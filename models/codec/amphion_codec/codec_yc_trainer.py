from itertools import chain
import logging
import pickle
import torch
import pyworld as pw
import numpy as np

from torchtts.nn.criterions import GANLoss
from torchtts.nn.criterions import (
    MultiResolutionSTFTLoss,
    MultiResolutionMelSpectrogramLoss,
)
from torchtts.nn.metrics import Mean
from torchtts.nn.optim.lr_schedulers import PowerLR, WarmupLR
from torchtts.trainers.base_trainer import Trainer
from torch.optim import Adam
from einops import rearrange

from torchaudio.functional import pitch_shift

logger = logging.getLogger(__name__)


class CodecTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train_step(self, batch, acc_step=0):
        if "audio" not in batch:
            batch["audio"] = batch.get("speech", None)

        with self.engine.context():
            vq_emb = self.model["CodecEnc"](batch["audio"].unsqueeze(1))

            vq_post_emb, vq_indices, commit_losses, codebook_losses, _ = self.model[
                "generator"
            ](vq_emb, vq=True, eval_vq=False)

            vq_loss = commit_losses + codebook_losses
            y_ = self.model["generator"](vq_post_emb, vq=False)

            y = batch["audio"].unsqueeze(1)

        self.train_disc_step(y, y_, acc_step=acc_step)

        # l1 loss for reconstruction
        abs_loss = self.criteria["l1_loss"](y_, y)

        self.train_gen_step(
            y,
            y_,
            vq_loss,
            abs_loss,
            acc_step=acc_step,
            batch=batch,
        )

        if self.gradient_accumulate > 1 and acc_step == 0:
            self.engine.optimize_gradients(optimizer=self.optimizers["disc"])
            self.engine.optimize_gradients(optimizer=self.optimizers["gen"])

        return {k: m.result() for k, m in self.metrics.items()}

    def set_discriminator_gradients(self, flag=True):
        for p in self.model["discriminator"].parameters():
            p.requires_grad = flag

        if "spec_discriminator" in self.model:
            for p in self.model["spec_discriminator"].parameters():
                p.requires_grad = flag

        if "MPD_spec_discriminator" in self.model:
            for p in self.model["MPD_spec_discriminator"].parameters():
                p.requires_grad = flag

    def cal_gen_loss(self, y, y_, vq_loss=None, batch=None, gen_out=None):
        gen_loss = 0.0
        self.set_discriminator_gradients(flag=False)

        # stft loss
        if (
            self._config["use_stft_loss"]
            and self.global_steps < self._config["stft_loss_steps"]
        ):
            with self.engine.context():
                sc_loss, mag_loss = self.criteria["stft_loss"](
                    y_.squeeze(1), y.squeeze(1)
                )
                gen_loss += (sc_loss + mag_loss) * self._config["lambda"][
                    "lambda_stft_loss"
                ]

            self.metrics["sc_loss"].update_state(sc_loss)
            self.metrics["mag_loss"].update_state(mag_loss)

        # mel loss
        if self._config["use_mel_loss"]:
            with self.engine.context():
                mel_loss = self.criteria["mel_loss"](y_.squeeze(1), y.squeeze(1))
                gen_loss += mel_loss * self._config["lambda"]["lambda_mel_loss"]

            self.metrics["mel_loss"].update_state(mel_loss)

        # gan loss
        if self.global_steps >= self._config["disc_train_start_steps"]:
            with self.engine.context():
                # Multi-period waveform discriminators (MPD)
                p_ = self.model["discriminator"](y_)

                adv_loss_list = []
                for i in range(len(p_)):
                    adv_loss_list.append(self.criteria["gan_loss"].gen_loss(p_[i][-1]))

                # Multi-band multi-scale STFT discriminator
                if "spec_discriminator" in self.model:
                    sd_p_ = self.model["spec_discriminator"](y_)
                    for i in range(len(sd_p_)):
                        adv_loss_list.append(
                            self.criteria["gan_loss"].gen_loss(sd_p_[i][-1])
                        )

                if "MPD_spec_discriminator" in self.model:
                    mpd_sd_p_ = self.model["MPD_spec_discriminator"](y_)
                    for i in range(len(mpd_sd_p_)):
                        adv_loss_list.append(
                            self.criteria["gan_loss"].gen_loss(mpd_sd_p_[i][-1])
                        )
                adv_loss = sum(adv_loss_list)
                gen_loss += adv_loss * self._config["lambda"]["lambda_adv"]

                self.metrics["adv_loss"].update_state(adv_loss)

            if self._config["use_feat_match_loss"]:
                fm_loss = 0.0
                with self.engine.context():
                    with torch.no_grad():
                        p = self.model["discriminator"](y)
                    for i in range(len(p_)):
                        for j in range(len(p_[i]) - 1):
                            fm_loss += self.criteria["fm_loss"](
                                p_[i][j], p[i][j].detach()
                            )
                    # fm_loss /= (i + 1) * (j + 1)

                    self.metrics["fm_loss"].update_state(fm_loss)
                    gen_loss += (
                        fm_loss * self._config["lambda"]["lambda_feat_match_loss"]
                    )

                    if "spec_discriminator" in self.model:
                        spec_fm_loss = 0.0
                        with torch.no_grad():
                            sd_p = self.model["spec_discriminator"](y)
                        for i in range(len(sd_p_)):
                            for j in range(len(sd_p_[i]) - 1):
                                spec_fm_loss += self.criteria["fm_loss"](
                                    sd_p_[i][j], sd_p[i][j].detach()
                                )
                        # spec_fm_loss /= (i + 1) * (j + 1)

                        self.metrics["spec_fm_loss"].update_state(spec_fm_loss)
                        gen_loss += (
                            spec_fm_loss
                            * self._config["lambda"]["lambda_feat_match_loss"]
                        )
                    if "MPD_spec_discriminator" in self.model:
                        mpd_spec_fm_loss = 0.0
                        with torch.no_grad():
                            mpd_sd_p = self.model["MPD_spec_discriminator"](y)
                        for i in range(len(mpd_sd_p_)):
                            for j in range(len(mpd_sd_p_[i]) - 1):
                                spec_fm_loss += self.criteria["fm_loss"](
                                    mpd_sd_p_[i][j], mpd_sd_p[i][j].detach()
                                )
                        # mpd_spec_fm_loss /= (i + 1) * (j + 1)

                        self.metrics["mpd_spec_fm_loss"].update_state(mpd_spec_fm_loss)
                        gen_loss += (
                            mpd_spec_fm_loss
                            * self._config["lambda"]["lambda_feat_match_loss"]
                        )

        if vq_loss is not None:
            # if type(vq_loss) == dict:
            #     disagreement_loss = vq_loss["disagreement_loss"]
            #     vq_loss = vq_loss["vq_loss"]
            gen_loss += sum(vq_loss)
            self.metrics["vq_loss"].update_state(sum(vq_loss))
            # if self._config["use_disagreement_loss"] and "disagreement_loss" in vq_loss:
            #     # gen_loss += disagreement_loss
            #     self.metrics["disagreement_loss"].update_state(disagreement_loss)

        self.set_discriminator_gradients(flag=True)
        self.metrics["gen_loss"].update_state(gen_loss)

        return gen_loss

    def train_disc_step(self, y, y_, acc_step=0):
        if self.global_steps >= self._config["disc_train_start_steps"]:
            with self.engine.context():
                # Multi-period waveform discriminators (MPD)
                p = self.model["discriminator"](y)
                p_ = self.model["discriminator"](y_.detach())

                real_loss_list = []
                fake_loss_list = []
                for i in range(len(p)):
                    real_loss, fake_loss = self.criteria["gan_loss"].disc_loss(
                        p[i][-1], p_[i][-1]
                    )
                    real_loss_list.append(real_loss)
                    fake_loss_list.append(fake_loss)

                # Multi-band multi-scale STFT discriminator
                if "spec_discriminator" in self.model:
                    sd_p = self.model["spec_discriminator"](y)
                    sd_p_ = self.model["spec_discriminator"](y_.detach())

                    for i in range(len(sd_p)):
                        real_loss, fake_loss = self.criteria["gan_loss"].disc_loss(
                            sd_p[i][-1], sd_p_[i][-1]
                        )
                        real_loss_list.append(real_loss)
                        fake_loss_list.append(fake_loss)

                if "MPD_spec_discriminator" in self.model:
                    mpd_sd_p = self.model["MPD_spec_discriminator"](y)
                    mpd_sd_p_ = self.model["MPD_spec_discriminator"](y_.detach())

                    for i in range(len(mpd_sd_p)):
                        real_loss, fake_loss = self.criteria["gan_loss"].disc_loss(
                            mpd_sd_p[i][-1], mpd_sd_p_[i][-1]
                        )
                        real_loss_list.append(real_loss)
                        fake_loss_list.append(fake_loss)

                real_loss = sum(real_loss_list)
                fake_loss = sum(fake_loss_list)

                disc_loss = real_loss + fake_loss
                disc_loss = disc_loss * self._config["lambda"]["lambda_disc"]

            self.metrics["real_loss"].update_state(real_loss)
            self.metrics["fake_loss"].update_state(fake_loss)
            self.metrics["disc_loss"].update_state(disc_loss)

            self.engine.optimize_step(
                loss=disc_loss,
                optimizer=self.optimizers["disc"],
                lr_scheduler=self.lr_schedulers["disc"],
                current_step=acc_step,
                grad_accumulate=self.gradient_accumulate,
                donot_optimize=True if self.gradient_accumulate > 1 else False,
                grad_norm=2e2,
            )

    def train_gen_step(
        self,
        y,
        y_,
        vq_loss=None,
        abs_loss=None,
        acc_step=0,
        batch=None,
    ):
        gen_loss = self.cal_gen_loss(y, y_, vq_loss, batch)

        if abs_loss is not None:
            # gen_loss += abs_loss
            self.metrics["abs_loss"].update_state(abs_loss)

        # self.engine.optimize_step(loss=gen_loss,
        #                           optimizer=self.optimizers['gen'])
        self.engine.optimize_step(
            loss=gen_loss,
            optimizer=self.optimizers["gen"],
            lr_scheduler=self.lr_schedulers["gen"],
            current_step=acc_step,
            grad_accumulate=self.gradient_accumulate,
            donot_optimize=True if self.gradient_accumulate > 1 else False,
            grad_norm=2e3,
        )

    def configure_optimizers(self):
        disc_params = self.model["discriminator"].parameters()
        if "spec_discriminator" in self.model:
            disc_params = chain(
                disc_params, self.model["spec_discriminator"].parameters()
            )
        if "MPD_spec_discriminator" in self.model:
            disc_params = chain(
                disc_params, self.model["MPD_spec_discriminator"].parameters()
            )

        gen_params = self.model["CodecEnc"].parameters()
        gen_params = chain(gen_params, self.model["generator"].parameters())

        logger.info(
            "Encoder parameters count: {} M".format(
                sum(
                    p.numel()
                    for p in self.model["CodecEnc"].parameters()
                    if p.requires_grad
                )
                / 1e6
            )
        )
        logger.info(
            "Generator parameters count: {} M".format(
                sum(
                    p.numel()
                    for p in self.model["generator"].parameters()
                    if p.requires_grad
                )
                / 1e6
            )
        )
        logger.info(
            "Spec discriminator (MBMSSTFTD) parameters count: {} M".format(
                sum(
                    p.numel()
                    for p in self.model["spec_discriminator"].parameters()
                    if p.requires_grad
                )
                / 1e6
            )
        )
        logger.info(
            "Time discriminator (MPD) parameters count: {} M".format(
                sum(
                    p.numel()
                    for p in self.model["discriminator"].parameters()
                    if p.requires_grad
                )
                / 1e6
            )
        )

        return {
            "gen": Adam(gen_params, **self._config["gen_optim_params"]),
            "disc": Adam(disc_params, **self._config["disc_optim_params"]),
        }

    def configure_lr_schedulers(self):
        """
        return {
            'gen': PowerLR(self.optimizers['gen'],
                           **self._config["gen_schedule_params"]),
            'disc': PowerLR(self.optimizers['disc'],
                            **self._config["disc_schedule_params"]),
        }
        """
        return {
            "gen": WarmupLR(
                self.optimizers["gen"], **self._config["gen_schedule_params"]
            ),
            "disc": WarmupLR(
                self.optimizers["disc"], **self._config["disc_schedule_params"]
            ),
        }

    def configure_criteria(self):
        criteria = {
            "gan_loss": GANLoss(mode="lsgan"),
        }
        if self._config["use_stft_loss"]:
            criteria["stft_loss"] = MultiResolutionSTFTLoss(
                **self._config["stft_loss_params"]
            )
        if self._config["use_mel_loss"]:
            criteria["mel_loss"] = MultiResolutionMelSpectrogramLoss(
                **self._config["mel_loss_params"]
            )
        if self._config["use_feat_match_loss"]:
            criteria["fm_loss"] = torch.nn.L1Loss()

        criteria["l1_loss"] = torch.nn.L1Loss(reduction="mean")
        criteria["l2_loss"] = torch.nn.MSELoss(reduction="mean")
        criteria["bce_loss"] = torch.nn.BCEWithLogitsLoss(reduction="mean")
        criteria["ce_loss"] = torch.nn.CrossEntropyLoss(reduction="mean")
        return criteria

    def configure_metrics(self):
        metrics = {
            "real_loss": Mean(),
            "fake_loss": Mean(),
            "disc_loss": Mean(),
            "gen_loss": Mean(),
            "adv_loss": Mean(),
            "vq_loss": Mean(),
            "abs_loss": Mean(),
        }
        if self._config["use_feat_match_loss"]:
            metrics["fm_loss"] = Mean()
            if "spec_discriminator" in self.model:
                metrics["spec_fm_loss"] = Mean()
            # if "MPD_spec_discriminator" in self.model:
            #     metrics["mpd_spec_fm_loss"] = Mean()
        if self._config["use_stft_loss"]:
            metrics["sc_loss"] = Mean()
            metrics["mag_loss"] = Mean()
        if self._config["use_mel_loss"]:
            metrics["mel_loss"] = Mean()
        # if self._config["use_disagreement_loss"]:
        #     metrics["disagreement_loss"] = Mean()

        return metrics
