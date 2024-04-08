from itertools import chain
import logging
import pickle
import torch
import pyworld as pw
import numpy as np
import soundfile as sf
import os

from torchtts.nn.criterions import GANLoss
from torchtts.nn.criterions import (
    MultiResolutionSTFTLoss,
    MultiResolutionMelSpectrogramLoss,
    SpeakerLoss,
)
from torchtts.nn.metrics import Mean
from torchtts.nn.optim.lr_schedulers import PowerLR, WarmupLR, WarmupLR_2
from torchtts.trainers.base_trainer import Trainer
from torchtts.nn.criterions.duration_loss import DurationPredictorLoss
from torch.optim import Adam, AdamW
from einops import rearrange

from torchaudio.functional import pitch_shift
from librosa.filters import mel as librosa_mel_fn

logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


class GPTTTSTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        from torchtts.models.codec_yc_mel import (
            MelCodecEncoder,
            MelCodecDecoderWithTimbre,
        )
        from torchtts.models.codec_yc import (
            CodecEncoder,
            CodecDecoder,
        )

        if self._config["codec"]["codec_type"] == "melgan":
            self.CodecEnc = MelCodecEncoder(
                d_mel=80,
                d_model=96,
                num_blocks=4,
                out_channels=self._config["codec"]["latent_channels"],
            )
            self.generator = MelCodecDecoderWithTimbre(
                in_channels=self._config["codec"]["latent_channels"],
                num_quantizers=self._config["codec"]["num_quantizers"],
                codebook_size=self._config["codec"]["codebook_size"],
                codebook_dim=self._config["codec"]["codebook_dim"],
                quantizer_type="fvq",
                use_l2_normlize=True,
                vocos_dim=512,
                vocos_intermediate_dim=4096,
                vocos_num_layers=16,
                ln_before_vq=True,
                use_pe=False,
            )

        elif self._config["codec"]["codec_type"] == "codec":
            self.CodecEnc = CodecEncoder(
                d_model=96,
                up_ratios=[2, 4, 5, 5],
                out_channels=self._config["codec"]["latent_channels"],
            )
            self.generator = CodecDecoder(
                in_channels=self._config["codec"]["latent_channels"],
                up_ratios=[5, 5, 4, 2],
                num_quantizers=self._config["codec"]["num_quantizers"],
                codebook_size=self._config["codec"]["codebook_size"],
                codebook_dim=self._config["codec"]["codebook_dim"],
                quantizer_type="fvq",
                use_l2_normlize=True,
                use_vocos=True,
                vocos_dim=512,
                vocos_intermediate_dim=4096,
                vocos_num_layers=24,
            )

        pretrained_model_ckpt = torch.load(
            self._config["codec"]["pretrained_model_path"],
        )

        codec_enc_dict = pretrained_model_ckpt["model"]["CodecEnc"]
        generator_dict = pretrained_model_ckpt["model"]["generator"]

        self.CodecEnc.load_state_dict(codec_enc_dict)
        self.generator.load_state_dict(generator_dict)

        self.CodecEnc = self.CodecEnc.to("cuda")
        self.generator = self.generator.to("cuda")

        self.CodecEnc.requires_grad_(False)
        self.generator.requires_grad_(False)

        self.CodecEnc.eval()
        self.generator.eval()

    def train_step(self, batch, acc_step=0):
        with self.engine.context():
            with torch.no_grad():
                speech = batch["speech"]
                # TODO: get target
                if self._config["codec"]["codec_type"] == "melgan":
                    vq_emb = self.CodecEnc(speech.unsqueeze(1))
                    (
                        _,
                        vq_indices,
                        _,
                        _,
                        _,
                        _,
                    ) = self.generator(vq_emb, vq=True, eval_vq=False)
                    frame_nums = speech.shape[-1] // 200
                    vq_emb_spk = self.CodecEnc(
                        speech.unsqueeze(1)[
                            :,
                            :,
                            : min(
                                np.random.randint(
                                    min(240, int(0.5 * frame_nums)),
                                    max(240, int(0.5 * frame_nums)) + 1,
                                ),
                                frame_nums,
                            )
                            * 200,
                        ]
                    )
                    (
                        _,
                        _,
                        _,
                        _,
                        _,
                        speaker_embedding,
                    ) = self.generator(vq_emb_spk, vq=True, eval_vq=False)
                elif self._config["codec"]["codec_type"] == "codec":
                    vq_emb = self.CodecEnc(speech.unsqueeze(1))
                    (
                        _,
                        vq_indices,
                        _,
                        _,
                        _,
                    ) = self.generator(vq_emb, vq=True, eval_vq=False)
                target = vq_indices[0, :, :]

            phone_id = batch["phone_id"]
            phone_id_mask = batch["phone_id_mask"]
            mask = batch["mask"]

            # print(phone_id[0], phone_id.shape)
            # print(phone_id_mask[0], phone_id_mask.shape)
            # print(target[0], target.shape)
            # print(mask[0], mask.shape)

            if self._config["use_timbre_embedding"] == True:
                speaker_embedding = speaker_embedding.unsqueeze(
                    1
                )  # (B, d) -> (B, 1, d)
                out = self.model["generator"](
                    phone_ids=phone_id.long(),
                    phone_mask=phone_id_mask.long(),
                    target_ids=target.long(),
                    target_mask=mask.long(),
                    input_embeds=speaker_embedding,
                )
            else:
                out = self.model["generator"](
                    phone_ids=phone_id.long(),
                    phone_mask=phone_id_mask.long(),
                    target_ids=target.long(),
                    target_mask=mask.long(),
                )

            gen_loss = 0.0
            gen_loss += out.loss
            self.metrics["ce_loss"].update_state(out.loss)

            self.engine.optimize_step(
                loss=gen_loss,
                optimizer=self.optimizers["gen"],
                lr_scheduler=self.lr_schedulers["gen"],
                current_step=acc_step,
                grad_accumulate=self.gradient_accumulate,
                donot_optimize=True if self.gradient_accumulate > 1 else False,
                grad_norm=2e3,
            )

            if self.gradient_accumulate > 1 and acc_step == 0:
                self.engine.optimize_gradients(optimizer=self.optimizers["gen"])

            return {k: m.result() for k, m in self.metrics.items()}

    def configure_optimizers(self):
        gen_params = self.model["generator"].parameters()
        logger.info(
            "generator parameters count: {} M".format(
                sum(
                    p.numel()
                    for p in self.model["generator"].parameters()
                    if p.requires_grad
                )
                / 1e6
            )
        )

        return {
            "gen": AdamW(gen_params, **self._config["gen_optim_params"]),
        }

    def configure_lr_schedulers(self):
        """
        return {
            'gen': PowerLR(self.optimizers['gen'],
                           **self._config["gen_schedule_params"]),
        }
        """
        if self._config["use_inverse_square_root"]:
            return {
                "gen": WarmupLR_2(
                    self.optimizers["gen"], **self._config["gen_schedule_params"]
                )
            }
        return {
            "gen": WarmupLR(
                self.optimizers["gen"], **self._config["gen_schedule_params"]
            )
        }

    def configure_criteria(self):
        criteria = {
            "l1_loss": torch.nn.L1Loss(reduction="mean"),
            "l2_loss": torch.nn.MSELoss(reduction="mean"),
        }
        return criteria

    def configure_metrics(self):
        metrics = {
            "ce_loss": Mean(),
        }

        return metrics
