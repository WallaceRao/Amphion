from functools import partial
import logging
import multiprocessing as mp
import numpy as np
import os
import random

from torchtts.data.core import audio
from torchtts.data.core import features
from torchtts.data.core.dataset_builder import GeneratorBasedBuilder
from torchtts.data.core.dataset_info import DatasetInfo
from torchtts.utils.data_utils import get_bucket_scheme

logger = logging.getLogger(__name__)


class GPTTTSDataset(GeneratorBasedBuilder):
    def _info(self):
        return DatasetInfo(
            builder=self,
            description="Codec dataset builder",
            features=features.FeaturesDict(
                {
                    "speech": features.Audio(),
                    "phone_id": features.Tensor(shape=(None,), dtype=np.int64),
                    "duration": features.Tensor(shape=(None,), dtype=np.int64),
                }
            ),
        )

    def _target_suffixs(self):
        return ["speech", "phone_id", "duration"]

    def _split_generators(self):
        path = self._config.get("raw_data", None)
        if path is None:
            raise ValueError("You should specify raw_data in dataset builder")
        return {"train": self._raw_data_generator(split="train", path=path)}

    def _raw_data_generator(self, split, path):
        example_index = 0
        num_workers = self._config.get("preprocess_workers", os.cpu_count())
        if num_workers > 1:
            with mp.Pool(num_workers) as pool:
                for root, _, files in os.walk(path):
                    extract_fn = partial(
                        self._extract_feature,
                        wav_dir=root,
                        audio_config=self._config["audio_config"],
                    )
                    for result in pool.imap_unordered(extract_fn, files):
                        if result is not None:
                            yield f"{example_index:010}", result
                            example_index += 1
        else:
            for root, _, files in os.walk(path):
                for wav_file in files:
                    result = self._extract_feature(
                        wav_file=wav_file,
                        wav_dir=root,
                        audio_config=self._config["audio_config"],
                    )
                    if result is not None:
                        yield f"{example_index:010}", result
                        example_index += 1

    def _data_pipeline(self, datapipe, shuffle):
        shuffle = True
        if shuffle:
            datapipe = datapipe.shuffle(buffer_size=200)

        # filter min length
        min_sample_per_sent = self._config["audio_config"].get("sample_rate", 16000) * 3
        datapipe = datapipe.filter(
            self._filter_min_len, fn_kwargs={"min_len": min_sample_per_sent}
        )

        # filter max length
        max_sample_per_sent = (
            self._config["audio_config"].get("sample_rate", 16000) * 25
        )
        datapipe = datapipe.filter(
            self._filter_max_len, fn_kwargs={"max_len": max_sample_per_sent}
        )

        # filter min phone nums
        datapipe = datapipe.filter(
            self._filter_min_phone_num, fn_kwargs={"min_phone_num": 7}
        )

        # filter duration align phone nums
        datapipe = datapipe.filter(self._filter_dur_align_phone_num)

        # filter duration align frame nums
        hop_length = self._config["audio_config"].get("hop_length", 200)
        datapipe = datapipe.filter(
            self._filter_dur_align_frame_num, fn_kwargs={"hop_length": hop_length}
        )
        datapipe = datapipe.map(self._align_len, fn_kwargs={"hop_length": hop_length})

        datapipe = datapipe.map(self._get_mask, fn_kwargs={"hop_length": hop_length})

        batch_size = self._config["batch_size"]
        bucket_step = self._config.get("bucket_step", 1.1)
        bucket_scheme = get_bucket_scheme(batch_size, 8, bucket_step)
        datapipe = datapipe.dynamic_batch(
            group_key_fn=self.get_frames,
            bucket_boundaries=bucket_scheme["boundaries"],
            batch_sizes=bucket_scheme["batch_sizes"],
        )

        # Shuffle on batch
        if shuffle:
            datapipe = datapipe.shuffle(buffer_size=32)
        # datapipe = datapipe.collate(
        #     fn_kwargs={
        #         "padding_axes": {
        #             "speech": 0,
        #             "phone_id": 0,
        #             "duration": 0,
        #             "mask": 0,
        #             "phone_id_mask": 0,
        #             "ref_speech": 0,
        #             "ref_mask": 0,
        #         },
        #         "padding_values": {
        #             "speech": 0,
        #             "phone_id": 7,
        #             "duration": 0,
        #             "mask": 0,
        #             "phone_id_mask": 0,
        #             "ref_speech": 0,
        #             "ref_mask": 0,
        #         },
        #     }
        # )  # padding <PAD> is 7

        datapipe = datapipe.collate(
            fn_kwargs={
                "padding_axes": {
                    "speech": 0,
                    "phone_id": 0,
                    "duration": 0,
                    "mask": 0,
                    "phone_id_mask": 0,
                    # "ref_speech": 0,
                    # "ref_mask": 0,
                },
                "padding_values": {
                    "speech": 0,
                    "phone_id": 7,
                    "duration": 0,
                    "mask": 0,
                    "phone_id_mask": 0,
                    # "ref_speech": 0,
                    # "ref_mask": 0,
                },
            }
        )  # padding <PAD> is 7

        return datapipe

    @staticmethod
    def _filter_min_len(data, min_len):
        return bool(len(data["speech"]) > min_len)

    @staticmethod
    def _filter_max_len(x, max_len):
        return bool(len(x["speech"]) < max_len)

    @staticmethod
    def _filter_min_phone_num(data, min_phone_num):
        return bool(len(data["phone_id"]) > min_phone_num)

    @staticmethod
    def _filter_dur_align_phone_num(data):
        return bool(len(data["phone_id"]) == len(data["duration"]))

    @staticmethod
    def _filter_dur_align_frame_num(data, hop_length):
        return bool(abs(sum(data["duration"]) - len(data["speech"]) // hop_length) <= 3)

    @staticmethod
    def _extract_feature(wav_file, wav_dir, audio_config):
        if os.path.splitext(wav_file)[1] != ".wav":
            return None
        res_type = audio_config.get("res_type", "soxr_hq")
        wav_path = os.path.join(wav_dir, wav_file)
        # Vocoder target synthesis sample rate
        target_wav_data = audio.load_wav(
            wav_path, audio_config["target_sample_rate"], res_type=res_type
        )
        # Maybe down-sampled audio data for mel extraction
        # feats_wav_data = audio.load_wav(wav_path, audio_config['sample_rate'],
        #                                res_type=res_type)
        # mel_spec = audio.mel_spectrogram(feats_wav_data, **audio_config)
        # return {'audio': target_wav_data, 'mel': mel_spec}
        return {"audio": target_wav_data}

    @staticmethod
    def _align_len(data, hop_length):
        frame_num = sum(data["duration"])
        sample_num = len(data["speech"])

        expected_sample_num = frame_num * hop_length
        if expected_sample_num > sample_num:
            data["speech"] = np.pad(
                data["speech"],
                (0, expected_sample_num - sample_num),
                "constant",
                constant_values=(0, data["speech"][-1]),
            )
        else:
            data["speech"] = data["speech"][:expected_sample_num]

        # add mask
        data["duration"] = np.array(data["duration"])
        data["phone_id"] = np.array(data["phone_id"])

        return data

    @staticmethod
    def get_frames(x):
        return len(x["speech"])

    @staticmethod
    def _get_mask(data, hop_length):
        data["mask"] = np.ones((len(data["speech"]) // hop_length))
        data["phone_id_mask"] = np.ones(data["phone_id"].shape)
        return data
