# Copyright (c) 2022 NVIDIA CORPORATION.
#   Licensed under the MIT license.

# Adapted from https://github.com/jik876/hifi-gan under the MIT license.
#   LICENSE is in incl_licenses directory.

import math
import os
import random
import torch
import torch.utils.data
import numpy as np
from librosa.util import normalize
import librosa
from scipy.io.wavfile import read
from librosa.filters import mel as librosa_mel_fn
import pathlib
from tqdm import tqdm

MAX_WAV_VALUE = 32768.0


# def load_wav(full_path, sr_target):
#     sampling_rate, data = read(full_path)
#     if sampling_rate != sr_target:
#         raise RuntimeError("Sampling rate of the file {} is {} Hz, but the model requires {} Hz".
#               format(full_path, sampling_rate, sr_target))
#     return data, sampling_rate


def load_wav(full_path, sr_target):
    data = librosa.load(full_path, sr=sr_target)[0]
    return data, sr_target


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}


def mel_spectrogram(
    y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False
):
    if torch.min(y) < -1.0:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.0:
        print("max value is ", torch.max(y))

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[str(fmax) + "_" + str(y.device)] = (
            torch.from_numpy(mel).float().to(y.device)
        )
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)

    # complex tensor as default, then use view_as_real for future pytorch compatibility
    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window[str(y.device)],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    spec = torch.view_as_real(spec)

    spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))

    spec = torch.matmul(mel_basis[str(fmax) + "_" + str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec


def get_dataset_filelist(a):
    with open(a.input_training_file, "r", encoding="utf-8") as fi:
        lines = fi.readlines()

        training_files = []

        for line in lines:
            line = line.strip()
            if line == "":
                continue
            training_files.append(line)
        print("first training file: {}".format(training_files[0]))
        print("Loading {} training files".format(len(training_files)))

    with open(a.input_validation_file, "r", encoding="utf-8") as fi:
        lines = fi.readlines()

        valid_files = []

        for line in lines:
            line = line.strip()
            if line == "":
                continue
            valid_files.append(line)
        print("first valid file: {}".format(valid_files[0]))
        print("Loading {} valid files".format(len(valid_files)))

    with open(a.input_test_file, "r", encoding="utf-8") as fi:
        lines = fi.readlines()

        test_files = []

        for line in lines:
            line = line.strip()
            if line == "":
                continue
            test_files.append(line)
        print("first test file: {}".format(test_files[0]))
        print("Loading {} test files".format(len(test_files)))

    return training_files, valid_files, test_files


class MelDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        training_files,
        hparams,
        segment_size,
        n_fft,
        num_mels,
        hop_size,
        win_size,
        sampling_rate,
        fmin,
        fmax,
        split=True,
        shuffle=True,
        n_cache_reuse=1,
        device=None,
        fmax_loss=None,
        fine_tuning=False,
    ):
        self.audio_files = training_files
        random.seed(1234)
        if shuffle:
            random.shuffle(self.audio_files)
        self.hparams = hparams

        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.split = split
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.fmax_loss = fmax_loss
        self.cached_wav = None
        self.n_cache_reuse = n_cache_reuse
        self._cache_ref_count = 0
        self.device = device
        self.fine_tuning = fine_tuning

        # print("INFO: checking dataset integrity...")
        # for i in tqdm(range(len(self.audio_files))):
        #     assert os.path.exists(self.audio_files[i]), "{} not found".format(self.audio_files[i])

    def __getitem__(self, index):
        filename = self.audio_files[index]
        if self._cache_ref_count == 0:
            audio, sampling_rate = load_wav(filename, self.sampling_rate)
            # audio = audio / MAX_WAV_VALUE
            # if not self.fine_tuning:
            #     audio = normalize(audio) * 0.95
            self.cached_wav = audio
            if sampling_rate != self.sampling_rate:
                raise ValueError(
                    "{} SR doesn't match target {} SR".format(
                        sampling_rate, self.sampling_rate
                    )
                )
            self._cache_ref_count = self.n_cache_reuse
        else:
            audio = self.cached_wav
            self._cache_ref_count -= 1

        audio = torch.FloatTensor(audio)
        audio = audio.unsqueeze(0)

        if not self.fine_tuning:
            if self.split:
                if audio.size(1) >= self.segment_size:
                    max_audio_start = audio.size(1) - self.segment_size
                    audio_start = random.randint(0, max_audio_start)
                    audio = audio[:, audio_start : audio_start + self.segment_size]
                else:
                    audio = torch.nn.functional.pad(
                        audio, (0, self.segment_size - audio.size(1)), "constant"
                    )

                mel = mel_spectrogram(
                    audio,
                    self.n_fft,
                    self.num_mels,
                    self.sampling_rate,
                    self.hop_size,
                    self.win_size,
                    self.fmin,
                    self.fmax,
                    center=False,
                )
            else:  # validation step
                # match audio length to self.hop_size * n for evaluation
                if (audio.size(1) % self.hop_size) != 0:
                    audio = audio[:, : -(audio.size(1) % self.hop_size)]
                mel = mel_spectrogram(
                    audio,
                    self.n_fft,
                    self.num_mels,
                    self.sampling_rate,
                    self.hop_size,
                    self.win_size,
                    self.fmin,
                    self.fmax,
                    center=False,
                )
                assert (
                    audio.shape[1] == mel.shape[2] * self.hop_size
                ), "audio shape {} mel shape {}".format(audio.shape, mel.shape)

        else:
            mel = np.load(
                os.path.join(
                    self.base_mels_path,
                    os.path.splitext(os.path.split(filename)[-1])[0] + ".npy",
                )
            )
            mel = torch.from_numpy(mel)

            if len(mel.shape) < 3:
                mel = mel.unsqueeze(0)

            if self.split:
                frames_per_seg = math.ceil(self.segment_size / self.hop_size)

                if audio.size(1) >= self.segment_size:
                    mel_start = random.randint(0, mel.size(2) - frames_per_seg - 1)
                    mel = mel[:, :, mel_start : mel_start + frames_per_seg]
                    audio = audio[
                        :,
                        mel_start
                        * self.hop_size : (mel_start + frames_per_seg)
                        * self.hop_size,
                    ]
                else:
                    mel = torch.nn.functional.pad(
                        mel, (0, frames_per_seg - mel.size(2)), "constant"
                    )
                    audio = torch.nn.functional.pad(
                        audio, (0, self.segment_size - audio.size(1)), "constant"
                    )

        mel_loss = mel_spectrogram(
            audio,
            self.n_fft,
            self.num_mels,
            self.sampling_rate,
            self.hop_size,
            self.win_size,
            self.fmin,
            self.fmax_loss,
            center=False,
        )

        return (mel.squeeze(), audio.squeeze(0), filename, mel_loss.squeeze())

    def __len__(self):
        return len(self.audio_files)
