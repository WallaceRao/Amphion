# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import glob
import json
import torch
import time
import numpy as np
import ssl
import sys

from tqdm import tqdm
from datetime import datetime
from models.svc.diffusion.diffusion_inference import DiffusionInference
from models.svc.comosvc.comosvc_inference import ComoSVCInference
from models.svc.vits.vits_inference import VitsInference
from models.svc.transformer.transformer_inference import TransformerInference
from utils.util import load_config
from utils.audio_slicer import split_audio, merge_segments_encodec
from processors import acoustic_extractor, content_extractor
from processors.content_extractor import WhisperExtractor, ContentvecExtractor
from utils.audio import load_audio_torch
from pydub import AudioSegment

import logging

sys.path.insert(0,os.getcwd())
os.environ["WORK_DIR"] = os.getcwd()
ssl._create_default_https_context = ssl._create_unverified_context

service_logger = logging.getLogger("svc_service")
service_logger.setLevel(logging.INFO)
log_path = os.getcwd() + "/work_dir/logs/svc.log"
handler = logging.FileHandler(log_path, mode='a')
formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
handler.setFormatter(formatter)
service_logger.addHandler(handler)

inference_map = {}
singger_name_mapping = {"vocalist_l1_王菲":"vocalist_l1_wangfei",
                        "vocalist_l1_张学友":"vocalist_l1_zhangxueyou",
                        "vocalist_l1_李健":"vocalist_l1_lijian",
                        "vocalist_l1_汪峰":"vocalist_l1_wangfeng",
                        "vocalist_l1_石倚洁":"vocalist_l1_shiyijie",
                        "vocalist_l1_蔡琴":"vocalist_l1_caiqin",
                        "vocalist_l1_那英":"vocalist_l1_naying",
                        "vocalist_l1_陈奕迅":"vocalist_l1_chenyixun",
                        "vocalist_l1_陶喆":"vocalist_l1_taozhe",
                        "vocalist_l1_Adele": "vocalist_l1_Adele",
                        "vocalist_l1_Beyonce": "vocalist_l1_Beyonce",
                        "vocalist_l1_BrunoMars": "vocalist_l1_BrunoMars",
                        "vocalist_l1_JohnMayer": "vocalist_l1_JohnMayer",
                        "vocalist_l1_MichaelJackson": "vocalist_l1_MichaelJackson",
                        "vocalist_l1_TaylorSwift": "vocalist_l1_TaylorSwift"}

whisper_extractor = None
contentvec_extractor = None


def build_inference(args, cfg, infer_type="from_dataset"):
    global inference_map
    supported_inference = {
        "DiffWaveNetSVC": DiffusionInference,
        "DiffComoSVC": ComoSVCInference,
        "TransformerSVC": TransformerInference,
        "VitsSVC": VitsInference,
    }
    if cfg.model_type not in inference_map.keys():
        print("cfg.model_type:", cfg.model_type)
        inference_class = supported_inference[cfg.model_type]
        inference_map[cfg.model_type] = inference_class(args, cfg, infer_type)
    return inference_map[cfg.model_type]


def prepare_for_audio_file(args, cfg, num_workers=1):
    global whisper_extractor, contentvec_extractor
    preprocess_path = cfg.preprocess.processed_dir
    audio_name = cfg.inference.source_audio_name
    temp_audio_dir = os.path.join(preprocess_path, audio_name)
    
    ### eval file
    t = time.time()
    eval_file = prepare_source_eval_file(cfg, temp_audio_dir, audio_name)
    args.source = eval_file
    with open(eval_file, "r") as f:
        metadata = json.load(f)
    ### acoustic features
    t = time.time()
    acoustic_extractor.extract_utt_acoustic_features_serial(
        metadata, temp_audio_dir, cfg
    )
    if cfg.preprocess.use_min_max_norm_mel == True:
        acoustic_extractor.cal_mel_min_max(
            dataset=audio_name, output_path=preprocess_path, cfg=cfg, metadata=metadata
        )
    acoustic_extractor.cal_pitch_statistics_svc(
        dataset=audio_name, output_path=preprocess_path, cfg=cfg, metadata=metadata
    )
    ### content features
    t = time.time()
    if whisper_extractor is None:
        whisper_extractor = WhisperExtractor(cfg)
        whisper_extractor.load_model()

    if contentvec_extractor is None:
        contentvec_extractor = ContentvecExtractor(cfg)
        contentvec_extractor.load_model()

    content_extractor.extract_utt_content_features_dataloader(
        cfg, metadata, num_workers, whisper_extractor, contentvec_extractor
    )
    return args, cfg, temp_audio_dir


def merge_for_audio_segments(audio_files, args, cfg):
    audio_name = cfg.inference.source_audio_name
    target_singer_name = args.target_singer
    merge_segments_encodec(
        wav_files=audio_files,
        fs=cfg.preprocess.sample_rate,
        output_path=os.path.join(
            args.output_dir, "{}_{}.wav".format(audio_name,
                                                singger_name_mapping[target_singer_name])
        ),
        overlap_duration=cfg.inference.segments_overlap_duration,
    )
    for tmp_file in audio_files:
        os.remove(tmp_file)


def merge_for_pitch_segments(pitch_folder, args, cfg):
    sample_rate = cfg.preprocess.sample_rate
    overlap_duration=cfg.inference.segments_overlap_duration
    hop_size = cfg.preprocess.hop_size
    overlap_frame = 1.0 * overlap_duration * sample_rate / hop_size 
    overlap_frame = 88.16666666666 # this is a magic number
    mod = overlap_frame - int(overlap_frame)
    f0_files = [w for w in glob.glob(os.path.join(pitch_folder, '*.npy')) if os.path.isfile(w)]
    f0_files.sort()
    total_f0 = np.load(f0_files[0])
    for i in range(1, len(f0_files)):
        f0_data = np.load(f0_files[i])
        clipped_start = int(int(overlap_frame) + mod * (i))
        total_f0 = np.concatenate((total_f0, f0_data[clipped_start:]), axis=0)
    return total_f0


def overwrite_pitch(total_f0, pitch_folder, args, cfg):
    f0_files = [w for w in glob.glob(os.path.join(pitch_folder, '*.npy')) if os.path.isfile(w)]
    f0_files.sort()
    if len(f0_files) == 0:
        service_logger.info(f"failed to overwrite pitch, no file found from folder:{pitch_folder}")
        return
    if len(f0_files) == 1:
        np.save(f0_files[0], total_f0)
        return
    offset = 0
    f0_file_idx = 0
    total_f0_len = total_f0.shape[0]
    total_f0_len_with_overlap = 0
    for f0_file in f0_files:
        total_f0_len_with_overlap += np.load(f0_file).shape[0]
    overlap_count = len(f0_files) - 1
    overlap_len = (total_f0_len_with_overlap - total_f0_len) * 1.0 / overlap_count
    mod = overlap_len - int(overlap_len)

    for f0_file in f0_files:
        original_f0 = np.load(f0_file)
        f0_len = original_f0.shape[0]
        if offset + f0_len > total_f0_len:
            if total_f0_len - f0_len < 0:
                service_logger.error(f"overwrite pitch failed, bad total f0 len:{total_f0_len}")
                offset = 0
                return
            offset = total_f0_len - f0_len
        np.save(f0_file, total_f0[offset:offset + f0_len])
        offset = offset + f0_len - int(overlap_len) - int(mod * f0_file_idx)
        f0_file_idx = f0_file_idx + 1


def prepare_source_eval_file(cfg, temp_audio_dir, audio_name):
    """
    Prepare the eval file (json) for an audio
    """

    audio_chunks_results = split_audio(
        wav_file=cfg.inference.source_audio_path,
        target_sr=cfg.preprocess.sample_rate,
        output_dir=os.path.join(temp_audio_dir, "wavs"),
        max_duration_of_segment=cfg.inference.segments_max_duration,
        overlap_duration=cfg.inference.segments_overlap_duration,
    )

    metadata = []
    for i, res in enumerate(audio_chunks_results):
        res["index"] = i
        res["Dataset"] = audio_name
        res["Singer"] = audio_name
        res["Uid"] = "{}_{}".format(audio_name, res["Uid"])
        metadata.append(res)

    eval_file = os.path.join(temp_audio_dir, "eval.json")
    with open(eval_file, "w") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False, sort_keys=True)

    return eval_file


def cuda_relevant(deterministic=False):
    torch.cuda.empty_cache()
    # TF32 on Ampere and above
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.allow_tf32 = True
    # Deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    torch.use_deterministic_algorithms(deterministic)


def infer(args, cfg, infer_type):
    # Build inference
    trainer = build_inference(args, cfg, infer_type)
    # Run inference
    output_audio_files = trainer.inference()
    service_logger.info(f"model infer finished, output audio files:{len(output_audio_files)}")
    return output_audio_files


def build_parser():
    r"""Build argument parser for inference.py.
    Anything else should be put in an extra config YAML file.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        help="JSON/YAML file for configurations.",
    )
    parser.add_argument(
        "--acoustics_dir",
        type=str,
        help="Acoustics model checkpoint directory. If a directory is given, "
        "search for the latest checkpoint dir in the directory. If a specific "
        "checkpoint dir is given, directly load the checkpoint.",
    )
    parser.add_argument(
        "--vocoder_dir",
        type=str,
        required=False,
        help="Vocoder checkpoint directory. Searching behavior is the same as "
        "the acoustics one.",
    )
    parser.add_argument(
        "--target_singer",
        type=str,
        required=False,
        help="convert to a specific singer (e.g. --target_singers singer_id).",
    )
    parser.add_argument(
        "--trans_key",
        default=0,
        help="0: no pitch shift; autoshift: pitch shift;  int: key shift.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="",
        help="Source audio file or directory. If a JSON file is given, "
        "inference from dataset is applied. If a directory is given, "
        "inference from all wav/flac/mp3 audio files in the directory is applied. "
        "Default: inference from all wav/flac/mp3 audio files in ./source_audio",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Output directory. Default: ./conversion_results",
    )
    parser.add_argument(
        "--work_dir",
        type=str,
        default="work_dir",
        help="Output directory. Default: ./conversion_results",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="warning",
        help="Logging level. Default: warning",
    )
    parser.add_argument(
        "--keep_cache",
        action="store_true",
        default=True,
        help="Keep cache files. Only applicable to inference from files.",
    )
    parser.add_argument(
        "--diffusion_inference_steps",
        type=int,
        default=1000,
        help="Number of inference steps. Only applicable to diffusion inference.",
    )
    return parser


def process_request(request_work_folder, action, singer, pitch_np, format):
    ### Parse arguments and config
    service_logger.info(f"begin to process request in folder:{request_work_folder}")
    if singer not in singger_name_mapping.keys():
        service_logger.info(f"failed to process request in folder:{request_work_folder}, unknown singer:{singer}")
        return None, None
    args = build_parser().parse_args()
    args.config = "ckpts/svc/vocalist_l1_contentvec+whisper/args.json"
    args.target_singer = singer
    args.acoustics_dir = "ckpts/svc/vocalist_l1_contentvec+whisper"
    args.vocoder_dir = "pretrained/bigvgan_singing"
    args.trans_key = "autoshift"
    cfg = load_config(args.config)
    cfg.preprocess.processed_dir = request_work_folder
    if len(args.source) == 0:
        args.source = request_work_folder + "/input_wav"
    if len(args.output_dir) == 0:
        args.output_dir = request_work_folder + "/output_wav"
    ### Infer from file
    # Get all the source audio files (.wav, .flac, .mp3)
    source_audio_dir = args.source
    audio_list = []
    for suffix in ["wav", "mp3"]:
        audio_list += glob.glob(
            os.path.join(source_audio_dir, "*.{}".format(suffix)), recursive=True
        )
    if len(audio_list) != 1:
        service_logger.info(f"expect only 1 audio file in folder:{source_audio_dir}, but {len(audio_list)} audio files found")
        return None, None
    f0_bytes = None
    audio_bytes = None
    output_root_path = args.output_dir
    audio_path = audio_list[0]
    audio_name = audio_path.split("/")[-1].split(".")[0]
    pitch_folder = request_work_folder + "/" + audio_name + "/pitches"
    args.output_dir = os.path.join(output_root_path, audio_name)
    os.makedirs(args.output_dir, exist_ok=True)
    cfg.inference.source_audio_path = audio_path
    cfg.inference.source_audio_name = audio_name
    cfg.inference.segments_max_duration = 10.0
    cfg.inference.segments_overlap_duration = 1.0
    # Prepare metadata and features
    service_logger.info(f"begin prepare_for_audio_file for folder:{request_work_folder}")
    args, cfg, cache_dir = prepare_for_audio_file(args, cfg)
    # Overwrite pitch features if they are provided
    if pitch_np is not None and action != "pitch_only":
        overwrite_pitch(pitch_np, pitch_folder, args, cfg)
        service_logger.info(f"Overwrite pitch finished to folder:{pitch_folder}")
    service_logger.info(f"finished prepare_for_audio_file for folder:{request_work_folder}")
    if action == "pitch_only":
        f0 = merge_for_pitch_segments(pitch_folder, args, cfg)
        f0_bytes = f0.tobytes()
    else :
        # Create link to vocalist_l1's min-max folder
        src = os.path.abspath("./ckpts/svc/vocalist_l1_contentvec+whisper/data/vocalist_l1")
        dst = request_work_folder + "/vocalist_l1"
        try:
            os.symlink(src, dst)
        except Exception as e:
            service_logger.info(f"failed to create symbol link from: {src} to {dst}, exception: {e}")
        # Infer from file
        service_logger.info(f"begin infer for folder:{request_work_folder}")
        output_audio_files = infer(args, cfg, infer_type="from_file")
        service_logger.info(f"finished inferring for folder:{request_work_folder}")
        # Merge the split segments
        merge_for_audio_segments(output_audio_files, args, cfg)
        output_wav_dir = os.path.join(
            args.output_dir,
            "{}_{}.wav".format(audio_name, singger_name_mapping[args.target_singer]))
        if format == "wav":
            with open(output_wav_dir, "rb") as f:
                audio_bytes = f.read()
        elif format == "mp3":
            output_mp3_dir = os.path.join(
                args.output_dir,
                "{}_{}.wav".format(audio_name, singger_name_mapping[args.target_singer]))
            AudioSegment.from_wav(output_wav_dir).export(output_mp3_dir, format="mp3")
            with open(output_mp3_dir, "rb") as f:
                audio_bytes = f.read()
        else:
            service_logger.info(f"unsupported foramt:{format}")

    service_logger.info(f"finish processing request in folder:{request_work_folder}")
    return f0_bytes, audio_bytes


if __name__ == "__main__":
    process_request("/mnt/data2/share/raoyonghui/svc_service/Amphion/work_dir/20240202_121212", "all")
    process_request("/mnt/data2/share/raoyonghui/svc_service/Amphion/work_dir/20240202_121212", "all")
