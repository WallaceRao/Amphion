import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
import os
from IPython.display import Audio
import matplotlib.pyplot as plt
import soundfile as sf
import pickle

from models.codec.kmeans.kmeans_model import KMeans, KMeansEMA
from models.tts.soundstorm.soundstorm_model import SoundStorm
from models.codec.amphion_codec.codec import CodecEncoder, CodecDecoder
from models.tts.text2semantic.t2s_model import T2SLlama
from transformers import Wav2Vec2BertModel
import safetensors
from utils.util import load_config
import wave

from utils.g2p_new.g2p import phonemizer_g2p
LANG2CODE = {
    'zh': 349,
    'en': 350,
    'ja': 351,
    'ko': 352,
    'fr': 353,
    'de': 354,
}
def g2p(text, language):
    return phonemizer_g2p(text, language)


cfg = load_config("egs/tts/SoundStorm/exp_config_16k_emilia_llama_new_semantic.json")
t2s_cfg = load_config("egs/tts/Text2Semantic/exp_config_16k_emilia_new_semantic.json")

def build_soundstorm(cfg, pretrained_path, device):
    soundstorm_model = SoundStorm(cfg=cfg.model.soundstorm)
    # if ".bin" in pretrained_path:
    #     soundstorm_model .load_state_dict(torch.load(pretrained_path))
    # elif ".safetensors" in pretrained_path:
    #     safetensors.torch.load_model(soundstorm_model, pretrained_path)
    soundstorm_model.eval()
    soundstorm_model.to(device)
    return soundstorm_model

def build_kmeans_model(cfg, device):
    if cfg.model.kmeans.type == "kmeans":
        kmeans_model = KMeans(cfg=cfg.model.kmeans.kmeans)
    elif cfg.model.kmeans.type == "kmeans_ema":
        kmeans_model = KMeansEMA(cfg=cfg.model.kmeans.kmeans)
    kmeans_model.eval()
    pretrained_path =cfg.model.kmeans.pretrained_path
    if ".bin" in pretrained_path:
        kmeans_model.load_state_dict(torch.load(pretrained_path))
    elif ".safetensors" in pretrained_path:
        safetensors.torch.load_model(kmeans_model, pretrained_path)
    kmeans_model.to(device)
    return kmeans_model

def build_semantic_model(cfg, device):
    semantic_model = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0")
    semantic_model.eval()
    semantic_model.to(device)
    # layer_idx = 15
    # if layer_idx == 23:
    #     output_idx = 0
    # else:
    #     output_idx = layer_idx + 2
    layer_idx = 15
    output_idx = 17
    stat_mean_var = torch.load(cfg.model.kmeans.stat_mean_var_path)
    semantic_mean = stat_mean_var["mean"]
    semantic_std = torch.sqrt(stat_mean_var["var"])
    semantic_mean = semantic_mean.to(device)
    semantic_std = semantic_std.to(device)
    # print(
    #     "semantic mean: ", semantic_mean, "semantic std: ", semantic_std
    # )
    return semantic_model, semantic_mean, semantic_std

def build_codec_model(cfg, device):
    codec_encoder = CodecEncoder(cfg=cfg.model.codec.encoder)
    codec_decoder = CodecDecoder(cfg=cfg.model.codec.decoder)
    codec_encoder.load_state_dict(
        torch.load(cfg.model.codec.encoder.pretrained_path)
    )
    codec_decoder.load_state_dict(
        torch.load(cfg.model.codec.decoder.pretrained_path)
    )
    # codec_decoder = codec_decoder.quantizer  # we only need the quantizer
    codec_encoder.eval()
    codec_decoder.eval()
    codec_encoder.to(device)
    codec_decoder.to(device)
    return codec_encoder, codec_decoder

def build_t2s_model(cfg, device):
    t2s_model = T2SLlama(cfg=cfg.model.t2sllama)
    t2s_model.eval()
    t2s_model.to(device)
    return t2s_model

device = torch.device("cuda")
soundstorm_pretrained_path = "/mnt/data2/share/raoyonghui/yuancheng/SoundStorm/ckpt:soundstorm:soundstorm_16k_kmeans_2048_emilia_50k_llama_new_semantic:checkpoint:epoch-0011_step-0174000_loss-4.735856/model.safetensors"
soundstorm_model = build_soundstorm(cfg, soundstorm_pretrained_path, device)
semantic_model, semantic_mean, semantic_std = build_semantic_model(cfg, device)
kmeans_model = build_kmeans_model(cfg, device)
codec_encoder, codec_decoder = build_codec_model(cfg, device)
t2s_model = build_t2s_model(t2s_cfg, device)


semantic_mean = semantic_mean.to(device)
semantic_std = semantic_std.to(device)

safetensors.torch.load_model(soundstorm_model, "/mnt/data2/share/raoyonghui/yuancheng/SoundStorm/ckpt:soundstorm:soundstorm_16k_kmeans_2048_emilia_50k_llama_new_semantic:checkpoint:epoch-0011_step-0174000_loss-4.735856/model.safetensors")
safetensors.torch.load_model(t2s_model, "/mnt/data2/share/raoyonghui/yuancheng/SoundStorm/ckpt:text2semantic:t2s_16k_kmeans_2048_emilia_50k_new_semantic:checkpoint:epoch-0006_step-0296000_loss-1.705253/model.safetensors")


from transformers import SeamlessM4TFeatureExtractor
processor = SeamlessM4TFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")


@torch.no_grad()
def extract_acoustic_code(speech):
    vq_emb = codec_encoder(speech.unsqueeze(1))
    _, vq, _, _, _ = codec_decoder.quantizer(vq_emb)
    acoustic_code = vq.permute(
        1, 2, 0
    )  # (num_quantizer, T, C) -> (T, C, num_quantizer)
    return acoustic_code

@torch.no_grad()
def extract_semantic_code(semantic_mean, semantic_std, input_features, attention_mask):
    vq_emb = semantic_model(
        input_features=input_features,
        attention_mask=attention_mask,
        output_hidden_states=True,
    )
    feat = vq_emb.hidden_states[17]  # (B, T, C)
    feat = (feat - semantic_mean.to(feat)) / semantic_std.to(feat)

    semantic_code, _ = kmeans_model.quantize(feat)  # (B, T)
    return semantic_code

@torch.no_grad()
def extract_features(speech, processor):
    inputs = processor(speech, sampling_rate=16000, return_tensors="pt")
    input_features = inputs["input_features"][0]
    attention_mask = inputs["attention_mask"][0]
    return input_features, attention_mask

def synthesis(prompt_wav_path, prompt_text, target_text, output_wav_path):
    #prompt_wav_path = "./en.wav"
    prompt_speech, sr = librosa.load(prompt_wav_path, sr=16000)
    #prompt_speech = prompt_speech[0:48000]
    uid = prompt_wav_path.split("/")[-1].split(".")[0]
    prompt_phone_id = g2p(prompt_text, 'zh')[1]   # use 'en' if prompt text is en
    prompt_phone_id = torch.tensor(prompt_phone_id, dtype=torch.long)
    prompt_phone_id = torch.cat([torch.tensor(LANG2CODE['en'], dtype=torch.long).reshape(1), prompt_phone_id])
    target_phone_id = g2p(target_text, 'zh')[1]   # use 'en' if target text is en
    target_phone_id = torch.tensor(target_phone_id, dtype=torch.long)
    phone_id = torch.cat([prompt_phone_id, target_phone_id])
    device = torch.device("cuda")
    phone_id = phone_id.to(device)
    text = prompt_text + target_text
    Audio(prompt_speech, rate=16000)


    input_fetures, attention_mask = extract_features(prompt_speech, processor)
    input_fetures = input_fetures.unsqueeze(0).to(device)
    attention_mask = attention_mask.unsqueeze(0).to(device)
    prompt_semantic_code = extract_semantic_code(semantic_mean, semantic_std, input_fetures, attention_mask)

    predict_semantic = t2s_model.sample_hf(phone_ids=phone_id.unsqueeze(0), prompt_ids=prompt_semantic_code, temperature=1.0, top_k=100, top_p=0.8)
    semantic_code = torch.cat([prompt_semantic_code, predict_semantic], dim=-1)


    acoustic_code = extract_acoustic_code(torch.tensor(prompt_speech).unsqueeze(0).to(device))
    print(acoustic_code.shape)
    cond = soundstorm_model.cond_emb(semantic_code.to(device))
    print(cond.shape)


    prompt = acoustic_code
    predict = soundstorm_model.reverse_diffusion(cond=cond, prompt=prompt, temp=1.5, filter_thres=0.98, n_timesteps=[50, 10, 1, 1, 1, 1, 1, 1], cfg=1.0, rescale_cfg=1.0)
    print(predict.shape)


    vq_emb = codec_decoder.vq2emb(predict.permute(2,0,1))
    recovered_audio = codec_decoder(vq_emb)
    recovered_audio = recovered_audio[0][0].cpu().detach().numpy()
    Audio(recovered_audio, rate=16000)

    prompt_vq_emb = codec_decoder.vq2emb(prompt.permute(2,0,1))
    recovered_prompt_audio = codec_decoder(prompt_vq_emb)
    recovered_prompt_audio = recovered_prompt_audio[0][0].cpu().detach().numpy()
    Audio(recovered_prompt_audio, rate=16000)

    combine_audio = np.concatenate([recovered_prompt_audio, recovered_audio])

    print("samples count:", len(combine_audio))
    pcm_bytes = combine_audio.tobytes()
    #result_file = "/mnt/data2/share/raoyonghui/yuancheng/Amphion/output/result.wav"
    with wave.open(output_wav_path, "wb") as out_f:
        out_f.setnchannels(1)
        out_f.setsampwidth(2)
        out_f.setframerate(16000)
        out_f.writeframesraw(pcm_bytes)
    return output_wav_path
