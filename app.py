import torch
import torchaudio
import librosa
import os
import json
import glob
import numpy as np
import ssl
import gradio as gr
from inference import synthesis
random_index = 0
import random
ssl._create_default_https_context = ssl._create_unverified_context


def process(wav_path, prompt_text, synthesis_text):
    save_path = "/mnt/data2/share/raoyonghui/yuancheng/Amphion/output/result.wav"
    try:
        synthesis(wav_path, prompt_text, synthesis_text, save_path)
    except Exception as e:
        print("got exception when process, e:", e)
        return ""
    return save_path

def generate_input():
    global random_index
    audio_files = ["prompts/1.wav"]
    prompt_texts = ["第一局,还是要付出一些代价,你才能去制衡,呃"]
    target_texts = ["第一局,还是要付出一些代价,你才能去制衡,呃,制衡一下科技演员。我觉得首先,你杂技演员这个角色,我觉得该拿还是得拿,就不要,不要审。然后如果你真的想审的话,我觉得上周那个魔术师出现,我觉得后面的队伍也真香模仿嘛。我觉得魔术师这个角色,第一局抛出来也不亏。"]
    return audio_files[0], prompt_texts[0], target_texts[0]

if __name__ == "__main__":
    with gr.Blocks() as demo:
        gr.Markdown("<center><font color=red size=48>Amphion Demo of SoundStorm Voice Cloning</font></center>")
        demo_inputs =[
            gr.Audio(
                    label="Reference wav",
                    type="filepath",
                ),
            gr.Textbox(
                label="Prompt text",
                type="text",
                placeholder=""
            ),
            gr.Textbox(
                label="Synthesis text",
                type="text",
                placeholder=""
            )
        ]

        demo_outputs = gr.Audio(
                    label="Trans-Wav",
                    format="wav",
                )
        #demo = gr.Interface(
        #    fn=process,
        #    inputs=demo_inputs,
        #    outputs=demo_outputs,
        #    title="SoundStorm Clone TTS",
        #)
        btn = gr.Button("Generate Input")
        btn.click(fn=generate_input, inputs=[], outputs=[demo_inputs[0],demo_inputs[1], demo_inputs[2]])
        submit_btn = gr.Button("Submit")
        submit_btn.click(fn=process, inputs=demo_inputs, outputs=demo_outputs)

        demo.launch(share=False, server_name="0.0.0.0", server_port=80)
    #demo.launch(share=True, inbrowser=True)
