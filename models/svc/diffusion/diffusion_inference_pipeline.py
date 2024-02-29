# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import tensorrt as trt
import numpy as np
from diffusers import DiffusionPipeline
import os

class DiffusionInferencePipeline(DiffusionPipeline):
    def __init__(self, network, scheduler, acoustic_mapper_trt, num_inference_timesteps=1000):
        super().__init__()
        self.register_modules(network=network, scheduler=scheduler)
        self.num_inference_timesteps = num_inference_timesteps
        logger = trt.Logger(trt.Logger.ERROR)
        runtime = trt.Runtime(logger)
        self.ctx = None
        if len(acoustic_mapper_trt):
            with open(acoustic_mapper_trt, "rb") as f:
                self.acoustic_mapper_trt_engine = runtime.deserialize_cuda_engine(f.read())
            self.trt_context = self.acoustic_mapper_trt_engine.create_execution_context()

    def acoustic_mapper_trt_infer(self, mel, timestep, conditioner):
        model_output = torch.zeros(mel.shape, dtype=torch.float32).cuda()
        for i in range(3):
            if i == 3:
                self.trt_context.set_binding_shape(i, self.trt_context.get_binding_shape(0))
            else:
                binding_shape = self.trt_context.get_binding_shape(i)
                binding_shape[0] = mel.shape[0]
                self.trt_context.set_binding_shape(i, binding_shape)
        trt_bindings=[int(mel.data_ptr()),
                      int(timestep.data_ptr()),
                      int(conditioner.data_ptr()),
                      int(model_output.data_ptr())]
        self.trt_context.execute_v2(bindings=trt_bindings)
        return model_output

    @torch.inference_mode()
    def __call__(
        self,
        initial_noise: torch.Tensor,
        conditioner: torch.Tensor = None,
    ):
        r"""
        Args:
            initial_noise: The initial noise to be denoised.
            conditioner:The conditioner.
            n_inference_steps: The number of denoising steps. More denoising steps
                usually lead to a higher quality at the expense of slower inference.
        """
        mel = initial_noise
        batch_size = mel.size(0)
        self.scheduler.set_timesteps(self.num_inference_timesteps)
        if self.acoustic_mapper_trt_engine:
            print("predict noise model_output with tensorrt")
        for t in self.progress_bar(self.scheduler.timesteps):
            model_output = None
            # 1. predict noise model_output
            if self.acoustic_mapper_trt_engine:
                timestep = torch.full((batch_size,), t, device=mel.device, dtype=torch.int32)
                model_output = self.acoustic_mapper_trt_infer(mel, timestep, conditioner)
            else:
                timestep = torch.full((batch_size,), t, device=mel.device, dtype=torch.long)
                model_output = self.network(mel, timestep, conditioner)
            # 2. denoise, compute previous step: x_t -> x_t-1
            mel = self.scheduler.step(model_output, t, mel).prev_sample
            # 3. clamp
            mel = mel.clamp(-1.0, 1.0)
        return mel
