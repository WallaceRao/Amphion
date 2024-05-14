import torch
import numpy as np
import torch.nn as nn
import math
from einops import rearrange


def top_k(logits, thres=0.9):
    k = math.ceil((1 - thres) * logits.shape[-1])
    val, ind = logits.topk(k, dim=-1)
    probs = torch.full_like(logits, float("-inf"))
    probs.scatter_(2, ind, val)
    return probs


def log(t, eps=1e-10):
    return torch.log(t + eps)


def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def gumbel_sample(t, temperature=1.0, dim=-1):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim=dim)


class SoundStorm(nn.module):
    def __init__(self, num_quantizer=8, hidden_size=1024, codebook_size=1024, cfg=None):
        super().__init__()

        self.num_quantizer = num_quantizer
        self.hidden_size = hidden_size
        self.codebook_size = codebook_size

        # conformer backbone settings
        self.diff_diff_estimator = ...

        self.layer_emb = nn.Embedding(self.num_quantizer, self.hidden_size)
        self.mask_emb = nn.Embedding(1, self.hidden_size)

        self.token_emb = torch.nn.ModuleList(
            [
                nn.Embedding(self.codebook_size, self.hidden_size)
                for _ in range(self.num_quantizer)
            ]
        )

        self.to_logits = torch.nn.ModuleList(
            [
                nn.Linear(self.hidden_size, self.codebook_size)
                for _ in range(self.num_quantizer)
            ]
        )

    def mask_prob(self, t):
        return torch.sin(t * np.pi / 2)

    def mask_layer(self, t):
        if self.mask_layer_schedule == "uniform":
            mask_layer = torch.randint(0, self.num_quantizer, (1,)).to(t.device)
        elif self.mask_layer_schedule == "cosine":
            weights = torch.tensor(
                [
                    np.cos(i / self.num_quantizer * np.pi / 2)
                    for i in range(self.num_quantizer)
                ]
            )
            mask_layer = torch.multinomial(weights, 1).to(t.device)
        elif self.mask_layer_schedule == "linear":
            weights = torch.tensor(
                [self.num_quantizer - i for i in range(self.num_quantizer)]
            )
            weights = weights / weights.sum()
            mask_layer = torch.multinomial(weights, 1).to(t.device)

        return mask_layer, t

    @torch.no_grad()
    def forward_diffusion(self, x0, t):
        # x0: (B, T, num_quantizer)
        ...
