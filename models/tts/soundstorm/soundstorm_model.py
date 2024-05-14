import torch
import numpy as np
import torch.nn as nn
import math
from einops import rearrange
from models.tts.soundstorm.transformer import DiffTransformer


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


class SoundStorm(nn.Module):
    def __init__(
        self,
        num_quantizer=8,
        hidden_size=1024,
        num_layers=16,
        num_heads=16,
        codebook_size=1024,
        cfg_scale=0.15,
        mask_layer_schedule="cosine",  # "uniform", "cosine", "linear
        cfg=None,
    ):
        super().__init__()

        self.num_quantizer = num_quantizer
        self.hidden_size = hidden_size
        self.codebook_size = codebook_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.cfg_scale = cfg_scale
        self.mask_layer_schedule = mask_layer_schedule

        # conformer backbone settings
        self.diff_estimator = DiffTransformer(
            hidden_size=hidden_size,
            num_heads=16,
            num_layers=num_layers,
            dropout=0.1,
            ffn_dropout=0.1,
            attention_dropout=0.0,
        )

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

        self.reset_parameters()

    def mask_prob(self, t):
        return torch.sin(t * np.pi / 2).to(t.device)

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

        new_t = t

        return mask_layer, new_t

    @torch.no_grad()
    def forward_diffusion(self, x0, t):
        # x0: (B, T, num_quantizer)
        mask_layer, new_t = self.mask_layer(t)  # (1,)
        mask_prob = self.mask_prob(new_t)  # (B,)
        mask_token = self.mask_emb(torch.zeros_like(mask_layer))  # (1, hidden_size)

        xt = torch.zeros(x0.shape[0], x0.shape[1], self.hidden_size).to(x0.device)

        cfg_scale = self.cfg_scale

        # get prompt len
        if torch.rand(1) > cfg_scale:
            prompt_len = torch.randint(
                min(x0.shape[1] // 4, 5), x0.shape[1] // 2, (x0.shape[0],)
            ).to(
                x0.device
            )  # (B,)
        else:
            prompt_len = torch.zeros(x0.shape[0]).to(x0)  # (B,)

        # get is prompt
        is_prompt = torch.zeros_like(x0[:, :, 0])  # (B, T)
        col_indices = (
            torch.arange(is_prompt.shape[1])
            .repeat(is_prompt.shape[0], 1)
            .to(prompt_len)
        )  # (B, T)
        is_prompt[col_indices < prompt_len.unsqueeze(1)] = 1  # (B, T) 1 if prompt

        for idx, token_emb_idx in enumerate(self.token_emb):
            if idx < mask_layer:
                xt = xt + token_emb_idx(x0[:, :, idx])  # (B, T, hidden_size)

            elif idx == mask_layer:
                mask = torch.bernoulli(
                    torch.ones_like(x0[:, :, idx]) * mask_prob[..., None]
                )  # mask if 1, not mask if 0
                # prompt part don't need to be masked
                mask[is_prompt.bool()] = 0
                # Ensure at least one token is masked
                mask_num = mask[:,].sum(dim=1, keepdim=False)
                all_zero_mask = (mask_num == 0).bool()
                row_indices_to_modify = torch.nonzero(all_zero_mask)
                # mask the first token if all tokens are not masked (may mask pad if random indices)
                mask[row_indices_to_modify, prompt_len[row_indices_to_modify]] = 1

                mask = mask[..., None]  # (B, T, 1)
                xt = (
                    xt
                    + mask * mask_token[:, None, :]
                    + (1 - mask) * token_emb_idx(x0[:, :, idx])
                )  # (B, T, hidden_size)

            else:
                # prompt part don't need to be masked
                xt = (
                    xt
                    + token_emb_idx(x0[:, :, idx]) * is_prompt[..., None]
                    + mask_token * (1 - is_prompt[..., None])
                )

        return xt, new_t, mask_layer, mask, prompt_len, mask_prob

    def loss_t(self, x0, x_mask, t, cond=None):
        xt, new_t, mask_layer, mask, prompt_len, mask_prob = self.forward_diffusion(
            x0, t
        )
        # xt: (B, T, hidden_size)
        # new_t: (B,)
        # mask_layer: (1,)
        # mask: (B, T, 1)   mask if 1, not mask if 0
        # prompt_len: (B,)
        # mask_prob: (B,)

        if cond is None:
            cond = torch.zeros_like(xt).to(xt.device)  # (B, T, hidden_size)
        mask_layer_cond = self.layer_emb(mask_layer).unsqueeze(1)  # (1, 1, hidden_size)
        cond = cond + mask_layer_cond  # (B, T, hidden_size)

        embeds = self.diff_estimator(xt, new_t, cond, x_mask)  # (B, T, hidden_size)

        logits = self.to_logits[mask_layer.item()](embeds)  # (B, T, codebook_size)

        # final mask used for loss calculation
        final_mask = mask * x_mask[..., None]  # (B, T, 1)

        return logits, mask_layer, final_mask, x0, prompt_len, mask_prob

    def compute_loss(self, x0, x_mask, cond=None):
        # x0: (B, T, num_quantizer)
        # x_mask: (B, T) mask is 0 for padding
        t = torch.rand(x0.shape[0], device=x0.device, requires_grad=False)
        t = torch.clamp(t, 1e-5, 1.0)
        return self.loss_t(x0, x_mask, t, cond)

    def reset_parameters(self):
        def _reset_parameters(m):
            if isinstance(m, nn.MultiheadAttention):
                if m._qkv_same_embed_dim:
                    nn.init.normal_(m.in_proj_weight, std=0.02)
                else:
                    nn.init.normal_(m.q_proj_weight, std=0.02)
                    nn.init.normal_(m.k_proj_weight, std=0.02)
                    nn.init.normal_(m.v_proj_weight, std=0.02)

                if m.in_proj_bias is not None:
                    nn.init.constant_(m.in_proj_bias, 0.0)
                    nn.init.constant_(m.out_proj.bias, 0.0)
                if m.bias_k is not None:
                    nn.init.xavier_normal_(m.bias_k)
                if m.bias_v is not None:
                    nn.init.xavier_normal_(m.bias_v)

            elif (
                isinstance(m, nn.Conv1d)
                or isinstance(m, nn.ConvTranspose1d)
                or isinstance(m, nn.Conv2d)
                or isinstance(m, nn.ConvTranspose2d)
            ):
                m.weight.data.normal_(0.0, 0.02)

            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(mean=0.0, std=0.02)
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.Embedding):
                m.weight.data.normal_(mean=0.0, std=0.02)
                if m.padding_idx is not None:
                    m.weight.data[m.padding_idx].zero_()

        self.apply(_reset_parameters)


# if __name__ == "__main__":
#     model = SoundStorm()
#     x0 = torch.randint(0, 1024, (4, 1000, 8))
#     x_mask = torch.ones(4, 1000)
#     cond = torch.randn(4, 1000, 1024)
#     logits, mask_layer, final_mask, x0, prompt_len, mask_prob = model.compute_loss(x0, x_mask, cond)
#     print(logits.shape)
#     print(mask_layer)
#     print(final_mask.shape)
#     print("prompt_len:", prompt_len)
#     print("mask tokens:", final_mask.squeeze(2).sum(-1))
#     print(x0.shape)
#     print(mask_prob)
