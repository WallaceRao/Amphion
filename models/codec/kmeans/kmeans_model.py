import numpy as np
import torch
import torch.nn as nn

from torch.nn import functional as F
from einops import rearrange, repeat


def compute_codebook_perplexity(indices, codebook_size):
    indices = indices.flatten()
    prob = torch.bincount(indices, minlength=codebook_size).float() / indices.size(0)
    perp = torch.exp(-torch.sum(prob * torch.log(prob + 1e-10)))
    return perp


class KMeans(nn.Module):
    def __init__(
        self,
        codebook_size=1024,
        codebook_dim=1024,
        cfg=None,
    ):
        super().__init__()
        codebook_size = (
            cfg.codebook_size
            if cfg is not None and hasattr(cfg, "codebook_size")
            else codebook_size
        )
        codebook_dim = (
            cfg.codebook_dim
            if cfg is not None and hasattr(cfg, "codebook_dim")
            else codebook_dim
        )

        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim

        self.codebook = nn.Embedding(codebook_size, codebook_dim)

    def forward(self, x):
        shape, dtype = x.shape, x.dtype  # shape: (B, T, D)
        flatten = rearrange(x, "... d -> (...) d")
        embed = self.codebook.weight.t()

        dist = -(
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ embed
            + embed.pow(2).sum(0, keepdim=True)
        )

        embed_ind = dist.max(dim=-1).indices
        embed_onehot = F.one_hot(embed_ind, self.codebook_size).type(dtype)
        embed_ind = embed_ind.view(*shape[:-1])

        quantize = self.codebook(embed_ind)

        perp = compute_codebook_perplexity(embed_ind, self.codebook_size)

        codebook_loss = F.mse_loss(quantize, x.detach(), reduction="none")

        return quantize, codebook_loss, perp


def l2norm(t):
    return F.normalize(t, p=2, dim=-1)


def ema_inplace(moving_avg, new, decay):
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))


def laplace_smoothing(x, n_categories, eps=1e-5):
    return (x + eps) / (x.sum() + n_categories * eps)


def sample_vectors(samples, num):
    num_samples, device = samples.shape[0], samples.device

    if num_samples >= num:
        indices = torch.randperm(num_samples, device=device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device=device)

    return samples[indices]


def kmeans(samples, num_clusters, num_iters=10, use_cosine_sim=False):
    dim, dtype, device = samples.shape[-1], samples.dtype, samples.device

    means = sample_vectors(samples, num_clusters)

    for _ in range(num_iters):
        print("kmeans init iter", _)
        if use_cosine_sim:
            dists = samples @ means.t()
        else:
            diffs = rearrange(samples, "n d -> n () d") - rearrange(
                means, "c d -> () c d"
            )
            dists = -(diffs**2).sum(dim=-1)

        buckets = dists.max(dim=-1).indices
        bins = torch.bincount(buckets, minlength=num_clusters)
        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)

        new_means = buckets.new_zeros(num_clusters, dim, dtype=dtype)
        new_means.scatter_add_(0, repeat(buckets, "n -> n d", d=dim), samples)
        new_means = new_means / bins_min_clamped[..., None]

        if use_cosine_sim:
            new_means = l2norm(new_means)

        means = torch.where(zero_mask[..., None], means, new_means)

    return means, bins


class KMeansEMA(nn.Module):
    def __init__(
        self,
        codebook_size=1024,
        codebook_dim=1024,
        kmeans_init=True,
        kmeans_iters=10,
        decay=0.8,
        eps=1e-5,
        weight_init=False,
        cfg=None,
    ):
        super().__init__()

        codebook_size = (
            cfg.codebook_size
            if cfg is not None and hasattr(cfg, "codebook_size")
            else codebook_size
        )
        codebook_dim = (
            cfg.codebook_dim
            if cfg is not None and hasattr(cfg, "codebook_dim")
            else codebook_dim
        )
        kmeans_iters = (
            cfg.kmeans_iters
            if cfg is not None and hasattr(cfg, "kmeans_iters")
            else kmeans_iters
        )
        decay = cfg.decay if cfg is not None and hasattr(cfg, "decay") else decay
        eps = cfg.eps if cfg is not None and hasattr(cfg, "eps") else eps

        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.kmeans_iters = kmeans_iters
        self.decay = decay
        self.eps = eps

        init_fn = torch.randn if not kmeans_init else torch.zeros
        embed = init_fn(codebook_size, codebook_dim)
        if weight_init:
            nn.init.uniform_(embed, -1 / codebook_size, 1 / codebook_size)

        # dummy emb for ddp warp
        self.dumy_emb = nn.Embedding(1, 1)

        self.register_buffer("initted", torch.Tensor([not kmeans_init]))
        self.register_buffer("cluster_size", torch.zeros(codebook_size))
        self.register_buffer("embed", embed)
        self.register_buffer("embed_avg", embed.clone())

    def init_embed_(self, data):
        embed, cluster_size = kmeans(
            data, self.codebook_size, self.kmeans_iters, use_cosine_sim=False
        )
        self.embed.data.copy_(embed)
        self.embed_avg.data.copy_(embed.clone())
        self.cluster_size.data.copy_(cluster_size)
        self.initted.data.copy_(torch.Tensor([True]))

    def replace(self, samples, mask):
        modified_codebook = torch.where(
            mask[..., None], sample_vectors(samples, self.codebook_size), self.embed
        )
        self.embed.data.copy_(modified_codebook)

    def forward(self, x):
        shape, dtype = x.shape, x.dtype
        flatten = rearrange(x, "... d -> (...) d")
        embed = self.embed.t()

        if not self.initted:
            if flatten.shape[0] >= self.codebook_size * 2:
                self.init_embed_(flatten[: self.codebook_size * 2, :])
            else:
                self.init_embed_(flatten)

        dist = -(
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ embed
            + embed.pow(2).sum(0, keepdim=True)
        )

        embed_ind = dist.max(dim=-1).indices
        embed_onehot = F.one_hot(embed_ind, self.codebook_size).type(dtype)
        embed_ind = embed_ind.view(*shape[:-1])
        quantize = F.embedding(embed_ind, self.embed)
        # quantize = self.embed(embed_ind)

        if self.training:
            ema_inplace(self.cluster_size, embed_onehot.sum(0), self.decay)
            embed_sum = flatten.t() @ embed_onehot
            ema_inplace(self.embed_avg, embed_sum.t(), self.decay)
            cluster_size = (
                laplace_smoothing(self.cluster_size, self.codebook_size, self.eps)
                * self.cluster_size.sum()
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
            self.embed.data.copy_(embed_normalized)

        perp = compute_codebook_perplexity(embed_ind, self.codebook_size)

        # codebook_loss = F.mse_loss(quantize, x.detach(), reduction="none")
        codebook_loss = None

        return quantize, codebook_loss, perp

    def quantize(self, x):
        shape, dtype = x.shape, x.dtype
        flatten = rearrange(x, "... d -> (...) d")
        embed = self.embed.t()

        if not self.initted:
            if flatten.shape[0] >= self.codebook_size * 2:
                self.init_embed_(flatten[: self.codebook_size * 2, :])
            else:
                self.init_embed_(flatten)

        dist = -(
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ embed
            + embed.pow(2).sum(0, keepdim=True)
        )

        embed_ind = dist.max(dim=-1).indices
        embed_onehot = F.one_hot(embed_ind, self.codebook_size).type(dtype)
        embed_ind = embed_ind.view(*shape[:-1])
        quantize = F.embedding(embed_ind, self.embed)

        return embed_ind, quantize


if __name__ == "__main__":
    model = KMeans()
    x = torch.randn(2, 100, 1024)
    quantize, codebook_loss, perp = model(x)
    print(quantize.shape, codebook_loss.shape, perp)
    # torch.Size([2, 10, 1024]) torch.Size([2, 10, 1024]) tensor(6.9078)

    model = KMeansEMA()
    x = torch.randn(2, 100, 1024)
    quantize, codebook_loss, perp = model(x)
    print(quantize.shape, codebook_loss, perp)
