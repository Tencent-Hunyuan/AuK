"""MLX port of AuK's BigVGAN-Flow VAE.

Only the inference path is ported: ``encode`` (waveform -> normalised latent) and
``decode`` (latent -> waveform). The normalising flow and the KL term exist only
in the training forward pass, so they are deliberately absent here.

Weight-norm parameters (``weight_g`` / ``weight_v``) are folded into plain
weights by ``convert.py``; this module expects already-folded weights.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import mlx.core as mx
from mlx import nn

from auk_mlx.layers import Activation1d, Conv1d, ConvTranspose1d


@dataclass
class VAEConfig:
    upsample_rates: list = field(default_factory=lambda: [5, 4, 3, 2, 2, 2])
    upsample_kernel_sizes: list = field(default_factory=lambda: [10, 8, 6, 4, 4, 4])
    upsample_initial_channel: int = 1536
    resblock_kernel_sizes: list = field(default_factory=lambda: [3, 7, 11])
    resblock_dilation_sizes: list = field(default_factory=lambda: [[1, 3, 5], [1, 3, 5], [1, 3, 5]])
    downsample_rates: list = field(default_factory=lambda: [2, 2, 2, 3, 4, 5])
    downsample_channels: list = field(default_factory=lambda: [12, 24, 48, 96, 192, 384, 768])
    snake_logscale: bool = True
    latent_dim: int = 64
    causal: bool = True
    act_causal: bool = True

    @classmethod
    def from_dict(cls, d: dict) -> VAEConfig:
        valid = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in valid})


def _leaky_relu(x: mx.array, slope: float = 0.2) -> mx.array:
    return mx.where(x >= 0, x, x * slope)


class ResStack(nn.Module):
    """Encoder residual stack: 4 blocks of (LReLU -> dilated conv -> LReLU -> conv)."""

    def __init__(self, channels: int, kernel_size: int = 3, base: int = 3, nums: int = 4):
        super().__init__()
        # nn.Conv1d here is the encoder's non-causal Conv1d_S (padding = dilation)
        self.layers = [
            [
                Conv1d(channels, channels, kernel_size, dilation=base**i, padding=base**i),
                Conv1d(channels, channels, kernel_size, dilation=1, padding=1),
            ]
            for i in range(nums)
        ]

    def __call__(self, x: mx.array) -> mx.array:
        for c1, c2 in self.layers:
            # LeakyReLU default slope is 0.01 inside ResStack (torch nn.LeakyReLU())
            h = c1(_leaky_relu(x, 0.01))
            h = c2(_leaky_relu(h, 0.01))
            x = x + h
        return x


class Encoder(nn.Module):
    """Waveform [B, T, 1] -> latent stats [B, T/hop, 2*latent_dim]."""

    def __init__(self, cfg: VAEConfig):
        super().__init__()
        ch = cfg.downsample_channels
        # Conv1d_S pads by dilation * (kernel_size - 1) // 2
        self.pre = Conv1d(1, ch[0], 3, padding=1)
        self.stages = []
        for (in_c, out_c), f in zip(zip(ch[:-1], ch[1:]), cfg.downsample_rates):
            k = f * 2
            self.stages.append(
                {
                    "down": Conv1d(in_c, out_c, kernel_size=k, stride=f, padding=(k - 1) // 2),
                    "stack": ResStack(out_c, 3, 2, 6),
                }
            )
        self.post = Conv1d(ch[-1], cfg.latent_dim * 2, 3, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        x = _leaky_relu(self.pre(x), 0.2)
        for st in self.stages:
            x = st["down"](x)
            x = st["stack"](x)
            x = _leaky_relu(x, 0.2)
        return self.post(x)


class AMPBlock1(nn.Module):
    """BigVGAN anti-aliased multi-periodicity block."""

    def __init__(self, channels: int, kernel_size: int, dilation: list, cfg: VAEConfig):
        super().__init__()
        causal, act_causal = cfg.causal, cfg.act_causal
        self.convs1 = [Conv1d(channels, channels, kernel_size, dilation=d, causal=causal) for d in dilation]
        self.convs2 = [Conv1d(channels, channels, kernel_size, dilation=1, causal=causal) for _ in dilation]
        # activations alternate: acts[0::2] before convs1, acts[1::2] before convs2
        self.activations = [Activation1d(channels, cfg.snake_logscale, causal=act_causal) for _ in range(2 * len(dilation))]

    def __call__(self, x: mx.array) -> mx.array:
        a1, a2 = self.activations[::2], self.activations[1::2]
        for c1, c2, act1, act2 in zip(self.convs1, self.convs2, a1, a2):
            h = c1(act1(x))
            h = c2(act2(h))
            x = x + h
        return x


class Decoder(nn.Module):
    """Latent [B, T, latent_dim] -> waveform [B, T*hop, 1]."""

    def __init__(self, cfg: VAEConfig):
        super().__init__()
        self.cfg = cfg
        causal = cfg.causal
        ch = cfg.upsample_initial_channel
        self.num_kernels = len(cfg.resblock_kernel_sizes)
        self.num_upsamples = len(cfg.upsample_rates)

        # NOTE: conv_pre is explicitly causal=False upstream (centre-padded), unlike
        # conv_post and the AMP blocks, which are causal. Getting this wrong shifts
        # the whole decoder by 6 samples and silently degrades the audio.
        self.conv_pre = Conv1d(cfg.latent_dim, ch, 7, causal=False)

        self.ups = []
        self.resblocks = []
        cur = ch
        for i, (u, k) in enumerate(zip(cfg.upsample_rates, cfg.upsample_kernel_sizes)):
            nxt = cur // 2
            self.ups.append(ConvTranspose1d(cur, nxt, k, stride=u, causal=causal))
            cur = nxt
            for kk, dd in zip(cfg.resblock_kernel_sizes, cfg.resblock_dilation_sizes):
                self.resblocks.append(AMPBlock1(cur, kk, dd, cfg))

        self.activation_post = Activation1d(cur, cfg.snake_logscale, causal=cfg.act_causal)
        self.conv_post = Conv1d(cur, 1, 7, causal=causal, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = self.ups[i](x)
            acc = None
            for j in range(self.num_kernels):
                r = self.resblocks[i * self.num_kernels + j](x)
                acc = r if acc is None else acc + r
            x = acc / self.num_kernels
        x = self.conv_post(self.activation_post(x))
        return mx.clip(x, -1.0, 1.0)


class BigVGANFlowVAE(nn.Module):
    def __init__(self, cfg: VAEConfig):
        super().__init__()
        self.cfg = cfg
        self.hop_size = math.prod(cfg.downsample_rates)
        self.audio_encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)
        self.global_mean = mx.zeros((cfg.latent_dim,))
        self.global_log_std = mx.ones((cfg.latent_dim,))

    def encode(self, wav: mx.array, *, sample_noise: bool = False) -> mx.array:
        """wav [B, T, 1] -> normalised latent [B, T/hop, latent_dim].

        The training encoder draws z = mean + eps*std. At inference AuK keeps the
        stochastic draw, so ``sample_noise`` mirrors it; the default is the mean,
        which is deterministic and what you want for reproducible output.
        """
        stats = self.audio_encoder(wav)
        d = self.cfg.latent_dim
        mean, log_std = stats[..., :d], stats[..., d:]
        z = mean + mx.random.normal(mean.shape) * mx.exp(log_std) if sample_noise else mean
        return (z - self.global_mean) / mx.sqrt(self.global_log_std)

    def denormalize(self, latents: mx.array) -> mx.array:
        return latents * mx.sqrt(self.global_log_std) + self.global_mean

    def decode(self, latents: mx.array) -> mx.array:
        """Normalised latent [B, T, D] -> waveform [B, T*hop, 1]."""
        return self.decoder(self.denormalize(latents))
