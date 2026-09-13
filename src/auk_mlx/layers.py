"""Shared MLX building blocks for the AuK port.

MLX convolutions are channels-last (NLC) while PyTorch is channels-first (NCL).
Every conv weight is therefore transposed at load time, never at call time:

    Conv1d            torch (O, I, K)  ->  mlx (O, K, I)      .transpose(0, 2, 1)
    ConvTranspose1d   torch (I, O, K)  ->  mlx (O, K, I)      .transpose(1, 2, 0)
    depthwise ConvT   torch (C, 1, K)  ->  mlx (C, K, 1)      .transpose(0, 2, 1)

The layouts above were verified numerically against torch (max abs diff ~1e-6);
see tests/test_parity.py.
"""

from __future__ import annotations

import math

import mlx.core as mx
from mlx import nn


def silu(x: mx.array) -> mx.array:
    return x * mx.sigmoid(x)


class Conv1d(nn.Module):
    """Channels-last Conv1d. Set ``causal`` to left-pad instead of centre-pad."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        causal: bool = False,
        padding: int | None = None,
    ):
        super().__init__()
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.causal = causal

        if causal:
            self.left_padding = dilation * (kernel_size - 1)
            self.padding = 0
        else:
            self.left_padding = 0
            # torch's get_padding(k, d) == int((k * d - d) / 2)
            self.padding = int((kernel_size * dilation - dilation) / 2) if padding is None else padding

        self.weight = mx.zeros((out_channels, kernel_size, in_channels // groups))
        self.bias = mx.zeros((out_channels,)) if bias else None

    def __call__(self, x: mx.array) -> mx.array:
        # x: [B, T, C]
        if self.causal and self.left_padding:
            x = mx.pad(x, [(0, 0), (self.left_padding, 0), (0, 0)])
        y = mx.conv1d(x, self.weight, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
        if self.bias is not None:
            y = y + self.bias
        return y


class ConvTranspose1d(nn.Module):
    """Channels-last ConvTranspose1d; ``causal`` trims the trailing ``stride`` frames."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        bias: bool = True,
        causal: bool = False,
        padding: int | None = None,
    ):
        super().__init__()
        self.stride = stride
        self.causal = causal
        if causal:
            assert kernel_size == 2 * stride, "causal ConvTranspose1d requires kernel_size == 2 * stride"
            self.padding = 0
        else:
            self.padding = (kernel_size - stride) // 2 if padding is None else padding

        self.weight = mx.zeros((out_channels, kernel_size, in_channels))
        self.bias = mx.zeros((out_channels,)) if bias else None

    def __call__(self, x: mx.array) -> mx.array:
        y = mx.conv_transpose1d(x, self.weight, stride=self.stride, padding=self.padding)
        if self.bias is not None:
            y = y + self.bias
        if self.causal:
            y = y[:, : -self.stride, :]
        return y


class SnakeBeta(nn.Module):
    """x + 1/b * sin^2(a*x), with per-channel a, b held in log space."""

    def __init__(self, channels: int, alpha_logscale: bool = True):
        super().__init__()
        self.alpha_logscale = alpha_logscale
        self.alpha = mx.zeros((channels,))
        self.beta = mx.zeros((channels,))

    def __call__(self, x: mx.array) -> mx.array:
        # x: [B, T, C] -- alpha/beta broadcast over the channel axis
        alpha, beta = self.alpha, self.beta
        if self.alpha_logscale:
            alpha = mx.exp(alpha)
            beta = mx.exp(beta)
        s = mx.sin(x * alpha)
        return x + (1.0 / (beta + 1e-9)) * s * s


def kaiser_sinc_filter1d(cutoff: float, half_width: float, kernel_size: int) -> mx.array:
    """Windowed-sinc low-pass filter, matching alias_free_torch. Returns [K]."""
    even = kernel_size % 2 == 0
    half_size = kernel_size // 2

    delta_f = 4 * half_width
    A = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95
    if A > 50.0:
        beta = 0.1102 * (A - 8.7)
    elif A >= 21.0:
        beta = 0.5842 * (A - 21) ** 0.4 + 0.07886 * (A - 21.0)
    else:
        beta = 0.0

    # Kaiser window: I0(beta * sqrt(1 - r^2)) / I0(beta), r spanning [-1, 1].
    # MLX has no kaiser_window, so build it from the series expansion of I0.
    n = mx.arange(kernel_size, dtype=mx.float32)
    r = (n - (kernel_size - 1) / 2.0) / ((kernel_size - 1) / 2.0)
    arg = beta * mx.sqrt(mx.maximum(1.0 - r * r, 0.0))
    window = _i0(arg) / _i0(mx.array(beta, dtype=mx.float32))

    if even:
        time = mx.arange(-half_size, half_size, dtype=mx.float32) + 0.5
    else:
        time = mx.arange(kernel_size, dtype=mx.float32) - half_size

    if cutoff == 0:
        return mx.zeros((kernel_size,))
    # sinc(x) = sin(pi x) / (pi x), with the removable singularity at 0 filled in
    arg2 = 2 * cutoff * time
    pi_arg = math.pi * arg2
    sinc = mx.where(
        mx.abs(arg2) < 1e-12, mx.ones_like(arg2), mx.sin(pi_arg) / mx.where(mx.abs(pi_arg) < 1e-12, mx.ones_like(pi_arg), pi_arg)
    )
    filt = 2 * cutoff * window * sinc
    return filt / mx.sum(filt)


def _i0(x: mx.array) -> mx.array:
    """Modified Bessel function I0 via its power series (ample for kaiser betas <= ~20)."""
    total = mx.ones_like(x)
    term = mx.ones_like(x)
    quarter_sq = (x * x) / 4.0
    for k in range(1, 40):
        term = term * quarter_sq / (k * k)
        total = total + term
    return total


class UpSample1d(nn.Module):
    """2x alias-free upsampling: replicate-pad, transposed sinc conv, trim."""

    def __init__(self, ratio: int = 2, kernel_size: int = 12, causal: bool = False):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = kernel_size
        self.stride = ratio
        self.causal = causal
        if causal:
            self.pad = 0
        else:
            self.pad = kernel_size // ratio - 1
            self.pad_left = self.pad * self.stride + (kernel_size - self.stride) // 2
            self.pad_right = self.pad * self.stride + (kernel_size - self.stride + 1) // 2
        self._filter = kaiser_sinc_filter1d(0.5 / ratio, 0.6 / ratio, kernel_size)

    def __call__(self, x: mx.array) -> mx.array:
        # x: [B, T, C]; depthwise transposed conv, so weight is (C, K, 1)
        C = x.shape[-1]
        if self.pad:
            x = mx.pad(x, [(0, 0), (self.pad, self.pad), (0, 0)], mode="edge")
        w = mx.broadcast_to(self._filter.reshape(1, self.kernel_size, 1), (C, self.kernel_size, 1))
        y = self.ratio * mx.conv_transpose1d(x, w, stride=self.stride, groups=C)
        if self.causal:
            y = y[:, : -(self.kernel_size - self.stride), :]
        else:
            y = y[:, self.pad_left : -self.pad_right, :]
        return y


class DownSample1d(nn.Module):
    """2x alias-free downsampling: replicate-pad then strided depthwise sinc conv."""

    def __init__(self, ratio: int = 2, kernel_size: int = 12, causal: bool = False):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = kernel_size
        self.causal = causal
        if causal:
            self.pad_left, self.pad_right = kernel_size - 1, 0
        else:
            even = kernel_size % 2 == 0
            self.pad_left = kernel_size // 2 - int(even)
            self.pad_right = kernel_size // 2
        self._filter = kaiser_sinc_filter1d(0.5 / ratio, 0.6 / ratio, kernel_size)

    def __call__(self, x: mx.array) -> mx.array:
        C = x.shape[-1]
        x = mx.pad(x, [(0, 0), (self.pad_left, self.pad_right), (0, 0)], mode="edge")
        w = mx.broadcast_to(self._filter.reshape(1, self.kernel_size, 1), (C, self.kernel_size, 1))
        return mx.conv1d(x, w, stride=self.ratio, groups=C)


class Activation1d(nn.Module):
    """Anti-aliased activation: upsample 2x, apply, downsample 2x."""

    def __init__(self, channels: int, alpha_logscale: bool = True, causal: bool = False):
        super().__init__()
        self.act = SnakeBeta(channels, alpha_logscale=alpha_logscale)
        self.upsample = UpSample1d(2, 12, causal=False)
        self.downsample = DownSample1d(2, 12, causal=causal)

    def __call__(self, x: mx.array) -> mx.array:
        return self.downsample(self.act(self.upsample(x)))
