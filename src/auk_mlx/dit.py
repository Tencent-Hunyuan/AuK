"""MLX port of AuK's Flux2Edit DiT backbone.

Two phases, as in the torch original:
  1. ``num_layers`` double-stream MMDiT blocks (text and audio keep separate
     weights but attend jointly).
  2. ``num_single_layers`` single-stream DiT blocks over ``concat(text, audio)``.

Only the inference path is ported (no gradient checkpointing, no dropout).
Rotary embeddings use ``mx.fast.rope(traditional=True)``, which was verified to
match x_transformers' ``apply_rotary_pos_emb`` to ~1e-6.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx
import numpy as np
from mlx import nn

from auk_mlx.layers import Conv1d, silu


@dataclass
class DiTConfig:
    dim: int = 1536
    heads: int = 24
    dim_head: int = 64
    ff_mult: float = 2.0
    text_hidden_dim: int = 2048
    latent_dim: int = 64
    num_layers: int = 10
    num_single_layers: int = 20

    @classmethod
    def from_dict(cls, d: dict) -> DiTConfig:
        valid = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in valid})


class _RopeTable:
    """Interleaved rotary cos/sin tables, matching x_transformers' RotaryEmbedding.

    ``inv_freq`` MUST come from the checkpoint. AuK's stored buffer is a bf16-rounded
    copy of the analytic 10000-base schedule (0.75 where the formula gives 0.74989),
    and recomputing it analytically shifts every rotation angle enough to move the
    attention logits by ~5e-3 — small per layer, but it compounds over 30 blocks.

    Angles are built in float64 and only then cast to float32; ``mx.fast.rope``
    computes them internally at lower precision and cannot take a custom inv_freq.
    """

    def __init__(self, inv_freq: np.ndarray):
        self.inv_freq = np.asarray(inv_freq, dtype=np.float64)
        self.dim_head = len(self.inv_freq) * 2
        self._cos: mx.array | None = None
        self._sin: mx.array | None = None
        self._n = 0

    def _grow(self, n: int) -> None:
        if self._cos is not None and n <= self._n:
            return
        ang = np.arange(n, dtype=np.float64)[:, None] * self.inv_freq[None, :]
        self._cos = mx.array(np.repeat(np.cos(ang), 2, axis=-1).astype(np.float32))
        self._sin = mx.array(np.repeat(np.sin(ang), 2, axis=-1).astype(np.float32))
        self._n = n

    def apply(self, x: mx.array) -> mx.array:
        """x: [B, H, N, D] -> rotated, using the interleaved (traditional) convention."""
        n = x.shape[-2]
        self._grow(n)
        cos, sin = self._cos[:n], self._sin[:n]
        x1, x2 = x[..., 0::2], x[..., 1::2]
        rot = mx.stack([-x2, x1], axis=-1).reshape(x.shape)
        return x * cos + rot * sin

    @staticmethod
    def analytic(dim_head: int, base: float = 10000.0) -> _RopeTable:
        return _RopeTable(base ** (-np.arange(0, dim_head, 2, dtype=np.float64) / dim_head))


class RMSNorm(nn.Module):
    """Matches torch.nn.RMSNorm(elementwise_affine=True) with its default eps=None,
    which resolves to torch.finfo(float32).eps rather than the usual 1e-5/1e-6."""

    TORCH_DEFAULT_EPS = 1.1920928955078125e-07

    def __init__(self, dim: int, eps: float | None = None):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = self.TORCH_DEFAULT_EPS if eps is None else eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, self.weight, self.eps)


def _layer_norm(x: mx.array) -> mx.array:
    """LayerNorm with elementwise_affine=False, eps=1e-6 (as used throughout)."""
    return mx.fast.layer_norm(x, None, None, 1e-6)


def _mish(x: mx.array) -> mx.array:
    """Mish: x * tanh(softplus(x)). softplus via logaddexp for numerical stability."""
    return x * mx.tanh(mx.logaddexp(x, mx.zeros_like(x)))


class AdaLayerNorm(nn.Module):
    """Produces the 6 modulation tensors of a DiT block."""

    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, dim * 6)

    def __call__(self, x: mx.array, emb: mx.array):
        e = self.linear(silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mx.split(e, 6, axis=1)
        x = _layer_norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class AdaLayerNormFinal(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, dim * 2)

    def __call__(self, x: mx.array, emb: mx.array) -> mx.array:
        e = self.linear(silu(emb))
        scale, shift = mx.split(e, 2, axis=1)
        return _layer_norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]


class SwiGLUFeedForward(nn.Module):
    def __init__(self, dim: int, mult: float = 2.0):
        super().__init__()
        inner = int(dim * mult)
        self.linear_in = nn.Linear(dim, inner * 2, bias=False)
        self.linear_out = nn.Linear(inner, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        x1, x2 = mx.split(self.linear_in(x), 2, axis=-1)
        return self.linear_out(silu(x1) * x2)


def _heads(x: mx.array, heads: int) -> mx.array:
    """[B, N, H*D] -> [B, H, N, D]"""
    B, N, _ = x.shape
    return x.reshape(B, N, heads, -1).transpose(0, 2, 1, 3)


def _unheads(x: mx.array) -> mx.array:
    """[B, H, N, D] -> [B, N, H*D]"""
    B, H, N, D = x.shape
    return x.transpose(0, 2, 1, 3).reshape(B, N, H * D)


class Attention(nn.Module):
    """Self-attention; when ``context_dim`` is set it also carries the joint (text) branch."""

    def __init__(self, dim: int, heads: int, dim_head: int, context_dim: int | None = None):
        super().__init__()
        self.heads = heads
        self.scale = dim_head**-0.5
        inner = heads * dim_head

        self.to_qkv = nn.Linear(dim, 3 * inner)
        self.q_norm = RMSNorm(dim_head)
        self.k_norm = RMSNorm(dim_head)
        self.to_out = [nn.Linear(inner, dim)]

        self.context_dim = context_dim
        if context_dim is not None:
            self.to_qkv_c = nn.Linear(context_dim, 3 * inner)
            self.c_q_norm = RMSNorm(dim_head)
            self.c_k_norm = RMSNorm(dim_head)
            self.to_out_c = nn.Linear(inner, context_dim)

    def _qkv(self, x: mx.array, proj, qn, kn):
        q, k, v = mx.split(proj(x), 3, axis=-1)
        q, k, v = _heads(q, self.heads), _heads(k, self.heads), _heads(v, self.heads)
        return qn(q), kn(k), v

    def single(self, x: mx.array, rope: _RopeTable, mask: mx.array | None = None) -> mx.array:
        q, k, v = self._qkv(x, self.to_qkv, self.q_norm, self.k_norm)
        q, k = rope.apply(q), rope.apply(k)
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        return self.to_out[0](_unheads(o))

    def joint(self, x: mx.array, c: mx.array, rope: _RopeTable, mask: mx.array | None = None):
        """Audio and text are roped independently, then concatenated for one attention."""
        q, k, v = self._qkv(x, self.to_qkv, self.q_norm, self.k_norm)
        cq, ck, cv = self._qkv(c, self.to_qkv_c, self.c_q_norm, self.c_k_norm)
        q, k = rope.apply(q), rope.apply(k)
        cq, ck = rope.apply(cq), rope.apply(ck)

        q = mx.concatenate([q, cq], axis=2)
        k = mx.concatenate([k, ck], axis=2)
        v = mx.concatenate([v, cv], axis=2)

        o = _unheads(mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask))
        n_x = x.shape[1]
        return self.to_out[0](o[:, :n_x]), self.to_out_c(o[:, n_x:])


class DiTBlock(nn.Module):
    """Single-stream block over the concatenated text+audio sequence."""

    def __init__(self, dim: int, heads: int, dim_head: int, ff_mult: float):
        super().__init__()
        self.attn_norm = AdaLayerNorm(dim)
        self.attn = Attention(dim, heads, dim_head)
        self.ff = SwiGLUFeedForward(dim, ff_mult)

    def __call__(self, x: mx.array, t: mx.array, rope: _RopeTable, mask: mx.array | None = None) -> mx.array:
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(x, t)
        x = x + gate_msa[:, None] * self.attn.single(norm, rope, mask)
        norm = _layer_norm(x) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        return x + gate_mlp[:, None] * self.ff(norm)


class MMDiTBlock(nn.Module):
    """Double-stream block: separate weights for text (c) and audio (x), joint attention."""

    def __init__(self, dim: int, heads: int, dim_head: int, ff_mult: float):
        super().__init__()
        self.attn_norm_c = AdaLayerNorm(dim)
        self.attn_norm_x = AdaLayerNorm(dim)
        self.attn = Attention(dim, heads, dim_head, context_dim=dim)
        self.ff_c = SwiGLUFeedForward(dim, ff_mult)
        self.ff_x = SwiGLUFeedForward(dim, ff_mult)

    def __call__(self, x: mx.array, c: mx.array, t: mx.array, rope: _RopeTable, mask: mx.array | None = None):
        norm_c, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.attn_norm_c(c, t)
        norm_x, x_gate_msa, x_shift_mlp, x_scale_mlp, x_gate_mlp = self.attn_norm_x(x, t)

        x_attn, c_attn = self.attn.joint(norm_x, norm_c, rope, mask)

        c = c + c_gate_msa[:, None] * c_attn
        norm_c = _layer_norm(c) * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
        c = c + c_gate_mlp[:, None] * self.ff_c(norm_c)

        x = x + x_gate_msa[:, None] * x_attn
        norm_x = _layer_norm(x) * (1 + x_scale_mlp[:, None]) + x_shift_mlp[:, None]
        x = x + x_gate_mlp[:, None] * self.ff_x(norm_x)
        return c, x


class ConvPositionEmbedding(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 31, groups: int = 16):
        super().__init__()
        self.conv1d = [
            Conv1d(dim, dim, kernel_size, groups=groups, padding=kernel_size // 2),
            Conv1d(dim, dim, kernel_size, groups=groups, padding=kernel_size // 2),
        ]

    def __call__(self, x: mx.array) -> mx.array:
        # x is already [B, T, C]; torch permutes because it is channels-first
        h = self.conv1d[0](x)
        h = _mish(h)
        h = self.conv1d[1](h)
        return _mish(h)


class AudioPromptEmbedding(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.conv_pos_embed = ConvPositionEmbedding(out_dim)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.linear(x)
        return self.conv_pos_embed(x) + x


class TimestepEmbedding(nn.Module):
    def __init__(self, dim: int, freq_embed_dim: int = 256):
        super().__init__()
        self.freq_embed_dim = freq_embed_dim
        self.time_mlp = [nn.Linear(freq_embed_dim, dim), nn.Linear(dim, dim)]

    def __call__(self, t: mx.array) -> mx.array:
        half = self.freq_embed_dim // 2
        e = math.log(10000) / (half - 1)
        e = mx.exp(mx.arange(half, dtype=mx.float32) * -e)
        e = 1000.0 * t[:, None] * e[None, :]
        e = mx.concatenate([mx.sin(e), mx.cos(e)], axis=-1)
        return self.time_mlp[1](silu(self.time_mlp[0](e)))


class Flux2Edit(nn.Module):
    def __init__(self, cfg: DiTConfig, inv_freq: np.ndarray | None = None):
        super().__init__()
        self.cfg = cfg
        d = cfg.dim
        # inv_freq comes from the checkpoint (see _RopeTable); the analytic fallback
        # is only for tests that build the model without weights.
        self.rope = _RopeTable(inv_freq) if inv_freq is not None else _RopeTable.analytic(cfg.dim_head)
        self.time_embed = TimestepEmbedding(d)
        self.txt_norm = RMSNorm(d)
        self.txt_proj = nn.Linear(cfg.text_hidden_dim, d)
        self.audio_embed = AudioPromptEmbedding(cfg.latent_dim, d)

        self.transformer_blocks = [MMDiTBlock(d, cfg.heads, cfg.dim_head, cfg.ff_mult) for _ in range(cfg.num_layers)]
        self.single_transformer_blocks = [DiTBlock(d, cfg.heads, cfg.dim_head, cfg.ff_mult) for _ in range(cfg.num_single_layers)]

        self.norm_out = AdaLayerNormFinal(d)
        self.proj_out = nn.Linear(d, cfg.latent_dim)

        self._text_cond: mx.array | None = None
        self._text_uncond: mx.array | None = None

    def clear_cache(self) -> None:
        self._text_cond = self._text_uncond = None

    def project_text(self, text: mx.array, drop_text: bool = False) -> mx.array:
        c = self.txt_norm(self.txt_proj(text))
        return mx.zeros_like(c) if drop_text else c

    def __call__(
        self,
        x: mx.array,  # [B, N, latent_dim] noised target
        text: mx.array,  # [B, NT, text_hidden_dim] LLM hidden states
        t: mx.array,  # [B] timestep
        ref: mx.array | None = None,  # [B, NP, latent_dim] reference latent
        cfg_infer: bool = False,
        cache: bool = True,
    ) -> mx.array:
        temb = self.time_embed(t)
        has_ref = ref is not None and ref.shape[1] > 0

        def embed(drop_audio_cond: bool):
            x_emb = self.audio_embed(x)
            if not has_ref:
                return x_emb, 0
            r = mx.zeros_like(ref) if drop_audio_cond else ref
            ref_emb = self.audio_embed(r)
            return mx.concatenate([ref_emb, x_emb], axis=1), ref_emb.shape[1]

        if cfg_infer:
            # cond / uncond stacked along the batch axis, one attention call per block
            if cache and self._text_cond is not None:
                c_cond, c_uncond = self._text_cond, self._text_uncond
            else:
                c_cond = self.project_text(text, drop_text=False)
                c_uncond = self.project_text(text, drop_text=True)
                if cache:
                    self._text_cond, self._text_uncond = c_cond, c_uncond
            x_cond, prompt_len = embed(drop_audio_cond=False)
            x_uncond, _ = embed(drop_audio_cond=True)
            h = mx.concatenate([x_cond, x_uncond], axis=0)
            c = mx.concatenate([c_cond, c_uncond], axis=0)
            temb = mx.concatenate([temb, temb], axis=0)
        else:
            c = self.project_text(text, drop_text=False)
            h, prompt_len = embed(drop_audio_cond=False)

        for block in self.transformer_blocks:
            c, h = block(h, c, temb, self.rope)

        text_len = c.shape[1]
        h = mx.concatenate([c, h], axis=1)
        for block in self.single_transformer_blocks:
            h = block(h, temb, self.rope)

        h = h[:, text_len + prompt_len :]
        return self.proj_out(self.norm_out(h, temb))
