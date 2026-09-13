"""MLX port of the Qwen2.5-Omni Thinker encoder used by AuK as a text/audio encoder.

AuK never generates with this model: it runs one forward pass and consumes every
layer's hidden states, which an ELMo-style weighted sum collapses into the
conditioning tensor. So this port is encode-only -- no KV cache, no sampling, no
lm_head, no talker, no vision tower.

Two pieces:
  * ``AudioTower``   - Whisper-style conv frontend + 32 bidirectional layers.
  * ``TextModel``    - Qwen2 decoder, 36 layers, GQA (16 q heads / 2 kv heads).

Positions: Qwen2.5-Omni uses M-RoPE (3 sections over t/h/w). For text-only and
text+audio input there is no spatial extent, so all three sections carry the same
position index and M-RoPE collapses to ordinary 1-D RoPE. This port implements
that collapsed case and rejects image/video input rather than pretending to
support it.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import mlx.core as mx
import numpy as np
from mlx import nn


@dataclass
class TextConfig:
    hidden_size: int = 2048
    num_hidden_layers: int = 36
    num_attention_heads: int = 16
    num_key_value_heads: int = 2
    intermediate_size: int = 11008
    vocab_size: int = 151936
    rope_theta: float = 1000000.0
    rms_norm_eps: float = 1e-6
    head_dim: int | None = None

    @property
    def dim_head(self) -> int:
        return self.head_dim or self.hidden_size // self.num_attention_heads


@dataclass
class AudioConfig:
    d_model: int = 1280
    encoder_layers: int = 32
    encoder_attention_heads: int = 20
    encoder_ffn_dim: int = 5120
    output_dim: int = 2048
    n_window: int = 100
    num_mel_bins: int = 128
    scale_embedding: bool = False


def _rms_norm(x: mx.array, w: mx.array, eps: float) -> mx.array:
    return mx.fast.rms_norm(x, w, eps)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return _rms_norm(x, self.weight, self.eps)


class TextAttention(nn.Module):
    """Qwen2 GQA self-attention. q/k/v carry biases, o_proj does not."""

    def __init__(self, cfg: TextConfig):
        super().__init__()
        self.n_heads = cfg.num_attention_heads
        self.n_kv = cfg.num_key_value_heads
        d = cfg.dim_head
        self.dim_head = d
        self.scale = d**-0.5
        self.q_proj = nn.Linear(cfg.hidden_size, self.n_heads * d, bias=True)
        self.k_proj = nn.Linear(cfg.hidden_size, self.n_kv * d, bias=True)
        self.v_proj = nn.Linear(cfg.hidden_size, self.n_kv * d, bias=True)
        self.o_proj = nn.Linear(self.n_heads * d, cfg.hidden_size, bias=False)

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array, mask: mx.array | None) -> mx.array:
        B, N, _ = x.shape
        q = self.q_proj(x).reshape(B, N, self.n_heads, self.dim_head).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, N, self.n_kv, self.dim_head).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, N, self.n_kv, self.dim_head).transpose(0, 2, 1, 3)

        q, k = _apply_rope_half(q, cos, sin), _apply_rope_half(k, cos, sin)
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        o = o.transpose(0, 2, 1, 3).reshape(B, N, -1)
        return self.o_proj(o)


def _apply_rope_half(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """HF "half" rotary convention: rotate_half splits the head dim in two halves.

    This is NOT the interleaved convention the AuK DiT uses -- HF's Qwen2 does
    ``cat(-x2, x1)`` over halves, so mx.fast.rope(traditional=True) is wrong here.
    """
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    rot = mx.concatenate([-x2, x1], axis=-1)
    return x * cos + rot * sin


class TextMLP(nn.Module):
    def __init__(self, cfg: TextConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        g = self.gate_proj(x)
        return self.down_proj((g * mx.sigmoid(g)) * self.up_proj(x))


class TextLayer(nn.Module):
    def __init__(self, cfg: TextConfig):
        super().__init__()
        self.self_attn = TextAttention(cfg)
        self.mlp = TextMLP(cfg)
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array, mask: mx.array | None) -> mx.array:
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, mask)
        return x + self.mlp(self.post_attention_layernorm(x))


class AudioAttention(nn.Module):
    """Whisper-style bidirectional attention; k_proj has no bias."""

    def __init__(self, cfg: AudioConfig):
        super().__init__()
        self.n_heads = cfg.encoder_attention_heads
        self.dim_head = cfg.d_model // cfg.encoder_attention_heads
        self.scale = self.dim_head**-0.5
        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=True)
        self.k_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=True)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=True)

    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        B, N, _ = x.shape
        shape = (B, N, self.n_heads, self.dim_head)
        q = self.q_proj(x).reshape(shape).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(shape).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(shape).transpose(0, 2, 1, 3)
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        return self.out_proj(o.transpose(0, 2, 1, 3).reshape(B, N, -1))


class AudioLayer(nn.Module):
    def __init__(self, cfg: AudioConfig):
        super().__init__()
        self.self_attn = AudioAttention(cfg)
        self.self_attn_layer_norm = nn.LayerNorm(cfg.d_model)
        self.fc1 = nn.Linear(cfg.d_model, cfg.encoder_ffn_dim)
        self.fc2 = nn.Linear(cfg.encoder_ffn_dim, cfg.d_model)
        self.final_layer_norm = nn.LayerNorm(cfg.d_model)

    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        x = x + self.self_attn(self.self_attn_layer_norm(x), mask)
        h = self.final_layer_norm(x)
        return x + self.fc2(nn.gelu(self.fc1(h)))


def _sinusoids(length: int, channels: int, max_timescale: float = 10000.0) -> mx.array:
    log_ts = np.log(max_timescale) / (channels // 2 - 1)
    inv = np.exp(-log_ts * np.arange(channels // 2, dtype=np.float64))
    scaled = np.arange(length, dtype=np.float64)[:, None] * inv[None, :]
    return mx.array(np.concatenate([np.sin(scaled), np.cos(scaled)], axis=1).astype(np.float32))


class AudioTower(nn.Module):
    """Mel spectrogram -> audio token embeddings, matching Qwen2_5OmniAudioEncoder.

    The encoder is *windowed*, not a plain Whisper stack: the mel frames are split
    into chunks of ``2 * n_window``, each chunk is convolved and positionally
    embedded independently, and attention is block-diagonal so no chunk attends to
    another. After the layers, an avg-pool of stride 2 halves the length again, so
    the final token rate is one per 4 mel frames.
    """

    def __init__(self, cfg: AudioConfig):
        super().__init__()
        self.cfg = cfg
        self.n_window = cfg.n_window
        self.conv1 = nn.Conv1d(cfg.num_mel_bins, cfg.d_model, 3, padding=1)
        self.conv2 = nn.Conv1d(cfg.d_model, cfg.d_model, 3, stride=2, padding=1)
        self.layers = [AudioLayer(cfg) for _ in range(cfg.encoder_layers)]
        self.ln_post = nn.LayerNorm(cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.output_dim)
        self._pos: mx.array | None = None

    def _positional(self, n: int) -> mx.array:
        if self._pos is None or self._pos.shape[0] < n:
            self._pos = _sinusoids(max(n, self.n_window), self.cfg.d_model)
        return self._pos[:n]

    def __call__(self, mel: mx.array, feature_len: int | None = None) -> mx.array:
        """mel: [1, T, n_mel] channels-last. Returns [T_tok, output_dim]."""
        T = mel.shape[1] if feature_len is None else feature_len
        mel = mel[:, :T]

        win = self.n_window * 2
        # chunk lengths: full windows, with the remainder carried by the last chunk
        lens = [win] * (T // win)
        rest = T % win
        if rest:
            lens.append(rest)
        elif not lens:
            lens = [T]

        # Convolve each chunk on its own, right-padded to the window size, so the
        # conv receptive field never crosses a chunk boundary.
        embeds, keep = [], []
        for off, L in zip(np.cumsum([0] + lens[:-1]), lens):
            c = mel[:, int(off) : int(off) + L]
            if L < win:
                c = mx.pad(c, [(0, 0), (0, win - L), (0, 0)])
            e = nn.gelu(self.conv1(c))
            if L < win:
                m = mx.array(np.concatenate([np.ones(L, np.float32), np.zeros(win - L, np.float32)]))
                e = e * m[None, :, None]
            e = nn.gelu(self.conv2(e))
            e = e + self._positional(e.shape[1])[None]
            embeds.append(e)
            keep.append((L + 1) // 2)  # conv2 stride-2 output length for this chunk

        h = mx.concatenate([e[:, :k] for e, k in zip(embeds, keep)], axis=1)

        # block-diagonal mask: bidirectional inside a chunk, nothing across chunks
        n = h.shape[1]
        blk = np.full((n, n), -np.inf, dtype=np.float32)
        pos = 0
        for k in keep:
            blk[pos : pos + k, pos : pos + k] = 0.0
            pos += k
        mask = mx.array(blk)[None, None]

        for layer in self.layers:
            h = layer(h, mask)

        # avg_pooler: AvgPool1d(kernel=2, stride=2) per audio, then norm + proj
        h = h[0]
        n2 = (h.shape[0] // 2) * 2
        h = h[:n2].reshape(n2 // 2, 2, -1).mean(axis=1)
        return self.proj(self.ln_post(h))


class ThinkerEncoder(nn.Module):
    """Qwen2.5-Omni Thinker, encode-only, returning every layer's hidden states."""

    def __init__(self, text_cfg: TextConfig, audio_cfg: AudioConfig | None = None):
        super().__init__()
        self.cfg = text_cfg
        self.embed_tokens = nn.Embedding(text_cfg.vocab_size, text_cfg.hidden_size)
        self.layers = [TextLayer(text_cfg) for _ in range(text_cfg.num_hidden_layers)]
        self.norm = RMSNorm(text_cfg.hidden_size, text_cfg.rms_norm_eps)
        self.audio_tower = AudioTower(audio_cfg) if audio_cfg is not None else None
        self._rope_cache: tuple[int, mx.array, mx.array] | None = None

    def _rope(self, n: int):
        """Returns cos/sin shaped [1, 1, n, dim_head], grown on demand and cached."""
        if self._rope_cache is None or self._rope_cache[0] < n:
            d = self.cfg.dim_head
            inv = self.cfg.rope_theta ** (-np.arange(0, d, 2, dtype=np.float64) / d)
            ang = np.arange(n, dtype=np.float64)[:, None] * inv[None, :]
            # HF "half" layout duplicates the angle block rather than interleaving it
            full = np.concatenate([ang, ang], axis=-1)
            cos = mx.array(np.cos(full).astype(np.float32))[None, None]
            sin = mx.array(np.sin(full).astype(np.float32))[None, None]
            self._rope_cache = (n, cos, sin)
        _, cos, sin = self._rope_cache
        # slice the sequence axis (2), not the head axis -- a shorter prompt after a
        # longer one must still get exactly n positions
        return cos[:, :, :n], sin[:, :, :n]

    def __call__(
        self,
        input_ids: mx.array,  # [B, N]
        audio_features: mx.array | None = None,  # [1, T, n_mel] mel spectrogram
        audio_token_mask: mx.array | None = None,  # [B, N] True at audio placeholders
        audio_feature_len: int | None = None,  # unpadded mel length
    ) -> list[mx.array]:
        """Returns the 1 + num_layers hidden states, matching HF's output_hidden_states."""
        h = self.embed_tokens(input_ids)

        if audio_features is not None:
            if self.audio_tower is None:
                raise ValueError("audio input given but this encoder was built without an audio tower")
            audio_emb = self.audio_tower(audio_features, audio_feature_len)
            h = _scatter_audio(h, audio_emb, audio_token_mask)

        n = h.shape[1]
        cos, sin = self._rope(n)
        mask = _causal_mask(n, h.dtype)

        hidden_states = [h]
        for layer in self.layers:
            h = layer(h, cos, sin, mask)
            hidden_states.append(h)
        # HF applies the final norm to the last layer output but reports the
        # PRE-norm tensor as hidden_states[-1]; keep that contract.
        hidden_states[-1] = self.norm(hidden_states[-1])
        return hidden_states


def _causal_mask(n: int, dtype) -> mx.array:
    m = np.triu(np.full((n, n), -np.inf, dtype=np.float32), k=1)
    return mx.array(m).astype(dtype)


def _scatter_audio(h: mx.array, audio_emb: mx.array, mask: mx.array | None) -> mx.array:
    """Replace the audio placeholder positions in h with the tower's embeddings."""
    if mask is None:
        raise ValueError("audio_token_mask is required when passing audio_features")
    B, N, D = h.shape
    idx = np.nonzero(np.array(mask).reshape(-1))[0]
    if len(idx) != audio_emb.shape[0]:
        raise ValueError(
            f"audio placeholder count ({len(idx)}) != audio tower output length ({audio_emb.shape[0]}); "
            "the mel length and the chat template disagree"
        )
    flat = np.array(h.reshape(-1, D))
    flat[idx] = np.array(audio_emb).astype(flat.dtype)
    return mx.array(flat).reshape(B, N, D)


def load_thinker(mlx_dir: str) -> ThinkerEncoder:
    """Load a converted Thinker from ``mlx_dir`` (thinker.safetensors + config.json)."""
    with open(os.path.join(mlx_dir, "thinker_config.json")) as f:
        cfg = json.load(f)
    tc = TextConfig(**cfg["text"])
    ac = AudioConfig(**cfg["audio"]) if cfg.get("audio") else None
    m = ThinkerEncoder(tc, ac)
    m.load_weights(os.path.join(mlx_dir, "thinker.safetensors"))
    m.eval()
    return m
