"""End-to-end MLX inference for AuK: instruction + optional reference audio -> waveform.

Mirrors ``auk.infer.infer_auk.AukInfer``:
  1. Qwen Thinker encodes the chat-formatted instruction (+ reference audio).
  2. An ELMo-style weighted sum over its 36 layers forms the conditioning tensor.
  3. A Euler ODE over the Flux2Edit DiT integrates the target latent, with CFG.
  4. The VAE decodes the generated slice back to a waveform.

Tokenisation and mel extraction still go through the HF processor: it owns the
chat template and the audio front end, neither of which is a model weight.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from omegaconf import OmegaConf

from auk_mlx.dit import DiTConfig, Flux2Edit
from auk_mlx.qwen_thinker import AudioConfig, TextConfig, ThinkerEncoder
from auk_mlx.vae import BigVGANFlowVAE, VAEConfig


# AuK-Flash is distilled onto a fixed 4-step grid with CFG disabled; re-adding
# guidance there clips the amplitude hard.
FLASH_T_GRID = [0.0, 0.07612049579620361, 0.2928932309150696, 0.6173166036605835, 1.0]


@dataclass
class GenerateOptions:
    gen_seconds: float | None = None
    nfe: int = 32
    cfg_strength: float = 2.0
    sway_sampling_coef: float | None = -1.0
    t_grid: list[float] | None = None
    seed: int | None = None


class AukMLX:
    def __init__(
        self,
        mlx_dir: str,
        config_path: str,
        qwen_dir: str,
        *,
        bits: int | None = None,
        group_size: int = 64,
        sequential: bool = False,
    ):
        """``bits`` quantizes the DiT and the Thinker to 8 or 4 bits (None = fp32).

        Only ``nn.Linear`` and ``nn.Embedding`` are quantizable in MLX, which suits
        this model: the DiT is 99.4% Linear and the Thinker 99.8% Linear+Embedding.
        The VAE is 99.9% Conv1d and stays fp32 either way -- at 147M params it is
        the small one, so nothing is lost by leaving it alone.

        ``sequential`` holds only one large stack in memory at a time. The Thinker
        runs to completion first and produces a [1, N, 2048] conditioning tensor;
        nothing downstream needs its 3.7B weights, so they are dropped before the
        DiT is built. This is the MLX analogue of the CUDA ``cpu_offload`` path,
        and cheaper: MLX's unified memory means freeing is a deallocation, not a
        device-to-host copy. The cost is re-reading weights from disk per call,
        so it pays off for one-shot runs and loses for batch work.
        """
        self.mlx_dir = mlx_dir
        self.qwen_dir = qwen_dir
        self.bits = bits
        self.group_size = group_size
        self.sequential = sequential

        cfg = OmegaConf.load(config_path)
        self.is_flash = cfg.model.get("name", "") == "AuK-Flash"
        self.variant = "flash" if self.is_flash else "base"

        vae_cfg = cfg.model.vae
        self.sample_rate = int(vae_cfg.target_sample_rate)
        self.downsample_rate = int(vae_cfg.downsample_rate)
        self.latent_dim = int(vae_cfg.latent_dim)
        self._vae_kwargs = OmegaConf.to_container(vae_cfg.get("model_init_kwargs", OmegaConf.create({})), resolve=True)

        # layer fusion is tiny (37 floats) and always resident
        fusion = mx.load(os.path.join(mlx_dir, f"fusion_{self.variant}.safetensors"))
        self.layer_weights = fusion["layer_weights"]
        self.layer_scale = fusion["layer_scale"]
        self._inv_freq = np.array(fusion["inv_freq"])
        self._arch = OmegaConf.to_container(cfg.model.arch, resolve=True)
        self._arch["latent_dim"] = self.latent_dim

        # HF processor owns the chat template + mel front end (no model weights)
        from transformers import Qwen2_5OmniProcessor

        self.processor = Qwen2_5OmniProcessor.from_pretrained(qwen_dir)
        with open(os.path.join(qwen_dir, "config.json")) as f:
            self.audio_token_id = json.load(f)["thinker_config"]["audio_token_index"]

        self.vae = self.dit = self.thinker = None
        if not sequential:
            self.vae = self._build_vae()
            self.dit = self._build_dit()
            self.thinker = self._build_thinker()

    # ------------------------------------------------------------------ builders

    def _build_vae(self) -> BigVGANFlowVAE:
        m = BigVGANFlowVAE(VAEConfig.from_dict(self._vae_kwargs))
        m.load_weights(os.path.join(self.mlx_dir, "vae.safetensors"))
        m.eval()
        return m

    def _build_dit(self) -> Flux2Edit:
        m = Flux2Edit(DiTConfig.from_dict(self._arch), inv_freq=self._inv_freq)
        q = os.path.join(self.mlx_dir, f"dit_{self.variant}.q{self.bits}.safetensors") if self.bits else None
        if q and os.path.isfile(q):
            # pre-quantized on disk: build the quantized skeleton first, then load
            nn.quantize(m, group_size=self.group_size, bits=self.bits)
            m.load_weights(q)
        else:
            m.load_weights(os.path.join(self.mlx_dir, f"dit_{self.variant}.safetensors"))
            if self.bits:
                nn.quantize(m, group_size=self.group_size, bits=self.bits)
        m.eval()
        return m

    def _build_thinker(self) -> ThinkerEncoder:
        tdir = os.path.join(self.mlx_dir, "thinker")
        with open(os.path.join(tdir, "thinker_config.json")) as f:
            tmeta = json.load(f)
        m = ThinkerEncoder(TextConfig(**tmeta["text"]), AudioConfig(**tmeta["audio"]))
        q = os.path.join(tdir, f"thinker.q{self.bits}.safetensors") if self.bits else None
        if q and os.path.isfile(q):
            nn.quantize(m, group_size=self.group_size, bits=self.bits)
            m.load_weights(q)
        else:
            m.load_weights(os.path.join(tdir, "thinker.safetensors"), strict=False)
            if self.bits:
                nn.quantize(m, group_size=self.group_size, bits=self.bits)
        m.eval()
        return m

    @staticmethod
    def _release(*names_and_owner) -> None:
        """Drop references and hand the buffers back to MLX."""
        import gc

        gc.collect()
        mx.clear_cache()

    # ------------------------------------------------------------------ text

    def encode_text(self, messages: list) -> mx.array:
        """Chat messages -> fused conditioning tensor [1, N, 2048].

        When the turn carries reference audio, the processor emits mel features plus
        ``<|AUDIO|>`` placeholders; the audio tower fills those positions.
        """
        texts = self.processor.apply_chat_template([messages], tokenize=False, add_generation_prompt=True)
        has_audio = any(c.get("type") == "audio" for m in messages for c in m.get("content", []))

        if has_audio:
            from qwen_omni_utils import process_mm_info

            audios, images, videos = process_mm_info([messages], use_audio_in_video=True)
            inputs = self.processor(
                text=texts,
                audio=audios,
                images=images,
                videos=videos,
                padding=True,
                return_tensors="np",
                use_audio_in_video=True,
            )
        else:
            inputs = self.processor(text=texts, padding=True, return_tensors="np")

        ids = mx.array(inputs["input_ids"])
        if has_audio and "input_features" in inputs:
            # processor gives [1, n_mel, T]; the MLX tower is channels-last
            mel = mx.array(np.asarray(inputs["input_features"]).transpose(0, 2, 1).astype(np.float32))
            feat_len = int(np.asarray(inputs["feature_attention_mask"]).sum())
            audio_mask = mx.array(np.asarray(inputs["input_ids"]) == self.audio_token_id)
            hidden = self.thinker(ids, audio_features=mel, audio_token_mask=audio_mask, audio_feature_len=feat_len)
        else:
            hidden = self.thinker(ids)

        # ELMo-style fusion over layers 1..L (the embedding output is excluded),
        # each layer-normed first, then softmax-weighted and rescaled.
        stacked = mx.stack([mx.fast.layer_norm(h, None, None, 1e-5) for h in hidden[1:]], axis=0)
        w = mx.softmax(self.layer_weights, axis=0)
        return (stacked * w[:, None, None, None]).sum(axis=0) * self.layer_scale

    # ----------------------------------------------------------------- audio

    def load_audio(self, path: str) -> np.ndarray:
        """Read a wav for the VAE: downmix to mono, then resample to 24 kHz.

        Order matters -- it mirrors ``AukInfer._load_audio``, which averages the
        channels before resampling. The Qwen side is unaffected: the processor
        loads the same file by path at its own 16 kHz rate.
        """
        import soundfile as sf

        wav, sr = sf.read(path, dtype="float32", always_2d=True)
        wav = wav.mean(axis=1)
        if sr != self.sample_rate:
            import soxr

            wav = soxr.resample(wav, sr, self.sample_rate, quality="VHQ")
        return np.ascontiguousarray(wav, dtype=np.float32)

    def encode_audio(self, wav: np.ndarray) -> mx.array:
        """Mono float waveform at ``self.sample_rate`` -> normalised latent [1, T, D]."""
        x = mx.array(wav.reshape(1, -1, 1).astype(np.float32))
        return self.vae.encode(x)

    # --------------------------------------------------------------- sampling

    def sample(
        self,
        text_embed: mx.array,
        ref_latent: mx.array,
        gen_len: int,
        opts: GenerateOptions,
    ) -> mx.array:
        if opts.seed is not None:
            mx.random.seed(opts.seed)

        nfe, cfg_strength, t_grid = opts.nfe, opts.cfg_strength, opts.t_grid
        sway = opts.sway_sampling_coef
        if self.is_flash:
            nfe, cfg_strength, sway, t_grid = 4, 0.0, None, FLASH_T_GRID

        if t_grid is not None:
            t = mx.array(np.asarray(t_grid, dtype=np.float32))
        else:
            t = mx.array(np.linspace(0.0, 1.0, nfe + 1, dtype=np.float32))
            if sway is not None:
                t = t + sway * (mx.cos(math.pi / 2 * t) - 1 + t)

        y = mx.random.normal((1, gen_len, self.latent_dim))
        use_cfg = cfg_strength >= 1e-5
        self.dit.clear_cache()

        # fixed-step Euler, matching torchdiffeq's method="euler"
        for i in range(t.size - 1):
            ti = t[i : i + 1]
            dt = t[i + 1] - t[i]
            if use_cfg:
                pred = self.dit(x=y, text=text_embed, t=ti, ref=ref_latent, cfg_infer=True, cache=True)
                v_cond, v_uncond = pred[0:1], pred[1:2]
                v = v_cond + (v_cond - v_uncond) * cfg_strength
            else:
                v = self.dit(x=y, text=text_embed, t=ti, ref=ref_latent, cache=True)
            y = y + v * dt
            mx.eval(y)

        self.dit.clear_cache()
        return y

    # ----------------------------------------------------------------- public

    def generate(
        self,
        instruction: str,
        *,
        audio: np.ndarray | None = None,
        audio_path: str | None = None,
        opts: GenerateOptions | None = None,
    ) -> tuple[np.ndarray, int]:
        """Generate audio. Pass ``audio_path`` for reference-conditioned tasks; the
        waveform is both VAE-encoded (prefix latent) and fed to the Qwen audio tower.
        Any sample rate and channel count is accepted -- see ``load_audio``.

        In ``sequential`` mode the three stacks are built and dropped in the order
        they are used (VAE encode -> Thinker -> DiT -> VAE decode), so only one
        large model is resident at a time.
        """
        opts = opts or GenerateOptions()
        seq = self.sequential

        if audio_path is not None and audio is None:
            audio = self.load_audio(audio_path)

        content: list[dict] = []
        if audio is None:
            # text-only instruct TTS carries an explicit marker upstream
            content.append({"type": "text", "text": instruction + "|<no_prompt_audio>|"})
            ref_latent = mx.zeros((1, 0, self.latent_dim))
            ref_len = 0
        else:
            content.append({"type": "text", "text": instruction})
            if audio_path is None:
                raise ValueError("reference audio must be given as audio_path (the processor reads it by path)")
            content.append({"type": "audio", "audio": audio_path})
            if seq:
                self.vae = self._build_vae()
            ref_latent = self.encode_audio(audio)
            mx.eval(ref_latent)
            if seq:
                self.vae = None
                self._release()
            ref_len = ref_latent.shape[1]
        messages = [{"role": "user", "content": content}]

        if opts.gen_seconds is not None:
            gen_len = max(1, math.ceil(opts.gen_seconds * self.sample_rate / self.downsample_rate))
        else:
            gen_len = max(1, ref_len)

        # Thinker -> conditioning tensor. Nothing downstream reads its weights, so
        # they can go before the DiT (3.7B params) is built.
        if seq:
            self.thinker = self._build_thinker()
        text_embed = self.encode_text(messages)
        mx.eval(text_embed)
        if seq:
            self.thinker = None
            self._release()

        if seq:
            self.dit = self._build_dit()
        gen_latent = self.sample(text_embed, ref_latent, gen_len, opts)
        if seq:
            self.dit = None
            self._release()

        if not bool(mx.all(mx.isfinite(gen_latent))):
            raise RuntimeError("generated latent contains NaN/Inf")

        if seq:
            self.vae = self._build_vae()
        wav = self.vae.decode(gen_latent)
        mx.eval(wav)
        if seq:
            self.vae = None
            self._release()

        out = np.array(wav).reshape(-1)
        if not np.isfinite(out).all():
            raise RuntimeError("generated audio contains NaN/Inf")
        return out, self.sample_rate
