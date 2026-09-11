from __future__ import annotations

import logging
import math
import os
import re

import torch
import torchaudio
from omegaconf import OmegaConf
from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration

from auk.model import CFMEdit, Flux2Edit
from auk.model.vae import load_vae_model
from auk.model.vae.bigvgan_flow_vae import BigVGANFlowVAEConfig


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

logging.getLogger().addFilter(lambda record: "System prompt modified" not in record.getMessage())


_DTYPE_MAP = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}

# the three independently placeable residents: the flow-matching DiT, the LLM text encoder, the VAE
_DEVICE_KEYS = ("dit", "qwen", "vae")


def _auto_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return f"cuda:{torch.cuda.current_device()}"
    return "cpu"


_DEVICE_RE = re.compile(r"^(cpu|mps|cuda(:\d+)?)$")

# AuK-Flash is a DMD student: it bakes in its own guidance, so re-adding CFG blows up the
# amplitude. Its recipe is fixed and overrides anything the caller passes.
FLASH_T_GRID = [0.0, 0.07612049579620361, 0.2928932309150696, 0.6173166036605835, 1.0]


def apply_flash_recipe(is_flash: bool, nfe, cfg_strength, sway_sampling_coef, t_grid):
    """Pin the distilled 4-step / CFG-off recipe for AuK-Flash; pass base AuK through untouched."""
    if not is_flash:
        return nfe, cfg_strength, sway_sampling_coef, t_grid
    return 4, 0.0, None, list(FLASH_T_GRID)


def qwen_num_layers(qwen_path: str) -> int:
    """Read ``num_hidden_layers`` from a Qwen2.5-Omni snapshot without loading any weights."""
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(qwen_path)
    thinker = getattr(cfg, "thinker_config", cfg)
    text = getattr(thinker, "text_config", thinker)
    return int(text.num_hidden_layers)


def _check_device(name: str) -> str:
    """Normalise + reject nonsense early instead of failing deep inside ``.to(...)``."""
    value = str(name).strip()
    if not _DEVICE_RE.match(value.lower()):
        raise ValueError(f"Invalid device {name!r}; expected 'cpu', 'mps', 'cuda' or 'cuda:N'")
    return value


def resolve_device_map(
    device: str | None = None,
    device_map: str | dict | None = None,
) -> dict[str, str]:
    """Resolve the placement of the three sub-models.

    ``device_map`` accepts:
      * ``None`` / ``""``                    -> everything on ``device`` (or the auto-detected device)
      * a bare device string (``"cuda:1"``)  -> everything on that device
      * ``"auto"``                           -> spread the residents over the visible CUDA devices
      * ``"dit=cuda:0,qwen=cuda:1,vae=cpu"`` -> per-component override ("cpu" is allowed)
      * a dict with any subset of the ``dit`` / ``qwen`` / ``vae`` keys
    """
    fallback = device or _auto_device()
    resolved = {k: fallback for k in _DEVICE_KEYS}
    if not device_map:
        return resolved

    if isinstance(device_map, dict):
        unknown = set(device_map) - set(_DEVICE_KEYS)
        if unknown:
            raise ValueError(f"Unknown device_map key(s) {sorted(unknown)}; expected any of {list(_DEVICE_KEYS)}")
        resolved.update({k: _check_device(v) for k, v in device_map.items() if v})
        return resolved

    spec = str(device_map).strip()
    if not spec:
        return resolved

    if spec.lower() == "auto":
        n = torch.cuda.device_count()
        if n < 2:
            logger.info("device_map='auto' but only %d CUDA device(s) visible — keeping everything on %s", n, fallback)
            return resolved
        # The DiT is the only resident that runs once per ODE step, so it keeps its own card and
        # gets the room for its activation peak. Qwen (biggest weights, runs once) shares the
        # second card with the tiny VAE, whose encode/decode never overlaps the DiT anyway.
        resolved["dit"] = "cuda:0"
        resolved["qwen"] = "cuda:1"
        resolved["vae"] = "cuda:1"
        return resolved

    if "=" not in spec:
        # bare device string -> put everything there
        return {k: _check_device(spec) for k in _DEVICE_KEYS}

    for fragment in spec.split(","):
        fragment = fragment.strip()
        if not fragment:
            continue
        key, sep, value = fragment.partition("=")
        key = key.strip().lower()
        if not sep or key not in _DEVICE_KEYS or not value.strip():
            raise ValueError(
                f"Cannot parse device_map fragment {fragment!r}; expected e.g. 'dit=cuda:0,qwen=cuda:1,vae=cuda:1' (or 'auto')"
            )
        resolved[key] = _check_device(value)
    return resolved


class AukInfer:
    def __init__(
        self,
        config_path: str,
        ckpt_path: str,
        *,
        device: str | None = None,
        dtype: str = "bf16",
        qwen_path: str | None = None,
        device_map: str | dict | None = None,
        weight_dtype: str | None = None,
        load_qwen: bool = True,
    ):
        """``load_qwen=False`` builds a text-encoder-free worker for a split deployment."""
        self.devices = resolve_device_map(device, device_map)
        self.dit_device = self.devices["dit"]
        self.qwen_device = self.devices["qwen"]
        self.vae_device = self.devices["vae"]
        # ``self.device`` stays the DiT device: that is where the latents / ODE state live.
        self.device = self.dit_device
        self.dtype = _DTYPE_MAP.get(dtype, torch.bfloat16)
        # Resident weight dtype. fp32 is the historical behaviour; bf16 halves resident VRAM and is
        # effectively free in quality because sampling already runs under a bf16 autocast.
        self.weight_dtype = _DTYPE_MAP.get(weight_dtype, torch.float32) if weight_dtype else torch.float32

        logger.info(
            "Devices | dit=%s | qwen=%s | vae=%s | autocast=%s | weights=%s",
            self.dit_device,
            self.qwen_device,
            self.vae_device,
            dtype,
            weight_dtype or "fp32",
        )
        if self.weight_dtype is torch.float32 and self.device.startswith("cuda"):
            logger.info("Tip: weight_dtype='bf16' roughly halves the resident weights (compute is already bf16 under autocast).")

        config = OmegaConf.load(config_path)
        if qwen_path:
            config.model.text_encoder.text_encoder_path = qwen_path
        # the VAE ships next to the checkpoint as vae.safetensors; use it if present, else keep config's path
        ckpt_dir_vae = os.path.join(os.path.dirname(os.path.abspath(ckpt_path)), "vae.safetensors")
        if os.path.isfile(ckpt_dir_vae):
            config.model.vae.vae_model_path = ckpt_dir_vae

        self.config = config
        # AuK-Flash is a distilled release that only works under a fixed (t_grid, cfg); detect it
        self.is_flash = config.model.get("name", "") == "AuK-Flash"
        if self.is_flash:
            logger.info("Detected AuK-Flash release — locking sampling to the 4-step / CFG-off recipe.")

        vae_config = config.model.vae
        self.target_sample_rate = vae_config.target_sample_rate
        self.downsample_rate = vae_config.downsample_rate
        self.latent_dim = vae_config.latent_dim

        # --- text encoder (Qwen2.5-Omni) ---
        text_encoder_config = config.model.text_encoder

        if load_qwen:
            logger.info(f"Loading Qwen text encoder from {text_encoder_config.text_encoder_path} ...")
            thinker = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
                text_encoder_config.text_encoder_path,
                torch_dtype=self.weight_dtype,
            )
            # keep the full multimodal Thinker (text + ref_audio); drop the unused vision tower
            if thinker.visual is not None:
                del thinker.visual
                thinker.visual = None
            text_encoder = thinker
            text_processor = Qwen2_5OmniProcessor.from_pretrained(text_encoder_config.text_encoder_path)
        else:
            # split deployment: embeddings arrive pre-fused from the text-encoder node
            logger.info("Skipping Qwen load — text embeddings are supplied by the caller.")
            text_encoder = None
            text_processor = None
        num_text_layers = None if load_qwen else qwen_num_layers(text_encoder_config.text_encoder_path)

        # --- VAE model ---
        logger.info(f"Loading VAE from {vae_config.vae_model_path} ...")
        model_init_kwargs = OmegaConf.to_container(vae_config.get("model_init_kwargs", OmegaConf.create({})), resolve=True)
        vae_model_config = BigVGANFlowVAEConfig.from_dict(model_init_kwargs)
        vae_model = load_vae_model(
            vae_name=vae_config.vae_name,
            vae_cfg=vae_model_config,
            vae_ckpt=vae_config.vae_model_path,
            map_location="cpu",
        )
        vae_model = vae_model.to(self.vae_device).eval()
        vae_model.requires_grad_(False)
        self.vae_model = vae_model

        # build CFMEdit (VAE-latent); Flux2Edit is the only supported backbone
        model_arc = OmegaConf.to_container(config.model.arch, resolve=True)
        model_arc["attn_backend"] = "torch"  # inference does not depend on flash_attn
        schedule_config = OmegaConf.to_container(config.model.get("schedule", OmegaConf.create({})), resolve=True)

        logger.info("Building CFMEdit model ...")
        model = CFMEdit(
            transformer=Flux2Edit(
                **model_arc,
                latent_dim=self.latent_dim,
            ),
            text_encoder=text_encoder,
            text_processor=text_processor,
            num_channels=self.latent_dim,
            num_text_layers=num_text_layers,
            **schedule_config,
        )
        # Only the DiT needs an fp32 master so the state-dict load is dtype-exact; the LLM is
        # already in its target dtype. Everything is still on CPU here — each submodule is
        # placed on (possibly different) accelerator(s) only after the weights are in.
        model.transformer.to(torch.float32)

        # --- load EMA weights (strip "ema_model." prefix; text_encoder.* comes from Qwen snapshot) ---
        self._load_ema_weights(model, ckpt_path)

        if self.weight_dtype is not torch.float32:
            # downcast on CPU so the host-to-device transfer is half the size
            model.transformer.to(self.weight_dtype)

        self._place_submodules(model)

        self.model = model
        self.model.eval()
        self._log_memory()

    def _place_submodules(self, model: CFMEdit) -> None:
        """Put the DiT, the LLM and the fusion weights onto their (possibly different) devices.

        ``nn.Parameter`` needs ``.data = ...`` rather than ``.to()`` — the latter returns a new
        parameter and silently unregisters it from the module.
        """
        model.transformer.to(self.dit_device)

        if model.text_encoder is not None:
            model.text_encoder.to(self.qwen_device)
            # the fusion weights are consumed next to the LLM hidden states
            fusion_device = self.qwen_device
        else:
            # split deployment: no LLM here, and the fusion weights are never used
            fusion_device = self.dit_device

        model.layer_weights.data = model.layer_weights.data.to(fusion_device)
        model.layer_scale.data = model.layer_scale.data.to(fusion_device)

    def _load_ema_weights(self, model: CFMEdit, ckpt_path: str):
        logger.info(f"Loading model checkpoint from {ckpt_path} ...")
        if ckpt_path.endswith(".safetensors"):
            # clean weights-only export: already stripped of the "ema_model." prefix
            from safetensors.torch import load_file

            state_dict = load_file(ckpt_path, device="cpu")
        else:
            # training checkpoint: pull the EMA weights and strip the "ema_model." prefix
            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            ema = checkpoint["ema_model_state_dict"]
            state_dict = {k.replace("ema_model.", ""): v for k, v in ema.items() if k not in ("initted", "step")}
            del checkpoint

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        n_missing_te = sum(1 for k in missing if k.startswith("text_encoder."))
        n_missing_other = len(missing) - n_missing_te
        logger.info(
            f"Loaded EMA weights | missing={len(missing)} (text_encoder.*={n_missing_te}, other={n_missing_other}) "
            f"| unexpected={len(unexpected)}"
        )
        if n_missing_other:
            logger.warning(
                "Some non-text-encoder weights are missing; check config/arch matches the checkpoint. "
                f"Examples: {[k for k in missing if not k.startswith('text_encoder.')][:10]}"
            )
        if unexpected:
            logger.warning(f"Unexpected keys in checkpoint: {unexpected[:10]}")
        self._sweep_caches()

    # ------------------------------------------------------------------ helpers

    def _sweep_caches(self):
        if not torch.cuda.is_available():
            return
        for dev in dict.fromkeys((self.dit_device, self.qwen_device, self.vae_device)):
            if dev.startswith("cuda"):
                with torch.cuda.device(dev):
                    torch.cuda.empty_cache()

    def _log_memory(self):
        """Log resident VRAM per device so a bad device_map is obvious straight away."""
        if not torch.cuda.is_available():
            return
        for dev in dict.fromkeys((self.dit_device, self.qwen_device, self.vae_device)):
            if not dev.startswith("cuda"):
                continue
            with torch.cuda.device(dev):
                index = torch.cuda.current_device()
                free, total = torch.cuda.mem_get_info()
            logger.info(
                "VRAM %s | allocated %.2f GB | reserved %.2f GB | free %.2f / %.2f GB",
                dev,
                torch.cuda.memory_allocated(index) / 1024**3,
                torch.cuda.memory_reserved(index) / 1024**3,
                free / 1024**3,
                total / 1024**3,
            )

    def _load_audio(self, source: str | tuple[torch.Tensor, int]) -> tuple[torch.Tensor, float]:
        if isinstance(source, str):
            audio, sr = torchaudio.load(source)
        else:
            audio, sr = source
            audio = audio.detach().to(device="cpu", dtype=torch.float32)
            if audio.ndim == 1:
                audio = audio.unsqueeze(0)
            if audio.ndim != 2:
                raise ValueError(f"Audio tensor must have shape [channels, samples], got {tuple(audio.shape)}.")
            if not isinstance(sr, int) or sr <= 0:
                raise ValueError(f"Audio sample rate must be a positive integer, got {sr!r}.")
            if audio.shape[-1] == 0:
                raise ValueError("Audio tensor is empty.")
            if not torch.isfinite(audio).all():
                raise ValueError("Audio tensor contains NaN or Inf.")
        return self._prepare_audio(audio, sr)

    def _prepare_audio(self, audio: torch.Tensor, sr: int) -> tuple[torch.Tensor, float]:
        """Mono downmix + RMS measurement + resample to the VAE rate."""
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        ref_rms = torch.sqrt(torch.mean(torch.square(audio)))
        if sr != self.target_sample_rate:
            audio = torchaudio.transforms.Resample(sr, self.target_sample_rate)(audio)
        return audio, float(ref_rms)

    @torch.inference_mode()
    def _run(
        self,
        ref_audio: torch.Tensor,  # [1, T] on cpu
        ref_rms: float | None,  # None => no reference audio, skip output RMS restore
        messages: list,  # single-sample chat messages (list of turns)
        gen_latent_len: int,
        *,
        nfe: int,
        cfg_strength: float,
        sway_sampling_coef: float,
        t_grid: list[float] | None,
        seed: int | None,
    ) -> torch.Tensor:
        ref_latents, ref_latent_lens_t, total_latent_lens_t = self._encode_reference(ref_audio, ref_rms, gen_latent_len)

        with torch.autocast("cuda", dtype=self.dtype, enabled=self.device.startswith("cuda")):
            cond_inputs = self.model.build_cond_inputs([messages], self.model.text_processor)
            text_embeds, context_mask = self.model.encode_text(cond_inputs, self.device)

        return self._run_from_embeds(
            ref_latents,
            ref_latent_lens_t,
            total_latent_lens_t,
            text_embeds,
            context_mask,
            nfe=nfe,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
            t_grid=t_grid,
            seed=seed,
        )

    def _encode_reference(self, ref_audio, ref_rms, gen_latent_len):
        """VAE-encode the reference clip → (ref_latents, ref_lens, total_lens) on the DiT device."""
        if ref_rms is None:
            ref_latent_lens_t = torch.zeros(1, dtype=torch.long, device=self.device)
            total_latent_lens_t = torch.tensor([gen_latent_len], dtype=torch.long, device=self.device)
            ref_latents = torch.zeros(1, 0, self.latent_dim, device=self.device, dtype=torch.float32)
            return ref_latents, ref_latent_lens_t, total_latent_lens_t

        ref_audio = ref_audio.to(self.vae_device).unsqueeze(0)  # [1, 1, T]
        ref_latent_len = ref_audio.shape[-1] // self.downsample_rate
        total_latent_len = ref_latent_len + gen_latent_len

        ref_latent_lens_t = torch.tensor([ref_latent_len], dtype=torch.long, device=self.device)
        total_latent_lens_t = torch.tensor([total_latent_len], dtype=torch.long, device=self.device)
        audio_lens_t = (ref_latent_lens_t * self.downsample_rate).to(self.vae_device)

        # --- online VAE encode + normalize (on the VAE's device, then hand over to the DiT) ---
        ref_latents, enc_latent_lens = self.vae_model.encoding_and_normalization(
            ref_audio,
            sample_lengths=audio_lens_t,
        )
        ref_latents = ref_latents.to(self.device)
        ref_latent_lens_t = torch.minimum(ref_latent_lens_t, enc_latent_lens.to(ref_latent_lens_t.device))
        return ref_latents, ref_latent_lens_t, total_latent_lens_t

    def _run_from_embeds(
        self,
        ref_latents,
        ref_latent_lens_t,
        total_latent_lens_t,
        text_embeds,
        context_mask,
        *,
        nfe: int,
        cfg_strength: float,
        sway_sampling_coef: float,
        t_grid: list[float] | None,
        seed: int | None,
    ) -> torch.Tensor:
        """ODE + VAE decode, starting from an already VAE-encoded reference."""
        # --- CFM sample in latent space (the whole ODE loop lives here, never in the caller) ---
        with torch.autocast("cuda", dtype=self.dtype, enabled=self.device.startswith("cuda")):
            generated, _ = self.model.sample_from_embeds(
                cond=ref_latents,
                text_embeds=text_embeds,
                context_mask=context_mask,
                duration=total_latent_lens_t,
                lens=ref_latent_lens_t,
                steps=nfe,
                cfg_strength=cfg_strength,
                sway_sampling_coef=sway_sampling_coef,
                t_grid=t_grid,
                no_ref_audio=False,
                seed=seed,
            )  # [1, T_total, D]

        gen = generated[0]
        rl = ref_latent_lens_t[0].item()
        tl = total_latent_lens_t[0].item()
        gen_latent = gen[rl:tl, :].unsqueeze(0)  # [1, T_new, D]
        if gen_latent.shape[1] == 0:
            raise RuntimeError("Empty generated latent (target duration collapsed to 0).")
        if torch.isnan(gen_latent).any() or torch.isinf(gen_latent).any():
            raise RuntimeError("Generated latent contains NaN/Inf.")

        # --- VAE decode (back on the VAE's device) ---
        gen_latent = self.vae_model.denormalize(gen_latent.to(self.vae_device))
        gen_latent = gen_latent.permute(0, 2, 1)  # [1, D, T_new]

        gen_audio = self.vae_model.inference_from_latents(gen_latent).cpu()
        if gen_audio.ndim == 3:
            gen_audio = gen_audio.squeeze(0)  # [1, T_wav]
        if torch.isnan(gen_audio).any() or torch.isinf(gen_audio).any():
            raise RuntimeError("Generated audio contains NaN/Inf.")

        return gen_audio.to(torch.float32)

    # ------------------------------------------------------------------ public API

    def generate_from_embeds(
        self,
        ref_audio: torch.Tensor,  # [1, T] on CPU, or [1, 0] for text-only Instruct TTS
        sample_rate: int,
        text_embeds: torch.Tensor,  # [1, Nt, d_llm] already fused
        context_mask: torch.Tensor,  # [1, Nt] bool
        *,
        gen_latent_len: int,
        nfe: int = 32,
        cfg_strength: float = 2.0,
        sway_sampling_coef: float = -1.0,
        t_grid: list[float] | None = None,
        seed: int | None = None,
    ) -> tuple[torch.Tensor, int]:
        """Worker-side entry point: VAE + DiT + decode, with text already encoded.

        Used by the split deployment. The ODE loop runs here in full, so one generation is one
        call rather than one call per solver step.
        """
        if ref_audio is None or ref_audio.shape[-1] == 0:
            ref, ref_rms = torch.zeros(1, 0), None
        else:
            ref, ref_rms = self._prepare_audio(ref_audio.detach().to("cpu", torch.float32), int(sample_rate))

        nfe, cfg_strength, sway_sampling_coef, t_grid = apply_flash_recipe(
            self.is_flash, nfe, cfg_strength, sway_sampling_coef, t_grid
        )

        ref_latents, ref_latent_lens_t, total_latent_lens_t = self._encode_reference(ref, ref_rms, gen_latent_len)
        audio_out = self._run_from_embeds(
            ref_latents,
            ref_latent_lens_t,
            total_latent_lens_t,
            text_embeds,
            context_mask,
            nfe=nfe,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
            t_grid=t_grid,
            seed=seed,
        )
        return audio_out, self.target_sample_rate

    def generate(
        self,
        messages: list,  # caller-composed ChatML turns (must carry a user audio item)
        *,
        audio: str | tuple[torch.Tensor, int] | None = None,
        gen_seconds: float | None = None,
        nfe: int = 32,
        cfg_strength: float = 2.0,
        sway_sampling_coef: float = -1.0,
        t_grid: list[float] | None = None,
        seed: int | None = None,
    ) -> tuple[torch.Tensor, int]:
        wav_path = audio or extract_audio_path(messages, required=False)
        if wav_path is not None:
            ref_audio, ref_rms = self._load_audio(wav_path)
        else:
            # no reference audio (text-only instruct TTS): empty reference, ref_rms=None
            ref_audio = torch.zeros(1, 0)
            ref_rms = None
            for m in messages:
                if m.get("role") != "user":
                    continue
                for c in m.get("content", []):
                    if isinstance(c, dict) and c.get("type") == "text" and not c["text"].endswith("|<no_prompt_audio>|"):
                        c["text"] = c["text"] + "|<no_prompt_audio>|"
        ref_latent_len = ref_audio.shape[-1] // self.downsample_rate  # 0 when no reference

        if gen_seconds is not None:
            gen_latent_len = max(1, int(math.ceil(gen_seconds * self.target_sample_rate / self.downsample_rate)))
        else:
            # default: regenerate a segment as long as the source clip
            gen_latent_len = max(1, ref_latent_len)

        nfe, cfg_strength, sway_sampling_coef, t_grid = apply_flash_recipe(
            self.is_flash, nfe, cfg_strength, sway_sampling_coef, t_grid
        )

        audio_out = self._run(
            ref_audio,
            ref_rms,
            messages,
            gen_latent_len,
            nfe=nfe,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
            t_grid=t_grid,
            seed=seed,
        )
        return audio_out, self.target_sample_rate


def extract_audio_path(messages: list, *, required: bool = True) -> str | None:
    for m in messages:
        if m.get("role") != "user":
            continue
        content = m.get("content")
        if not isinstance(content, list):
            continue
        for c in content:
            if isinstance(c, dict) and c.get("type") == "audio":
                path = c.get("audio") or c.get("audio_url")
                if path:
                    return path
    if required:
        raise ValueError("generate() needs a user audio item (type=audio) in messages to VAE-encode.")
    return None


def get_gen_duration(
    audio: str | None = None,
    ref_text: str | None = None,
    gen_text: str | None = None,
    gen_seconds: float | None = None,
    speed: float = 1.0,
) -> float | None:
    if gen_seconds:
        return float(gen_seconds)

    ref_seconds = None
    if audio:
        info = torchaudio.info(audio)
        ref_seconds = info.num_frames / info.sample_rate

    if ref_text and gen_text and ref_seconds is not None:
        return ref_seconds * len(gen_text.encode("utf-8")) / max(1, len(ref_text.encode("utf-8"))) / speed
    return ref_seconds


def save_audio(audio: torch.Tensor, sample_rate: int, output_path: str):
    """Save a [1, T] / [T] float tensor to ``output_path`` (creates parent dirs)."""
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)
    torchaudio.save(output_path, audio.to(torch.float32), sample_rate)
