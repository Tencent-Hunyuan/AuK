"""Server-side workers for the split deployment.

Two nodes, matching the two natural boundaries in the pipeline:

* :class:`TextEncoderNode` — Qwen2.5-Omni Thinker **plus** the ELMo layer-fusion weights.
  Both live together on purpose: fusing on this side shrinks the response from ~117 MB
  (36 layers of hidden states) to ~3 MB.
* :class:`WorkerNode` — VAE + DiT + the whole ODE loop. The ODE loop is here, never in the
  caller, so one generation is one RPC instead of one per solver step.
"""

from __future__ import annotations

import logging
import os

import torch

from auk.infer.infer_auk import AukInfer, qwen_num_layers
from auk.model.cfm_edit import fuse_hidden_states
from auk.serve.codec import pack

logger = logging.getLogger(__name__)


def load_layer_fusion(ckpt_path: str) -> dict[str, torch.Tensor]:
    """Pull just ``layer_weights`` / ``layer_scale`` out of an AuK checkpoint.

    Uses ``safe_open`` so a 6 GB export is never fully read for two tiny tensors.
    """
    out: dict[str, torch.Tensor] = {}
    if ckpt_path.endswith(".safetensors"):
        from safetensors import safe_open

        with safe_open(ckpt_path, framework="pt", device="cpu") as f:
            keys = set(f.keys())
            for name in ("layer_weights", "layer_scale"):
                if name in keys:
                    out[name] = f.get_tensor(name)
        return out

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("ema_model_state_dict", checkpoint)
    for key, value in state.items():
        clean = key.replace("ema_model.", "")
        if clean in ("layer_weights", "layer_scale"):
            out[clean] = value
    return out


class TextEncoderNode:
    """Qwen2.5-Omni Thinker + layer fusion. Runs once per generation."""

    def __init__(
        self,
        qwen_path: str,
        ckpt_path: str | None = None,
        *,
        device: str | None = None,
        weight_dtype: str | None = None,
    ):
        from transformers import Qwen2_5OmniThinkerForConditionalGeneration

        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}.get(weight_dtype or "bf16", torch.bfloat16)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        logger.info("Loading Qwen text encoder from %s ...", qwen_path)
        thinker = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(qwen_path, torch_dtype=dtype)
        if thinker.visual is not None:
            del thinker.visual
            thinker.visual = None
        thinker.requires_grad_(False)
        thinker.eval()
        self.text_encoder = thinker.to(self.device)

        num_layers = qwen_num_layers(qwen_path)
        self.layer_weights = torch.zeros(num_layers, dtype=dtype, device=self.device)
        self.layer_scale = torch.ones(1, dtype=dtype, device=self.device)
        if ckpt_path:
            fusion = load_layer_fusion(ckpt_path)
            missing = [k for k in ("layer_weights", "layer_scale") if k not in fusion]
            if missing:
                raise ValueError(f"{ckpt_path} does not contain the layer-fusion weights: {missing}")
            self.layer_weights.copy_(fusion["layer_weights"].to(self.device))
            self.layer_scale.copy_(fusion["layer_scale"].to(self.device))
        else:
            logger.warning("No --ckpt given: layer-fusion weights stay zeroed. Output will be wrong.")

    @torch.no_grad()
    def encode(self, cond_inputs) -> bytes:
        """Run the LLM and return a packed frame with the fused embedding."""
        if hasattr(cond_inputs, "to"):
            cond_inputs = cond_inputs.to(self.device)
        else:
            cond_inputs = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in cond_inputs.items()}
        attention_mask = cond_inputs["attention_mask"]

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.startswith("cuda")):
            outputs = self.text_encoder(**cond_inputs, output_hidden_states=True)
            hidden = fuse_hidden_states(outputs.hidden_states, self.layer_weights, self.layer_scale)

        # bool is not portable across every safetensors build — ship masks as uint8
        return pack(
            {"shape": list(hidden.shape), "dtype": str(hidden.dtype)},
            {"hidden": hidden, "attention_mask": attention_mask.to(torch.uint8)},
        )


class WorkerNode:
    """VAE + DiT + ODE loop. The compute bottleneck; scale this one horizontally."""

    def __init__(
        self,
        config_path: str,
        ckpt_path: str,
        *,
        qwen_path: str | None = None,
        device: str | None = None,
        dtype: str = "bf16",
        weight_dtype: str | None = None,
        device_map: str | dict | None = None,
    ):
        self.engine = AukInfer(
            config_path=config_path,
            ckpt_path=ckpt_path,
            device=device,
            dtype=dtype,
            qwen_path=qwen_path,
            device_map=device_map,
            weight_dtype=weight_dtype,
            load_qwen=False,
        )

    def info(self) -> dict:
        vae = self.engine.config.model.vae
        return {
            "target_sample_rate": int(vae.target_sample_rate),
            "downsample_rate": int(vae.downsample_rate),
            "latent_dim": int(vae.latent_dim),
            "is_flash": bool(self.engine.is_flash),
            "device": self.engine.device,
        }

    def generate(
        self,
        ref_audio: torch.Tensor | None,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        gen_latent_len: int,
        nfe: int,
        cfg_strength: float,
        sway_sampling_coef: float | None,
        t_grid: list[float] | None,
        seed: int | None,
        sample_rate: int = 0,
    ) -> tuple[torch.Tensor, int]:
        """Returns ``(audio [1, T] float32, sample_rate)``.

        ``ref_audio`` of length 0 (or ``None``) means text-only Instruct TTS.
        """
        ref = torch.zeros(1, 0) if ref_audio is None else ref_audio.to(torch.float32)
        if ref.ndim == 3:
            ref = ref.squeeze(0)
        if ref.dim() == 1:
            ref = ref.unsqueeze(0)
        return self.engine.generate_from_embeds(
            ref,
            int(sample_rate) or self.engine.target_sample_rate,
            hidden,
            attention_mask.bool(),
            gen_latent_len=gen_latent_len,
            nfe=nfe,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
            t_grid=t_grid,
            seed=seed,
        )


def default_config_path(ckpt_path: str) -> str:
    path = os.path.join(os.path.dirname(os.path.abspath(ckpt_path)), "config.yaml")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"config.yaml not found next to --ckpt: {path}")
    return path
