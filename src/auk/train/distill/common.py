"""Flow parameterization, conditioning, and run plumbing shared by both distillation stages."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from ema_pytorch import EMA
from omegaconf import DictConfig, OmegaConf
from torch import nn
from transformers import Qwen2_5OmniThinkerForConditionalGeneration
from transformers.feature_extraction_utils import BatchFeature

from auk.model import Flux2Edit
from auk.model.utils import lens_to_mask


# Per-step generator seeds are laid out on these strides so no two ranks, updates,
# or loss terms ever draw the same noise.
SEED_STRIDE_PER_UPDATE = 1_000_003
SEED_STRIDE_PER_RANK = 10_000_019
SEED_STRIDE_PER_STUDENT_UPDATE = 97

WEIGHT_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    # The Qwen processor re-emits this warning for every batch it templates.
    logging.getLogger().addFilter(lambda record: "System prompt modified" not in record.getMessage())


def load_text_encoder(text_encoder_path: str) -> Qwen2_5OmniThinkerForConditionalGeneration:
    """Load the Thinker for text + reference audio, dropping the unused vision tower."""
    thinker = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(text_encoder_path, torch_dtype=torch.bfloat16)
    del thinker.visual
    thinker.visual = None
    return thinker


@dataclass
class DistillScriptArgs:
    """Paths and resources shared by both distillation stages."""

    train_jsonl: str
    config: str = field(default="ckpts/AuK/config.yaml", metadata={"help": "model config yaml (reads model.* only)"})
    teacher_ckpt: str = field(
        default="ckpts/AuK/auk_base.safetensors",
        metadata={"help": "teacher weights; .safetensors release or training .pt"},
    )
    teacher_use_ema: bool = field(
        default=False,
        metadata={"help": "read ema_model_state_dict instead of model_state_dict (.pt teachers only)"},
    )
    weight_type: Literal["bf16", "fp16", "fp32"] = field(
        default="bf16",
        metadata={"help": "backbone weight dtype (flash_attn requires bf16/fp16)"},
    )
    attn_backend: str = field(default="torch", metadata={"help": "override model.arch.attn_backend"})


@dataclass(kw_only=True)
class EMAConfig:
    """Student EMA settings."""

    beta: float
    update_after_step: int
    update_every: int


def build_student_ema(student: nn.Module, device: torch.device, ema_kwargs: EMAConfig) -> EMA:
    return EMA(
        student,
        beta=ema_kwargs.beta,
        update_after_step=ema_kwargs.update_after_step,
        update_every=ema_kwargs.update_every,
        include_online_model=False,
    ).to(device)


def build_backbone(
    model_config: DictConfig,
    vae_config: DictConfig,
    attn_backend: str,
    checkpoint_every_n_layers: int | None = None,
) -> Flux2Edit:
    arch = OmegaConf.to_container(model_config.arch, resolve=True)
    arch["attn_backend"] = attn_backend
    # DMD holds three backbones, so its activation peak is far above the single-model
    # figure the config comment was tuned for. Lowering the stride checkpoints more
    # blocks (1 = every block) and trades recompute for activation memory.
    if checkpoint_every_n_layers is not None:
        arch["checkpoint_every_n_layers"] = checkpoint_every_n_layers
    return Flux2Edit(**arch, latent_dim=vae_config.latent_dim)


@dataclass
class PreparedBatch:
    """Conditioning shared by teacher, student, and fake-score roles."""

    clean: torch.Tensor
    ref: torch.Tensor
    text: torch.Tensor
    lens: torch.Tensor
    ref_lens: torch.Tensor
    valid_mask: torch.Tensor
    ref_mask: torch.Tensor
    c_mask: torch.Tensor
    metadata: Mapping[str, Any] | None = None

    @classmethod
    def from_tensors(
        cls,
        *,
        clean: torch.Tensor,
        ref: torch.Tensor,
        text: torch.Tensor,
        lens: torch.Tensor,
        ref_lens: torch.Tensor,
        c_mask: torch.Tensor,
        metadata: Mapping[str, Any] | None = None,
    ) -> "PreparedBatch":
        lens = lens.to(device=clean.device, dtype=torch.long)
        ref_lens = ref_lens.to(device=clean.device, dtype=torch.long)
        return cls(
            clean=clean,
            ref=ref,
            text=text,
            lens=lens,
            ref_lens=ref_lens,
            valid_mask=lens_to_mask(lens, length=clean.shape[1]),
            ref_mask=lens_to_mask(ref_lens, length=ref.shape[1]),
            c_mask=c_mask.to(device=clean.device, dtype=torch.bool),
            metadata=None if metadata is None else dict(metadata),
        )

    @property
    def loss_mask(self) -> torch.Tensor:
        return self.valid_mask.unsqueeze(-1)

    def to(self, device: torch.device | str, *, dtype: torch.dtype | None = None) -> "PreparedBatch":
        value_dtype = dtype or self.clean.dtype
        return PreparedBatch(
            clean=self.clean.to(device=device, dtype=value_dtype),
            ref=self.ref.to(device=device, dtype=value_dtype),
            text=self.text.to(device=device, dtype=value_dtype),
            lens=self.lens.to(device=device),
            ref_lens=self.ref_lens.to(device=device),
            valid_mask=self.valid_mask.to(device=device),
            ref_mask=self.ref_mask.to(device=device),
            c_mask=self.c_mask.to(device=device),
            metadata=self.metadata,
        )


def task_selection_weight(batch: PreparedBatch, tasks: Sequence[str]) -> torch.Tensor:
    """Per-sample 1.0 for the selected tasks, 0.0 otherwise."""
    if batch.metadata is None:
        raise ValueError("task selection is set but the batch carries no metadata; the dataloader must provide _tasks")
    values = batch.metadata["_tasks"]
    if len(values) != batch.clean.shape[0]:
        raise ValueError(f"_tasks has {len(values)} entries for a batch of {batch.clean.shape[0]}")
    selected = {str(task) for task in tasks}
    return torch.tensor(
        [1.0 if str(task) in selected else 0.0 for task in values],
        device=batch.clean.device,
        dtype=torch.float32,
    )


def expand_time(time: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    while time.ndim < target.ndim:
        time = time.unsqueeze(-1)
    return time.to(device=target.device, dtype=target.dtype)


def batch_time(time: torch.Tensor, batch_size: int, reference: torch.Tensor) -> torch.Tensor:
    if time.ndim == 0:
        time = time[None]
    if time.numel() == 1:
        time = time.expand(batch_size)
    return time.to(device=reference.device, dtype=reference.dtype)


def normal_like(value: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    return torch.randn(value.shape, device=value.device, dtype=value.dtype, generator=generator)


def add_velocity_noise(clean: torch.Tensor, noise: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    """Linear flow interpolation with noise at t=0 and data at t=1."""
    expanded = expand_time(time, clean)
    return (1.0 - expanded) * noise + expanded * clean


def velocity_to_clean(noisy: torch.Tensor, velocity: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    """Convert a velocity prediction to the implied clean endpoint."""
    return noisy + (1.0 - expand_time(time, noisy)) * velocity


def guided_velocity(conditional: torch.Tensor, unconditional: torch.Tensor, cfg_scale: float) -> torch.Tensor:
    """Blend the two CFG branches into one guided velocity.

    Training uses v_g = v_u + s*(v_c-v_u), which equals the inference-side
    v_c + (s-1)*(v_c-v_u); a cfg_scale of s corresponds to w = s-1.
    """
    return unconditional + cfg_scale * (conditional - unconditional)


def clean_prediction(
    noisy: torch.Tensor,
    velocity: torch.Tensor,
    time: torch.Tensor,
    batch: PreparedBatch,
) -> torch.Tensor:
    """Clean endpoint of a velocity, with padded frames restored to the ground truth."""
    clean = velocity_to_clean(noisy, velocity, time)
    return torch.where(batch.valid_mask.unsqueeze(-1), clean, batch.clean)


def predict_clean(
    model: nn.Module,
    noisy: torch.Tensor,
    batch: PreparedBatch,
    time: torch.Tensor,
    *,
    conditional: bool,
) -> torch.Tensor:
    velocity = model(
        x=noisy,
        text=batch.text,
        time=batch_time(time, noisy.shape[0], noisy),
        mask=batch.valid_mask,
        c_mask=batch.c_mask,
        ref=batch.ref,
        ref_mask=batch.ref_mask,
        drop_audio_cond=not conditional,
        drop_text=not conditional,
    )
    return clean_prediction(noisy, velocity, time, batch)


class FrozenQwenConditioner(nn.Module):
    """One shared frozen Qwen pass with checkpoint-owned layer fusion."""

    def __init__(self, *, text_encoder: nn.Module, layer_weights: torch.Tensor, layer_scale: torch.Tensor):
        super().__init__()
        self.text_encoder = text_encoder.eval().requires_grad_(False)
        self.register_buffer("layer_weights", layer_weights.detach().float().clone(), persistent=True)
        self.register_buffer("layer_scale", layer_scale.detach().float().clone(), persistent=True)

    @torch.no_grad()
    def forward(self, inputs: BatchFeature) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = inputs.to(next(self.text_encoder.parameters()).device)
        attention_mask = inputs["attention_mask"]
        outputs = self.text_encoder(**inputs, output_hidden_states=True)
        hidden_states = list(outputs.hidden_states[1:])
        if len(hidden_states) != self.layer_weights.numel():
            raise ValueError(
                f"text layer count does not match checkpoint fusion weights: {len(hidden_states)} != {self.layer_weights.numel()}"
            )
        width = hidden_states[0].shape[-1]
        stacked = torch.stack([F.layer_norm(value, [width]) for value in hidden_states])
        weights = F.softmax(self.layer_weights, dim=0).to(stacked.dtype)
        fused = (stacked * weights[:, None, None, None]).sum(0) * self.layer_scale.to(stacked.dtype)
        return fused.detach(), attention_mask.bool().detach()


class VAEConditioningAdapter(nn.Module):
    """Encode each physical input once and share it across all distillation roles."""

    def __init__(self, *, vae_model: nn.Module, text_conditioner: FrozenQwenConditioner):
        super().__init__()
        self.vae_model = vae_model.eval().requires_grad_(False)
        self.text_conditioner = text_conditioner.eval().requires_grad_(False)
        self.latent_dim = int(vae_model.h.latent_dim)

    @property
    def device(self) -> torch.device:
        return next(self.vae_model.parameters()).device

    @torch.no_grad()
    def encode_audio(self, audio: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if audio.shape[-1] == 0:
            batch = audio.shape[0]
            return (
                torch.zeros(batch, 0, self.latent_dim, device=audio.device, dtype=torch.float32),
                torch.zeros(batch, device=audio.device, dtype=torch.long),
            )
        return self.vae_model.encoding_and_normalization(audio, sample_lengths=lengths)

    @torch.no_grad()
    def prepare_batch(self, batch: Mapping[str, Any]) -> PreparedBatch:
        device = self.device
        clean, lens = self.encode_audio(
            batch["audio"].to(device),
            batch["audio_lengths"].to(device=device, dtype=torch.long),
        )
        ref, ref_lens = self.encode_audio(
            batch["ref_audio"].to(device),
            batch["ref_audio_lengths"].to(device=device, dtype=torch.long),
        )
        text, c_mask = self.text_conditioner(batch["cond_inputs"])
        # Only _tasks is consumed (task routing); the plain jsonl reader omits it.
        metadata = {"_tasks": batch["_tasks"]} if "_tasks" in batch else None
        return PreparedBatch.from_tensors(
            clean=clean.detach(),
            ref=ref.detach(),
            text=text.to(device).detach(),
            lens=lens,
            ref_lens=ref_lens,
            c_mask=c_mask.to(device),
            metadata=metadata,
        )


def reduce_metrics(
    accelerator: Accelerator,
    metrics: Mapping[str, float],
    *,
    sum_keys: Mapping[str, str] | None = None,
    max_keys: Sequence[str] = (),
) -> dict[str, float]:
    """All-reduce every scalar in three grouped collectives instead of one each.

    Keys in sum_keys are summed and renamed to their mapped name, keys in
    max_keys take the maximum, everything else is averaged.
    """
    sum_keys = sum_keys or {}
    grouped: dict[str, list[str]] = {"sum": [], "max": [], "mean": []}
    for name in metrics:
        if name in sum_keys:
            grouped["sum"].append(name)
        elif name in max_keys:
            grouped["max"].append(name)
        else:
            grouped["mean"].append(name)

    reduced: dict[str, float] = {}
    for reduction, names in grouped.items():
        if not names:
            continue
        stacked = torch.tensor(
            [float(metrics[name]) for name in names],
            device=accelerator.device,
            dtype=torch.float64,
        )
        stacked = accelerator.reduce(stacked, reduction=reduction)
        for name, value in zip(names, stacked.tolist()):
            reduced[sum_keys[name] if reduction == "sum" else name] = float(value)
    return reduced
