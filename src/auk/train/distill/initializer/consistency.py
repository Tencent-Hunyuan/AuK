"""Stage-one initializer objective: consistency distillation.

Adapted to this repository's flow direction (t=0 noise, t=1 data) and to
comparing clean predictions rather than velocities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

import torch
from torch import nn

from auk.train.distill.common import (
    PreparedBatch,
    add_velocity_noise,
    batch_time,
    expand_time,
    guided_velocity,
    normal_like,
    predict_clean,
    task_selection_weight,
)


CMWeighting = Literal["default", "sqrt", "one"]

# The shrink schedule weights the sigmoid tail by this gain and counts stage
# progress in thousands of samples; both are fixed by the schedule shape, not
# tuned per run.
RATIO_SIGMOID_GAIN = 8.0
SAMPLES_PER_STAGE_UNIT = 1000


@dataclass(kw_only=True)
class CMConfig:
    """Consistency objective settings: time pairing, distance, and weighting."""

    tasks: Sequence[str] = field(default_factory=tuple)
    use_cd: bool = True
    teacher_cfg_scale: float = 3.0
    train_p_mean: float = 0.0
    train_p_std: float = 1.0
    min_r: float = 1.0e-4
    min_time: float = 1.0e-4
    max_time: float = 0.999
    ct_resume_updates: int = 0
    ct_q: float = 2.0
    ct_ratio_limit: float = 0.999
    ct_kimg_per_stage: int = 12500
    huber_const: float = 1.0e-8
    use_squared_l2: bool = False
    weighting: CMWeighting = "default"


def stage_shrink_ratio(iteration: int, batch_size: int, *, q: float, ratio_limit: float, kimg_per_stage: int) -> float:
    """Shrink ratio for the r/t time pair, advancing one stage per kimg_per_stage."""
    stage = iteration * batch_size // (kimg_per_stage * SAMPLES_PER_STAGE_UNIT)
    return min(1.0 - 1.0 / q ** (stage + 1), ratio_limit)


def endpoint_time_from_current(t: torch.Tensor, ratio: float, min_r: float) -> torch.Tensor:
    """Map the current time to its endpoint via the sigmoid-weighted shrink."""
    return (t - t * (1.0 - ratio) * (1.0 + RATIO_SIGMOID_GAIN * torch.sigmoid(-t))).clamp_min(min_r)


class ConsistencyInitSystem(nn.Module):
    """Stage-one objective: pull the student towards a few-step solution.

    The student's clean prediction at current is regressed onto its own
    prediction at endpoint, so the two collapse onto one consistent
    trajectory before DMD starts.
    """

    def __init__(self, *, student: nn.Module, teacher: nn.Module, config: CMConfig, eps: float = 1e-5):
        super().__init__()
        self.student = student
        self.teacher = teacher.eval().requires_grad_(False)
        self.config = config
        self.eps = eps

    def train(self, mode: bool = True) -> "ConsistencyInitSystem":
        super().train(mode)
        self.teacher.eval()
        return self

    def pair_times(
        self,
        batch: PreparedBatch,
        iteration: int,
        generator: torch.Generator | None,
        *,
        global_batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        """Sample the (current, endpoint) time pair on the staged shrink schedule."""
        cfg = self.config
        normal = torch.randn(
            batch.clean.shape[0],
            device=batch.clean.device,
            dtype=batch.clean.dtype,
            generator=generator,
        )
        canonical_t = torch.sigmoid(normal * cfg.train_p_std + cfg.train_p_mean)
        canonical_t = canonical_t * (cfg.max_time - cfg.min_time) + cfg.min_time
        ratio = stage_shrink_ratio(
            iteration + cfg.ct_resume_updates,
            global_batch_size,
            q=cfg.ct_q,
            ratio_limit=cfg.ct_ratio_limit,
            kimg_per_stage=cfg.ct_kimg_per_stage,
        )
        canonical_r = endpoint_time_from_current(canonical_t, ratio, cfg.min_r)
        invalid = canonical_r >= canonical_t - self.eps
        canonical_t = torch.where(invalid, (canonical_r + self.eps).clamp(max=cfg.max_time), canonical_t)
        canonical_r = canonical_r.minimum(canonical_t - self.eps)
        return 1.0 - canonical_t, 1.0 - canonical_r, ratio

    def cm_distance(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        batch: PreparedBatch,
        current: torch.Tensor,
        endpoint: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-sample (weighted, unweighted) consistency distance."""
        cfg = self.config
        difference = prediction.float() - target.float()
        mask = batch.loss_mask.expand_as(difference).to(difference.dtype)
        distance = ((difference * mask).square().sum(dim=tuple(range(1, difference.ndim)))).sqrt()
        if cfg.huber_const > 0:
            unweighted = torch.sqrt(distance.square() + cfg.huber_const**2) - cfg.huber_const
        elif cfg.use_squared_l2:
            unweighted = distance.square()
        else:
            unweighted = distance
        delta = (endpoint - current).clamp_min(self.eps)
        if cfg.weighting == "default":
            return unweighted / delta, unweighted
        if cfg.weighting == "sqrt":
            return unweighted / delta.sqrt(), unweighted
        if cfg.weighting == "one":
            return unweighted, unweighted
        raise ValueError(f"unsupported CM weighting: {cfg.weighting}")

    def forward(
        self,
        batch: PreparedBatch,
        iteration: int,
        generator: torch.Generator | None = None,
        *,
        global_batch_size: int,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        cfg = self.config
        current, endpoint, ratio = self.pair_times(batch, iteration, generator, global_batch_size=global_batch_size)

        noise = normal_like(batch.clean, generator)
        noisy_current = add_velocity_noise(batch.clean, noise, current)
        if cfg.use_cd:
            with torch.no_grad():
                both = self.teacher(
                    x=noisy_current,
                    text=batch.text,
                    time=batch_time(current, noisy_current.shape[0], noisy_current),
                    mask=batch.valid_mask,
                    c_mask=batch.c_mask,
                    ref=batch.ref,
                    ref_mask=batch.ref_mask,
                    cfg_infer=True,
                )
                conditional, unconditional = both.chunk(2)
                velocity = guided_velocity(conditional, unconditional, cfg.teacher_cfg_scale)
                noisy_endpoint = noisy_current + expand_time(endpoint - current, noisy_current) * velocity
        else:
            noisy_endpoint = add_velocity_noise(batch.clean, noise, endpoint)

        # Both student passes must see the same dropout draw, otherwise the pair
        # differs by dropout noise rather than by the time step alone.
        with torch.random.fork_rng(devices=[noisy_current.device] if noisy_current.device.type == "cuda" else []):
            prediction = predict_clean(self.student, noisy_current, batch, current, conditional=True)
        with torch.no_grad():
            target = predict_clean(self.student, noisy_endpoint, batch, endpoint, conditional=True)
        per_sample, unweighted = self.cm_distance(prediction, target, batch, current, endpoint)

        if cfg.tasks:
            weight = task_selection_weight(batch, cfg.tasks)
        else:
            weight = torch.ones_like(per_sample)
        sample_weight = weight.to(per_sample.dtype)
        total = (per_sample * sample_weight).sum() / sample_weight.sum().clamp_min(self.eps)
        metrics = {
            "current_time": current.mean(),
            "endpoint_time": endpoint.mean(),
            "ct_ratio": torch.tensor(ratio, device=current.device),
            "consistency_unweighted": unweighted.mean(),
            "total_loss": total.detach(),
            "active_fraction": (weight > 0).float().mean(),
        }
        return total, {name: float(value.detach()) for name, value in metrics.items()}
