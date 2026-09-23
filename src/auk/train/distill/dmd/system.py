"""DMD distillation core for velocity-flow edit models.

The module is intentionally independent of Qwen and the VAE. A prepared batch
contains one shared set of frozen conditioning features, while only the three
acoustic backbones have role-specific state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

import torch
from torch import nn

from auk.train.distill.common import (
    PreparedBatch,
    add_velocity_noise,
    batch_time,
    clean_prediction,
    expand_time,
    guided_velocity,
    normal_like,
    predict_clean,
    task_selection_weight,
)

APG_EPS = 1e-12


def get_epss_timesteps(n: int, device: torch.device | str, dtype: torch.dtype) -> torch.Tensor:
    """Inference step grid for n function evaluations."""
    # The table stores indices on a 1/EPSS_LATTICE_STEPS lattice, so only the listed
    # step counts have a tuned grid; anything else takes the uniform one.
    EPSS_LATTICE_STEPS = 32
    predefined_timesteps = {
        5: [0, 2, 4, 8, 16, 32],
        6: [0, 2, 4, 6, 8, 16, 32],
        7: [0, 2, 4, 6, 8, 16, 24, 32],
        10: [0, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32],
        12: [0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32],
        16: [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32],
    }
    if n not in predefined_timesteps:
        return torch.linspace(0, 1, n + 1, device=device, dtype=dtype)
    lattice_indices = torch.tensor(predefined_timesteps[n], device=device, dtype=dtype)
    return lattice_indices / EPSS_LATTICE_STEPS


def make_dmd_time_grid(
    nfe: int = 4,
    sway_sampling_coef: float = -1.0,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return the exact inference grid, including terminal t=1."""
    if nfe < 1:
        raise ValueError("nfe must be positive")
    time = get_epss_timesteps(nfe, device=device, dtype=dtype)
    return time + sway_sampling_coef * (torch.cos(torch.pi / 2 * time) - 1 + time)


def routing_weights(
    batch: PreparedBatch,
    regression_only_tasks: Sequence[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-sample (dmd, regression) weights; both all-ones when routing is off."""
    ones = torch.ones(batch.clean.shape[0], device=batch.clean.device, dtype=torch.float32)
    if not regression_only_tasks:
        return ones, ones
    regression = task_selection_weight(batch, regression_only_tasks)
    return ones - regression, regression


def weighted_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    loss_mask: torch.Tensor,
    sample_weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """MSE over masked frames, every sample's frames scaled by its routing weight."""
    weight = expand_time(sample_weight, prediction) * loss_mask.expand_as(prediction).to(prediction.dtype)
    return ((prediction - target).square() * weight).sum() / weight.sum().clamp_min(eps)


def sample_decoupled_time(
    batch: PreparedBatch,
    *,
    schedule: str,
    generator: torch.Generator | None,
    lower_bound: torch.Tensor | None = None,
) -> torch.Tensor:
    raw = torch.rand(
        (batch.clean.shape[0],),
        device=batch.clean.device,
        dtype=batch.clean.dtype,
        generator=generator,
    )
    # "decoupled-hybrid" draws from U(lower_bound, 1) so CA only aligns the
    # guidance signal on states cleaner than the current rollout.
    if schedule == "decoupled-hybrid":
        bound = lower_bound.to(raw).clamp(0.0, 1.0 - 1e-6)
        return bound + (1.0 - bound) * raw
    if schedule == "uniform":
        return raw
    raise ValueError(f"unsupported decoupled time schedule: {schedule!r}")


@dataclass
class RolloutResult:
    clean: torch.Tensor
    time: torch.Tensor
    step_index: int


class DMDEditSystem(nn.Module):
    """Three-role acoustic DMD system for a velocity-prediction backbone."""

    def __init__(
        self,
        *,
        student: nn.Module,
        teacher: nn.Module,
        fake_score: nn.Module,
        eps: float = 1e-5,
        regression_only_tasks: Sequence[str] = (),
        simulation_free: bool = True,
    ):
        super().__init__()
        self.student = student
        self.teacher = teacher.eval().requires_grad_(False)
        self.fake_score = fake_score
        self.eps = eps
        # True: noise the ground truth and take one student forward pass.
        # False: roll the student out from pure noise over several steps.
        self.simulation_free = simulation_free
        # Samples belonging to these tasks only contribute reg_loss, never DMD or critic.
        self.regression_only_tasks = tuple(regression_only_tasks)

    def train(self, mode: bool = True) -> "DMDEditSystem":
        # nn.Module.train recurses into children, so the teacher needs pinning back.
        super().train(mode)
        self.teacher.eval()
        return self

    def forward(
        self,
        mode: Literal["fake_score", "student"],
        **kwargs: Any,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if mode == "fake_score":
            return self.fake_score_loss(**kwargs)
        return self.student_loss(**kwargs)

    def student_rollout(
        self,
        batch: PreparedBatch,
        *,
        time_grid: torch.Tensor,
        generator: torch.Generator | None = None,
        with_grad: bool,
    ) -> RolloutResult:
        """Produce a student clean prediction at one inference-grid time."""
        step_index = int(
            torch.randint(
                time_grid.numel() - 1,
                (1,),
                device=batch.clean.device,
                generator=generator,
            ).item()
        )
        selected_time = time_grid[step_index : step_index + 1]
        valid_mask = batch.valid_mask.unsqueeze(-1)
        if self.simulation_free:
            # Noising the ground truth avoids compounding the student's own
            # multi-step rollout error.
            noise = normal_like(batch.clean, generator=generator)
            current = add_velocity_noise(batch.clean, noise, selected_time) * valid_mask
        else:
            current = normal_like(batch.clean, generator=generator) * valid_mask
            with torch.no_grad():
                for index in range(step_index):
                    step_time = time_grid[index : index + 1]
                    velocity = self.student(
                        x=current,
                        text=batch.text,
                        time=batch_time(step_time, current.shape[0], current),
                        mask=batch.valid_mask,
                        c_mask=batch.c_mask,
                        ref=batch.ref,
                        ref_mask=batch.ref_mask,
                        drop_audio_cond=False,
                        drop_text=False,
                    )
                    current = (current + (time_grid[index + 1] - time_grid[index]) * velocity) * valid_mask

        with torch.set_grad_enabled(with_grad):
            clean = predict_clean(self.student, current, batch, selected_time, conditional=True)
        return RolloutResult(clean=clean, time=selected_time, step_index=step_index)

    def fake_score_loss(
        self,
        batch: PreparedBatch,
        *,
        time_grid: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        rollout = self.student_rollout(batch, time_grid=time_grid, generator=generator, with_grad=False)
        time = torch.rand(
            (batch.clean.shape[0],),
            device=batch.clean.device,
            dtype=batch.clean.dtype,
            generator=generator,
        )
        noise = normal_like(rollout.clean, generator=generator)
        noisy = add_velocity_noise(rollout.clean, noise, time)
        prediction = self.fake_score(
            x=noisy,
            text=batch.text,
            time=batch_time(time, noisy.shape[0], noisy),
            mask=batch.valid_mask,
            c_mask=batch.c_mask,
            ref=batch.ref,
            ref_mask=batch.ref_mask,
            drop_audio_cond=False,
            drop_text=False,
        )
        dmd_weight, _ = routing_weights(batch, self.regression_only_tasks)
        loss = weighted_mse(prediction, rollout.clean - noise, batch.loss_mask, dmd_weight, self.eps)
        return loss, {
            "fake_score_loss": float(loss.detach()),
            "fake_score_time": float(time.detach().mean()),
            "student_rollout_time": float(rollout.time.detach().mean()),
            "student_rollout_step": float(rollout.step_index),
        }

    def dm_gradient_term(
        self,
        detached_clean: torch.Tensor,
        batch: PreparedBatch,
        *,
        schedule: str,
        generator: torch.Generator | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Distribution-matching direction, its teacher endpoint, and the time drawn."""
        time = sample_decoupled_time(batch, schedule=schedule, generator=generator)
        noise = normal_like(detached_clean, generator=generator)
        noisy = add_velocity_noise(detached_clean, noise, time)
        fake_clean = predict_clean(self.fake_score, noisy, batch, time, conditional=True)
        teacher_clean = predict_clean(self.teacher, noisy, batch, time, conditional=True)
        return fake_clean - teacher_clean, teacher_clean, time

    def ca_gradient_term(
        self,
        detached_clean: torch.Tensor,
        batch: PreparedBatch,
        *,
        schedule: str,
        generator: torch.Generator | None,
        teacher_cfg_scale: float,
        rollout_time: torch.Tensor | None,
        apg_enabled: bool,
        apg_eta: float,
        apg_norm_threshold: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Classifier-alignment direction, its un-projected baseline, and the time drawn."""
        time = sample_decoupled_time(
            batch,
            schedule=schedule,
            generator=generator,
            lower_bound=rollout_time,
        )
        noise = normal_like(detached_clean, generator=generator)
        noisy = add_velocity_noise(detached_clean, noise, time)
        both = self.teacher(
            x=noisy,
            text=batch.text,
            time=batch_time(time, noisy.shape[0], noisy),
            mask=batch.valid_mask,
            c_mask=batch.c_mask,
            ref=batch.ref,
            ref_mask=batch.ref_mask,
            cfg_infer=True,
        )
        conditional, unconditional = both.chunk(2)
        cond_clean = clean_prediction(noisy, conditional, time, batch)
        uncond_clean = clean_prediction(noisy, unconditional, time, batch)
        cfg_clean = clean_prediction(noisy, guided_velocity(conditional, unconditional, teacher_cfg_scale), time, batch)
        baseline = cond_clean - cfg_clean
        if not apg_enabled or (apg_eta == 1.0 and apg_norm_threshold == 0.0):
            return baseline, baseline, time

        mask = batch.loss_mask.expand_as(cond_clean).to(torch.float32)
        reduce_dims = tuple(range(1, cond_clean.ndim))
        guidance = (cond_clean.float() - uncond_clean.float()) * mask
        if apg_norm_threshold > 0:
            guidance_norm = guidance.square().sum(dim=reduce_dims).sqrt()
            cap = torch.minimum(
                torch.ones_like(guidance_norm),
                apg_norm_threshold / guidance_norm.clamp_min(APG_EPS),
            )
            guidance = guidance * cap[:, None, None]

        anchor = cond_clean.float() * mask
        anchor_norm_sq = anchor.square().sum(dim=reduce_dims, keepdim=True)
        parallel = torch.where(
            anchor_norm_sq > APG_EPS,
            (guidance * anchor).sum(dim=reduce_dims, keepdim=True) / anchor_norm_sq.clamp_min(APG_EPS) * anchor,
            torch.zeros_like(guidance),
        )
        applied = (guidance - parallel + apg_eta * parallel).to(baseline.dtype)
        return -(teacher_cfg_scale - 1.0) * applied, baseline, time

    def dmd_loss(
        self,
        student_clean: torch.Tensor,
        batch: PreparedBatch,
        *,
        teacher_cfg_scale: float,
        generator: torch.Generator | None,
        ca_weight: float = 1.0,
        dm_weight: float = 1.0,
        ca_time_schedule: str = "uniform",
        dm_time_schedule: str = "uniform",
        ca_apg_enabled: bool = False,
        ca_apg_eta: float = 0.0,
        ca_apg_norm_threshold: float = 0.0,
        rollout_time: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Decompose DMD into CA/DM terms with independent re-noising and one norm."""
        detached_clean = student_clean.detach()
        with torch.no_grad():
            dm_raw, teacher_dm_cond_clean, dm_time = self.dm_gradient_term(
                detached_clean,
                batch,
                schedule=dm_time_schedule,
                generator=generator,
            )
            ca_raw, baseline_ca_raw, ca_time = self.ca_gradient_term(
                detached_clean,
                batch,
                schedule=ca_time_schedule,
                generator=generator,
                teacher_cfg_scale=teacher_cfg_scale,
                rollout_time=rollout_time,
                apg_enabled=ca_apg_enabled,
                apg_eta=ca_apg_eta,
                apg_norm_threshold=ca_apg_norm_threshold,
            )

            residual = (detached_clean - (teacher_dm_cond_clean - baseline_ca_raw)).abs()
            mask = batch.loss_mask.expand_as(residual).to(residual.dtype)
            reduce_dims = tuple(range(1, residual.ndim))
            denominator = (residual * mask).sum(dim=reduce_dims) / mask.sum(dim=reduce_dims).clamp_min(self.eps)
            denominator = torch.nan_to_num(
                denominator,
                nan=self.eps,
                posinf=torch.finfo(denominator.dtype).max,
                neginf=self.eps,
            ).clamp_min(self.eps)

            # Each term is divided before the weighted sum: in bf16 the two orders
            # do not agree, and this is the one the trained runs used.
            ca_gradient = ca_raw / denominator[:, None, None]
            dm_gradient = dm_raw / denominator[:, None, None]
            dmd_gradient = ca_weight * ca_gradient + dm_weight * dm_gradient
            dmd_gradient = torch.nan_to_num(dmd_gradient, nan=0.0, posinf=0.0, neginf=0.0) * batch.loss_mask

        surrogate_target = (student_clean - dmd_gradient).detach()
        dmd_weight, _ = routing_weights(batch, self.regression_only_tasks)
        loss = 0.5 * weighted_mse(
            student_clean.float(),
            surrogate_target.float(),
            batch.loss_mask,
            dmd_weight,
            self.eps,
        )
        return loss, {
            "dmd_loss": float(loss.detach()),
            "dmd_denominator": float(denominator.detach().mean()),
            "decoupled_ca_time": float(ca_time.detach().mean()),
            "decoupled_dm_time": float(dm_time.detach().mean()),
        }

    def student_loss(
        self,
        batch: PreparedBatch,
        *,
        time_grid: torch.Tensor,
        teacher_cfg_scale: float = 3.0,
        generator: torch.Generator | None = None,
        ca_weight: float = 1.0,
        dm_weight: float = 1.0,
        ca_time_schedule: str = "uniform",
        dm_time_schedule: str = "uniform",
        ca_apg_enabled: bool = False,
        ca_apg_eta: float = 0.0,
        ca_apg_norm_threshold: float = 0.0,
        regression_weight: float = 0.0,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        rollout = self.student_rollout(batch, time_grid=time_grid, generator=generator, with_grad=True)
        dmd_loss, metrics = self.dmd_loss(
            rollout.clean,
            batch,
            teacher_cfg_scale=teacher_cfg_scale,
            generator=generator,
            ca_weight=ca_weight,
            dm_weight=dm_weight,
            ca_time_schedule=ca_time_schedule,
            dm_time_schedule=dm_time_schedule,
            ca_apg_enabled=ca_apg_enabled,
            ca_apg_eta=ca_apg_eta,
            ca_apg_norm_threshold=ca_apg_norm_threshold,
            rollout_time=rollout.time,
        )
        # Masked L2 between the generated and ground-truth latents. Under task
        # routing this weights the regression-only samples, complementing
        # dmd_loss which weights the rest.
        _, reg_sample_weight = routing_weights(batch, self.regression_only_tasks)
        reg_loss = weighted_mse(
            rollout.clean.float(),
            batch.clean.float(),
            batch.loss_mask,
            reg_sample_weight,
            self.eps,
        )
        metrics.update(
            {
                "student_rollout_time": float(rollout.time.detach().mean()),
                "student_rollout_step": float(rollout.step_index),
                "reg_loss": float(reg_loss.detach()),
            }
        )
        return dmd_loss + regression_weight * reg_loss, metrics
