"""Training adapter and alternating optimizer loop for VAE DMD."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from accelerate import Accelerator
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from auk.train.distill.checkpoint import safe_torch_load
from auk.train.distill.common import (
    SEED_STRIDE_PER_RANK,
    SEED_STRIDE_PER_STUDENT_UPDATE,
    SEED_STRIDE_PER_UPDATE,
    EMAConfig,
    PreparedBatch,
    VAEConditioningAdapter,
    build_student_ema,
    reduce_metrics,
)
from auk.train.distill.dmd.checkpoint import load_training_checkpoint, save_training_checkpoint
from auk.train.distill.dmd.system import DMDEditSystem


@dataclass(kw_only=True)
class DecoupledDMDConfig:
    """CA/DM decomposition weights, time schedules, and APG settings."""

    ca_weight: float = 1.0
    dm_weight: float = 1.0
    ca_time_schedule: str = "uniform"
    dm_time_schedule: str = "uniform"
    ca_apg_enabled: bool = False
    ca_apg_eta: float = 0.0
    ca_apg_norm_threshold: float = 0.0


@dataclass
class DMDUpdateState:
    fake_score_updates: int = 0
    student_updates: int = 0
    fake_updates_since_student: int = 0


class DMDVAETrainer:
    """Five fake-score updates followed by one student update by default."""

    def __init__(
        self,
        *,
        system: DMDEditSystem,
        adapter: VAEConditioningAdapter,
        student_optimizer: Optimizer,
        fake_score_optimizer: Optimizer,
        student_scheduler: LRScheduler,
        fake_score_scheduler: LRScheduler,
        time_grid: torch.Tensor,
        teacher_provenance: Mapping[str, Any],
        resolved_config: Mapping[str, Any],
        student_update_interval: int = 5,
        teacher_cfg_scale: float = 3.0,
        decoupled: DecoupledDMDConfig,
        regression_weight: float = 0.0,
        max_grad_norm: float = 1.0,
        ema_kwargs: EMAConfig,
        accelerator: Accelerator,
    ):
        self.accelerator = accelerator
        self.adapter = adapter.to(self.accelerator.device)
        self.system, self.student_optimizer, self.fake_score_optimizer = self.accelerator.prepare(
            system, student_optimizer, fake_score_optimizer
        )
        self.student_scheduler = student_scheduler
        self.fake_score_scheduler = fake_score_scheduler
        self.time_grid = time_grid.to(device=self.accelerator.device, dtype=torch.float32)
        self.teacher_provenance = teacher_provenance
        self.resolved_config = dict(resolved_config)
        self.student_update_interval = student_update_interval
        self.teacher_cfg_scale = teacher_cfg_scale
        self.decoupled = decoupled
        self.regression_weight = regression_weight
        self.max_grad_norm = max_grad_norm
        self.update_state = DMDUpdateState()
        self.epoch = 0
        self.batch_cursor = 0

        unwrapped = self.accelerator.unwrap_model(self.system)
        self.student_ema = build_student_ema(unwrapped.student, self.accelerator.device, ema_kwargs)

    def step_generator(self, stream: int) -> torch.Generator:
        generator = torch.Generator(device=self.accelerator.device)
        seed = (
            SEED_STRIDE_PER_UPDATE * (self.update_state.fake_score_updates + 1)
            + SEED_STRIDE_PER_STUDENT_UPDATE * self.update_state.student_updates
            + SEED_STRIDE_PER_RANK * self.accelerator.process_index
            + stream
        )
        return generator.manual_seed(seed)

    def step(self, prepared: PreparedBatch) -> dict[str, float]:
        step_start = time.perf_counter()
        self.system.train()
        unwrapped = self.accelerator.unwrap_model(self.system)
        prepared = prepared.to(
            self.accelerator.device,
            dtype=next(unwrapped.student.parameters()).dtype,
        )
        self.fake_score_optimizer.zero_grad(set_to_none=True)
        fake_loss, metrics = self.system(
            mode="fake_score",
            batch=prepared,
            time_grid=self.time_grid,
            generator=self.step_generator(stream=1),
        )
        self.accelerator.backward(fake_loss)
        fake_grad_norm = self.accelerator.clip_grad_norm_(unwrapped.fake_score.parameters(), self.max_grad_norm)
        self.fake_score_optimizer.step()
        self.fake_score_scheduler.step()
        self.fake_score_optimizer.zero_grad(set_to_none=True)
        self.update_state.fake_score_updates += 1
        self.update_state.fake_updates_since_student += 1

        student_updated = self.update_state.fake_updates_since_student >= self.student_update_interval
        if student_updated:
            self.student_optimizer.zero_grad(set_to_none=True)
            student_loss, student_metrics = self.system(
                mode="student",
                batch=prepared,
                time_grid=self.time_grid,
                teacher_cfg_scale=self.teacher_cfg_scale,
                generator=self.step_generator(stream=2),
                ca_weight=self.decoupled.ca_weight,
                dm_weight=self.decoupled.dm_weight,
                ca_time_schedule=self.decoupled.ca_time_schedule,
                dm_time_schedule=self.decoupled.dm_time_schedule,
                ca_apg_enabled=self.decoupled.ca_apg_enabled,
                ca_apg_eta=self.decoupled.ca_apg_eta,
                ca_apg_norm_threshold=self.decoupled.ca_apg_norm_threshold,
                regression_weight=self.regression_weight,
            )
            self.accelerator.backward(student_loss)
            student_grad_norm = self.accelerator.clip_grad_norm_(unwrapped.student.parameters(), self.max_grad_norm)
            self.student_optimizer.step()
            self.student_scheduler.step()
            self.student_ema.update()
            self.student_optimizer.zero_grad(set_to_none=True)
            self.update_state.student_updates += 1
            self.update_state.fake_updates_since_student -= self.student_update_interval
            metrics.update(student_metrics)
            metrics["student_grad_norm"] = float(student_grad_norm.detach())

        self.batch_cursor += 1
        metrics.update(
            {
                "student_updated": float(student_updated),
                "fake_score_updates": float(self.update_state.fake_score_updates),
                "student_updates": float(self.update_state.student_updates),
                "fake_updates_since_student": float(self.update_state.fake_updates_since_student),
                "fake_score_grad_norm": float(fake_grad_norm.detach()),
                "fake_score_lr": float(self.fake_score_scheduler.get_last_lr()[0]),
                "student_lr": float(self.student_scheduler.get_last_lr()[0]),
                "local_samples": float(prepared.clean.shape[0]),
                "local_target_frames": float(prepared.lens.sum().detach()),
                "local_reference_frames": float(prepared.ref_lens.sum().detach()),
                "target_frames_max": float(prepared.lens.max().detach()),
                "target_frames_mean": float(prepared.lens.float().mean().detach()),
            }
        )
        torch.cuda.synchronize(self.accelerator.device)
        metrics["gpu_memory_peak_gib"] = float(torch.cuda.max_memory_allocated(self.accelerator.device) / 2**30)
        metrics["step_time_sec"] = float(time.perf_counter() - step_start)
        return reduce_metrics(
            self.accelerator,
            metrics,
            sum_keys={
                "local_samples": "global_samples",
                "local_target_frames": "global_target_frames",
                "local_reference_frames": "global_reference_frames",
            },
            max_keys=("target_frames_max",),
        )

    def save_checkpoint(
        self,
        path: Path,
        *,
        initializer_provenance: Mapping[str, Any] | None = None,
    ) -> None:
        save_training_checkpoint(
            path,
            accelerator=self.accelerator,
            system=self.accelerator.unwrap_model(self.system),
            student_optimizer=self.student_optimizer,
            fake_score_optimizer=self.fake_score_optimizer,
            student_scheduler=self.student_scheduler,
            fake_score_scheduler=self.fake_score_scheduler,
            student_ema_state=self.student_ema.ema_model.state_dict(),
            update_state=asdict(self.update_state),
            epoch=self.epoch,
            batch_cursor=self.batch_cursor,
            teacher_provenance=self.teacher_provenance,
            resolved_config=self.resolved_config,
            initializer_provenance=initializer_provenance,
        )

    def load_checkpoint(self, path: Path) -> None:
        restored = load_training_checkpoint(
            safe_torch_load(path, map_location="cpu"),
            system=self.accelerator.unwrap_model(self.system),
            student_optimizer=self.student_optimizer,
            fake_score_optimizer=self.fake_score_optimizer,
            student_scheduler=self.student_scheduler,
            fake_score_scheduler=self.fake_score_scheduler,
        )
        self.student_ema.ema_model.load_state_dict(restored["student_ema_state"], strict=True)
        self.update_state = DMDUpdateState(**restored["update_state"])
        self.epoch = restored["epoch"]
        self.batch_cursor = restored["batch_cursor"]
