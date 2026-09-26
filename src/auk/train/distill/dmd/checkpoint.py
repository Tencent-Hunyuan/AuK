"""Checkpoint IO for DMD training: three-role init, resume state, RNG-free like the mainline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
from accelerate import Accelerator
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from auk.train.distill.checkpoint import extract_teacher_state
from auk.train.distill.dmd.system import DMDEditSystem


def initialize_dmd_roles(
    system: DMDEditSystem,
    checkpoint: Mapping[str, Any],
    *,
    use_ema: bool = True,
) -> dict[str, torch.Tensor]:
    """Strictly initialize all three acoustic roles from one teacher state."""
    backbone, fusion = extract_teacher_state(checkpoint, use_ema=use_ema)
    for role in (system.teacher, system.student, system.fake_score):
        role.load_state_dict(backbone, strict=True)
    return fusion


def initialize_dmd_from_initializer(
    system: DMDEditSystem,
    *,
    teacher_checkpoint: Mapping[str, Any],
    teacher_use_ema: bool,
    initializer_state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Teacher from the base weights, student and fake_score from the stage-one artifact."""
    teacher_state, fusion = extract_teacher_state(teacher_checkpoint, use_ema=teacher_use_ema)
    system.teacher.load_state_dict(teacher_state, strict=True)
    system.student.load_state_dict(initializer_state, strict=True)
    system.fake_score.load_state_dict(initializer_state, strict=True)
    return fusion


def save_training_checkpoint(
    path: Path,
    *,
    accelerator: Accelerator,
    system: DMDEditSystem,
    student_optimizer: Optimizer,
    fake_score_optimizer: Optimizer,
    student_scheduler: LRScheduler,
    fake_score_scheduler: LRScheduler,
    student_ema_state: Mapping[str, torch.Tensor],
    update_state: Mapping[str, int],
    epoch: int,
    batch_cursor: int,
    teacher_provenance: Mapping[str, Any],
    resolved_config: Mapping[str, Any],
    initializer_provenance: Mapping[str, Any] | None = None,
) -> None:
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        checkpoint = {
            "student_model_state_dict": system.student.state_dict(),
            "fake_score_model_state_dict": system.fake_score.state_dict(),
            "student_ema_model_state_dict": dict(student_ema_state),
            "student_optimizer_state_dict": student_optimizer.state_dict(),
            "fake_score_optimizer_state_dict": fake_score_optimizer.state_dict(),
            "student_scheduler_state_dict": student_scheduler.state_dict(),
            "fake_score_scheduler_state_dict": fake_score_scheduler.state_dict(),
            "update_state": dict(update_state),
            "epoch": epoch,
            "batch_cursor": batch_cursor,
            "teacher_provenance": dict(teacher_provenance),
            "resolved_config": dict(resolved_config),
        }
        if initializer_provenance is not None:
            checkpoint["initializer_provenance"] = dict(initializer_provenance)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write beside the target and rename, so a crash mid-write cannot leave a
        # half-written model_last.pt as the only resume point.
        temporary = path.with_name(f".{path.name}.tmp")
        accelerator.save(checkpoint, temporary)
        temporary.replace(path)
    accelerator.wait_for_everyone()


def load_training_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    system: DMDEditSystem,
    student_optimizer: Optimizer,
    fake_score_optimizer: Optimizer,
    student_scheduler: LRScheduler,
    fake_score_scheduler: LRScheduler,
) -> dict[str, Any]:
    system.student.load_state_dict(checkpoint["student_model_state_dict"], strict=True)
    system.fake_score.load_state_dict(checkpoint["fake_score_model_state_dict"], strict=True)
    student_optimizer.load_state_dict(checkpoint["student_optimizer_state_dict"])
    fake_score_optimizer.load_state_dict(checkpoint["fake_score_optimizer_state_dict"])
    student_scheduler.load_state_dict(checkpoint["student_scheduler_state_dict"])
    fake_score_scheduler.load_state_dict(checkpoint["fake_score_scheduler_state_dict"])
    return {
        "student_ema_state": checkpoint["student_ema_model_state_dict"],
        "update_state": dict(checkpoint["update_state"]),
        "epoch": int(checkpoint["epoch"]),
        "batch_cursor": int(checkpoint["batch_cursor"]),
    }
