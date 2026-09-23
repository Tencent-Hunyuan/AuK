"""Entry point for stage-two DMD distillation."""

from __future__ import annotations

import logging
import os
import shutil
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from omegaconf import OmegaConf
from torch.optim import AdamW
from torch.utils.data import DataLoader, SequentialSampler
from transformers import HfArgumentParser, Qwen2_5OmniProcessor, get_scheduler

from auk.model.vae import load_vae_model
from auk.model.vae.bigvgan_flow_vae import BigVGANFlowVAEConfig
from auk.train.distill.checkpoint import (
    load_initializer_artifact,
    load_teacher_checkpoint,
    make_student_export,
    teacher_provenance,
)
from auk.train.distill.common import (
    WEIGHT_DTYPES,
    DistillScriptArgs,
    EMAConfig,
    FrozenQwenConditioner,
    VAEConditioningAdapter,
    build_backbone,
    configure_logging,
    load_text_encoder,
)
from auk.train.distill.dmd.checkpoint import initialize_dmd_from_initializer, initialize_dmd_roles
from auk.train.distill.dmd.system import DMDEditSystem, make_dmd_time_grid
from auk.train.distill.dmd.trainer import DecoupledDMDConfig, DMDVAETrainer

# Data loading is shared with regular fine-tuning: same jsonl schema, same
# frame-length packing. Only the training objective differs.
from auk.train.train import AukJsonlDataset, DynamicBatchSampler


configure_logging()
logger = logging.getLogger(__name__)


@dataclass
class DistillConfig:
    """DMD hyper-parameters."""

    nfe: int = 4
    sway_sampling_coef: float = -1.0
    teacher_cfg_scale: float = 4.0
    fake_score_updates_per_student: int = 5
    simulation_free: bool = True
    regression_weight: float = 1.0
    eps: float = 1.0e-5

    task_routing: bool = False
    regression_only_tasks: str = "ss,music,fisher"

    # CA/DM decomposition + APG on the CA branch
    ca_weight: float = 1.0
    dm_weight: float = 1.0
    ca_time_schedule: str = "decoupled-hybrid"  # 'uniform' | 'decoupled-hybrid'
    dm_time_schedule: str = "uniform"
    ca_apg: bool = True
    ca_apg_eta: float = 0.0  # 0 = pure orthogonal projection
    ca_apg_norm_threshold: float = 0.0

    student_learning_rate: float = 1.0e-5
    fake_score_learning_rate: float = 1.0e-5
    weight_decay: float = 0.0
    scheduler_type: str = "constant"
    warmup_updates: int = 0
    max_student_updates: int = 2500
    max_grad_norm: float = 1.0

    frames_threshold: int = 2700
    max_samples: int = 8
    dataloader_num_workers: int = 4
    seed: int = 666

    ema_beta: float = 0.999
    ema_update_after_step: int = 0
    ema_update_every: int = 1

    save_per_student_updates: int = 500
    last_per_student_updates: int = 100
    logging_steps: int = 1


@dataclass
class ScriptArgs(DistillScriptArgs):
    """Stage-two paths. Kept separate from DistillConfig, which holds hyper-params."""

    output_dir: str = "ckpts/auk_dmd"
    checkpoint_every_n_layers: int | None = field(
        default=None,
        metadata={
            "help": "override model.arch.checkpoint_every_n_layers; 1 checkpoints every block "
            "(lowest activation memory, most recompute). Unset keeps the config value."
        },
    )
    initializer: str | None = field(
        default=None,
        metadata={
            "help": "consistency artifact from train_init (e.g. ckpts/auk_cm_init/model_500.pt). "
            "Warm-starts student+fake_score; strongly recommended. Without it both roles start "
            "from the teacher and DMD converges noticeably slower."
        },
    )
    initializer_use_ema: bool = True


def save_inference_export(
    trainer: DMDVAETrainer,
    fusion: Mapping[str, torch.Tensor],
    initializer_provenance: Mapping[str, Any] | None,
    path: Path,
) -> None:
    trainer.accelerator.wait_for_everyone()
    if trainer.accelerator.is_main_process:
        student_state = {
            key.replace("ema_model.", ""): value
            for key, value in trainer.student_ema.ema_model.state_dict().items()
            if key not in ("initted", "step")
        }
        metadata = {
            "time_grid": trainer.time_grid.tolist(),
            "nfe": trainer.time_grid.numel() - 1,
            "teacher_provenance": dict(trainer.teacher_provenance),
        }
        if initializer_provenance is not None:
            metadata["initializer_provenance"] = dict(initializer_provenance)
        export = make_student_export(
            ema_state=student_state,
            online_state=student_state,
            fusion_state=fusion,
            update=trainer.update_state.student_updates,
            export_type="dmd_student_inference",
            metadata=metadata,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        trainer.accelerator.save(export, path)
        logger.info(f"saved inference export {path}")
    trainer.accelerator.wait_for_everyone()


def main() -> None:
    parser = HfArgumentParser((ScriptArgs, DistillConfig))
    args, distill_config = parser.parse_args_into_dataclasses()

    model_config = OmegaConf.load(args.config).model
    text_encoder_config = model_config.text_encoder
    vae_config = model_config.vae

    torch.manual_seed(distill_config.seed)

    saved_config = os.path.join(args.output_dir, "config.yaml")
    if os.environ.get("RANK", "0") == "0" and not os.path.exists(saved_config):
        os.makedirs(args.output_dir, exist_ok=True)
        shutil.copy(args.config, saved_config)

    logger.info(f"Loading Qwen text encoder from {text_encoder_config.text_encoder_path} ...")
    thinker = load_text_encoder(text_encoder_config.text_encoder_path)
    text_processor = Qwen2_5OmniProcessor.from_pretrained(text_encoder_config.text_encoder_path)

    logger.info(f"Loading VAE from {vae_config.vae_model_path} ...")
    model_init_kwargs = OmegaConf.to_container(vae_config.get("model_init_kwargs", OmegaConf.create({})), resolve=True)
    vae_model = load_vae_model(
        vae_name=vae_config.vae_name,
        vae_cfg=BigVGANFlowVAEConfig.from_dict(model_init_kwargs),
        vae_ckpt=vae_config.vae_model_path,
        map_location="cpu",
    )
    vae_model.eval().requires_grad_(False)

    logger.info("Building teacher / student / fake_score backbones ...")
    system = DMDEditSystem(
        student=build_backbone(model_config, vae_config, args.attn_backend, args.checkpoint_every_n_layers),
        teacher=build_backbone(model_config, vae_config, args.attn_backend, args.checkpoint_every_n_layers),
        fake_score=build_backbone(model_config, vae_config, args.attn_backend, args.checkpoint_every_n_layers),
        eps=distill_config.eps,
        regression_only_tasks=(
            tuple(task.strip() for task in distill_config.regression_only_tasks.split(",") if task.strip())
            if distill_config.task_routing
            else ()
        ),
        simulation_free=distill_config.simulation_free,
    )

    logger.info(f"Loading teacher weights from {args.teacher_ckpt} ...")
    checkpoint = load_teacher_checkpoint(args.teacher_ckpt)
    if args.initializer:
        initializer_state, initializer_provenance = load_initializer_artifact(
            args.initializer,
            use_ema=args.initializer_use_ema,
        )
        fusion = initialize_dmd_from_initializer(
            system,
            teacher_checkpoint=checkpoint,
            teacher_use_ema=args.teacher_use_ema,
            initializer_state=initializer_state,
        )
        logger.info(
            f"teacher(real_score)={args.teacher_ckpt}; "
            f"student+fake_score={initializer_provenance['source']} "
            f"update={initializer_provenance['update']}"
        )
    else:
        initializer_provenance = None
        fusion = initialize_dmd_roles(system, checkpoint, use_ema=args.teacher_use_ema)
        logger.warning(
            "no --initializer given: student and fake_score start from the teacher. "
            "Run auk.train.distill.initializer.train first for faster convergence."
        )

    # The three roles run in a single dtype (no autocast): the DMD losses compare
    # teacher/student/fake_score outputs directly, and flash_attn needs bf16/fp16.
    system = system.to(dtype=WEIGHT_DTYPES[args.weight_type])

    adapter = VAEConditioningAdapter(
        vae_model=vae_model,
        text_conditioner=FrozenQwenConditioner(
            text_encoder=thinker,
            layer_weights=fusion["layer_weights"],
            layer_scale=fusion["layer_scale"],
        ),
    )

    student_optimizer = AdamW(
        system.student.parameters(),
        lr=distill_config.student_learning_rate,
        betas=(0.9, 0.95),
        weight_decay=distill_config.weight_decay,
    )
    fake_score_optimizer = AdamW(
        system.fake_score.parameters(),
        lr=distill_config.fake_score_learning_rate,
        betas=(0.9, 0.95),
        weight_decay=distill_config.weight_decay,
    )
    interval = distill_config.fake_score_updates_per_student
    max_fake_updates = distill_config.max_student_updates * interval
    student_scheduler = get_scheduler(
        distill_config.scheduler_type,
        student_optimizer,
        num_warmup_steps=distill_config.warmup_updates,
        num_training_steps=distill_config.max_student_updates,
    )
    fake_score_scheduler = get_scheduler(
        distill_config.scheduler_type,
        fake_score_optimizer,
        num_warmup_steps=distill_config.warmup_updates,
        num_training_steps=max_fake_updates,
    )

    decoupled = DecoupledDMDConfig(
        ca_weight=distill_config.ca_weight,
        dm_weight=distill_config.dm_weight,
        ca_time_schedule=distill_config.ca_time_schedule,
        dm_time_schedule=distill_config.dm_time_schedule,
        ca_apg_enabled=distill_config.ca_apg,
        ca_apg_eta=distill_config.ca_apg_eta,
        ca_apg_norm_threshold=distill_config.ca_apg_norm_threshold,
    )
    resolved_config = {
        "dmd": {
            **asdict(decoupled),
            "fake_score_updates_per_student": interval,
            "teacher_cfg_scale": distill_config.teacher_cfg_scale,
            "nfe": distill_config.nfe,
            "sway_sampling_coef": distill_config.sway_sampling_coef,
            "simulation_free": distill_config.simulation_free,
            "regression_weight": distill_config.regression_weight,
        },
        "optim": {
            "student_learning_rate": distill_config.student_learning_rate,
            "fake_score_learning_rate": distill_config.fake_score_learning_rate,
        },
        "script_args": asdict(args),
    }

    trainer = DMDVAETrainer(
        system=system,
        adapter=adapter,
        student_optimizer=student_optimizer,
        fake_score_optimizer=fake_score_optimizer,
        student_scheduler=student_scheduler,
        fake_score_scheduler=fake_score_scheduler,
        time_grid=make_dmd_time_grid(
            nfe=distill_config.nfe,
            sway_sampling_coef=distill_config.sway_sampling_coef,
        ),
        teacher_provenance=teacher_provenance(args.teacher_ckpt, use_ema=args.teacher_use_ema),
        resolved_config=resolved_config,
        student_update_interval=interval,
        teacher_cfg_scale=distill_config.teacher_cfg_scale,
        decoupled=decoupled,
        regression_weight=distill_config.regression_weight,
        max_grad_norm=distill_config.max_grad_norm,
        ema_kwargs=EMAConfig(
            beta=distill_config.ema_beta,
            update_after_step=distill_config.ema_update_after_step,
            update_every=distill_config.ema_update_every,
        ),
        accelerator=Accelerator(kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)]),
    )

    train_dataset = AukJsonlDataset(args.train_jsonl, text_processor)
    trainer.accelerator.even_batches = False
    batch_sampler = DynamicBatchSampler(
        SequentialSampler(train_dataset),
        distill_config.frames_threshold,
        max_samples=distill_config.max_samples,
        random_seed=distill_config.seed,
    )
    dataloader = trainer.accelerator.prepare(
        DataLoader(
            train_dataset,
            collate_fn=train_dataset.collate_fn,
            num_workers=distill_config.dataloader_num_workers,
            pin_memory=True,
            persistent_workers=distill_config.dataloader_num_workers > 0,
            batch_sampler=batch_sampler,
        )
    )

    logger.info(
        f"train={len(train_dataset)} batches/epoch={len(batch_sampler)} | "
        f"student_updates={distill_config.max_student_updates} (fake={max_fake_updates}) | "
        f"config: {asdict(distill_config)}"
    )

    output_dir = Path(args.output_dir)
    resume_ckpt = output_dir / "model_last.pt"
    if resume_ckpt.exists():
        trainer.load_checkpoint(resume_ckpt)
        logger.info(
            f"resumed from {resume_ckpt} | fake_update={trainer.update_state.fake_score_updates} "
            f"student_update={trainer.update_state.student_updates} epoch={trainer.epoch}"
        )
    trainer.accelerator.wait_for_everyone()

    saved_resume_at = -1
    saved_export_at = -1
    while trainer.update_state.fake_score_updates < max_fake_updates:
        batch_sampler.set_epoch(trainer.epoch)
        reached_update_limit = False
        for batch in dataloader:
            metrics = trainer.step(trainer.adapter.prepare_batch(batch))

            update = trainer.update_state.fake_score_updates
            if trainer.accelerator.is_main_process and update % distill_config.logging_steps == 0:
                logger.info(
                    f"[dmd] fake_update={update}/{max_fake_updates} "
                    f"student_update={trainer.update_state.student_updates} "
                    + " ".join(f"{key}={value:.6g}" for key, value in sorted(metrics.items()))
                )

            student_update = trainer.update_state.student_updates
            if metrics["student_updated"] and student_update % distill_config.last_per_student_updates == 0:
                trainer.save_checkpoint(resume_ckpt, initializer_provenance=initializer_provenance)
                saved_resume_at = student_update
            if metrics["student_updated"] and student_update % distill_config.save_per_student_updates == 0:
                save_inference_export(
                    trainer,
                    fusion,
                    initializer_provenance,
                    output_dir / f"model_{student_update}.pt",
                )
                saved_export_at = student_update
            if update >= max_fake_updates:
                reached_update_limit = True
                break
        if reached_update_limit:
            break
        trainer.epoch += 1

    # The periodic saves above already cover the final update whenever the limit
    # is a multiple of the save interval.
    final_update = trainer.update_state.student_updates
    if saved_resume_at != final_update:
        trainer.save_checkpoint(resume_ckpt, initializer_provenance=initializer_provenance)
    if saved_export_at != final_update:
        save_inference_export(trainer, fusion, initializer_provenance, output_dir / f"model_{final_update}.pt")


if __name__ == "__main__":
    sys.exit(main())
