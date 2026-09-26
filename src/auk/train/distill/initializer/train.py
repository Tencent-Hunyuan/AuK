"""Entry point for stage-one consistency initializer training."""

from __future__ import annotations

import logging
import os
import shutil
import sys
from dataclasses import asdict, dataclass
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
    extract_teacher_state,
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
from auk.train.distill.initializer.consistency import CMConfig, CMWeighting, ConsistencyInitSystem
from auk.train.distill.initializer.trainer import InitializerTrainer
from auk.train.train import AukJsonlDataset, DynamicBatchSampler


configure_logging()
logger = logging.getLogger(__name__)


@dataclass
class InitConfig:
    """Initializer hyper-parameters."""

    eps: float = 1.0e-5
    max_updates: int = 500  # measured as the point where CM loss flattens
    learning_rate: float = 1.0e-5
    weight_decay: float = 0.0
    scheduler_type: str = "constant"
    warmup_updates: int = 0
    max_grad_norm: float = 1.0

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

    # restrict the objective to these tasks; needs a per-sample _tasks batch
    # field, so it is a no-op on the plain jsonl reader
    tasks: str = ""

    frames_threshold: int = 2700
    max_samples: int = 8
    dataloader_num_workers: int = 4
    seed: int = 666

    ema_beta: float = 0.999
    ema_update_after_step: int = 0
    ema_update_every: int = 1

    save_per_updates: int = 500
    logging_steps: int = 1


@dataclass
class ScriptArgs(DistillScriptArgs):
    """Stage-one paths. Kept separate from InitConfig, which holds hyper-params."""

    output_dir: str = "ckpts/auk_cm_init"


def method_config(config: InitConfig) -> CMConfig:
    return CMConfig(
        tasks=tuple(task.strip() for task in config.tasks.split(",") if task.strip()),
        use_cd=config.use_cd,
        teacher_cfg_scale=config.teacher_cfg_scale,
        train_p_mean=config.train_p_mean,
        train_p_std=config.train_p_std,
        min_r=config.min_r,
        min_time=config.min_time,
        max_time=config.max_time,
        ct_resume_updates=config.ct_resume_updates,
        ct_q=config.ct_q,
        ct_ratio_limit=config.ct_ratio_limit,
        ct_kimg_per_stage=config.ct_kimg_per_stage,
        huber_const=config.huber_const,
        use_squared_l2=config.use_squared_l2,
        weighting=config.weighting,
    )


def save_initializer(
    trainer: InitializerTrainer,
    fusion: Mapping[str, torch.Tensor],
    provenance: Mapping[str, Any],
    output_dir: Path,
) -> None:
    trainer.accelerator.wait_for_everyone()
    if trainer.accelerator.is_main_process:
        update = trainer.updates
        ema_state = {
            key.replace("ema_model.", ""): value
            for key, value in trainer.student_ema.ema_model.state_dict().items()
            if key not in ("initted", "step")
        }
        export = make_student_export(
            ema_state=ema_state,
            online_state=trainer.accelerator.unwrap_model(trainer.system).student.state_dict(),
            fusion_state=fusion,
            update=update,
            export_type="dmd_initializer",
            metadata={
                "teacher_provenance": dict(provenance),
                "resolved_config": trainer.resolved_config,
            },
        )
        path = output_dir / f"model_{update}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        trainer.accelerator.save(export, path)
        logger.info(f"saved initializer artifact {path}")
    trainer.accelerator.wait_for_everyone()


def main() -> None:
    parser = HfArgumentParser((ScriptArgs, InitConfig))
    args, init_config = parser.parse_args_into_dataclasses()

    torch.manual_seed(init_config.seed)

    model_config = OmegaConf.load(args.config).model
    text_encoder_config = model_config.text_encoder
    vae_config = model_config.vae

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

    logger.info("Building student / teacher backbones ...")
    system = ConsistencyInitSystem(
        student=build_backbone(model_config, vae_config, args.attn_backend),
        teacher=build_backbone(model_config, vae_config, args.attn_backend),
        config=method_config(init_config),
        eps=init_config.eps,
    )

    logger.info(f"Loading teacher weights from {args.teacher_ckpt} ...")
    teacher_state, fusion = extract_teacher_state(
        load_teacher_checkpoint(args.teacher_ckpt),
        use_ema=args.teacher_use_ema,
    )
    system.teacher.load_state_dict(teacher_state, strict=True)
    system.student.load_state_dict(teacher_state, strict=True)
    system = system.to(dtype=WEIGHT_DTYPES[args.weight_type])

    adapter = VAEConditioningAdapter(
        vae_model=vae_model,
        text_conditioner=FrozenQwenConditioner(
            text_encoder=thinker,
            layer_weights=fusion["layer_weights"],
            layer_scale=fusion["layer_scale"],
        ),
    )

    optimizer = AdamW(
        system.student.parameters(),
        lr=init_config.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=init_config.weight_decay,
    )
    scheduler = get_scheduler(
        init_config.scheduler_type,
        optimizer,
        num_warmup_steps=init_config.warmup_updates,
        num_training_steps=init_config.max_updates,
    )

    provenance = teacher_provenance(args.teacher_ckpt, use_ema=args.teacher_use_ema)
    trainer = InitializerTrainer(
        system=system,
        adapter=adapter,
        optimizer=optimizer,
        scheduler=scheduler,
        resolved_config={
            "initializer": asdict(method_config(init_config)),
            "optim": {"learning_rate": init_config.learning_rate, "max_updates": init_config.max_updates},
            "script_args": asdict(args),
        },
        max_grad_norm=init_config.max_grad_norm,
        ema_kwargs=EMAConfig(
            beta=init_config.ema_beta,
            update_after_step=init_config.ema_update_after_step,
            update_every=init_config.ema_update_every,
        ),
        accelerator=Accelerator(kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)]),
    )

    train_dataset = AukJsonlDataset(args.train_jsonl, text_processor)
    trainer.accelerator.even_batches = False
    batch_sampler = DynamicBatchSampler(
        SequentialSampler(train_dataset),
        init_config.frames_threshold,
        max_samples=init_config.max_samples,
        random_seed=init_config.seed,
    )
    dataloader = trainer.accelerator.prepare(
        DataLoader(
            train_dataset,
            collate_fn=train_dataset.collate_fn,
            num_workers=init_config.dataloader_num_workers,
            pin_memory=True,
            persistent_workers=init_config.dataloader_num_workers > 0,
            batch_sampler=batch_sampler,
        )
    )

    logger.info(
        f"train={len(train_dataset)} batches/epoch={len(batch_sampler)} | "
        f"max_updates={init_config.max_updates} | config: {asdict(init_config)}"
    )

    output_dir = Path(args.output_dir)
    trainer.accelerator.wait_for_everyone()

    saved_at = -1
    while trainer.updates < init_config.max_updates:
        batch_sampler.set_epoch(trainer.epoch)
        reached_limit = False
        for batch in dataloader:
            metrics = trainer.step(trainer.adapter.prepare_batch(batch))

            update = trainer.updates
            if trainer.accelerator.is_main_process and update % init_config.logging_steps == 0:
                logger.info(
                    f"[init] update={update}/{init_config.max_updates} "
                    + " ".join(f"{key}={value:.6g}" for key, value in sorted(metrics.items()))
                )

            if update % init_config.save_per_updates == 0:
                save_initializer(trainer, fusion, provenance, output_dir)
                saved_at = update
            if update >= init_config.max_updates:
                reached_limit = True
                break
        if reached_limit:
            break
        trainer.epoch += 1

    # The periodic save above already covers the final update whenever
    # max_updates is a multiple of save_per_updates.
    if saved_at != trainer.updates:
        save_initializer(trainer, fusion, provenance, output_dir)


if __name__ == "__main__":
    sys.exit(main())
