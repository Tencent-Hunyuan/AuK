"""LoRA fine-tuning entry point for AuK.

The official ``auk.train.train`` module is left untouched. This script rewrites
``train_loop`` so that the saved checkpoint keeps the official key layout
(``ema_model.*``), which keeps the inference loader and
``auk.train.merge_lora`` compatible.

Usage:
    accelerate launch ... -m auk.train.lora_train ...
"""

from __future__ import annotations

import math
import os
import random
import shutil
from dataclasses import dataclass, field

import torch
import torchaudio
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from auk.model import CFMEdit, Flux2Edit
from auk.model.vae import load_vae_model
from auk.model.vae.bigvgan_flow_vae import BigVGANFlowVAEConfig
from auk.train.train import (
    AukJsonlDataset,
    ScriptArgs,
    TrainConfig,
    evaluate,
    log_val_curve,
    logger,
    prepare_batch,
)
from omegaconf import OmegaConf
from safetensors.torch import load_file
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import HfArgumentParser, Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration

from auk.train.lora import apply_lora, freeze_for_lora


def save_checkpoint(accelerator, model, optimizer, scheduler, update, output_dir, last=False):
    """Save a ZeRO-3 safe checkpoint.

    ``accelerator.get_state_dict`` performs an all-gather, which is a collective
    operation and therefore has to be executed by every rank.
    """
    model_state = accelerator.get_state_dict(model)
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return
    model_state = {k: v for k, v in model_state.items() if not k.startswith("text_encoder.")}
    ema_state = {("ema_model." + k): v for k, v in model_state.items()}
    checkpoint = {
        "model_state_dict": model_state,
        "ema_model_state_dict": ema_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "update": update,
    }
    os.makedirs(output_dir, exist_ok=True)
    if last:
        accelerator.save(checkpoint, f"{output_dir}/model_last.pt")
        logger.info(f"[update {update}] saved model_last.pt")
        return
    accelerator.save(checkpoint, f"{output_dir}/model_{update}.pt")
    logger.info(f"[update {update}] saved model_{update}.pt")


@torch.no_grad()
def synthesize_and_save(accelerator, model, batch, vae_model, update, samples_dir):
    """Generate one sample with every rank so that ZeRO-3 gathers stay in sync."""
    unwrapped = accelerator.unwrap_model(model)
    unwrapped.eval()

    ref_audio = batch["ref_audio"][:1].to(accelerator.device)
    ref_audio_lengths = batch["ref_audio_lengths"][:1].to(accelerator.device)
    has_ref_audio = ref_audio.size(-1) > 0 and ref_audio_lengths[0].item() > 0
    if has_ref_audio:
        ref_latent, ref_lens = vae_model.encoding_and_normalization(ref_audio, sample_lengths=ref_audio_lengths)
        ref_len = ref_lens[0].item()
    else:
        ref_latent = torch.zeros(1, 0, vae_model.h.latent_dim, device=accelerator.device, dtype=torch.float32)
        ref_lens = torch.zeros(1, dtype=torch.long, device=accelerator.device)
        ref_len = 0

    target_audio = batch["audio"][:1].to(accelerator.device)
    target_lengths = batch["audio_lengths"][:1].to(accelerator.device)
    target_latent, target_latent_lens = vae_model.encoding_and_normalization(target_audio, sample_lengths=target_lengths)
    target_len = target_latent_lens[0].item()

    cond_inputs = unwrapped.build_cond_inputs(batch["messages"][:1], unwrapped.text_processor)

    mp = accelerator.mixed_precision
    autocast_dtype = torch.bfloat16 if mp == "bf16" else torch.float16 if mp == "fp16" else torch.float32
    with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=autocast_dtype != torch.float32):
        generated, _ = unwrapped.sample(
            cond=ref_latent,
            text=cond_inputs,
            duration=ref_len + target_len,
            lens=ref_lens,
            steps=32,
            cfg_strength=2.0,
            sway_sampling_coef=-1.0,
        )

    model.train()

    if not accelerator.is_main_process:
        return

    generated = generated.to(torch.float32)

    os.makedirs(samples_dir, exist_ok=True)
    for name, latent in [
        ("gen", generated[:, ref_len : ref_len + target_len, :]),
        ("tgt", target_latent[:1, :target_len, :]),
    ]:
        latent = vae_model.denormalize(latent.to(accelerator.device))
        audio = vae_model.inference_from_latents(latent.permute(0, 2, 1)).cpu().to(torch.float32)
        if audio.ndim == 3:
            audio = audio.squeeze(0)
        if torch.isnan(audio).any() or torch.isinf(audio).any():
            logger.warning(f"[update {update}] {name} audio has NaN/Inf, skipping")
            continue
        torchaudio.save(f"{samples_dir}/update_{update}_{name}.wav", audio, 24000)
    logger.info(f"[update {update}] wrote synthesis samples to {samples_dir}")


class BackboneForEMA(nn.Module):
    """Generation backbone only, excluding the frozen text encoder."""

    def __init__(self, cfm_model):
        super().__init__()
        self.transformer = cfm_model.transformer
        self.layer_weights = cfm_model.layer_weights
        self.layer_scale = cfm_model.layer_scale


@dataclass
class LoraArgs:
    lora_r: int = field(default=32, metadata={"help": "LoRA rank"})
    lora_alpha: int = field(default=32, metadata={"help": "LoRA alpha (scaling = alpha / r)"})
    lora_dropout: float = field(default=0.0, metadata={"help": "LoRA dropout"})
    lora_targets: str = field(
        default=None,
        metadata={"help": "Comma-separated path suffixes; empty uses auk.train.lora.DEFAULT_TARGETS"},
    )


def train_loop(model, vae_model, train_dataset, val_dataset, train_config, output_dir):
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], rng_types=[])

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=train_config.learning_rate,
        betas=(0.9, 0.95),
    )
    model, optimizer = accelerator.prepare(model, optimizer)

    vae_model = vae_model.to(accelerator.device)
    vae_model.requires_grad_(False)

    writer = SummaryWriter(f"{output_dir}/tensorboard") if accelerator.is_main_process else None

    accelerator.even_batches = True
    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=train_dataset.collate_fn,
        num_workers=train_config.dataloader_num_workers,
        pin_memory=True,
        persistent_workers=False,
        batch_size=train_config.max_samples,
        shuffle=True,
    )

    total_updates = (
        math.ceil(len(train_dataloader) / train_config.gradient_accumulation_steps / accelerator.num_processes)
        * train_config.num_train_epochs
    )
    warmup_updates = train_config.warmup_steps
    decay_updates = max(total_updates - warmup_updates, 1)
    warmup_scheduler = LinearLR(optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_updates)
    decay_scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=1e-8, total_iters=decay_updates)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, decay_scheduler], milestones=[warmup_updates])
    train_dataloader, scheduler = accelerator.prepare(train_dataloader, scheduler)

    val_loader = None
    val_dummy_input = None
    if val_dataset is not None:
        saved_even_batches = accelerator.even_batches
        accelerator.even_batches = True
        val_loader = DataLoader(
            val_dataset,
            collate_fn=val_dataset.collate_fn,
            num_workers=train_config.dataloader_num_workers,
            batch_size=4,
            shuffle=False,
        )
        val_loader = accelerator.prepare(val_loader)
        accelerator.even_batches = saved_even_batches
        val_dummy_input = next(iter(val_loader))

    samples_dir = f"{output_dir}/samples"
    val_curves_dir = f"{output_dir}/val_curves"
    updates_per_epoch = math.ceil(len(train_dataloader) / train_config.gradient_accumulation_steps)

    start_epoch = 0
    global_update = 0
    updates_done_in_epoch = 0
    resume_ckpt = os.path.join(output_dir, "model_last.pt")
    if os.path.exists(resume_ckpt):
        state = torch.load(resume_ckpt, map_location="cpu", weights_only=False)
        try:
            from deepspeed.runtime.zero.partition_parameters import GatheredParameters
        except ImportError:
            from deepspeed.zero import GatheredParameters
        raw_model = accelerator.unwrap_model(model)
        with GatheredParameters(raw_model.parameters(), modifier_rank=0):
            if accelerator.is_main_process:
                raw_model.load_state_dict(state["model_state_dict"], strict=False)
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        global_update = int(state["update"])
        start_epoch = global_update // updates_per_epoch
        updates_done_in_epoch = global_update % updates_per_epoch
        logger.info(
            f"resumed from {resume_ckpt} | update={global_update} "
            f"start_epoch={start_epoch} skip_batches={updates_done_in_epoch * train_config.gradient_accumulation_steps}"
        )

    for epoch in range(start_epoch, train_config.num_train_epochs):
        model.train()
        epoch_loader = train_dataloader
        if epoch == start_epoch and updates_done_in_epoch > 0:
            epoch_loader = accelerator.skip_first_batches(
                train_dataloader, updates_done_in_epoch * train_config.gradient_accumulation_steps
            )
        for batch in epoch_loader:
            with accelerator.accumulate(model):
                latent, latent_lens, cond_inputs, extras = prepare_batch(batch, vae_model)
                with accelerator.autocast():
                    loss, _, _ = model(latent, text=cond_inputs, lens=latent_lens, **extras)
                accelerator.backward(loss)
                if train_config.max_grad_norm > 0 and accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), train_config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_update += 1

                if accelerator.is_main_process and global_update % train_config.logging_steps == 0:
                    batch_size = latent.shape[0]
                    noise_len = latent.shape[1]
                    ref_len = extras["ref_latent"].shape[1]
                    text_len = cond_inputs["input_ids"].shape[1]
                    logger.info(
                        f"[epoch {epoch + 1}/{train_config.num_train_epochs} | update {global_update}/{total_updates}] "
                        f"loss={loss.item():.4f} lr={scheduler.get_last_lr()[0]:.2e} "
                        f"bsz={batch_size} seq_len=noise:{noise_len}/ref:{ref_len}/text:{text_len}"
                    )

                if global_update % train_config.last_per_updates == 0:
                    save_checkpoint(accelerator, model, optimizer, scheduler, global_update, output_dir, last=True)
                if global_update % train_config.save_per_updates == 0:
                    save_checkpoint(accelerator, model, optimizer, scheduler, global_update, output_dir)
                if val_loader is not None and global_update % train_config.val_per_updates == 0:
                    t_grid, val_losses = evaluate(accelerator, model, val_loader, vae_model)
                    if accelerator.is_main_process:
                        log_val_curve(writer, val_curves_dir, t_grid, val_losses, global_update)
                    synthesize_and_save(accelerator, model, val_dummy_input, vae_model, global_update, samples_dir)

    save_checkpoint(accelerator, model, optimizer, scheduler, global_update, output_dir, last=True)
    if writer is not None:
        writer.close()
    accelerator.end_training()


def main():
    parser = HfArgumentParser((ScriptArgs, TrainConfig, LoraArgs))
    args, train_config, lora_args = parser.parse_args_into_dataclasses()

    random.seed(train_config.seed)
    torch.manual_seed(train_config.seed)

    model_config = OmegaConf.load(args.config).model
    text_encoder_config = model_config.text_encoder
    vae_config = model_config.vae

    saved_config = os.path.join(args.output_dir, "config.yaml")
    if os.environ.get("RANK", "0") == "0" and not os.path.exists(saved_config):
        os.makedirs(args.output_dir, exist_ok=True)
        shutil.copy(args.config, saved_config)

    logger.info(f"Loading Qwen text encoder from {text_encoder_config.text_encoder_path} ...")
    thinker = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        text_encoder_config.text_encoder_path, torch_dtype=torch.bfloat16
    )
    if thinker.visual is not None:
        del thinker.visual
        thinker.visual = None
    text_encoder = thinker
    text_processor = Qwen2_5OmniProcessor.from_pretrained(text_encoder_config.text_encoder_path)

    logger.info(f"Loading VAE from {vae_config.vae_model_path} ...")
    model_init_kwargs = OmegaConf.to_container(vae_config.get("model_init_kwargs", OmegaConf.create({})), resolve=True)
    vae_model = load_vae_model(
        vae_name=vae_config.vae_name,
        vae_cfg=BigVGANFlowVAEConfig.from_dict(model_init_kwargs),
        vae_ckpt=vae_config.vae_model_path,
        map_location="cpu",
    )
    vae_model.eval()
    vae_model.requires_grad_(False)

    logger.info("Building CFMEdit model ...")
    model_arc = OmegaConf.to_container(model_config.arch, resolve=True)
    model_arc["attn_backend"] = "torch"
    model_arc["checkpoint_activations"] = True
    model_arc["checkpoint_every_n_layers"] = 1
    schedule_config = OmegaConf.to_container(model_config.get("schedule", OmegaConf.create({})), resolve=True)
    model = CFMEdit(
        transformer=Flux2Edit(**model_arc, latent_dim=vae_config.latent_dim),
        text_encoder=text_encoder,
        text_processor=text_processor,
        num_channels=vae_config.latent_dim,
        **schedule_config,
    )

    if os.path.exists(os.path.join(args.output_dir, "model_last.pt")):
        logger.info(f"found {args.output_dir}/model_last.pt | will resume, skipping init_ckpt cold-start load")
    else:
        logger.info(f"Loading init weights from {args.init_ckpt} ...")
        if args.init_ckpt.endswith(".safetensors"):
            state_dict = load_file(args.init_ckpt, device="cpu")
        else:
            checkpoint = torch.load(args.init_ckpt, map_location="cpu", weights_only=False)
            ema = checkpoint["ema_model_state_dict"]
            state_dict = {k.replace("ema_model.", ""): v for k, v in ema.items() if k not in ("initted", "step")}
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        non_te_missing = [k for k in missing if not k.startswith("text_encoder.")]
        logger.info(f"init weights loaded | missing(non-text-encoder)={len(non_te_missing)} unexpected={len(unexpected)}")
        if non_te_missing:
            logger.warning(f"missing non-text-encoder keys, examples: {non_te_missing[:10]}")

    n_adapted = apply_lora(
        model.transformer,
        r=lora_args.lora_r,
        alpha=lora_args.lora_alpha,
        dropout=lora_args.lora_dropout,
        targets=lora_args.lora_targets,
    )
    n_trainable = freeze_for_lora(model.transformer)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        f"LoRA: adapted={n_adapted} layers | trainable={trainable:,}/{total:,} "
        f"({100 * trainable / total:.2f}%) | trainable_params={n_trainable}"
    )

    train_dataset = AukJsonlDataset(args.train_jsonl, text_processor)
    val_dataset = None
    if args.val_jsonl:
        val_dataset = AukJsonlDataset(args.val_jsonl, text_processor)

    logger.info(f"train={len(train_dataset)} val={len(val_dataset) if val_dataset else 0}")
    train_loop(model, vae_model, train_dataset, val_dataset, train_config, args.output_dir)


if __name__ == "__main__":
    main()
