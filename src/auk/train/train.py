import json
import logging
import math
import os
import random
import shutil
import sys
from dataclasses import asdict, dataclass, field
from typing import Optional

import matplotlib.pyplot as plt
import torch
import torchaudio
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from ema_pytorch import EMA
from omegaconf import OmegaConf
from safetensors.torch import load_file
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset, Sampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import HfArgumentParser, Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration

from auk.model import CFMEdit, Flux2Edit
from auk.model.vae import load_vae_model
from auk.model.vae.bigvgan_flow_vae import BigVGANFlowVAEConfig


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.getLogger().addFilter(lambda record: "System prompt modified" not in record.getMessage())
logger = logging.getLogger(__name__)

plt.switch_backend("Agg")  # headless training nodes have no display


@dataclass
class TrainConfig:
    learning_rate: float = field(default=2e-5, metadata={"aliases": ["--lr"]})
    max_grad_norm: float = 1.0
    num_train_epochs: int = field(default=200, metadata={"aliases": ["--epochs"]})
    gradient_accumulation_steps: int = 1
    warmup_steps: int = 100
    frames_threshold: int = 2700  # VAE latent frames per GPU (OOM-safe with ckpt every_n=4)
    max_samples: int = 8
    ema_beta: float = 0.9999
    ema_update_after_step: int = 100
    ema_update_every: int = 10
    save_per_updates: int = 1000
    last_per_updates: int = 1000
    logging_steps: int = 10
    val_per_updates: int = 200
    dataloader_num_workers: int = 4
    seed: int = 666


@dataclass
class ScriptArgs:
    """Paths and resources (not hyper-params). Kept separate from TrainConfig."""

    train_jsonl: str
    val_jsonl: Optional[str] = None
    config: str = field(default="ckpts/AuK/config.yaml", metadata={"help": "model config yaml (reads model.* only)"})
    init_ckpt: str = "ckpts/AuK/auk_base.safetensors"
    output_dir: str = "ckpts/auk_finetune"


class AukJsonlDataset(Dataset):
    def __init__(self, jsonl_path, processor):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            self.data = [json.loads(line) for line in f if line.strip()]
        self.durations = [row["duration"] for row in self.data]
        self.target_sample_rate = 24000
        self.vae_downsample_rate = 480
        self.processor = processor
        self.resamplers = {}

    def __len__(self):
        return len(self.data)

    def get_frame_len(self, index):
        return self.durations[index] * self.target_sample_rate / self.vae_downsample_rate

    def load_wav(self, path):
        audio, source_sample_rate = torchaudio.load(path)
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        if source_sample_rate != self.target_sample_rate:
            if source_sample_rate not in self.resamplers:
                self.resamplers[source_sample_rate] = torchaudio.transforms.Resample(source_sample_rate, self.target_sample_rate)
            audio = self.resamplers[source_sample_rate](audio)
        return audio

    def __getitem__(self, index):
        row = self.data[index]
        messages = row["messages"]
        if isinstance(messages, str):
            messages = json.loads(messages)

        ref_url, gen_url = None, None
        for msg in messages:
            for c in msg["content"]:
                if c["type"] != "audio":
                    continue
                url = c.get("audio") or c.get("audio_url")
                if msg["role"] == "user":
                    ref_url = url
                elif msg["role"] == "assistant":
                    gen_url = url

        ref_audio = torch.empty(1, 0)
        if ref_url is not None:
            ref_audio = self.load_wav(ref_url)

        gen_audio = torch.empty(1, 0)
        if gen_url is not None:
            gen_audio = self.load_wav(gen_url)

        for msg in messages:
            if msg["role"] != "user" or any(c["type"] == "audio" for c in msg["content"]):
                continue
            for c in msg["content"]:
                if c["type"] == "text" and not c["text"].endswith("|<no_prompt_audio>|"):
                    c["text"] = c["text"] + "|<no_prompt_audio>|"

        cond_messages = [m for m in messages if m["role"] != "assistant"]
        return {"audio": gen_audio, "ref_audio": ref_audio, "messages": cond_messages}

    def collate_fn(self, batch):
        audios = [item["audio"] for item in batch]
        audio_lengths = torch.LongTensor([a.shape[-1] for a in audios])
        max_len = audio_lengths.amax()
        audios = torch.stack([torch.nn.functional.pad(a, (0, max_len - a.size(-1))) for a in audios])

        ref_audios = [item["ref_audio"] for item in batch]
        ref_audio_lengths = torch.LongTensor([a.shape[-1] for a in ref_audios])
        ref_max = ref_audio_lengths.amax()
        ref_audios = torch.stack([torch.nn.functional.pad(a, (0, ref_max - a.size(-1))) for a in ref_audios])

        conversations = [item["messages"] for item in batch]
        cond_inputs = CFMEdit.build_cond_inputs(conversations, self.processor)
        return {
            "audio": audios,
            "audio_lengths": audio_lengths,
            "ref_audio": ref_audios,
            "ref_audio_lengths": ref_audio_lengths,
            "messages": conversations,
            "cond_inputs": cond_inputs,
        }


class DynamicBatchSampler(Sampler[list[int]]):
    """Sort by frame length, then greedily pack samples until frames_threshold, keeping padding low."""

    def __init__(self, sampler, frames_threshold, max_samples=0, random_seed=None, drop_residual=False):
        self.sampler = sampler
        self.frames_threshold = frames_threshold
        self.max_samples = max_samples
        self.random_seed = random_seed
        self.epoch = 0

        data_source = self.sampler.data_source
        downsample_rate = data_source.vae_downsample_rate
        min_frame_length = 0.3 * data_source.target_sample_rate / downsample_rate
        max_frame_length = 30 * data_source.target_sample_rate / downsample_rate

        indices = []
        for idx in tqdm(self.sampler, desc="Sorting by frame length"):
            frame_len = data_source.get_frame_len(idx)
            if min_frame_length <= frame_len <= max_frame_length:
                indices.append((idx, frame_len))
        indices.sort(key=lambda elem: elem[1])

        batches = []
        batch = []
        batch_frames = 0
        for idx, frame_len in tqdm(indices, desc=f"Packing batches ({frames_threshold} frames/gpu)"):
            if batch_frames + frame_len <= self.frames_threshold and (max_samples == 0 or len(batch) < max_samples):
                batch.append(idx)
                batch_frames += frame_len
            else:
                if len(batch) > 0:
                    batches.append(batch)
                if frame_len <= self.frames_threshold:
                    batch = [idx]
                    batch_frames = frame_len
                else:
                    batch = []
                    batch_frames = 0
        if not drop_residual and len(batch) > 0:
            batches.append(batch)

        self.batches = batches
        self.drop_last = True

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        if self.random_seed is not None:
            g = torch.Generator()
            g.manual_seed(self.random_seed + self.epoch)
            indices = torch.randperm(len(self.batches), generator=g).tolist()
            return iter([self.batches[i] for i in indices])
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


def prepare_batch(batch, vae_model):
    vae_model.eval()
    with torch.no_grad():
        latent, latent_lengths = vae_model.encoding_and_normalization(
            batch["audio"],
            sample_lengths=batch["audio_lengths"],
        )
        if batch["ref_audio"].size(-1) == 0:
            B = batch["ref_audio"].size(0)
            ref_latent = torch.zeros(B, 0, latent.size(-1), device=latent.device, dtype=latent.dtype)
            ref_lens = torch.zeros(B, dtype=torch.long, device=latent.device)
        else:
            ref_latent, ref_lens = vae_model.encoding_and_normalization(
                batch["ref_audio"],
                sample_lengths=batch["ref_audio_lengths"],
            )
    return latent, latent_lengths, batch["cond_inputs"], {"ref_latent": ref_latent, "ref_lens": ref_lens}


def save_checkpoint(accelerator, model, ema_model, optimizer, scheduler, update, output_dir, last=False):
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return
    unwrapped = accelerator.unwrap_model(model)
    model_state = {k: v for k, v in unwrapped.state_dict().items() if not k.startswith("text_encoder.")}
    ema_state = {k: v for k, v in ema_model.state_dict().items() if "text_encoder." not in k}
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
def evaluate(accelerator, model, val_loader, vae_model):
    """Per-t val loss: for each t in the grid, compute flow-matching loss with t (and x0) fixed.

    x0 is drawn once per batch and shared across all t so the curve reflects t alone, not
    per-t noise variance. Returns (t_grid, mean_losses) with the losses aligned to t_grid.
    """
    t_grid = [round(0.1 * i, 1) for i in range(10)]
    model.eval()
    loss_sums = torch.zeros(len(t_grid), device=accelerator.device)
    count = 0
    for batch in val_loader:
        latent, latent_lens, cond_inputs, extras = prepare_batch(batch, vae_model)
        B = latent.size(0)
        x0 = torch.randn_like(latent)  # shared across all t for this batch
        per_t = []
        for t_val in t_grid:
            time = torch.full((B,), float(t_val), device=latent.device, dtype=latent.dtype)
            with accelerator.autocast():
                loss, _, _ = model(latent, text=cond_inputs, lens=latent_lens, time=time, x0=x0, apply_cond_drop=False, **extras)
            per_t.append(loss.detach())
        gathered = accelerator.gather_for_metrics(torch.stack(per_t).unsqueeze(0).repeat(B, 1))  # [sum_B, n_t]
        loss_sums += gathered.sum(0)
        count += gathered.size(0)
    model.train()
    return t_grid, (loss_sums / max(count, 1)).tolist()


def log_val_curve(writer, val_curves_dir, t_grid, losses, update):
    """Emit per-t val loss three ways: log line, PNG snapshot (loss vs t), TensorBoard (loss vs update)."""
    mean = sum(losses) / len(losses)
    pairs = " ".join(f"t{t:.1f}={loss:.4f}" for t, loss in zip(t_grid, losses))
    logger.info(f"[update {update}] val loss per-t | mean={mean:.4f} | {pairs}")

    if writer is not None:
        writer.add_scalars("val/loss_per_t", {f"t{t:.1f}": loss for t, loss in zip(t_grid, losses)}, update)
        writer.add_scalar("val/loss_mean", mean, update)

    os.makedirs(val_curves_dir, exist_ok=True)
    fig, ax = plt.subplots()
    ax.plot(t_grid, losses, marker="o")
    ax.set_xlabel("t")
    ax.set_ylabel("val loss")
    ax.set_title(f"update {update}")
    fig.savefig(f"{val_curves_dir}/update_{update}.png")
    plt.close(fig)


@torch.no_grad()
def synthesize_and_save(accelerator, model, batch, vae_model, update, samples_dir):
    if not accelerator.is_main_process:
        return
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
    model.train()


def train_loop(model, vae_model, train_dataset, val_dataset, train_config, output_dir):
    # CFG training randomly drops the text/audio condition, so some params (e.g. text proj)
    # skip the graph on those steps — DDP must tolerate unused params.
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(gradient_accumulation_steps=train_config.gradient_accumulation_steps, kwargs_handlers=[ddp_kwargs])

    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=train_config.learning_rate, betas=(0.9, 0.95))
    model, optimizer = accelerator.prepare(model, optimizer)

    vae_model = vae_model.to(accelerator.device)
    vae_model.requires_grad_(False)

    if accelerator.is_main_process:
        ema_model = EMA(
            accelerator.unwrap_model(model),
            beta=train_config.ema_beta,
            update_after_step=train_config.ema_update_after_step,
            update_every=train_config.ema_update_every,
            include_online_model=False,
        )
        ema_model.to(accelerator.device)
    else:
        ema_model = None

    writer = SummaryWriter(f"{output_dir}/tensorboard") if accelerator.is_main_process else None

    accelerator.even_batches = False
    sampler = SequentialSampler(train_dataset)
    batch_sampler = DynamicBatchSampler(
        sampler, train_config.frames_threshold, max_samples=train_config.max_samples, random_seed=train_config.seed
    )
    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=train_dataset.collate_fn,
        num_workers=train_config.dataloader_num_workers,
        pin_memory=True,
        persistent_workers=train_config.dataloader_num_workers > 0,
        batch_sampler=batch_sampler,
    )

    warmup_updates = train_config.warmup_steps * accelerator.num_processes
    total_updates = math.ceil(len(train_dataloader) / train_config.gradient_accumulation_steps) * train_config.num_train_epochs
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
        accelerator.unwrap_model(model).load_state_dict(state["model_state_dict"], strict=False)
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        if accelerator.is_main_process:
            ema_model.load_state_dict(state["ema_model_state_dict"], strict=False)
        global_update = int(state["update"])
        start_epoch = global_update // updates_per_epoch
        updates_done_in_epoch = global_update % updates_per_epoch
        logger.info(
            f"resumed from {resume_ckpt} | update={global_update} "
            f"start_epoch={start_epoch} skip_batches={updates_done_in_epoch * train_config.gradient_accumulation_steps}"
        )

    for epoch in range(start_epoch, train_config.num_train_epochs):
        model.train()
        batch_sampler.set_epoch(epoch)
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
                if accelerator.is_main_process:
                    ema_model.update()
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
                    save_checkpoint(accelerator, model, ema_model, optimizer, scheduler, global_update, output_dir, last=True)
                if global_update % train_config.save_per_updates == 0:
                    save_checkpoint(accelerator, model, ema_model, optimizer, scheduler, global_update, output_dir)
                if val_loader is not None and global_update % train_config.val_per_updates == 0:
                    t_grid, val_losses = evaluate(accelerator, model, val_loader, vae_model)
                    if accelerator.is_main_process:
                        log_val_curve(writer, val_curves_dir, t_grid, val_losses, global_update)
                    synthesize_and_save(accelerator, model, val_dummy_input, vae_model, global_update, samples_dir)

    save_checkpoint(accelerator, model, ema_model, optimizer, scheduler, global_update, output_dir, last=True)
    if writer is not None:
        writer.close()
    accelerator.end_training()


def main():
    parser = HfArgumentParser((ScriptArgs, TrainConfig))
    args, train_config = parser.parse_args_into_dataclasses()

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
    # keep the full multimodal Thinker (text + ref_audio); drop the unused vision tower
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
    schedule_config = OmegaConf.to_container(model_config.get("schedule", OmegaConf.create({})), resolve=True)
    model = CFMEdit(
        transformer=Flux2Edit(**model_arc, latent_dim=vae_config.latent_dim),
        text_encoder=text_encoder,
        text_processor=text_processor,
        num_channels=vae_config.latent_dim,
        **schedule_config,
    )

    # If output_dir already holds a model_last.pt, train_loop resumes from it (full training
    # state); skip the init_ckpt cold-start load here so we don't clobber the resumed weights.
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

    train_dataset = AukJsonlDataset(args.train_jsonl, text_processor)
    val_dataset = None
    if args.val_jsonl:
        val_dataset = AukJsonlDataset(args.val_jsonl, text_processor)

    logger.info(f"train={len(train_dataset)} val={len(val_dataset) if val_dataset else 0} | config: {asdict(train_config)}")
    train_loop(model, vae_model, train_dataset, val_dataset, train_config, args.output_dir)


if __name__ == "__main__":
    sys.exit(main())
