# Fine-tuning AuK

AuK fine-tuning uses the same ChatML-style multimodal conditioning format as inference. Each training sample contains a natural-language instruction, optional reference audio, and a target audio.

This guide covers data preparation, training, checkpointing, and inference with a fine-tuned model. For a minimal example, see the [Fine-tuning](../README.md#fine-tuning) section of the README.

## Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Training Configuration](#training-configuration)
- [Dynamic batching & memory](#dynamic-batching--memory)
- [Checkpoints, resuming & EMA](#checkpoints-resuming--ema)
- [Monitoring](#monitoring)
- [Inference with a fine-tuned checkpoint](#inference-with-a-fine-tuned-checkpoint)

## Overview

The training entry point is
[`src/auk/train/train.py`](../src/auk/train/train.py), launched through
[`scripts/train.sh`](../scripts/train.sh).

By default, fine-tuning updates the AuK generation backbone and its
conditioning-fusion parameters, while keeping the Qwen2.5-Omni encoder
and the AuK VAE frozen.

Training is implemented with [🤗 Accelerate](https://github.com/huggingface/accelerate), DDP, and `bf16` mixed precision.

## Quick Start

### 1. Install training dependencies
In addition to the base inference install, fine-tuning needs a few more packages (declared under `[train]` in [`pyproject.toml`](../pyproject.toml)):

```bash
pip install -e ".[train]"
```

### 2. Prepare a training manifest

Training data is a **JSONL** file, one JSON object per line. Each line is a single
source→target pair:

- `duration` — target-audio duration in **seconds** (float). **Required** — it
  drives duration-based dynamic batching.
- `messages` — a ChatML-style list:
  - The **`user`** message carries the instruction (`text`) and, when the task
    needs it, a reference/source audio.
  - The **`assistant`** message carries the target audio the model should learn
    to produce.

Audio is referenced by file path under either an `audio_url` or `audio` key (both
are accepted). Paths should be absolute. Audio is loaded, downmixed to mono, and
resampled to 24 kHz automatically, so source files can be any sample rate.

**Task with a reference/source audio** (e.g. voice cloning, editing, enhancement):

```json
{
  "duration": 5.66,
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Say the following with the same voice: \"你好，这是一次语音合成测试。\""},
        {"type": "audio", "audio_url": "/abs/path/ref.wav"}
      ]
    },
    {
      "role": "assistant",
      "content": [
        {"type": "audio", "audio_url": "/abs/path/target.wav"}
      ]
    }
  ]
}
```

**Task without a reference audio** (e.g. instruction-only / description-driven TTS):

```json
{
  "duration": 5.66,
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Say “Even the darkest night eventually gives way to morning” in Evelyn’s warm, clear female voice with a soft, slightly husky timbre, gentle resonance, and calm, expressive delivery."}
      ]
    },
    {
      "role": "assistant",
      "content": [
        {"type": "audio", "audio_url": "/abs/path/target.wav"}
      ]
    }
  ]
}
```

Every supported task uses this one format; you only change the instruction and
the source/target audio. Prepare a `train.jsonl` and, optionally, a `val.jsonl`
(used for periodic per-t validation loss and sample synthesis).

### 3. Launch training

Edit the resource paths at the top of [`scripts/train.sh`](../scripts/train.sh),
then run it:

```bash
train_jsonl=/path/to/train.jsonl # train dataset
val_jsonl=/path/to/val.jsonl # optional validation dataset
config=ckpts/AuK/config.yaml  # model config path
init_ckpt=ckpts/AuK/auk_base.safetensors # pretrain ckpt
output_dir=ckpts/auk_finetune # save dir
num_gpus=8 # GPUs
```
Then launch training:
```bash
bash scripts/train.sh
```

## Training Configuration

A training run is configured by:
1. The model architecture in the YAML file
2. The training hyperparameters passed through command-line flags

### Model configuration (`--config`)

`--config` points at a model YAML (default `ckpts/AuK/config.yaml`, the
config bundled with the downloaded weights). Only the model.* section is used by the trainer. It defines:

- The AuK generation architecture
- The flow-matching time-sampling schedule
- The Qwen2.5-Omni encoder path
- The VAE configuration
- Activation-checkpointing settings

### Common training options

Defaults come from the `TrainConfig` dataclass in
[`src/auk/train/train.py`](../src/auk/train/train.py). Paths come from
`ScriptArgs`.

| Flag | Default | Meaning |
| --- | --- | --- |
| `--train_jsonl` | *(required)* | Path to the training JSONL. |
| `--val_jsonl` | `None` | Optional validation JSONL; enables per-t val loss + sample synthesis. |
| `--config` | `ckpts/AuK/config.yaml` | Model config YAML (reads `model.*` only). |
| `--init_ckpt` | `ckpts/AuK/auk_base.safetensors` | Weights to initialize from (cold start). |
| `--output_dir` | `ckpts/auk_finetune` | Where checkpoints, logs, samples, the run script, and a copy of `config.yaml` are written. |
| `--learning_rate` / `--lr` | `2e-5` | AdamW learning rate (`betas=(0.9, 0.95)`). |
| `--max_grad_norm` | `1.0` | Gradient-clipping threshold (`0` disables). |
| `--num_train_epochs` / `--epochs` | `200` | Number of epochs. |
| `--gradient_accumulation_steps` | `1` | Micro-batches accumulated per optimizer step. |
| `--warmup_steps` | `100` | Warmup steps; the effective count is multiplied by the number of GPUs. |
| `--frames_threshold` | `2700` | VAE latent frames per GPU per batch — the main memory/throughput knob. |
| `--max_samples` | `8` | Hard cap on samples per batch (`0` = unlimited). |
| `--ema_beta` | `0.9999` | EMA decay. |
| `--ema_update_after_step` | `100` | Start updating EMA after this many sync steps. |
| `--ema_update_every` | `10` | Update EMA every N sync steps. |
| `--save_per_updates` | `1000` | Save a numbered snapshot `model_{update}.pt` every N updates. |
| `--last_per_updates` | `1000` | Refresh the `model_last.pt` resume point every N updates. |
| `--logging_steps` | `10` | Log loss/lr/shapes every N updates. |
| `--val_per_updates` | `200` | Run validation + synthesis every N updates (if `val_jsonl` set). |
| `--dataloader_num_workers` | `4` | DataLoader worker processes. |
| `--seed` | `666` | Random seed (also seeds batch shuffling). |

The learning-rate schedule is linear warmup (from ~0 to `learning_rate` over
`warmup_steps × num_gpus` updates) followed by linear decay back toward ~0 over
the remaining updates.

Flags you will most often tune for fine-tuning: `--learning_rate`,
`--num_train_epochs` (smaller datasets need far fewer than 200),
`--frames_threshold` / `--max_samples` (memory), and the save/val cadence.

## Dynamic batching & memory

AuK does **not** use a fixed batch size. Instead, `DynamicBatchSampler` sorts
samples by latent-frame length and greedily packs each batch until it hits
`frames_threshold` (VAE latent frames per GPU), capped at `max_samples` samples.
Packing similar-length clips together keeps padding low and GPU utilization high.

- Latent frames ≈ `duration × 24000 / 480` (the VAE downsamples 24 kHz audio by
  480×, i.e. to a 50 Hz latent). So `frames_threshold=2700` ≈ 54 seconds of audio
  packed per GPU per batch.
- Only clips between **0.3 s and 30 s** are included; anything outside that range
  is skipped. Make sure your `duration` values are accurate.

**To reduce memory (avoid OOM):** lower `--frames_threshold` first, then
`--max_samples`. To recover the effective batch size, raise
`--gradient_accumulation_steps`. Activation checkpointing is already enabled in
the model config (`checkpoint_activations: True`, every 4 layers), so you don't
need to toggle it.

## Checkpoints, resuming & EMA

All artifacts are written under `--output_dir`:

| Path | What it is |
| --- | --- |
| `model_last.pt` | Rolling **resume point**, refreshed every `last_per_updates`. Contains model + EMA + optimizer + scheduler state + update count. |
| `model_{update}.pt` | Numbered **snapshots**, saved every `save_per_updates`. Same contents, for keeping intermediate checkpoints. |
| `tensorboard/` | TensorBoard event files. |
| `val_curves/` | Per-t validation-loss PNGs (`update_{N}.png`). |
| `samples/` | Synthesized `update_{N}_gen.wav` / `_tgt.wav` pairs. |
| `<timestamp>_run.sh` | The exact launch command for this run. |

**Automatic resume.** If `--output_dir` already contains `model_last.pt`, training
resumes from it — restoring model/optimizer/scheduler/EMA and skipping the already-seen
batches — and the `--init_ckpt` cold start is skipped. To start a *fresh* run from
the base weights, use a new empty `--output_dir` (or remove the old `model_last.pt`).

**EMA is what you deploy.** An exponential-moving-average copy of the weights is
maintained throughout training; inference loads the EMA weights, not the raw
online weights (see below).

**Missing `text_encoder.*` keys are expected.** Both the init checkpoint and saved
checkpoints deliberately exclude the Qwen text encoder and the VAE (they are loaded
separately at runtime). Checkpoints load with `strict=False`, so the log will report
missing `text_encoder.*` keys — this is normal, not an error. The trainer only warns
about missing keys that are *not* under `text_encoder.`.

## Monitoring

With `--val_jsonl` set, every `val_per_updates` the trainer:

1. Computes a **per-t validation loss** across a `t ∈ {0.0, 0.1, …, 0.9}` grid
   (shared noise per batch so the curve reflects `t` alone), logged to the console
   and TensorBoard and saved as a PNG under `val_curves/`.
2. **Synthesizes a sample** from the first validation batch and writes the
   generated and target wavs to `samples/`, so you can listen to progress directly.

Launch TensorBoard to watch training/validation curves:

```bash
tensorboard --logdir ckpts/auk_finetune/tensorboard
```

Training-loss lines (`loss`, `lr`, batch size, sequence lengths) are printed every
`logging_steps` updates.

## Inference with a fine-tuned checkpoint

Your fine-tuned weights land as `.pt` files. The only thing that differs from
base-model inference is where you point the checkpoint: the inference code
([`src/auk/infer/infer_auk.py`](../src/auk/infer/infer_auk.py)) automatically reads
the **EMA** weights from a training `.pt` (it pulls `ema_model_state_dict` and strips
the `ema_model.` prefix), so you can point the CLI or `AukInfer` straight at a checkpoint:

```bash
auk-infer \
    --ckpt ckpts/auk_finetune/model_last.pt \
    --audio assets/ref.wav \
    --instruction "Say the following with the same voice: '...'" \
    --output out.wav \
    --gen_seconds 6.0
```

```python
from auk.infer.infer_auk import AukInfer

engine = AukInfer(
    "ckpts/AuK/config.yaml",
    "ckpts/auk_finetune/model_last.pt",
)
```

The VAE (`vae.safetensors`) is auto-detected next to the checkpoint if present,
otherwise the path from the config is used.

Everything else is identical to base-model inference — the full task catalog,
instruction templates, duration hints, and CLI/Python usage all live in the
[Inference Cookbook](COOKBOOK.md). Just swap in your fine-tuned `.pt` wherever a
checkpoint path appears.