# LoRA fine-tuning

LoRA adapts a small set of linear layers while the Qwen2.5-Omni encoder, the
AuK VAE, and the remaining backbone weights stay frozen. It is aimed at
cross-lingual adaptation, where a full fine-tune of the generation backbone
would drift the timbre of the reference speaker.

This guide covers the adapter layout, the launch script, merging the adapters
back into the backbone, and the ZeRO-3 constraints that the trainer has to
respect. Data preparation is identical to [FINETUNING.md](FINETUNING.md).

## Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Adapter targets](#adapter-targets)
- [Training configuration](#training-configuration)
- [Merging and inference](#merging-and-inference)
- [Memory and throughput](#memory-and-throughput)
- [Notes](#notes)
- [Reference results](#reference-results)

## Overview

The entry point is
[`src/auk/train/lora_train.py`](../src/auk/train/lora_train.py), launched through
[`scripts/train_lora.sh`](../scripts/train_lora.sh). The adapter implementation
lives in [`src/auk/train/lora.py`](../src/auk/train/lora.py) and does not depend
on `peft`: an adapted layer is an `nn.Linear` subclass with `lora_A`, `lora_B`,
and a `lora_scaling` buffer, so the checkpoint keeps the official key layout.

Two parameter groups remain fully trainable rather than adapted:
`layer_weights` and `layer_scale`, the encoder layer-fusion parameters. They
hold a few dozen values in total, so training them directly is cheaper and more
stable than wrapping them in an adapter.

Training uses [🤗 Accelerate](https://github.com/huggingface/accelerate),
[DeepSpeed ZeRO-3](https://www.deepspeed.ai/tutorials/zero/), and `bf16`.
ZeRO-3 is what keeps the frozen 5.6B backbone affordable on smaller cards.

## Quick Start

### 1. Install the training and DeepSpeed dependencies

```bash
pip install -e ".[train]"
pip install deepspeed
```

### 2. Prepare a training manifest

Use the same JSONL format as full fine-tuning: `duration`, a `user` message
carrying the instruction and reference audio, and an `assistant` message
carrying the target audio. See [FINETUNING.md](FINETUNING.md#2-prepare-a-training-manifest)
for the schema and examples.

### 3. Launch training

Edit the paths at the top of [`scripts/train_lora.sh`](../scripts/train_lora.sh):

```bash
train_jsonl=/path/to/train.jsonl # train dataset
val_jsonl=/path/to/val.jsonl     # optional validation dataset
config=ckpts/AuK/config.yaml     # model config path
init_ckpt=ckpts/AuK/auk_base.safetensors # pretrain ckpt
output_dir=ckpts/auk_lora         # save dir
num_gpus=8                        # GPUs
```

Then launch training:

```bash
bash scripts/train_lora.sh
```

A healthy start prints the number of adapted layers and the trainable share:

```
LoRA: adapted=200 layers | trainable=41,287,717/5,606,606,437 (0.74%) | trainable_params=400
```

## Adapter targets

Targets match the full module path suffix of every `nn.Linear`, so layers nested
in `ModuleList` or `Sequential` (for example `attn.to_out.0`) are reachable.

| Suffix | Module |
| --- | --- |
| `.attn.to_qkv`, `.attn.to_qkv_c` | audio / context QKV projections |
| `.attn.to_out.0`, `.attn.to_out_c` | attention output projections |
| `.ff.linear_in`, `.ff.linear_out` | DiT feed-forward |
| `.ff_c.*`, `.ff_x.*` | MMDiT dual-stream feed-forward |
| `.txt_proj` | text projection into the DiT width |
| `.proj_out` | final projection to latent space |
| `.audio_embed.linear` | audio prompt embedding |
| `.attn_norm*.linear`, `.norm_out.linear` | adaLN modulation |
| `.time_embed.time_mlp.0`, `.time_embed.time_mlp.2` | timestep MLP |

`LEGACY_TARGETS` in [`lora.py`](../src/auk/train/lora.py) keeps a smaller set
(attention QKV and feed-forward only) for ablations. Leave `lora_targets` empty
to use the full set, or pass a comma-separated suffix list to override it.

Adapting `audio_embed.linear` improves text and duration alignment but can cost
speaker similarity on language-transfer runs. If similarity matters more than
WER, drop that one suffix and retrain.

## Training configuration

LoRA-specific flags are defined in the `LoraArgs` dataclass; the remaining flags
come from `TrainConfig` and are documented in
[FINETUNING.md](FINETUNING.md#common-training-options).

| Flag | Default | Meaning |
| --- | --- | --- |
| `--lora_r` | `32` | Adapter rank. |
| `--lora_alpha` | `32` | Adapter scaling; effective scale is `alpha / r`. |
| `--lora_dropout` | `0.0` | Dropout applied before the low-rank path. |
| `--lora_targets` | `None` | Comma-separated suffix list; empty uses the full set. |
| `--learning_rate` | `5e-5` in `train_lora.sh` | Language adaptation is stable at this rate. |
| `--num_train_epochs` | `1` in `train_lora.sh` | One pass over a large corpus is usually enough. |
| `--max_samples` | `4` in `train_lora.sh` | Must equal `train_micro_batch_size_per_gpu` in `scripts/ds_zero3.json`. |

`max_samples` and the DeepSpeed micro-batch size are coupled: the fixed-size
DataLoader and the ZeRO-3 config must agree, otherwise the effective batch and
the learning-rate schedule diverge.

## Merging and inference

The inference loader builds a `Flux2Edit` without adapter layers, so pointing it
at `model_last.pt` silently ignores every `lora_*` key. Merge first:

```bash
python -m auk.train.merge_lora \
  --ckpt ckpts/auk_lora/model_last.pt \
  --out ckpts/auk_lora/auk_lora_merged.safetensors
```

The merged file matches the layout of `auk_base.safetensors`: adapters folded
into `weight`, `lora_*` keys removed, `text_encoder.*` dropped. A copy of
`config.yaml` is written into the output directory at training time, so the pair
can be passed to the inference CLI directly.

## Memory and throughput

- The frozen backbone is still resident and still gathered during forward
  passes. ZeRO-3 shards it across ranks; `--num_processes` therefore sets the
  per-GPU footprint as much as the throughput.
- Activation checkpointing is forced on in the trainer
  (`checkpoint_every_n_layers=1`), because the training loop does not use the
  dynamic batch sampler and activation memory would otherwise dominate.
- Each checkpoint stores the full backbone, not only the adapters. On a language
  adaptation run with 200 adapted layers the adapters themselves are about
  41M parameters (0.74%), but a checkpoint is still several gigabytes. Place
  `output_dir` on a volume with room for several snapshots, or raise
  `save_per_updates`.

## Notes

- **ZeRO-3 collectives.** `accelerator.get_state_dict` and `model.sample()` both
  gather sharded weights, so they are collectives and must run on every rank.
  Guarding them with `is_main_process` deadlocks the job after the NCCL timeout.
  `synthesize_and_save` therefore runs the forward pass on all ranks and only
  writes audio on the main process.
- **Resume.** Loading a checkpoint into a sharded model requires
  `GatheredParameters(..., modifier_rank=0)`; a plain `load_state_dict` fails
  with a size mismatch because non-owning ranks hold empty tensors.
- **Batch counts.** Every rank must see the same number of batches per epoch,
  otherwise one rank enters the next forward gather while the others wait in the
  gradient reduction. `even_batches` stays enabled for this reason.
- **Validation cost.** `evaluate()` computes the loss at ten noise levels, so one
  validation pass costs about ten forward passes over the validation set.
  Increase `val_per_updates` when validation dominates wall-clock time.

## Reference results

French language adaptation from `auk_base`, evaluated on the French subset of
X-Voice-Testset (500 prompts). WER uses `whisper-large-v3` with text
normalisation, SIM uses WavLM-ECAPA cosine similarity, and UTMOS uses
SpeechMOS. Generation used 32 NFE, CFG 2.0, and a fixed seed; training ran on
4 GPUs with `bf16` and ZeRO-3.

| Run | Adapters | Training data | WER ↓ | SIM ↑ | UTMOS ↑ |
| --- | --- | --- | --- | --- | --- |
| Base | none | none | 11.078 | 0.7617 | — |
| LoRA r=32 | attention + FFN | 69k utterances | 10.369 | 0.7578 | 3.4431 |
| LoRA r=32 | full set | 327k utterances | 9.028 | 0.7464 | 3.4752 |

The trend is consistent across runs: larger learning rates and the full target
set lower WER, while speaker similarity declines slowly. Runs differ in data and
learning rate, so the table is a design reference rather than a controlled
ablation.
