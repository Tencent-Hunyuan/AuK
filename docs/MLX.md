# AuK on Apple Silicon — native MLX backend

A from-scratch MLX implementation of AuK's three model stacks, validated
layer-by-layer against the PyTorch reference. Every weight is converted to MLX
layout and every op runs through MLX.

This backend is the supported way to run AuK on an Apple Silicon Mac. It does not
use the PyTorch MPS backend, and it does not change anything under `src/auk/` —
the CUDA path is untouched.

Verified on an M4 Pro (48 GB), macOS 26.4, Python 3.10, mlx 0.32.2.

## What was ported

| Component | File | Notes |
| --- | --- | --- |
| BigVGAN-Flow VAE | `src/auk_mlx/vae.py` | encoder + decoder; weight-norm folded at conversion |
| Flux2Edit DiT (1.53B) | `src/auk_mlx/dit.py` | 10 MMDiT + 20 DiT blocks, CFG batching, text cache |
| Qwen2.5-Omni Thinker | `src/auk_mlx/qwen_thinker.py` | 36-layer LLM + 32-layer audio tower, all hidden states |
| Euler sampler + pipeline | `src/auk_mlx/infer.py` | layer fusion, CFG, sway sampling, VAE decode |
| Weight converter | `src/auk_mlx/convert.py` | torch → MLX layout |
| CLI | `src/auk_mlx/cli.py` | `auk-mlx-infer` equivalent |

Deliberately not ported: the normalising flow and KL term (training only), the
talker / token2wav stacks (not used by AuK), and the vision tower (deleted
upstream too).

## Setup

```bash
uv venv --python 3.10 && source .venv/bin/activate
uv pip install -e . && uv pip install mlx
# pin transformers back if mlx-lm pulled v5
uv pip install "transformers>=4.52,<5"
```

Download the weights as in the main README (`ckpts/AuK`, `ckpts/AuK-Flash`,
`ckpts/Qwen2.5-Omni-3B`), then convert once:

```bash
export PYTHONPATH=src
python -m auk_mlx.convert vae     ckpts/AuK/vae.safetensors        ckpts/mlx/vae.safetensors
python -m auk_mlx.convert dit     ckpts/AuK/auk_base.safetensors   ckpts/mlx/dit_base.safetensors
python -m auk_mlx.convert dit     ckpts/AuK-Flash/auk_flash.safetensors ckpts/mlx/dit_flash.safetensors
python -m auk_mlx.convert thinker ckpts/Qwen2.5-Omni-3B            ckpts/mlx/thinker
```

Conversion writes about 28 GB of fp32 MLX weights (base and Flash DiT, Thinker
and VAE). The DiT converter also emits
`fusion_<variant>.safetensors` (`layer_weights`, `layer_scale`, `inv_freq`).

## Running

```bash
export PYTHONPATH=src
python -m auk_mlx.cli \
    --instruction "Say the following with the same voice: 'Hello, this is a test.'" \
    --audio ref_24k.wav --gen_seconds 4.0 --nfe 16 -o out.wav

# AuK-Flash: 4 fixed steps, CFG off
python -m auk_mlx.cli --flash --instruction "..." --audio ref_24k.wav -o out.wav
```

Reference audio of any sample rate or channel count is accepted: the engine
downmixes to mono and resamples to 24 kHz with `soxr`, mirroring what
`AukInfer._load_audio` does upstream (mono first, then resample). Omit `--audio`
for Instruct TTS and give `--gen_seconds`.

## Running the cookbook

```bash
PYTHONPATH=src .venv/bin/python scripts/run_cookbook_mlx.py --flash
PYTHONPATH=src .venv/bin/python scripts/run_cookbook_mlx.py --base --nfe 32
PYTHONPATH=src .venv/bin/python scripts/verify_cookbook_mlx.py /tmp/cookbook_mlx/flash
```

All 17 `docs/COOKBOOK.md` examples run end-to-end on both variants. The verifier
checks each task against its input rather than only asserting the output is
finite — content edits are transcribed, pitch and volume are measured in cents
and dB, separation is checked for the expected drop in active frames.

| | Flash, 4 steps | base, 32 steps | base, 32 steps, 8-bit |
| --- | --- | --- | --- |
| total wall clock | 237 s | 776 s | not measured separately |
| verifier checks passed | 15/17 | 16/17 | 16/17 |

The 8-bit total is left unmeasured rather than estimated: quantization does not
change wall clock here, which the RTF table below shows directly (6.05 vs 6.02
against fp32), so the cookbook total carries no information the RTF does not.

base at 32 steps tracks the requested edit more closely where it is measurable:
volume $+9.7$ dB against Flash's $+7.4$ (target $+10$), the whisper HF/LF ratio
opens two orders of magnitude instead of one, and the de-accented Mandarin
transcribes more cleanly.

The one case that fails on both is 2.2 lyric editing ("rear view" → "like you"):
Flash alters the phrase into something else, base leaves it unchanged. The torch
reference, run at the same 32 steps and seed, also fails to make the edit, so
this is a model/instruction limit rather than a port defect.

## Quantization

MLX quantizes `nn.Linear` and `nn.Embedding`, but not `nn.Conv1d`. That maps well
onto this model: the DiT is 99.4% Linear and the Thinker 99.8% Linear+Embedding,
so both shrink; the VAE is 99.9% Conv1d and stays fp32, which costs little
because it is only 147M parameters.

```bash
# quantize at load time
python -m auk_mlx.cli --bits 8 --instruction "..." --audio ref.wav -o out.wav

# or write quantized weights once, so the download shrinks too
python -m auk_mlx.convert quantize ckpts/mlx --bits 8 --variant base
```

Measured on the zero-shot TTS example, base at 32 steps:

| | peak memory | DiT on disk | Thinker on disk | RTF |
| --- | --- | --- | --- | --- |
| fp32 | 24.3 GB | 6.12 GB | 14.9 GB | 6.05 |
| 8-bit | 9.1 GB | 1.75 GB | 4.21 GB | 6.02 |
| 4-bit | 6.5 GB | — | — | 6.06 |

**Quantization buys memory, not speed.** Wall clock is flat across all three
because this workload is compute-bound in the ODE, not bandwidth-bound; the win
is that 8-bit fits comfortably in a 16 GB machine where fp32 does not.

**Use 8-bit; avoid 4-bit.** At 8 bits the audio is indistinguishable from fp32 —
across the cookbook tasks, mean waveform correlation 0.989 and spectral cosine
1.0000, with identical ASR transcripts. At 4 bits English still transcribes
correctly, but Chinese pronunciation degrades: the de-accent example turns
"論文…導師" (*thesis… advisor*) into "任務…倒死" (*task… nonsense*). The single-step
DiT error tells the same story — 8-bit keeps correlation at 0.9998 against fp32,
4-bit drops to 0.953.

Pre-quantized weights load into a quantized skeleton and reproduce the
quantize-at-load-time output to a correlation of 0.99999999.

## Sequential mode (the MLX analogue of `--cpu_offload`)

The three stacks are used in sequence and never simultaneously: the Thinker
produces a `[1, N, 2048]` conditioning tensor and nothing downstream reads its
3.7B weights again. `--sequential` builds each stack when it is needed and drops
it immediately after, so only one large model is resident at a time.

On CUDA, `--cpu_offload` moves weights to host RAM and back. MLX has unified
memory, so there is no transfer to pay for — freeing is a deallocation, and the
next stack is read straight from disk.

```bash
python -m auk_mlx.cli --bits 8 --sequential --instruction "..." --audio ref.wav -o out.wav
```

Peak memory for base at 32 steps, 6 s of audio:

| | resident | sequential |
| --- | --- | --- |
| fp32 | 24.3 GB | 15.4 GB |
| 8-bit | 9.1 GB | 6.1 GB |

Output is **bit-identical** either way (max abs diff 0.00e+00, correlation
1.00000000), and wall clock is unchanged — the weights are read once per call in
both modes. 8-bit plus `--sequential` brings the whole pipeline to 6.1 GB, which
fits a 16 GB Mac with room to spare.

The trade-off is that sequential mode re-reads weights from disk on every call,
so it suits one-shot runs and loses to resident mode for batch work.

## Accuracy

`python tests/test_parity.py` compares every stage against PyTorch on identical
inputs. Measured relative error:

| Stage | rel. error |
| --- | --- |
| conv / activation / resample primitives | ~1e-7 |
| VAE encode / decode | 7e-7 / 8e-5 |
| DiT forward, with and without CFG | ~5e-6 |
| 16-step CFG Euler trajectory | 2.4e-5 |
| Fused conditioning, text and text+audio | ~5e-6 |

End-to-end, Whisper transcribes the MLX output as the exact requested sentence,
in the reference speaker's voice, for English and Chinese alike.

## Speed (M4 Pro, 4 s of audio)

| Configuration | wall clock | RTF |
| --- | --- | --- |
| MLX, AuK-Flash, 4 steps | ~2–7 s | 0.6–1.8 |
| MLX, base, 16 steps | ~8–16 s | 2.0–4.0 |
| MLX, base, 32 steps | ~17–29 s | 4.2–7.3 |

These spans are wide because the machine is not quiet: repeated runs of the same
configuration on the same input vary by more than 2x, with no competing load
visible in `top`. The spans are min-to-max over 8 runs after warm-up; the low end
is the number to trust, since contention can only add time.

Stage profile for a 4-step Flash run: VAE decode ~60 %, ODE ~32 %, Qwen encode
~6 %. The vocoder, not the diffusion backbone, is the thing worth optimising
next — which is why Flash's 4× fewer steps buy well under 4× end-to-end.

## Things that bite

Four bugs cost real debugging time; each is a trap for anyone porting this.

**`inv_freq` must come from the checkpoint.** AuK stores a bf16-rounded copy of
the 10000-base rotary schedule (0.75 where the formula gives 0.74989). Computing
it analytically looks obviously right and moves the DiT output by 5e-3 — small
per layer, compounding over 30 blocks. This was the single largest error source.

**`conv_pre` in the VAE decoder is non-causal.** Everything around it is causal.
Getting it wrong shifts the decoder by 6 samples; the audio still sounds like
speech, so it will not announce itself.

**Two different rotary conventions in one pipeline.** The AuK DiT uses the
interleaved convention (`mx.fast.rope(traditional=True)`); HF's Qwen2 uses the
half-split convention. Using either one everywhere silently degrades output.

**The Qwen audio tower is windowed, not a plain Whisper stack.** Mel frames are
chunked into `2 * n_window` blocks, each convolved and positionally embedded on
its own, attention is block-diagonal across chunks, and a stride-2 avg-pool
follows the layers. A straight Whisper encoder produces the wrong token count
and the wrong embeddings.

Two smaller ones: `torch.nn.RMSNorm(eps=None)` resolves to `finfo(float32).eps`
(1.19e-7), not 1e-5; and `layer_weights` / `layer_scale` differ between base and
Flash, so the fusion file is written per variant.

## MLX weight layouts

Conversion transposes convolution weights; everything else carries over as-is.

| Layer | torch | MLX |
| --- | --- | --- |
| `Conv1d` | `(O, I, K)` | `(O, K, I)` — `permute(0, 2, 1)` |
| `ConvTranspose1d` | `(I, O, K)` | `(O, K, I)` — `permute(1, 2, 0)` |
| depthwise `ConvTranspose1d` | `(C, 1, K)` | `(C, K, 1)` — `permute(0, 2, 1)` |

`nn.Linear` needs no transpose. MLX has no `kaiser_window`, so the alias-free
resampler builds it from an I0 power series (matches torch to 3e-8).
