# Split deployment

Run AuK as two nodes plus a thin orchestrator, instead of one process holding every weight on a
single card.

| Component | Holds | Resident (bf16) | Runs |
|---|---|---|---|
| **Text-encoder node** | Qwen2.5-Omni Thinker + `layer_weights` / `layer_scale` | ~8.1 GB | once per generation |
| **Worker node** | BigVGAN VAE + AuK DiT + the whole ODE loop | ~3.4 GB | every ODE step |
| **Orchestrator** | `Qwen2_5OmniProcessor` only (no GPU) | — | — |

One generation costs **two RPCs**: the orchestrator tokenizes locally, asks the text-encoder node
for the fused embedding, then hands everything to the worker, which returns the waveform.

## Why these two cuts

The pipeline is strictly sequential — `VAE encode → Qwen → DiT ODE → VAE decode` — so almost any
split is technically possible. Two pairs are *not*:

- **Qwen and the layer fusion must stay together.** The fusion is an ELMo-style weighted average
  over all 36 hidden layers. Fusing server-side turns ~117 MB of hidden states into ~3 MB.
- **The DiT and the ODE loop must stay together.** `torchdiffeq.odeint` calls the DiT once per
  step. Splitting them would turn one call into 32 round trips.

The VAE *could* be a third node, but it is only 0.6 GB and splitting it would push the 960 KB
waveform across the wire twice. Only do it if you want N workers sharing one VAE.

## Running it

```bash
conda activate auk

# node A — small card
python -m auk.serve text-encoder \
    --qwen_path ckpts/Qwen2.5-Omni-3B \
    --ckpt ckpts/AuK/auk_base.safetensors \
    --device cuda:1 --port 8001

# node B — big card
python -m auk.serve worker \
    --ckpt ckpts/AuK/auk_base.safetensors \
    --device cuda:0 --port 8002
```

Then either the CLI or the web UI, both of which need no GPU of their own:

```bash
python -m auk.infer.infer_cli \
    --text_encoder_url http://hostA:8001 --worker_url http://hostB:8002 \
    --instruction "Replace 'rear view' with 'like you' in the lyrics" \
    --audio ref.wav -o out.wav

python -m auk.infer.infer_gradio \
    --text_encoder_url http://hostA:8001 --worker_url http://hostB:8002
```

Notes:

- The text-encoder node needs `--ckpt` **only** to read `layer_weights` / `layer_scale`. It uses
  `safe_open`, so the 6 GB export is never fully read.
- Both nodes need the AuK `config.yaml` (the worker finds it next to `--ckpt`).
- One text-encoder node can serve many workers, and both AuK and AuK-Flash at once.

## Options

| Flag | Node | Meaning |
|---|---|---|
| `--device` | both | `cuda:N` / `cpu` / `mps` |
| `--weight_dtype` | both | `fp32` (default, historical) or `bf16` (halves resident VRAM) |
| `--device_map` | worker | still spreads VAE/DiT across that machine's GPUs, e.g. `auto` |
| `--dtype` | worker | autocast dtype for sampling |
| `--timeout` | client | per-request timeout in seconds |

`--device_map` composes with the split: a worker can put the DiT on `cuda:0` and the VAE on
`cuda:1` of its own box while Qwen lives on another machine entirely.

## Wire format

`b"AUK1" | uint32 header_len | JSON header | safetensors payload`

Tensors travel as safetensors rather than pickled torch objects — it is already a dependency, it
is not executable, and it preserves dtype. Scalars ride in the JSON header. Masks are sent as
`uint8` for portability. Transport is plain HTTP/1.1 with keep-alive; at a few MB per request it
does not justify gRPC's codegen and dependencies.

## Operational notes

- **Concurrency.** Each node serialises inference behind a lock. `Flux2Edit.text_cond` /
  `text_uncond` are module-level state reused through `cache=True`, so two overlapping CFG samples
  would silently reuse each other's text projection. Scale by running more worker processes, not
  more threads.
- **No auth or TLS.** Keep the nodes on a private network.
- **Resampling.** The orchestrator sends the reference clip at its *original* sample rate and the
  worker resamples; if the client claimed the target rate instead, the duration and pitch would
  both be wrong.
- **Restart after editing.** The nodes are long-lived processes; nothing is hot-reloaded.

## Troubleshooting

**`RuntimeError: Generated latent contains NaN/Inf.`**

The text-encoder node runs Qwen in bf16 by default (`--weight_dtype`), so the embedding it returns
is bf16, while the DiT holds fp32 weights. `sample_from_embeds` casts the incoming embedding to
the DiT's dtype before use, which is what makes this safe — if you see this error, the worker is
running a build without that cast. Restart it.

To confirm where a NaN comes from:

```python
h, m = RemoteTextEncoder("http://host:8001").encode(cond_inputs)
print(h.dtype, torch.isfinite(h).all(), h.abs().max())   # embeddings sane?
```

If the embedding is finite but generation still fails, the cause is downstream (ODE / VAE), not
the encoder. Note that an *all-zero* embedding is a false alarm: it makes the text mask all-`False`,
which NaNs the attention softmax by construction.

## Tests

```bash
python tests/test_serve_split.py
```

Covers the codec (round trip, malformed frames), `CFMEdit` without a text encoder and
`sample_from_embeds`, the local `sample()` path as a regression guard, the latent-length planner,
and the HTTP nodes end to end. It runs on CPU with tiny random weights — it validates plumbing,
not audio quality.

### Checking the split output against local inference

The tests cannot prove the two paths produce the same audio. To do that, run the same prompt
through both and compare waveforms:

```bash
# split
python -m auk.infer.infer_cli --text_encoder_url http://host:8001 --worker_url http://host:8002 \
    --instruction "..." --audio ref.wav --seed 0 -o out_split.wav

# local (single process)
python -m auk.infer.infer_cli --instruction "..." --audio ref.wav --seed 0 -o out_local.wav
```

Then compare:

```python
import torchaudio
a, _ = torchaudio.load("out_local.wav")
b, _ = torchaudio.load("out_split.wav")
n = min(a.shape[-1], b.shape[-1])
err = (a[..., :n] - b[..., :n]).abs().max()
print("max abs diff:", float(err))
```

Expect small-but-nonzero differences: the text-encoder node runs Qwen in bf16 by default while the
local path keeps fp32 weights, so the fused embedding is not bit-identical. A large difference or
a difference in duration means a real bug.
