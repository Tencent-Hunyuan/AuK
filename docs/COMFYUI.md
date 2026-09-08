# ComfyUI-AuK

ComfyUI-AuK exposes the existing AuK inference and Prompt Enhancer paths as two
ComfyUI V3 nodes using the stable `comfy_api.v0_0_2` interface:

- **AuK Model Loader** loads AuK or AuK-Flash and returns an `AUK_ENGINE`.
- **AuK Generate / Edit** runs instruction TTS, zero-shot TTS, or speech
  editing and returns standard ComfyUI `AUDIO`.

The generated waveform has shape `[B, C, T]` and can be connected directly to
**Preview Audio** or **Save Audio (Advanced)**.

## Installation

Use an existing dedicated ComfyUI environment. Install the PyTorch/CUDA build
required by your server first, then install AuK and its ComfyUI dependencies
through the repository's single optional dependency group:

```bash
cd /path/to/ComfyUI
source .venv/bin/activate

cd /path/to/AuK
pip install -e ".[comfyui]"

test ! -e /path/to/ComfyUI/custom_nodes/ComfyUI-AuK
ln -s /path/to/AuK/comfyui/ComfyUI-AuK \
  /path/to/ComfyUI/custom_nodes/ComfyUI-AuK
```

Do not install a second copy of this node under `custom_nodes`; duplicate copies
can register the same node IDs twice. The package has not been published to
ComfyUI Manager or the Registry.

## Weights

Download the AuK, optional AuK-Flash, and Qwen2.5-Omni-3B files as described in
the root [README](../README.md#download-the-weights). The default node values
expect:

```text
ckpts/
├── AuK/
│   ├── auk_base.safetensors
│   ├── config.yaml
│   └── vae.safetensors
├── AuK-Flash/
│   ├── auk_flash.safetensors
│   ├── config.yaml
│   └── vae.safetensors
└── Qwen2.5-Omni-3B/
```

Relative loader paths resolve against `AUK_HOME`, then against the installed
AuK checkout. Set it before starting ComfyUI when the weights are stored with a
different checkout:

```bash
export AUK_HOME=/path/to/AuK
```

You can also enter absolute checkpoint, config, and Qwen paths in the loader.
The VAE is selected from the checkpoint directory by the existing AuK loader.

## Start ComfyUI

Start ComfyUI from its own directory:

```bash
cd /path/to/ComfyUI
source .venv/bin/activate
python main.py --listen 127.0.0.1 --port 8188
```

`8188` is ComfyUI's default port. Bind to `0.0.0.0` only after confirming the
platform access controls; open the actual forwarded address or
`http://127.0.0.1:PORT`, not `0.0.0.0`.

## Workflows

Import the single reusable UI workflow:

- [`auk.json`](../comfyui/workflows/auk.json)

This JSON file is a ComfyUI canvas export, not a Python dependency. It stores
the nodes, links, widget values, and layout needed for one-click browser import.
The Python nodes still work if it is deleted.

The workflow leaves **Load Audio** disconnected by default:

- For instruction TTS, leave it disconnected.
- For zero-shot TTS, connect a reference recording to **input audio**.
- For editing, enhancement, and separation, connect the audio to process.

Change the instruction and target duration for the selected task. Separate
workflow or API-prompt JSON copies are intentionally not kept because the same
two AuK nodes execute every task.

## Generate and edit

For instruction TTS, leave **input audio** unconnected. For zero-shot TTS,
connect a voice reference. For editing, connect the source recording. The task
is carried by the instruction, using the same message format and templates as
the AuK CLI, Python API, and [Cookbook](COOKBOOK.md). The node accepts
one audio sample per execution and averages multichannel input to mono.

The loader caches one engine for each resolved checkpoint, config, Qwen path,
device, and dtype combination. Repeated generations with the same loader
settings reuse it. Models remain resident on the selected device; there is no
automatic ComfyUI VRAM offload. Restart ComfyUI to release them. Loading another
combination can keep another complete engine in memory.

Base AuK exposes NFE, CFG, and sway controls. AuK-Flash uses its fixed 4-step,
CFG-off recipe; set NFE to `4`, CFG to `0`, and sway to `-1`. The node rejects
other values instead of silently ignoring them.

## Prompt Enhancer

**Enable Prompt Enhancer** defaults to enabled, matching the Gradio demo. It
uses the same classification, ASR, rewrite, duration, audio preprocessing, and
error paths. The node also returns the model instruction and a compact PE
summary. Disabling it skips PE construction and network/ASR calls; enter an
explicit positive duration.

Load credentials only into the server process:

```bash
set -a
source /path/to/AuK/.env
set +a
python main.py --listen 127.0.0.1 --port 8188
```

Keys are not node inputs, workflow values, log fields, or repository files.
Audio preprocessing uses temporary files only when PE requires the existing
file-based path, and removes them after generation.

## 30-second limit

AuK was trained with audio targets up to 30 seconds. During inference, source or
reference latents are prepended to generated target latents in the model
sequence, so this integration enforces one shared 30-second budget after
resampling and latent-frame rounding:

```text
source/reference duration + generated target duration <= 30 seconds
```

Instruction TTS has no audio prefix, so its target can use the full budget.
Audio-backed tasks have less target time available. The node checks both the
actual connected audio and the manual or PE-derived target duration before
inference. It raises a clear error when the sequence is too long; it never
truncates or automatically splits and joins long audio.
