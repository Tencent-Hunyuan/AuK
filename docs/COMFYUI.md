# AuK in ComfyUI

AuK and AuK-Flash use the same models, Prompt Enhancer (PE), and task
instructions as the [Gradio demo](../README.md#interactive-gradio-demo).
Follow the README for [installation](../README.md#installation),
[weights](../README.md#download-the-weights), and PE configuration; use the
[Cookbook](COOKBOOK.md) for task examples.

## Connect to ComfyUI

Install `.[comfyui]` in the Python environment that runs ComfyUI, then link
the nodes and start ComfyUI from that environment:

```bash
ln -s /path/to/AuK/comfyui/ComfyUI-AuK \
  /path/to/ComfyUI/custom_nodes/ComfyUI-AuK

export AUK_HOME=/path/to/AuK
cd /path/to/ComfyUI
python main.py --listen 127.0.0.1 --port 8188
```

Keep one node-package copy under `custom_nodes`. To enable PE, load the same
AuK `.env` used by Gradio into this shell **before starting ComfyUI**; the nodes
do not load it automatically. Open the local address or the server's forwarded
address, using its configured port.

## Run a workflow

Open [`auk.json`](../comfyui/workflows/auk.json), check the paths and device in
**AuK Model Loader**, and click **Run**. The included example uses **AuK Base**,
**PE disabled**, and a **3-second** target. Its output connects to **Preview
Audio** and **Save Audio (Advanced)**, which saves FLAC files with the
`auk/output` prefix under ComfyUI's output directory.

In **AuK Generate / Edit**, enter an instruction from the Cookbook:

- **Instruction TTS:** leave `input_audio` disconnected.
- **Zero-shot TTS:** upload a reference voice with **Load Audio** and connect it
  to `input_audio`.
- **Editing, enhancement, or separation:** connect the audio to process.

Replace the **Load Audio** placeholder with an uploaded file before using it.
With PE enabled, `generation_seconds=0` estimates the target duration; a
positive value overrides the estimate. With PE disabled, supply a positive
value. Newly added Generate / Edit nodes default to PE enabled, unlike the
included workflow. Connect the two text outputs to **Preview as Text** to see
the final instruction and PE/ASR summary.

To use **AuK-Flash**, change `checkpoint_path` to
`ckpts/AuK-Flash/auk_flash.safetensors` and set `nfe_steps=4`,
`cfg_strength=0`, and `sway_sampling_coef=-1`. Leave `config_path` empty to
load the config beside the selected checkpoint. Relative model paths use
`AUK_HOME`, or the installed AuK checkout when it is unset; absolute paths
are also supported.

## ComfyUI-specific behavior

- **Duration:** source/reference + generated target must fit within **30
  seconds**, after PE preprocessing, resampling, and model-frame rounding.
  Long Cookbook examples may need shorter inputs; the node does not split
  audio automatically.
- **Memory:** models stay resident on the selected device and are reused.
  Switching model settings may load another engine; restart ComfyUI to release
  them. Automatic ComfyUI VRAM offload is not supported.
- **Audio:** each execution accepts one input sample and averages channels to
  mono. The node returns the generated waveform directly, without Gradio's
  extra loudness processing for lyric editing and vocal extraction; output
  levels may differ.
