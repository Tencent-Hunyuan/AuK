from __future__ import annotations

import logging
import math
import os
import tempfile
import threading
from pathlib import Path

import torch
import torchaudio
from comfy_api.v0_0_2 import ComfyExtension, io
from omegaconf import OmegaConf

import auk
from auk.infer.infer_auk import AukInfer


logger = logging.getLogger("ComfyUI-AuK")

MAX_SEQUENCE_SECONDS = 30.0
QWEN_AUDIO_SAMPLE_RATE = 16_000
AUK_ENGINE = io.Custom("AUK_ENGINE")


class AuKEngine:
    def __init__(self, inference: AukInfer):
        self.inference = inference
        self.lock = threading.Lock()


ENGINE_CACHE: dict[tuple[str, str, str, str, str, bool], AuKEngine] = {}
ENGINE_CACHE_LOCK = threading.Lock()


def resolve_path(path: str) -> Path:
    expanded = Path(os.path.expandvars(path)).expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    auk_home = os.environ.get("AUK_HOME")
    base = Path(auk_home).expanduser() if auk_home else Path(auk.__file__).resolve().parents[2]
    return (base / expanded).resolve()


def normalize_audio(audio: dict | None) -> tuple[torch.Tensor, int] | None:
    if audio is None:
        return None
    if not isinstance(audio, dict) or "waveform" not in audio or "sample_rate" not in audio:
        raise ValueError("input_audio must be a ComfyUI AUDIO value with waveform and sample_rate.")

    waveform = audio["waveform"]
    sample_rate = audio["sample_rate"]
    if not torch.is_tensor(waveform) or waveform.ndim != 3:
        shape = tuple(waveform.shape) if torch.is_tensor(waveform) else type(waveform).__name__
        raise ValueError(f"input_audio waveform must have shape [B, C, T], got {shape}.")
    if waveform.shape[0] != 1:
        raise ValueError(f"AuK accepts one audio sample per run, got batch size {waveform.shape[0]}.")
    if waveform.shape[1] < 1 or waveform.shape[2] < 1:
        raise ValueError(f"input_audio waveform is empty: {tuple(waveform.shape)}.")
    if not isinstance(sample_rate, int) or sample_rate <= 0:
        raise ValueError(f"input_audio sample_rate must be a positive integer, got {sample_rate!r}.")

    waveform = waveform[0].detach().to(device="cpu", dtype=torch.float32)
    if not torch.isfinite(waveform).all():
        raise ValueError("input_audio waveform contains NaN or Inf.")
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform.contiguous(), sample_rate


def prepare_model_audio(
    audio: tuple[torch.Tensor, int] | None,
    target_sample_rate: int,
) -> tuple[tuple[torch.Tensor, int] | None, object | None]:
    if audio is None:
        return None, None
    waveform, sample_rate = audio
    model_waveform = waveform
    if sample_rate != target_sample_rate:
        model_waveform = torchaudio.functional.resample(waveform, sample_rate, target_sample_rate)
    qwen_waveform = model_waveform
    if target_sample_rate != QWEN_AUDIO_SAMPLE_RATE:
        qwen_waveform = torchaudio.functional.resample(model_waveform, target_sample_rate, QWEN_AUDIO_SAMPLE_RATE)
    return (model_waveform.contiguous(), target_sample_rate), qwen_waveform.squeeze(0).contiguous().numpy()


def validate_sequence_duration(engine: AuKEngine, source_audio: tuple[torch.Tensor, int] | None, target_seconds: float) -> None:
    if not math.isfinite(target_seconds) or target_seconds <= 0:
        raise ValueError(f"Target duration must be a finite value greater than 0 seconds, got {target_seconds!r}.")
    sample_rate = engine.inference.target_sample_rate
    downsample_rate = engine.inference.downsample_rate
    source_frames = source_audio[0].shape[-1] // downsample_rate if source_audio is not None else 0
    target_frames = max(1, math.ceil(target_seconds * sample_rate / downsample_rate))
    max_frames = int(MAX_SEQUENCE_SECONDS * sample_rate / downsample_rate)
    if source_frames + target_frames > max_frames:
        source_seconds = source_frames * downsample_rate / sample_rate
        applied_target_seconds = target_frames * downsample_rate / sample_rate
        raise ValueError(
            f"AuK's source/reference plus generated target sequence is "
            f"{source_seconds + applied_target_seconds:.2f}s "
            f"({source_seconds:.2f}s + {applied_target_seconds:.2f}s after model-frame rounding), "
            f"exceeding the {MAX_SEQUENCE_SECONDS:.0f}s limit."
        )


class AuKModelLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        devices = [f"cuda:{index}" for index in range(torch.cuda.device_count())]
        devices.append("cpu")
        default_device = devices[0]
        default_dtype = "bf16" if default_device.startswith("cuda") else "fp32"
        return io.Schema(
            node_id="AuKModelLoader",
            display_name="AuK Model Loader",
            category="AuK",
            description=(
                "Load AuK or AuK-Flash and reuse the same engine for identical settings. "
                "Engines remain cached until ComfyUI exits; changing settings can load another copy."
            ),
            inputs=[
                io.String.Input(
                    "checkpoint_path",
                    default="ckpts/AuK/auk_base.safetensors",
                    tooltip="AuK or AuK-Flash checkpoint. Relative paths use AUK_HOME, then the installed AuK checkout.",
                ),
                io.String.Input(
                    "config_path",
                    default="",
                    tooltip="Optional config.yaml override. Empty uses config.yaml beside the checkpoint.",
                ),
                io.String.Input(
                    "qwen_path",
                    default="ckpts/Qwen2.5-Omni-3B",
                    tooltip="Qwen2.5-Omni-3B directory. Relative paths use AUK_HOME, then the installed AuK checkout.",
                ),
                io.Combo.Input(
                    "device",
                    options=devices,
                    default=default_device,
                    tooltip="Visible CUDA device or CPU.",
                ),
                io.Combo.Input(
                    "dtype",
                    options=["bf16", "fp16", "fp32"],
                    default=default_dtype,
                    tooltip="Inference autocast dtype. The Qwen encoder is loaded in bf16 by AuK.",
                ),
                io.Boolean.Input(
                    "cpu_offload",
                    default=False,
                    optional=True,
                    tooltip=(
                        "Offload the Qwen text encoder and DiT to CPU when idle to reduce VRAM use. "
                        "The VAE remains on CUDA (slower inference)."
                    ),
                ),
            ],
            outputs=[AUK_ENGINE.Output("engine", display_name="engine")],
        )

    @classmethod
    def execute(
        cls,
        checkpoint_path: str,
        config_path: str,
        qwen_path: str,
        device: str,
        dtype: str,
        cpu_offload: bool = False,
    ) -> io.NodeOutput:
        checkpoint = resolve_path(checkpoint_path.strip())
        if not checkpoint.is_file():
            raise ValueError(f"AuK checkpoint not found: {checkpoint}")
        vae = checkpoint.parent / "vae.safetensors"
        if not vae.is_file():
            raise ValueError(f"AuK VAE not found next to the checkpoint: {vae}")

        config = resolve_path(config_path.strip()) if config_path.strip() else checkpoint.parent / "config.yaml"
        config = config.resolve()
        if not config.is_file():
            raise ValueError(f"AuK config not found: {config}")

        model_config = OmegaConf.load(config)
        model_name = str(model_config.model.get("name", ""))
        if model_name not in {"AuK", "AuK-Flash"}:
            raise ValueError(f"Unsupported AuK model name in {config}: {model_name!r}")

        qwen_setting = qwen_path.strip() or str(model_config.model.text_encoder.text_encoder_path)
        qwen = resolve_path(qwen_setting)
        if not qwen.is_dir():
            raise ValueError(f"Qwen2.5-Omni-3B directory not found: {qwen}")

        device = device.strip()
        if device != "cpu" and device != "cuda" and not (device.startswith("cuda:") and device.removeprefix("cuda:").isdigit()):
            raise ValueError(f"Unsupported device {device!r}; use cpu, cuda, or cuda:N.")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise ValueError(f"CUDA device requested but CUDA is unavailable: {device}")
        if device.startswith("cuda:") and int(device.removeprefix("cuda:")) >= torch.cuda.device_count():
            raise ValueError(f"CUDA device is out of range: {device}; visible device count is {torch.cuda.device_count()}.")
        if dtype not in {"bf16", "fp16", "fp32"}:
            raise ValueError(f"Unsupported dtype {dtype!r}; use bf16, fp16, or fp32.")
        if device == "cpu" and dtype != "fp32":
            raise ValueError("CPU inference requires dtype=fp32; fp16/bf16 autocast is only applied on CUDA.")
        if device.startswith("cuda") and dtype == "bf16" and not torch.cuda.is_bf16_supported():
            raise ValueError(f"The selected CUDA device does not support bf16: {device}.")

        cache_key = (str(checkpoint), str(config), str(qwen), device, dtype, cpu_offload)
        with ENGINE_CACHE_LOCK:
            engine = ENGINE_CACHE.get(cache_key)
            if engine is None:
                logger.info("Loading %s from %s on %s (%s)", model_name, checkpoint, device, dtype)
                engine = AuKEngine(
                    inference=AukInfer(
                        config_path=str(config),
                        ckpt_path=str(checkpoint),
                        device=device,
                        dtype=dtype,
                        qwen_path=str(qwen),
                        cpu_offload=cpu_offload,
                    )
                )
                ENGINE_CACHE[cache_key] = engine
        return io.NodeOutput(engine)


class AuKGenerateEdit(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="AuKGenerateEdit",
            display_name="AuK Generate / Edit",
            category="AuK",
            description=(
                "Run instruction TTS, zero-shot TTS, or speech editing. "
                "The source/reference and generated target share one 30-second sequence budget."
            ),
            inputs=[
                AUK_ENGINE.Input("engine"),
                io.String.Input(
                    "instruction",
                    multiline=True,
                    default=(
                        'Generate speech based on the following description: "a calm, warm female voice". '
                        'The content to speak is: "Hello, welcome to AuK.".'
                    ),
                    tooltip="Natural-language AuK request. Prompt Enhancer can convert a free-form request into the model instruction.",
                ),
                io.Float.Input(
                    "generation_seconds",
                    default=0.0,
                    min=0.0,
                    max=MAX_SEQUENCE_SECONDS,
                    step=0.1,
                    display_mode=io.NumberDisplay.slider,
                    tooltip=(
                        "Generated target duration in seconds. 0 lets Prompt Enhancer estimate it. "
                        "When Prompt Enhancer is disabled, enter a value above 0."
                    ),
                ),
                io.Boolean.Input(
                    "use_prompt_enhancer",
                    display_name="Enable Prompt Enhancer",
                    default=True,
                    label_on="enabled",
                    label_off="disabled",
                    tooltip="Uses the same PE preparation path as the AuK Gradio demo. Credentials are read from server environment variables.",
                ),
                io.Int.Input(
                    "seed",
                    default=42,
                    min=0,
                    max=0x7FFFFFFFFFFFFFFF,
                    control_after_generate=io.ControlAfterGenerate.fixed,
                    tooltip="Seeds both reference VAE encoding and target sampling.",
                ),
                io.Audio.Input(
                    "input_audio",
                    optional=True,
                    tooltip=(
                        "Reference audio for zero-shot TTS or source audio for editing. "
                        "Leave unconnected for instruction TTS. Batch size must be 1; channels are averaged to mono."
                    ),
                ),
                io.Int.Input(
                    "nfe_steps",
                    default=32,
                    min=4,
                    max=64,
                    step=1,
                    advanced=True,
                    tooltip="Base AuK sampling steps. AuK-Flash requires 4.",
                ),
                io.Float.Input(
                    "cfg_strength",
                    default=2.0,
                    min=0.0,
                    max=5.0,
                    step=0.1,
                    advanced=True,
                    tooltip="Base AuK classifier-free guidance. AuK-Flash requires 0.",
                ),
                io.Float.Input(
                    "sway_sampling_coef",
                    default=-1.0,
                    min=-1.0,
                    max=1.0,
                    step=0.1,
                    advanced=True,
                    tooltip="Base AuK sway sampling coefficient. AuK-Flash requires the default -1 placeholder.",
                ),
            ],
            outputs=[
                io.Audio.Output("generated_audio", display_name="generated audio"),
                io.String.Output("enhanced_instruction", display_name="model instruction"),
                io.String.Output("prompt_enhancer_info", display_name="Prompt Enhancer info"),
            ],
        )

    @classmethod
    def execute(
        cls,
        engine: AuKEngine,
        instruction: str,
        generation_seconds: float,
        use_prompt_enhancer: bool,
        seed: int,
        input_audio: dict | None = None,
        nfe_steps: int = 32,
        cfg_strength: float = 2.0,
        sway_sampling_coef: float = -1.0,
    ) -> io.NodeOutput:
        instruction = instruction.strip()
        if not instruction:
            raise ValueError("AuK instruction is empty.")
        if not math.isfinite(float(generation_seconds)) or not 0.0 <= float(generation_seconds) <= MAX_SEQUENCE_SECONDS:
            raise ValueError(f"generation_seconds must be between 0 and {MAX_SEQUENCE_SECONDS:.0f}, got {generation_seconds!r}.")
        if engine.inference.is_flash and (int(nfe_steps) != 4 or float(cfg_strength) != 0.0 or float(sway_sampling_coef) != -1.0):
            raise ValueError(
                "AuK-Flash requires NFE steps=4, CFG strength=0, and sway=-1; these controls are fixed by its recipe."
            )

        audio = normalize_audio(input_audio)
        prepared = None
        bridge_path = None
        final_instruction = instruction
        prompt_enhancer_info = "Prompt Enhancer: disabled"
        target_seconds = float(generation_seconds)

        try:
            if use_prompt_enhancer:
                if audio is not None:
                    waveform, sample_rate = audio
                    descriptor, bridge_path = tempfile.mkstemp(prefix="auk_comfy_pe_", suffix=".wav")
                    os.close(descriptor)
                    torchaudio.save(
                        bridge_path,
                        waveform,
                        sample_rate,
                        encoding="PCM_F",
                        bits_per_sample=32,
                    )

                from auk.infer.pe import PromptEnhancer, PromptEnhancerError

                try:
                    prepared = PromptEnhancer().prepare(
                        instruction,
                        bridge_path,
                        target_duration=target_seconds if target_seconds > 0 else None,
                    )
                except (PromptEnhancerError, ValueError, FileNotFoundError) as error:
                    raise ValueError(f"Prompt Enhancer failed: {type(error).__name__}: {error}") from error
                final_instruction = prepared.instruction
                target_seconds = target_seconds if target_seconds > 0 else prepared.gen_seconds
                if prepared.audio:
                    prepared_waveform, prepared_sample_rate = torchaudio.load(prepared.audio)
                    audio = normalize_audio({"waveform": prepared_waveform.unsqueeze(0), "sample_rate": prepared_sample_rate})
                else:
                    audio = None

                task = prepared.task_type
                if prepared.operation_subtype:
                    task = f"{task} / {prepared.operation_subtype}"
                asr_text = prepared.asr.text if prepared.asr and prepared.asr.text else ""
                prompt_enhancer_info = f"Task: {task}\nTarget Duration: {target_seconds:.2f} s\nASR Content: {asr_text}"
            elif target_seconds <= 0:
                raise ValueError("Duration must be greater than 0 when Prompt Enhancer is disabled.")

            model_audio, qwen_audio = prepare_model_audio(audio, engine.inference.target_sample_rate)
            validate_sequence_duration(engine, model_audio, target_seconds)

            content = [{"type": "text", "text": final_instruction}]
            if qwen_audio is not None:
                content.append({"type": "audio", "audio": qwen_audio})
            messages = [{"role": "user", "content": content}]

            with engine.lock:
                torch.manual_seed(int(seed))
                generated_waveform, sample_rate = engine.inference.generate(
                    messages,
                    audio=model_audio,
                    gen_seconds=target_seconds,
                    nfe=int(nfe_steps),
                    cfg_strength=float(cfg_strength),
                    sway_sampling_coef=float(sway_sampling_coef),
                    seed=int(seed),
                )
        finally:
            if prepared is not None:
                prepared.cleanup()
            if bridge_path is not None:
                try:
                    os.remove(bridge_path)
                except FileNotFoundError:
                    pass

        generated_waveform = generated_waveform.detach().to(device="cpu", dtype=torch.float32)
        if generated_waveform.ndim == 1:
            generated_waveform = generated_waveform.unsqueeze(0)
        if generated_waveform.ndim != 2 or generated_waveform.shape[-1] == 0:
            raise RuntimeError(f"AuK returned invalid audio shape: {tuple(generated_waveform.shape)}.")
        if not torch.isfinite(generated_waveform).all():
            raise RuntimeError("AuK returned audio containing NaN or Inf.")

        audio_output = {
            "waveform": generated_waveform.unsqueeze(0),
            "sample_rate": int(sample_rate),
        }
        return io.NodeOutput(audio_output, final_instruction, prompt_enhancer_info)


class AuKExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [AuKModelLoader, AuKGenerateEdit]


async def comfy_entrypoint() -> AuKExtension:
    return AuKExtension()
