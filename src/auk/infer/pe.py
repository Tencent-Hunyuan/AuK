from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import os
import random
import re
import tempfile
import threading
import time
import wave
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

import torch
import torchaudio
import yaml
from openai import OpenAI, OpenAIError
from tencentcloud.asr.v20190614 import asr_client, models as asr_models
from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile

_CONFIG_PATH = Path(__file__).with_name("pe.config.yaml")
_PE_CONFIG = yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8"))
_LLM_CONFIG = _PE_CONFIG["api"]["llm"]
_ASR_CONFIG = _PE_CONFIG["api"]["asr"]
_PROMPTS = _PE_CONFIG["prompts"]
_RUNTIME = _PE_CONFIG["runtime"]
_TASKS = _PE_CONFIG["tasks"]
_TEMPLATE_POOLS = _PE_CONFIG["template_pools"]
_EDITING_TEMPLATE_POOLS = _TEMPLATE_POOLS["editing"]
_RESTORATION_TEMPLATE_POOLS = _TEMPLATE_POOLS["restoration"]
_DURATION_CONFIG = _RUNTIME["duration"]
_F5_CONFIG = _DURATION_CONFIG["f5"]
_VAD_CONFIG = _RUNTIME["vad"]
_WHISPER_CONFIG = _RUNTIME["whisper"]

LLM_MAX_TOKENS = int(_LLM_CONFIG["max_tokens"])
LLM_TEMPERATURE = float(_LLM_CONFIG["temperature"])
LLM_TIMEOUT = int(_LLM_CONFIG["timeout_sec"])
ASR_ENDPOINT = str(_ASR_CONFIG["endpoint"])
ASR_ENGINE_MODEL_TYPE = str(_ASR_CONFIG["engine_model_type"])
ASR_SAMPLE_RATE = int(_ASR_CONFIG["sample_rate"])
ASR_MAX_DURATION_SEC = float(_ASR_CONFIG["max_duration_sec"])
ASR_MAX_DATA_BYTES = int(_ASR_CONFIG["max_data_bytes"])
ASR_TIMEOUT = int(_ASR_CONFIG["timeout_sec"])
ASR_POLL_INTERVAL_SEC = float(_ASR_CONFIG["poll_interval_sec"])
INSTRUCT_DURATION_SYSTEM_PROMPT = str(_PROMPTS["instruct_tts_duration"])

TTS_SEC_PER_UTF8_BYTE = {language: float(value) for language, value in _DURATION_CONFIG["seconds_per_utf8_byte"].items()}
F5_SHORT_TEXT_BYTE_THRESHOLD = int(_F5_CONFIG["short_text_byte_threshold"])
F5_SHORT_TEXT_SPEED = float(_F5_CONFIG["short_text_speed"])
F5_SAMPLE_RATE = int(_F5_CONFIG["sample_rate"])
F5_HOP_LENGTH = int(_F5_CONFIG["hop_length"])
MODEL_LATENT_FRAMES_PER_SECOND = int(_DURATION_CONFIG["output_frames_per_second"])
NO_VAD_TASK_TYPES = frozenset(_VAD_CONFIG["skip_tasks"])
WHISPER_TO_NORMAL = "to_normal"
WHISPER_TO_WHISPER = "to_whisper"
WHISPER_TARGET_LUFS = float(_WHISPER_CONFIG["target_lufs"])
WHISPER_TARGET_RMS = float(_WHISPER_CONFIG["target_rms"])
WHISPER_TO_NORMAL_TARGET_RMS = float(_WHISPER_CONFIG["to_normal_target_rms"])
WHISPER_PEAK_CEILING = float(_WHISPER_CONFIG["peak_ceiling"])
VAD_TRIM_PAD_SEC = float(_VAD_CONFIG["trim_padding_sec"])
VAD_NORM_RMS_THRESHOLD = float(_VAD_CONFIG["low_rms_threshold"])
VAD_NORM_TARGET_PEAK = float(_VAD_CONFIG["normalized_peak"])
INSTRUCT_DURATION_PROMPT_VERSION = str(_DURATION_CONFIG["instruct_tts_prompt_version"])

EMOTIONS = next(param["labels"] for param in _TASKS["emotion_edit"]["params"] if param["name"] == "emotion")
_NONVERBAL_SUBTYPES = {
    "delete": ("delete", "global", {}),
    "add_before": ("add", "anchor", {"side": {"zh": "前", "en": "before"}}),
    "add_after": ("add", "anchor", {"side": {"zh": "后", "en": "after"}}),
    "add_head": ("add", "positional", {"pos": {"zh": "开头", "en": "the beginning"}}),
    "add_tail": ("add", "positional", {"pos": {"zh": "结尾", "en": "the end"}}),
}

_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_EN_WORD_RE = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)*")
_DIGIT_SPACED_RUN_RE = re.compile(r"\d+(?:[ \t　]+\d+)+")
_MULTIGROUP_JOIN_SUFFIXES = frozenset("年月日号点分秒元块角分吨米公里厘米毫米千米次个岁届位名层楼度")
_ZH_TN_SPACE_RE = re.compile(r"(?<=[一-鿿])[ \t　]+(?=[一-鿿0-9])|(?<=[0-9])[ \t　]+(?=[一-鿿])")
_RECORDING_RESULT_PREFIX_RE = re.compile(r"^\s*\[\d+:\d+(?:\.\d+)?,\d+:\d+(?:\.\d+)?\]\s*")
_VOCAL_EDIT_MARKERS = (
    "lyric",
    "lyrics",
    "singing",
    "sung",
    "vocal recording",
    "歌词",
    "歌唱",
    "演唱",
    "唱的",
    "这首歌",
    "歌曲",
)

_TN_MODELS: dict[str, Any] = {}
_TN_LOCK = threading.Lock()
_VAD_MODEL = None
_VAD_LOCK = threading.Lock()


def _task_param(task_type: str, param_name: str) -> dict[str, Any]:
    for param in _TASKS[task_type].get("params") or []:
        if param["name"] == param_name:
            return param
    raise KeyError(f"{task_type}.{param_name}")


def _render_capabilities() -> str:
    blocks = []
    for task_type, task in _TASKS.items():
        lines = [
            (
                f"- {task_type}（{task['name']}） | "
                f"需要输入音频: {'是' if task['needs_audio'] else '否'} | "
                f"需要待念文本: {'是' if task.get('needs_text') else '否'}"
            ),
            f"  说明: {str(task['description']).strip()}",
        ]
        subtypes = task.get("subtypes") or []
        if subtypes:
            lines.append(f"  子类型: {' | '.join(subtypes)}")
        params = task.get("params") or []
        if params:
            param_lines = []
            hints = []
            for param in params:
                required_for = param.get("required_for") or []
                requirement = "必填" if param.get("required") else (f"{'/'.join(required_for)} 必填" if required_for else "可选")
                line = f"{param['name']}({param['type']},{requirement})"
                allowed_values = param.get("choices")
                if allowed_values:
                    line += f" 合法档位{allowed_values}"
                if param.get("labels"):
                    line += f" 取值{list(param['labels'])}"
                param_lines.append(line)
                hints.extend(param.get("normalization_hints") or [])
            lines.append("  槽位: " + "; ".join(param_lines))
            if hints:
                lines.append("  归一参考: " + "；".join(hints))
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _build_classify_prompt() -> str:
    return str(_PROMPTS["classify"]).format(capabilities=_render_capabilities())


def _task_normalization_context(
    task_type: str,
    language: str,
    *,
    output_language: str | None = None,
) -> str:
    task = _TASKS[task_type]
    templates = task["templates"]
    template = templates.get(language) or templates.get("en") or templates.get("zh")
    lines = [
        f"任务类型：{task_type}（{task['name']}）",
        f"标准 instruction 模板：{template}",
    ]
    rewrite = task.get("rewrite")
    if rewrite:
        template_rules = str(rewrite.get("template_rules") or "").strip()
        if template_rules:
            lines.append(f"填充规则：{template_rules}")
        lines.append(f"只扩写槽位“{rewrite['slot']}”，其余逐字保留。")
        resolved_output_language = output_language or rewrite.get("output_language")
        if resolved_output_language == "zh":
            lines.append(f"扩写后的“{rewrite['slot']}”必须使用中文。")
        elif resolved_output_language == "en":
            lines.append(f"扩写后的“{rewrite['slot']}”必须使用英文。")
        rewrite_rules = str(rewrite.get("instructions") or "").strip()
        if rewrite_rules:
            lines.append(f"扩写要求：{rewrite_rules}")
        example = rewrite.get("example")
        if example:
            lines.append(f"示例 · 修改前：{example.get('before', '')}")
            lines.append(f"示例 · 修改后：{example.get('after', '')}")
    return "\n".join(lines)


def _build_rewrite_prompt(
    task_type: str,
    language: str,
    *,
    output_language: str | None = None,
) -> str:
    rewrite = _TASKS[task_type]["rewrite"]
    slot = str(rewrite["slot"])
    resolved_output_language = output_language or rewrite.get("output_language")
    output_language_rule = str(_PROMPTS["rewrite_language_rules"].get(resolved_output_language, "")).format(slot=slot)
    return str(_PROMPTS["rewrite"]).format(
        task_context=_task_normalization_context(
            task_type,
            language,
            output_language=resolved_output_language,
        ),
        slot=slot,
        output_language_rule=output_language_rule,
    )


def _nonverbal_events() -> dict[str, dict[str, Any]]:
    return _PE_CONFIG["nonverbal_events"]


def _render_nonverbal_event_catalog() -> str:
    lines = []
    for event, spec in _nonverbal_events().items():
        aliases = [*spec.get("zh", []), *spec.get("en", [])]
        lines.append(f"- {event}: {spec.get('desc', '')}; aliases={aliases}")
    return "\n".join(lines)


def _build_nonverbal_event_prompt() -> str:
    return str(_PROMPTS["nonverbal_event_match"]).format(
        events=_render_nonverbal_event_catalog(),
    )


def _canonical_nonverbal_event(value: Any) -> str | None:
    normalized = str(value or "").strip().casefold()
    if not normalized:
        return None
    unwrapped = normalized.strip("[](){}<>\"' ")
    events = _nonverbal_events()
    if unwrapped in events:
        return unwrapped
    for event, spec in events.items():
        aliases = [*spec.get("zh", []), *spec.get("en", [])]
        normalized_aliases = {str(alias).strip().casefold().strip("[](){}<>\"' ") for alias in aliases}
        if unwrapped in normalized_aliases:
            return event
    return None


def _stable_rng(*values: Any) -> random.Random:
    serialized = json.dumps(values, ensure_ascii=False, sort_keys=True, default=str)
    seed = int.from_bytes(hashlib.sha256(serialized.encode("utf-8")).digest()[:8], "big")
    return random.Random(seed)


def _by_language(node: dict[str, Any], language: str) -> list[str] | None:
    other = "en" if language == "zh" else "zh"
    return node.get(language) or node.get(other)


def _fill_template(template: str, values: dict[str, Any]) -> str | None:
    result = template
    for placeholder in re.findall(r"\{(\w+)\}", template):
        value = values.get(placeholder)
        if value is None:
            return None
        result = result.replace(f"{{{placeholder}}}", str(value))
    return result


def _render_wenming_template(
    task_type: str,
    subtype: str | None,
    params: dict[str, Any],
    language: str,
    rng: random.Random,
) -> str | None:
    task = _EDITING_TEMPLATE_POOLS.get(task_type)
    if not task:
        return None
    if task_type == "speed_edit":
        key = f"{float(params['speed_multiplier']):.2f}x"
        values = (task.get("values") or {}).get(key)
        frames = (task.get("templates") or {}).get(language)
        if not values or not values.get(language) or not frames:
            return None
        value = rng.choice(values[language])
        return _fill_template(rng.choice(frames), {"value": value, "value_cap": value[:1].upper() + value[1:]})
    if task_type in ("pitch_edit", "volume_edit"):
        if subtype not in ("increase", "decrease"):
            return None
        prefix = "p" if subtype == "increase" else ("n" if task_type == "pitch_edit" else "m")
        amount = int(params["semitones"] if task_type == "pitch_edit" else params["gain_db"])
        key = f"{prefix}{amount:02d}" if task_type == "pitch_edit" else f"{prefix}{amount:02d}dB"
        values = (task.get("values") or {}).get(key)
        frames = (task.get("templates") or {}).get(language)
        if not values or not values.get(language) or not frames:
            return None
        value = rng.choice(values[language])
        return _fill_template(rng.choice(frames), {"value": value, "value_cap": value[:1].upper() + value[1:]})
    if task_type == "emotion_edit":
        values = (task.get("values") or {}).get(params.get("emotion"))
        frames = (task.get("templates") or {}).get(language)
        if not values or not values.get(language) or not frames:
            return None
        return _fill_template(rng.choice(frames), {"emo": values[language][0]})
    if task_type == "voice_edit":
        frames = (task.get("templates") or {}).get(language)
        return _fill_template(rng.choice(frames), {"desc": params.get("timbre_desc")}) if frames else None
    if task_type == "accent_edit":
        frames = (task.get("templates") or {}).get(language)
        return rng.choice(frames) if frames else None
    if task_type == "nonverbal_edit":
        mapping = _NONVERBAL_SUBTYPES.get(subtype)
        if not mapping:
            return None
        operation, variant, extra = mapping
        language_templates = (task.get("templates") or {}).get(language) or {}
        if subtype == "delete" and params.get("anchor"):
            variant = "anchor"
            extra = {"side": {"zh": "后", "en": "after"}}
        frames = (language_templates.get(operation) or {}).get(variant)
        if not frames:
            return None
        slots = {"nv": params.get("sound"), "anchor": params.get("anchor")}
        for name, tokens in extra.items():
            slots[name] = tokens[language]
        return _fill_template(rng.choice(frames), slots)
    return None


def _effect_bucket(effect: Any) -> str:
    text = str(effect or "").casefold()
    for bucket, keywords in _RUNTIME["effect_aliases"].items():
        if any(str(keyword).casefold() in text for keyword in keywords):
            return str(bucket)
    return str(_RUNTIME["effect_fallback"])


def _explicit_cleanup_mode(instruction: str) -> str | None:
    text = str(instruction or "").casefold()
    denoise = any(str(keyword).casefold() in text for keyword in _RUNTIME["cleanup_keywords"]["denoise"])
    dereverb = any(str(keyword).casefold() in text for keyword in _RUNTIME["cleanup_keywords"]["dereverb"])
    if denoise and dereverb:
        return "denoise_dereverb"
    if denoise:
        return "denoise"
    if dereverb:
        return "dereverb"
    return None


def _plain_prompt_pool(node: dict[str, Any], language: str) -> list[str] | None:
    pool = _by_language(node, language)
    if not pool:
        return None
    markers = ("去噪", "噪声", "混响", "denoise", "dereverb", "background noise", "room reverberation")
    plain = [frame for frame in pool if not any(marker in frame.casefold() for marker in markers)]
    return plain or pool


def _render_reference_prompt(
    task_type: str,
    subtype: str | None,
    params: dict[str, Any],
    language: str,
    rng: random.Random,
) -> str | None:
    if task_type not in _RESTORATION_TEMPLATE_POOLS:
        return None
    task = _RESTORATION_TEMPLATE_POOLS[task_type]
    cleanup_mode = params.get("cleanup_mode")
    if task_type == "enhance_speech":
        pool = _by_language(((task.get("cleanup") or {}).get(cleanup_mode) or {}), language) if cleanup_mode else None
        pool = pool or _by_language(task, language)
    elif task_type == "extract_vocals":
        pool = _by_language(task.get(subtype) or {}, language)
    elif task_type == "separate_speech":
        node = task.get(subtype) or {}
        pool = _by_language(((node.get("cleanup") or {}).get(cleanup_mode) or {}), language) if cleanup_mode else None
        pool = pool or _plain_prompt_pool(node, language)
    elif task_type == "improve_quality":
        if subtype == "bandwidth_extension":
            node = task["bandwidth_extension"]
        elif subtype == "remove_effect":
            node = task["remove_effect"].get(_effect_bucket(params.get("effect"))) or {}
        else:
            return None
        pool = _by_language(((node.get("cleanup") or {}).get(cleanup_mode) or {}), language) if cleanup_mode else None
        pool = pool or (_plain_prompt_pool(node, language) if cleanup_mode else _by_language(node, language))
    else:
        return None
    if not pool:
        return None
    return _fill_template(
        rng.choice(pool),
        {
            "text": params.get("text"),
            "n": params.get("n"),
            "n_zh": _integer_to_zh(params["n"]) if params.get("n") is not None else None,
            "n_ord": _ordinal_en(params["n"]) if params.get("n") is not None else None,
        },
    )


class PromptEnhancerError(RuntimeError):
    pass


class UnsupportedRequestError(PromptEnhancerError):
    pass


@dataclass(frozen=True)
class LLMCall:
    stage: str
    requested_model: str
    returned_model: str | None
    content: str
    reasoning_content: str | None
    usage: dict[str, Any] | None
    raw_response: dict[str, Any]


@dataclass(frozen=True)
class ASRCall:
    model: str
    text: str | None
    language: str | None
    raw_response: dict[str, Any] | None
    error: str | None = None


class ASRProvider(Protocol):
    def transcribe(self, audio_path: str) -> ASRCall: ...


class FallbackASRProvider:
    """Try ASR providers in order and return the first non-empty transcript."""

    def __init__(self, providers: list[ASRProvider]):
        self.providers = providers

    def transcribe(self, audio_path: str) -> ASRCall:
        attempts: list[str] = []
        for provider in self.providers:
            result = provider.transcribe(audio_path)
            if result.text:
                return result
            attempts.append(f"{result.model}: {result.error or 'ASR 返回空文本'}")
        return ASRCall(
            model=" -> ".join(type(provider).__name__ for provider in self.providers) or "none",
            text=None,
            language=None,
            raw_response=None,
            error="; ".join(attempts) or "没有可用的 ASR provider",
        )


class TencentCloudRecordingASR:
    """Tencent Cloud recording-file ASR using CreateRecTask and polling."""

    def __init__(
        self,
        *,
        secret_id: str,
        secret_key: str,
        engine_model_type: str = ASR_ENGINE_MODEL_TYPE,
        endpoint: str = ASR_ENDPOINT,
        region: str = "",
        timeout: int = ASR_TIMEOUT,
        poll_interval: float = ASR_POLL_INTERVAL_SEC,
        client: Any | None = None,
    ):
        self.engine_model_type = engine_model_type
        self.timeout = max(0.0, float(timeout))
        self.poll_interval = max(0.0, float(poll_interval))
        if client is None:
            http_profile = HttpProfile(endpoint=endpoint, reqTimeout=timeout)
            client_profile = ClientProfile(httpProfile=http_profile)
            client = asr_client.AsrClient(
                credential.Credential(secret_id, secret_key),
                region,
                client_profile,
            )
        self._client = client

    def transcribe(self, audio_path: str) -> ASRCall:
        try:
            wav_bytes = _load_asr_wav(audio_path)
            if len(wav_bytes) > ASR_MAX_DATA_BYTES:
                raise ValueError(f"ASR 音频为 {len(wav_bytes)} bytes，超过限制 {ASR_MAX_DATA_BYTES} bytes")

            request = asr_models.CreateRecTaskRequest()
            request.EngineModelType = self.engine_model_type
            request.ChannelNum = 1
            request.ResTextFormat = 0
            request.SourceType = 1
            request.Data = base64.b64encode(wav_bytes).decode("ascii")
            request.DataLen = len(wav_bytes)
            request.FilterDirty = 0
            request.FilterModal = 0
            request.FilterPunc = 0
            request.ConvertNumMode = 1

            submit_response = self._client.CreateRecTask(request)
            task_id = submit_response.Data.TaskId
            submit_payload = json.loads(submit_response.to_json_string())
            deadline = time.monotonic() + self.timeout

            while time.monotonic() < deadline:
                if self.poll_interval:
                    time.sleep(min(self.poll_interval, max(0.0, deadline - time.monotonic())))
                status_request = asr_models.DescribeTaskStatusRequest()
                status_request.TaskId = task_id
                status_response = self._client.DescribeTaskStatus(status_request)
                status = status_response.Data
                status_payload = json.loads(status_response.to_json_string())

                if status.Status == 2:
                    text = _clean_recording_asr_result(status.Result)
                    return ASRCall(
                        model=self.engine_model_type,
                        text=text or None,
                        language=_resolve_text_language(text) if text else None,
                        raw_response={"submit": submit_payload, "status": status_payload},
                        error=None if text else "ASR 返回空文本",
                    )
                if status.Status == 3:
                    raise RuntimeError(status.ErrorMsg or status.StatusStr or "录音文件识别失败")

            raise TimeoutError(f"录音文件识别轮询超过 {self.timeout:g} 秒")
        except Exception as exc:
            return ASRCall(
                model=self.engine_model_type,
                text=None,
                language=None,
                raw_response=None,
                error=f"{type(exc).__name__}: {exc}",
            )


class SenseVoiceSmallASR:
    """Lazy, CPU-first local ASR provider backed by FunASR SenseVoiceSmall."""

    _models: dict[tuple[str, str, int], Any] = {}
    _model_lock = threading.Lock()
    _inference_lock = threading.Lock()

    def __init__(
        self,
        *,
        model: str = "iic/SenseVoiceSmall",
        device: str = "cpu",
        ncpu: int = 4,
        model_instance: Any | None = None,
    ):
        self.model_name = model
        self.device = device
        self.ncpu = max(1, int(ncpu))
        self._model_instance = model_instance

    def _get_model(self):
        if self._model_instance is not None:
            return self._model_instance
        key = (self.model_name, self.device, self.ncpu)
        with self._model_lock:
            if key not in self._models:
                try:
                    from funasr import AutoModel
                except ImportError as exc:
                    raise RuntimeError("缺少 FunASR 依赖，请重新安装 AuK") from exc
                self._models[key] = AutoModel(
                    model=self.model_name,
                    device=self.device,
                    ncpu=self.ncpu,
                    disable_update=True,
                    disable_pbar=True,
                )
            return self._models[key]

    def transcribe(self, audio_path: str) -> ASRCall:
        try:
            model = self._get_model()
            with self._inference_lock:
                results = model.generate(
                    input=[audio_path],
                    cache={},
                    batch_size=1,
                    language="auto",
                    use_itn=True,
                )
            if not results or not isinstance(results[0], dict):
                raise ValueError("FunASR 返回空结果")
            raw_text = str(results[0].get("text") or "").strip()
            text = re.sub(r"<\|[^|]+\|>", "", raw_text).replace("</s>", "").strip()
            language = _sensevoice_language(raw_text, text)
            return ASRCall(
                model=self.model_name,
                text=text or None,
                language=language,
                raw_response={"text": raw_text},
                error=None if text else "FunASR 返回空文本",
            )
        except Exception as exc:
            return ASRCall(
                model=self.model_name,
                text=None,
                language=None,
                raw_response=None,
                error=f"{type(exc).__name__}: {exc}",
            )


@dataclass
class PromptEnhancerOutput:
    audio: str | None
    instruction: str
    gen_seconds: float
    ref_text: str | None
    gen_text: str | None
    task_type: str
    operation_subtype: str | None
    params_extracted: dict[str, Any]
    prepared_params: dict[str, Any]
    language: str
    text_language: str | None
    reasoning: str
    duration_source: str
    duration_details: dict[str, Any]
    asr: ASRCall | None
    llm_calls: list[LLMCall]
    cleanup_paths: list[str] = field(default_factory=list, repr=False)

    @property
    def generation_inputs(self) -> tuple[str | None, str, float, str | None, str | None]:
        """Return the five PE-owned inputs accepted by ``infer_gradio.run_generate``."""
        return self.audio, self.instruction, self.gen_seconds, self.ref_text, self.gen_text

    def to_run_generate_args(
        self,
        variant: str,
        nfe: int,
        cfg: float,
        seed: int | None,
    ) -> tuple[Any, ...]:
        return variant, *self.generation_inputs, nfe, cfg, seed

    def to_run_generate_kwargs(
        self,
        *,
        variant: str,
        nfe: int,
        cfg: float,
        seed: int | None,
    ) -> dict[str, Any]:
        return dict(
            zip(
                ("variant", "audio", "instruction", "gen_seconds", "ref_text", "gen_text", "nfe", "cfg", "seed"),
                self.to_run_generate_args(variant, nfe, cfg, seed),
            )
        )

    def debug_dict(self, *, include_raw_responses: bool = True) -> dict[str, Any]:
        output = asdict(self)
        output.pop("cleanup_paths", None)
        if not include_raw_responses:
            for call in output["llm_calls"]:
                call.pop("raw_response", None)
            if output["asr"]:
                output["asr"].pop("raw_response", None)
        return output

    @property
    def llm_outputs(self) -> list[dict[str, Any]]:
        """Return the model answers, including visible reasoning when the gateway provides it."""
        return [
            {
                "stage": call.stage,
                "requested_model": call.requested_model,
                "returned_model": call.returned_model,
                "content": call.content,
                "reasoning_content": call.reasoning_content,
                "usage": call.usage,
            }
            for call in self.llm_calls
        ]

    def cleanup(self) -> None:
        for path in self.cleanup_paths:
            try:
                os.remove(path)
            except OSError:
                pass
        self.cleanup_paths.clear()

    def __enter__(self) -> PromptEnhancerOutput:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.cleanup()


@dataclass(frozen=True)
class _Classified:
    supported: bool
    task_type: str
    operation_subtype: str | None
    params_extracted: dict[str, Any]
    needs_text: bool
    reasoning: str
    language: str
    text_language: str | None


class PromptEnhancer:
    """Convert ``instruction + optional audio`` into arguments accepted by Gradio's ``run_generate``.

    Prompts, task capabilities, template pools, and duration rules are loaded from
    ``pe.config.yaml``. Text LLM calls use an OpenAI-compatible Chat Completions
    endpoint configured by ``LLM_API_KEY``, ``LLM_BASE_URL``, and
    ``LLM_MODEL_NAME``. Uploaded audio uses Tencent Cloud recording-file ASR
    when credentials are available, then falls back to SenseVoiceSmall on CPU.
    """

    def __init__(
        self,
        *,
        llm_api_key: str | None = None,
        llm_base_url: str | None = None,
        llm_model: str | None = None,
        asr_provider: ASRProvider | None = None,
        asr_secret_id: str | None = None,
        asr_secret_key: str | None = None,
        asr_engine_model_type: str | None = None,
        llm_timeout: int = LLM_TIMEOUT,
        asr_timeout: int = ASR_TIMEOUT,
    ):
        self.llm_api_key = str(llm_api_key or os.environ.get("LLM_API_KEY") or "").strip()
        self.llm_base_url = str(llm_base_url or os.environ.get("LLM_BASE_URL") or "").strip().rstrip("/")
        self.llm_model = str(llm_model or os.environ.get("LLM_MODEL_NAME") or "").strip()
        missing_llm_config = [
            name
            for name, value in (
                ("LLM_API_KEY", self.llm_api_key),
                ("LLM_BASE_URL", self.llm_base_url),
                ("LLM_MODEL_NAME", self.llm_model),
            )
            if not value
        ]
        if missing_llm_config:
            raise ValueError(f"缺少 OpenAI-compatible LLM 配置: {missing_llm_config}")

        self.asr_secret_id = str(
            asr_secret_id or os.environ.get("TENCENTCLOUD_SECRET_ID") or os.environ.get("SecretId") or ""
        ).strip()
        self.asr_secret_key = str(
            asr_secret_key or os.environ.get("TENCENTCLOUD_SECRET_KEY") or os.environ.get("SecretKey") or ""
        ).strip()
        self.asr_engine_model_type = str(
            asr_engine_model_type or os.environ.get("ASR_ENGINE_MODEL_TYPE") or ASR_ENGINE_MODEL_TYPE
        ).strip()
        self._asr_provider = asr_provider
        self.llm_timeout = llm_timeout
        self.asr_timeout = asr_timeout
        self._llm_client = OpenAI(
            api_key=self.llm_api_key,
            base_url=self.llm_base_url,
            timeout=self.llm_timeout,
        )

    def prepare(
        self,
        instruction: str,
        audio_path: str | None = None,
        *,
        target_duration: float | None = None,
    ) -> PromptEnhancerOutput:
        user_instruction = str(instruction or "").strip()
        if not user_instruction:
            raise ValueError("instruction 不能为空")
        if audio_path and not Path(audio_path).is_file():
            raise FileNotFoundError(audio_path)

        llm_calls: list[LLMCall] = []
        asr = self._transcribe(audio_path) if audio_path else None
        classified, classify_call = self._classify(
            user_instruction,
            audio_path=audio_path,
            asr=asr,
        )
        llm_calls.append(classify_call)
        if not classified.supported:
            raise UnsupportedRequestError(classified.reasoning or "请求不在 AuK 单步能力范围内")
        if _TASKS[classified.task_type]["needs_audio"] and not audio_path:
            message = "Zero-shot TTS 需要参考音频" if classified.task_type == "zero_shot_tts" else "该任务需要输入音频"
            raise UnsupportedRequestError(message)

        params = self._normalize_params(classified, user_instruction=user_instruction)
        instruction_language = classified.language
        if classified.task_type == "nonverbal_edit":
            params, event_call, instruction_language = self._match_nonverbal_event(
                user_instruction,
                classified,
                params,
                audio_language=asr.language if asr else None,
            )
            llm_calls.append(event_call)
        if _TASKS[classified.task_type].get("rewrite"):
            params, rewrite_call = self._rewrite_description(
                classified,
                params,
                asr_text=asr.text if asr else None,
            )
            llm_calls.append(rewrite_call)

        if classified.task_type in ("instruct_tts", "zero_shot_tts"):
            params["text"] = _normalize_tts_text(
                str(params["text"]),
                classified.text_language or classified.language,
            )
        normalized_asr_text = None
        if asr and asr.text:
            normalized_asr_text = (
                _normalize_tts_text(asr.text, asr.language or classified.language)
                if classified.task_type == "zero_shot_tts"
                else asr.text
            )

        original_duration = _audio_duration(audio_path) if audio_path else None
        vad_bounds, base_duration = self._resolve_audio_duration(
            classified.task_type,
            classified.operation_subtype,
            audio_path,
            original_duration,
        )

        if target_duration is not None and float(target_duration) > 0:
            duration = float(target_duration)
            duration_source = "user"
            duration_details = {"requested_duration_sec": duration}
        else:
            duration, duration_source, duration_details = self._compute_duration(
                classified,
                params,
                base_duration=base_duration,
                original_duration=original_duration,
                asr_text=normalized_asr_text,
                asr_language=asr.language if asr else None,
                llm_calls=llm_calls,
            )
        target_len, duration = _quantize_model_duration(duration)
        duration_details.setdefault("original_duration_sec", original_duration)
        duration_details.setdefault("base_duration_sec", base_duration)
        duration_details["vad_bounds_sec"] = list(vad_bounds) if vad_bounds else None
        duration_details["applied_duration_sec"] = target_len / MODEL_LATENT_FRAMES_PER_SECOND
        duration_details["target_len"] = target_len

        model_instruction = _render_instruction(
            classified.task_type,
            classified.operation_subtype,
            params,
            instruction_language,
            classified.text_language,
        )
        processed_audio, cleanup_paths = _prepare_audio(
            audio_path,
            task_type=classified.task_type,
            operation_subtype=classified.operation_subtype,
            vad_bounds=vad_bounds,
        )
        if classified.task_type == "instruct_tts":
            processed_audio = None

        is_zero_shot = classified.task_type == "zero_shot_tts"
        return PromptEnhancerOutput(
            audio=processed_audio,
            instruction=model_instruction,
            gen_seconds=duration,
            ref_text=normalized_asr_text if is_zero_shot else None,
            gen_text=str(params.get("text") or "") if classified.task_type in ("instruct_tts", "zero_shot_tts") else None,
            task_type=classified.task_type,
            operation_subtype=classified.operation_subtype,
            params_extracted=classified.params_extracted,
            prepared_params=params,
            language=instruction_language,
            text_language=classified.text_language,
            reasoning=classified.reasoning,
            duration_source=duration_source,
            duration_details=duration_details,
            asr=asr,
            llm_calls=llm_calls,
            cleanup_paths=cleanup_paths,
        )

    def _call_llm(self, stage: str, messages: list[dict[str, str]]) -> LLMCall:
        request_kwargs = {
            "model": self.llm_model,
            "messages": messages,
            "max_tokens": LLM_MAX_TOKENS,
            "temperature": LLM_TEMPERATURE,
        }
        if any(provider in self.llm_base_url.lower() for provider in ("deepseek", "tokenhub.tencentmaas.com")):
            request_kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        try:
            response = self._llm_client.chat.completions.create(**request_kwargs)
            data = response.model_dump(mode="json")
        except OpenAIError as exc:
            raise PromptEnhancerError(f"{stage} LLM 调用失败: {type(exc).__name__}: {exc}") from exc
        if not response.choices:
            raise PromptEnhancerError(f"{stage} LLM 响应缺少 choices: {str(data)[:300]}")
        message = response.choices[0].message
        content = str(message.content or "").strip()
        if not content:
            raise PromptEnhancerError(f"{stage} LLM 返回空 content")
        return LLMCall(
            stage=stage,
            requested_model=self.llm_model,
            returned_model=str(response.model or "") or None,
            content=content,
            reasoning_content=None,
            usage=data.get("usage") if isinstance(data.get("usage"), dict) else None,
            raw_response=data,
        )

    def _classify(
        self,
        instruction: str,
        *,
        audio_path: str | None,
        asr: ASRCall | None,
    ) -> tuple[_Classified, LLMCall]:
        if audio_path:
            user_lines = ["【用户已上传参考/输入音频】", instruction]
            if asr and asr.text:
                user_lines.append(f"【输入音频 ASR 转写】{asr.text}")
                user_lines.append(f"【ASR 检测语种】{asr.language or '未知'}")
        else:
            user_lines = ["【用户未上传任何参考/输入音频，只能做纯文本合成 instruct_tts】", instruction]
        call = self._call_llm(
            "classify",
            [
                {"role": "system", "content": _build_classify_prompt()},
                {"role": "user", "content": "\n".join(user_lines)},
            ],
        )
        obj = _extract_json(call.content)
        supported = obj.get("supported")
        if not isinstance(supported, bool):
            raise PromptEnhancerError("分类响应 supported 必须是 boolean")
        task_type = str(obj.get("task_type") or "").strip()
        if supported and task_type not in _TASKS:
            raise PromptEnhancerError(f"分类响应包含未知 task_type: {task_type!r}")
        params = obj.get("params_extracted") or {}
        if not isinstance(params, dict):
            raise PromptEnhancerError("分类响应 params_extracted 必须是 object")
        language = _normalize_language(obj.get("language")) or "zh"
        text_language = None
        if task_type in ("instruct_tts", "zero_shot_tts"):
            text_language = _resolve_text_language(
                params.get("text"),
                declared_language=obj.get("text_language"),
                fallback_language=language,
            )
        subtype = obj.get("operation_subtype")
        subtype = str(subtype).strip() if subtype else None
        if (
            supported
            and task_type == "content_edit"
            and subtype == "replace"
            and any(marker in instruction.casefold() for marker in _VOCAL_EDIT_MARKERS)
        ):
            task_type = "vocal_edit"
            subtype = None
            params = {"orig": params.get("orig"), "new": params.get("new")}
        return (
            _Classified(
                supported=supported,
                task_type=task_type,
                operation_subtype=subtype,
                params_extracted=params,
                needs_text=bool(obj.get("needs_text", False)),
                reasoning=str(obj.get("reasoning") or ""),
                language=language,
                text_language=text_language,
            ),
            call,
        )

    def _rewrite_description(
        self,
        classified: _Classified,
        params: dict[str, Any],
        *,
        asr_text: str | None,
    ) -> tuple[dict[str, Any], LLMCall]:
        rewrite = _TASKS[classified.task_type]["rewrite"]
        slot = str(rewrite["slot"])
        configured_output_language = rewrite.get("output_language")
        output_language = (
            classified.text_language or classified.language
            if configured_output_language == "text_language"
            else configured_output_language
        )
        template_language = (
            classified.text_language or classified.language if classified.task_type == "instruct_tts" else classified.language
        )
        user_lines = [f"原始 {slot}：{params.get(slot, '')}"]
        if params.get("text"):
            user_lines.append(f"要念的文本（仅供参考，不要修改或放进描述里）：{params['text']}")
        if classified.task_type == "voice_edit":
            user_lines.append(f"输入音频 ASR（仅供理解内容）：{asr_text or '未获取'}")
        call = self._call_llm(
            f"rewrite_{slot}",
            [
                {
                    "role": "system",
                    "content": _build_rewrite_prompt(
                        classified.task_type,
                        template_language,
                        output_language=output_language,
                    ),
                },
                {"role": "user", "content": "\n".join(user_lines)},
            ],
        )
        expanded = str(_extract_json(call.content).get(slot) or "").strip()
        if not expanded:
            raise PromptEnhancerError(f"LLM 扩写返回空 {slot}")
        if output_language == "zh" and len(_CJK_RE.findall(expanded)) < 4:
            raise PromptEnhancerError(f"LLM 的 {slot} 未按要求输出中文")
        if output_language == "en" and (_CJK_RE.search(expanded) or not _EN_WORD_RE.search(expanded)):
            raise PromptEnhancerError(f"LLM 的 {slot} 未按要求输出英文")
        return {**params, slot: expanded}, call

    def _match_nonverbal_event(
        self,
        user_instruction: str,
        classified: _Classified,
        params: dict[str, Any],
        *,
        audio_language: str | None,
    ) -> tuple[dict[str, Any], LLMCall, str]:
        original_sound = str(params["sound"]).strip()
        preferred_language = _normalize_language(audio_language) or classified.language
        call = self._call_llm(
            "match_nonverbal_event",
            [
                {"role": "system", "content": _build_nonverbal_event_prompt()},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "user_instruction": user_instruction,
                            "operation_subtype": classified.operation_subtype,
                            "extracted_sound": original_sound,
                            "language": preferred_language,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
        )
        returned_event = _extract_json(call.content).get("event")
        event = _canonical_nonverbal_event(returned_event)
        events = _nonverbal_events()
        if event is None:
            raise PromptEnhancerError(f"LLM 返回未知 nonverbal event: {returned_event!r}")
        event_spec = events[event]
        language = "en" if preferred_language == "en" else "zh"
        names = event_spec.get(language) or []
        if not names:
            language = "en" if language == "zh" else "zh"
            names = event_spec.get(language) or []
        if not names:
            raise PromptEnhancerError(f"nonverbal event={event!r} 没有可用名称")
        sound = _stable_rng(user_instruction, event, language).choice(names)
        return (
            {
                **params,
                "event": event,
                "sound_original": original_sound,
                "sound": sound,
            },
            call,
            language,
        )

    def _transcribe(self, audio_path: str) -> ASRCall:
        if self._asr_provider is None:
            providers = []
            if self.asr_secret_id and self.asr_secret_key:
                providers.append(
                    TencentCloudRecordingASR(
                        secret_id=self.asr_secret_id,
                        secret_key=self.asr_secret_key,
                        engine_model_type=self.asr_engine_model_type,
                        timeout=self.asr_timeout,
                    )
                )
            providers.append(SenseVoiceSmallASR())
            self._asr_provider = FallbackASRProvider(providers)
        return self._asr_provider.transcribe(audio_path)

    def _normalize_params(
        self,
        classified: _Classified,
        *,
        user_instruction: str,
    ) -> dict[str, Any]:
        task_type = classified.task_type
        subtype = classified.operation_subtype
        params = dict(classified.params_extracted)
        spec = _TASKS[task_type]
        subtypes = spec.get("subtypes") or []
        if subtypes and subtype not in subtypes:
            raise PromptEnhancerError(f"{task_type} 缺少或包含非法 operation_subtype: {subtype!r}")

        if task_type == "zero_shot_tts":
            params = {"text": params.get("text")}
        if task_type == "speed_edit":
            speed_multiplier = _as_float(params.get("speed_multiplier"), "speed_multiplier")
            if speed_multiplier == 1.0:
                raise UnsupportedRequestError("语速 1.0 倍属于无操作")
            allowed = tuple(float(value) for value in _task_param(task_type, "speed_multiplier")["choices"])
            params["speed_multiplier"] = min(
                allowed,
                key=lambda candidate: abs(candidate - speed_multiplier),
            )
        elif task_type == "volume_edit":
            gain_db = _as_float(params.get("gain_db"), "gain_db")
            if gain_db == 0:
                raise UnsupportedRequestError("音量 0 dB 属于无操作")
            allowed = tuple(int(value) for value in _task_param(task_type, "gain_db")["choices"])
            params["gain_db"] = min(allowed, key=lambda candidate: abs(candidate - gain_db))
        elif task_type == "pitch_edit":
            semitones = _as_float(params.get("semitones"), "semitones")
            if semitones == 0:
                raise UnsupportedRequestError("音调 0 半音属于无操作")
            allowed = tuple(int(value) for value in _task_param(task_type, "semitones")["choices"])
            params["semitones"] = min(allowed, key=lambda candidate: abs(candidate - semitones))
        elif task_type == "emotion_edit":
            emotion = str(params.get("emotion") or "").casefold()
            allowed = _task_param(task_type, "emotion")["choices"]
            if emotion not in allowed:
                raise PromptEnhancerError(f"非法 emotion: {emotion!r}")
            params["emotion"] = emotion
        elif task_type in ("enhance_speech", "separate_speech", "improve_quality"):
            cleanup_mode = _explicit_cleanup_mode(user_instruction)
            if cleanup_mode:
                params["cleanup_mode"] = cleanup_mode
            else:
                params.pop("cleanup_mode", None)

        required = _required_params(task_type, subtype, params)
        missing = [
            name
            for name in required
            if params.get(name) is None or (isinstance(params.get(name), str) and not params[name].strip())
        ]
        if missing:
            raise PromptEnhancerError(f"{task_type} 缺少参数: {missing}")
        return params

    def _resolve_audio_duration(
        self,
        task_type: str,
        operation_subtype: str | None,
        audio_path: str | None,
        original_duration: float | None,
    ) -> tuple[tuple[float, float] | None, float | None]:
        if (
            not audio_path
            or task_type in NO_VAD_TASK_TYPES
            or (task_type == "whisper_edit" and operation_subtype == WHISPER_TO_NORMAL)
        ):
            return None, original_duration
        bounds = _vad_speech_bounds(audio_path)
        return (bounds, bounds[1] - bounds[0]) if bounds else (None, original_duration)

    def _compute_duration(
        self,
        classified: _Classified,
        params: dict[str, Any],
        *,
        base_duration: float | None,
        original_duration: float | None,
        asr_text: str | None,
        asr_language: str | None,
        llm_calls: list[LLMCall],
    ) -> tuple[float, str, dict[str, Any]]:
        task_type = classified.task_type
        rule = _TASKS[task_type]["duration"]
        category = rule["strategy"]
        if task_type == "instruct_tts":
            content = str(params["text"])
            f5_duration = _estimate_f5_instruct_duration(content, classified.text_language)
            try:
                call = self._call_llm(
                    "instruct_tts_duration",
                    [
                        {"role": "system", "content": INSTRUCT_DURATION_SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": json.dumps(
                                {
                                    "items": [
                                        {
                                            "key": "request",
                                            "language": classified.text_language or classified.language,
                                            "content": content,
                                            "style_instruction": str(params["style_desc"]),
                                            "f5_duration_sec": round(f5_duration, 6),
                                        }
                                    ]
                                },
                                ensure_ascii=False,
                            ),
                        },
                    ],
                )
                llm_calls.append(call)
                predicted_duration, ratio = _parse_duration_prediction(call.content, f5_duration)
            except PromptEnhancerError as exc:
                return (
                    f5_duration,
                    "f5_fallback",
                    {
                        "f5_duration_sec": f5_duration,
                        "duration_llm_error": str(exc),
                        "prompt_version": INSTRUCT_DURATION_PROMPT_VERSION,
                    },
                )
            return (
                predicted_duration,
                "hy3_adjust_f5",
                {
                    "f5_duration_sec": f5_duration,
                    "llm_predicted_duration_sec": predicted_duration,
                    "ratio_vs_f5": ratio,
                    "prompt_version": INSTRUCT_DURATION_PROMPT_VERSION,
                },
            )

        if task_type == "zero_shot_tts":
            target_duration = _estimate_f5_instruct_duration(str(params["text"]), classified.text_language)
            if base_duration and asr_text:
                target_duration = _estimate_f5_zero_shot_duration(
                    base_duration,
                    str(params["text"]),
                    asr_text,
                    target_language=classified.text_language,
                    reference_language=asr_language,
                )
                source = "reference_duration_x_weighted_text_ratio"
            else:
                source = "f5_target_text_fallback"
            return target_duration, source, {"base_duration_sec": base_duration, "reference_text": asr_text}

        if not base_duration:
            raise PromptEnhancerError(f"{task_type} 无法获得输入音频时长")
        if category == "speed_scaled":
            duration = base_duration / float(params["speed_multiplier"])
            return duration, "input_duration_div_speed", {"base_duration_sec": base_duration}
        if category == "content_scaled":
            duration = _content_scaled_duration(
                task_type,
                classified.operation_subtype,
                params,
                base_duration,
                asr_text,
                classified.language,
            )
            return duration, "content_ratio", {"base_duration_sec": base_duration, "asr_text": asr_text}
        if rule.get("emotion_multipliers"):
            coefficient = float(
                rule["emotion_multipliers"].get(
                    str(params["emotion"]),
                    rule["default_emotion_multiplier"],
                )
            )
            return (
                base_duration * coefficient,
                "input_duration_x_emotion_coefficient",
                {"base_duration_sec": base_duration, "coefficient": coefficient},
            )
        if rule.get("nonverbal_adjustments"):
            delta = _nonverbal_delta(classified.operation_subtype, params)
            return (
                max(0.1, base_duration + delta),
                "input_duration_plus_nonverbal_delta",
                {
                    "base_duration_sec": base_duration,
                    "event": params["event"],
                    "sound": params["sound"],
                    "sound_original": params["sound_original"],
                    "delta_sec": delta,
                },
            )
        if task_type == "whisper_edit" and classified.operation_subtype == WHISPER_TO_NORMAL:
            duration = original_duration or base_duration
            return duration, "whisper_to_normal_original_duration", {"original_duration_sec": duration}
        if category == "equal_length":
            return base_duration, "input_duration", {"base_duration_sec": base_duration}
        raise PromptEnhancerError(f"{task_type} 包含未知 duration category: {category!r}")


def prepare_generation(
    instruction: str,
    audio_path: str | None = None,
    **enhancer_kwargs,
) -> PromptEnhancerOutput:
    """One-shot convenience API.

    Example::

        with prepare_generation("用这个人的声音说“你好”", "/path/ref.wav") as prepared:
            output_path = run_generate(
                *prepared.to_run_generate_args("AuK (Base)", 32, 2.0, 44)
            )
            print(prepared.debug_dict())
    """
    return PromptEnhancer(**enhancer_kwargs).prepare(instruction, audio_path)


def _extract_json(text: str) -> dict[str, Any]:
    stripped = str(text or "").strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, re.DOTALL)
        if not match:
            raise PromptEnhancerError("LLM 响应没有 JSON object") from None
        payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise PromptEnhancerError("LLM 响应 JSON 不是 object")
    return payload


def _normalize_language(language: Any) -> str | None:
    value = str(language or "").strip().lower()
    if value.startswith("en"):
        return "en"
    if value.startswith(("zh", "cn")) or "chinese" in value:
        return "zh"
    return None


def _resolve_text_language(text: Any, *, declared_language: Any = None, fallback_language: Any = None) -> str:
    value = str(text or "")
    num_zh = len(_CJK_RE.findall(value))
    num_en = len(_EN_WORD_RE.findall(value))
    if num_zh and not num_en:
        return "zh"
    if num_en and not num_zh:
        return "en"
    if num_zh and num_en:
        return "zh" if num_zh * 0.21 >= num_en * 0.30 else "en"
    return _normalize_language(declared_language) or _normalize_language(fallback_language) or "zh"


def _sensevoice_language(raw_text: str, clean_text: str) -> str:
    tag = re.search(r"<\|(zh|en|yue|ja|ko|nospeech)\|>", raw_text, re.IGNORECASE)
    if tag:
        language = tag.group(1).lower()
        if language == "en":
            return "en"
        if language in {"zh", "yue"}:
            return "zh"
    return _resolve_text_language(clean_text)


def _as_float(value: Any, name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise PromptEnhancerError(f"{name}={value!r} 不是数字") from exc


def _required_params(task_type: str, subtype: str | None, params: dict[str, Any]) -> tuple[str, ...]:
    required = []
    for param in _TASKS[task_type].get("params") or []:
        if param.get("required") or subtype in (param.get("required_for") or []):
            required.append(param["name"])
    return tuple(required)


def _render_instruction(
    task_type: str,
    subtype: str | None,
    params: dict[str, Any],
    language: str,
    text_language: str | None,
) -> str:
    spec = _TASKS[task_type]
    lang = text_language if task_type in ("instruct_tts", "zero_shot_tts") else language
    lang = "en" if lang == "en" else "zh"
    if task_type == "zero_shot_tts":
        selected = spec["templates"][lang]
        return str(selected).format(text=params["text"])
    rng = _stable_rng(task_type, subtype, params, lang)
    instruction = _render_wenming_template(task_type, subtype, params, lang, rng)
    if instruction is None:
        instruction = _render_reference_prompt(task_type, subtype, params, lang, rng)
    if instruction is not None:
        return instruction
    templates = spec["templates"]
    selected = templates.get(lang) or templates.get("en" if lang == "zh" else "zh")
    if isinstance(selected, dict):
        selected = selected.get(subtype) if subtype is not None else next(iter(selected.values()), None)
    if selected is None:
        raise PromptEnhancerError(f"{task_type} 无可用模板")
    values = dict(params)
    if "emotion" in values and values["emotion"] in EMOTIONS:
        values["emotion"] = EMOTIONS[values["emotion"]][lang]
    if "n" in values:
        values["n_zh"] = _integer_to_zh(values["n"])
        values["n_en"] = _ordinal_en(values["n"])
    try:
        instruction = str(selected).format(**values)
    except KeyError as exc:
        raise PromptEnhancerError(f"{task_type} 渲染模板缺少参数 {exc.args[0]}") from exc
    return instruction


def _integer_to_zh(value: Any) -> str:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise PromptEnhancerError(f"非法说话人序号: {value!r}") from exc
    if number <= 0:
        raise PromptEnhancerError(f"说话人序号必须大于 0: {number}")
    if number >= 10000:
        return str(number)
    digits = "零一二三四五六七八九"
    units = ("", "十", "百", "千")
    pieces = []
    zero_pending = False
    chars = [int(char) for char in str(number)]
    for index, digit in enumerate(chars):
        unit_index = len(chars) - index - 1
        if digit == 0:
            zero_pending = bool(pieces)
            continue
        if zero_pending:
            pieces.append("零")
            zero_pending = False
        if not (digit == 1 and unit_index == 1 and not pieces):
            pieces.append(digits[digit])
        pieces.append(units[unit_index])
    return "".join(pieces)


def _ordinal_en(value: Any) -> str:
    number = int(value)
    words = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth", 6: "sixth"}
    if number in words:
        return words[number]
    suffix = "th" if 10 <= number % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(number % 10, "th")
    return f"{number}{suffix}"


def _get_tn_model(language: str):
    lang = "en" if language == "en" else "zh"
    with _TN_LOCK:
        if lang not in _TN_MODELS:
            if lang == "en":
                from tn.english.normalizer import Normalizer

                _TN_MODELS[lang] = Normalizer(overwrite_cache=False)
            else:
                from tn.chinese.normalizer import Normalizer

                _TN_MODELS[lang] = Normalizer(
                    remove_interjections=False,
                    remove_erhua=False,
                    remove_puncts=False,
                    overwrite_cache=False,
                )
    return _TN_MODELS[lang]


def _normalize_tts_text(text: str, language: str, *, join_split_digits: bool = False) -> str:
    value = str(text or "")

    def join_digits(match: re.Match) -> str:
        groups = re.split(r"[ \t　]+", match.group(0))
        next_char = value[match.end() : match.end() + 1]
        return "".join(groups) if len(groups) == 2 or next_char in _MULTIGROUP_JOIN_SUFFIXES else match.group(0)

    source = _DIGIT_SPACED_RUN_RE.sub(join_digits, value) if join_split_digits else value
    if not source.strip():
        return source
    lang = _resolve_text_language(source, declared_language=language, fallback_language=language)
    try:
        if lang == "zh":
            source = _ZH_TN_SPACE_RE.sub("", source)
        normalized = _get_tn_model(lang).normalize(source)
        return _ZH_TN_SPACE_RE.sub("", normalized) if lang == "zh" else normalized
    except Exception:
        return source


def _utf8_byte_count(text: str | None) -> int:
    return len(str(text or "").encode("utf-8")) if str(text or "").strip() else 0


def _tts_utf8_weight(text: str | None, language: str | None) -> float:
    value = str(text or "")
    if not value.strip():
        return 0.0
    fallback = _resolve_text_language(value, declared_language=language)
    script_languages = [
        "zh" if _CJK_RE.fullmatch(char) else ("en" if char.isascii() and char.isalpha() else None) for char in value
    ]
    next_languages: list[str | None] = [None] * len(value)
    next_language = None
    for index in range(len(value) - 1, -1, -1):
        if script_languages[index]:
            next_language = script_languages[index]
        next_languages[index] = next_language
    weight = 0.0
    previous_language = None
    for index, char in enumerate(value):
        char_language = script_languages[index]
        if char_language is None:
            char_language = previous_language or next_languages[index] or fallback
        else:
            previous_language = char_language
        weight += len(char.encode("utf-8")) * TTS_SEC_PER_UTF8_BYTE[char_language]
    return weight


def _f5_local_speed(text: str | None) -> float:
    return F5_SHORT_TEXT_SPEED if _utf8_byte_count(text) < F5_SHORT_TEXT_BYTE_THRESHOLD else 1.0


def _estimate_f5_instruct_duration(text: str, language: str | None) -> float:
    weight = _tts_utf8_weight(text, language)
    frames = int(weight * F5_SAMPLE_RATE / F5_HOP_LENGTH / _f5_local_speed(text))
    return frames * F5_HOP_LENGTH / F5_SAMPLE_RATE


def _estimate_f5_zero_shot_duration(
    reference_duration: float,
    target_text: str,
    reference_text: str,
    *,
    target_language: str | None,
    reference_language: str | None,
) -> float:
    target_weight = _tts_utf8_weight(target_text, target_language)
    reference_weight = _tts_utf8_weight(reference_text, reference_language)
    if target_weight <= 0 or reference_weight <= 0:
        return 0.0
    reference_frames = int(reference_duration * F5_SAMPLE_RATE / F5_HOP_LENGTH)
    target_frames = int(reference_frames * target_weight / reference_weight / _f5_local_speed(target_text))
    return target_frames * F5_HOP_LENGTH / F5_SAMPLE_RATE


def _spoken_duration(text: str | None, fallback_language: str | None) -> float:
    value = str(text or "")
    num_zh = len(_CJK_RE.findall(value))
    num_en = len(_EN_WORD_RE.findall(value))
    duration = num_zh * 0.21 + num_en * 0.30
    if duration > 0:
        return duration
    language = _resolve_text_language(value, fallback_language=fallback_language)
    units = len(value.split()) if language == "en" else len(re.findall(r"\S", value))
    return units * (0.30 if language == "en" else 0.21)


def _content_scaled_duration(
    task_type: str,
    subtype: str | None,
    params: dict[str, Any],
    base_duration: float,
    asr_text: str | None,
    language: str,
) -> float:
    if task_type == "vocal_edit":
        add_slot, delete_slot = "new", "orig"
    else:
        add_slot, delete_slot = {
            "insert_before": ("text", None),
            "insert_after": ("text", None),
            "delete": (None, "target"),
            "delete_before": (None, "target"),
            "delete_after": (None, "target"),
            "replace": ("new", "orig"),
        }[str(subtype)]
    if asr_text:
        original = _spoken_duration(asr_text, language)
        if original <= 0:
            return base_duration
        edited = original
        if add_slot:
            edited += _spoken_duration(params.get(add_slot), language)
        if delete_slot:
            edited -= _spoken_duration(params.get(delete_slot), language)
        return base_duration * max(0.05, edited) / original
    if add_slot and delete_slot:
        original = _spoken_duration(params.get(delete_slot), language)
        replacement = _spoken_duration(params.get(add_slot), language)
        return base_duration * replacement / original if original > 0 else base_duration
    return base_duration


def _nonverbal_delta(subtype: str | None, params: dict[str, Any]) -> float:
    operation = "delete" if str(subtype).startswith("delete") else "add"
    rule = _TASKS["nonverbal_edit"]["duration"]
    event = str(params.get("event") or "").casefold()
    event_spec = _nonverbal_events().get(event) or {}
    match_text = " ".join(
        [
            event,
            *(str(value) for value in event_spec.get("zh", [])),
            *(str(value) for value in event_spec.get("en", [])),
        ]
    ).casefold()
    for family in rule["nonverbal_adjustments"]:
        if any(str(keyword).casefold() in match_text for keyword in family["keywords"]):
            return float(family[operation])
    return float(rule["default_nonverbal_adjustment"][operation])


def _parse_duration_prediction(content: str, f5_duration: float) -> tuple[float, float]:
    results = _extract_json(content).get("results")
    if not isinstance(results, list) or len(results) != 1 or not isinstance(results[0], dict):
        raise PromptEnhancerError("InstructTTS 时长响应必须包含一个 result")
    result = results[0]
    if str(result.get("key") or "") != "request":
        raise PromptEnhancerError("InstructTTS 时长响应 key 不匹配")
    try:
        duration = float(result["duration_sec"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PromptEnhancerError("InstructTTS duration_sec 非数字") from exc
    ratio = duration / f5_duration
    if not math.isfinite(duration) or duration < 0.4 or not 0.45 <= ratio <= 2.20:
        raise PromptEnhancerError(f"InstructTTS 时长非法: duration={duration}, ratio={ratio}")
    if result.get("ratio") is not None and abs(float(result["ratio"]) - ratio) > 0.03:
        raise PromptEnhancerError("InstructTTS 返回的 duration 与 ratio 不一致")
    return duration, ratio


def _quantize_model_duration(duration: float) -> tuple[int, float]:
    if not math.isfinite(float(duration)) or float(duration) <= 0:
        raise PromptEnhancerError(f"非法目标时长: {duration}")
    target_len = max(1, round(float(duration) * MODEL_LATENT_FRAMES_PER_SECOND))
    nominal_seconds = target_len / MODEL_LATENT_FRAMES_PER_SECOND
    if math.ceil(nominal_seconds * MODEL_LATENT_FRAMES_PER_SECOND) != target_len:
        nominal_seconds = math.nextafter(nominal_seconds, 0.0)
    return target_len, nominal_seconds


def _audio_duration(audio_path: str) -> float:
    info = torchaudio.info(audio_path)
    if info.sample_rate <= 0:
        raise PromptEnhancerError(f"输入音频采样率非法: {info.sample_rate}")
    return info.num_frames / info.sample_rate


def _load_asr_pcm16(audio_path: str) -> bytes:
    waveform, sample_rate = torchaudio.load(audio_path)
    if waveform.ndim != 2 or waveform.shape[-1] == 0:
        raise ValueError(f"ASR 输入音频形状非法: {tuple(waveform.shape)}")
    if sample_rate <= 0:
        raise ValueError(f"ASR 输入音频采样率非法: {sample_rate}")

    duration = waveform.shape[-1] / sample_rate
    if duration > ASR_MAX_DURATION_SEC:
        raise ValueError(f"ASR 输入音频时长 {duration:.2f}s，超过限制 {ASR_MAX_DURATION_SEC:.0f}s")

    mono = waveform.mean(dim=0, keepdim=True)
    if sample_rate != ASR_SAMPLE_RATE:
        mono = torchaudio.transforms.Resample(sample_rate, ASR_SAMPLE_RATE)(mono)
    mono = mono.squeeze(0).to(torch.float64)
    if not torch.isfinite(mono).all():
        raise ValueError("ASR 输入音频包含 NaN/Inf")
    pcm16 = torch.round(mono.clamp(-1.0, 1.0) * 32767.0).to(torch.int16).contiguous().numpy()
    return pcm16.astype("<i2", copy=False).tobytes()


def _load_asr_wav(audio_path: str) -> bytes:
    pcm_bytes = _load_asr_pcm16(audio_path)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(ASR_SAMPLE_RATE)
        wav_file.writeframes(pcm_bytes)
    return buffer.getvalue()


def _clean_recording_asr_result(result: str | None) -> str:
    lines = []
    for line in str(result or "").splitlines():
        text = _RECORDING_RESULT_PREFIX_RE.sub("", line).strip()
        if text:
            lines.append(text)
    return "\n".join(lines)


def _get_vad_model():
    global _VAD_MODEL
    with _VAD_LOCK:
        if _VAD_MODEL is None:
            from silero_vad import load_silero_vad

            _VAD_MODEL = load_silero_vad()
        return _VAD_MODEL


def _vad_speech_bounds(audio_path: str) -> tuple[float, float] | None:
    try:
        from silero_vad import get_speech_timestamps

        model = _get_vad_model()
        audio, sample_rate = torchaudio.load(audio_path)
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        if sample_rate != 16_000:
            audio = torchaudio.transforms.Resample(sample_rate, 16_000)(audio)
        signal = audio.squeeze(0)
        rms = float((signal.to(torch.float64) ** 2).mean().sqrt())
        peak = float(signal.abs().max())
        if 0 < rms < VAD_NORM_RMS_THRESHOLD and peak > 0:
            signal = signal * (VAD_NORM_TARGET_PEAK / peak)
        with _VAD_LOCK:
            timestamps = get_speech_timestamps(
                signal,
                model,
                sampling_rate=16_000,
                return_seconds=True,
                time_resolution=3,
            )
        if not timestamps:
            return None
        return float(timestamps[0]["start"]), float(timestamps[-1]["end"])
    except Exception:
        return None


def _prepare_audio(
    audio_path: str | None,
    *,
    task_type: str,
    operation_subtype: str | None,
    vad_bounds: tuple[float, float] | None,
) -> tuple[str | None, list[str]]:
    if not audio_path:
        return None, []
    current = audio_path
    cleanup_paths: list[str] = []
    if task_type not in NO_VAD_TASK_TYPES and not (task_type == "whisper_edit" and operation_subtype == WHISPER_TO_NORMAL):
        trimmed = _trim_audio(audio_path, vad_bounds)
        if trimmed != audio_path:
            cleanup_paths.append(trimmed)
            current = trimmed
    if task_type == "whisper_edit":
        if operation_subtype == WHISPER_TO_NORMAL:
            normalized = _normalize_audio_level(
                audio_path,
                target_rms=WHISPER_TO_NORMAL_TARGET_RMS,
                use_lufs=False,
            )
        elif operation_subtype == WHISPER_TO_WHISPER:
            normalized = _normalize_audio_level(
                current,
                target_rms=WHISPER_TARGET_RMS,
                target_lufs=WHISPER_TARGET_LUFS,
                use_lufs=True,
            )
        else:
            normalized = current
        if normalized != current:
            cleanup_paths.append(normalized)
            current = normalized
    return current, cleanup_paths


def _trim_audio(audio_path: str, bounds: tuple[float, float] | None) -> str:
    if not bounds:
        return audio_path
    try:
        audio, sample_rate = torchaudio.load(audio_path)
        total = audio.shape[-1]
        padding = round(VAD_TRIM_PAD_SEC * sample_rate)
        start = max(0, round(bounds[0] * sample_rate) - padding)
        end = min(total, round(bounds[1] * sample_rate) + padding)
        if end <= start or (start == 0 and end == total):
            return audio_path
        handle, path = tempfile.mkstemp(prefix="auk_pe_vad_", suffix=".wav")
        os.close(handle)
        torchaudio.save(path, audio[:, start:end], sample_rate, encoding="PCM_S", bits_per_sample=16)
        return path
    except Exception:
        return audio_path


def _normalize_audio_level(
    audio_path: str,
    *,
    target_rms: float,
    target_lufs: float | None = None,
    use_lufs: bool,
) -> str:
    try:
        waveform, sample_rate = torchaudio.load(audio_path)
        mono = waveform.mean(dim=0).to(torch.float64)
        gain = None
        if use_lufs and target_lufs is not None:
            try:
                import pyloudnorm as pyln

                measured_lufs = pyln.Meter(sample_rate).integrated_loudness(mono.numpy())
                if math.isfinite(measured_lufs):
                    gain = 10.0 ** ((target_lufs - measured_lufs) / 20.0)
            except Exception:
                gain = None
        if gain is None:
            rms = float(torch.sqrt(torch.mean(mono**2)))
            gain = target_rms / max(rms, 1e-9)
        output = mono * gain
        peak = float(output.abs().max())
        if peak > WHISPER_PEAK_CEILING:
            output *= WHISPER_PEAK_CEILING / peak
        handle, path = tempfile.mkstemp(prefix="auk_pe_level_", suffix=".wav")
        os.close(handle)
        torchaudio.save(
            path,
            output.to(torch.float32).unsqueeze(0),
            sample_rate,
            encoding="PCM_S",
            bits_per_sample=16,
        )
        return path
    except Exception:
        return audio_path
