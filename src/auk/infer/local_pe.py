"""Local Prompt Enhancer backend: MiniCPM5-2B for the LLM stages, SenseVoice for ASR.

Opt-in via ``auk-gradio --local`` (or ``build_local_enhancer()``). The cloud path
in ``pe.py`` is untouched: this module only supplies a drop-in replacement for the
``openai`` client that ``PromptEnhancer`` talks to, plus the local ASR provider
that already ships in ``pe.py``.

Small local models emit JSON that is occasionally malformed in ways the cloud
models are not. Those repairs live here, at the client boundary, rather than in
``pe.py``'s shared parser -- so a well-formed response (every cloud response) is
returned byte-for-byte unchanged.
"""

from __future__ import annotations

import json
import re
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch


DEFAULT_MODEL_REPO = "openbmb/MiniCPM5-2B"
DEFAULT_CKPT_DIR = Path("ckpts")
# Generation is stopped by eos_token_id; this only bounds a runaway decode.
MAX_NEW_TOKENS = 4096

# Cut a continuation if the model starts a fresh turn after its answer.
_STOP_MARKERS = ("\nHuman:", "\nUser:", "\nAssistant:", "<|im_end|>")

# --- JSON repairs, in order of specificity -----------------------------------
# A schema hint copied into the payload: `"text_language":"zh" 或 "en" 或 null`.
_ALTERNATION_RE = re.compile(
    r'(?<=:)\s*("(?:[^"\\]|\\.)*"|null|true|false|-?\d+(?:\.\d+)?)'
    r'(?:\s*(?:或|or|\||/)\s*(?:"(?:[^"\\]|\\.)*"|null|true|false|-?\d+(?:\.\d+)?))+'
)
# A closing quote dropped after CJK punctuation, letting a brace fall inside the
# value: `"text":"宁可我负天下人。}",` -- the quote belongs before the brace.
_BRACE_IN_STRING_RE = re.compile(r'([}\])]+)"(\s*[,}\]])')
# The model quoting the user back inside a value without escaping:
# `"reasoning":"...要念的文本（"Welcome back"），符合条件。"`
_INNER_QUOTE_RE = re.compile(r'(:\s*")(.*?)("\s*(?:,\s*"[\w$]+"\s*:|\}\s*$|\}\s*,))', re.DOTALL)


def _truncate_at_markers(text: str) -> str:
    cut = len(text)
    for marker in _STOP_MARKERS:
        index = text.find(marker)
        if index != -1:
            cut = min(cut, index)
    return text[:cut].strip()


def _strip_think_block(text: str) -> str:
    """Drop a leading <think>...</think> block if the chat template emitted one."""
    stripped = text.lstrip()
    if not stripped.startswith("<think>"):
        return text.strip()
    end = stripped.find("</think>")
    return stripped[end + len("</think>") :].strip() if end != -1 else ""


def _merge_objects(text: str) -> str | None:
    """Merge a response split across consecutive top-level objects.

    `{"reasoning": ...}\\n{"task_type": ..., "params_extracted": ...}` -- each part
    parses on its own, so the naive reader keeps only the first and loses the
    fields that matter.
    """
    decoder = json.JSONDecoder()
    merged: dict[str, Any] = {}
    index = 0
    while index < len(text):
        start = text.find("{", index)
        if start < 0:
            break
        try:
            payload, end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            break
        if isinstance(payload, dict):
            merged = {**payload, **merged} if merged else payload
        index = start + end
    return json.dumps(merged, ensure_ascii=False) if len(merged) > 1 else None


def _repair_candidates(text: str):
    """Yield progressively more aggressive repairs of a non-JSON response."""
    merged = _merge_objects(text)
    if merged:
        yield merged
    collapsed = _ALTERNATION_RE.sub(lambda m: m.group(1), text)
    if collapsed != text:
        yield collapsed
    for base in dict.fromkeys((collapsed, text)):
        moved = _BRACE_IN_STRING_RE.sub(lambda m: f'"{m.group(1)}{m.group(2)}', base)
        if moved != base:
            yield moved
    start = text.find("{")
    if start >= 0:
        head = text[start:]
        # Truncated mid-object: close the open braces, optionally dropping a
        # trailing partial field.
        if head.count("{") > head.count("}"):
            for attempt in (head, head[: head.rfind(",")] if "," in head else head):
                closed = attempt.rstrip().rstrip(",")
                yield closed + "}" * (closed.count("{") - closed.count("}"))
        match = re.search(r"\{.*\}", head, re.DOTALL)
        if match:
            yield match.group(0)
    # Broadest, so last: escape stray quotes inside values.
    escaped = _INNER_QUOTE_RE.sub(
        lambda m: (
            f"{m.group(1)}{m.group(2).replace(chr(34), chr(92) + chr(34))}{m.group(3)}" if '"' in m.group(2) else m.group(0)
        ),
        text,
    )
    if escaped != text:
        yield escaped


def _sanitize_json(text: str) -> str:
    """Return `text` unchanged when it already parses, else the first repair that does."""
    body = text.strip()
    if body.startswith("```"):
        body = re.sub(r"^```(?:json)?\s*", "", body)
        body = re.sub(r"\s*```$", "", body).strip()
    try:
        json.loads(body)
        return body
    except json.JSONDecodeError:
        pass
    for candidate in _repair_candidates(body):
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            continue
    return body  # let PromptEnhancer report it


def resolve_model_dir(
    repo_id: str = DEFAULT_MODEL_REPO,
    ckpt_dir: Path | str = DEFAULT_CKPT_DIR,
) -> Path:
    """Return the local snapshot dir, downloading it on first use."""
    target = Path(ckpt_dir) / repo_id.split("/")[-1]
    if (target / "config.json").is_file():
        return target
    print(f"[AuK] Local PE model not found; downloading {repo_id} -> {target} (~4.7 GB, first run only) ...")
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=repo_id, local_dir=str(target))
    print(f"[AuK] Download complete: {target}")
    return target


@dataclass
class _Message:
    content: str
    role: str = "assistant"


@dataclass
class _Choice:
    message: _Message
    index: int = 0
    finish_reason: str = "stop"


@dataclass
class _Usage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class _Completion:
    """Duck-types the parts of ``openai.types.chat.ChatCompletion`` that PE reads."""

    choices: list[_Choice]
    model: str
    usage: _Usage
    id: str = field(default_factory=lambda: f"local-{uuid.uuid4().hex[:12]}")
    object: str = "chat.completion"
    created: int = 0

    def model_dump(self, mode: str = "python") -> dict[str, Any]:
        return {
            "id": self.id,
            "object": self.object,
            "created": self.created,
            "model": self.model,
            "choices": [
                {
                    "index": c.index,
                    "finish_reason": c.finish_reason,
                    "message": {"role": c.message.role, "content": c.message.content},
                }
                for c in self.choices
            ],
            "usage": {
                "prompt_tokens": self.usage.prompt_tokens,
                "completion_tokens": self.usage.completion_tokens,
                "total_tokens": self.usage.total_tokens,
            },
        }


class _Completions:
    def __init__(self, client: LocalLLMClient):
        self._client = client

    def create(
        self,
        *,
        model: str | None = None,
        messages: list[dict[str, str]],
        max_tokens: int = MAX_NEW_TOKENS,
        temperature: float = 0.0,
        extra_body: dict[str, Any] | None = None,
        **_ignored: Any,
    ) -> _Completion:
        text, prompt_tokens, completion_tokens = self._client.generate(
            messages,
            max_new_tokens=int(max_tokens or MAX_NEW_TOKENS),
            temperature=temperature,
        )
        return _Completion(
            choices=[_Choice(message=_Message(content=text))],
            model=model or self._client.model_name,
            usage=_Usage(prompt_tokens, completion_tokens, prompt_tokens + completion_tokens),
        )


class _Chat:
    def __init__(self, client: LocalLLMClient):
        self.completions = _Completions(client)


class LocalLLMClient:
    """Drop-in stand-in for ``openai.OpenAI``, backed by a local causal LM."""

    def __init__(
        self,
        model_dir: str | Path | None = None,
        *,
        device: str | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        path = str(model_dir or resolve_model_dir())
        self.model_name = Path(path).name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._lock = threading.RLock()
        self._tokenizer = AutoTokenizer.from_pretrained(path)
        self._model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype).to(self.device).eval()
        self.chat = _Chat(self)

    def _eos_ids(self) -> list[int]:
        ids: set[int] = set()
        eos = getattr(self._tokenizer, "eos_token_id", None)
        if isinstance(eos, int):
            ids.add(eos)
        elif isinstance(eos, (list, tuple)):
            ids.update(i for i in eos if isinstance(i, int))
        for token in ("<|im_end|>", "<|endoftext|>"):
            token_id = self._tokenizer.convert_tokens_to_ids(token)
            if isinstance(token_id, int) and token_id >= 0:
                ids.add(token_id)
        return sorted(ids)

    @torch.inference_mode()
    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_new_tokens: int = MAX_NEW_TOKENS,
        temperature: float = 0.0,
    ) -> tuple[str, int, int]:
        try:
            prompt = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:  # template without a thinking switch
            prompt = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        encoded = self._tokenizer([prompt], return_tensors="pt")
        # Llama-family generate() rejects token_type_ids outright.
        inputs = {k: v.to(self.device) for k, v in encoded.items() if k in ("input_ids", "attention_mask")}
        with self._lock:
            output = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                eos_token_id=self._eos_ids(),
                pad_token_id=self._tokenizer.pad_token_id or self._tokenizer.eos_token_id,
            )
        prompt_len = int(inputs["input_ids"].shape[-1])
        generated = output[0][prompt_len:]
        raw = self._tokenizer.decode(generated, skip_special_tokens=True)
        text = _sanitize_json(_truncate_at_markers(_strip_think_block(raw)))
        return text, prompt_len, int(generated.shape[-1])


def build_local_enhancer(
    model_dir: str | Path | None = None,
    *,
    device: str | None = None,
    asr_provider: Any | None = None,
    **enhancer_kwargs: Any,
):
    """Build a ``PromptEnhancer`` that uses the local LLM and local SenseVoice ASR.

    Pass ``asr_provider`` to override ASR only (e.g. the CLI's ``--asr cloud``);
    the LLM stages stay local either way.
    """
    from auk.infer.pe import PromptEnhancer, SenseVoiceSmallASR

    return PromptEnhancer(
        llm_client=LocalLLMClient(model_dir, device=device),
        asr_provider=asr_provider or SenseVoiceSmallASR(),
        **enhancer_kwargs,
    )
