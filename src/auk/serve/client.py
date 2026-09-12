"""Client side of the split deployment.

:class:`AukDistributedInfer` is the drop-in orchestrator: it mirrors :class:`AukInfer.generate`
but owns no GPU. It tokenizes locally (the processor is CPU-only and small), asks the text-encoder
node for the fused embedding, then hands everything to the worker node, which runs the ODE loop
in full and returns the waveform.
"""

from __future__ import annotations

import http.client
import logging
import math
import os
from urllib.parse import urlparse

import torch

from auk.infer.infer_auk import apply_flash_recipe, extract_audio_path
from auk.model.cfm_edit import CFMEdit
from auk.serve.codec import pack, unpack

logger = logging.getLogger(__name__)


class _Connection:
    """Keep-alive HTTP connection to one node."""

    def __init__(self, url: str, timeout: float | None = None):
        parsed = urlparse(url if "://" in url else f"http://{url}")
        self.host = parsed.hostname or "127.0.0.1"
        self.port = parsed.port or 80
        self.path_prefix = parsed.path.rstrip("/")
        self.timeout = timeout
        self._conn: http.client.HTTPConnection | None = None

    def _get(self) -> http.client.HTTPConnection:
        if self._conn is None:
            self._conn = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)
        return self._conn

    def _request(self, method: str, route: str, body: bytes | None = None) -> bytes:
        # A long-lived client will eventually hit an idle-timeout close; reconnect once rather
        # than failing the generation.
        for attempt in (0, 1):
            conn = self._get()
            headers = {"Content-Type": "application/octet-stream"} if body is not None else {}
            try:
                conn.request(method, f"{self.path_prefix}{route}", body=body, headers=headers)
                response = conn.getresponse()
                data = response.read()
            except (http.client.HTTPException, OSError):
                self.close()
                if attempt:
                    raise
                continue
            if response.status != 200:
                raise RuntimeError(f"{self.host}:{self.port}{route} -> HTTP {response.status}: {data[:400]!r}")
            return data
        raise RuntimeError("unreachable")

    def post(self, route: str, body: bytes) -> bytes:
        return self._request("POST", route, body)

    def get_json(self, route: str) -> dict:
        import json

        return json.loads(self._request("GET", route))

    def close(self):
        if self._conn is not None:
            self._conn.close()
            self._conn = None


def plan_latent_lengths(
    ref_num_samples: int,
    ref_sample_rate: int,
    gen_seconds: float | None,
    target_sample_rate: int,
    downsample_rate: int,
) -> tuple[int, int]:
    """Latent length of the reference clip and of the segment to generate.

    Mirrors ``AukInfer.generate``, but works from the *source* sample rate: the client does not
    resample, it just tells the worker what rate the waveform is in and lets the worker do it.
    """
    if ref_num_samples and ref_sample_rate > 0:
        ref_latent_len = ref_num_samples * target_sample_rate // (ref_sample_rate * downsample_rate)
    else:
        ref_latent_len = 0
    if gen_seconds is not None:
        gen_latent_len = max(1, int(math.ceil(gen_seconds * target_sample_rate / downsample_rate)))
    else:
        gen_latent_len = max(1, ref_latent_len)
    return ref_latent_len, gen_latent_len


class RemoteTextEncoder:
    """Client for the Qwen + layer-fusion node."""

    def __init__(self, url: str, timeout: float | None = None):
        self.url = url
        self.conn = _Connection(url, timeout)

    @torch.no_grad()
    def encode(self, cond_inputs) -> tuple[torch.Tensor, torch.Tensor]:
        tensors = {k: v for k, v in dict(cond_inputs).items() if torch.is_tensor(v)}
        _, out = unpack(self.conn.post("/encode", pack({}, tensors)))
        return out["hidden"], out["attention_mask"].bool()


class RemoteWorker:
    """Client for the VAE + DiT + ODE node."""

    def __init__(self, url: str, timeout: float | None = None):
        self.url = url
        self.conn = _Connection(url, timeout)

    def info(self) -> dict:
        return self.conn.get_json("/info")

    def generate(
        self,
        ref_audio: torch.Tensor | None,
        sample_rate: int,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        gen_latent_len: int,
        nfe: int,
        cfg_strength: float,
        sway_sampling_coef: float | None,
        t_grid: list[float] | None,
        seed: int | None,
    ) -> tuple[torch.Tensor, int]:
        tensors = {"hidden": hidden, "attention_mask": attention_mask.to(torch.uint8)}
        if ref_audio is not None and ref_audio.numel():
            tensors["ref_audio"] = ref_audio
        header = {
            "sample_rate": int(sample_rate),
            "gen_latent_len": int(gen_latent_len),
            "nfe": int(nfe),
            "cfg_strength": float(cfg_strength),
            "sway_sampling_coef": sway_sampling_coef,
            "t_grid": t_grid,
            "seed": seed,
        }
        head, out = unpack(self.conn.post("/generate", pack(header, tensors)))
        return out["audio"], int(head["sample_rate"])


class AukDistributedInfer:
    """Thin orchestrator: tokenize here, everything GPU-heavy happens on the two nodes."""

    def __init__(
        self,
        text_encoder_url: str,
        worker_url: str,
        qwen_path: str,
        *,
        timeout: float | None = None,
    ):
        from transformers import Qwen2_5OmniProcessor

        self.text_encoder = RemoteTextEncoder(text_encoder_url, timeout)
        self.worker = RemoteWorker(worker_url, timeout)
        # tokenizer + feature extractor only: no LLM weights, CPU only
        self.processor = Qwen2_5OmniProcessor.from_pretrained(qwen_path)

        info = self.worker.info()
        self.target_sample_rate = info["target_sample_rate"]
        self.downsample_rate = info["downsample_rate"]
        self.latent_dim = info["latent_dim"]
        self.is_flash = info["is_flash"]
        logger.info("Connected to worker %s | %s", worker_url, info)

    def close(self):
        self.text_encoder.conn.close()
        self.worker.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    @torch.no_grad()
    def generate(
        self,
        messages: list,
        *,
        audio: str | tuple[torch.Tensor, int] | None = None,
        gen_seconds: float | None = None,
        nfe: int = 32,
        cfg_strength: float = 2.0,
        sway_sampling_coef: float = -1.0,
        t_grid: list[float] | None = None,
        seed: int | None = None,
    ) -> tuple[torch.Tensor, int]:
        import torchaudio

        wav_path = audio or extract_audio_path(messages, required=False)
        if wav_path is not None:
            if isinstance(wav_path, str):
                ref_audio, sr = torchaudio.load(wav_path)
            else:
                ref_audio, sr = wav_path
            ref_audio = ref_audio.detach().to("cpu", torch.float32)
            if ref_audio.shape[0] > 1:
                ref_audio = ref_audio.mean(dim=0, keepdim=True)
        else:
            # text-only Instruct TTS: empty reference, plus the marker the model was trained with
            ref_audio, sr = torch.zeros(1, 0), self.target_sample_rate
            for m in messages:
                if m.get("role") != "user":
                    continue
                for c in m.get("content", []):
                    if isinstance(c, dict) and c.get("type") == "text" and not c["text"].endswith("|<no_prompt_audio>|"):
                        c["text"] = c["text"] + "|<no_prompt_audio>|"

        ref_latent_len, gen_latent_len = plan_latent_lengths(
            ref_audio.shape[-1], sr, gen_seconds, self.target_sample_rate, self.downsample_rate
        )

        nfe, cfg_strength, sway_sampling_coef, t_grid = apply_flash_recipe(
            self.is_flash, nfe, cfg_strength, sway_sampling_coef, t_grid
        )

        # --- 1 RPC: tokenize locally, let the Qwen node fuse the layers ---
        cond_inputs = CFMEdit.build_cond_inputs([messages], self.processor)
        hidden, context_mask = self.text_encoder.encode(cond_inputs)

        # --- 1 RPC: the worker runs VAE-encode + the whole ODE + VAE-decode ---
        # ``sr`` is the clip's *original* rate — the worker resamples it to the VAE rate.
        return self.worker.generate(
            ref_audio,
            sr,
            hidden,
            context_mask,
            gen_latent_len=gen_latent_len,
            nfe=nfe,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
            t_grid=t_grid,
            seed=seed,
        )


def build_infer(  # convenience used by the CLI
    text_encoder_url: str | None,
    worker_url: str | None,
    *,
    config_path: str | None = None,
    ckpt_path: str | None = None,
    qwen_path: str | None = None,
    device=None,
    dtype: str = "bf16",
    device_map=None,
    weight_dtype: str | None = None,
    timeout: float | None = None,
    root: str | None = None,
):
    """Return a distributed orchestrator when both URLs are given, else a local ``AukInfer``.

    Split mode needs neither a checkpoint nor a config: the orchestrator only tokenizes.
    """
    if not (text_encoder_url and worker_url):
        from auk.infer.infer_auk import AukInfer

        return AukInfer(
            config_path=config_path,
            ckpt_path=ckpt_path,
            device=device,
            dtype=dtype,
            qwen_path=qwen_path,
            device_map=device_map,
            weight_dtype=weight_dtype,
        )

    resolved_qwen = qwen_path or os.path.join(root or ".", "ckpts", "Qwen2.5-Omni-3B")
    return AukDistributedInfer(text_encoder_url, worker_url, resolved_qwen, timeout=timeout)


__all__ = [
    "AukDistributedInfer",
    "RemoteTextEncoder",
    "RemoteWorker",
    "build_infer",
    "plan_latent_lengths",
]
