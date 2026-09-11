"""Tests for the split (two-node) deployment.

Runs on CPU with tiny random weights — it validates the *plumbing and the seams*, not audio
quality:

* the wire codec (frame format, dtype preservation, malformed input)
* ``CFMEdit`` without a text encoder + ``sample_from_embeds`` (the seam that lets the ODE loop
  stay on the worker instead of becoming one RPC per solver step)
* ``CFMEdit.sample`` still working end-to-end with a local text encoder (no regression)
* the HTTP nodes: routing, keep-alive client, error propagation, text-only path

    python tests/test_serve_split.py
"""

from __future__ import annotations

import os
import sys
import threading
import time
import traceback
import types
import urllib.request

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from auk.model import CFMEdit, Flux2Edit  # noqa: E402
from auk.serve.codec import pack, unpack  # noqa: E402
from auk.serve.server import serve  # noqa: E402

LATENT_DIM = 8
TEXT_DIM = 16


# --------------------------------------------------------------------------- codec


def test_codec_roundtrip():
    header = {"gen_latent_len": 123, "t_grid": [0.0, 0.5, 1.0], "seed": 7, "sway": -1.0}
    tensors = {
        "hidden": torch.randn(1, 40, 2048),
        "attention_mask": torch.ones(1, 40, dtype=torch.uint8),
        "audio": torch.randn(1, 24000),
    }
    head, out = unpack(pack(header, tensors))
    assert head == header
    for name, value in tensors.items():
        assert torch.equal(out[name], value), name
        assert out[name].dtype == value.dtype, (name, out[name].dtype)


def test_codec_rejects_malformed():
    for bad in (b"", b"XXXX" + b"\x00" * 40, b"AUK1\x00\x00\x00\x05ab", b"AUK1\xff\xff\xff\xffx"):
        try:
            unpack(bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad!r}")


# --------------------------------------------------------------------------- model seams


class _FakeTextEncoder(torch.nn.Module):
    """Minimal stand-in with the interface ``CFMEdit`` expects of the LLM."""

    def __init__(self, num_layers: int, hidden: int):
        super().__init__()
        self._num_layers = num_layers
        self._hidden = hidden
        self.config = types.SimpleNamespace(text_config=types.SimpleNamespace(num_hidden_layers=num_layers))
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids=None, attention_mask=None, output_hidden_states=False, **_):
        batch, seq = input_ids.shape
        states = tuple(torch.randn(batch, seq, self._hidden) for _ in range(self._num_layers + 1))
        return types.SimpleNamespace(hidden_states=states)


def _tiny_cfm(num_layers: int = 3):
    transformer = Flux2Edit(
        dim=32,
        heads=4,
        dim_head=8,
        ff_mult=2,
        latent_dim=LATENT_DIM,
        text_hidden_dim=TEXT_DIM,
        num_layers=1,
        num_single_layers=1,
        attn_backend="torch",
        attn_mask_enabled=True,
    )
    return CFMEdit(
        transformer=transformer,
        text_encoder=None,
        text_processor=None,
        num_channels=LATENT_DIM,
        num_text_layers=num_layers,
    )


def test_cfm_edit_without_text_encoder():
    """The worker never loads the LLM: embeddings arrive pre-fused."""
    model = _tiny_cfm().eval()
    assert model.text_encoder is None
    assert model.layer_weights.numel() == 3  # sized from num_text_layers, so the ckpt loads

    cond = torch.randn(1, 5, LATENT_DIM)
    text_embeds = torch.randn(1, 7, TEXT_DIM)
    context_mask = torch.ones(1, 7, dtype=torch.bool)
    out, _ = model.sample_from_embeds(
        cond,
        text_embeds,
        context_mask,
        torch.tensor([9]),
        lens=torch.tensor([5]),
        steps=2,
        cfg_strength=2.0,
        seed=0,
    )
    assert out.shape == (1, 9, LATENT_DIM), out.shape
    # the text-projection cache is module state — it must not survive the call
    assert model.transformer.text_cond is None, "clear_cache() should have run"


def test_cfm_edit_sample_with_local_text_encoder():
    """Regression guard: the local (single-process) path still works unchanged."""
    num_layers = 3
    model = _tiny_cfm(num_layers)
    model.text_encoder = _FakeTextEncoder(num_layers, TEXT_DIM)
    model.eval()

    cond = torch.randn(1, 4, LATENT_DIM)
    cond_inputs = {
        "input_ids": torch.randint(0, 50, (1, 6)),
        "attention_mask": torch.ones(1, 6, dtype=torch.long),
    }
    out, _ = model.sample(cond, cond_inputs, torch.tensor([8]), lens=torch.tensor([4]), steps=2, cfg_strength=2.0, seed=0)
    assert out.shape == (1, 8, LATENT_DIM), out.shape


def test_place_submodules_without_text_encoder():
    """Regression: a worker with no LLM must not try to move it.

    ``AukInfer(load_qwen=False)`` leaves ``text_encoder`` as ``None``; the placement step used to
    call ``.to()`` on it unconditionally.
    """
    from auk.infer.infer_auk import AukInfer

    engine = AukInfer.__new__(AukInfer)
    engine.dit_device = engine.qwen_device = engine.vae_device = "cpu"

    model = _tiny_cfm()
    engine._place_submodules(model)
    assert model.layer_weights.data.device.type == "cpu"

    # and with an LLM present, the fusion weights follow the encoder instead
    model2 = _tiny_cfm(num_layers=3)
    model2.text_encoder = _FakeTextEncoder(3, TEXT_DIM)
    engine._place_submodules(model2)
    assert next(model2.text_encoder.parameters()).device.type == "cpu"


def test_flash_recipe_pins_sampling():
    from auk.infer.infer_auk import apply_flash_recipe

    nfe, cfg, sway, grid = apply_flash_recipe(True, 32, 2.0, -1.0, [0.0, 1.0])
    assert (nfe, cfg, sway) == (4, 0.0, None) and len(grid) == 5
    assert apply_flash_recipe(False, 32, 2.0, -1.0, [0.0, 1.0]) == (32, 2.0, -1.0, [0.0, 1.0])


# --------------------------------------------------------------------------- http nodes


class _FakeTextEncoderNode:
    def encode(self, cond_inputs):
        assert "input_ids" in cond_inputs
        seq = cond_inputs["input_ids"].shape[1]
        return pack(
            {"shape": [1, seq, TEXT_DIM]},
            {"hidden": torch.randn(1, seq, TEXT_DIM), "attention_mask": torch.ones(1, seq, dtype=torch.uint8)},
        )


class _FakeWorkerNode:
    def info(self):
        return {
            "target_sample_rate": 24000,
            "downsample_rate": 480,
            "latent_dim": LATENT_DIM,
            "is_flash": False,
            "device": "cpu",
        }

    def generate(
        self,
        ref_audio,
        hidden,
        attention_mask,
        *,
        gen_latent_len,
        nfe,
        cfg_strength,
        sway_sampling_coef,
        t_grid,
        seed,
        sample_rate,
    ):
        assert hidden.shape[-1] == TEXT_DIM
        assert attention_mask.dtype == torch.uint8, "masks travel as uint8"
        assert isinstance(gen_latent_len, int) and gen_latent_len > 0
        assert (nfe, cfg_strength, seed) == (8, 2.0, 42)
        return torch.randn(1, gen_latent_len * 480), sample_rate


def _serve_bg(node, kind, port):
    threading.Thread(target=serve, args=(node, kind, "127.0.0.1", port), daemon=True).start()
    for _ in range(200):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=0.5).read()
            return
        except Exception:  # noqa: BLE001 - retry until the socket accepts
            time.sleep(0.05)
    raise RuntimeError(f"{kind} node never came up on port {port}")


def test_http_roundtrip():
    from auk.serve.client import RemoteTextEncoder, RemoteWorker

    _serve_bg(_FakeTextEncoderNode(), "text-encoder", 8801)
    _serve_bg(_FakeWorkerNode(), "worker", 8802)

    encoder = RemoteTextEncoder("http://127.0.0.1:8801")
    worker = RemoteWorker("http://127.0.0.1:8802")

    info = worker.info()
    assert (info["target_sample_rate"], info["downsample_rate"]) == (24000, 480), info

    cond = {"input_ids": torch.randint(0, 100, (1, 17)), "attention_mask": torch.ones(1, 17, dtype=torch.long)}
    hidden, mask = encoder.encode(cond)
    assert hidden.shape == (1, 17, TEXT_DIM)
    assert mask.dtype is torch.bool and mask.shape == (1, 17)

    audio, sr = worker.generate(
        torch.randn(1, 24000),
        24000,
        hidden,
        mask,
        gen_latent_len=50,
        nfe=8,
        cfg_strength=2.0,
        sway_sampling_coef=-1.0,
        t_grid=None,
        seed=42,
    )
    assert audio.shape == (1, 50 * 480) and sr == 24000

    # text-only Instruct TTS: no ref_audio key is sent at all
    audio2, _ = worker.generate(
        None,
        24000,
        hidden,
        mask,
        gen_latent_len=10,
        nfe=8,
        cfg_strength=2.0,
        sway_sampling_coef=-1.0,
        t_grid=None,
        seed=42,
    )
    assert audio2.shape == (1, 10 * 480)

    # keep-alive: many requests over one connection
    for _ in range(5):
        encoder.encode(cond)
        worker.info()

    try:
        worker.conn.post("/nope", pack({}, {}))
    except RuntimeError as exc:
        assert "404" in str(exc), exc
    else:
        raise AssertionError("expected a 404 from an unknown route")


def test_plan_latent_lengths():
    from auk.serve.client import plan_latent_lengths

    # 1 s of 24 kHz audio -> 24000 // 480 = 50 latent frames
    assert plan_latent_lengths(24000, 24000, None, 24000, 480) == (50, 50)
    # same clip at 16 kHz must end up with the same latent length once resampled
    assert plan_latent_lengths(16000, 16000, None, 24000, 480) == (50, 50)
    # an explicit duration overrides the reference length
    assert plan_latent_lengths(24000, 24000, 2.0, 24000, 480) == (50, 100)
    # no reference at all
    assert plan_latent_lengths(0, 24000, 3.0, 24000, 480) == (0, 150)
    assert plan_latent_lengths(0, 24000, None, 24000, 480) == (0, 1)


class _FakeProcessor:
    """Keeps ``CFMEdit.build_cond_inputs`` working without the real Qwen snapshot."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "prompt"

    def __call__(self, text=None, padding=False, return_tensors=None, **_):
        return {
            "input_ids": torch.randint(0, 50, (1, 8)),
            "attention_mask": torch.ones(1, 8, dtype=torch.long),
        }


def test_client_forwards_original_sample_rate():
    """Regression: the client must not claim a 16 kHz clip is already 24 kHz.

    Claiming the target rate makes the worker skip resampling, which silently changes both the
    generated duration and the pitch.
    """
    from auk.serve.client import AukDistributedInfer, RemoteTextEncoder, RemoteWorker

    captured: dict = {}

    class _CaptureWorker(_FakeWorkerNode):
        def generate(
            self,
            ref_audio,
            hidden,
            attention_mask,
            *,
            gen_latent_len,
            nfe,
            cfg_strength,
            sway_sampling_coef,
            t_grid,
            seed,
            sample_rate,
        ):
            captured.update(
                sr=sample_rate,
                gen=gen_latent_len,
                ref_len=0 if ref_audio is None else ref_audio.shape[-1],
            )
            return torch.randn(1, 480), sample_rate

    _serve_bg(_FakeTextEncoderNode(), "text-encoder", 8811)
    _serve_bg(_CaptureWorker(), "worker", 8812)

    # build the orchestrator without __init__ so the test needs no Qwen snapshot
    engine = AukDistributedInfer.__new__(AukDistributedInfer)
    engine.text_encoder = RemoteTextEncoder("http://127.0.0.1:8811")
    engine.worker = RemoteWorker("http://127.0.0.1:8812")
    engine.processor = _FakeProcessor()
    engine.target_sample_rate = 24000
    engine.downsample_rate = 480
    engine.latent_dim = LATENT_DIM
    engine.is_flash = False

    messages = [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]
    engine.generate(messages, audio=(torch.randn(1, 16000), 16000), gen_seconds=None)

    assert captured["sr"] == 16000, captured
    assert captured["ref_len"] == 16000, captured
    assert captured["gen"] == 50, captured  # 1 s at 24 kHz -> 50 latent frames


def main() -> int:
    """Run every ``test_*`` in this module without requiring pytest.

    Keeps the suite runnable in a bare checkout: ``python tests/test_serve_split.py``.
    """
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = []
    for test in tests:
        try:
            test()
        except Exception:  # report every failure, don't stop at the first one
            failed.append(test.__name__)
            print(f"  FAIL  {test.__name__}", file=sys.stderr)
            traceback.print_exc()
        else:
            print(f"  PASS  {test.__name__}")
    print(f"\n{len(tests) - len(failed)} passed, {len(failed)} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
