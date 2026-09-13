"""Numerical parity tests: MLX port vs the PyTorch reference.

Run with the converted weights in place:

    PYTHONPATH=src .venv/bin/python tests/test_parity.py

Every check compares against the torch implementation on identical inputs and
reports a relative error. Tolerances are set just above the float32 accumulation
noise actually observed, so a real regression trips them.
"""

from __future__ import annotations

import logging
import os
import sys

import mlx.core as mx
import numpy as np
import torch


logging.disable(logging.INFO)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

MLX_DIR = os.path.join(ROOT, "ckpts", "mlx")
BASE_DIR = os.path.join(ROOT, "ckpts", "AuK")
QWEN_DIR = os.path.join(ROOT, "ckpts", "Qwen2.5-Omni-3B")

FAILURES: list[str] = []


def check(name: str, got, want, tol: float) -> None:
    a = np.asarray(got, dtype=np.float32)
    b = np.asarray(want, dtype=np.float32)
    if a.shape != b.shape:
        FAILURES.append(f"{name}: shape {a.shape} != {b.shape}")
        print(f"  FAIL {name}: shape {a.shape} != {b.shape}")
        return
    scale = max(float(np.abs(b).max()), 1e-9)
    rel = float(np.abs(a - b).max()) / scale
    status = "ok  " if rel <= tol else "FAIL"
    if rel > tol:
        FAILURES.append(f"{name}: rel {rel:.2e} > tol {tol:.0e}")
    print(f"  {status} {name}: rel={rel:.2e} (tol {tol:.0e})")


# --------------------------------------------------------------------- layers


def test_primitives() -> None:
    """Conv/activation/resample primitives against their torch originals."""
    from auk.model.vae.modules.bigvgan import activations as tact
    from auk.model.vae.modules.bigvgan.alias_free_torch.act import Activation1d as TAct
    from auk.model.vae.modules.bigvgan.alias_free_torch.filter import kaiser_sinc_filter1d as t_kaiser
    from auk.model.vae.modules.bigvgan.alias_free_torch.resample import DownSample1d as TDown
    from auk.model.vae.modules.bigvgan.alias_free_torch.resample import UpSample1d as TUp
    from auk.model.vae.modules.commons.layers import Conv1d as TConv
    from auk.model.vae.modules.commons.layers import ConvTranspose1d as TConvT
    from auk_mlx.layers import Activation1d, Conv1d, ConvTranspose1d, DownSample1d, UpSample1d, kaiser_sinc_filter1d

    print("[primitives]")
    check("kaiser_sinc", kaiser_sinc_filter1d(0.25, 0.3, 12), t_kaiser(0.25, 0.3, 12).reshape(-1), 1e-5)

    torch.manual_seed(0)
    C, T = 8, 64
    x = torch.randn(1, C, T)
    xm = mx.array(x.permute(0, 2, 1).numpy())

    def tr(m):
        """MLX channels-last [B, T, C] -> numpy channels-first, for comparison with torch."""
        return np.array(mx.transpose(m, (0, 2, 1)))

    with torch.no_grad():
        check("UpSample1d", tr(UpSample1d(2, 12)(xm)), TUp(2, 12)(x), 1e-5)
        check("DownSample1d causal", tr(DownSample1d(2, 12, causal=True)(xm)), TDown(2, 12, causal=True)(x), 1e-5)

        al, be = torch.randn(C) * 0.3, torch.randn(C) * 0.3
        ta = TAct(tact.SnakeBeta(C, alpha_logscale=True), causal=True)
        ta.act.alpha.copy_(al)
        ta.act.beta.copy_(be)
        ma = Activation1d(C, True, causal=True)
        ma.act.alpha, ma.act.beta = mx.array(al.numpy()), mx.array(be.numpy())
        check("Activation1d", tr(ma(xm)), ta(x), 1e-5)

        tc = TConv(C, 16, 3, dilation=3, causal=True)
        mc = Conv1d(C, 16, 3, dilation=3, causal=True)
        mc.weight = mx.array(tc.weight.detach().permute(0, 2, 1).numpy())
        mc.bias = mx.array(tc.bias.detach().numpy())
        check("Conv1d causal dil3", tr(mc(xm)), tc(x), 1e-5)

        tt = TConvT(C, 16, kernel_size=10, stride=5, causal=True)
        mt = ConvTranspose1d(C, 16, 10, stride=5, causal=True)
        mt.weight = mx.array(tt.weight.detach().permute(1, 2, 0).numpy())
        mt.bias = mx.array(tt.bias.detach().numpy())
        check("ConvTranspose1d causal", tr(mt(xm)), tt(x), 1e-5)


# ------------------------------------------------------------------------ vae


def test_vae() -> None:
    from omegaconf import OmegaConf

    from auk.model.vae import load_vae_model
    from auk.model.vae.bigvgan_flow_vae import BigVGANFlowVAEConfig
    from auk_mlx.vae import BigVGANFlowVAE, VAEConfig

    print("[vae]")
    kw = OmegaConf.to_container(OmegaConf.load(os.path.join(BASE_DIR, "config.yaml")).model.vae.model_init_kwargs, resolve=True)
    mm = BigVGANFlowVAE(VAEConfig.from_dict(kw))
    mm.load_weights(os.path.join(MLX_DIR, "vae.safetensors"))
    mm.eval()
    tm = load_vae_model(
        "BigVGANFlowVAE", BigVGANFlowVAEConfig.from_dict(kw), os.path.join(BASE_DIR, "vae.safetensors"), map_location="cpu"
    ).eval()

    np.random.seed(0)
    wav = (np.random.randn(1, 1, 24000) * 0.05).astype(np.float32)
    with torch.no_grad():
        stats = tm.audio_encoder(torch.from_numpy(wav))
        mean = stats[:, :64]
        t_lat = ((mean.transpose(1, 2).float() - tm.global_mean.float()) / torch.sqrt(tm.global_log_std.float())).numpy()
    m_lat = np.array(mm.encode(mx.array(wav.transpose(0, 2, 1))))
    check("vae encode", m_lat, t_lat, 1e-4)

    with torch.no_grad():
        t_wav = tm.inference_from_latents(tm.denormalize(torch.from_numpy(t_lat)).permute(0, 2, 1)).numpy()
    m_wav = np.array(mm.decode(mx.array(t_lat))).transpose(0, 2, 1)
    check("vae decode", m_wav, t_wav, 1e-4)


# ------------------------------------------------------------------------ dit


def _load_dits():
    from omegaconf import OmegaConf
    from safetensors.torch import load_file

    from auk.model import Flux2Edit as TFlux
    from auk_mlx.dit import DiTConfig
    from auk_mlx.dit import Flux2Edit as MFlux

    arch = OmegaConf.to_container(OmegaConf.load(os.path.join(BASE_DIR, "config.yaml")).model.arch, resolve=True)
    marc = dict(arch)
    marc["latent_dim"] = 64
    fusion = mx.load(os.path.join(MLX_DIR, "fusion_base.safetensors"))
    mm = MFlux(DiTConfig.from_dict(marc), inv_freq=np.array(fusion["inv_freq"]))
    mm.load_weights(os.path.join(MLX_DIR, "dit_base.safetensors"))
    mm.eval()

    tarc = dict(arch)
    tarc["attn_backend"] = "torch"
    tarc["checkpoint_activations"] = False
    tm = TFlux(**tarc, latent_dim=64).eval()
    sd = load_file(os.path.join(BASE_DIR, "auk_base.safetensors"), device="cpu")
    tm.load_state_dict({k[len("transformer.") :]: v for k, v in sd.items() if k.startswith("transformer.")}, strict=False)
    return mm, tm.to(torch.float32)


def test_dit() -> None:
    print("[dit]")
    mm, tm = _load_dits()
    np.random.seed(0)
    x = np.random.randn(1, 40, 64).astype(np.float32)
    ref = np.random.randn(1, 20, 64).astype(np.float32)
    txt = (np.random.randn(1, 15, 2048) * 0.5).astype(np.float32)
    t = np.array([0.3], dtype=np.float32)

    with torch.no_grad():
        ot = tm(
            x=torch.from_numpy(x),
            text=torch.from_numpy(txt),
            time=torch.from_numpy(t),
            ref=torch.from_numpy(ref),
            cache=False,
        ).numpy()
    om = np.array(mm(x=mx.array(x), text=mx.array(txt), t=mx.array(t), ref=mx.array(ref), cache=False))
    check("dit forward", om, ot, 1e-4)

    tm.clear_cache()
    mm.clear_cache()
    with torch.no_grad():
        ot2 = tm(
            x=torch.from_numpy(x),
            text=torch.from_numpy(txt),
            time=torch.from_numpy(t),
            ref=torch.from_numpy(ref),
            cfg_infer=True,
            cache=True,
        ).numpy()
    om2 = np.array(mm(x=mx.array(x), text=mx.array(txt), t=mx.array(t), ref=mx.array(ref), cfg_infer=True, cache=True))
    check("dit cfg forward", om2, ot2, 1e-4)


def test_ode() -> None:
    """A full 16-step CFG Euler trajectory -- catches drift a single step hides."""
    print("[ode]")
    mm, tm = _load_dits()
    np.random.seed(7)
    ref = (np.random.randn(1, 100, 64) * 0.8).astype(np.float32)
    txt = (np.random.randn(1, 40, 2048) * 0.7).astype(np.float32)
    y0 = np.random.randn(1, 100, 64).astype(np.float32)

    nfe, cfg, sway = 16, 2.0, -1.0
    t = np.linspace(0, 1, nfe + 1, dtype=np.float32)
    t = t + sway * (np.cos(np.pi / 2 * t) - 1 + t)

    yt = torch.from_numpy(y0.copy())
    tm.clear_cache()
    with torch.no_grad():
        for i in range(nfe):
            p = tm(
                x=yt,
                text=torch.from_numpy(txt),
                time=torch.tensor([t[i]]),
                ref=torch.from_numpy(ref),
                cfg_infer=True,
                cache=True,
            )
            vc, vu = p.chunk(2, dim=0)
            yt = yt + (vc + (vc - vu) * cfg) * float(t[i + 1] - t[i])
    tm.clear_cache()

    ym = mx.array(y0.copy())
    mm.clear_cache()
    for i in range(nfe):
        p = mm(
            x=ym,
            text=mx.array(txt),
            t=mx.array(np.array([t[i]], dtype=np.float32)),
            ref=mx.array(ref),
            cfg_infer=True,
            cache=True,
        )
        vc, vu = p[0:1], p[1:2]
        ym = ym + (vc + (vc - vu) * cfg) * float(t[i + 1] - t[i])
        mx.eval(ym)
    mm.clear_cache()
    check("16-step cfg ode", np.array(ym), yt.numpy(), 1e-3)


# -------------------------------------------------------------------- thinker


def test_conditioning() -> None:
    """Fused text(+audio) conditioning -- the tensor the DiT is actually driven by."""
    import copy

    from auk.infer.infer_auk import AukInfer
    from auk_mlx.infer import AukMLX

    print("[conditioning]")
    ref_wav = os.path.join("/tmp", "_parity_ref.wav")
    if not os.path.isfile(ref_wav):
        import soundfile as sf

        src = os.path.join(ROOT, "assets", "demo-input-audio", "zero-shot-tts", "ref.wav")
        import torchaudio

        w, sr = torchaudio.load(src)
        w = w.mean(0, keepdim=True)
        if sr != 24000:
            w = torchaudio.transforms.Resample(sr, 24000)(w)
        sf.write(ref_wav, w.numpy().reshape(-1)[: 24000 * 4], 24000)

    te = AukInfer(
        os.path.join(BASE_DIR, "config.yaml"),
        os.path.join(BASE_DIR, "auk_base.safetensors"),
        device="cpu",
        dtype="fp32",
        qwen_path=QWEN_DIR,
    )
    me = AukMLX(MLX_DIR, os.path.join(BASE_DIR, "config.yaml"), QWEN_DIR)
    instr = "Say the following with the same voice: 'Hello, this is a test of the MLX port.'"

    for tag, msgs in (
        ("text only", [{"role": "user", "content": [{"type": "text", "text": instr + "|<no_prompt_audio>|"}]}]),
        ("text+audio", [{"role": "user", "content": [{"type": "text", "text": instr}, {"type": "audio", "audio": ref_wav}]}]),
    ):
        ci = te.model.build_cond_inputs([copy.deepcopy(msgs)], te.model.text_processor)
        with torch.no_grad():
            tb, _ = te.model.encode_text(ci, "cpu")
        ma = me.encode_text(copy.deepcopy(msgs))
        mx.eval(ma)
        check(f"conditioning {tag}", np.array(ma), tb.numpy(), 1e-4)


def main() -> None:
    test_primitives()
    test_vae()
    test_dit()
    test_ode()
    test_conditioning()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S):")
        for f in FAILURES:
            print("  -", f)
        sys.exit(1)
    print("all parity checks passed")


if __name__ == "__main__":
    main()
