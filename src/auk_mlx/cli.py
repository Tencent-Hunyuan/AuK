"""CLI for MLX AuK inference.

    auk-mlx-infer --instruction "..." --audio ref.wav --output out.wav --gen_seconds 6

Weights must be converted first (see auk_mlx/convert.py).
"""

from __future__ import annotations

import argparse
import os
import time


ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="AuK MLX inference (Apple Silicon)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mlx_dir", default=os.path.join(ROOT, "ckpts", "mlx"), help="converted MLX weights")
    p.add_argument("--config", default=None, help="config.yaml (defaults to the base release next to ckpts/)")
    p.add_argument("--qwen_path", default=os.path.join(ROOT, "ckpts", "Qwen2.5-Omni-3B"), help="Qwen snapshot (processor)")
    p.add_argument("--flash", action="store_true", help="use the AuK-Flash config/weights (4-step, CFG off)")
    p.add_argument(
        "--bits",
        type=int,
        choices=[4, 8],
        default=None,
        help="quantize the DiT and Thinker to 4 or 8 bits (default: fp32). The VAE is convolutional and stays fp32.",
    )
    p.add_argument("--group_size", type=int, default=64, help="quantization group size")
    p.add_argument(
        "--sequential",
        action="store_true",
        help="hold one stack in memory at a time (Thinker, then DiT); the MLX analogue of CUDA cpu_offload",
    )

    p.add_argument("--instruction", required=True, help="natural-language task instruction")
    p.add_argument("--audio", default=None, help="reference/source wav at 24 kHz; omit for Instruct TTS")
    p.add_argument("--gen_seconds", type=float, default=None, help="target generated duration in seconds")

    p.add_argument("--nfe", type=int, default=32, help="ODE steps (ignored for Flash)")
    p.add_argument("--cfg", type=float, default=2.0, help="classifier-free guidance strength")
    p.add_argument("--sway", type=float, default=-1.0, help="sway sampling coefficient")
    p.add_argument("--seed", type=int, default=None, help="random seed")

    p.add_argument("--output", "-o", required=True, help="output wav path")
    return p


def main() -> None:
    args = build_parser().parse_args()

    if not args.audio and not args.gen_seconds:
        raise SystemExit("Without --audio (Instruct TTS), set a target length via --gen_seconds.")

    config = args.config or os.path.join(ROOT, "ckpts", "AuK-Flash" if args.flash else "AuK", "config.yaml")
    if not os.path.isfile(config):
        raise SystemExit(f"config.yaml not found: {config}")

    import soundfile as sf

    # check the input before paying for a ~6 s model load
    gen_seconds = args.gen_seconds
    if args.audio:
        if not os.path.isfile(args.audio):
            raise SystemExit(f"error: no such file: {args.audio}")
        info = sf.info(args.audio)
        # any rate / channel count is fine; the engine downmixes and resamples
        if gen_seconds is None:
            gen_seconds = info.frames / info.samplerate

    from auk_mlx.infer import AukMLX, GenerateOptions

    t0 = time.time()
    engine = AukMLX(args.mlx_dir, config, args.qwen_path, bits=args.bits, group_size=args.group_size, sequential=args.sequential)
    prec = f"{args.bits}-bit" if args.bits else "fp32"
    mode = ", sequential" if args.sequential else ""
    print(f"loaded in {time.time() - t0:.1f}s (flash={engine.is_flash}, {prec}{mode})")

    t0 = time.time()
    try:
        audio, sr = engine.generate(
            args.instruction,
            audio_path=args.audio,
            opts=GenerateOptions(
                gen_seconds=gen_seconds,
                nfe=args.nfe,
                cfg_strength=args.cfg,
                sway_sampling_coef=args.sway,
                seed=args.seed,
            ),
        )
    except ValueError as e:
        raise SystemExit(f"error: {e}")
    dt = time.time() - t0

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    sf.write(args.output, audio, sr)
    dur = len(audio) / sr
    print(f"Saved {args.output}  ({dur:.2f}s @ {sr} Hz) in {dt:.1f}s  RTF={dt / dur:.2f}")


if __name__ == "__main__":
    main()
