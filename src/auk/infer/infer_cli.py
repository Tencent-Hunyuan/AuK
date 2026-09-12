from __future__ import annotations

import argparse
import os

from auk.infer.infer_auk import AukInfer, get_gen_duration, save_audio


# repo root: src/auk/infer/infer_cli.py -> up 3 levels
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="AuK message-driven inference CLI (TTS / speech editing / SE / … by instruction)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--ckpt",
        default=os.path.join(ROOT, "ckpts", "AuK", "auk_base.safetensors"),
        help="model weights (.safetensors clean export, or a training .pt with ema_model_state_dict)",
    )
    p.add_argument("--qwen_path", default=None, help="override Qwen2.5-Omni-3B snapshot path")

    # inputs (task carried by the instruction)
    p.add_argument(
        "--audio",
        default=None,
        help="input audio wav (VAE-encoded to the reference/prefix latent); omit for Instruct TTS (no reference)",
    )
    p.add_argument("--instruction", required=True, help="natural-language task instruction")

    # duration control (priority: gen_seconds > ref_text+gen_text > match source length)
    p.add_argument("--gen_seconds", type=float, default=None, help="target generated duration in seconds")
    p.add_argument("--ref_text", default=None, help="transcript of --audio (with --gen_text, estimates duration)")
    p.add_argument("--gen_text", default=None, help="target text (with --ref_text, estimates duration)")

    # sampling knobs — used as-is for base AuK; ignored for AuK-Flash (locked by config name)
    p.add_argument("--nfe", type=int, default=32, help="number of function evaluations (ODE steps)")
    p.add_argument("--cfg", type=float, default=2.0, help="classifier-free guidance strength")
    p.add_argument("--sway", type=float, default=-1.0, help="sway sampling coefficient")
    p.add_argument(
        "--t_grid",
        type=str,
        default=None,
        help="explicit comma-separated sampling times, e.g. '0.0,0.076,0.293,0.617,1.0'; overrides --nfe/--sway",
    )
    p.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="bf16", help="autocast dtype")
    p.add_argument("--seed", type=int, default=None, help="random seed")
    p.add_argument("--device", default=None, help="cuda / cuda:0 / cpu (auto if unset)")
    p.add_argument(
        "--cpu_offload",
        action="store_true",
        help="offload the Qwen text encoder and DiT to CPU when idle; keep the VAE on CUDA",
    )

    p.add_argument("--output", "-o", required=True, help="output wav path")
    return p


def main():
    args = build_parser().parse_args()

    # Instruct TTS (no reference audio) can't match the source length, so it needs an explicit target
    if not args.audio and not args.gen_seconds and not args.gen_text:
        raise SystemExit("Without --audio (Instruct TTS), set a target length via --gen_seconds or --gen_text.")

    # config: config.yaml ships next to --ckpt (release dirs bundle their own)
    config_path = os.path.join(os.path.dirname(os.path.abspath(args.ckpt)), "config.yaml")
    if not os.path.isfile(config_path):
        raise SystemExit(f"config.yaml not found next to --ckpt: {config_path}")
    t_grid = [float(x) for x in args.t_grid.split(",")] if args.t_grid else None

    engine = AukInfer(
        config_path=config_path,
        ckpt_path=args.ckpt,
        device=args.device,
        dtype=args.dtype,
        qwen_path=args.qwen_path,
        cpu_offload=args.cpu_offload,
    )

    # no --audio => Instruct TTS; generate() appends the |<no_prompt_audio>| marker for the text-only turn
    content = [{"type": "text", "text": args.instruction}]
    if args.audio:
        content.append({"type": "audio", "audio": args.audio})
    messages = [{"role": "user", "content": content}]

    gen_seconds = get_gen_duration(
        audio=args.audio,
        ref_text=args.ref_text,
        gen_text=args.gen_text,
        gen_seconds=args.gen_seconds,
    )
    audio, sr = engine.generate(
        messages,
        audio=args.audio,
        gen_seconds=gen_seconds,
        nfe=args.nfe,
        cfg_strength=args.cfg,
        sway_sampling_coef=args.sway,
        t_grid=t_grid,
        seed=args.seed,
    )

    save_audio(audio, sr, args.output)
    print(f"Saved output -> {args.output}  ({audio.shape[-1] / sr:.2f}s @ {sr} Hz)")


if __name__ == "__main__":
    main()
