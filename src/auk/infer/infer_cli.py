from __future__ import annotations

import argparse
import os

from auk.infer.infer_auk import get_gen_duration, save_audio
from auk.serve.client import build_infer


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
        "--device_map",
        default=None,
        help=(
            "Split the three sub-models across devices: 'auto' spreads them over the visible GPUs "
            "(dit=cuda:0, qwen=cuda:1, vae=cuda:1), or set them explicitly, e.g. "
            "'dit=cuda:0,qwen=cuda:1,vae=cpu'. Default: everything on --device."
        ),
    )
    p.add_argument(
        "--weight_dtype",
        choices=["fp32", "bf16", "fp16"],
        default=None,
        help=(
            "Resident weight dtype for the DiT + Qwen (default fp32 = current behaviour). "
            "'bf16' roughly halves resident VRAM; sampling already computes in bf16 under autocast. "
            "The VAE is always fp32 (it runs with autocast disabled)."
        ),
    )

    p.add_argument(
        "--text_encoder_url",
        default=None,
        help="remote Qwen+layer-fusion node (with --worker_url, runs in split mode; this process needs no GPU)",
    )
    p.add_argument("--worker_url", default=None, help="remote VAE+DiT+ODE node")
    p.add_argument("--timeout", type=float, default=None, help="per-request timeout in seconds (split mode)")

    p.add_argument("--output", "-o", required=True, help="output wav path")
    return p


def main():
    args = build_parser().parse_args()

    # Instruct TTS (no reference audio) can't match the source length, so it needs an explicit target
    if not args.audio and not args.gen_seconds and not args.gen_text:
        raise SystemExit("Without --audio (Instruct TTS), set a target length via --gen_seconds or --gen_text.")

    split = bool(args.text_encoder_url or args.worker_url)
    if split and not (args.text_encoder_url and args.worker_url):
        raise SystemExit("Split mode needs BOTH --text_encoder_url and --worker_url (or neither).")

    config_path = None
    if not split:
        # config.yaml ships next to --ckpt (release dirs bundle their own)
        config_path = os.path.join(os.path.dirname(os.path.abspath(args.ckpt)), "config.yaml")
        if not os.path.isfile(config_path):
            raise SystemExit(f"config.yaml not found next to --ckpt: {config_path}")
    else:
        # the orchestrator only tokenizes; it still needs the Qwen tokenizer, but no weights
        qwen_dir = args.qwen_path or os.path.join(ROOT, "ckpts", "Qwen2.5-Omni-3B")
        if not os.path.isdir(qwen_dir):
            raise SystemExit(f"Split mode needs the Qwen tokenizer at {qwen_dir} — pass --qwen_path to point at a snapshot.")
    t_grid = [float(x) for x in args.t_grid.split(",")] if args.t_grid else None

    engine = build_infer(
        args.text_encoder_url,
        args.worker_url,
        config_path=config_path,
        ckpt_path=args.ckpt,
        qwen_path=args.qwen_path,
        device=args.device,
        dtype=args.dtype,
        device_map=args.device_map,
        weight_dtype=args.weight_dtype,
        timeout=args.timeout,
        root=ROOT,
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
