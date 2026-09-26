from __future__ import annotations

import argparse
import logging
import sys

from auk.serve.nodes import TextEncoderNode, WorkerNode, default_config_path
from auk.serve.server import serve


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m auk.serve",
        description="AuK split deployment: run one node per process.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = p.add_subparsers(dest="node", required=True)

    te = sub.add_parser("text-encoder", help="Qwen2.5-Omni Thinker + ELMo layer fusion")
    te.add_argument("--qwen_path", required=True, help="Qwen2.5-Omni-3B snapshot directory")
    te.add_argument(
        "--ckpt",
        required=True,
        help="AuK checkpoint — needed for the layer-fusion weights (layer_weights / layer_scale)",
    )
    te.add_argument("--device", default=None)
    te.add_argument("--weight_dtype", choices=["fp32", "bf16", "fp16"], default="bf16")
    te.add_argument("--host", default="0.0.0.0")
    te.add_argument("--port", type=int, default=8001)

    wk = sub.add_parser("worker", help="VAE + DiT + ODE loop (the compute bottleneck)")
    wk.add_argument("--ckpt", required=True)
    wk.add_argument("--config", default=None, help="config.yaml (default: next to --ckpt)")
    wk.add_argument("--qwen_path", default=None, help="only needed to read num_hidden_layers")
    wk.add_argument("--device", default=None)
    wk.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="bf16", help="autocast dtype")
    wk.add_argument("--weight_dtype", choices=["fp32", "bf16", "fp16"], default=None)
    wk.add_argument(
        "--device_map",
        default=None,
        help="spread VAE/DiT across this machine's GPUs, e.g. 'auto' or 'dit=cuda:0,vae=cuda:1'",
    )
    wk.add_argument("--host", default="0.0.0.0")
    wk.add_argument("--port", type=int, default=8002)
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = build_parser().parse_args(argv)

    if args.node == "text-encoder":
        node = TextEncoderNode(
            qwen_path=args.qwen_path,
            ckpt_path=args.ckpt,
            device=args.device,
            weight_dtype=args.weight_dtype,
        )
        serve(node, "text-encoder", args.host, args.port)
    else:
        config_path = args.config or default_config_path(args.ckpt)
        node = WorkerNode(
            config_path=config_path,
            ckpt_path=args.ckpt,
            qwen_path=args.qwen_path,
            device=args.device,
            dtype=args.dtype,
            weight_dtype=args.weight_dtype,
            device_map=args.device_map,
        )
        serve(node, "worker", args.host, args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
