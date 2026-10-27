"""Merge LoRA weights back into the backbone and export a plain safetensors file.

The inference loader builds a ``Flux2Edit`` without LoRA layers, so pointing it
at a training checkpoint would silently ignore every ``lora_*`` key and keep the
frozen base weights. This script folds the adapters into the base weights
(``W' = W + B @ A * scaling``) and drops the text encoder, matching the layout of
``auk_base.safetensors``.

Usage:
    python -m auk.train.merge_lora --ckpt <model_last.pt> --out <merged.safetensors>
"""

from __future__ import annotations

import argparse

import torch
from safetensors.torch import save_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Training checkpoint (model_last.pt)")
    ap.add_argument("--out", required=True, help="Output safetensors path")
    ap.add_argument("--ema-key", default="ema_model_state_dict", help="Checkpoint field holding the weights")
    ap.add_argument("--ema-prefix", default="ema_model.", help="Prefix stripped from weight keys")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = ckpt[args.ema_key]
    sd = {k.replace(args.ema_prefix, ""): v for k, v in state.items() if k not in ("initted", "step")}

    lora = {}
    for k in list(sd.keys()):
        for suf in ("lora_A", "lora_B", "lora_scaling"):
            if k.endswith("." + suf):
                prefix = k[: -len(suf) - 1]
                lora.setdefault(prefix, {})[suf] = sd.pop(k)
                break

    for prefix, m in lora.items():
        a = m["lora_A"]
        b = m["lora_B"]
        scaling = m.get("lora_scaling", torch.tensor(1.0))
        delta = (b @ a) * scaling.to(b.dtype)
        w = sd[prefix + ".weight"]
        sd[prefix + ".weight"] = (w + delta.to(w.dtype)).contiguous()

    sd = {k: v for k, v in sd.items() if not k.startswith("text_encoder.")}
    save_file(sd, args.out)
    print(f"saved {args.out} | {len(sd)} tensors | merged lora layers: {len(lora)}")


if __name__ == "__main__":
    main()
