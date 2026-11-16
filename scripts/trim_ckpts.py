"""Free disk space after MLX conversion by deleting weights inference no longer reads.

The conversion step (`python -m auk_mlx.convert ...`) reads the torch originals
and writes `ckpts/mlx/`. After it finishes, inference reads only:

  - ckpts/<variant>/config.yaml            architecture (kept)
  - ckpts/mlx/...                          converted weights (kept)
  - Qwen2.5-Omni-3B processor files        tokenizer, chat template, mel front end (kept)

Everything else is dead weight: the torch DiT/VAE safetensors (~13 GB), the
Qwen shards and spk_dict.pt (~11 GB). Running only 8-bit (`--bits 8`) also
makes the fp32 MLX copies (~25 GB) redundant. This was verified by running
`auk-mlx-infer` against a trimmed tree with identical output.

Deleting the torch originals means a future re-conversion needs a fresh
download — that is the only thing lost.

Usage:
    python scripts/trim_ckpts.py                     # report only (default)
    python scripts/trim_ckpts.py --apply             # delete torch originals
    python scripts/trim_ckpts.py --apply --fp32-mlx  # also drop fp32 MLX weights
"""

from __future__ import annotations

import argparse
import glob
import os


ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# Torch checkpoints: read by the converter, never by inference.
TORCH_FILES = [
    "AuK/auk_base.safetensors",
    "AuK/vae.safetensors",
    "AuK-Flash/auk_flash.safetensors",
    "AuK-Flash/vae.safetensors",
    "Qwen2.5-Omni-3B/spk_dict.pt",
]
TORCH_GLOBS = ["Qwen2.5-Omni-3B/model-*.safetensors", "Qwen2.5-Omni-3B/model.safetensors.index.json"]

# fp32 MLX weights: redundant once inference always passes --bits 8.
FP32_MLX_FILES = [
    "mlx/dit_base.safetensors",
    "mlx/dit_flash.safetensors",
    "mlx/thinker/thinker.safetensors",
]


def _gb(path: str) -> float:
    return os.path.getsize(path) / 1e9


def collect(ckpts: str, fp32_mlx: bool) -> list[str]:
    paths: list[str] = []
    for rel in TORCH_FILES + [p for pattern in TORCH_GLOBS for p in glob.glob(os.path.join(ckpts, pattern))]:
        full = os.path.join(ckpts, rel)
        if os.path.isfile(full):
            paths.append(full)
    if fp32_mlx:
        for rel in FP32_MLX_FILES:
            full = os.path.join(ckpts, rel)
            if os.path.isfile(full):
                paths.append(full)
    return paths


def conversion_complete(ckpts: str) -> bool:
    """Refuse to trim unless the converted artifacts inference reads are present."""

    def have(rel: str) -> bool:
        return os.path.exists(os.path.join(ckpts, rel))

    dit = any(have(f"mlx/dit_{variant}{q}.safetensors") for variant in ("base", "flash") for q in ("", ".q8"))
    thinker = have("mlx/thinker/thinker.safetensors") or have("mlx/thinker/thinker.q8.safetensors")
    fusion = have("mlx/fusion_base.safetensors") or have("mlx/fusion_flash.safetensors")
    return dit and thinker and fusion and have("mlx/vae.safetensors") and have("mlx/thinker/thinker_config.json")


def main() -> None:
    p = argparse.ArgumentParser(description="Report or delete checkpoints that are dead weight after MLX conversion")
    p.add_argument("--ckpts", default=os.path.join(ROOT, "ckpts"), help="checkpoint root (default: ./ckpts)")
    p.add_argument("--apply", action="store_true", help="delete the files (default: report only)")
    p.add_argument("--fp32-mlx", action="store_true", help="with --apply: also delete the fp32 MLX weights (8-bit-only setups)")
    a = p.parse_args()

    if not conversion_complete(a.ckpts):
        raise SystemExit("error: converted MLX weights not found under ckpts/mlx/ — run auk_mlx.convert first.")

    paths = collect(a.ckpts, a.fp32_mlx)
    if not paths:
        print("Nothing to trim: already trimmed (or no checkpoints found).")
        return

    total = sum(_gb(f) for f in paths)
    print(f"{len(paths)} files, {total:.1f} GB reclaimable" + (" (+fp32 MLX)" if a.fp32_mlx else "") + ":\n")
    for f in sorted(paths, key=_gb, reverse=True):
        print(f"  {_gb(f):6.2f} GB  {os.path.relpath(f, a.ckpts)}")

    if not a.apply:
        print(
            "\nDry run — re-run with --apply to delete."
            + (" Use --fp32-mlx to also free the fp32 MLX copies when you only run --bits 8." if not a.fp32_mlx else "")
        )
        return

    freed = 0.0
    for f in paths:
        os.remove(f)
        freed += _gb(f)
    print(f"\nDeleted {len(paths)} files, {freed:.1f} GB freed.")
    if a.fp32_mlx:
        print("Note: re-running with --bits omitted now requires re-converting from fresh torch weights.")


if __name__ == "__main__":
    main()
