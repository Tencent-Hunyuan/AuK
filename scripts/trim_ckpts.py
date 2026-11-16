"""Free disk space after MLX conversion by deleting weights inference no longer reads.

The conversion step (`python -m auk_mlx.convert ...`) reads the torch originals
and writes `ckpts/mlx/`. After it finishes, inference reads only:

  - ckpts/<variant>/config.yaml            architecture (kept)
  - ckpts/mlx/...                          converted weights (kept)
  - Qwen2.5-Omni-3B processor files        tokenizer, chat template, mel front end (kept)

Once each installed variant has been converted, its Torch weights and the
shared Qwen weights can be removed. To also remove the fp32 MLX weights,
each installed variant and the shared Thinker need 8-bit weights.

PyTorch inference and a future re-conversion need the original weights.

Usage:
    python scripts/trim_ckpts.py                     # report only (default)
    python scripts/trim_ckpts.py --apply             # delete torch originals
    python scripts/trim_ckpts.py --apply --fp32-mlx  # also drop fp32 MLX weights
"""

from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VARIANT_DIRS = {"base": "AuK", "flash": "AuK-Flash"}

TORCH_FILES = [
    f"{directory}/{filename}"
    for variant, directory in VARIANT_DIRS.items()
    for filename in (f"auk_{variant}.safetensors", "vae.safetensors")
] + ["Qwen2.5-Omni-3B/spk_dict.pt", "Qwen2.5-Omni-3B/model.safetensors.index.json"]

FP32_MLX_FILES = [
    "mlx/dit_base.safetensors",
    "mlx/dit_flash.safetensors",
    "mlx/thinker/thinker.safetensors",
]


def missing_artifacts(ckpts: Path, fp32_mlx: bool) -> list[str]:
    """List the files needed to keep each installed variant usable."""
    shared = ("mlx/vae.safetensors", "mlx/thinker/thinker_config.json", "Qwen2.5-Omni-3B/config.json")
    missing = [path for path in shared if not (ckpts / path).is_file()]
    thinker = "mlx/thinker/thinker"
    if not (ckpts / f"{thinker}.q8.safetensors").is_file() and (fp32_mlx or not (ckpts / f"{thinker}.safetensors").is_file()):
        missing.append(f"{thinker}.q8.safetensors" if fp32_mlx else f"{thinker}[.q8].safetensors")

    variants = []
    for variant, directory in VARIANT_DIRS.items():
        dit = f"mlx/dit_{variant}"
        fusion = f"mlx/fusion_{variant}.safetensors"
        fp32 = (ckpts / f"{dit}.safetensors").is_file()
        q8 = (ckpts / f"{dit}.q8.safetensors").is_file()
        torch_weights = (f"{directory}/auk_{variant}.safetensors", f"{directory}/vae.safetensors")
        if not (fp32 or q8 or (ckpts / fusion).is_file() or any((ckpts / path).is_file() for path in torch_weights)):
            continue
        variants.append(variant)
        missing.extend(path for path in (fusion, f"{directory}/config.yaml") if not (ckpts / path).is_file())
        if not q8 and (fp32_mlx or not fp32):
            missing.append(f"{dit}.q8.safetensors" if fp32_mlx else f"{dit}[.q8].safetensors")
    if not variants:
        missing.append("a converted Base or Flash DiT with matching fusion weights")
    return missing


def main() -> None:
    p = argparse.ArgumentParser(description="Report or delete checkpoints that are dead weight after MLX conversion")
    p.add_argument("--ckpts", type=Path, default=ROOT / "ckpts", help="checkpoint root (default: ./ckpts)")
    p.add_argument("--apply", action="store_true", help="delete the files (default: report only)")
    p.add_argument("--fp32-mlx", action="store_true", help="with --apply: also delete the fp32 MLX weights (8-bit-only setups)")
    a = p.parse_args()

    ckpts = a.ckpts.resolve()
    missing = missing_artifacts(ckpts, a.fp32_mlx)
    if missing:
        raise SystemExit("error: required files are missing; nothing was deleted:\n  " + "\n  ".join(missing))

    candidates = [ckpts / name for name in TORCH_FILES]
    candidates.extend((ckpts / "Qwen2.5-Omni-3B").glob("model-*.safetensors"))
    if a.fp32_mlx:
        candidates.extend(ckpts / name for name in FP32_MLX_FILES)
    paths = [path for path in candidates if path.is_file()]
    if not paths:
        print("Nothing to trim: already trimmed (or no checkpoints found).")
        return

    for path in paths:
        if not path.parent.resolve().is_relative_to(ckpts):
            raise SystemExit(f"error: refusing to delete outside {ckpts}: {path}; nothing was deleted")

    files = sorted(((path, path.lstat().st_size) for path in paths), key=lambda item: item[1], reverse=True)
    total = sum(size for _, size in files) / 1e9
    print(f"{len(files)} files, {total:.1f} GB reclaimable" + (" (+fp32 MLX)" if a.fp32_mlx else "") + ":\n")
    for path, size in files:
        print(f"  {size / 1e9:6.2f} GB  {path.relative_to(ckpts)}")

    if not a.apply:
        print(
            "\nDry run — re-run with --apply to delete."
            + (" Use --fp32-mlx to also free the fp32 MLX copies when you only run --bits 8." if not a.fp32_mlx else "")
        )
        return

    for path, _ in files:
        path.unlink()
    print(f"\nDeleted {len(files)} files, {total:.1f} GB removed.")
    if a.fp32_mlx:
        print("Note: re-running with --bits omitted now requires re-converting from fresh torch weights.")


if __name__ == "__main__":
    main()
