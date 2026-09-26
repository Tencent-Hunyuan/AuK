"""Teacher loading and student export shared by both distillation stages."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from numpy.core.multiarray import _reconstruct  # noqa: PLC2701 - the name old pickles reference
from safetensors.torch import load_file


# Bumped whenever the layout of a training checkpoint or an exported artifact changes.
CHECKPOINT_SCHEMA = 1


def safe_torch_load(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    mmap: bool = False,
) -> dict[str, Any]:
    """Load tensor checkpoints without enabling arbitrary pickle execution."""
    allowed = [
        _reconstruct,
        np.ndarray,
        np.dtype,
        *(type(np.dtype(name)) for name in ("float64", "uint32", "int64")),
    ]
    with torch.serialization.safe_globals(allowed):
        return torch.load(path, map_location=map_location, weights_only=True, mmap=mmap)


def teacher_provenance(source: str | Path, *, use_ema: bool) -> dict[str, Any]:
    """Identity of the teacher weights, stored in every artifact and checked on resume."""
    return {"source": str(Path(source).resolve()), "update": 0, "use_ema": use_ema}


def load_teacher_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load either an AuK .safetensors release or a training .pt."""
    path = Path(path)
    if path.suffix == ".safetensors":
        return {"model_state_dict": load_file(str(path), device="cpu")}
    return safe_torch_load(path)


def extract_teacher_state(
    checkpoint: Mapping[str, Any],
    *,
    use_ema: bool = True,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Return (backbone, text_fusion) from full or online artifacts."""
    if use_ema:
        root_key = "ema_model_state_dict"
        prefix = "ema_model.transformer."
        fusion_prefix = "ema_model."
    else:
        root_key = "model_state_dict"
        prefix = "transformer."
        fusion_prefix = ""
    if root_key not in checkpoint:
        raise KeyError(f"checkpoint does not contain {root_key!r}")

    state = checkpoint[root_key]
    backbone = {key[len(prefix) :]: value for key, value in state.items() if key.startswith(prefix)}
    if not backbone:
        raise KeyError(f"no backbone keys with prefix {prefix!r}")
    fusion = {name: state[f"{fusion_prefix}{name}"] for name in ("layer_weights", "layer_scale")}
    return backbone, fusion


def make_student_export(
    *,
    ema_state: Mapping[str, torch.Tensor],
    online_state: Mapping[str, torch.Tensor],
    fusion_state: Mapping[str, torch.Tensor],
    update: int,
    export_type: str,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Build an export carrying the student backbone under both state keys."""
    ema = {f"ema_model.transformer.{key}": value for key, value in ema_state.items()}
    online = {f"transformer.{key}": value for key, value in online_state.items()}
    for name in ("layer_weights", "layer_scale"):
        ema[f"ema_model.{name}"] = fusion_state[name]
        online[name] = fusion_state[name]
    ema["initted"] = torch.tensor(True)
    ema["step"] = torch.tensor(update)
    return {
        "ema_model_state_dict": ema,
        "model_state_dict": online,
        "update": update,
        "schema_version": CHECKPOINT_SCHEMA,
        "export_type": export_type,
        "metadata": dict(metadata),
    }


def load_initializer_artifact(
    path: str | Path,
    *,
    use_ema: bool,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Return the stage-one student backbone and the provenance to record downstream."""
    checkpoint = safe_torch_load(path, mmap=True)
    export_type = str(checkpoint["export_type"])
    schema_version = int(checkpoint["schema_version"])
    if export_type != "dmd_initializer":
        raise ValueError(f"expected dmd_initializer artifact, got {export_type!r}")
    if schema_version != CHECKPOINT_SCHEMA:
        raise ValueError(f"unsupported initializer schema {schema_version}; expected {CHECKPOINT_SCHEMA}")
    # The initializer artifact uses the same two-root layout as a teacher export.
    backbone, _ = extract_teacher_state(checkpoint, use_ema=use_ema)
    provenance = {
        "source": str(path),
        "use_ema": use_ema,
        "export_type": export_type,
        "schema_version": schema_version,
        "update": int(checkpoint["update"]),
        "teacher_provenance": dict(checkpoint["metadata"]["teacher_provenance"]),
    }
    return backbone, provenance
