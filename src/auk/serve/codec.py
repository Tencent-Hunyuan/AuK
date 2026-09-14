"""Wire format for the AuK split deployment.

A frame is::

    b"AUK1" | uint32 header_len (big endian) | header (JSON) | payload (safetensors)

Tensors travel as safetensors rather than pickled torch objects: it is already a dependency,
it is not executable, and it preserves dtype exactly. Scalars / small metadata ride in the
JSON header so a request can be introspected without touching the tensor payload.

The transport is deliberately plain HTTP: payloads are a few MB and the call rate is a handful
per generation, so HTTP/2 or gRPC buys nothing worth the codegen and extra dependency.
"""

from __future__ import annotations

import json
import struct

import torch
from safetensors.torch import load as _st_load
from safetensors.torch import save as _st_save

MAGIC = b"AUK1"
_HEADER = struct.Struct(">I")


def pack(header: dict, tensors: dict[str, torch.Tensor] | None = None) -> bytes:
    """Serialise a JSON header plus an optional dict of tensors into one frame."""
    head = json.dumps(header, separators=(",", ":")).encode("utf-8")
    payload = b""
    if tensors:
        payload = _st_save({k: _prep(v) for k, v in tensors.items()})
    return MAGIC + _HEADER.pack(len(head)) + head + payload


def unpack(data: bytes) -> tuple[dict, dict[str, torch.Tensor]]:
    """Inverse of :func:`pack`. Raises ``ValueError`` on a malformed frame."""
    if not isinstance(data, (bytes, bytearray)) or len(data) < 4 + _HEADER.size:
        raise ValueError("Truncated AuK frame")
    if bytes(data[:4]) != MAGIC:
        raise ValueError(f"Bad AuK frame magic: {bytes(data[:4])!r}")
    (head_len,) = _HEADER.unpack(data[4 : 4 + _HEADER.size])
    start = 4 + _HEADER.size
    if len(data) < start + head_len:
        raise ValueError("Truncated AuK frame header")
    header = json.loads(bytes(data[start : start + head_len]).decode("utf-8"))
    body = bytes(data[start + head_len :])
    tensors = _st_load(body) if body else {}
    return header, tensors


def _prep(tensor: torch.Tensor) -> torch.Tensor:
    # safetensors refuses non-contiguous tensors and tensors sharing storage
    return tensor.detach().to("cpu").contiguous()
