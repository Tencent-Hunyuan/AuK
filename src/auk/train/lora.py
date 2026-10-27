"""Minimal LoRA implementation for the AuK Flux2Edit backbone (no peft dependency).

Only ``nn.Linear`` modules whose full path ends with one of the configured
suffixes are adapted. Base weights and biases stay frozen, and ``lora_B`` is
zero-initialised so that an adapted model is numerically identical to the
original model at step 0.
"""

from __future__ import annotations

import math

import torch
from torch import nn

DEFAULT_TARGETS = [
    ".attn.to_qkv",
    ".attn.to_qkv_c",
    ".attn.to_out.0",
    ".attn.to_out_c",
    ".ff.linear_in",
    ".ff.linear_out",
    ".ff_c.linear_in",
    ".ff_c.linear_out",
    ".ff_x.linear_in",
    ".ff_x.linear_out",
    ".txt_proj",
    ".proj_out",
    ".audio_embed.linear",
    ".attn_norm.linear",
    ".attn_norm_c.linear",
    ".attn_norm_x.linear",
    ".norm_out.linear",
    ".time_embed.time_mlp.0",
    ".time_embed.time_mlp.2",
]

LEGACY_TARGETS = [
    ".attn.to_qkv",
    ".attn.to_qkv_c",
    ".attn.to_out_c",
    ".ff.linear_in",
    ".ff.linear_out",
    ".ff_c.linear_in",
    ".ff_c.linear_out",
    ".ff_x.linear_in",
    ".ff_x.linear_out",
]


class LoRALinear(nn.Linear):
    """Linear layer with a frozen base weight and a trainable low-rank update."""

    def __init__(self, in_features, out_features, r=32, alpha=32, dropout=0.0, bias=True):
        super().__init__(in_features, out_features, bias=bias)
        self.r = int(r)
        self.alpha = int(alpha)
        self.register_buffer("lora_scaling", torch.tensor(alpha / r, dtype=torch.float32))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        out = super().forward(x)
        if self.r > 0:
            x = self.lora_dropout(x)
            out = out + (x @ self.lora_A.t() @ self.lora_B.t()) * self.lora_scaling
        return out


def _parse_targets(targets):
    """Normalise ``targets`` (None, comma-separated string or list) to a list."""
    if targets is None:
        return list(DEFAULT_TARGETS)
    if isinstance(targets, str):
        return [t.strip() for t in targets.split(",") if t.strip()]
    return [t for t in targets]


def apply_lora(model: nn.Module, r=32, alpha=32, dropout=0.0, targets=None) -> int:
    """Replace every matching ``nn.Linear`` with a ``LoRALinear`` and return the count."""
    targets = _parse_targets(targets)
    count = 0

    for full_name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not any(full_name.endswith(suf) for suf in targets):
            continue

        parent_name, _, leaf = full_name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model

        new = LoRALinear(
            module.in_features,
            module.out_features,
            r=r,
            alpha=alpha,
            dropout=dropout,
            bias=module.bias is not None,
        )
        new.weight.data.copy_(module.weight.data)
        if module.bias is not None:
            new.bias.data.copy_(module.bias.data)

        setattr(parent, leaf, new)
        count += 1

    return count


def freeze_for_lora(model: nn.Module) -> int:
    """Freeze every parameter except ``lora_A`` / ``lora_B``; return the trainable tensor count."""
    for p in model.parameters():
        p.requires_grad_(False)
    n = 0
    for name, p in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            p.requires_grad_(True)
            n += 1
    return n
