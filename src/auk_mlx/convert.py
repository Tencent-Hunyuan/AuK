"""Convert AuK PyTorch checkpoints to MLX safetensors.

Two jobs:
  1. Fold weight-norm pairs (``weight_g``, ``weight_v``) into a single ``weight``.
  2. Transpose conv weights from PyTorch's channels-first layout to MLX's
     channels-last (see auk_mlx/layers.py for the layout table).

Usage:
    python -m auk_mlx.convert vae   ckpts/AuK/vae.safetensors        ckpts/mlx/vae.safetensors
    python -m auk_mlx.convert dit   ckpts/AuK/auk_base.safetensors   ckpts/mlx/dit.safetensors
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from safetensors import safe_open
from safetensors.numpy import save_file


def _fold_weight_norm(g: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """weight = g * v / ||v||, norm taken over every axis but dim 0."""
    dims = tuple(range(1, v.ndim))
    norm = v.float().pow(2).sum(dim=dims, keepdim=True).sqrt()
    return (g.float() / norm) * v.float()


def convert_vae(src: str, dst: str) -> None:
    with safe_open(src, "pt") as f:
        sd = {k: f.get_tensor(k) for k in f.keys()}

    # 1. fold every weight_g / weight_v pair
    folded: dict[str, torch.Tensor] = {}
    for k in list(sd):
        if k.endswith(".weight_g"):
            base = k[: -len(".weight_g")]
            folded[base + ".weight"] = _fold_weight_norm(sd[k], sd[base + ".weight_v"])
        elif not k.endswith(".weight_v"):
            folded[k] = sd[k]

    out: dict[str, np.ndarray] = {}

    def put(name: str, t: torch.Tensor) -> None:
        out[name] = t.detach().float().numpy()

    def conv(name: str, t: torch.Tensor) -> None:
        """torch Conv1d (O, I, K) -> mlx (O, K, I)"""
        put(name, t.permute(0, 2, 1).contiguous())

    def convT(name: str, t: torch.Tensor) -> None:
        """torch ConvTranspose1d (I, O, K) -> mlx (O, K, I)"""
        put(name, t.permute(1, 2, 0).contiguous())

    put("global_mean", folded["global_mean"])
    put("global_log_std", folded["global_log_std"])

    # ---- encoder: generator.<i> is a flat nn.Sequential of Conv1d_S / ResStack / LeakyReLU
    # layout: 0=pre conv, then per stage (down conv, ResStack, LReLU), finally post conv.
    # indices present in the checkpoint: 0, 2,3, 5,6, 8,9, 11,12, 14,15, 17,18, 20
    enc_prefix = "audio_encoder.generator."
    conv("audio_encoder.pre.weight", folded[enc_prefix + "0.layer.weight"])
    put("audio_encoder.pre.bias", folded[enc_prefix + "0.layer.bias"])

    down_idx = [2, 5, 8, 11, 14, 17]  # Conv1d_S per stage
    stack_idx = [3, 6, 9, 12, 15, 18]  # ResStack per stage
    for s, (di, si) in enumerate(zip(down_idx, stack_idx)):
        conv(f"audio_encoder.stages.{s}.down.weight", folded[f"{enc_prefix}{di}.layer.weight"])
        put(f"audio_encoder.stages.{s}.down.bias", folded[f"{enc_prefix}{di}.layer.bias"])
        # ResStack.layers.<n> is Sequential(LReLU, conv@1, LReLU, conv@3)
        n = 0
        while f"{enc_prefix}{si}.layers.{n}.1.weight" in folded:
            conv(f"audio_encoder.stages.{s}.stack.layers.{n}.0.weight", folded[f"{enc_prefix}{si}.layers.{n}.1.weight"])
            put(f"audio_encoder.stages.{s}.stack.layers.{n}.0.bias", folded[f"{enc_prefix}{si}.layers.{n}.1.bias"])
            conv(f"audio_encoder.stages.{s}.stack.layers.{n}.1.weight", folded[f"{enc_prefix}{si}.layers.{n}.3.weight"])
            put(f"audio_encoder.stages.{s}.stack.layers.{n}.1.bias", folded[f"{enc_prefix}{si}.layers.{n}.3.bias"])
            n += 1
    conv("audio_encoder.post.weight", folded[enc_prefix + "20.layer.weight"])
    put("audio_encoder.post.bias", folded[enc_prefix + "20.layer.bias"])

    # ---- decoder
    conv("decoder.conv_pre.weight", folded["conv_pre.weight"])
    put("decoder.conv_pre.bias", folded["conv_pre.bias"])
    conv("decoder.conv_post.weight", folded["conv_post.weight"])  # bias=False

    i = 0
    while f"ups.{i}.0.weight" in folded:
        convT(f"decoder.ups.{i}.weight", folded[f"ups.{i}.0.weight"])
        put(f"decoder.ups.{i}.bias", folded[f"ups.{i}.0.bias"])
        i += 1

    r = 0
    while f"resblocks.{r}.convs1.0.weight" in folded:
        for j in range(3):
            conv(f"decoder.resblocks.{r}.convs1.{j}.weight", folded[f"resblocks.{r}.convs1.{j}.weight"])
            put(f"decoder.resblocks.{r}.convs1.{j}.bias", folded[f"resblocks.{r}.convs1.{j}.bias"])
            conv(f"decoder.resblocks.{r}.convs2.{j}.weight", folded[f"resblocks.{r}.convs2.{j}.weight"])
            put(f"decoder.resblocks.{r}.convs2.{j}.bias", folded[f"resblocks.{r}.convs2.{j}.bias"])
        for a in range(6):
            put(f"decoder.resblocks.{r}.activations.{a}.act.alpha", folded[f"resblocks.{r}.activations.{a}.act.alpha"])
            put(f"decoder.resblocks.{r}.activations.{a}.act.beta", folded[f"resblocks.{r}.activations.{a}.act.beta"])
        r += 1

    put("decoder.activation_post.act.alpha", folded["activation_post.act.alpha"])
    put("decoder.activation_post.act.beta", folded["activation_post.act.beta"])

    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    save_file(out, dst)
    print(f"VAE -> {dst}  ({len(out)} tensors, {sum(v.nbytes for v in out.values()) / 1e6:.0f} MB)")


def convert_dit(src: str, dst: str) -> None:
    """Flux2Edit + layer-fusion weights. Linear layers need no transpose in MLX.

    Two index remaps, because torch's ``nn.Sequential`` counts activation modules
    as layers while the MLX port stores only the parametrised ones in a list:
        time_mlp.2   -> time_mlp.1   (SiLU sits at index 1 upstream)
        conv1d.2     -> conv1d.1     (Mish sits at index 1 upstream)
    """
    with safe_open(src, "pt") as f:
        keys = [k for k in f.keys() if not k.startswith("text_encoder.")]
        sd = {k: f.get_tensor(k) for k in keys}

    fusion: dict[str, np.ndarray] = {}
    out: dict[str, np.ndarray] = {}
    for k, v in sd.items():
        if k in ("layer_weights", "layer_scale"):
            fusion[k] = v.detach().float().numpy()
            continue
        name = k.removeprefix("transformer.")
        t = v.detach().float()
        # inv_freq is kept: AuK's buffer is a bf16-rounded 10000-base schedule, and
        # recomputing it analytically visibly shifts the attention logits.
        if name == "rotary_embed.inv_freq":
            fusion["inv_freq"] = t.numpy()
            continue
        if ".conv1d." in name and name.endswith(".weight"):
            t = t.permute(0, 2, 1).contiguous()
        name = name.replace("time_mlp.2.", "time_mlp.1.").replace("conv_pos_embed.conv1d.2.", "conv_pos_embed.conv1d.1.")
        out[name] = t.numpy()

    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    save_file(out, dst)
    print(f"DiT -> {dst}  ({len(out)} tensors, {sum(v.nbytes for v in out.values()) / 1e6:.0f} MB)")

    if fusion:
        # Named after the DiT it belongs to: layer_weights / layer_scale are trained
        # per release (Flash's layer_scale differs from base by 4.0), so a shared
        # fusion file would silently mis-condition one of the two.
        stem = os.path.splitext(os.path.basename(dst))[0]
        fdst = os.path.join(os.path.dirname(os.path.abspath(dst)), f"fusion_{stem.replace('dit_', '')}.safetensors")
        save_file(fusion, fdst)
        print(f"layer fusion -> {fdst}  ({', '.join(f'{k}{v.shape}' for k, v in fusion.items())})")


def convert_thinker(src_dir: str, dst_dir: str) -> None:
    """Convert the Qwen2.5-Omni Thinker (text model + audio tower) to MLX.

    Dropped on purpose: ``lm_head`` (AuK reads hidden states, never logits),
    ``visual`` (the vision tower is deleted upstream too), and the talker /
    token2wav stacks (not part of the Thinker).
    """
    import glob

    shards = sorted(glob.glob(os.path.join(src_dir, "*.safetensors")))
    if not shards:
        raise SystemExit(f"no safetensors found in {src_dir}")

    with open(os.path.join(src_dir, "config.json")) as f:
        full_cfg = json.load(f)
    tcfg = full_cfg["thinker_config"]["text_config"]
    acfg = full_cfg["thinker_config"]["audio_config"]

    out: dict[str, np.ndarray] = {}
    kept = skipped = 0
    for shard in shards:
        with safe_open(shard, "pt") as f:
            for k in f.keys():
                if not k.startswith("thinker."):
                    skipped += 1
                    continue
                name = k[len("thinker.") :]
                if name.startswith("visual.") or name == "lm_head.weight":
                    skipped += 1
                    continue
                t = f.get_tensor(k).detach().float()
                # model.* -> flat; audio_tower.* keeps its prefix
                name = name.removeprefix("model.")
                # audio conv1/conv2: torch (O, I, K) -> mlx (O, K, I)
                if name.startswith("audio_tower.conv") and name.endswith(".weight"):
                    t = t.permute(0, 2, 1).contiguous()
                out[name] = t.numpy()
                kept += 1

    os.makedirs(dst_dir, exist_ok=True)
    save_file(out, os.path.join(dst_dir, "thinker.safetensors"))

    keep_text = (
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "intermediate_size",
        "vocab_size",
        "rope_theta",
        "rms_norm_eps",
        "head_dim",
    )
    keep_audio = (
        "d_model",
        "encoder_layers",
        "encoder_attention_heads",
        "encoder_ffn_dim",
        "output_dim",
        "n_window",
        "num_mel_bins",
        "scale_embedding",
    )
    meta = {
        "text": {k: tcfg[k] for k in keep_text if k in tcfg},
        "audio": {k: acfg[k] for k in keep_audio if k in acfg},
    }
    with open(os.path.join(dst_dir, "thinker_config.json"), "w") as f:
        json.dump(meta, f, indent=2)

    gb = sum(v.nbytes for v in out.values()) / 1e9
    print(f"Thinker -> {dst_dir}  ({kept} tensors kept, {skipped} skipped, {gb:.2f} GB fp32)")


def quantize_weights(mlx_dir: str, bits: int, group_size: int = 64, variant: str = "base") -> None:
    """Write quantized copies of the DiT and Thinker next to the fp32 ones.

    Quantizing at load time already cuts peak memory; doing it once here also cuts
    the bytes on disk, which is what a user downloading the weights actually pays.
    The VAE is skipped: it is 99.9% Conv1d, which MLX cannot quantize.
    """
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from omegaconf import OmegaConf

    from auk_mlx.dit import DiTConfig, Flux2Edit
    from auk_mlx.qwen_thinker import AudioConfig, TextConfig, ThinkerEncoder

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    cfg_dir = "AuK-Flash" if variant == "flash" else "AuK"
    arch = OmegaConf.to_container(OmegaConf.load(os.path.join(root, "ckpts", cfg_dir, "config.yaml")).model.arch, resolve=True)
    arch["latent_dim"] = 64

    fusion = mx.load(os.path.join(mlx_dir, f"fusion_{variant}.safetensors"))
    dit = Flux2Edit(DiTConfig.from_dict(arch), inv_freq=np.array(fusion["inv_freq"]))
    dit.load_weights(os.path.join(mlx_dir, f"dit_{variant}.safetensors"))
    dit.eval()
    nn.quantize(dit, group_size=group_size, bits=bits)
    dst = os.path.join(mlx_dir, f"dit_{variant}.q{bits}.safetensors")
    mx.save_safetensors(dst, dict(tree_flatten(dit.parameters())))
    print(f"DiT q{bits} -> {dst}  ({os.path.getsize(dst) / 1e9:.2f} GB)")
    del dit

    tdir = os.path.join(mlx_dir, "thinker")
    with open(os.path.join(tdir, "thinker_config.json")) as f:
        tmeta = json.load(f)
    th = ThinkerEncoder(TextConfig(**tmeta["text"]), AudioConfig(**tmeta["audio"]))
    th.load_weights(os.path.join(tdir, "thinker.safetensors"), strict=False)
    th.eval()
    nn.quantize(th, group_size=group_size, bits=bits)
    dst = os.path.join(tdir, f"thinker.q{bits}.safetensors")
    mx.save_safetensors(dst, dict(tree_flatten(th.parameters())))
    print(f"Thinker q{bits} -> {dst}  ({os.path.getsize(dst) / 1e9:.2f} GB)")


def main() -> None:
    p = argparse.ArgumentParser(description="Convert AuK torch checkpoints to MLX")
    p.add_argument("kind", choices=["vae", "dit", "thinker", "quantize"])
    p.add_argument("src", help="source checkpoint, or the MLX dir when kind=quantize")
    p.add_argument("dst", nargs="?", default=None, help="destination (unused for kind=quantize)")
    p.add_argument("--bits", type=int, choices=[4, 8], default=8, help="kind=quantize only")
    p.add_argument("--group_size", type=int, default=64, help="kind=quantize only")
    p.add_argument("--variant", choices=["base", "flash"], default="base", help="kind=quantize only")
    a = p.parse_args()

    if a.kind == "quantize":
        quantize_weights(a.src, a.bits, a.group_size, a.variant)
        return
    if a.dst is None:
        raise SystemExit(f"kind={a.kind} needs a destination path")
    {"vae": convert_vae, "dit": convert_dit, "thinker": convert_thinker}[a.kind](a.src, a.dst)


if __name__ == "__main__":
    main()
