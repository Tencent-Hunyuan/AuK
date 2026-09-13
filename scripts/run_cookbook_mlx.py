"""Run every COOKBOOK example through the MLX port.

    PYTHONPATH=src .venv/bin/python scripts/run_cookbook_mlx.py --flash
    PYTHONPATH=src .venv/bin/python scripts/run_cookbook_mlx.py --base --nfe 32

Writes wavs to an output directory and prints a per-task status table.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

import numpy as np


logging.disable(logging.INFO)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

A = "assets/demo-input-audio"

# (id, section, instruction, audio, gen_seconds) -- transcribed from docs/COOKBOOK.md
CASES = [
    (
        "1.1-zeroshot-tts",
        "Zero-shot TTS",
        "Say the following with the same voice: 'Ladies and gentlemen, it's an honor to have the "
        "opportunity to address such a distinguished audience'",
        f"{A}/zero-shot-tts/ref.wav",
        6.0,
    ),
    (
        "1.2-instruct-tts",
        "Instruct TTS",
        'Based on the following description: "一位二十多岁的女性，面对刚到家的伴侣，以温柔、关心且略带撒娇的语气轻声诉说。'
        '她的声音柔和亲密，语速稍缓，音量适中，音色甜美自然，在短语结尾处语调温暖地上扬。她正向伴侣问候回家，并温柔询问今日工作如何。", '
        'generate speech content "Welcome home, how was work today?".',
        None,
        1.7,
    ),
    (
        "2.1-content-edit",
        "Speech Content Editing",
        "Replace 'but accepting what we cannot have' with 'and living well with dreams unmet'.",
        f"{A}/content-edit/content.wav",
        7.0,
    ),
    (
        "2.2-lyric-edit",
        "Lyric Editing",
        'Change "rear view" to "like you" in the vocal recording.',
        f"{A}/vocal-edit/vocaledit-en-1-input.wav",
        None,
    ),
    ("3.1-pitch", "Pitch Editing", "Raise the pitch by 2 semitones.", f"{A}/pitch/pitch-1-input.wav", None),
    ("3.2-speed", "Speed Editing", "Adjust the speech speed to 1.5x.", f"{A}/speed/speed-edit-1-input.wav", 6.86),
    ("3.3-volume", "Volume Editing", "Increase the volume by 10 dB.", f"{A}/energy/energy-edit-1-input.wav", None),
    ("4.1-emotion", "Emotion", "Change the emotion to happy.", f"{A}/emotion-edit/en-1-input.wav", None),
    (
        "4.2-timbre",
        "Timbre",
        'Keep the spoken content unchanged and change the timbre to: "a deep, calm male voice".',
        f"{A}/vc/vc-1-input.wav",
        None,
    ),
    ("4.3-deaccent", "De-accent", "请把方言腔改成标准普通话发音,音色维持一致。", f"{A}/accent/accent-sichuan-input.wav", None),
    ("4.4-nonverbal-remove", "Nonverbal Remove", "Remove the humming from the audio.", f"{A}/nv/en-d-input.wav", 22.0),
    ("4.4-nonverbal-add", "Nonverbal Add", "Add a cough before 'We tested'", f"{A}/nv/en-c-input.wav", 10.44),
    ("4.5-whisper", "Whisper Conversion", "用小声耳语的方式把这段话说出来。", f"{A}/whisper/wh-w2n-zh-input.wav", None),
    (
        "5.1-enhance",
        "Speech Enhancement",
        "Preserve all speakers, remove noise and reverberation, and output clean speech of the same length.",
        f"{A}/se/se-zh-1-input.wav",
        None,
    ),
    (
        "5.2-separation",
        "Speech Separation",
        "Please keep the second speaker to start talking and remove the other speakers, outputting a single clean speech track.",
        f"{A}/ss/zh-1-input.wav",
        None,
    ),
    (
        "5.3-music-sep",
        "Music Separation",
        "Keep the clean singing voice, drop all other audio.",
        f"{A}/vocal-extraction/vocal-1-input.wav",
        None,
    ),
    (
        "5.4-tse",
        "Target Speaker Extraction",
        'Keep only the speaker who says "get what" and remove all other speakers.',
        f"{A}/ss/en-1-input.wav",
        None,
    ),
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--flash", action="store_true", help="AuK-Flash, 4 fixed steps")
    p.add_argument("--base", action="store_true", help="AuK base")
    p.add_argument("--nfe", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bits", type=int, choices=[4, 8], default=None, help="quantize DiT + Thinker")
    p.add_argument("--group_size", type=int, default=64)
    p.add_argument("--out", default="/tmp/cookbook_mlx")
    p.add_argument("--only", default=None, help="substring filter on the case id")
    args = p.parse_args()
    if not (args.flash or args.base):
        args.flash = True

    import soundfile as sf

    from auk_mlx.infer import AukMLX, GenerateOptions

    variant = "flash" if args.flash else "base"
    cfg = os.path.join(ROOT, "ckpts", "AuK-Flash" if args.flash else "AuK", "config.yaml")
    out_dir = os.path.join(args.out, variant)
    os.makedirs(out_dir, exist_ok=True)

    os.chdir(ROOT)  # the cookbook paths are repo-relative
    eng = AukMLX(
        os.path.join(ROOT, "ckpts", "mlx"),
        cfg,
        os.path.join(ROOT, "ckpts", "Qwen2.5-Omni-3B"),
        bits=args.bits,
        group_size=args.group_size,
    )
    prec = f"{args.bits}-bit" if args.bits else "fp32"
    print(f"# {variant} ({'4 steps' if args.flash else f'{args.nfe} steps'}), {prec}, seed {args.seed}\n")

    rows, total = [], 0.0
    for cid, name, instr, audio, secs in CASES:
        if args.only and args.only not in cid:
            continue
        gen_secs = secs
        if gen_secs is None and audio:
            info = sf.info(audio)
            gen_secs = info.frames / info.samplerate
        opts = GenerateOptions(gen_seconds=gen_secs, seed=args.seed)
        if not args.flash:
            opts.nfe = args.nfe
        try:
            t0 = time.time()
            wav, sr = eng.generate(instr, audio_path=audio, opts=opts)
            dt = time.time() - t0
            total += dt
            path = os.path.join(out_dir, f"{cid}.wav")
            sf.write(path, wav, sr)
            dur = len(wav) / sr
            rows.append((cid, name, "ok", dt, dur, float(np.sqrt((wav**2).mean())), float(np.abs(wav).max())))
            print(f"  ok   {cid:22s} {dt:6.1f}s  {dur:5.2f}s out  rms={rows[-1][5]:.4f} peak={rows[-1][6]:.3f}")
        except Exception as e:
            rows.append((cid, name, f"FAIL {type(e).__name__}", 0.0, 0.0, 0.0, 0.0))
            print(f"  FAIL {cid:22s} {type(e).__name__}: {str(e)[:100]}")

    n_ok = sum(1 for r in rows if r[2] == "ok")
    print(f"\n{n_ok}/{len(rows)} ok, {total:.0f}s total -> {out_dir}")


if __name__ == "__main__":
    main()
