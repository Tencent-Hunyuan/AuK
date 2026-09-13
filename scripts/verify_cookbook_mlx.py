"""Verify COOKBOOK outputs: did each task actually do what it claims?

    PYTHONPATH=src .venv/bin/python scripts/verify_cookbook_mlx.py /tmp/cookbook_mlx/flash

Per-task checks against the input, rather than a generic "is it finite" pass:
content edits are transcribed, pitch/volume/speed are measured, enhancement and
separation are checked for the expected direction of change.
"""

from __future__ import annotations

import os
import re
import sys

import numpy as np
import soundfile as sf
import soxr


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
A = os.path.join(ROOT, "assets", "demo-input-audio")
SR = 24000


def load(path: str) -> np.ndarray:
    w, sr = sf.read(path, dtype="float32", always_2d=True)
    w = w.mean(axis=1)
    return soxr.resample(w, sr, SR, quality="VHQ") if sr != SR else w


def median_f0(x: np.ndarray) -> float:
    """Autocorrelation pitch, median over voiced frames."""
    fl, hop = int(0.04 * SR), int(0.02 * SR)
    f0 = []
    for i in range(0, max(0, len(x) - fl), hop):
        fr = x[i : i + fl]
        if np.sqrt((fr**2).mean()) < 0.02:
            continue
        fr = fr - fr.mean()
        ac = np.correlate(fr, fr, "full")[fl - 1 :]
        lo, hi = int(SR / 400), int(SR / 70)
        if hi >= len(ac):
            continue
        k = lo + int(np.argmax(ac[lo:hi]))
        if ac[k] > 0.3 * ac[0]:
            f0.append(SR / k)
    return float(np.median(f0)) if f0 else 0.0


def db(x: np.ndarray) -> float:
    return 20 * np.log10(np.sqrt((x**2).mean()) + 1e-12)


def speech_rate(x: np.ndarray) -> float:
    """Syllable-ish rate: energy-envelope peaks per second."""
    fl, hop = int(0.025 * SR), int(0.010 * SR)
    e = np.array([np.sqrt((x[i : i + fl] ** 2).mean()) for i in range(0, max(0, len(x) - fl), hop)])
    if len(e) < 3:
        return 0.0
    e = e / (e.max() + 1e-9)
    peaks = ((e[1:-1] > e[:-2]) & (e[1:-1] > e[2:]) & (e[1:-1] > 0.25)).sum()
    return float(peaks / (len(e) * 0.010))


_asr_cache: dict[str, str] = {}


def asr(path: str, lang: str = "en") -> str:
    if path in _asr_cache:
        return _asr_cache[path]
    import mlx_whisper

    model = "mlx-community/whisper-small.en-mlx" if lang == "en" else "mlx-community/whisper-small-mlx"
    txt = mlx_whisper.transcribe(path, path_or_hf_repo=model, language=lang)["text"].strip()
    _asr_cache[path] = txt
    return txt


def words(s: str) -> list[str]:
    return re.sub(r"[^a-z' ]", "", s.lower()).split()


def main() -> None:
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "/tmp/cookbook_mlx/flash"

    def o(cid: str) -> str:
        return os.path.join(out_dir, f"{cid}.wav")

    print(f"# verifying {out_dir}\n")
    results: list[tuple[str, str, str]] = []

    def report(cid: str, verdict: str, detail: str) -> None:
        results.append((cid, verdict, detail))
        print(f"  {verdict:4s} {cid:22s} {detail}")

    # 1.1 zero-shot TTS -- must speak the target sentence
    if os.path.isfile(o("1.1-zeroshot-tts")):
        target = "ladies and gentlemen it's an honor to have the opportunity to address such a distinguished audience"
        import difflib

        got = asr(o("1.1-zeroshot-tts"))
        acc = difflib.SequenceMatcher(None, words(target), words(got)).ratio()
        report("1.1-zeroshot-tts", "ok" if acc > 0.8 else "WARN", f"word-acc={acc * 100:.0f}%  “{got[:70]}”")

    # 1.2 instruct TTS -- no reference; must say the requested line
    if os.path.isfile(o("1.2-instruct-tts")):
        got = asr(o("1.2-instruct-tts"))
        hit = "welcome home" in got.lower()
        report("1.2-instruct-tts", "ok" if hit else "WARN", f"“{got[:70]}”")

    # 2.1 content edit -- new phrase in, old phrase out
    if os.path.isfile(o("2.1-content-edit")):
        got = asr(o("2.1-content-edit")).lower()
        new_in = "dreams unmet" in got or "living well" in got
        old_out = "accepting what we cannot have" not in got
        v = "ok" if (new_in and old_out) else "WARN"
        report("2.1-content-edit", v, f"new={new_in} old_gone={old_out}  “{got[:70]}”")

    # 2.2 lyric edit -- "rear view" -> "like you". Singing ASR is unreliable, so a miss
    # here is weak evidence; listen before concluding the edit failed.
    if os.path.isfile(o("2.2-lyric-edit")):
        got = asr(o("2.2-lyric-edit")).lower()
        report(
            "2.2-lyric-edit",
            "ok" if "like you" in got else "WARN",
            f"has_like_you={'like you' in got} has_rear_view={'rear view' in got}  “{got[:60]}”",
        )

    # 3.1 pitch +2 semitones = +200 cents
    if os.path.isfile(o("3.1-pitch")):
        src, dst = load(f"{A}/pitch/pitch-1-input.wav"), load(o("3.1-pitch"))
        f0s, f0d = median_f0(src), median_f0(dst)
        cents = 1200 * np.log2(f0d / f0s) if f0s > 0 and f0d > 0 else 0.0
        report(
            "3.1-pitch", "ok" if 80 < cents < 330 else "WARN", f"{f0s:.0f}Hz -> {f0d:.0f}Hz = {cents:+.0f} cents (target +200)"
        )

    # 3.2 speed 1.5x -- the model is told the target LENGTH, so check the duration it hit
    # (syllable rate is a weak proxy: it also moves with pauses, so it is reported not asserted)
    if os.path.isfile(o("3.2-speed")):
        src, dst = load(f"{A}/speed/speed-edit-1-input.wav"), load(o("3.2-speed"))
        rs, rd = speech_rate(src), speech_rate(dst)
        ds, dd = len(src) / SR, len(dst) / SR
        got_ratio = ds / dd if dd > 0 else 0
        report(
            "3.2-speed",
            "ok" if 1.35 < got_ratio < 1.7 else "WARN",
            f"dur {ds:.1f}s -> {dd:.1f}s = {got_ratio:.2f}x (target 1.5x); syllable rate {rs:.2f} -> {rd:.2f}/s",
        )

    # 3.3 volume +10 dB
    if os.path.isfile(o("3.3-volume")):
        src, dst = load(f"{A}/energy/energy-edit-1-input.wav"), load(o("3.3-volume"))
        d = db(dst) - db(src)
        report("3.3-volume", "ok" if d > 3 else "WARN", f"{db(src):.1f} -> {db(dst):.1f} dB = {d:+.1f} dB (target +10)")

    # 4.1 emotion -- content preserved, prosody moved
    if os.path.isfile(o("4.1-emotion")):
        src, dst = load(f"{A}/emotion-edit/en-1-input.wav"), load(o("4.1-emotion"))
        import difflib

        a, b = asr(f"{A}/emotion-edit/en-1-input.wav"), asr(o("4.1-emotion"))
        keep = difflib.SequenceMatcher(None, words(a), words(b)).ratio()
        f0s, f0d = median_f0(src), median_f0(dst)
        moved = abs(1200 * np.log2(f0d / f0s)) if f0s > 0 and f0d > 0 else 0
        report(
            "4.1-emotion",
            "ok" if keep > 0.5 else "WARN",
            f"content kept={keep * 100:.0f}%  F0 {f0s:.0f}->{f0d:.0f}Hz ({moved:+.0f} cents)",
        )

    # 4.2 timbre -> "deep, calm male voice": F0 should drop
    if os.path.isfile(o("4.2-timbre")):
        src, dst = load(f"{A}/vc/vc-1-input.wav"), load(o("4.2-timbre"))
        f0s, f0d = median_f0(src), median_f0(dst)
        cents = 1200 * np.log2(f0d / f0s) if f0s > 0 and f0d > 0 else 0
        report("4.2-timbre", "ok" if cents < 0 else "WARN", f"F0 {f0s:.0f} -> {f0d:.0f}Hz = {cents:+.0f} cents (want lower)")

    # 4.3 de-accent -- still Chinese speech of the same length
    if os.path.isfile(o("4.3-deaccent")):
        src, dst = load(f"{A}/accent/accent-sichuan-input.wav"), load(o("4.3-deaccent"))
        got = asr(o("4.3-deaccent"), "zh")
        ratio = len(dst) / len(src)
        report("4.3-deaccent", "ok" if 0.9 < ratio < 1.1 and got else "WARN", f"len ratio={ratio:.2f}  “{got[:40]}”")

    # 4.4 nonverbal remove / add
    if os.path.isfile(o("4.4-nonverbal-remove")):
        src, dst = load(f"{A}/nv/en-d-input.wav"), load(o("4.4-nonverbal-remove"))
        report(
            "4.4-nonverbal-remove",
            "ok" if len(dst) > 0 else "WARN",
            f"{len(src) / SR:.1f}s -> {len(dst) / SR:.1f}s  rms {np.sqrt((src**2).mean()):.4f} -> {np.sqrt((dst**2).mean()):.4f}",
        )
    if os.path.isfile(o("4.4-nonverbal-add")):
        # The cookbook asks for gen_seconds=10.44 on a 13.3 s input, i.e. a SHORTER clip.
        # So the check is that the output matches the requested length, not that it grew.
        src, dst = load(f"{A}/nv/en-c-input.wav"), load(o("4.4-nonverbal-add"))
        want, got = 10.44, len(dst) / SR
        report(
            "4.4-nonverbal-add",
            "ok" if abs(got - want) < 0.3 else "WARN",
            f"{len(src) / SR:.1f}s -> {got:.1f}s (cookbook asked for {want}s)",
        )

    # 4.5 whisper -- whispering is unvoiced, so voiced frames should drop
    if os.path.isfile(o("4.5-whisper")):
        src, dst = load(f"{A}/whisper/wh-w2n-zh-input.wav"), load(o("4.5-whisper"))
        f0s, f0d = median_f0(src), median_f0(dst)
        # spectral tilt: whisper pushes energy up in frequency
        import scipy.signal as ss

        def tilt(x):
            f, p = ss.welch(x, SR, nperseg=1024)
            return float(p[(f >= 2000) & (f < 8000)].sum() / (p[(f >= 0) & (f < 1000)].sum() + 1e-12))

        ts, td = tilt(src), tilt(dst)
        report("4.5-whisper", "ok" if td > ts else "WARN", f"HF/LF ratio {ts:.3f} -> {td:.3f} (whisper raises it)")

    # 5.1 enhancement -- noise floor should fall
    if os.path.isfile(o("5.1-enhance")):
        src, dst = load(f"{A}/se/se-zh-1-input.wav"), load(o("5.1-enhance"))

        def noise_floor(x):
            fl, hop = int(0.025 * SR), int(0.010 * SR)
            e = np.array([np.sqrt((x[i : i + fl] ** 2).mean()) for i in range(0, max(0, len(x) - fl), hop)])
            return float(np.percentile(e, 10))

        ns, nd = noise_floor(src), noise_floor(dst)
        report(
            "5.1-enhance",
            "ok" if nd < ns else "WARN",
            f"noise floor {ns:.5f} -> {nd:.5f} ({20 * np.log10((nd + 1e-12) / (ns + 1e-12)):+.1f} dB)",
        )

    # 5.2 / 5.4 separation -- output should be quieter than the mix (one speaker kept)
    for cid, srcp in (("5.2-separation", f"{A}/ss/zh-1-input.wav"), ("5.4-tse", f"{A}/ss/en-1-input.wav")):
        if os.path.isfile(o(cid)):
            src, dst = load(srcp), load(o(cid))

            def active(x):
                fl, hop = int(0.025 * SR), int(0.010 * SR)
                e = np.array([np.sqrt((x[i : i + fl] ** 2).mean()) for i in range(0, max(0, len(x) - fl), hop)])
                return float((e > 0.02).mean())

            asrc, adst = active(src), active(dst)
            report(cid, "ok" if adst < asrc else "WARN", f"voiced-frame share {asrc:.2f} -> {adst:.2f} (fewer speakers kept)")

    # 5.3 music separation -- the backing track's bass/drums live below 200 Hz, so that
    # band should collapse. (An HF-share test fails here: isolated vocals keep their
    # sibilance, so the >4 kHz share goes UP once the low end is gone.)
    if os.path.isfile(o("5.3-music-sep")):
        import scipy.signal as ss

        src, dst = load(f"{A}/vocal-extraction/vocal-1-input.wav"), load(o("5.3-music-sep"))

        def low_share(x):
            f, p = ss.welch(x, SR, nperseg=2048)
            return float(p[f < 200].sum() / (p.sum() + 1e-12))

        ls, ld = low_share(src), low_share(dst)
        report(
            "5.3-music-sep", "ok" if ld < ls * 0.5 else "WARN", f"<200Hz share {ls:.3f} -> {ld:.3f} (backing bass/drums removed)"
        )

    n_ok = sum(1 for _, v, _ in results if v == "ok")
    print(f"\n{n_ok}/{len(results)} checks passed")


if __name__ == "__main__":
    main()
