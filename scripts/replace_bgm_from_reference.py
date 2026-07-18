#!/usr/bin/env python3
"""Keep the material-based edit and replace only its synthetic BGM.

The reference contributes Demucs non-vocal stems only.  Video is stream-copied.
"""
from __future__ import annotations

import hashlib
import sys
import tempfile
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wavfile
import scipy.signal as sig

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import rebuild_prototype as rb  # noqa: E402

OUT = ROOT / "output" / "prototype.mp4"
REF = ROOT / "reference" / "reference.mp4"
SEP = Path(Path("/private/tmp/sakaba_bgm_tmp_path").read_text().strip()) / "separated" / "htdemucs" / "reference"
SR = 48000
MUTE_START = 8.066667
MUTE_END = 9.980
PRE_FADE = 0.300
RESUME_FADE = 0.600
MUSIC_GAIN_DB = -1.5
DUCK_DB = -3.5


def load(path: Path) -> tuple[int, np.ndarray]:
    rate, x = wavfile.read(path)
    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float64) / max(abs(np.iinfo(x.dtype).min), np.iinfo(x.dtype).max)
    else:
        x = x.astype(np.float64)
    if x.ndim == 1:
        x = x[:, None]
    return rate, x


def rms_db(x: np.ndarray) -> float:
    return float(20 * np.log10(np.sqrt(np.mean(np.square(x))) + 1e-12))


def smooth_transients(x: np.ndarray, rate: int) -> np.ndarray:
    """Reduce likely SFX leakage without changing the musical time line."""
    mono = np.mean(x, axis=1)
    win = max(1, int(.040 * rate))
    env = np.sqrt(sig.fftconvolve(mono * mono, np.ones(win) / win, mode="same") + 1e-12)
    hop = win
    ds = env[::hop]
    med = sig.medfilt(ds, kernel_size=51)
    floor = np.percentile(ds[ds > 1e-7], 20)
    target = np.maximum(med, floor) * 1.995  # allow musical attacks up to +6 dB
    gain_ds = np.minimum(1.0, target / np.maximum(ds, 1e-9))
    gain = np.interp(np.arange(len(x)), np.arange(len(ds)) * hop, gain_ds)
    k = max(1, int(.080 * rate))
    gain = sig.fftconvolve(gain, np.ones(k) / k, mode="same")
    return x * gain[:, None]


def envelope(n: int, speech: np.ndarray) -> np.ndarray:
    t = np.arange(n) / SR
    g = np.full(n, 10 ** (MUSIC_GAIN_DB / 20), dtype=np.float64)
    # Equal-power fade into the exact monochrome thinking-scene mute.
    a = (t >= MUTE_START - PRE_FADE) & (t < MUTE_START)
    g[a] *= np.cos(.5 * np.pi * (t[a] - (MUTE_START - PRE_FADE)) / PRE_FADE) ** 2
    g[(t >= MUTE_START) & (t < MUTE_END)] = 0.0
    b = (t >= MUTE_END) & (t < MUTE_END + RESUME_FADE)
    g[b] *= np.sin(.5 * np.pi * (t[b] - MUTE_END) / RESUME_FADE) ** 2
    # The same speech-aware ducking design used by the material rebuild.
    grown = rb.speech_mask(speech, n)
    g *= rb.mask_to_gain(grown, n, 10 ** (DUCK_DB / 20), .25)
    return g


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    for p in (OUT, REF, SEP / "bass.wav", SEP / "drums.wav", SEP / "other.wav", SEP / "vocals.wav"):
        if not p.exists():
            raise FileNotFoundError(p)
    old_video_sha = sha(OUT)
    with tempfile.TemporaryDirectory(prefix="reference_bgm_") as td:
        work = Path(td)
        total = rb.TOTAL_SAMPLES
        print("[1/4] rebuilding the material-derived audio bed (without synthetic BGM)")
        speech = rb.build_stem(total)
        room = rb.build_roomtone_bed(total)
        foley = rb.build_foley(total)
        grown = rb.speech_mask(speech, total)
        foley_gain = rb.mask_to_gain(grown, total, rb.DUCK_GAIN, rb.DUCK_RAMP)
        base = speech + rb.synth_thinking_sfx(total) + room + foley * foley_gain * rb.level_curve(total)

        print("[2/4] loading aligned Demucs non-vocal stems")
        rate, music = load(SEP / "bass.wav")
        for name in ("drums.wav", "other.wav"):
            r, part = load(SEP / name)
            assert r == rate
            music += part
        _, vocals = load(SEP / "vocals.wav")
        music = smooth_transients(music, rate)
        music = sig.resample_poly(music, SR, rate, axis=0)
        if len(music) < total:
            music = np.pad(music, ((0, total - len(music)), (0, 0)))
        music = music[:total]
        gain = envelope(total, speech)
        music *= gain[:, None]

        stereo_base = np.repeat(base[:, None], 2, axis=1)
        mix = stereo_base + music
        peak_before = float(np.max(np.abs(mix)))
        # Preserve program level; only catch sparse inter-sample/transient overs.
        mix = np.clip(mix, -.999, .999)

        print("[3/4] stream-copying current video and encoding replacement audio")
        raw = work / "mix.f32"
        raw.write_bytes(mix.astype(np.float32).tobytes())
        candidate = work / "prototype.mp4"
        rb.run(["-i", str(OUT), "-f", "f32le", "-ar", str(SR), "-ac", "2", "-i", str(raw),
                "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                "-t", f"{rb.TOTAL_SEC:.7f}", "-movflags", "+faststart", str(candidate)])
        rb.run(["-v", "error", "-i", str(candidate), "-f", "null", "-"])

        # Verify exact mute in the new BGM layer and quantify vocal-stem similarity.
        i0, i1 = int((MUTE_START + .05) * SR), int((MUTE_END - .05) * SR)
        assert np.max(np.abs(music[i0:i1])) == 0.0
        vm = np.mean(vocals, axis=1)
        mm = np.mean(load(SEP / "bass.wav")[1] + load(SEP / "drums.wav")[1] + load(SEP / "other.wav")[1], axis=1)
        corr = float(np.corrcoef(vm, mm)[0, 1])
        report = work / "report.txt"
        report.write_text(
            f"source={REF}\nmodel=htdemucs shifts=2 overlap=0.5\n"
            f"music_stems=bass+drums+other (vocals excluded)\n"
            f"music_gain_db={MUSIC_GAIN_DB}\nduck_db={DUCK_DB}\n"
            f"mute={MUTE_START:.6f}-{MUTE_END:.3f}\nresume_fade={RESUME_FADE}\n"
            f"nonvocal_vocal_corr={corr:.6f}\nmusic_rms_db={rms_db(music):.2f}\n"
            f"mix_peak_before_limiter={peak_before:.6f}\n"
            f"old_container_sha={old_video_sha}\n", encoding="utf-8")
        staging = OUT.with_suffix(".bgm.tmp.mp4")
        staging.write_bytes(candidate.read_bytes())
        staging.replace(OUT)
        (OUT.parent / "bgm_replacement_report.txt").write_text(report.read_text(), encoding="utf-8")

    print("[4/4] complete")
    print(f"prototype_sha256={sha(OUT)}")
    print(f"reference_sha256={sha(REF)}")


if __name__ == "__main__":
    main()
