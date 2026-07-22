#!/usr/bin/env python3
"""見本タイムラインを基準に、指定差分だけを残した提出版を生成する。"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np

import rebuild_prototype as rb

ROOT = Path(__file__).resolve().parent.parent
REF = ROOT / "reference" / "reference.mp4"
DESIGN = ROOT / "output" / "prototype.mp4"
OUT = ROOT / "output" / "final_submission.mp4"
FFMPEG, SR = rb.FFMPEG, rb.SR


def call(args: list[str]) -> None:
    p = subprocess.run(args, capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError(p.stderr)


def stereo_audio(path: Path) -> np.ndarray:
    p = subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-i", str(path),
                        "-vn", "-ac", "2", "-ar", str(SR), "-f", "f32le", "pipe:1"],
                       capture_output=True)
    if p.returncode:
        raise RuntimeError(p.stderr.decode(errors="replace"))
    return np.frombuffer(p.stdout, np.float32).reshape(-1, 2).astype(np.float64)


def make_audio(work: Path) -> Path:
    """見本音声を維持し、8.067〜9.95秒の効果音だけ独自音へ置換する。"""
    mix = stereo_audio(REF)[:rb.TOTAL_SAMPLES].copy()
    a, b = round(rb.SFX_START * SR), round(9.95 * SR)
    # 素材由来の無発話ルームトーンに、独自マリンバSFXを重ねる。
    room = rb.roomtone_tile()
    bed = np.tile(room, int(np.ceil((b - a) / len(room))))[:b-a] * .55
    sfx = rb.synth_thinking_sfx(rb.TOTAL_SAMPLES)[a:b]
    replacement = np.repeat((bed + sfx)[:, None], 2, axis=1)
    xf = round(.025 * SR)
    ramp = np.linspace(0, 1, xf)[:, None]
    replacement[:xf] = mix[a:a+xf] * (1-ramp) + replacement[:xf] * ramp
    replacement[-xf:] = replacement[-xf:] * (1-ramp) + mix[b-xf:b] * ramp
    mix[a:b] = replacement
    raw = work / "final_audio.f32"
    raw.write_bytes(np.clip(mix, -.999, .999).astype(np.float32).tobytes())
    audio = work / "final_audio.m4a"
    call([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-f", "f32le",
          "-ar", str(SR), "-ac", "2", "-i", str(raw), "-c:a", "aac", "-b:a", "256k",
          str(audio)])
    return audio


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="sakaba_final_") as td:
        work = Path(td)
        video = work / "final_video.mp4"
        # 通常映像は見本そのものを30fps化。字幕帯だけ新デザイン版から移植する。
        # 特殊テロップ区間は帯外にも文字があるため、その区間のみデザイン版を採用。
        band_ranges = [
            (0.92, 8.07), (10.23, 15.13),
            (1206/30, 1290/30), (1418/30, 1704/30),
            (1806/30, 1856/30), (1978/30, 2103/30), (2168/30, 2206/30),
        ]
        band_enable = "+".join(f"between(t\\,{a:.6f}\\,{b:.6f})" for a, b in band_ranges)
        fc = (
            "[0:v]fps=30,scale=540:960,setsar=1[r];"
            "[1:v]fps=30,scale=540:960,setsar=1,split=5[d0][d1][d2][d3][d4];"
            "[d0]crop=540:120:0:400[band];"
            f"[r][band]overlay=0:400:enable='{band_enable}'[v0];"
            "[v0][d1]overlay=0:0:enable=between(t\\,61.866667\\,62.8)[v1];"
            "[v1][d2]overlay=0:0:enable=between(t\\,63.566667\\,64.9)[v2];"
            "[v2][d3]overlay=0:0:enable=between(t\\,70.1\\,72.266667)[v3];"
            "[v3][d4]overlay=0:0:enable=between(t\\,80.0\\,81.666667)[v]"
        )
        call([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-i", str(REF),
              "-i", str(DESIGN), "-filter_complex", fc, "-map", "[v]", "-an",
              "-frames:v", str(rb.TOTAL_FRAMES), "-c:v", "libx264", "-preset", "medium",
              "-crf", "18", "-pix_fmt", "yuv420p", str(video)])
        audio = make_audio(work)
        candidate = work / "final.mp4"
        call([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-i", str(video),
              "-i", str(audio), "-map", "0:v:0", "-map", "1:a:0", "-c", "copy",
              "-t", f"{rb.TOTAL_SEC:.7f}", "-movflags", "+faststart", str(candidate)])
        call([FFMPEG, "-v", "error", "-i", str(candidate), "-f", "null", "-"])
        staging = OUT.with_suffix(".staging.mp4")
        staging.write_bytes(candidate.read_bytes())
        staging.replace(OUT)
    print(f"generated: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
