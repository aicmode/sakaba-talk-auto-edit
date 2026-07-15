#!/usr/bin/env python3
"""冒頭「あっ、一人です」の「あっ」だけを音声から除去して prototype.mp4 を更新する。

- 「あっ」= 出力 2.860〜3.085 秒(素材4とのxcorr照合+ピッチ検定で確定)。
- 同シーンのルームトーン(3.635〜3.890秒)で置換し、両端15msの等パワークロスフェード。
- 「一人です」(3.19秒〜)と映像・他の音声は無変更。映像は stream copy。
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np
from imageio_ffmpeg import get_ffmpeg_exe

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "output" / "prototype.mp4"
NEW = ROOT / "output" / "prototype_new.mp4"
SR = 48000
CHANNELS = 2
FFMPEG = get_ffmpeg_exe()

CUT_IN, CUT_OUT = 2.860, 3.085   # 「あっ」区間(声帯音2.888-3.07+余白)
FILL_IN = 3.635                  # ルームトーン採取開始(〜3.890、無音部)
FADE = 0.015                     # クロスフェード長


def run(args: list[str], *, capture: bool = False) -> subprocess.CompletedProcess:
    p = subprocess.run(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", *args],
        capture_output=capture,
    )
    if p.returncode:
        raise RuntimeError(p.stderr.decode(errors="replace") if capture else "ffmpeg failed")
    return p


def main() -> int:
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)

    decoded = run([
        "-i", str(SOURCE), "-vn", "-ac", str(CHANNELS), "-ar", str(SR),
        "-f", "f32le", "pipe:1",
    ], capture=True).stdout
    audio = np.frombuffer(decoded, dtype=np.float32).copy().reshape(-1, CHANNELS)

    i0 = int(round((CUT_IN - FADE) * SR))
    i1 = int(round((CUT_OUT + FADE) * SR))
    n = i1 - i0
    j0 = int(round((FILL_IN - FADE) * SR))
    fill = audio[j0:j0 + n].copy()

    ramp = np.linspace(0.0, 1.0, int(FADE * 2 * SR), dtype=np.float32)
    fade_in = np.sqrt(ramp)[:, None]
    patch = fill.copy()
    patch[: len(ramp)] = audio[i0:i0 + len(ramp)] * np.sqrt(1 - ramp)[:, None] + fill[: len(ramp)] * fade_in
    patch[-len(ramp):] = audio[i1 - len(ramp):i1] * fade_in + fill[-len(ramp):] * np.sqrt(1 - ramp)[:, None]
    audio[i0:i1] = patch

    with tempfile.TemporaryDirectory(prefix="remove_atto_") as tmp:
        raw = Path(tmp) / "patched.f32"
        raw.write_bytes(audio.tobytes())
        run([
            "-i", str(SOURCE), "-f", "f32le", "-ar", str(SR), "-ac", str(CHANNELS),
            "-i", str(raw), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k", "-ar", str(SR), "-ac", str(CHANNELS),
            "-t", "81.666667", "-movflags", "+faststart", str(NEW),
        ])

    print(f"generated: {NEW}")
    print(f"removed 「あっ」 {CUT_IN:.3f}-{CUT_OUT:.3f}s, room tone from {FILL_IN:.3f}s, {FADE*1000:.0f}ms crossfades")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
