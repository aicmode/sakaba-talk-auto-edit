#!/usr/bin/env python3
"""指定された3つの現場効果音だけを下げて prototype.mp4 を生成する。"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np
from imageio_ffmpeg import get_ffmpeg_exe

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "output" / "prototype_dialogue_cut_fixed.mp4"
OUT = ROOT / "output" / "prototype.mp4"
SR = 48000
CHANNELS = 2
FFMPEG = get_ffmpeg_exe()

# 動作カットの境界に合わせた調整値。切替点の120msで滑らかに遷移させる。
# 会話が始まる40.000秒以降は常に0dB（変更なし）。
POINTS = [
    (0.000, 0.0),
    (21.500, 0.0), (21.620, -5.0),                 # 冷蔵庫から取り出す音
    (24.013, -5.0), (24.133, -8.0),               # 製氷機・氷の音
    (28.613, -8.0), (28.733, 0.0),
    (37.200, 0.0), (37.320, -6.0),                 # マドラーで混ぜる音
    (39.880, -6.0), (40.000, 0.0),
]


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

    times = np.arange(len(audio), dtype=np.float64) / SR
    point_t = np.array([p[0] for p in POINTS] + [len(audio) / SR], dtype=np.float64)
    point_db = np.array([p[1] for p in POINTS] + [0.0], dtype=np.float64)
    gain_db = np.interp(times, point_t, point_db)
    audio *= np.power(10.0, gain_db / 20.0)[:, None].astype(np.float32)

    OUT.parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="effect_levels_") as tmp:
        raw = Path(tmp) / "adjusted.f32"
        raw.write_bytes(audio.tobytes())
        run([
            "-i", str(SOURCE), "-f", "f32le", "-ar", str(SR), "-ac", str(CHANNELS),
            "-i", str(raw), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k", "-ar", str(SR), "-ac", str(CHANNELS),
            "-t", "81.666667", "-movflags", "+faststart", str(OUT),
        ])

    print(f"generated: {OUT}")
    print("fridge -5 dB / ice -8 dB / stirring -6 dB (120 ms transitions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
