#!/usr/bin/env python3
"""思案シーンの素材会話を、見本の思案効果音だけに置き換える。

映像はストリームコピーし、音声の 8.067〜9.980 秒だけを変更する。
8.067〜9.800 秒は見本音声（思案効果音）を -4 dB で使用し、末尾 180 ms
を等電力フェードアウトする。9.980 秒以降の「二階堂で」を含む会話と、
それ以外の全区間は現行 prototype の音声を維持する。
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from imageio_ffmpeg import get_ffmpeg_exe

ROOT = Path(__file__).resolve().parent.parent
CURRENT = ROOT / "output" / "prototype.mp4"
REFERENCE = ROOT / "reference" / "reference.mp4"
OUT = CURRENT
FFMPEG = get_ffmpeg_exe()

START = 8.066667
SFX_END = 9.800
REPLACE_END = 9.980
SFX_GAIN_DB = -4.0


def run(args: list[str]) -> None:
    process = subprocess.run(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", *args],
        capture_output=True,
        text=True,
    )
    if process.returncode:
        raise RuntimeError(process.stderr)


def main() -> None:
    if not CURRENT.exists():
        raise FileNotFoundError(CURRENT)
    if not REFERENCE.exists():
        raise FileNotFoundError(REFERENCE)

    with tempfile.TemporaryDirectory(prefix="thinking_sfx_") as tmp:
        candidate = Path(tmp) / "prototype_new.mp4"

        # 現行音声を3分割し、中央を見本の思案効果音へ完全置換する。
        # SFX_END〜REPLACE_END は無音なので、元の「なににしようかな〜」は
        # 「二階堂で」が始まる直前まで残らない。
        filter_complex = (
            f"[0:a]atrim=start=0:end={START:.6f},asetpts=PTS-STARTPTS[a0];"
            f"[1:a]atrim=start={START:.6f}:end={SFX_END:.6f},asetpts=PTS-STARTPTS,"
            f"volume={SFX_GAIN_DB}dB,afade=t=out:st={SFX_END - START - (REPLACE_END - SFX_END):.6f}:"
            f"d={REPLACE_END - SFX_END:.6f}:curve=qsin,apad=pad_dur={REPLACE_END - SFX_END:.6f},"
            f"atrim=end={REPLACE_END - START:.6f}[sfx];"
            f"[0:a]atrim=start={REPLACE_END:.6f},asetpts=PTS-STARTPTS[a2];"
            "[a0][sfx][a2]concat=n=3:v=0:a=1[a]"
        )
        run([
            "-i", str(CURRENT), "-i", str(REFERENCE),
            "-filter_complex", filter_complex,
            "-map", "0:v:0", "-map", "[a]",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
            "-movflags", "+faststart", str(candidate),
        ])

        # 完全デコードできた新版だけを採用する。replace は同一ファイルシステム上で
        # 原子的に行われるため、検証前に旧版を失わない。
        run(["-v", "error", "-i", str(candidate), "-f", "null", "-"])
        candidate.replace(OUT)

    print(f"generated: {OUT}")
    print(f"removed dialogue: {START:.3f}-{REPLACE_END:.3f}s")
    print(f"sfx: reference {START:.3f}-{SFX_END:.3f}s, {SFX_GAIN_DB:.1f} dB, 180 ms fade-out")


if __name__ == "__main__":
    main()
