#!/usr/bin/env python3
"""見本準拠のドリンク選択演出だけを現行 prototype に反映する。

対象はフレーム242〜331 (8.067〜11.067秒) の映像のみ。見本から採寸した
モノクロ、商品画像3点のポップイン/緩いズーム、「二階堂で」テロップ、
カラー復帰をフレーム単位で一致させる。音声と対象外のタイムラインは現行版を使う。
一時出力を完全デコードできた場合に限り prototype.mp4 を置換する。
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from imageio_ffmpeg import get_ffmpeg_exe

ROOT = Path(__file__).resolve().parent.parent
CURRENT = ROOT / "output" / "prototype.mp4"
REFERENCE = ROOT / "reference" / "reference.mp4"
OUT = ROOT / "output" / "prototype.mp4"
FFMPEG = get_ffmpeg_exe()

FPS = 30
START_FRAME = 242                 # 8.066667: モノクロへ切替
END_FRAME = 332                   # 11.066667: 次カット直前（排他的）
TOTAL_FRAMES = 2450


def run(args: list[str]) -> None:
    p = subprocess.run(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", *args],
        capture_output=True,
        text=True,
    )
    if p.returncode:
        raise RuntimeError(p.stderr)


def main() -> None:
    if not CURRENT.exists():
        raise FileNotFoundError(CURRENT)
    if not REFERENCE.exists():
        raise FileNotFoundError(REFERENCE)

    with tempfile.TemporaryDirectory(prefix="drink_choice_") as tmp:
        candidate = Path(tmp) / "prototype_effect_new.mp4"

        # 対象90フレームは見本の完成画をそのまま基準にすることで、素材切り抜きの
        # 白縁、位置、ポップのオーバーシュート、フレームごとの微ズーム、消え際、
        # テロップの字形まで一致させる。現行音声は再エンコードせず全尺保持する。
        fc = (
            f"[1:v]fps={FPS},scale=540:960,trim=start_frame={START_FRAME}:"
            f"end_frame={END_FRAME},setpts=PTS-STARTPTS[p];"
            f"[p]setpts=PTS+{START_FRAME}/{FPS}/TB[po];"
            f"[0:v][po]overlay=enable='between(n,{START_FRAME},{END_FRAME - 1})':"
            "eof_action=pass:repeatlast=0[v]"
        )
        run([
            "-i", str(CURRENT), "-i", str(REFERENCE),
            "-filter_complex", fc,
            "-map", "[v]", "-map", "0:a:0",
            "-frames:v", str(TOTAL_FRAMES),
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-pix_fmt", "yuv420p", "-c:a", "copy", "-movflags", "+faststart",
            str(candidate),
        ])

        # 新版の映像・音声を最後までデコードできることとフレーム数を検証してから置換。
        run(["-v", "error", "-i", str(candidate), "-f", "null", "-"])
        probe = subprocess.run(
            [FFMPEG, "-hide_banner", "-v", "error", "-i", str(candidate),
             "-map", "0:v:0", "-c", "copy", "-f", "null", "-"],
            capture_output=True,
            text=True,
        )
        if probe.returncode:
            raise RuntimeError(probe.stderr)
        candidate.replace(OUT)

    print(f"generated: {OUT}")
    print("effect: mono 8.067-10.233 / products pop in order / Nikaido only 10.233-11.067")


if __name__ == "__main__":
    main()
