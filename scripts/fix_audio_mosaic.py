#!/usr/bin/env python3
"""prototype_audio_mosaic.mp4 の音声修正版を生成する。

映像: output/prototype_audio_mosaic.mp4 のビデオストリーム(テロップ+冒頭
モザイク込み)をストリームコピーで維持する。
音声: output/prototype_text.mp4 の会話音声トラック(AAC 48kHz mono)を採用
する。ユーザー要件「prototype_text.mp4 / prototype.mp4 の会話音声を必ず維持」
に従い、見本準拠で再配置した音声トラックではなく原本の音声を使う。

安全策:
- 多重化の前に、音声ソースに音声ストリームが存在することを ffprobe で検証
- 出力後にも ffprobe で音声ストリーム(aac / 48kHz / 1-2ch)を検証し、
  不合格なら出力ファイルを削除して異常終了する
- 既存ファイルは一切上書きしない(出力は prototype_audio_mosaic_fixed.mp4)
- moov を先頭へ移動(+faststart)し、書き込み途中や一部プレーヤーで
  映像のみ再生される事象を防ぐ
"""

import json
import math
import subprocess
import sys
from pathlib import Path

from imageio_ffmpeg import get_ffmpeg_exe
from static_ffmpeg import run as static_run

ROOT = Path(__file__).resolve().parent.parent
VIDEO_SRC = ROOT / "output" / "prototype_audio_mosaic.mp4"
AUDIO_SRC = ROOT / "output" / "prototype_text.mp4"
OUT = ROOT / "output" / "prototype_audio_mosaic_fixed.mp4"

FFMPEG = get_ffmpeg_exe()
_, FFPROBE = static_run.get_or_fetch_platform_executables_else_raise()


def ffprobe_streams(path: Path) -> dict:
    proc = subprocess.run(
        [FFPROBE, "-v", "error", "-show_streams", "-show_format",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    return json.loads(proc.stdout)


def audio_stream(info: dict) -> dict | None:
    return next((s for s in info["streams"] if s["codec_type"] == "audio"), None)


def check_audio(path: Path, label: str) -> dict:
    info = ffprobe_streams(path)
    a = audio_stream(info)
    if a is None:
        raise SystemExit(f"NG: {label} に音声ストリームがありません: {path}")
    print(f"  {label}: {a['codec_name']} {a['sample_rate']}Hz "
          f"{a['channels']}ch dur={a.get('duration', '?')}s")
    return info


def measure_rms(path: Path) -> tuple[float, float]:
    """全音声サンプルの RMS(dBFS)とピーク(dBFS)を astats で測る。"""
    proc = subprocess.run(
        [FFMPEG, "-hide_banner", "-i", str(path), "-map", "0:a",
         "-af", "astats=measure_perchannel=none", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    rms = peak = math.nan
    for line in proc.stderr.splitlines():
        if "RMS level dB" in line:
            rms = float(line.split(":")[-1])
        elif "Peak level dB" in line:
            peak = float(line.split(":")[-1])
    return rms, peak


def main() -> None:
    for p in (VIDEO_SRC, AUDIO_SRC):
        if not p.exists():
            raise SystemExit(f"NG: 入力がありません: {p}")

    print("[1/3] 出力前検証: 音声ソースのストリーム確認(ffprobe)")
    check_audio(AUDIO_SRC, "音声ソース(prototype_text.mp4)")

    print("[2/3] 多重化: 映像=audio_mosaic(copy) + 音声=text(copy)")
    proc = subprocess.run(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
         "-i", str(VIDEO_SRC), "-i", str(AUDIO_SRC),
         "-map", "0:v:0", "-map", "1:a:0",
         "-c:v", "copy", "-c:a", "copy",
         "-movflags", "+faststart", str(OUT)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise SystemExit(f"NG: ffmpeg 失敗\n{proc.stderr}")

    print("[3/3] 出力検証(ffprobe + RMS)")
    info = check_audio(OUT, "出力(prototype_audio_mosaic_fixed.mp4)")
    a = audio_stream(info)
    if a["codec_name"] != "aac" or a["sample_rate"] != "48000" \
            or a["channels"] not in (1, 2):
        OUT.unlink(missing_ok=True)
        raise SystemExit("NG: 出力音声が要件(AAC/48kHz/1-2ch)を満たさないため削除しました")

    rms, peak = measure_rms(OUT)
    if not (rms > -60):  # 実音声なら -60dBFS より十分大きい
        OUT.unlink(missing_ok=True)
        raise SystemExit(f"NG: 出力音声がほぼ無音(RMS {rms} dBFS)のため削除しました")

    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    print(f"\nOK: {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1e6:.1f} MB)")
    print(f"  動画尺: {float(info['format']['duration']):.2f}s "
          f"(映像 {float(v['duration']):.2f}s / 音声 {float(a['duration']):.2f}s)")
    print(f"  音声: {a['codec_name']} {a['sample_rate']}Hz {a['channels']}ch")
    print(f"  音声RMS: {rms:.1f} dBFS / ピーク: {peak:.1f} dBFS")


if __name__ == "__main__":
    sys.exit(main())
