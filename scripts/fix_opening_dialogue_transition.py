#!/usr/bin/env python3
"""冒頭の「かしこまりました」→「こちらの席へどうぞ」を音画同期で修正する。"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
from imageio_ffmpeg import get_ffmpeg_exe

import fix_opening_audio
import render_prototype_text as telops

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "output" / "prototype.mp4"
VIDEO_TMP = ROOT / "output" / "prototype_video_transition_fixed.mp4"
OUT_TMP = ROOT / "output" / "prototype_transition_new.mp4"
LEVEL_TMP = ROOT / "output" / "prototype_transition_levels.mp4"
OUT = ROOT / "output" / "prototype.mp4"
FFMPEG = get_ffmpeg_exe()
FPS, W, H = 30, 540, 960
START_FRAME, CUT_FRAME, END_FRAME = 77, 145, 202


def run(args: list[str]) -> None:
    p = subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", *args],
                       capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError(p.stderr)


def render_opening_video(work: Path) -> None:
    """該当125フレームだけを素材4から再構成し、修正時刻のテロップを焼く。"""
    raw = work / "opening_raw.mp4"
    # 2.567〜4.833は前ショットを延長し、「こちらの席へどうぞ」実発話(4.83秒)に
    # 合わせて4.833秒(フレーム145)で次ショットへ切る。次ショットの取り出し位置は
    # 旧版(ss4.600/カット137)とソースマッピングが一致するよう 4.600+8/30 とし、
    # フレーム145以降の映像内容とフレーム202以降との連続性を現行版のまま保つ。
    fc = work / "first.mp4"
    sc = work / "second.mp4"
    run(["-ss", "2.550", "-i", str(ROOT / "videos/課題素材4.mov"),
         "-an", "-vf", f"scale={W}:{H},fps={FPS}", "-frames:v", str(CUT_FRAME-START_FRAME),
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", str(fc)])
    run(["-ss", "4.866667", "-i", str(ROOT / "videos/課題素材4.mov"),
         "-an", "-vf", f"scale={W}:{H},fps={FPS}", "-frames:v", str(END_FRAME-CUT_FRAME),
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", str(sc)])
    listing = work / "opening.ffconcat"
    listing.write_text(f"ffconcat version 1.0\nfile '{fc}'\nfile '{sc}'\n")
    run(["-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", str(raw)])

    # 当該3枚だけ修正。その他のテロップ定義には触れない。
    original = telops.BAND_TELOPS
    telops.BAND_TELOPS = [
        (0.0, 0.593, [[("1人です", "w")]]),
        # 「かしこまりました」は新カット(4.833秒)まで保持し、
        # 「こちらの席へどうぞ」は実発話開始(4.83秒)、
        # 「ありがとうございます」は実発話開始(5.83秒)直前に切り替える。
        (0.593, (CUT_FRAME-START_FRAME)/FPS, [[("かしこまりました", "w")]]),
        ((CUT_FRAME-START_FRAME)/FPS, 3.233, [[("こちらの席へどうぞ", "w")]]),
        (3.233, (END_FRAME-START_FRAME)/FPS, [[("ありがとうございます", "w")]]),
    ]
    overlays = telops.build_overlays(work)
    telops.BAND_TELOPS = original
    transparent = work / "transparent.png"
    from PIL import Image
    Image.new("RGBA", (W, H), (0, 0, 0, 0)).save(transparent)
    lines = ["ffconcat version 1.0"]
    t = 0.0
    for ov in overlays:
        if ov["start"] > t:
            lines += [f"file '{transparent}'", f"duration {ov['start']-t:.6f}"]
        lines += [f"file '{ov['path']}'", f"duration {ov['end']-ov['start']:.6f}"]
        t = ov["end"]
    lines += [f"file '{transparent}'", "duration 1", f"file '{transparent}'"]
    ol = work / "overlay.ffconcat"
    ol.write_text("\n".join(lines) + "\n")
    patched = work / "opening_telop.mp4"
    run(["-i", str(raw), "-f", "concat", "-safe", "0", "-i", str(ol),
         "-filter_complex", "[1:v]fps=30,format=rgba[o];[0:v][o]overlay=eof_action=pass:repeatlast=0[v]",
         "-map", "[v]", "-an", "-frames:v", str(END_FRAME-START_FRAME),
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", str(patched)])

    # 中間区間のみ差し替え。前後は現行映像をそのまま内容保持する。
    run(["-i", str(SOURCE), "-i", str(patched),
         "-filter_complex", f"[1:v]setpts=PTS+{START_FRAME}/{FPS}/TB[p];[0:v][p]overlay=enable='between(n,{START_FRAME},{END_FRAME-1})':eof_action=pass[v]",
         "-map", "[v]", "-map", "0:a:0", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
         "-c:a", "copy", "-frames:v", "2450", "-movflags", "+faststart", str(VIDEO_TMP)])


def preserve_effect_levels(work: Path) -> None:
    """現行版で確定済みの3区間の効果音レベルをそのまま再適用する。"""
    points = [
        (0.000, 0.0), (21.500, 0.0), (21.620, -5.0),
        (24.013, -5.0), (24.133, -8.0), (28.613, -8.0), (28.733, 0.0),
        (37.200, 0.0), (37.320, -6.0), (39.880, -6.0), (40.000, 0.0),
    ]
    p = subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-i", str(OUT_TMP),
                        "-vn", "-ac", "2", "-ar", "48000", "-f", "f32le", "pipe:1"],
                       capture_output=True)
    if p.returncode:
        raise RuntimeError(p.stderr.decode(errors="replace"))
    audio = np.frombuffer(p.stdout, dtype=np.float32).copy().reshape(-1, 2)
    times = np.arange(len(audio), dtype=np.float64) / 48000
    pt = np.array([x[0] for x in points] + [len(audio) / 48000])
    pd = np.array([x[1] for x in points] + [0.0])
    audio *= np.power(10.0, np.interp(times, pt, pd) / 20.0)[:, None].astype(np.float32)
    raw = work / "leveled.f32"
    raw.write_bytes(audio.tobytes())
    run(["-i", str(OUT_TMP), "-f", "f32le", "-ar", "48000", "-ac", "2", "-i", str(raw),
         "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
         "-t", "81.666667", "-movflags", "+faststart", str(LEVEL_TMP)])


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)
    with tempfile.TemporaryDirectory(prefix="opening_transition_") as tmp:
        render_opening_video(Path(tmp))
        first = fix_opening_audio.build_first_half(Path(tmp))
        fix_opening_audio.mux(first, OUT_TMP)
        preserve_effect_levels(Path(tmp))
    # 完全デコード検証後だけ現行版を置換する。
    run(["-v", "error", "-i", str(LEVEL_TMP), "-f", "null", "-"])
    LEVEL_TMP.replace(OUT)
    OUT_TMP.unlink(missing_ok=True)
    VIDEO_TMP.unlink(missing_ok=True)
    print(f"generated: {OUT}")


if __name__ == "__main__":
    main()
