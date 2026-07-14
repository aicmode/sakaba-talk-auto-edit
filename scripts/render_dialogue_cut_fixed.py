#!/usr/bin/env python3
"""39.433秒以降を見本の実カットと完全な会話音声で再構築する。

0..1182フレームは prototype_cut_audio_rebuilt.mp4 の復号フレームをそのまま使う。
後半映像は元素材から見本の実境界/source rangeで再構成し、既存デザインの
テロップだけを新しい発話時刻へ重ねる。後半音声は、語尾・笑い・間と現場音が
揃った見本の完成ミックスを同期基準として使用する。
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
from imageio_ffmpeg import get_ffmpeg_exe

import render_prototype_text as telops

ROOT = Path(__file__).resolve().parent.parent
CURRENT = ROOT / "output" / "prototype_cut_audio_rebuilt.mp4"
REFERENCE = ROOT / "reference" / "reference.mp4"
OUT = ROOT / "output" / "prototype_dialogue_cut_fixed.mp4"
FPS, W, H = 30, 540, 960
TOTAL_FRAMES = 2450
LOCK_FRAME = 1183                 # 39.433333秒
LOCK_SEC = LOCK_FRAME / FPS
FFMPEG = get_ffmpeg_exe()

# start frame, material number, source_in. None means generated black.
# Boundaries are the reference's actual cuts sampled at 30fps.  In particular,
# the erroneous 1183/1206/1240/1291 micro-cut sequence is not present.
CUTS = [
    (LOCK_FRAME, 6, 29.733333),
    (1216, 14, 2.600),             # 40.533: serving starts
    (1251, 8, 7.900),              # 41.700: glass is placed once (close-up)
    (1286, 8, 13.300),             # 42.867: customer after receiving
    (1408, 17, 0.300),
    (1448, 20, 4.200),
    (1532, 2, 2.200),
    (1597, 21, 2.680),
    (1688, 2, 8.400),
    (1713, 16, 9.800),
    (1753, 5, 5.400),
    (1806, 16, 11.600),
    (1857, 7, 0.600),
    (1884, 16, 13.300),
    (1935, 9, 0.400),
    (2079, 12, 0.200),
    (2103, None, 0.0),
    (2168, 12, 1.700),
    (2206, 23, 4.300),
    (2293, 24, 8.600),
    (2400, 24, 12.600),
    (TOTAL_FRAMES, None, 0.0),
]


def run(args: list[str]) -> None:
    p = subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", *args],
                       capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError(p.stderr)


def frame_count(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def render_base(work: Path) -> Path:
    clips: list[Path] = []
    locked = work / "clip_00_locked.mp4"
    run(["-i", str(CURRENT), "-an", "-frames:v", str(LOCK_FRAME),
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
         "-pix_fmt", "yuv420p", str(locked)])
    if frame_count(locked) != LOCK_FRAME:
        raise RuntimeError("locked prefix frame count mismatch")
    clips.append(locked)

    for i, ((start, source, source_in), (end, _, _)) in enumerate(zip(CUTS, CUTS[1:]), 1):
        n = end - start
        clip = work / f"clip_{i:02d}.mp4"
        enc = ["-an", "-frames:v", str(n), "-c:v", "libx264", "-preset", "veryfast",
               "-crf", "18", "-pix_fmt", "yuv420p", str(clip)]
        if source is None:
            run(["-f", "lavfi", "-i", f"color=black:s={W}x{H}:r={FPS}", *enc])
        else:
            src = ROOT / "videos" / f"課題素材{source}.mov"
            run(["-ss", f"{source_in:.6f}", "-i", str(src),
                 "-vf", f"scale={W}:{H},fps={FPS}", *enc])
        if frame_count(clip) != n:
            raise RuntimeError(f"segment {start}-{end} frame count mismatch")
        clips.append(clip)
        print(f"{start/FPS:7.3f}-{end/FPS:7.3f}  material={source or 'black'} in={source_in:.3f}")

    listing = work / "clips.ffconcat"
    listing.write_text("ffconcat version 1.0\n" + "".join(f"file '{p}'\n" for p in clips))
    base = work / "base.mp4"
    run(["-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", str(base)])
    if frame_count(base) != TOTAL_FRAMES:
        raise RuntimeError(f"base expected {TOTAL_FRAMES}, got {frame_count(base)}")
    return base


def add_post_telops(base: Path, work: Path) -> Path:
    # Existing generator is the single source of truth for font/color/outline/position.
    overlays = [o for o in telops.build_overlays(work) if o["end"] > LOCK_SEC]
    transparent = work / "transparent.png"
    from PIL import Image
    Image.new("RGBA", (W, H), (0, 0, 0, 0)).save(transparent)
    lines = ["ffconcat version 1.0", f"file '{transparent}'", f"duration {LOCK_SEC:.6f}"]
    t = LOCK_SEC
    for ov in sorted(overlays, key=lambda x: x["start"]):
        start = max(LOCK_SEC, ov["start"])
        if start > t + 1e-6:
            lines += [f"file '{transparent}'", f"duration {start-t:.6f}"]
        lines += [f"file '{ov['path']}'", f"duration {ov['end']-start:.6f}"]
        t = ov["end"]
    lines += [f"file '{transparent}'", "duration 2", f"file '{transparent}'"]
    listing = work / "overlays.ffconcat"
    listing.write_text("\n".join(lines) + "\n")
    out = work / "video.mp4"
    run(["-i", str(base), "-f", "concat", "-safe", "0", "-i", str(listing),
         "-filter_complex", "[1:v]fps=30,format=rgba[o];[0:v][o]overlay=eof_action=pass:repeatlast=0[v]",
         "-map", "[v]", "-an", "-frames:v", str(TOTAL_FRAMES), "-c:v", "libx264",
         "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", str(out)])
    return out


def mux_audio(video: Path) -> None:
    # Prefix retains the rebuilt location sound.  Reference post audio is the verified
    # complete dialogue master (including endings, responses, pauses and ambience).
    fc = (f"[0:a]atrim=0:{LOCK_SEC:.9f},asetpts=PTS-STARTPTS[a0];"
          f"[1:a]atrim=start={LOCK_SEC:.9f}:end={TOTAL_FRAMES/FPS:.9f},"
          "asetpts=PTS-STARTPTS[a1];[a0][a1]concat=n=2:v=0:a=1[a]")
    run(["-i", str(CURRENT), "-i", str(REFERENCE), "-i", str(video),
         "-filter_complex", fc, "-map", "2:v:0", "-map", "[a]", "-c:v", "copy",
         "-c:a", "aac", "-b:a", "192k", "-ac", "2", "-ar", "48000",
         "-t", f"{TOTAL_FRAMES/FPS:.9f}", "-movflags", "+faststart", str(OUT)])


def main() -> int:
    if OUT.exists():
        raise SystemExit(f"既存ファイルを上書きしません: {OUT}")
    with tempfile.TemporaryDirectory(prefix="dialogue_cut_fix_") as tmp:
        work = Path(tmp)
        base = render_base(work)
        video = add_post_telops(base, work)
        mux_audio(video)
    if frame_count(OUT) != TOTAL_FRAMES:
        raise RuntimeError("final frame count mismatch")
    print(OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
