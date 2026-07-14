#!/usr/bin/env python3
"""見本の実フレーム照合に基づくカット修正版を新規生成する。"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
from imageio_ffmpeg import get_ffmpeg_exe

import render_prototype_text as telops
import render_prototype_audio_mosaic as mosaic

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output" / "prototype_cut_fixed.mp4"
AUDIO = ROOT / "output" / "prototype_audio_mosaic_fixed.mp4"
FPS, W, H = 30, 540, 960
FFMPEG = get_ffmpeg_exe()

# start_frame, source number (None=black), source_in seconds.
# Boundaries are the first 30-fps output frame belonging to the new shot.  They
# were obtained by inspecting the 60-fps reference's first changed frame and
# sampling its effective 30-fps cadence (ceil(reference_frame / 2)).
CUTS = [
    (0, 4, .300), (28, 10, 1.100), (77, 4, 2.550), (122, 4, 4.100),
    (202, 11, .400), (242, 25, 2.800), (307, 25, 6.300),
    (327, 11, 7.000), (351, 3, .200), (387, 22, 2.900),
    (419, 19, .400), (454, 15, 4.600), (496, 15, 6.200),
    (543, 15, 8.300), (578, 15, 13.300), (622, 15, 15.000),
    (645, 13, .700), (724, 1, 1.200), (812, 1, 4.900),
    (862, 6, .900), (920, 6, 3.200), (1064, 6, 17.500),
    (1116, 6, 27.500),
    # Reference-only micro cuts around serving (missing from the old plan).
    (1183, 8, 4.500), (1206, 14, 3.533), (1240, 8, 7.900),
    (1291, 8, 16.263), (1370, 8, 18.234),
    (1408, 17, .300), (1459, 20, 6.233), (1532, 2, 3.417),
    (1597, 21, 3.100), (1688, 2, 8.867),
    # Reference returns to an earlier take after the close mixing shot.
    (1705, 16, 9.800), (1753, 5, 4.367), (1806, 16, 8.833),
    (1856, 7, .683), (1885, 16, 12.400),
    # Two internal jumps in material 9 reproduce the reference action cadence.
    (1978, 9, 2.017), (2029, 9, 4.166), (2069, 12, 1.234),
    (2103, None, 0.0), (2168, 12, 2.483), (2206, 23, 5.100),
    (2293, 24, 8.600), (2399, 24, 12.180), (2450, None, 0.0),
]


def run(args: list[str]) -> None:
    p = subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", *args],
                       capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError(p.stderr)


def render_base(work: Path) -> Path:
    clips: list[Path] = []
    for i, ((start, source, source_in), (end, _, _)) in enumerate(zip(CUTS, CUTS[1:])):
        n = end - start
        clip = work / f"clip_{i:02d}.mp4"
        enc = ["-an", "-frames:v", str(n), "-c:v", "libx264", "-preset", "veryfast",
               "-crf", "21", "-pix_fmt", "yuv420p", str(clip)]
        if source is None:
            run(["-f", "lavfi", "-i", f"color=black:s={W}x{H}:r={FPS}", *enc])
        else:
            src = ROOT / "videos" / f"課題素材{source}.mov"
            run(["-ss", f"{source_in:.3f}", "-i", str(src),
                 "-vf", f"scale={W}:{H},fps={FPS}", *enc])
        actual = int(cv2.VideoCapture(str(clip)).get(cv2.CAP_PROP_FRAME_COUNT))
        if actual != n:
            raise RuntimeError(f"clip {i} expected {n} frames, got {actual}: source={source}")
        clips.append(clip)
        print(f"  {start:4d}-{end-1:4d}  material={source or 'black'} in={source_in:.3f}")
    listing = work / "clips.ffconcat"
    listing.write_text("ffconcat version 1.0\n" + "".join(f"file '{p}'\n" for p in clips))
    base = work / "base.mp4"
    run(["-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", base])
    return base


def add_telops(base: Path, work: Path) -> Path:
    overlays = telops.build_overlays(work)
    transparent = work / "transparent.png"
    from PIL import Image
    Image.new("RGBA", (W, H), (0, 0, 0, 0)).save(transparent)
    lines = ["ffconcat version 1.0"]
    t = 0.0
    for ov in sorted(overlays, key=lambda x: x["start"]):
        if ov["start"] > t + 1e-6:
            lines += [f"file '{transparent}'", f"duration {ov['start'] - t:.4f}"]
        lines += [f"file '{ov['path']}'", f"duration {ov['end'] - ov['start']:.4f}"]
        t = ov["end"]
    lines += [f"file '{transparent}'", "duration 2.0", f"file '{transparent}'"]
    listing = work / "overlays.ffconcat"
    listing.write_text("\n".join(lines) + "\n")
    out = work / "text.mp4"
    run(["-i", str(base), "-f", "concat", "-safe", "0", "-i", str(listing),
         "-filter_complex", "[1:v]fps=30,format=rgba[o];[0:v][o]overlay=0:0:eof_action=pass:repeatlast=0[v]",
         "-map", "[v]", "-an", "-frames:v", "2450", "-c:v", "libx264",
         "-preset", "veryfast", "-crf", "21", "-pix_fmt", "yuv420p", str(out)])
    return out


def add_mosaic_and_audio(text_video: Path, work: Path) -> None:
    boxes = mosaic.detect_face_boxes()
    cap = cv2.VideoCapture(str(text_video))
    raw = work / "mosaic.mp4"
    cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
           "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "pipe:0",
           "-an", "-frames:v", "2450", "-c:v", "libx264", "-preset", "veryfast",
           "-crf", "21", "-pix_fmt", "yuv420p", str(raw)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    count = 0
    while count < 2450:
        ok, frame = cap.read()
        if not ok:
            break
        if count < len(boxes):
            mosaic.pixelate(frame, boxes[count])
        proc.stdin.write(frame.tobytes())
        count += 1
    cap.release()
    proc.stdin.close()
    err = proc.stderr.read().decode()
    if proc.wait() or count != 2450:
        raise RuntimeError(f"mosaic encode failed ({count} frames): {err}")
    run(["-i", str(raw), "-i", str(AUDIO), "-map", "0:v:0", "-map", "1:a:0",
         "-c:v", "copy", "-c:a", "copy", "-t", "81.6666667",
         "-movflags", "+faststart", str(OUT)])


def main() -> int:
    if OUT.exists():
        raise SystemExit(f"既存ファイルを上書きしません: {OUT}")
    if not AUDIO.exists():
        raise FileNotFoundError(AUDIO)
    with tempfile.TemporaryDirectory(prefix="cut_fixed_") as tmp:
        work = Path(tmp)
        print("[1/3] corrected cuts")
        base = render_base(work)
        print("[2/3] existing telops")
        text_video = add_telops(base, work)
        print("[3/3] existing opening mosaic + original fixed audio")
        add_mosaic_and_audio(text_video, work)
    print(OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
