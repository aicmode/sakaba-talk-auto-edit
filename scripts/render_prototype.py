#!/usr/bin/env python3
"""低解像度試作動画の生成(フェーズ3・試作)。

final_editing_plan.json の video_track / audio_track から
output/prototype.mp4 (540x960 / 30fps) を組み立てる。

この試作に含めるもの:
- 映像43カット(黒画面カット含む)のカット構成
- 会話音声23セグメントの絶対時刻配置

この試作に含めないもの(本レンダリングで追加):
- テロップ・字幕・商品画像オーバーレイ
- ズームイン/アウト・ズームクロップ・モノクロ・クロスフェード等のエフェクト
  (カット22のクロスフェードも通常カット扱い)
- BGM・効果音

タイムライン精度: 各カットのフレーム数を round(ref_out*30)-round(ref_in*30) で
確定させ、丸め誤差が累積しないようにする(音声は絶対時刻配置のため必須)。
"""

import json
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np
from imageio_ffmpeg import get_ffmpeg_exe

ROOT = Path(__file__).resolve().parent.parent
PLAN = ROOT / "analysis" / "final_editing_plan.json"
OUT = ROOT / "output" / "prototype.mp4"

W, H, FPS = 540, 960, 30
SR = 48000  # 音声サンプルレート
FFMPEG = get_ffmpeg_exe()


def run(args: list[str]) -> None:
    proc = subprocess.run(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", *args],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {' '.join(args)}\n{proc.stderr}")


def render_clips(video_track: list[dict], work: Path, warnings: list[str]) -> list[Path]:
    """各カットを 540x960/30fps の中間クリップに書き出す。"""
    clips = []
    for i, cut in enumerate(video_track, 1):
        n_frames = round(cut["ref_out"] * FPS) - round(cut["ref_in"] * FPS)
        dur = n_frames / FPS
        clip = work / f"clip_{i:02d}.mp4"
        enc = ["-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
               "-pix_fmt", "yuv420p", str(clip)]
        if cut["source"] is None:
            # カット39: 黒画面(テロップは本レンダリングで追加)
            run(["-f", "lavfi", "-i", f"color=black:s={W}x{H}:r={FPS}",
                 "-frames:v", str(n_frames), *enc])
        else:
            src = ROOT / cut["source"]
            if not src.exists():
                raise FileNotFoundError(src)
            src_dur = cut["source_out"] - cut["source_in"]
            if abs(src_dur - (cut["ref_out"] - cut["ref_in"])) > 0.02:
                warnings.append(f"カット{i}: 素材尺と見本尺の差が {src_dur - (cut['ref_out']-cut['ref_in']):+.3f}s")
            run(["-ss", f"{cut['source_in']:.3f}", "-i", str(src),
                 "-vf", f"scale={W}:{H},fps={FPS}",
                 "-frames:v", str(n_frames), *enc])
        clips.append(clip)
        print(f"  カット{i:2d}/{len(video_track)} {dur:6.3f}s  {cut['source'] or '(黒画面)'}")
    return clips


def concat_clips(clips: list[Path], work: Path) -> Path:
    listfile = work / "concat.txt"
    listfile.write_text("".join(f"file '{c}'\n" for c in clips))
    video = work / "video.mp4"
    run(["-f", "concat", "-safe", "0", "-i", str(listfile), "-c", "copy", str(video)])
    return video


def build_audio(audio_track: list[dict], total_dur: float, work: Path,
                warnings: list[str]) -> Path:
    """会話セグメントを 48kHz モノラルで絶対時刻に加算ミックスする。"""
    buf = np.zeros(int(round(total_dur * SR)), dtype=np.float64)
    fade = int(SR * 0.01)  # クリックノイズ防止の10msフェード
    for i, seg in enumerate(audio_track, 1):
        dur = seg["source_out"] - seg["source_in"]
        wav_path = work / f"seg_{i:02d}.wav"
        run(["-ss", f"{seg['source_in']:.3f}", "-i", str(ROOT / seg["source"]),
             "-t", f"{dur:.3f}", "-vn", "-ac", "1", "-ar", str(SR),
             "-c:a", "pcm_s16le", str(wav_path)])
        with wave.open(str(wav_path)) as w:
            data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        samples = data.astype(np.float64) / 32768.0
        if len(samples) > 2 * fade:
            samples[:fade] *= np.linspace(0, 1, fade)
            samples[-fade:] *= np.linspace(1, 0, fade)
        start = int(round(seg["ref_at"] * SR))
        end = start + len(samples)
        if end > len(buf):
            warnings.append(
                f"音声A{i}({seg.get('line','')[:20]}…): 終端が動画尺を {(end - len(buf)) / SR:.2f}s 超過→末尾トリム")
            samples = samples[: len(buf) - start]
            end = len(buf)
        buf[start:end] += samples
    peak = np.max(np.abs(buf))
    if peak > 0.99:
        warnings.append(f"音声ピーク {peak:.2f} → 0.95 に正規化")
        buf *= 0.95 / peak
    master = work / "audio.wav"
    with wave.open(str(master), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((buf * 32767).astype(np.int16).tobytes())
    return master


def main() -> None:
    plan = json.loads(PLAN.read_text())
    total_dur = plan["output"]["duration_sec"]
    warnings: list[str] = []
    OUT.parent.mkdir(exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="prototype_") as tmp:
        work = Path(tmp)
        print("[1/4] 映像カットを書き出し中…")
        clips = render_clips(plan["video_track"], work, warnings)
        print("[2/4] カットを連結中…")
        video = concat_clips(clips, work)
        print("[3/4] 音声トラックを構築中…")
        audio = build_audio(plan["audio_track"], total_dur, work, warnings)
        print("[4/4] 映像と音声を多重化中…")
        run(["-i", str(video), "-i", str(audio),
             "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
             "-shortest", str(OUT)])

    size_mb = OUT.stat().st_size / 1e6
    print(f"\n完了: {OUT.relative_to(ROOT)} ({size_mb:.1f} MB)")
    if warnings:
        print("\n警告:")
        for msg in warnings:
            print(f"  - {msg}")


if __name__ == "__main__":
    sys.exit(main())
