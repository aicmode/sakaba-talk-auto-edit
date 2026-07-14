#!/usr/bin/env python3
"""見本再照合に基づくカット修正 + 元素材音声(環境音)完全復元版を新規生成する。

映像: reference をフレーム照合し直した結果。15.133〜21.5秒は
  棚アップ(素材18ズームクロップ)→ミドル(素材15)→ボトル置きアップ(素材15クロップ)
  →お茶ミドル(素材15の2回目の起き上がり)→お茶アップ(素材15クロップ)。
  素材15の1回目と2回目のしゃがみを両方使っていた重複を排除。
  冷蔵庫(素材13)は0.85秒開始でグラス取り出しまで、氷(素材1)は0.433秒開始で
  蓋開けから、27.067秒でズームクロップに切り替え(見本と同じパンチイン)。

音声: 会話ステム(prototype_cut_fixed の音声、絶対時刻維持)に、映像で使う各素材の
  source_in/source_out と同区間のオリジナル音声を全編分つないだ現場音トラックを加算。
  会話がある間は現場音を -23dB にダッキングして二重会話を防ぐ。境界は12msフェード。
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from imageio_ffmpeg import get_ffmpeg_exe

import render_prototype_text as telops
import render_prototype_audio_mosaic as mosaic

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output" / "prototype_cut_audio_rebuilt.mp4"
STEM_SRC = ROOT / "output" / "prototype_cut_fixed.mp4"
FPS, W, H = 30, 540, 960
SR = 48000
TOTAL_FRAMES = 2450
TOTAL_SEC = TOTAL_FRAMES / FPS  # 81.666667
FFMPEG = get_ffmpeg_exe()

# 見本のズームクロップ(素材ネイティブ1080x1920上の crop w:h:x:y)。
# テンプレート照合(スコア0.986〜0.996)で推定。
CROP_SHELF = (586, 1042, 248, 184)    # 素材18 棚アップ scale1.84
CROP_BOTTLE = (568, 1010, 256, 906)   # 素材15 ボトル置きアップ scale1.90
CROP_TEA = (474, 842, 586, 836)       # 素材15 お茶アップ scale2.28
CROP_ICE = (614, 1090, 0, 374)        # 素材1 氷アップ scale1.76

# start_frame, source number (None=black), source_in seconds, crop or None.
CUTS = [
    (0, 4, .300, None), (28, 10, 1.100, None), (77, 4, 2.550, None),
    (122, 4, 4.100, None), (202, 11, .400, None), (242, 25, 2.800, None),
    (307, 25, 6.300, None), (327, 11, 7.000, None), (351, 3, .200, None),
    (387, 22, 2.900, None),
    # --- ここから見本 13.967〜30.667 秒の再照合結果 ---
    (419, 19, .533, None),          # 引き。旧 0.400 → 0.533
    (454, 18, .000, CROP_SHELF),    # 棚から二階堂を掴むアップ(旧: 素材15ミドル)
    (496, 15, 4.750, None),         # ラベル確認→カウンターへ(旧 4.600)
    (543, 15, 6.316, CROP_BOTTLE),  # ボトルを置き手が離れるアップ(旧: 素材15@8.3 しゃがみ)
    (578, 15, 13.367, None),        # お茶を持ち上げ確認して置くミドル(2回目の起き上がり)
    (622, 15, 14.849, CROP_TEA),    # お茶ボトルのアップ(旧: 素材15@15.0 ミドル)
    (645, 13, .850, None),          # 冷蔵庫: 開けて冷えたグラスを取り出す(旧 0.700)
    (724, 1, .433, None),           # アイスストッカー: 蓋開けから(旧 1.200)
    (812, 1, 3.417, CROP_ICE),      # 氷すくいのパンチイン(旧: フルフレーム@4.9)
    (862, 6, 1.583, None),          # カウンター引き(旧 0.900)
    (920, 6, 3.517, None),          # 連続再生(旧 3.200 は0.3秒巻き戻りだった)
    # --- 以降は既存どおり ---
    (1064, 6, 17.500, None), (1116, 6, 27.500, None),
    (1183, 8, 4.500, None), (1206, 14, 3.533, None), (1240, 8, 7.900, None),
    (1291, 8, 16.263, None), (1370, 8, 18.234, None),
    (1408, 17, .300, None), (1459, 20, 6.233, None), (1532, 2, 3.417, None),
    (1597, 21, 3.100, None), (1688, 2, 8.867, None),
    (1705, 16, 9.800, None), (1753, 5, 4.367, None), (1806, 16, 8.833, None),
    (1856, 7, .683, None), (1885, 16, 12.400, None),
    (1978, 9, 2.017, None), (2029, 9, 4.166, None), (2069, 12, 1.234, None),
    (2103, None, 0.0, None), (2168, 12, 2.483, None), (2206, 23, 5.100, None),
    (2293, 24, 8.600, None), (2399, 24, 12.180, None), (2450, None, 0.0, None),
]

# 黒画面(お会計中)の現場音: 直前素材12の音を途切れず継続、控えめな音量で敷く。
BLACK_AMBIENT = {2103: (12, 2.367, 0.5)}  # start_frame: (source, source_in, gain)

DUCK_GAIN = 0.07          # 会話中の現場音ゲイン(約-23dB)
SEG_FADE = 0.012          # 各素材音声セグメント両端のフェード秒
DUCK_RAMP = 0.12          # ダッキングの遷移秒
SPEECH_THRESH_DB = -55.0  # 会話ステムの有音判定


def run(args: list[str]) -> None:
    p = subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", *args],
                       capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError(p.stderr)


# ---------------------------------------------------------------- video

def render_base(work: Path) -> Path:
    clips: list[Path] = []
    for i, ((start, source, source_in, crop), (end, _, _, _)) in enumerate(
            zip(CUTS, CUTS[1:])):
        n = end - start
        clip = work / f"clip_{i:02d}.mp4"
        enc = ["-an", "-frames:v", str(n), "-c:v", "libx264", "-preset", "veryfast",
               "-crf", "21", "-pix_fmt", "yuv420p", str(clip)]
        if source is None:
            run(["-f", "lavfi", "-i", f"color=black:s={W}x{H}:r={FPS}", *enc])
        else:
            src = ROOT / "videos" / f"課題素材{source}.mov"
            vf = f"scale={W}:{H},fps={FPS}"
            if crop:
                cw, ch, cx, cy = crop
                vf = f"crop={cw}:{ch}:{cx}:{cy}," + vf
            run(["-ss", f"{source_in:.3f}", "-i", str(src), "-vf", vf, *enc])
        actual = int(cv2.VideoCapture(str(clip)).get(cv2.CAP_PROP_FRAME_COUNT))
        if actual != n:
            raise RuntimeError(f"clip {i} expected {n} frames, got {actual}: source={source}")
        clips.append(clip)
        print(f"  {start:4d}-{end-1:4d}  material={source or 'black'} "
              f"in={source_in:.3f}{' crop' if crop else ''}")
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
         "-filter_complex",
         "[1:v]fps=30,format=rgba[o];[0:v][o]overlay=0:0:eof_action=pass:repeatlast=0[v]",
         "-map", "[v]", "-an", "-frames:v", str(TOTAL_FRAMES), "-c:v", "libx264",
         "-preset", "veryfast", "-crf", "21", "-pix_fmt", "yuv420p", str(out)])
    return out


def add_mosaic(text_video: Path, work: Path) -> Path:
    boxes = mosaic.detect_face_boxes()
    cap = cv2.VideoCapture(str(text_video))
    raw = work / "mosaic.mp4"
    cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
           "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "pipe:0",
           "-an", "-frames:v", str(TOTAL_FRAMES), "-c:v", "libx264",
           "-preset", "veryfast", "-crf", "21", "-pix_fmt", "yuv420p", str(raw)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    count = 0
    while count < TOTAL_FRAMES:
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
    if proc.wait() or count != TOTAL_FRAMES:
        raise RuntimeError(f"mosaic encode failed ({count} frames): {err}")
    return raw


# ---------------------------------------------------------------- audio

def load_audio(path: Path, ss: float | None = None, t: float | None = None) -> np.ndarray:
    cmd = [FFMPEG, "-hide_banner", "-loglevel", "error"]
    if ss is not None:
        cmd += ["-ss", f"{ss:.6f}"]
    cmd += ["-i", str(path)]
    if t is not None:
        cmd += ["-t", f"{t:.6f}"]
    cmd += ["-vn", "-ac", "1", "-ar", str(SR), "-f", "f32le", "pipe:1"]
    p = subprocess.run(cmd, capture_output=True)
    if p.returncode:
        raise RuntimeError(p.stderr.decode())
    return np.frombuffer(p.stdout, dtype=np.float32).copy()


def fit(seg: np.ndarray, n: int) -> np.ndarray:
    if len(seg) < n:
        seg = np.pad(seg, (0, n - len(seg)))
    return seg[:n]


def build_ambient(total: int) -> np.ndarray:
    amb = np.zeros(total, dtype=np.float64)
    fade_n = int(SEG_FADE * SR)
    fade_in = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, fade_n))
    for (start, source, source_in, _), (end, _, _, _) in zip(CUTS, CUTS[1:]):
        n0, n1 = int(round(start / FPS * SR)), int(round(end / FPS * SR))
        n = n1 - n0
        gain = 1.0
        if source is None:
            if start not in BLACK_AMBIENT:
                continue
            source, source_in, gain = BLACK_AMBIENT[start]
        src = ROOT / "videos" / f"課題素材{source}.mov"
        seg = fit(load_audio(src, source_in, n / SR + 0.05), n).astype(np.float64)
        seg[:fade_n] *= fade_in
        seg[-fade_n:] *= fade_in[::-1]
        amb[n0:n1] = seg * gain
    return amb


def speech_gate(stem: np.ndarray, total: int) -> np.ndarray:
    """会話ステムの有音区間から現場音のゲインエンベロープを作る。"""
    hop = int(0.010 * SR)
    win = int(0.025 * SR)
    n_frames = total // hop + 1
    rms = np.full(n_frames, -120.0)
    for i in range(n_frames):
        a = i * hop
        b = min(a + win, len(stem))
        if a >= len(stem):
            break
        chunk = stem[a:b].astype(np.float64)
        if len(chunk):
            rms[i] = 20 * np.log10(np.sqrt(np.mean(chunk ** 2)) + 1e-12)
    active = rms > SPEECH_THRESH_DB
    # 前後0.12秒に拡張し、0.25秒未満の隙間を埋める
    pad = int(0.12 / 0.010)
    idx = np.where(active)[0]
    grown = np.zeros_like(active)
    for i in idx:
        grown[max(0, i - pad):i + pad + 1] = True
    gap = int(0.25 / 0.010)
    on = np.where(grown)[0]
    for a, b in zip(on, on[1:]):
        if 0 < b - a <= gap:
            grown[a:b] = True
    gain_f = np.where(grown, DUCK_GAIN, 1.0)
    # 10msフレーム列 → サンプル列へ展開し、120msで平滑化
    gain = np.repeat(gain_f, hop)[:total]
    if len(gain) < total:
        gain = np.pad(gain, (0, total - len(gain)), constant_values=gain[-1])
    k = int(DUCK_RAMP * SR)
    kernel = np.hanning(k)
    kernel /= kernel.sum()
    return np.convolve(gain, kernel, mode="same")


def build_audio(work: Path) -> Path:
    total = int(round(TOTAL_SEC * SR))
    stem = fit(load_audio(STEM_SRC), total).astype(np.float64)
    amb = build_ambient(total)
    gain = speech_gate(stem, total)
    # 会話レベルは変えず、クリップする場合は現場音側だけを下げる
    f = 1.0
    for _ in range(8):
        mix = stem + amb * gain * f
        peak = np.abs(mix).max()
        if peak <= 0.99:
            break
        f *= 0.99 / peak
    print(f"  stem peak {np.abs(stem).max():.3f}, amb peak {amb.max():.3f}, "
          f"mix peak {peak:.3f}, amb factor {f:.3f}")
    raw = work / "mix.f32"
    raw.write_bytes(mix.astype(np.float32).tobytes())
    aac = work / "mix.m4a"
    run(["-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", str(raw),
         "-c:a", "aac", "-b:a", "160k", str(aac)])
    return aac


def main() -> int:
    if OUT.exists():
        raise SystemExit(f"既存ファイルを上書きしません: {OUT}")
    if not STEM_SRC.exists():
        raise FileNotFoundError(STEM_SRC)
    with tempfile.TemporaryDirectory(prefix="cut_audio_rebuilt_") as tmp:
        work = Path(tmp)
        print("[1/4] corrected cuts (dedup + reference crops)")
        base = render_base(work)
        print("[2/4] existing telops")
        text_video = add_telops(base, work)
        print("[3/4] existing opening mosaic")
        vid = add_mosaic(text_video, work)
        print("[4/4] dialogue stem + rebuilt ambient audio")
        aac = build_audio(work)
        run(["-i", str(vid), "-i", str(aac), "-map", "0:v:0", "-map", "1:a:0",
             "-c:v", "copy", "-c:a", "copy", "-t", f"{TOTAL_SEC:.7f}",
             "-movflags", "+faststart", str(OUT)])
    print(OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
