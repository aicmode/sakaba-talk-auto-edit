#!/usr/bin/env python3
"""音声最終調整+冒頭モザイク版の生成(フェーズ3・確認版)。

output/prototype_text.mp4 の映像(テロップ込み・カット構成は不変)に対して
1. 会話音声を見本動画準拠で再配置した新しい音声トラック
2. カット1(0〜0.917秒)の入店客への顔追従モザイク
を適用し、output/prototype_audio_mosaic.mp4 を生成する。
prototype.mp4 / prototype_text.mp4 は上書きしない。

音声配置の根拠: 見本動画は同じ録音素材から編集されているため、高域通過
波形のFFTクロス相関で「見本内に同一録音が現れる位置」をサンプル精度で
特定した(sharp=ピーク/サイドローブ比。>3で同一録音と判定)。相関が
立たない箇所は mlx-whisper の単語タイムスタンプで裏取りした。主な調整:
- 冒頭「いらっしゃいませ」は見本では0.8秒から(旧配置は1.28秒相当)
- 多くのセリフは見本ではテロップより0.1〜0.6秒先行して始まる(Jカット)
- 45.2秒の客「ありがとうございます」(素材8)は見本の音声に存在しない→削除
- 素材14「お待たせ〜二階堂のお茶割り」と素材20「今年24…25」は見本内で
  内部カット(言い直し等の除去)がある→2分割配置で再現
- 「じゃあ二階堂で」の「じゃあ」は見本に無い→除去

モザイクは見本と同様、入店カット(カット1)のみ。素材4のネイティブ解像度
フレームを YuNet(FaceDetectorYN)で毎フレーム検出し、移動平均で平滑化
した矩形をピクセル化する。検出漏れフレームは前後から線形補間する。
"""

import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import cv2
import numpy as np
from imageio_ffmpeg import get_ffmpeg_exe

ROOT = Path(__file__).resolve().parent.parent
SRC_VIDEO = ROOT / "output" / "prototype_text.mp4"
OUT = ROOT / "output" / "prototype_audio_mosaic.mp4"
YUNET = ROOT / "assets" / "models" / "face_detection_yunet_2023mar.onnx"

W, H, FPS = 540, 960, 30
DURATION = 81.68
SR = 48000
FFMPEG = get_ffmpeg_exe()

# ---------------------------------------------------------------------------
# 音声トラック(見本準拠の最終調整版)
# 各行: (ref_at, source, source_in, source_out, 根拠, セリフ)
AUDIO_TRACK = [
    (0.000, "videos/課題素材10.mov", 1.545, 3.94,
     "部分xcorr sharp19.4(頭0.445sトリム)", "いらっしゃいませ 何名様ですか?"),
    (2.021, "videos/課題素材4.mov", 2.00, 4.00, "xcorr sharp24.2", "あ、一人です/かしこまりました"),
    (4.021, "videos/課題素材4.mov", 4.00, 6.00, "xcorr sharp22.5", "こちらの席へどうぞ/ありがとうございます"),
    (6.121, "videos/課題素材11.mov", 0.00, 2.00, "xcorr sharp7.6", "お飲み物どうされますか?"),
    (8.100, "videos/課題素材11.mov", 2.60, 4.50,
     "見本で相関・音声認識とも未検出のため据え置き(尾を短縮)", "何にしようかなぁ"),
    (10.010, "videos/課題素材11.mov", 5.92, 6.40,
     "見本語り出し10.04(whisper)に合わせ「じゃあ」を除去", "二階堂で"),
    (10.938, "videos/課題素材11.mov", 7.40, 8.30, "xcorr sharp10.0", "かしこまりました"),
    (10.638, "videos/課題素材3.mov", 0.20, 2.42, "xcorr sharp7.7(リードイン込みJカット)", "割り方はどうされますか?"),
    (12.755, "videos/課題素材22.mov", 2.85, 3.94, "xcorr sharp13.5", "お茶割りでお願いします"),
    (13.288, "videos/課題素材19.mov", 0.30, 2.00, "xcorr sharp5.6", "少々お待ちくださいませ"),
    (39.138, "videos/課題素材14.mov", 2.45, 4.41,
     "部分xcorr sharp8.6(前半)", "すみません、お待たせいたしました"),
    (41.091, "videos/課題素材14.mov", 4.95, 6.78,
     "部分xcorr(後半)語尾が見本42.90と一致。見本は中間0.55sを内部カット", "二階堂のお茶割りになります"),
    # 45.2s 客「ありがとうございます」(素材8 10.3-12.0)は見本音声に存在しないため削除
    (45.755, "videos/課題素材17.mov", 0.00, 2.56, "xcorr sharp9.9", "店員さんはいくつなんですか?"),
    (48.078, "videos/課題素材20.mov", 4.34, 5.09,
     "部分xcorr sharp6.4(前半)", "(えっと)今年"),
    (48.805, "videos/課題素材20.mov", 6.40, 8.48,
     "部分xcorr sharp4.9(後半)。見本は24…の言い直し1.33sを内部カット", "25の年になりますね"),
    (49.638, "videos/課題素材2.mov", 2.00, 5.72, "xcorr sharp18.1", "お若いんですね/よかったら店員さんも一杯どうぞ"),
    (52.835, "videos/課題素材21.mov", 2.68, 5.76, "xcorr sharp14.4", "ありがとうございます/私もお客様と同じものを…"),
    (55.805, "videos/課題素材2.mov", 8.40, 9.10, "xcorr+RMSオンセット一致", "はい"),
    (59.771, "videos/課題素材16.mov", 8.40, 10.30, "xcorr sharp11.5", "ありがとうございます/いただきます"),
    (61.780, "videos/課題素材7.mov", 0.60, 1.50, "xcorr sharp3.2", "(グラスの音)"),
    (63.471, "videos/課題素材9.mov", 0.00, 5.30,
     "広域xcorr sharp10.8(尾0.18s短縮で店員の重複「ありがとう」を回避)", "こんな時間か/ごちそうさまです楽しかったです"),
    (67.755, "videos/課題素材12.mov", 0.00, 1.60,
     "xcorr sharp5.6(見本では客のセリフ末尾と実際に重なる)", "ありがとうございます"),
    (71.371, "videos/課題素材12.mov", 1.60, 3.38, "xcorr sharp18.8", "またお越しくださいませ"),
    (79.855, "videos/課題素材24.mov", 12.00, 14.76, "部分xcorr sharp10.4", "ありがとうございました"),
]

# ---------------------------------------------------------------------------
# モザイク(カット1: 出力0〜0.917秒 = フレーム0〜27、素材4の0.3秒〜)
MOSAIC_SRC = ROOT / "videos" / "課題素材4.mov"
MOSAIC_SRC_IN = 0.3
MOSAIC_FRAMES = round(0.917 * FPS)  # 28
MOSAIC_BLOCK = 9  # 540px幅でのモザイク1ブロックの画素数(見本の粒度に合わせた)


def run(args: list[str]) -> None:
    proc = subprocess.run(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", *args],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {' '.join(map(str, args))}\n{proc.stderr}")


def detect_face_boxes() -> list[tuple[int, int, int, int]]:
    """素材4から顔矩形を毎フレーム検出し、平滑化して出力解像度の矩形を返す。"""
    cap = cv2.VideoCapture(str(MOSAIC_SRC))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    sw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    sh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    det = cv2.FaceDetectorYN_create(str(YUNET), "", (sw, sh), score_threshold=0.5)
    raw: list[tuple[float, float, float, float] | None] = []
    for i in range(MOSAIC_FRAMES):
        cap.set(cv2.CAP_PROP_POS_FRAMES, round((MOSAIC_SRC_IN + i / FPS) * src_fps))
        ok, frame = cap.read()
        if not ok:
            raw.append(None)
            continue
        _, faces = det.detect(frame)
        raw.append(tuple(faces[0][:4]) if faces is not None and len(faces) else None)
    cap.release()

    n_missing = sum(b is None for b in raw)
    if all(b is None for b in raw):
        raise RuntimeError("顔が1フレームも検出できませんでした")
    # 検出漏れは前後の検出値から線形補間(端は最寄り値で保持)
    idx = [i for i, b in enumerate(raw) if b is not None]
    boxes = np.array([raw[i] if raw[i] is not None else (0, 0, 0, 0) for i in range(len(raw))],
                     dtype=np.float64)
    for c in range(4):
        boxes[:, c] = np.interp(np.arange(len(raw)), idx, [raw[i][c] for i in idx])
    # 5フレーム移動平均で平滑化(ジッタ防止)
    kernel = np.ones(5) / 5
    pad = np.pad(boxes, ((2, 2), (0, 0)), mode="edge")
    smooth = np.stack([np.convolve(pad[:, c], kernel, "valid") for c in range(4)], axis=1)
    # 顔矩形→頭部全体をカバーする矩形へ拡張し、出力解像度へスケール
    scale = W / sw
    out = []
    for x, y, w, h in smooth:
        x0 = (x - 0.25 * w) * scale
        y0 = (y - 0.28 * h) * scale
        x1 = (x + w * 1.25) * scale
        y1 = (y + h * 1.12) * scale
        out.append((max(0, int(x0)), max(0, int(y0)),
                    min(W, int(x1)), min(H, int(y1))))
    if n_missing:
        print(f"  検出漏れ {n_missing}/{MOSAIC_FRAMES} フレームを補間")
    return out


def pixelate(frame: np.ndarray, box: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = box
    roi = frame[y0:y1, x0:x1]
    if roi.size == 0:
        return
    h, w = roi.shape[:2]
    small = cv2.resize(roi, (max(1, w // MOSAIC_BLOCK), max(1, h // MOSAIC_BLOCK)),
                       interpolation=cv2.INTER_AREA)
    frame[y0:y1, x0:x1] = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def build_audio(work: Path, warnings: list[str]) -> Path:
    """調整済みセグメントを 48kHz モノラルで絶対時刻に加算ミックスする。"""
    buf = np.zeros(int(round(DURATION * SR)), dtype=np.float64)
    fade = int(SR * 0.01)  # クリックノイズ防止の10msフェード
    for i, (ref_at, source, s_in, s_out, _basis, line) in enumerate(AUDIO_TRACK, 1):
        dur = s_out - s_in
        wav_path = work / f"seg_{i:02d}.wav"
        run(["-ss", f"{s_in:.3f}", "-i", str(ROOT / source),
             "-t", f"{dur:.3f}", "-vn", "-ac", "1", "-ar", str(SR),
             "-c:a", "pcm_s16le", str(wav_path)])
        with wave.open(str(wav_path)) as w_:
            data = np.frombuffer(w_.readframes(w_.getnframes()), dtype=np.int16)
        samples = data.astype(np.float64) / 32768.0
        if len(samples) > 2 * fade:
            samples[:fade] *= np.linspace(0, 1, fade)
            samples[-fade:] *= np.linspace(1, 0, fade)
        start = int(round(ref_at * SR))
        end = start + len(samples)
        if end > len(buf):
            warnings.append(f"音声{i}({line[:16]}…): 終端が動画尺を"
                            f" {(end - len(buf)) / SR:.2f}s 超過→末尾トリム")
            samples = samples[: len(buf) - start]
            end = len(buf)
        buf[start:end] += samples
    peak = np.max(np.abs(buf))
    if peak > 0.99:
        warnings.append(f"音声ピーク {peak:.2f} → 0.95 に正規化")
        buf *= 0.95 / peak
    master = work / "audio.wav"
    with wave.open(str(master), "wb") as w_:
        w_.setnchannels(1)
        w_.setsampwidth(2)
        w_.setframerate(SR)
        w_.writeframes((buf * 32767).astype(np.int16).tobytes())
    return master


def render(boxes: list[tuple[int, int, int, int]], audio: Path) -> None:
    """prototype_text.mp4 を読み、冒頭フレームへモザイクを適用して再エンコード。"""
    cap = cv2.VideoCapture(str(SRC_VIDEO))
    args = [FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
            "-r", str(FPS), "-i", "pipe:0", "-i", str(audio),
            "-map", "0:v", "-map", "1:a",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "21",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k",
            "-shortest", str(OUT)]
    proc = subprocess.Popen(args, stdin=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if n < len(boxes):
            pixelate(frame, boxes[n])
        proc.stdin.write(frame.tobytes())
        n += 1
    cap.release()
    proc.stdin.close()
    stderr = proc.stderr.read().decode()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed:\n{stderr}")
    print(f"  {n} フレーム処理(モザイク適用: 先頭 {min(n, len(boxes))} フレーム)")


def main() -> None:
    if not SRC_VIDEO.exists():
        raise FileNotFoundError(SRC_VIDEO)
    warnings: list[str] = []
    print("[1/3] カット1の顔を検出・平滑化中…")
    boxes = detect_face_boxes()
    with tempfile.TemporaryDirectory(prefix="audio_mosaic_") as tmp:
        work = Path(tmp)
        print("[2/3] 調整済み音声トラックを構築中…")
        audio = build_audio(work, warnings)
        print("[3/3] モザイク適用+映像音声を多重化中…")
        render(boxes, audio)
    size_mb = OUT.stat().st_size / 1e6
    print(f"\n完了: {OUT.relative_to(ROOT)} ({size_mb:.1f} MB)")
    if warnings:
        print("\n警告:")
        for msg in warnings:
            print(f"  - {msg}")


if __name__ == "__main__":
    sys.exit(main())
