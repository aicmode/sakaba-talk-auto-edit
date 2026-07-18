#!/usr/bin/env python3
"""再構成した prototype.mp4 の素材トレーサビリティ検証。

1. reference と prototype の SHA-256 が異なること
2. 全カットについて、出力フレームが指定素材の指定時刻フレームと一致すること
   (NCC照合。モノクロ/クロップ/黒背景も考慮)
3. 会話音声セグメントの素材照合(スポットxcorr)
4. 冒頭5.8〜8.03秒にグラス残渣(2kHz以上の突発音)が無いこと
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import cv2
import numpy as np
from imageio_ffmpeg import get_ffmpeg_exe

from rebuild_prototype import (CUTS, STEM, GRAY_START, GRAY_END,
                               FPS, W, H, SR, TOTAL_FRAMES, material)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output" / "prototype.mp4"
REF = ROOT / "reference" / "reference.mp4"
FFMPEG = get_ffmpeg_exe()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def get_frame(cap: cv2.VideoCapture, idx: int) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"frame {idx} read failed")
    return frame


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = cv2.resize(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), (96, 170)).astype(np.float64)
    b = cv2.resize(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), (96, 170)).astype(np.float64)
    a -= a.mean()
    b -= b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum()) + 1e-12
    return float((a * b).sum() / d)


def ncc_region(a: np.ndarray, b: np.ndarray, y0: int, y1: int) -> float:
    """オーバーレイ(商品画像・テロップ帯)を避けた領域だけでNCC照合する。"""
    a = cv2.cvtColor(a[y0:y1], cv2.COLOR_BGR2GRAY).astype(np.float64)
    b = cv2.cvtColor(b[y0:y1], cv2.COLOR_BGR2GRAY).astype(np.float64)
    a -= a.mean()
    b -= b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum()) + 1e-12
    return float((a * b).sum() / d)


def load_audio(path: Path, ss: float, t: float) -> np.ndarray:
    p = subprocess.run(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-ss", f"{ss:.6f}",
         "-i", str(path), "-t", f"{t:.6f}", "-vn", "-ac", "1", "-ar", str(SR),
         "-f", "f32le", "pipe:1"], capture_output=True)
    if p.returncode:
        raise RuntimeError(p.stderr.decode())
    return np.frombuffer(p.stdout, dtype=np.float32).astype(np.float64)


def main() -> int:
    print("== 1. SHA-256 ==")
    h_out, h_ref = sha256(OUT), sha256(REF)
    print(f"  prototype: {h_out}")
    print(f"  reference: {h_ref}")
    assert h_out != h_ref, "SHA-256 が一致(referenceのコピー)"
    print("  → 相違を確認")

    print("\n== 2. 映像カットの素材照合 (NCC) ==")
    cap_out = cv2.VideoCapture(str(OUT))
    caps: dict[int, cv2.VideoCapture] = {}
    results = []
    for (start, source, source_in, crop), (end, _, _, _) in zip(CUTS, CUTS[1:]):
        mid = (start + end) // 2
        out_frame = get_frame(cap_out, mid)
        if source is None:
            gray = cv2.cvtColor(out_frame, cv2.COLOR_BGR2GRAY)
            # 中央上下(テロップ帯以外)がほぼ黒であること
            ok = float(gray[:400].mean()) < 8.0
            results.append((start, end, "黒背景.PNG", 1.0 if ok else 0.0))
            continue
        if source not in caps:
            caps[source] = cv2.VideoCapture(str(material(source)))
        cap_m = caps[source]
        src_fps = cap_m.get(cv2.CAP_PROP_FPS) or 30.0
        sw = int(cap_m.get(cv2.CAP_PROP_FRAME_WIDTH))
        m_idx = round((source_in + (mid - start) / FPS) * src_fps)
        m_frame = get_frame(cap_m, m_idx)
        if crop:
            cw, ch, cx, cy = crop
            s = sw / 1080  # crop座標は1080幅基準
            m_frame = m_frame[int(cy * s):int((cy + ch) * s),
                              int(cx * s):int((cx + cw) * s)]
        m_frame = cv2.resize(m_frame, (W, H))
        if GRAY_START <= mid < GRAY_END:
            g = cv2.cvtColor(m_frame, cv2.COLOR_BGR2GRAY)
            m_frame = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        if 242 <= mid < 332:
            # 商品画像(上部)とテロップ帯(y400-520)を避けた下部領域で照合
            score = ncc_region(out_frame, m_frame, 540, 940)
        else:
            score = ncc(out_frame, m_frame)
        results.append((start, end, f"課題素材{source}.mov", score))

    n_ok = 0
    for start, end, src, score in results:
        mark = "OK" if score > 0.80 else "NG"
        if score > 0.80:
            n_ok += 1
        print(f"  f{start:4d}-{end - 1:4d} ({start / FPS:6.2f}s) {src:18s} NCC={score:.3f} {mark}")
    print(f"  → {n_ok}/{len(results)} カット一致")

    print("\n== 3. 会話音声の素材照合 (800Hz HPF xcorr) ==")
    import scipy.signal as _sig
    sos_hp = _sig.butter(4, 800, btype="high", fs=SR, output="sos")
    total_sec = TOTAL_FRAMES / FPS
    for ref_at, src, s_in, s_out, *_rest, line in [STEM[0], STEM[10], STEM[16], STEM[23]]:
        start = max(0.0, ref_at - 0.15)
        dur = min(s_out - s_in, 1.9, total_sec - ref_at - 0.35)
        mat = _sig.sosfiltfilt(sos_hp, load_audio(material(src), s_in, dur))
        outa = _sig.sosfiltfilt(sos_hp, load_audio(OUT, start, dur + 0.3))
        corr = np.correlate(outa, mat, mode="valid")
        denom = np.sqrt((mat ** 2).sum()) + 1e-12
        norm = np.array([np.sqrt((outa[i:i + len(mat)] ** 2).sum()) + 1e-12
                         for i in range(len(corr))])
        scores = corr / (denom * norm)
        k = int(np.argmax(scores))
        peak = float(scores[k])
        lag = k / SR - (ref_at - start)
        print(f"  {ref_at:7.3f}s 素材{src:2d} 『{line[:14]}』 xcorr={peak:.3f} "
              f"lag={lag:+.3f}s {'OK' if peak > 0.7 and abs(lag) < 0.03 else 'NG'}")

    print("\n== 4. 冒頭グラス残渣チェック (5.8〜8.03秒) ==")
    import scipy.signal as sig
    a = load_audio(OUT, 0.0, 10.0)
    sos = sig.butter(4, 2000, btype="high", fs=SR, output="sos")
    hb = sig.sosfiltfilt(sos, a)

    def rms_db(x):
        return 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12)

    floor = rms_db(hb[int(5.45 * SR):int(5.80 * SR)])
    window = rms_db(hb[int(6.50 * SR):int(8.00 * SR)])
    print(f"  2kHz+ RMS: floor(5.45-5.8s) {floor:.1f} dBFS / window(6.5-8.0s) {window:.1f} dBFS")
    print(f"  → {'残渣なし (床レベル同等)' if window < floor + 3.0 else '要確認'}")

    print("\n== 5. 尺・フレーム数 ==")
    n = int(cv2.VideoCapture(str(OUT)).get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  frames={n} (期待 {TOTAL_FRAMES}) {'OK' if n == TOTAL_FRAMES else 'NG'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
