#!/usr/bin/env python3
"""提出版と見本を全尺で比較し、機械検証レポートを出力する。"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
from imageio_ffmpeg import get_ffmpeg_exe
from scipy import signal

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output" / "final_submission.mp4"
REF = ROOT / "reference" / "reference.mp4"
REPORT = ROOT / "analysis" / "final_verification.json"
FFMPEG = get_ffmpeg_exe()
FPS, FRAMES, SR = 30, 2450, 48000

# 最終パッチを含む、見本実測の編集境界（30 fps タイムライン）。
CUTS = [
    0, 28, 77, 145, 202, 242, 307, 327, 351, 387, 419, 454, 496,
    543, 578, 622, 645, 724, 812, 862, 920, 1064, 1116, 1183, 1206,
    1240, 1290, 1370, 1459, 1532, 1597, 1688, 1704, 1753, 1806, 1856,
    1884, 1939, 1947, 1955, 1978, 2029, 2069, 2103, 2168, 2206, 2293,
    2400, 2450,
]

# セリフが十分含まれる検査窓。思案SFXなど意図的差分は除外する。
DIALOGUE_WINDOWS = [
    (0.15, 2.55), (3.10, 5.75), (6.00, 7.90), (10.25, 14.85),
    (39.10, 44.10), (45.70, 50.90), (52.80, 56.70), (59.70, 61.80),
    (63.95, 69.70), (71.30, 73.55), (79.80, 81.60),
]


def run(args: list[str]) -> bytes:
    p = subprocess.run(args, capture_output=True)
    if p.returncode:
        raise RuntimeError(p.stderr.decode(errors="replace"))
    return p.stdout


def audio(path: Path) -> np.ndarray:
    raw = run([FFMPEG, "-hide_banner", "-loglevel", "error", "-i", str(path),
               "-vn", "-ac", "1", "-ar", str(SR), "-f", "f32le", "pipe:1"])
    return np.frombuffer(raw, np.float32).astype(np.float64)


def norm_corr(a: np.ndarray, b: np.ndarray, max_lag: int) -> tuple[float, int]:
    a = a - a.mean()
    b = b - b.mean()
    c = signal.correlate(a, b, mode="full", method="fft")
    lags = signal.correlation_lags(len(a), len(b), mode="full")
    keep = np.abs(lags) <= max_lag
    c, lags = c[keep], lags[keep]
    k = int(np.argmax(c))
    score = c[k] / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    return float(score), int(lags[k])


def main() -> int:
    cap_o, cap_r = cv2.VideoCapture(str(OUT)), cv2.VideoCapture(str(REF))
    n_o = int(cap_o.get(cv2.CAP_PROP_FRAME_COUNT))
    n_r = int(cap_r.get(cv2.CAP_PROP_FRAME_COUNT))
    if n_o != FRAMES or n_r not in (4900, 4901):
        raise RuntimeError(f"unexpected frame counts: output={n_o}, reference={n_r}")

    # 見本60fpsの偶数フレームと提出版30fpsを全尺比較。字幕帯を避けた映像領域で
    # NCCを取り、各編集境界の前後が同じタイムライン位置か確認する。
    scores: list[float] = []
    boundary_scores: dict[str, float] = {}
    for f in range(FRAMES):
        ok_o, fo = cap_o.read()
        ok_r, fr = cap_r.read()
        # 60→30 fps: 対応する偶数フレームだけ比較し、次の奇数フレームを捨てる。
        cap_r.grab()
        if not ok_o or not ok_r:
            raise RuntimeError(f"frame decode failed at {f}")
        fr = cv2.resize(fr, (540, 960), interpolation=cv2.INTER_AREA)
        # 下部テロップと商品画像の主領域を避ける。カラー差にも頑健な輝度NCC。
        ao = cv2.resize(cv2.cvtColor(fo[80:380], cv2.COLOR_BGR2GRAY), (96, 64))
        ar = cv2.resize(cv2.cvtColor(fr[80:380], cv2.COLOR_BGR2GRAY), (96, 64))
        av, bv = ao.astype(float).ravel(), ar.astype(float).ravel()
        av -= av.mean(); bv -= bv.mean()
        score = float(np.dot(av, bv) / (np.linalg.norm(av) * np.linalg.norm(bv) + 1e-12))
        scores.append(score)
        if f in CUTS or f + 1 in CUTS:
            boundary_scores[str(f)] = score
    cap_o.release(); cap_r.release()

    ao, ar = audio(OUT), audio(REF)
    low_ranges = []
    low_start = None
    for f, score in enumerate(scores + [1.0]):
        if score < .35 and low_start is None:
            low_start = f
        elif score >= .35 and low_start is not None:
            low_ranges.append({"start_frame": low_start, "end_frame": f,
                               "start_sec": low_start / FPS, "end_sec": f / FPS})
            low_start = None
    dialogue = []
    for start, end in DIALOGUE_WINDOWS:
        a, b = round(start * SR), round(end * SR)
        # BGM差を抑え、会話子音に敏感な1kHzハイパスで同期を見る。
        sos = signal.butter(4, 1000, "highpass", fs=SR, output="sos")
        xo = signal.sosfiltfilt(sos, ao[a:b])
        xr = signal.sosfiltfilt(sos, ar[a:b])
        corr, lag = norm_corr(xo, xr, round(.10 * SR))
        dialogue.append({"start": start, "end": end, "corr": corr,
                         "lag_samples": lag, "lag_ms": lag / SR * 1000})

    # コンテナ全編デコード（書き出し破損、末尾欠落、音声エラーの確認）。
    run([FFMPEG, "-v", "error", "-i", str(OUT), "-f", "null", "-"])
    result = {
        "output": str(OUT), "reference": str(REF),
        "output_frames": n_o, "reference_frames": n_r,
        "timeline_seconds": FRAMES / FPS,
        "visual_ncc": {
            "mean": float(np.mean(scores)), "p01": float(np.percentile(scores, 1)),
            "minimum": float(np.min(scores)), "boundary_minimum": min(boundary_scores.values()),
        },
        "visual_low_correlation_ranges": low_ranges,
        "dialogue_sync": dialogue,
        "dialogue_max_abs_lag_ms": max(abs(x["lag_ms"]) for x in dialogue),
        "full_decode": "pass",
        "intentional_differences": ["sound effect", "telop font/color/design"],
    }
    REPORT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
