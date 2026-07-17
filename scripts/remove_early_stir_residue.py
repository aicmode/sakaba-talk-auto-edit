#!/usr/bin/env python3
"""冒頭に先行して聞こえるグラスを混ぜる音(BGMベッドの分離残渣)を除去する。

add_bgm.py の冒頭BGMベッドは見本 29.1 秒以降の音楽ステムを 0〜8.067 秒に
移植しているが、移植元の 35.0 秒以降(= 出力 5.9 秒以降)に、見本の
「グラスを混ぜる音」の demucs 分離残渣が含まれていた
(検証: 出力 6.5〜8.0 秒の 2kHz 以上と見本 +29.100 秒が NCC 0.968)。

本スクリプトは音声の 5.80〜8.03 秒の窓内だけ、500-2000Hz と 2kHz 以上の
2 帯域に「クリーン区間の床レベルへのトランジェントゲート」をかけ、
クリンク音だけを床まで潰す。500Hz 以下(BGM の主成分とルームトーン)は
無変更なので BGM は自然に鳴り続ける。窓外・会話・思案効果音(8.067秒〜)・
本来の混ぜシーン(35.5秒〜)・映像には一切触れない。映像は stream copy。
一時出力の完全デコード検証後に prototype.mp4 を原子的に置換する。
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wavfile
import scipy.signal as sig
from imageio_ffmpeg import get_ffmpeg_exe

ROOT = Path(__file__).resolve().parent.parent
CURRENT = ROOT / "output" / "prototype.mp4"
FFMPEG = get_ffmpeg_exe()

SR = 48000
WIN_START = 5.80          # 残渣の立ち上がり 5.85 秒の直前
WIN_END = 8.03            # 思案効果音 8.067 秒の直前。7.93 以降は床レベル
EDGE_RAMP = 0.15          # 窓両端の cos^2 ランプ
FLOOR_REF = (5.45, 5.80)  # 床レベルの計測区間(残渣・会話なし)
THR_DB = 2.0              # 床 +2dB を超えた分だけゲート
BANDS = [(500, 2000), (2000, None)]  # None = ハイパス
ENV_WIN = 0.02
GAIN_SMOOTH = 0.02


def run(args: list[str]) -> None:
    p = subprocess.run(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", *args],
        capture_output=True,
        text=True,
    )
    if p.returncode:
        raise RuntimeError(p.stderr)


def envelope(x: np.ndarray, sr: int) -> np.ndarray:
    win = int(ENV_WIN * sr)
    return np.sqrt(np.abs(sig.fftconvolve(x**2, np.ones(win) / win, mode="same")))


def main() -> None:
    if not CURRENT.exists():
        raise FileNotFoundError(CURRENT)

    with tempfile.TemporaryDirectory(prefix="stir_residue_") as tmpdir:
        tmp = Path(tmpdir)
        cur_wav = tmp / "cur.wav"
        run(["-i", str(CURRENT), "-ac", "1", "-ar", str(SR), str(cur_wav)])
        _, raw = wavfile.read(cur_wav)
        x = raw.astype(np.float64) / 32768.0

        n = len(x)
        t = np.arange(n) / SR

        # 窓マスク(窓外 0、両端 cos^2 ランプ)
        w = np.zeros(n)
        core = (t >= WIN_START + EDGE_RAMP) & (t < WIN_END - EDGE_RAMP)
        w[core] = 1.0
        up = (t >= WIN_START) & (t < WIN_START + EDGE_RAMP)
        w[up] = np.sin(0.5 * np.pi * (t[up] - WIN_START) / EDGE_RAMP) ** 2
        dn = (t >= WIN_END - EDGE_RAMP) & (t < WIN_END)
        w[dn] = np.cos(0.5 * np.pi * (t[dn] - (WIN_END - EDGE_RAMP)) / EDGE_RAMP) ** 2

        f0, f1 = int(FLOOR_REF[0] * SR), int(FLOOR_REF[1] * SR)
        y = x.copy()
        for lo, hi in BANDS:
            if hi is None:
                sos = sig.butter(4, lo, btype="high", fs=SR, output="sos")
            else:
                sos = sig.butter(4, [lo, hi], btype="band", fs=SR, output="sos")
            band = sig.sosfiltfilt(sos, x)
            env = envelope(band, SR)
            thr = np.median(env[f0:f1]) * 10 ** (THR_DB / 20)
            gain = np.minimum(1.0, thr / np.maximum(env, 1e-12))
            smooth = int(GAIN_SMOOTH * SR)
            gain = sig.fftconvolve(gain, np.ones(smooth) / smooth, mode="same")
            # 窓内だけ、ゲートで削る分を差し引く
            y -= w * (1.0 - gain) * band
            cut = 20 * np.log10(np.maximum(gain[int(6.5 * SR):int(8.0 * SR)].min(), 1e-12))
            print(f"band {lo}-{hi or 'nyq'} Hz: floor thr {20*np.log10(thr):.1f} dBFS, "
                  f"max cut {cut:.1f} dB")

        # 検証1: 窓外は完全一致
        mask_out = (t < WIN_START) | (t >= WIN_END)
        assert np.array_equal(x[mask_out], y[mask_out]), "window leak"

        # 検証2: 残渣(見本 +29.1 秒との高域相関)が消えたこと
        sos_hp = sig.butter(4, 2000, btype="high", fs=SR, output="sos")
        hb_old = sig.sosfiltfilt(sos_hp, x)
        hb_new = sig.sosfiltfilt(sos_hp, y)
        a0, a1 = int(6.5 * SR), int(8.0 * SR)
        drop = 20 * np.log10(
            (np.sqrt(np.mean(hb_new[a0:a1] ** 2)) + 1e-12)
            / (np.sqrt(np.mean(hb_old[a0:a1] ** 2)) + 1e-12))
        print(f"2kHz+ residue RMS change 6.5-8.0s: {drop:.1f} dB")
        if drop > -6:
            raise RuntimeError("residue not sufficiently removed")

        mix_wav = tmp / "mix.wav"
        wavfile.write(mix_wav, SR, (np.clip(y, -1, 1) * 32767).astype(np.int16))

        candidate = tmp / "prototype_new.mp4"
        run(["-i", str(CURRENT), "-i", str(mix_wav),
             "-map", "0:v:0", "-map", "1:a:0",
             "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ar", str(SR),
             "-movflags", "+faststart", str(candidate)])

        # 検証3: 完全デコード + 会話/効果音区間が AAC 誤差の範囲で無改変
        run(["-v", "error", "-i", str(candidate), "-f", "null", "-"])
        chk_wav = tmp / "chk.wav"
        run(["-i", str(candidate), "-ac", "1", "-ar", str(SR), str(chk_wav)])
        _, chk_raw = wavfile.read(chk_wav)
        chk = chk_raw.astype(np.float64) / 32768.0
        for label, s0, s1 in [("dialogue 2.5-5.5s", 2.5, 5.5),
                              ("thinking sfx 8.1-9.9s", 8.1, 9.9),
                              ("stir scene 35.5-38.5s", 35.5, 38.5)]:
            i0, i1 = int(s0 * SR), int(s1 * SR)
            a = x[i0:i1]
            lags = range(-240, 241, 8)
            best = max(lags, key=lambda L: float(np.dot(a, chk[i0 + L:i1 + L])))
            diff = 20 * np.log10(np.sqrt(np.mean((chk[i0 + best:i1 + best] - a) ** 2)) + 1e-12)
            print(f"{label}: diff vs current {diff:.1f} dBFS (AAC誤差のみ想定)")
            if diff > -40:
                raise RuntimeError(f"{label} was altered beyond codec error")

        candidate.replace(CURRENT)

    print(f"generated: {CURRENT}")
    print(f"removed: stirring-sound residue in {WIN_START:.2f}-{WIN_END:.2f}s "
          f"(bands 500-2000Hz / 2kHz+, gated to floor+{THR_DB:.0f}dB)")


if __name__ == "__main__":
    main()
