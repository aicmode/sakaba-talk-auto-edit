#!/usr/bin/env python3
"""動画全体にBGMを通す(前半0〜39.433秒へのBGM追加)。

現行 prototype.mp4 の音声は 39.433秒以降が見本の完成ミックス(BGM込み)で、
前半は再構築音声(BGMなし)。そのため BGM が途中から始まって聞こえる。

本スクリプトは見本 reference.mp4 を demucs (htdemucs) で音源分離し、
音楽ステム(bass+other+drums)から前半用のBGMベッドを作って加算する。

- 9.98〜39.433秒: 見本タイムライン同位相のステムを等倍で使用。
  39.433秒で焼き込み済みBGMへサンプル連続で接続される(曲の位置が同じ)。
- 0〜8.067秒: 見本の思案効果音より前は見本のBGMがほぼ無音のため、
  声・効果音の混入がない 29.1秒以降のきれいな区間を移植し -34dBFS に整音。
- 8.067〜9.98秒(モノクロ商品画像シーン): BGM完全ミュート。思案効果音を優先し、
  「二階堂で」開始直前の9.98秒からフェードで復帰。
- セリフ区間(0.80-8.12 / 10.02-15.04秒)は -3.5dB ダッキング。
- ステム全体に 50ms エンベロープのトランジェント抑制をかけ、
  分離残渣(氷・ボトル等の打撃音)による急な音量変化を防ぐ。

映像・既存音声(会話、効果音、後半ミックス)は一切変更せず、
音声は「現行音声 + BGMベッド」の加算のみ。映像は stream copy。
一時出力の完全デコード検証後に prototype.mp4 を原子的に置換する。
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wavfile
import scipy.signal as sig
from imageio_ffmpeg import get_ffmpeg_exe

ROOT = Path(__file__).resolve().parent.parent
CURRENT = ROOT / "output" / "prototype.mp4"
REFERENCE = ROOT / "reference" / "reference.mp4"
FFMPEG = get_ffmpeg_exe()

SR_OUT = 48000
BOUNDARY = 39.433333          # ここから先は見本ミックス(BGM焼き込み済み)
XFADE_KNEE = 39.875           # 旧クロスフェードの実測: ここまでw=0.5
BED_END = 40.025              # w=1.0 到達点。ベッドはここで0になる
SFX_MUTE_START = 8.066667     # 思案効果音の開始 = モノクロシーン開始
SFX_MUTE_END = 9.980          # 「二階堂で」直前。ここからBGM復帰
OPEN_FADE_IN = 1.2            # 冒頭フェードイン
PRE_FADE_OUT = 0.3            # ミュートへ向かうフェードアウト(7.767-8.067)
RESUME_FADE_IN = 0.6          # 9.98からの復帰フェード
PREPEND_SRC = 29.1            # 移植元(検証済み: 声の混入なし・氷音回避)
PREPEND_LEVEL_DB = -34.0      # 冒頭ベッドの実効RMS
DUCK_DB = -3.5                # セリフ中の追加ダッキング
DUCK_SPANS = [(0.80, 8.12), (10.02, 15.04)]  # reference_transcript.json 実測
DUCK_RAMP = 0.25


def run(args: list[str]) -> None:
    p = subprocess.run(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", *args],
        capture_output=True,
        text=True,
    )
    if p.returncode:
        raise RuntimeError(p.stderr)


def load_wav_mono(path: Path) -> tuple[int, np.ndarray]:
    sr, x = wavfile.read(path)
    if x.dtype == np.int16:
        x = x.astype(np.float64) / 32768.0
    else:
        x = x.astype(np.float64)
    if x.ndim == 2:
        x = x.mean(axis=1)
    return sr, x


def transient_clamp(x: np.ndarray, sr: int) -> np.ndarray:
    """50msエンベロープが局所中央値の+4dBを超える箇所を抑え込む。"""
    win = int(0.05 * sr)
    env = np.sqrt(sig.fftconvolve(x**2, np.ones(win) / win, mode="same"))
    hop = win
    env_ds = env[::hop]
    med = sig.medfilt(env_ds, kernel_size=81)  # 約4秒の中央値
    floor = np.percentile(env_ds[env_ds > 1e-6], 25)
    target = np.maximum(med, floor)
    gain_ds = np.minimum(1.0, (target * 1.585) / np.maximum(env_ds, 1e-9))
    gain = np.interp(np.arange(len(x)), np.arange(len(env_ds)) * hop, gain_ds)
    smooth = int(0.1 * sr)
    gain = sig.fftconvolve(gain, np.ones(smooth) / smooth, mode="same")
    return x * gain


def fade_curve(n: int, sr: int, spans_in: list, spans_out: list) -> np.ndarray:
    """cosランプのフェードイン/アウトを重ねたゲイン曲線。"""
    g = np.ones(n)
    t = np.arange(n) / sr
    for st, en in spans_in:
        m = (t >= st) & (t < en)
        g[m] *= np.sin(0.5 * np.pi * (t[m] - st) / (en - st)) ** 2
        g[t < st] = 0.0
    for st, en in spans_out:
        m = (t >= st) & (t < en)
        g[m] *= np.cos(0.5 * np.pi * (t[m] - st) / (en - st)) ** 2
        g[t >= en] = 0.0
    return g


def duck_curve(n: int, sr: int) -> np.ndarray:
    g = np.ones(n)
    t = np.arange(n) / sr
    duck = 10 ** (DUCK_DB / 20)
    for st, en in DUCK_SPANS:
        core = (t >= st) & (t < en)
        g[core] = np.minimum(g[core], duck)
        att = (t >= st - DUCK_RAMP) & (t < st)
        g[att] = np.minimum(
            g[att], 1 + (duck - 1) * (t[att] - (st - DUCK_RAMP)) / DUCK_RAMP)
        rel = (t >= en) & (t < en + DUCK_RAMP)
        g[rel] = np.minimum(g[rel], duck + (1 - duck) * (t[rel] - en) / DUCK_RAMP)
    return g


def rms_db(x: np.ndarray) -> float:
    return 20 * np.log10(np.sqrt(np.mean(x**2)) + 1e-12)


def main() -> None:
    if not CURRENT.exists():
        raise FileNotFoundError(CURRENT)
    if not REFERENCE.exists():
        raise FileNotFoundError(REFERENCE)

    with tempfile.TemporaryDirectory(prefix="add_bgm_") as tmpdir:
        tmp = Path(tmpdir)

        # 1) 音声抽出
        ref_wav = tmp / "ref.wav"
        cur_wav = tmp / "cur.wav"
        run(["-i", str(REFERENCE), "-ac", "1", "-ar", str(SR_OUT), str(ref_wav)])
        run(["-i", str(CURRENT), "-ac", "1", "-ar", str(SR_OUT), str(cur_wav)])
        _, cur = load_wav_mono(cur_wav)

        # 再適用ガード: 前半モンタージュ(30-35秒)にすでにBGM級の床があれば中断。
        # (BGMなし実測 -50dBFS / BGM追加後 -33dBFS)
        sos_g = sig.butter(4, [100, 2000], btype="band", fs=SR_OUT, output="sos")
        floor = rms_db(sig.sosfiltfilt(sos_g, cur[30 * SR_OUT:35 * SR_OUT]))
        if floor > -42.0:
            raise RuntimeError(
                f"BGM already present? 30-35s band floor {floor:.1f} dBFS > -42. "
                "二重適用を防ぐため中断します。")

        # 2) demucs で見本から音楽ステムを分離(44.1kHzで出力される)
        sep = tmp / "sep"
        p = subprocess.run(
            [sys.executable, "-m", "demucs", "-n", "htdemucs",
             "-o", str(sep), str(ref_wav)],
            capture_output=True, text=True)
        if p.returncode:
            raise RuntimeError(p.stderr)
        stem_dir = sep / "htdemucs" / "ref"
        sr_m, bass = load_wav_mono(stem_dir / "bass.wav")
        _, other = load_wav_mono(stem_dir / "other.wav")
        _, drums = load_wav_mono(stem_dir / "drums.wav")
        music = transient_clamp(bass + other + drums, sr_m)

        # 3) BGMベッドを44.1kHzタイムラインで構築
        n_m = len(music)
        bed = np.zeros(n_m)

        # 3a) 冒頭 0〜8.067秒: きれいな区間を移植して -34dBFS に整音
        pre_len = int(SFX_MUTE_START * sr_m)
        src0 = int(PREPEND_SRC * sr_m)
        pre = music[src0:src0 + pre_len].copy()
        pre *= 10 ** ((PREPEND_LEVEL_DB - rms_db(pre)) / 20)
        pre *= fade_curve(
            pre_len, sr_m,
            spans_in=[(0.0, OPEN_FADE_IN)],
            spans_out=[(SFX_MUTE_START - PRE_FADE_OUT, SFX_MUTE_START)])
        bed[:pre_len] = pre

        # 3b) 9.98〜40.025秒: 同位相ステム等倍。ミュート明けフェードイン。
        #     旧処理は39.433〜40.025秒で見本ミックスへクロスフェードしており
        #     (実測: 39.433-39.875秒はw=0.5、その後w=1.0へ直線上昇)、その間の
        #     焼き込みBGMは w 倍しかない。ベッドを (1-w) 倍で重ねて補償し、
        #     BGM総量を w + (1-w) = 1.0 に保つ。
        i0, i1 = int(SFX_MUTE_END * sr_m), int(BED_END * sr_m)
        seg = music[i0:i1] * fade_curve(
            i1 - i0, sr_m, spans_in=[(0.0, RESUME_FADE_IN)], spans_out=[])
        t_seg = SFX_MUTE_END + np.arange(i1 - i0) / sr_m
        comp = np.interp(
            t_seg,
            [BOUNDARY - 0.005, BOUNDARY + 0.005, XFADE_KNEE, BED_END],
            [1.0, 0.5, 0.5, 0.0])
        bed[i0:i1] = seg * comp

        # 3c) セリフ区間ダッキング
        bed *= duck_curve(n_m, sr_m)

        # 4) 48kHzへリサンプルして現行音声に加算
        bed48 = sig.resample_poly(bed, SR_OUT, sr_m)
        if len(bed48) < len(cur):
            bed48 = np.pad(bed48, (0, len(cur) - len(bed48)))
        bed48 = bed48[:len(cur)]
        bed48[int(BED_END * SR_OUT):] = 0.0
        mix = cur + bed48
        peak = np.abs(mix).max()
        if peak > 0.99:
            bed48 *= max(0.0, (0.99 - np.abs(cur).max()) / (np.abs(bed48).max() + 1e-9))
            mix = cur + bed48
            print(f"note: clip avoidance applied (peak was {peak:.3f})")

        mix_wav = tmp / "mix.wav"
        wavfile.write(mix_wav, SR_OUT, (np.clip(mix, -1, 1) * 32767).astype(np.int16))

        # 5) 映像 stream copy + 新音声で多重化
        candidate = tmp / "prototype_bgm.mp4"
        run(["-i", str(CURRENT), "-i", str(mix_wav),
             "-map", "0:v:0", "-map", "1:a:0",
             "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ar", str(SR_OUT),
             "-movflags", "+faststart", str(candidate)])

        # 6) 完全デコード検証 → 検証で問題なければ原子的に置換
        run(["-v", "error", "-i", str(candidate), "-f", "null", "-"])

        # ミュート区間とセリフの無改変を検証
        chk_wav = tmp / "chk.wav"
        run(["-i", str(candidate), "-ac", "1", "-ar", str(SR_OUT), str(chk_wav)])
        _, chk = load_wav_mono(chk_wav)
        # AACエンコード遅延の可能性に備え、±5msで最良ラグを合わせてから比較
        m0, m1 = int(8.15 * SR_OUT), int(9.90 * SR_OUT)
        a = cur[m0:m1]
        lags = range(-240, 241, 8)
        best = max(lags, key=lambda L: float(np.dot(a, chk[m0 + L:m1 + L])))
        mute_diff = rms_db(chk[m0 + best:m1 + best] - a)
        j0 = int((BOUNDARY - 0.5) * SR_OUT)
        j1 = int(BOUNDARY * SR_OUT)
        j2 = int((BOUNDARY + 0.5) * SR_OUT)
        sos = sig.butter(4, [100, 2000], btype="band", fs=SR_OUT, output="sos")
        f_chk = sig.sosfiltfilt(sos, chk)
        junction = (rms_db(f_chk[j0:j1]), rms_db(f_chk[j1:j2]))
        print(f"mute-zone diff vs current: {mute_diff:.1f} dBFS (AAC誤差のみ想定)")
        print(f"junction 100-2000Hz RMS: {junction[0]:.1f} -> {junction[1]:.1f} dBFS "
              "(比較対象は素材現場音を含むため±3dB程度は正常)")
        if mute_diff > -45:
            raise RuntimeError("mute zone was altered beyond codec error")

        candidate.replace(CURRENT)

    print(f"generated: {CURRENT}")
    print(f"bed: 0-{SFX_MUTE_START:.3f}s transplanted (src {PREPEND_SRC}s, "
          f"{PREPEND_LEVEL_DB} dBFS), mute {SFX_MUTE_START:.3f}-{SFX_MUTE_END:.3f}s, "
          f"aligned stem {SFX_MUTE_END:.3f}-{BOUNDARY:.3f}s (unity)")
    print(f"fades: in {OPEN_FADE_IN}s / out {PRE_FADE_OUT}s / resume {RESUME_FADE_IN}s")
    print(f"duck: {DUCK_DB} dB on {DUCK_SPANS}")


if __name__ == "__main__":
    main()
