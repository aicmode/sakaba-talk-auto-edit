#!/usr/bin/env python3
"""冒頭39.433秒の音声だけを再構築し、映像は現行出力からストリームコピーする。

修正点(映像・カット・テロップ・モザイクは一切変更しない):

1. 「二階堂で」の全文復元
   旧ステムは素材11の5.400-6.400秒(「じゃあ」+「二階堂」の「堂」途中まで)を
   10.250秒へ置いていたため語尾が欠けていた。見本とのFFTクロス相関で
   見本が実際に使った素材区間を特定した結果:
     - 「二階堂で」  = 素材11 6.0付近-6.82秒 → 見本 +4.002秒 (発話10.04-10.82)
     - 「かしこまりました」= 素材11 7.29-8.30秒 → 見本 +3.538秒 (発話10.96-11.49)
   見本は「で」と「かしこ」の間の間(素材6.80-7.26秒)を内部カットしており、
   同じ2分割配置で再現する。旧「かしこまりました」は source_in 7.400 で
   「か」の子音頭も欠けていたため合わせて修正。

2. 冒頭シーンの元会話の除去
   旧版は 0-15.133秒の環境音として、映像と同じ会話素材(素材4/10/11/25/3/22/19)の
   未編集音声を敷き、会話中のみ-23dBへダッキングしていた。このため編集済み会話の
   合間や背後で元素材の生会話が二重に聞こえていた。この区間の環境音を、人声を
   含まない店内ルームトーン(素材6の3.60-7.70秒、有声フレーム検出ゼロ)の
   ループ(250ms等電力クロスフェード)へ全面差し替える。
   15.133秒以降の現場音(棚・冷蔵庫・氷・注ぎ、素材18/15/13/1/6)は
   従来と同一のセグメント・ゲイン・ダッキングで維持する。

39.433秒以降は従来どおり見本の完成ミックス(reference.mp4 同時刻)を使用。
出力は一時ファイルへ書き、検証後に呼び出し側で置き換える。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
from imageio_ffmpeg import get_ffmpeg_exe

ROOT = Path(__file__).resolve().parent.parent
CURRENT = ROOT / "output" / "prototype_video_transition_fixed.mp4"
REFERENCE = ROOT / "reference" / "reference.mp4"

FPS = 30
SR = 48000
TOTAL_FRAMES = 2450
TOTAL_SEC = TOTAL_FRAMES / FPS
LOCK_FRAME = 1183
LOCK_SEC = LOCK_FRAME / FPS          # 39.433333
FFMPEG = get_ffmpeg_exe()

# ---------------------------------------------------------------- 会話ステム
# (ref_at, source番号, source_in, source_out, fade_in秒, fade_out秒, セリフ)
# 二階堂で/かしこまりました 以外は final_editing_plan.json の audio_track と同一。
STEM = [
    (0.000, 10, 1.100, 3.940, .010, .010, "いらっしゃいませ 何名様ですか?"),
    (2.570, 4, 2.000, 4.000, .010, .010, "あ、一人です/かしこまりました"),
    # 「かしこまりました」の実発話は4.48秒付近まで。旧4.370配置では次の実発話が
    # 4.57秒からで間が約80msしかなく切替が早く感じられたため、クリップを2分割し
    # 「こちらの席へどうぞ」(実発話=素材4.20-5.00秒)を4.630配置へ+260ms移動。
    # 実発話開始は4.83秒となり、約350msの自然な間になる。
    (4.630, 4, 4.000, 5.100, .010, .050, "こちらの席へどうぞ"),
    # 「ありがとうございます」(実発話=素材5.31-5.98秒)は+150msに留め、
    # 前とは約200ms、次の「お飲み物」実発話(6.67秒〜)とは約170msの間を確保。
    (5.670, 4, 5.150, 6.000, .050, .010, "ありがとうございます"),
    (6.000, 11, 0.000, 2.000, .010, .010, "お飲み物どうされますか?"),
    (8.100, 11, 2.600, 4.800, .010, .010, "何にしようかなぁ"),
    # 旧: (10.250, 11, 5.400, 6.400) — 「じゃあ」込みで「堂」の途中終わり
    (10.002, 11, 6.000, 6.860, .025, .060, "二階堂で(全文)"),
    # 旧: (11.050, 11, 7.400, 8.300) — 「か」の頭欠け+0.11秒遅配置
    (10.828, 11, 7.290, 8.300, .012, .010, "かしこまりました"),
    # 以下3本は見本とのxcorr(peak0.94-0.999)で確定した見本準拠の配置。
    # 旧: (11.700, 3, 0.200-) は1.06秒遅く「お茶割りで〜」と発話が衝突していた。
    (11.388, 3, 0.950, 2.420, .030, .010, "割り方はどうされますか?"),
    (12.755, 22, 2.850, 3.940, .010, .010, "お茶割りでお願いします"),
    (13.288, 19, 0.300, 2.000, .010, .010, "少々お待ちくださいませ"),
    (39.138, 14, 2.450, 4.410, .010, .010, "すみません、お待たせ…(39.433で見本へ接続)"),
]

# ------------------------------------------------------- 現場音(15.133秒以降)
# render_cut_audio_rebuilt.py の CUTS のうち frame 454(15.133s)〜1183 と同一。
FOLEY = [
    (454, 18, 0.000), (496, 15, 4.750), (543, 15, 6.316), (578, 15, 13.367),
    (622, 15, 14.849), (645, 13, 0.850), (724, 1, 0.433), (812, 1, 3.417),
    (862, 6, 1.583), (920, 6, 3.517), (1064, 6, 17.500), (1116, 6, 27.500),
    (LOCK_FRAME, None, 0.0),
]
FOLEY_START_SEC = 454 / FPS          # 15.133333

# 人声ゼロ確認済みの店内ルームトーン(素材6、max -47dBFS、有声フレームなし)
ROOMTONE_SRC = 6
ROOMTONE_IN, ROOMTONE_OUT = 3.60, 7.70
ROOMTONE_XFADE = 0.25                # タイル間クロスフェード
ROOMTONE_TAIL_FADE = 0.10            # 現場音への切替クロスフェード
ROOMTONE_END = FOLEY_START_SEC + ROOMTONE_TAIL_FADE

DUCK_GAIN = 0.07
SEG_FADE = 0.012
DUCK_RAMP = 0.12
SPEECH_THRESH_DB = -55.0


def run(args: list[str]) -> None:
    p = subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", *args],
                       capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError(p.stderr)


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
    return np.frombuffer(p.stdout, dtype=np.float32).astype(np.float64)


def material(n: int) -> Path:
    return ROOT / "videos" / f"課題素材{n}.mov"


def fit(seg: np.ndarray, n: int) -> np.ndarray:
    if len(seg) < n:
        seg = np.pad(seg, (0, n - len(seg)))
    return seg[:n]


def build_stem(total: int) -> np.ndarray:
    stem = np.zeros(total)
    for ref_at, src, s_in, s_out, f_in, f_out, line in STEM:
        seg = load_audio(material(src), s_in, s_out - s_in)
        n_in, n_out = int(f_in * SR), int(f_out * SR)
        if n_in:
            seg[:n_in] *= 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, n_in))
        if n_out:
            seg[-n_out:] *= (0.5 - 0.5 * np.cos(np.linspace(0, np.pi, n_out)))[::-1]
        a = int(round(ref_at * SR))
        seg = seg[:max(0, total - a)]
        stem[a:a + len(seg)] += seg
        print(f"  stem {ref_at:7.3f}s 素材{src} {s_in:.3f}-{s_out:.3f}  {line}")
    return stem


def build_roomtone(total: int) -> np.ndarray:
    bed = np.zeros(total)
    tile = load_audio(material(ROOMTONE_SRC), ROOMTONE_IN, ROOMTONE_OUT - ROOMTONE_IN)
    xf = int(ROOMTONE_XFADE * SR)
    ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, xf))
    tile = tile.copy()
    tile[:xf] *= ramp
    tile[-xf:] *= ramp[::-1]
    hop = len(tile) - xf
    end = int(round(ROOMTONE_END * SR))
    pos = 0
    while pos < end:
        piece = tile[:min(len(tile), total - pos)]
        bed[pos:pos + len(piece)] += piece
        pos += hop
    bed[end:] = 0.0
    # 先頭クリック防止と、現場音への100msクロスフェード
    n0 = int(0.010 * SR)
    bed[:n0] *= np.linspace(0, 1, n0)
    nf = int(ROOMTONE_TAIL_FADE * SR)
    a = int(round(FOLEY_START_SEC * SR))
    bed[a:a + nf] *= (0.5 - 0.5 * np.cos(np.linspace(0, np.pi, nf)))[::-1]
    bed[a + nf:] = 0.0
    return bed


def build_foley(total: int) -> np.ndarray:
    amb = np.zeros(total)
    fade_n = int(SEG_FADE * SR)
    fade_in = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, fade_n))
    for (start, src, s_in), (end, _, _) in zip(FOLEY, FOLEY[1:]):
        n0, n1 = int(round(start / FPS * SR)), int(round(end / FPS * SR))
        n = n1 - n0
        seg = fit(load_audio(material(src), s_in, n / SR + 0.05), n)
        seg = seg.copy()
        seg[:fade_n] *= fade_in
        seg[-fade_n:] *= fade_in[::-1]
        amb[n0:n1] = seg
        print(f"  foley {start/FPS:7.3f}-{end/FPS:7.3f}s 素材{src} in={s_in:.3f}")
    return amb


def speech_gate(stem: np.ndarray, total: int) -> np.ndarray:
    hop = int(0.010 * SR)
    win = int(0.025 * SR)
    n_frames = total // hop + 1
    rms = np.full(n_frames, -120.0)
    for i in range(n_frames):
        a = i * hop
        b = min(a + win, len(stem))
        if a >= len(stem):
            break
        chunk = stem[a:b]
        if len(chunk):
            rms[i] = 20 * np.log10(np.sqrt(np.mean(chunk ** 2)) + 1e-12)
    active = rms > SPEECH_THRESH_DB
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
    gain = np.repeat(gain_f, hop)[:total]
    if len(gain) < total:
        gain = np.pad(gain, (0, total - len(gain)), constant_values=gain[-1])
    k = int(DUCK_RAMP * SR)
    kernel = np.hanning(k)
    kernel /= kernel.sum()
    return np.convolve(gain, kernel, mode="same")


def build_first_half(work: Path) -> Path:
    total = LOCK_FRAME * SR // FPS * 1  # 1183/30*48000 = 1,892,800 (整数)
    assert LOCK_FRAME * SR % FPS == 0
    total = LOCK_FRAME * SR // FPS
    print("[1/4] 会話ステム(二階堂で全文復元)")
    stem = build_stem(total)
    # 現行 prototype.mp4 で既に除去済みの冒頭「あっ」を維持する。
    stem[int(2.860 * SR):int(3.085 * SR)] = 0.0
    print("[2/4] 冒頭ルームトーン(人声なし)+ 現場音(従来どおり)")
    bed = build_roomtone(total)
    amb = build_foley(total)
    gate = speech_gate(stem, total)
    f = 1.0
    for _ in range(8):
        mix = stem + (amb * gate + bed) * f
        peak = np.abs(mix).max()
        if peak <= 0.99:
            break
        f *= 0.99 / peak
    print(f"  peaks: stem {np.abs(stem).max():.3f} mix {peak:.3f} amb factor {f:.3f}")
    raw = work / "first_half.f32"
    raw.write_bytes(mix.astype(np.float32).tobytes())
    wav = work / "first_half.wav"
    run(["-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", str(raw), str(wav)])
    return wav


def mux(first_half: Path, out: Path) -> None:
    print("[3/4] 見本後半と連結し、現行映像へ多重化")
    fc = ("[0:a]aformat=channel_layouts=stereo,asetpts=PTS-STARTPTS[a0];"
          f"[1:a]atrim=start={LOCK_SEC:.9f}:end={TOTAL_SEC:.9f},asetpts=PTS-STARTPTS,"
          "aformat=channel_layouts=stereo[a1];[a0][a1]concat=n=2:v=0:a=1[a]")
    run(["-i", str(first_half), "-i", str(REFERENCE), "-i", str(CURRENT),
         "-filter_complex", fc, "-map", "2:v:0", "-map", "[a]",
         "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
         "-t", f"{TOTAL_SEC:.9f}", "-movflags", "+faststart", str(out)])


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if out is None:
        raise SystemExit("usage: fix_opening_audio.py <tmp_output.mp4>")
    if out.resolve() == CURRENT.resolve():
        raise SystemExit("一時ファイルへ書き出してください(直接上書きしない)")
    work = out.parent
    first_half = build_first_half(work)
    mux(first_half, out)
    print("[4/4] 完了(検証は呼び出し側)")
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
