#!/usr/bin/env python3
"""「あー、二階堂で」の「あー」だけを音声から除去して prototype.mp4 を更新する。

- 「あー」=「じゃあ二階堂で」(素材11)の「じゃあ」の尻尾。出力 9.97〜10.32 秒。
- 「に」の頭 = 出力 10.327 秒(素材11の 6.325 秒。whisper開始スイープ+ピッチ検定で確定)。
- 置換: 9.96〜10.233 秒(シーン7カットまで)はデジタル無音(直前 9.84〜9.97 秒と同じ扱い)、
  10.233〜10.322 秒は素材11のルームトーン(4.95 秒〜)をレベル合わせして充当。
- ルームトーンはカット位置で10msフェードイン、右端15msの等パワークロスフェードで原音へ接続。
- 「二階堂で」(10.327 秒〜)と映像・テロップ・他の音声は無変更。映像は stream copy。
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np
from imageio_ffmpeg import get_ffmpeg_exe

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "output" / "prototype.mp4"
MATERIAL = ROOT / "videos" / "課題素材11.mov"
NEW = ROOT / "output" / "prototype_new.mp4"
SR = 48000
CHANNELS = 2
FFMPEG = get_ffmpeg_exe()

PATCH_IN = 9.960        # 直前のデジタル無音内(あーの開始 9.97 より手前)
SCENE_CUT = 10.233      # シーン7(客+色付きテロップ)開始 = ルームトーン入り
PATCH_OUT = 10.322      # 「に」の頭 10.327 の直前
FILL_SRC = 4.95         # 素材11ルームトーン採取開始(4.85〜5.60 が無音部)
FADE_IN = 0.010         # ルームトーン立ち上がり
XFADE = 0.015           # 原音への等パワークロスフェード
GAIN_REF_OUT = (10.05, 10.30)   # 出力側「あー」区間(レベル照合用)
GAIN_REF_MAT = (6.052, 6.302)   # 対応する素材区間(出力-4.002秒)


def run(args: list[str], *, capture: bool = False) -> subprocess.CompletedProcess:
    p = subprocess.run(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", *args],
        capture_output=capture,
    )
    if p.returncode:
        raise RuntimeError(p.stderr.decode(errors="replace") if capture else "ffmpeg failed")
    return p


def decode(path: Path, ss: float | None = None, t: float | None = None) -> np.ndarray:
    args: list[str] = []
    if ss is not None:
        args += ["-ss", f"{ss:.6f}"]
    if t is not None:
        args += ["-t", f"{t:.6f}"]
    raw = run([*args, "-i", str(path), "-vn", "-ac", str(CHANNELS), "-ar", str(SR),
               "-f", "f32le", "pipe:1"], capture=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy().reshape(-1, CHANNELS)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt((x ** 2).mean()))


def main() -> int:
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)

    audio = decode(SOURCE)

    # 素材ルームトーンを出力内クリップの実レベルに合わせる
    out_ref = audio[int(GAIN_REF_OUT[0] * SR):int(GAIN_REF_OUT[1] * SR)]
    mat_ref = decode(MATERIAL, GAIN_REF_MAT[0], GAIN_REF_MAT[1] - GAIN_REF_MAT[0])
    gain = rms(out_ref) / (rms(mat_ref) + 1e-12)

    i0 = int(round(PATCH_IN * SR))
    i_cut = int(round(SCENE_CUT * SR))
    i1 = int(round(PATCH_OUT * SR))

    tone_len = i1 - i_cut
    tone = decode(MATERIAL, FILL_SRC, tone_len / SR + 0.05)[:tone_len] * gain

    # 無音区間
    audio[i0:i_cut] = 0.0

    # ルームトーン: カット位置で10msフェードイン
    nf = int(FADE_IN * SR)
    tone[:nf] *= np.linspace(0.0, 1.0, nf, dtype=np.float32)[:, None]

    # 右端15ms: ルームトーン→原音の等パワークロスフェード
    nx = int(XFADE * SR)
    ramp = np.linspace(0.0, 1.0, nx, dtype=np.float32)[:, None]
    orig_tail = audio[i1 - nx:i1].copy()
    tone[-nx:] = tone[-nx:] * np.sqrt(1.0 - ramp) + orig_tail * np.sqrt(ramp)

    audio[i_cut:i1] = tone

    with tempfile.TemporaryDirectory(prefix="remove_aa_") as tmp:
        raw = Path(tmp) / "patched.f32"
        raw.write_bytes(audio.tobytes())
        run([
            "-i", str(SOURCE), "-f", "f32le", "-ar", str(SR), "-ac", str(CHANNELS),
            "-i", str(raw), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k", "-ar", str(SR), "-ac", str(CHANNELS),
            "-t", "81.666667", "-movflags", "+faststart", str(NEW),
        ])

    print(f"generated: {NEW}")
    print(f"removed 「あー」 {PATCH_IN:.3f}-{PATCH_OUT:.3f}s "
          f"(silence to {SCENE_CUT:.3f}s, room tone gain x{gain:.2f}, "
          f"{XFADE*1000:.0f}ms crossfade into 「二階堂で」 at 10.327s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
