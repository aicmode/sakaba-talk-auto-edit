#!/usr/bin/env python3
"""素材のみから output/prototype.mp4 を完全再構成する。

reference/reference.mp4 は本スクリプトの入力に一切使用しない(比較・解析で
確定済みの数値パラメータのみ使用)。使用する素材は:
- videos/課題素材1〜25.mov (映像カット・会話音声・現場音・ルームトーン)
- images/課題素材(写真1〜3)  (商品画像ポップイン演出)
- assets/黒背景.PNG           (「お会計中」黒背景カット)

これまでの修正内容を素材ベースで統合して反映する:
- カット構成: render_cut_audio_rebuilt.py の照合済みカット(0〜1182フレーム、
  素材18/15/1のズームクロップ含む)+ render_dialogue_cut_fixed.py の
  後半実カット(1183〜2449フレーム)+ fix_opening_dialogue_transition.py の
  冒頭カット境界修正(フレーム145で素材4を4.867秒へ)
- テロップ: render_prototype_text.py のデザインそのまま+冒頭3枚の時刻修正
- 冒頭モザイク: render_prototype_audio_mosaic.py の顔追従モザイク
- 商品画像演出: 8.067〜11.067秒をモノクロ化し、images/ の3枚を
  ポップインで合成(従来は reference の完成画を流用していた箇所を素材から再構成)
- 会話音声: fix_opening_audio.py の修正済みステム(「あっ」「あー」除去、
  「二階堂で」全文復元)+ 後半は render_prototype_audio_mosaic.py の
  xcorr照合済み配置(従来 reference ミックスだった39.433秒以降を素材から再構成)
- 現場音: ルームトーン(素材6)+ 素材現場音、会話中は-23dBダッキング、
  冷蔵庫-5dB/氷-8dB/混ぜ-6dBのレベル調整
- 思案効果音・BGM: 素材に存在せず reference からも流用できないため、
  numpyで独自合成(マリンバ風アルペジオ / ローファイ・パッドのBGMベッド)。
  BGM構成(冒頭フェードイン、8.067〜9.98秒ミュート、会話ダッキング、全編通し)は
  add_bgm.py の設計を踏襲する。合成のため 5.8〜8.03 秒のグラス残渣も原理的に生じない。
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from imageio_ffmpeg import get_ffmpeg_exe

import render_prototype_text as telops
import render_prototype_audio_mosaic as mosaic

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output" / "prototype.mp4"
BLACK_PNG = ROOT / "assets" / "黒背景.PNG"

FPS, W, H = 30, 540, 960
SR = 48000
TOTAL_FRAMES = 2450
TOTAL_SEC = TOTAL_FRAMES / FPS          # 81.666667
TOTAL_SAMPLES = TOTAL_FRAMES * SR // FPS  # 3,920,000 (整数)
FFMPEG = get_ffmpeg_exe()

# ---------------------------------------------------------------- 映像カット
# 見本のズームクロップ(素材ネイティブ1080x1920上の crop w:h:x:y、照合済み)
CROP_SHELF = (586, 1042, 248, 184)
CROP_BOTTLE = (568, 1010, 256, 906)
CROP_TEA = (474, 842, 586, 836)
CROP_ICE = (614, 1090, 0, 374)

GRAY_START, GRAY_END = 242, 307      # モノクロ区間(思案シーン)
EFFECT_START, EFFECT_END = 242, 332  # 商品画像演出区間(排他的)

# (start_frame, source番号 (None=黒背景), source_in秒, crop)
CUTS = [
    (0, 4, .300, None), (28, 10, 1.100, None), (77, 4, 2.550, None),
    (145, 4, 4.866667, None),            # 冒頭カット境界修正(旧122/4.100)
    (202, 11, .400, None),
    (242, 25, 2.800, None),              # モノクロ+商品画像(素材から再構成)
    (307, 25, 6.300, None), (327, 11, 7.000, None), (351, 3, .200, None),
    (387, 22, 2.900, None), (419, 19, .533, None),
    (454, 18, .000, CROP_SHELF), (496, 15, 4.750, None),
    (543, 15, 6.316, CROP_BOTTLE), (578, 15, 13.367, None),
    (622, 15, 14.849, CROP_TEA), (645, 13, .850, None),
    (724, 1, .433, None), (812, 1, 3.417, CROP_ICE),
    (862, 6, 1.583, None), (920, 6, 3.517, None),
    (1064, 6, 17.500, None), (1116, 6, 27.500, None),
    # --- 後半: 見本実カット(全て素材から) ---
    (1183, 6, 29.733333, None), (1216, 14, 2.600, None),
    (1251, 8, 7.900, None), (1286, 8, 13.300, None),
    (1408, 17, 0.300, None), (1448, 20, 4.200, None),
    (1532, 2, 2.200, None), (1597, 21, 2.680, None),
    (1688, 2, 8.400, None), (1713, 16, 9.800, None),
    (1753, 5, 5.400, None), (1806, 16, 11.600, None),
    (1857, 7, 0.600, None), (1884, 16, 13.300, None),
    (1935, 9, 0.400, None), (2079, 12, 0.200, None),
    (2103, None, 0.0, None), (2168, 12, 1.700, None),
    (2206, 23, 4.300, None), (2293, 24, 8.600, None),
    (2400, 24, 12.600, None), (2450, None, 0.0, None),
]

# ---------------------------------------------------------------- 会話ステム
# (ref_at, source番号, source_in, source_out, fade_in, fade_out, セリフ)
STEM = [
    (0.000, 10, 1.100, 3.940, .010, .010, "いらっしゃいませ 何名様ですか?"),
    (2.570, 4, 2.000, 4.000, .010, .010, "あ、一人です/かしこまりました"),
    (4.630, 4, 4.000, 5.100, .010, .050, "こちらの席へどうぞ"),
    (5.670, 4, 5.150, 6.000, .050, .010, "ありがとうございます"),
    (6.000, 11, 0.000, 2.000, .010, .010, "お飲み物どうされますか?"),
    # 「何にしようかなぁ」は思案効果音に置換済みのため使用しない
    (10.327, 11, 6.325, 6.860, .010, .060, "二階堂で(「あー」除去済み)"),
    (10.828, 11, 7.290, 8.300, .012, .010, "かしこまりました"),
    (11.388, 3, 0.950, 2.420, .030, .010, "割り方はどうされますか?"),
    (12.755, 22, 2.850, 3.940, .010, .010, "お茶割りでお願いします"),
    (13.288, 19, 0.300, 2.000, .010, .010, "少々お待ちくださいませ"),
    (39.138, 14, 2.450, 4.410, .010, .010, "すみません、お待たせいたしました"),
    (41.091, 14, 4.950, 6.780, .010, .010, "二階堂のお茶割りになります"),
    (45.755, 17, 0.000, 2.560, .010, .010, "店員さんはいくつなんですか?"),
    (48.078, 20, 4.340, 5.090, .010, .010, "(えっと)今年"),
    (48.805, 20, 6.400, 8.480, .010, .010, "25の年になりますね"),
    (49.638, 2, 2.000, 5.720, .010, .010, "お若いんですね/よかったら一杯どうぞ"),
    (52.835, 21, 2.680, 5.760, .010, .010, "ありがとうございます/私もお客様と同じものを…"),
    (55.805, 2, 8.400, 9.100, .010, .010, "はい"),
    (59.771, 16, 8.400, 10.300, .010, .010, "ありがとうございます/いただきます"),
    (61.780, 7, 0.600, 1.500, .010, .010, "(グラスの音)"),
    (63.471, 9, 0.000, 5.300, .010, .010, "こんな時間か/ごちそうさまです楽しかったです"),
    (67.755, 12, 0.000, 1.600, .010, .010, "ありがとうございます"),
    (71.371, 12, 1.600, 3.380, .010, .010, "またお越しくださいませ"),
    (79.855, 24, 12.000, 14.760, .010, .010, "ありがとうございました"),
]
ATTO_ZERO = (2.860, 3.085)  # 冒頭「あっ」除去(ステム上で無音化)

# ------------------------------------------------------- ルームトーン・現場音
ROOMTONE_SRC, ROOMTONE_IN, ROOMTONE_OUT = 6, 3.60, 7.70
ROOMTONE_XFADE = 0.25
ROOMTONE_TAIL_FADE = 0.10
FOLEY_START_SEC = 454 / FPS              # 15.133333
ROOMTONE_END = FOLEY_START_SEC + ROOMTONE_TAIL_FADE

# 現場音セグメント(15.133秒〜)。後半の会話重複リスクがあるカットは
# ルームトーン("room")へ差し替え、(1688)は「はい」重複回避で採取位置をずらす。
FOLEY = [
    (454, 18, 0.000), (496, 15, 4.750), (543, 15, 6.316), (578, 15, 13.367),
    (622, 15, 14.849), (645, 13, 0.850), (724, 1, 0.433), (812, 1, 3.417),
    (862, 6, 1.583), (920, 6, 3.517), (1064, 6, 17.500), (1116, 6, 27.500),
    (1183, 6, 29.733), (1216, 14, 2.600), (1251, 8, 7.900), (1286, 8, 13.300),
    (1408, 17, 0.300), (1448, 20, 4.200), (1532, 2, 2.200), (1597, 21, 2.680),
    (1688, 2, 8.867),                    # 素材2の「はい」再生回避
    (1713, 16, 9.800), (1753, 5, 5.400), (1806, 16, 11.600),
    (1857, 7, 0.600), (1884, 16, 13.300), (1935, 9, 0.400),
    (2079, "room", 0.0),                 # 「ありがとう」重複回避
    (2103, "room", 0.0),                 # 黒背景(お会計中)
    (2168, "room", 0.0),                 # 「またお越し」重複回避
    (2206, 23, 4.300), (2293, 24, 8.600), (2400, 24, 12.600),
    (2450, None, 0.0),
]

DUCK_GAIN = 0.07           # 会話中の現場音(約-23dB)
SEG_FADE = 0.012
DUCK_RAMP = 0.12
SPEECH_THRESH_DB = -55.0

# 効果音レベル調整(冷蔵庫-5dB / 氷-8dB / 混ぜ-6dB、現場音のみに適用)
LEVEL_POINTS = [
    (0.000, 0.0), (21.500, 0.0), (21.620, -5.0),
    (24.013, -5.0), (24.133, -8.0), (28.613, -8.0), (28.733, 0.0),
    (37.200, 0.0), (37.320, -6.0), (39.880, -6.0), (40.000, 0.0),
]

# ------------------------------------------------------- 合成SFX・合成BGM
SFX_START, SFX_END, SFX_FADE = 8.066667, 9.800, 0.180
BGM_MUTE_START, BGM_MUTE_END = 8.066667, 9.980
BGM_OPEN_LEVEL_DB = -34.0
BGM_MAIN_LEVEL_DB = -30.0
BGM_OPEN_FADE_IN = 1.2
BGM_PRE_FADE_OUT = 0.3
BGM_RESUME_FADE_IN = 0.6
BGM_END_FADE = 1.5
BGM_DUCK_DB = -3.5

# ---------------------------------------------------------------- 商品画像
PRODUCTS = [
    # (画像, ポップイン開始frame, 表示終了frame(排他的), 中心x(1080), 中心y(1080))
    (ROOT / "images" / "課題素材(写真2).jpeg", 245, 332, 216, 430),  # 二階堂
    (ROOT / "images" / "課題素材(写真1).jpeg", 263, 307, 540, 430),  # 黒霧島
    (ROOT / "images" / "課題素材(写真3).webp", 285, 307, 864, 430),  # 角瓶
]
CARD_MAX_W, CARD_MAX_H = 280, 400   # 1080座標系での商品カード最大サイズ
CARD_BORDER = 10
POP_SCALES = [0.55, 0.85, 1.10, 1.14, 1.07, 1.01, 1.0]  # ポップインの7フレーム
SOLO_SCALE = 1.2                     # カラー復帰後の二階堂単独表示


def run(args: list[str]) -> None:
    p = subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", *args],
                       capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError(p.stderr)


def material(n: int) -> Path:
    return ROOT / "videos" / f"課題素材{n}.mov"


def frame_count(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


# ================================================================ 映像
def render_base(work: Path) -> Path:
    clips: list[Path] = []
    for i, ((start, source, source_in, crop), (end, _, _, _)) in enumerate(
            zip(CUTS, CUTS[1:])):
        n = end - start
        clip = work / f"clip_{i:02d}.mp4"
        enc = ["-an", "-frames:v", str(n), "-c:v", "libx264", "-preset", "veryfast",
               "-crf", "18", "-pix_fmt", "yuv420p", str(clip)]
        if source is None:
            # 黒背景カット: assets/黒背景.PNG を敷く
            # (-framerate 30 を明示しないと画像入力が25fps扱いになり、
            #  連結後のタイムスタンプがずれてオーバーレイ同期が壊れる)
            run(["-framerate", str(FPS), "-loop", "1", "-i", str(BLACK_PNG),
                 "-vf", f"scale={W}:{H}:force_original_aspect_ratio=increase,"
                        f"crop={W}:{H},fps={FPS}", *enc])
        else:
            vf = f"scale={W}:{H},fps={FPS}"
            if crop:
                cw, ch, cx, cy = crop
                vf = f"crop={cw}:{ch}:{cx}:{cy}," + vf
            if start < GRAY_END and end > GRAY_START:
                vf += ",hue=s=0"       # 思案シーンのモノクロ化
            run(["-ss", f"{source_in:.6f}", "-i", str(material(source)),
                 "-vf", vf, *enc])
        if frame_count(clip) != n:
            raise RuntimeError(f"clip {i} ({start}-{end}) frame count mismatch")
        clips.append(clip)
        print(f"  {start:4d}-{end - 1:4d}  素材{source if source else '黒背景'} "
              f"in={source_in:.3f}{' crop' if crop else ''}")
    listing = work / "clips.ffconcat"
    listing.write_text("ffconcat version 1.0\n" + "".join(f"file '{c}'\n" for c in clips))
    base = work / "base.mp4"
    run(["-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", str(base)])
    if frame_count(base) != TOTAL_FRAMES:
        raise RuntimeError("base frame count mismatch")
    return base


def load_product_card(path: Path) -> Image.Image:
    img = Image.open(path).convert("RGB")
    scale = min(CARD_MAX_W / img.width, CARD_MAX_H / img.height)
    img = img.resize((round(img.width * scale), round(img.height * scale)),
                     Image.LANCZOS)
    card = Image.new("RGBA",
                     (img.width + 2 * CARD_BORDER, img.height + 2 * CARD_BORDER),
                     (255, 255, 255, 255))
    card.paste(img, (CARD_BORDER, CARD_BORDER))
    return card


def build_product_frames(work: Path) -> list[Path]:
    """商品画像演出のフレーム連番PNG(EFFECT_START〜EFFECT_END-1)を生成する。"""
    cards = [load_product_card(p) for p, *_ in PRODUCTS]
    paths = []
    for f in range(EFFECT_START, EFFECT_END):
        canvas = Image.new("RGBA", (telops.DW, telops.DH), (0, 0, 0, 0))
        for (path, f_in, f_out, cx, cy), card in zip(PRODUCTS, cards):
            if not (f_in <= f < f_out):
                continue
            k = f - f_in
            scale = POP_SCALES[k] if k < len(POP_SCALES) else 1.0
            if f >= GRAY_END:  # カラー復帰後は二階堂のみ・少し大きく
                k2 = f - GRAY_END
                scale = [1.0, 1.12, 1.22, 1.2][k2] if k2 < 4 else SOLO_SCALE
            w = max(1, round(card.width * scale))
            h = max(1, round(card.height * scale))
            scaled = card.resize((w, h), Image.LANCZOS)
            canvas.alpha_composite(scaled, (cx - w // 2, cy - h // 2))
        out = work / f"product_{f:04d}.png"
        canvas.resize((W, H), Image.LANCZOS).save(out)
        paths.append(out)
    return paths


def compose_video(base: Path, work: Path) -> Path:
    """テロップ・商品画像・モザイクをフレーム番号ベースで合成する。

    ffmpegのoverlayフィルタはベースのPTSに同期するため、クリップ連結由来の
    タイムスタンプ揺れで後半のテロップが脱落することがある。ここでは
    cv2で全フレームを順読みし、numpyでアルファ合成する(PTS非依存)。
    """
    patched = list(telops.BAND_TELOPS)
    patched[3] = (3.16, 4.833, [[("かしこまりました", "w")]])
    patched[4] = (4.833, 5.80, [[("こちらの席へどうぞ", "w")]])
    patched[5] = (5.80, 6.73, [[("ありがとうございます", "w")]])
    original = telops.BAND_TELOPS
    telops.BAND_TELOPS = patched
    overlays = telops.build_overlays(work)
    telops.BAND_TELOPS = original

    # フレーム番号 → 合成するPNGパスのリスト(商品画像を先に、テロップを上に)
    schedule: dict[int, list[Path]] = {}
    for f, p in zip(range(EFFECT_START, EFFECT_END), build_product_frames(work)):
        schedule.setdefault(f, []).append(p)
    for ov in sorted(overlays, key=lambda x: x["start"]):
        f0, f1 = round(ov["start"] * FPS), round(ov["end"] * FPS)
        for f in range(f0, min(f1, TOTAL_FRAMES)):
            schedule.setdefault(f, []).append(ov["path"])

    cache: dict[Path, tuple[np.ndarray, np.ndarray]] = {}

    def blend(frame: np.ndarray, path: Path) -> np.ndarray:
        if path not in cache:
            png = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            alpha = (png[:, :, 3:4].astype(np.float32)) / 255.0
            cache[path] = (png[:, :, :3].astype(np.float32) * alpha, 1.0 - alpha)
        fg, inv = cache[path]
        return (frame.astype(np.float32) * inv + fg).astype(np.uint8)

    boxes = mosaic.detect_face_boxes()
    cap = cv2.VideoCapture(str(base))
    out = work / "video_final.mp4"
    cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
           "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "pipe:0",
           "-an", "-frames:v", str(TOTAL_FRAMES), "-c:v", "libx264",
           "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", str(out)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    count = 0
    while count < TOTAL_FRAMES:
        ok, frame = cap.read()
        if not ok:
            break
        for path in schedule.get(count, ()):
            frame = blend(frame, path)
        if count < len(boxes):
            mosaic.pixelate(frame, boxes[count])
        proc.stdin.write(frame.tobytes())
        count += 1
    cap.release()
    proc.stdin.close()
    err = proc.stderr.read().decode()
    if proc.wait() or count != TOTAL_FRAMES:
        raise RuntimeError(f"compose encode failed ({count} frames): {err}")
    return out


# ================================================================ 音声
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


def fit(seg: np.ndarray, n: int) -> np.ndarray:
    if len(seg) < n:
        seg = np.pad(seg, (0, n - len(seg)))
    return seg[:n]


def cos_ramp(n: int) -> np.ndarray:
    return 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, n))


def build_stem(total: int) -> np.ndarray:
    stem = np.zeros(total)
    for ref_at, src, s_in, s_out, f_in, f_out, line in STEM:
        seg = load_audio(material(src), s_in, s_out - s_in)
        n_in, n_out = int(f_in * SR), int(f_out * SR)
        if n_in:
            seg[:n_in] *= cos_ramp(n_in)
        if n_out:
            seg[-n_out:] *= cos_ramp(n_out)[::-1]
        a = int(round(ref_at * SR))
        seg = seg[:max(0, total - a)]
        stem[a:a + len(seg)] += seg
        print(f"  stem {ref_at:7.3f}s 素材{src} {s_in:.3f}-{s_out:.3f}  {line}")
    stem[int(ATTO_ZERO[0] * SR):int(ATTO_ZERO[1] * SR)] = 0.0
    return stem


def roomtone_tile() -> np.ndarray:
    tile = load_audio(material(ROOMTONE_SRC), ROOMTONE_IN, ROOMTONE_OUT - ROOMTONE_IN)
    xf = int(ROOMTONE_XFADE * SR)
    ramp = cos_ramp(xf)
    tile = tile.copy()
    tile[:xf] *= ramp
    tile[-xf:] *= ramp[::-1]
    return tile


def build_roomtone_bed(total: int) -> np.ndarray:
    """冒頭0〜15.233秒の人声なしルームトーンベッド。"""
    bed = np.zeros(total)
    tile = roomtone_tile()
    hop = len(tile) - int(ROOMTONE_XFADE * SR)
    end = int(round(ROOMTONE_END * SR))
    pos = 0
    while pos < end:
        piece = tile[:min(len(tile), total - pos)]
        bed[pos:pos + len(piece)] += piece
        pos += hop
    bed[end:] = 0.0
    n0 = int(0.010 * SR)
    bed[:n0] *= np.linspace(0, 1, n0)
    nf = int(ROOMTONE_TAIL_FADE * SR)
    a = int(round(FOLEY_START_SEC * SR))
    bed[a:a + nf] *= cos_ramp(nf)[::-1]
    bed[a + nf:] = 0.0
    return bed


def build_foley(total: int) -> np.ndarray:
    amb = np.zeros(total)
    fade_n = int(SEG_FADE * SR)
    fade_in = cos_ramp(fade_n)
    tile = roomtone_tile()
    for (start, src, s_in), (end, _, _) in zip(FOLEY, FOLEY[1:]):
        n0, n1 = int(round(start / FPS * SR)), int(round(end / FPS * SR))
        n = n1 - n0
        if src is None:
            continue
        if src == "room":
            reps = int(np.ceil(n / len(tile)))
            seg = np.tile(tile, reps)[:n].copy()
        else:
            seg = fit(load_audio(material(src), s_in, n / SR + 0.05), n).copy()
        seg[:fade_n] *= fade_in
        seg[-fade_n:] *= fade_in[::-1]
        amb[n0:n1] = seg
    return amb


def speech_mask(stem: np.ndarray, total: int) -> np.ndarray:
    """10msフレーム単位の有声マスク(前後拡張+隙間埋め済み)を返す。"""
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
    grown = np.zeros_like(active)
    for i in np.where(active)[0]:
        grown[max(0, i - pad):i + pad + 1] = True
    gap = int(0.25 / 0.010)
    on = np.where(grown)[0]
    for a, b in zip(on, on[1:]):
        if 0 < b - a <= gap:
            grown[a:b] = True
    return grown


def mask_to_gain(grown: np.ndarray, total: int, duck: float, ramp_sec: float) -> np.ndarray:
    hop = int(0.010 * SR)
    gain = np.repeat(np.where(grown, duck, 1.0), hop)[:total]
    if len(gain) < total:
        gain = np.pad(gain, (0, total - len(gain)), constant_values=gain[-1])
    k = int(ramp_sec * SR)
    kernel = np.hanning(k)
    kernel /= kernel.sum()
    return np.convolve(gain, kernel, mode="same")


def level_curve(total: int) -> np.ndarray:
    times = np.arange(total) / SR
    pt = np.array([p[0] for p in LEVEL_POINTS] + [TOTAL_SEC])
    pd = np.array([p[1] for p in LEVEL_POINTS] + [0.0])
    return np.power(10.0, np.interp(times, pt, pd) / 20.0)


def synth_thinking_sfx(total: int) -> np.ndarray:
    """マリンバ風の上昇アルペジオ×2(独自合成の思案効果音)。"""
    out = np.zeros(total)
    notes = [659.26, 783.99, 880.00, 1046.5]        # E5 G5 A5 C6
    motif_gap = 0.86
    for rep in range(2):
        for i, f0 in enumerate(notes):
            t0 = SFX_START + rep * motif_gap + i * 0.16
            n = int(0.45 * SR)
            t = np.arange(n) / SR
            tone = (np.sin(2 * np.pi * f0 * t) * np.exp(-t / 0.16)
                    + 0.35 * np.sin(2 * np.pi * f0 * 4.0 * t) * np.exp(-t / 0.05))
            tone[:int(0.002 * SR)] *= cos_ramp(int(0.002 * SR))
            a = int(t0 * SR)
            seg = tone[:max(0, total - a)]
            out[a:a + len(seg)] += seg
    # 区間整形: 8.067〜9.8秒、末尾180msフェードアウト
    w = np.zeros(total)
    i0, i1 = int(SFX_START * SR), int(SFX_END * SR)
    w[i0:i1] = 1.0
    nf = int(SFX_FADE * SR)
    w[i1 - nf:i1] *= cos_ramp(nf)[::-1]
    out *= w
    body = out[i0:i1]
    rms = np.sqrt(np.mean(body ** 2)) + 1e-12
    out *= 10 ** (-23.0 / 20) / rms
    return out


def synth_bgm(total: int, grown_speech: np.ndarray) -> np.ndarray:
    """ローファイ・パッド調の独自合成BGMベッド(全編)。"""
    rng = np.random.default_rng(7)
    beat = 60.0 / 88.0                    # 88 BPM
    bar = 4 * beat
    # Am7 → Fmaj7 → Cmaj7 → G (MIDIノート)
    chords = [[57, 60, 64, 67], [53, 57, 60, 64], [55, 60, 64, 67], [55, 59, 62, 67]]
    basses = [45, 41, 48, 43]

    def hz(m: int) -> float:
        return 440.0 * 2 ** ((m - 69) / 12)

    music = np.zeros(total)
    n_bars = int(np.ceil(TOTAL_SEC / bar))
    pad_n = int(bar * SR)
    t_pad = np.arange(pad_n) / SR
    env = np.minimum(1.0, t_pad / 0.6) * np.minimum(1.0, (bar - t_pad) / 0.8)
    for b in range(n_bars):
        chord = chords[b % 4]
        a = int(b * bar * SR)
        seg_n = min(pad_n, total - a)
        if seg_n <= 0:
            break
        t = t_pad[:seg_n]
        seg = np.zeros(seg_n)
        for m in chord:
            f0 = hz(m)
            vib = 1.0 + 0.002 * np.sin(2 * np.pi * 0.9 * t + rng.uniform(0, 6.28))
            seg += (np.sin(2 * np.pi * f0 * vib * t)
                    + 0.30 * np.sin(2 * np.pi * 2 * f0 * t)
                    + 0.10 * np.sin(2 * np.pi * 3 * f0 * t)) * 0.25
        fb = hz(basses[b % 4])
        seg += 0.6 * np.sin(2 * np.pi * fb * t) * np.minimum(1.0, t / 0.05)
        # 軽いアルペジオ(各拍で1音、チャイム風)
        for k in range(4):
            note = chord[(b + k) % 4] + 12
            t0 = int(k * beat * SR)
            n = min(int(0.8 * SR), seg_n - t0)
            if n <= 0:
                continue
            tt = np.arange(n) / SR
            seg[t0:t0 + n] += 0.22 * np.sin(2 * np.pi * hz(note) * tt) * np.exp(-tt / 0.35)
        music[a:a + seg_n] += seg * env[:seg_n]

    # 区間エンベロープ(add_bgm.py の設計に準拠)
    t_all = np.arange(total) / SR
    g = np.zeros(total)
    m_open = t_all < BGM_MUTE_START
    m_main = t_all >= BGM_MUTE_END
    g[m_open] = 10 ** (BGM_OPEN_LEVEL_DB / 20)
    g[m_main] = 10 ** (BGM_MAIN_LEVEL_DB / 20)
    # RMS基準化: 全体をmainレベルに合わせてから区間ゲインで整音
    body_rms = np.sqrt(np.mean(music ** 2)) + 1e-12
    music /= body_rms

    fade = np.ones(total)
    n = int(BGM_OPEN_FADE_IN * SR)
    fade[:n] *= cos_ramp(n)
    i0 = int((BGM_MUTE_START - BGM_PRE_FADE_OUT) * SR)
    i1 = int(BGM_MUTE_START * SR)
    fade[i0:i1] *= cos_ramp(i1 - i0)[::-1]
    j0, j1 = int(BGM_MUTE_END * SR), int((BGM_MUTE_END + BGM_RESUME_FADE_IN) * SR)
    fade[j0:j1] *= cos_ramp(j1 - j0)
    k0 = int((TOTAL_SEC - BGM_END_FADE) * SR)
    fade[k0:] *= cos_ramp(total - k0)[::-1]

    duck = mask_to_gain(grown_speech, total, 10 ** (BGM_DUCK_DB / 20), 0.25)
    return music * g * fade * duck


def build_audio(work: Path) -> Path:
    total = TOTAL_SAMPLES
    print("[a1] 会話ステム")
    stem = build_stem(total)
    print("[a2] ルームトーン+現場音")
    bed = build_roomtone_bed(total)
    amb = build_foley(total)
    grown = speech_mask(stem, total)
    gate = mask_to_gain(grown, total, DUCK_GAIN, DUCK_RAMP)
    amb_final = amb * gate * level_curve(total)
    print("[a3] 合成SFX+合成BGM")
    sfx = synth_thinking_sfx(total)
    bgm = synth_bgm(total, grown)

    f = 1.0
    for _ in range(8):
        mix = stem + sfx + bed + amb_final * f + bgm
        peak = np.abs(mix).max()
        if peak <= 0.99:
            break
        f *= 0.99 / peak
    print(f"  peaks: stem {np.abs(stem).max():.3f} mix {peak:.3f} amb factor {f:.3f}")

    raw = work / "mix.f32"
    raw.write_bytes(mix.astype(np.float32).tobytes())
    aac = work / "mix.m4a"
    run(["-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", str(raw),
         "-ac", "2", "-c:a", "aac", "-b:a", "192k", str(aac)])
    return aac


# ================================================================ main
def main() -> int:
    OUT.parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="rebuild_prototype_") as tmp:
        work = Path(tmp)
        print("[1/5] 映像カットを素材から書き出し")
        base = render_base(work)
        print("[2-3/5] テロップ+商品画像+モザイクをフレーム合成")
        video = compose_video(base, work)
        print("[4/5] 音声トラックを素材から構築")
        aac = build_audio(work)
        print("[5/5] 多重化+検証")
        candidate = work / "prototype_candidate.mp4"
        run(["-i", str(video), "-i", str(aac), "-map", "0:v:0", "-map", "1:a:0",
             "-c:v", "copy", "-c:a", "copy", "-t", f"{TOTAL_SEC:.7f}",
             "-movflags", "+faststart", str(candidate)])
        run(["-v", "error", "-i", str(candidate), "-f", "null", "-"])
        if frame_count(candidate) != TOTAL_FRAMES:
            raise RuntimeError("final frame count mismatch")
        # 検証済みの新版で置換(tmpfsまたぎのため一旦同一FSへコピーして原子的に置換)
        staging = OUT.with_suffix(".rebuilt.tmp.mp4")
        staging.write_bytes(candidate.read_bytes())
        staging.replace(OUT)
    print(f"generated: {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
