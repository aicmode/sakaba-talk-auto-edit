#!/usr/bin/env python3
"""prototype.mp4 の部分差し替えパッチ(2026-07-18)。

全体再レンダリングは行わない。確定済み区間はストリームコピーで 1 ビットも
変更せず、修正対象の時間帯だけを再構成して差し替える。

修正内容(すべて reference 実測値に基づく):
 ①③ 商品画像3枚シーン (frame 242-331 / 8.07-11.07s)
    - 白枠カード → 切り抜きボトル+白フチ(見本と同スタイル・同寸・同位置)
    - 出現時刻/線形ポップイン/カラー復帰後の二階堂ソロ拡大を見本実測に一致
    - 「二階堂で」テロップを見本サイズへ拡大 (二階堂140px赤+白縁 / で90px白)
    - 効果音: 合成マリンバを廃止し、見本の思案窓 (8.067-9.95s;
      BGM無音・無発話区間のピコッSFX+場内アンビエンス) を移植
 ① 後半会話カットを reference 実カットへ一致 (frame 1183-1856)
    - 39.42-48.63s は素材8の連続テイク (offset -27.433s) + ズームクロップ
    - 各スピーカーショットのイン点 = 会話ステムのオフセット (口パク一致)
    - 56.82-58.43s は素材6(32.2s〜)、58.43-60.2s は素材5@4.433
    - 60.2-61.85s は素材16@8.829 (口パク一致)、乾杯カットを 61.867s へ
 ① 終盤 (frame 1884-2205)
    - ごくっショットをズーム化(素材16 crop)+ごくっテロップ時刻/位置修正
    - 64.63-65.13s の黒への暗転(dip-to-black)を再現
    - 素材9 を 63.988s 開始+素材内 3.645-4.162s スキップ(見本と同じ間詰め)
    - 68.97s で素材12@1.212 へカット、72.27s は素材12@2.496 (口パク一致)
 ④ 帯テロップ時刻を reference 実測へ (40.2/41.317/43.017/47.283/48.633/
    51.05/51.783/53.233/54.117/56.267/56.817/60.2/61.167/61.85/65.917/
    67.633/68.967/72.25-73.533)
 ⑤ BGM は昨日の reference 由来 Demucs ステムのまま変更しない。

映像: コピー区間 [0,242),[387,1064),[1857,1884),[2206,2450) はストリームコピー。
      再エンコード区間 [242,387),[1064,1857),[1884,2206) のうち、実変更のない
      バッファ (332-386 / 1064-1182 / 2103-2167) は現行フレームをデコードして
      同一内容のまま再圧縮する(キーフレーム境界の都合による)。
音声: 昨日の決定論的ミックスを同一コードパスで再構築し、
      W1[8.067,9.95] W2[39.42,62.0] W3[62.7,69.7] W5[72.25,73.55] のみ変更。
      窓外は PCM ビット一致のまま一括 AAC エンコード。
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))

import rebuild_prototype as rb            # noqa: E402
import render_prototype_text as telops    # noqa: E402
import replace_bgm_from_reference as rbgm  # noqa: E402  (Demucsステム経路)

OUT = ROOT / "output" / "prototype.mp4"
REF = ROOT / "reference" / "reference.mp4"
FPS, W, H = 30, 540, 960
SR = 48000
TOTAL_FRAMES = 2450
FFMPEG = rb.FFMPEG

# ---------------------------------------------------------------- 映像: 設計値
# ズームクロップ (w,h,x,y) 素材ネイティブ1080x1920 (reference テンプレート照合)
CROP_A = (632, 1124, 344, 364)   # 素材8  39.42-40.20 z1.71
CROP_B = (480, 854, 565, 505)    # 素材8  41.32-43.00 z2.25
CROP_C = (676, 1200, 50, 103)    # 素材8  43.00-45.67 z1.60
CROP_D = (584, 1038, 173, 9)     # 素材16 62.80-64.90 z1.85 (ごくっ)
CROP_E = (618, 1098, 192, 126)   # 素材9  65.93-67.63 z1.75

# 再エンコードセグメント: (名前, 開始f, 終了f, カット列)
# カット: (開始f, 素材番号 or "cur", 素材イン秒, crop, mono)
SEGMENTS = [
    ("RA", 242, 387, [
        (242, 25, 2.800, None, True),
        (307, 25, 6.300, None, False),
        (327, 11, 7.000, None, False),
        (332, "cur", 0, None, False),          # 現行フレーム 332-386 (無変更)
    ]),
    ("RB", 1064, 1857, [
        (1064, "cur", 0, None, False),         # 現行フレーム 1064-1182 (無変更)
        (1183, 8, 12.000, CROP_A, False),      # 客待ち(ズーム)
        (1206, 14, 3.512, None, False),        # お待たせいたしました
        (1240, 8, 13.900, CROP_B, False),      # 提供マット寄り
        (1290, 8, 15.567, CROP_C, False),      # ストローで一口(ズーム)
        (1370, 8, 18.234, None, False),        # 引き
        (1459, 20, 6.228, None, False),        # 今年25の年
        (1532, 2, 3.429, None, False),         # お若いんですね
        (1597, 21, 3.078, None, False),        # 私も同じものを
        (1688, 2, 8.862, None, False),         # はい
        (1704, 6, 32.217, None, False),        # ドリンク作り寄り
        (1753, 5, 4.433, None, False),         # 後ろ姿
        (1806, 16, 8.829, None, False),        # ありがとうございます/いただきます
        (1856, 7, 0.567, None, False),         # 乾杯 1フレーム前倒し
    ]),
    ("RC", 1884, 2206, [
        (1884, 16, 13.300, CROP_D, False),     # ごくっ(ズーム)→暗転
        (1947, 9, 0.912, None, False),         # 明転: 客ワイド
        (1978, 9, 1.945, CROP_E, False),       # うわっもうこんな時間かぁ(ズーム)
        (2029, 9, 4.162, None, False),         # ご馳走様です(間スキップ後)
        (2069, 12, 1.212, None, False),        # ありがとうございます(店員)
        (2103, "cur", 0, None, False),         # 黒背景 お会計中 (無変更)
        (2168, 12, 2.496, None, False),        # またお越しくださいませ
    ]),
]
COPY_RANGES = [(0, 242), (387, 1064), (1857, 1884), (2206, 2450)]
BUFFER_RANGES = [(332, 387), (1064, 1183), (2103, 2168)]

# 暗転 (reference 実測: 64.633→64.883 黒、64.883→65.133 明転)
FADE_BLACK = 64.883
FADE_LEN = 0.25

def fade_factor(f: int) -> float:
    t = (f + 0.5) / FPS
    if 1939 <= f <= 1946:
        return max(0.0, min(1.0, (FADE_BLACK - t) / FADE_LEN))
    if 1947 <= f <= 1954:
        return max(0.0, min(1.0, (t - FADE_BLACK) / FADE_LEN))
    return 1.0

# ---------------------------------------------------------------- テロップ
BAND_SCHED = [
    (1206, 1240, [[("お待たせいたしました", "w")]]),
    (1240, 1290, [[("こちら", "w2"), ("二階堂", "red_s"), ("の", "w2"),
                   ("お茶割り", "lime_s"), ("になります", "w2")]]),
    (1418, 1459, [[("店員さんおいくつなんですか?", "w")]]),
    (1459, 1532, [[("今年25の年になりますね", "w")]]),
    (1532, 1554, [[("お若いんですね", "w")]]),
    (1554, 1597, [[("良かったら店員さんも一杯どうぞ", "w")]]),
    (1597, 1624, [[("ありがとうございます", "w")]]),
    (1624, 1688, [[("私もお客様と", "w2")],
                  [("同じものをいただいていいですか?", "w2")]]),
    (1688, 1704, [[("はい", "w")]]),
    (1806, 1835, [[("ありがとうございます", "w")]]),
    (1835, 1856, [[("いただきます", "w")]]),
    (1978, 2029, [[("うわっもうこんな時間かぁ、、、", "w")]]),
    (2029, 2069, [[("ご馳走様です!楽しかったです!", "w")]]),
    (2069, 2103, [[("ありがとうございます", "w")]]),
    (2168, 2206, [[("またお越しくださいませ", "w")]]),
]
GOKURI_1 = (1908, (245, 585))    # 63.583s〜
GOKURI_2 = (1930, (536, 860))    # 64.333s〜
GOKURI_END = 1947                # 暗転の黒まで表示
GOKURI_SIZE = 88

NIKAIDO_TELOP = (307, 332)       # 10.233-11.067 (大型「二階堂で」)

# ---------------------------------------------------------------- 商品画像
# (写真, 彩度bbox目標高, 彩度bbox中心(1080座標))
PRODUCTS = [
    (ROOT / "images" / "課題素材(写真2).jpeg", 427, (186.5, 325.5)),  # 二階堂
    (ROOT / "images" / "課題素材(写真1).jpeg", 328, (526.5, 335.0)),  # 黒霧島
    (ROOT / "images" / "課題素材(写真3).webp", 415, (867.5, 343.5)),  # 角瓶
]
POP_CURVES = [   # frame → scale (reference 実測、以降1.0)
    {243: 0.14, 244: 0.29, 245: 0.43, 246: 0.57, 247: 0.71, 248: 0.86},
    {266: 0.05, 267: 0.20, 268: 0.33, 269: 0.46, 270: 0.60, 271: 0.73, 272: 0.87},
    {290: 0.14, 291: 0.43, 292: 0.71},
]
APPEAR = [243, 266, 290]
SOLO_START, SOLO_RAMP_END, SOLO_SCALE = 307, 313, 1.17
OUTLINE_PX = 6                   # 白フチ(1080スケール)

# ---------------------------------------------------------------- 音声窓
AW1 = (387200, int(9.95 * SR))           # 思案SFX移植 (8.0667-9.95)
AW2 = (int(39.42 * SR), int(62.70 * SR))  # 後半 foley 再配置（AW3へ連続）
AW3 = (int(62.70 * SR), int(69.98 * SR))  # 素材9 ステム再配置+foley
AW5 = (int(72.25 * SR), int(73.55 * SR))  # 素材12 foley
XFADE = int(0.020 * SR)

STEM_NEW = [
    e for e in rb.STEM if not (e[1] == 9 and abs(e[0] - 63.471) < 1e-6)
]
STEM_NEW += [
    (63.988, 9, 0.000, 3.645, .010, .030, "うわっもうこんな時間かぁ(見本位置)"),
    (67.633, 9, 4.162, 5.300, .030, .010, "ご馳走様です楽しかったです(間スキップ)"),
]
STEM_NEW.sort(key=lambda e: e[0])

FOLEY_NEW = [
    (454, 18, 0.000), (496, 15, 4.750), (543, 15, 6.316), (578, 15, 13.367),
    (622, 15, 14.849), (645, 13, 0.850), (724, 1, 0.433), (812, 1, 3.417),
    (862, 6, 1.583), (920, 6, 3.517), (1064, 6, 17.500), (1116, 6, 27.500),
    # --- ここから新カット構成に合わせた現場音 ---
    (1183, 8, 12.000), (1206, 14, 3.512), (1240, 8, 13.900), (1290, 8, 15.567),
    (1370, 8, 18.234), (1459, 20, 6.228), (1532, 2, 3.429), (1597, 21, 3.078),
    (1688, 2, 8.862), (1704, 6, 32.217), (1753, 5, 4.433), (1806, 16, 8.829),
    (1856, 7, 0.567),
    (1884, 16, 13.300), (1947, 9, 0.912), (2029, 9, 4.162),
    (2069, "room", 0.0),                 # ありがとう重複回避 (見本はステムのみ)
    (2079, "room", 0.0),                 # 旧タイル位相と一致させ 69.97s 以降を不変に
    (2103, "room", 0.0),
    (2168, 12, 2.496),                   # またお越し: 映像と同オフセット
    (2206, 23, 4.300), (2293, 24, 8.600), (2400, 24, 12.600),
    (2450, None, 0.0),
]


# ================================================================ 共通
def load_audio_ref(ss: float, t: float) -> np.ndarray:
    return rb.load_audio(REF, ss, t)


def decode_frames(path: Path, f0: int, f1: int):
    """[f0,f1) を BGR フレームで順次 yield (frame-accurate)。"""
    cmd = [FFMPEG, "-hide_banner", "-loglevel", "error",
           "-i", str(path),
           "-vf", f"trim=start_frame={f0}:end_frame={f1},setpts=PTS-STARTPTS",
           "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    n = 0
    size = W * H * 3
    while n < f1 - f0:
        # pipeのreadは1フレーム未満の短いchunkを返すことがあるため、EOFまで
        # 即座に欠落扱いせず、必ず1フレーム分を集める。
        chunks = []
        remaining = size
        while remaining:
            chunk = proc.stdout.read(remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        buf = b"".join(chunks)
        if len(buf) < size:
            break
        yield np.frombuffer(buf, np.uint8).reshape(H, W, 3).copy()
        n += 1
    proc.stdout.close()
    proc.wait()
    if n != f1 - f0:
        raise RuntimeError(f"decode_frames {f0}-{f1}: got {n}")


# ================================================================ 商品ステッカー
def load_sticker(path: Path, target_sat_h: int) -> tuple[Image.Image, tuple[float, float]]:
    """白背景写真からボトル切り抜き+白フチのRGBAステッカーを作る。

    返り値: (ステッカー, ステッカー内の彩度bbox中心座標)。
    スケールは彩度bbox高が target_sat_h になるよう調整済み。
    """
    bgr = cv2.imread(str(path))
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    bg = (g >= 240).astype(np.uint8)
    # 縁からflood fillで背景を確定(ボトル内の白ハイライトは残す)
    mask = np.zeros((bg.shape[0] + 2, bg.shape[1] + 2), np.uint8)
    flood = bg.copy()
    for seed in [(0, 0), (bg.shape[1] - 1, 0), (0, bg.shape[0] - 1),
                 (bg.shape[1] - 1, bg.shape[0] - 1)]:
        if flood[seed[1], seed[0]]:
            cv2.floodFill(flood, mask, seed, 2)
    alpha = np.where(flood == 2, 0, 255).astype(np.uint8)
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    sat = (hsv[:, :, 1] > 80) & (hsv[:, :, 2] > 70) & (alpha > 0)
    ys, xs = np.where(sat)
    scale = target_sat_h / (ys.max() - ys.min())
    nw, nh = round(bgr.shape[1] * scale), round(bgr.shape[0] * scale)
    rgba = np.dstack([cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), alpha])
    img = Image.fromarray(rgba).resize((nw, nh), Image.LANCZOS)
    # 白フチ: alpha を膨張させた白レイヤーを下に敷く
    a = np.array(img.getchannel("A"))
    k = 2 * OUTLINE_PX + 1
    dil = cv2.dilate(a, np.ones((k, k), np.uint8))
    outline = Image.new("RGBA", img.size, (255, 255, 255, 0))
    outline.putalpha(Image.fromarray(dil))
    sticker = Image.alpha_composite(outline, img)
    cx = (xs.min() + xs.max()) / 2 * scale
    cy = (ys.min() + ys.max()) / 2 * scale
    return sticker, (cx, cy)


def product_overlay_frames(work: Path) -> dict[int, Path]:
    stickers = [load_sticker(p, h) for p, h, _ in PRODUCTS]
    out = {}
    for f in range(242, 332):
        items = []
        for i, ((sticker, (acx, acy)), (_, _, (tx, ty))) in enumerate(
                zip(stickers, PRODUCTS)):
            if f < APPEAR[i]:
                continue
            if f >= SOLO_START and i != 0:
                continue                       # カラー復帰後は二階堂のみ
            s = POP_CURVES[i].get(f, 1.0)
            if i == 0 and f >= SOLO_START:
                s = 1.0 + (SOLO_SCALE - 1.0) * min(
                    1.0, (f - (SOLO_START - 1)) / (SOLO_RAMP_END - SOLO_START + 1))
            w = max(1, round(sticker.width * s))
            h = max(1, round(sticker.height * s))
            scaled = sticker.resize((w, h), Image.LANCZOS)
            items.append((scaled, round(tx - acx * s), round(ty - acy * s)))
        if not items:
            continue
        canvas = Image.new("RGBA", (telops.DW, telops.DH), (0, 0, 0, 0))
        for im, x, y in items:
            canvas.alpha_composite(im, (x, y))
        p = work / f"prod_{f:04d}.png"
        canvas.resize((W, H), Image.LANCZOS).save(p)
        out[f] = p
    return out


def render_nikaido_big(work: Path) -> Path:
    """大型「二階堂で」帯テロップ (見本実測: 二階堂≈140px赤+白縁 / で≈90px白)。"""
    img, d = telops.new_canvas()
    d.rectangle([0, telops.BAND_TOP, telops.DW, telops.BAND_BOTTOM],
                fill=(0, 0, 0, telops.BAND_ALPHA))
    (path, index) = telops.MINCHO
    f_big = ImageFont.truetype(path, 140, index=index)
    f_de = ImageFont.truetype(path, 90, index=index)
    w1 = d.textlength("二階堂", font=f_big)
    w2 = d.textlength("で", font=f_de)
    x = telops.DW / 2 - (w1 + w2) / 2
    cy = telops.TEXT_CY
    for color, width in [(telops.BLACK, 12), (telops.WHITE, 6)]:
        d.text((x, cy), "二階堂", font=f_big, fill=color,
               stroke_width=width, stroke_fill=color, anchor="lm")
    d.text((x, cy), "二階堂", font=f_big, fill=telops.RED, anchor="lm")
    d.text((x + w1, cy), "で", font=f_de, fill=telops.BLACK,
           stroke_width=8, stroke_fill=telops.BLACK, anchor="lm")
    d.text((x + w1, cy), "で", font=f_de, fill=telops.WHITE, anchor="lm")
    p = work / "nikaido_big.png"
    img.resize((W, H), Image.LANCZOS).save(p)
    return p


def render_gokuri_big(work: Path, positions, name: str) -> Path:
    img, d = telops.new_canvas()
    (path, index) = telops.MINCHO
    font = ImageFont.truetype(path, GOKURI_SIZE, index=index)
    for cx, cy in positions:
        d.text((cx, cy), "ごくっ", font=font, fill=telops.LIGHTBLUE,
               stroke_width=8, stroke_fill=telops.WHITE, anchor="mm")
        d.text((cx, cy), "ごくっ", font=font, fill=telops.LIGHTBLUE, anchor="mm")
    p = work / f"{name}.png"
    img.resize((W, H), Image.LANCZOS).save(p)
    return p


def build_overlay_schedule(work: Path) -> dict[int, list[Path]]:
    sched: dict[int, list[Path]] = {}
    for f, p in product_overlay_frames(work).items():
        sched.setdefault(f, []).append(p)
    nik = render_nikaido_big(work)
    for f in range(*NIKAIDO_TELOP):
        sched.setdefault(f, []).append(nik)
    for i, (start, end, lines) in enumerate(BAND_SCHED):
        img = telops.render_band_telop(lines)
        p = work / f"band_{i:02d}.png"
        img.resize((W, H), Image.LANCZOS).save(p)
        for f in range(start, end):
            sched.setdefault(f, []).append(p)
    g1 = render_gokuri_big(work, [GOKURI_1[1]], "gokuri1")
    g12 = render_gokuri_big(work, [GOKURI_1[1], GOKURI_2[1]], "gokuri12")
    for f in range(GOKURI_1[0], GOKURI_2[0]):
        sched.setdefault(f, []).append(g1)
    for f in range(GOKURI_2[0], GOKURI_END):
        sched.setdefault(f, []).append(g12)
    kanpai = work / "kanpai.png"
    telops.render_kanpai().resize((W, H), Image.LANCZOS).save(kanpai)
    sched.setdefault(1856, []).append(kanpai)
    return sched


# ================================================================ 映像セグメント
def render_segment(name: str, f0: int, f1: int, cuts, sched, work: Path) -> Path:
    print(f"[video] {name} [{f0},{f1}) を構築")
    # 素材カットはクリップ書き出し、"cur" は現行prototypeからデコード
    sources = []
    bounds = [c[0] for c in cuts] + [f1]
    for (start, src, s_in, crop, mono), end in zip(cuts, bounds[1:]):
        n = end - start
        if src == "cur":
            sources.append(("cur", start, end))
            continue
        clip = work / f"{name}_{start}.mp4"
        vf = f"scale={W}:{H},fps={FPS}"
        if crop:
            cw, ch, cx, cy = crop
            vf = f"crop={cw}:{ch}:{cx}:{cy}," + vf
        if mono:
            vf += ",hue=s=0"
        rb.run(["-ss", f"{s_in:.6f}", "-i", str(rb.material(src)),
                "-vf", vf, "-an", "-frames:v", str(n), "-c:v", "libx264",
                "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
                str(clip)])
        if rb.frame_count(clip) != n:
            raise RuntimeError(f"{name} clip {start}: frame count mismatch")
        sources.append(("clip", clip, n))
        print(f"    f{start}-{end-1} 素材{src} in={s_in:.3f}"
              f"{' crop' if crop else ''}{' mono' if mono else ''}")

    cache: dict[Path, tuple[np.ndarray, np.ndarray]] = {}

    def blend(frame, path):
        if path not in cache:
            png = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            al = png[:, :, 3:4].astype(np.float32) / 255.0
            cache[path] = (png[:, :, :3].astype(np.float32) * al, 1.0 - al)
        fg, inv = cache[path]
        return (frame.astype(np.float32) * inv + fg).astype(np.uint8)

    seg = work / f"{name}.mp4"
    cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
           "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "pipe:0",
           "-an", "-frames:v", str(f1 - f0), "-c:v", "libx264",
           "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", str(seg)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    fcur = f0
    for entry in sources:
        if entry[0] == "cur":
            frames = decode_frames(OUT, entry[1], entry[2])
        else:
            cap = cv2.VideoCapture(str(entry[1]))
            frames = (cap.read()[1] for _ in range(entry[2]))
        for frame in frames:
            for p in sched.get(fcur, ()):  # 商品画像→テロップの順で合成
                frame = blend(frame, p)
            g = fade_factor(fcur)
            if g < 1.0:
                frame = (frame.astype(np.float32) * g).astype(np.uint8)
            proc.stdin.write(frame.tobytes())
            fcur += 1
        if entry[0] != "cur":
            cap.release()
    proc.stdin.close()
    err = proc.stderr.read().decode()
    if proc.wait() or fcur != f1:
        raise RuntimeError(f"{name} encode failed at f{fcur}: {err}")
    if rb.frame_count(seg) != f1 - f0:
        raise RuntimeError(f"{name}: final frame count mismatch")
    return seg


def extract_copy(f0: int, f1: int, work: Path) -> Path:
    seg = work / f"copy_{f0}.mp4"
    # COPY_RANGES の境界は現行マスターの実キーフレーム。入力側seekと明示durationで
    # GOPをそのままコピーする（出力側 -ss/-to はB-frameの時間基準で欠落する）。
    ss = f0 / FPS
    count = f1 - f0
    rb.run(["-ss", f"{ss:.6f}", "-i", str(OUT), "-frames:v", str(count),
            "-map", "0:v:0", "-c:v", "copy", "-an", "-avoid_negative_ts", "make_zero",
            str(seg)])
    n = rb.frame_count(seg)
    if n != f1 - f0:
        raise RuntimeError(f"copy {f0}-{f1}: {n} frames (expected {f1 - f0})")
    return seg


# ================================================================ 音声
def build_mix(stem_list, foley_list, with_sfx: bool) -> np.ndarray:
    total = rb.TOTAL_SAMPLES
    old_stem, old_foley = rb.STEM, rb.FOLEY
    rb.STEM, rb.FOLEY = stem_list, foley_list
    try:
        speech = rb.build_stem(total)
        room = rb.build_roomtone_bed(total)
        foley = rb.build_foley(total)
    finally:
        rb.STEM, rb.FOLEY = old_stem, old_foley
    grown = rb.speech_mask(speech, total)
    gate = rb.mask_to_gain(grown, total, rb.DUCK_GAIN, rb.DUCK_RAMP)
    base = speech + room + foley * gate * rb.level_curve(total)
    if with_sfx:
        base = base + rb.synth_thinking_sfx(total)
    rate, music = rbgm.load(rbgm.SEP / "bass.wav")
    for nm in ("drums.wav", "other.wav"):
        r, part = rbgm.load(rbgm.SEP / nm)
        music += part
    music = rbgm.smooth_transients(music, rate)
    import scipy.signal as sig
    music = sig.resample_poly(music, SR, rate, axis=0)
    if len(music) < total:
        music = np.pad(music, ((0, total - len(music)), (0, 0)))
    music = music[:total]
    gain = rbgm.envelope(total, speech)
    music = music * gain[:, None]
    mix = np.repeat(base[:, None], 2, axis=1) + music
    return np.clip(mix, -.999, .999)


def build_audio(work: Path) -> Path:
    print("[audio] 旧ミックスを決定論的に再構築")
    mix_old = build_mix(rb.STEM, rb.FOLEY, with_sfx=True)
    print("[audio] 新ミックス (素材9再配置+foley追従、思案窓のみ後で差替)")
    # 旧SFXを全体から外すと合成音の微小な裾まで窓外差分になる。まず旧経路を
    # 完全維持し、直後にAW1だけreference実測窓へ置換する。
    mix_new = build_mix(STEM_NEW, FOLEY_NEW, with_sfx=True)

    # W1: 見本の思案窓 (SFX+アンビエンス) を移植
    a, b = AW1
    ref_seg = load_audio_ref(a / SR, (b - a) / SR)
    ref_seg = rb.fit(ref_seg, b - a)
    win = np.repeat(ref_seg[:, None], 2, axis=1)
    r = np.linspace(0.0, 1.0, XFADE)[:, None]
    win[:XFADE] = mix_new[a:a + XFADE] * (1 - r) + win[:XFADE] * r
    win[-XFADE:] = win[-XFADE:] * (1 - r) + mix_new[b - XFADE:b] * r
    mix_new[a:b] = win

    # 窓外が旧ミックスとビット一致することを検証し、窓外は旧ミックスを採用
    windows = [AW1, AW2, AW3, AW5]
    mask = np.zeros(rb.TOTAL_SAMPLES, bool)
    for w0, w1 in windows:
        mask[w0:w1] = True
    delta = np.max(np.abs(mix_new - mix_old), axis=1)
    out_diff = delta[~mask].max()
    outside_indices = np.flatnonzero((~mask) & (delta > 1e-9))
    where = outside_indices[np.argmax(delta[outside_indices])] / SR if len(outside_indices) else -1
    print(f"[audio] 窓外 最大差 (再構築同士): {out_diff:.2e} at {where:.6f}s")
    if out_diff > 1e-9:
        raise RuntimeError("音声窓外に差分が発生 (設計エラー)")
    final = mix_old.copy()
    final[mask] = mix_new[mask]

    raw = work / "mix.f32"
    raw.write_bytes(final.astype(np.float32).tobytes())
    aac = work / "mix.m4a"
    rb.run(["-f", "f32le", "-ar", str(SR), "-ac", "2", "-i", str(raw),
            "-c:a", "aac", "-b:a", "192k", str(aac)])
    return aac


# ================================================================ main
def main() -> int:
    # 作業物はすべてOSの一時領域へ置き、成功・失敗を問わず自動削除する。
    # output/ には完成した prototype.mp4 以外を一切作らない。
    with tempfile.TemporaryDirectory(prefix="patch_latter_") as tmp:
        work = Path(tmp)
        print(f"work: {work}")
        sched = build_overlay_schedule(work)

        parts: list[tuple[int, Path]] = []
        for f0, f1 in COPY_RANGES:
            parts.append((f0, extract_copy(f0, f1, work)))
        for name, f0, f1, cuts in SEGMENTS:
            parts.append((f0, render_segment(name, f0, f1, cuts, sched, work)))
        parts.sort()

        aac = build_audio(work)

        listing = work / "concat.txt"
        listing.write_text("".join(f"file '{p}'\n" for _, p in parts))
        joined = work / "joined.mp4"
        rb.run(["-f", "concat", "-safe", "0", "-i", str(listing),
                "-c", "copy", str(joined)])
        n = rb.frame_count(joined)
        if n != TOTAL_FRAMES:
            raise RuntimeError(f"joined frame count {n} != {TOTAL_FRAMES}")

        candidate = work / "prototype_patched.mp4"
        rb.run(["-i", str(joined), "-i", str(aac), "-map", "0:v:0", "-map", "1:a:0",
                "-c:v", "copy", "-c:a", "copy", "-t", f"{rb.TOTAL_SEC:.7f}",
                "-movflags", "+faststart", str(candidate)])
        rb.run(["-v", "error", "-i", str(candidate), "-f", "null", "-"])
        if rb.frame_count(candidate) != TOTAL_FRAMES:
            raise RuntimeError("candidate frame count mismatch")

        # 候補が完全に検証された後にだけ、同一ファイルシステム上で原子的に置換する。
        staging = OUT.parent / ".prototype.patch.staging"
        try:
            shutil.copyfile(candidate, staging)
            staging.replace(OUT)
        finally:
            staging.unlink(missing_ok=True)
        print(f"patched: {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")
    print("temporary work files removed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
