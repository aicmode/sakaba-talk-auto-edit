#!/usr/bin/env python3
"""試作動画へのテロップ・字幕の追加(フェーズ3・テロップ確認用)。

output/prototype.mp4 (540x960/30fps) にテロップのみを重ね、
output/prototype_text.mp4 を生成する。prototype.mp4 は上書きしない。

テロップ様式は見本動画のフレーム観察(0.5秒間隔コンタクトシート+原寸確認)から採取:
- 会話字幕: 全幅の半透明黒帯(1080x1920換算で y800-1040)+中央揃え、
  ヒラギノ明朝 W6 相当の白文字+黒縁取り。文字中心は y≈920。
- 強調語: 「二階堂」=赤+白縁(大きめ)、「お茶割り」=緑+白縁(大きめ)。
  提供シーンの「お茶割り」は黄緑。
- 「乾杯」: 縦書き2文字・特大・半透明白・縁取りなし。
- 「ごくっ」×2: 水色+白縁。飲むカットの左上→中央に時間差で出現。
- 「お会計中」: 黒背景カットに角ゴシック W8 の白文字(帯なし)。
- 「ありがとうございました」: 縦書き・白文字+ピンク縁+ピンクグロー、
  フェードインして最終カット終端まで表示。

表示タイミングは見本の実測(コンタクトシート)に合わせた絶対時刻。
商品画像・モザイク・ズーム等のエフェクト、BGM・効果音は本スクリプトの対象外。
"""

import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from imageio_ffmpeg import get_ffmpeg_exe

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "output" / "prototype.mp4"
OUT = ROOT / "output" / "prototype_text.mp4"

# デザインは見本と同じ 1080x1920 座標系で行い、出力時に 540x960 へ縮小する
DW, DH = 1080, 1920
OW, OH = 540, 960

MINCHO = ("/System/Library/Fonts/ヒラギノ明朝 ProN.ttc", 2)   # W6
GOTHIC = ("/System/Library/Fonts/ヒラギノ角ゴシック W8.ttc", 0)  # W8

BAND_TOP, BAND_BOTTOM = 800, 1040   # 半透明黒帯の範囲
BAND_ALPHA = 135
TEXT_CY = 920                       # 帯テロップの文字中心

WHITE = (255, 255, 255, 255)
BLACK = (0, 0, 0, 255)
RED = (222, 30, 24, 255)        # 二階堂
GREEN = (56, 160, 48, 255)      # お茶割り(注文シーン)
LIME = (172, 214, 30, 255)      # お茶割り(提供シーン)
LIGHTBLUE = (150, 218, 250, 255)  # ごくっ
PINK = (232, 28, 196, 255)      # 縦書きありがとうございました の縁
PINK_GLOW = (255, 92, 226, 110)

# 文字種スタイル: (フォント, サイズ, 塗り, [(縁色, 縁太さ), ...] 外側から)
STYLES = {
    "w":      (MINCHO, 66, WHITE, [(BLACK, 6)]),
    "w2":     (MINCHO, 56, WHITE, [(BLACK, 5)]),       # 2行テロップ用
    "red":    (MINCHO, 96, RED, [(BLACK, 10), (WHITE, 5)]),
    "green":  (MINCHO, 84, GREEN, [(BLACK, 9), (WHITE, 5)]),
    "red_s":  (MINCHO, 78, RED, [(BLACK, 8), (WHITE, 4)]),
    "lime_s": (MINCHO, 78, LIME, [(BLACK, 8), (WHITE, 4)]),
    "gokuri": (MINCHO, 58, LIGHTBLUE, [(WHITE, 6)]),
    "kaikei": (GOTHIC, 148, WHITE, []),
}

# 帯付き会話テロップ: (開始, 終了, 行リスト)。行 = [(文字列, スタイル), ...]
BAND_TELOPS = [
    (0.92, 1.75, [[("いらっしゃいませ", "w")]]),
    (1.75, 2.57, [[("何名様ですか?", "w")]]),
    (2.57, 3.16, [[("1人です", "w")]]),
    (3.16, 4.25, [[("かしこまりました", "w")]]),
    (4.25, 5.20, [[("こちらの席へどうぞ", "w")]]),
    (5.20, 6.73, [[("ありがとうございます", "w")]]),
    (6.73, 8.07, [[("お飲み物どうされますか?", "w")]]),
    (10.23, 11.08, [[("二階堂", "red"), ("で", "w")]]),
    (11.08, 11.70, [[("かしこまりました", "w")]]),
    (11.70, 12.95, [[("割り方はどうされますか?", "w")]]),
    (12.95, 13.90, [[("お茶割り", "green"), ("でお願いします", "w")]]),
    (13.90, 15.13, [[("少々お待ちくださいませ", "w")]]),
    (40.53, 41.68, [[("お待たせいたしました", "w")]]),
    (41.68, 43.35, [[("こちら", "w2"), ("二階堂", "red_s"), ("の", "w2"),
                     ("お茶割り", "lime_s"), ("になります", "w2")]]),
    (47.15, 48.70, [[("店員さんおいくつなんですか?", "w")]]),
    (48.70, 51.05, [[("今年25の年になりますね", "w")]]),
    (51.05, 51.90, [[("お若いんですね", "w")]]),
    (51.90, 53.23, [[("良かったら店員さんも一杯どうぞ", "w")]]),
    (53.23, 54.30, [[("ありがとうございます", "w")]]),
    (54.30, 56.27, [[("私もお客様と", "w2")],
                    [("同じものをいただいていいですか?", "w2")]]),
    (56.27, 57.08, [[("はい", "w")]]),
    (60.30, 61.30, [[("ありがとうございます", "w")]]),
    (61.30, 61.90, [[("いただきます", "w")]]),
    (65.90, 68.00, [[("うわっもうこんな時間かぁ、、、", "w")]]),
    (68.00, 69.30, [[("ご馳走様です!楽しかったです!", "w")]]),
    (69.30, 70.08, [[("ありがとうございます", "w")]]),
    (72.25, 73.53, [[("またお越しくださいませ", "w")]]),
]


def font_of(style: str) -> ImageFont.FreeTypeFont:
    (path, index), size = STYLES[style][0], STYLES[style][1]
    return ImageFont.truetype(path, size, index=index)


def draw_segment(d: ImageDraw.ImageDraw, xy, text, style, anchor):
    _, _, fill, strokes = STYLES[style]
    font = font_of(style)
    for color, width in strokes:
        d.text(xy, text, font=font, fill=color, stroke_width=width,
               stroke_fill=color, anchor=anchor)
    d.text(xy, text, font=font, fill=fill, anchor=anchor)


def new_canvas() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGBA", (DW, DH), (0, 0, 0, 0))
    return img, ImageDraw.Draw(img)


def render_band_telop(lines) -> Image.Image:
    img, d = new_canvas()
    d.rectangle([0, BAND_TOP, DW, BAND_BOTTOM], fill=(0, 0, 0, BAND_ALPHA))
    if len(lines) == 1:
        centers = [TEXT_CY]
    else:  # 2行(見本では y≈882 / 958)
        centers = [882, 958]
    for line, cy in zip(lines, centers):
        widths = [d.textlength(t, font=font_of(s)) for t, s in line]
        x = DW / 2 - sum(widths) / 2
        for (text, style), w in zip(line, widths):
            draw_segment(d, (x, cy), text, style, anchor="lm")
            x += w
    return img


def render_kanpai() -> Image.Image:
    """特大「乾杯」縦書き・半透明白(縁取りなし)。"""
    img, d = new_canvas()
    (path, index) = MINCHO
    font = ImageFont.truetype(path, 430, index=index)
    for ch, cy in (("乾", 800), ("杯", 1215)):
        d.text((DW / 2, cy), ch, font=font,
               fill=(255, 255, 255, 185), anchor="mm")
    return img


def render_gokuri(positions) -> Image.Image:
    img, d = new_canvas()
    for cx, cy in positions:
        draw_segment(d, (cx, cy), "ごくっ", "gokuri", anchor="mm")
    return img


def render_kaikei() -> Image.Image:
    img, d = new_canvas()
    draw_segment(d, (DW / 2, 930), "お会計中", "kaikei", anchor="mm")
    return img


def render_vertical_thanks() -> Image.Image:
    """縦書き「ありがとうございました」白+ピンク縁+グロー。"""
    img, d = new_canvas()
    (path, index) = MINCHO
    font = ImageFont.truetype(path, 112, index=index)
    text = "ありがとうございました"
    for i, ch in enumerate(text):
        xy = (DW / 2, 320 + i * 126)
        d.text(xy, ch, font=font, fill=PINK_GLOW, stroke_width=18,
               stroke_fill=PINK_GLOW, anchor="mm")
        d.text(xy, ch, font=font, fill=PINK, stroke_width=9,
               stroke_fill=PINK, anchor="mm")
        d.text(xy, ch, font=font, fill=WHITE, anchor="mm")
    return img


def build_overlays(work: Path) -> list[dict]:
    """テロップPNG(540x960へ縮小済み)と表示区間のリストを返す。

    テロップ同士は時間的に重ならないため、透明PNGで隙間を埋めた
    単一の画像シーケンス(concatデマルチプレクサ)として合成する。
    フェードインはアルファを段階的に変えたPNG連番で表現する。
    """
    overlays = []

    def save(img: Image.Image, name: str) -> Path:
        path = work / name
        img.resize((OW, OH), Image.LANCZOS).save(path)
        return path

    def add(img: Image.Image, start: float, end: float, fade_in: float = 0.0):
        n = len(overlays) + 1
        if fade_in > 0:
            steps = max(1, round(fade_in * 30))
            for k in range(1, steps + 1):
                faded = img.copy()
                alpha = faded.getchannel("A").point(
                    lambda v, f=k / steps: int(v * f))
                faded.putalpha(alpha)
                overlays.append({
                    "path": save(faded, f"telop_{n:02d}_f{k:02d}.png"),
                    "start": start + (k - 1) * fade_in / steps,
                    "end": start + k * fade_in / steps,
                })
            start += fade_in
            n = len(overlays) + 1
        overlays.append({"path": save(img, f"telop_{n:02d}.png"),
                         "start": start, "end": end})

    for start, end, lines in BAND_TELOPS:
        add(render_band_telop(lines), start, end)
    add(render_kanpai(), 61.90, 62.80)
    add(render_gokuri([(270, 640)]), 63.80, 64.30)          # ごくっ 1個目
    add(render_gokuri([(270, 640), (565, 810)]), 64.30, 64.50)  # 2個目追加
    add(render_kaikei(), 70.08, 72.25)
    add(render_vertical_thanks(), 80.00, 81.68, fade_in=0.4)
    return overlays


def run_ffmpeg(overlays: list[dict], work: Path) -> None:
    transparent = work / "transparent.png"
    Image.new("RGBA", (OW, OH), (0, 0, 0, 0)).save(transparent)

    # 透明PNGで隙間を埋めた ffconcat リストを作る
    lines = ["ffconcat version 1.0"]
    t = 0.0
    for ov in sorted(overlays, key=lambda o: o["start"]):
        if ov["start"] > t + 1e-6:
            lines += [f"file '{transparent}'",
                      f"duration {ov['start'] - t:.4f}"]
        lines += [f"file '{ov['path']}'",
                  f"duration {ov['end'] - ov['start']:.4f}"]
        t = ov["end"]
    lines += [f"file '{transparent}'", "duration 2.0",
              f"file '{transparent}'"]
    listfile = work / "overlay.ffconcat"
    listfile.write_text("\n".join(lines) + "\n")

    args = [get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(SRC),
            "-f", "concat", "-safe", "0", "-i", str(listfile),
            # 出力の -shortest は終盤のオーバーレイを打ち切るため使わない。
            # 本編EOFで止まるよう overlay 側の shortest=1 で終端を揃える。
            "-filter_complex",
            "[1:v]fps=30,format=rgba[ov];[0:v][ov]overlay=0:0:shortest=1[v]",
            "-map", "[v]", "-map", "0:a",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "21",
            "-pix_fmt", "yuv420p", "-c:a", "copy", str(OUT)]
    proc = subprocess.run(args, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{proc.stderr}")


def main() -> None:
    if not SRC.exists():
        raise FileNotFoundError(SRC)
    with tempfile.TemporaryDirectory(prefix="telop_") as tmp:
        work = Path(tmp)
        print(f"[1/2] テロップ画像を生成中…")
        overlays = build_overlays(work)
        print(f"  {len(overlays)} 枚(フェード用連番を含む)")
        print("[2/2] 動画へ合成中…")
        run_ffmpeg(overlays, work)
    size_mb = OUT.stat().st_size / 1e6
    print(f"\n完了: {OUT.relative_to(ROOT)} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    sys.exit(main())
