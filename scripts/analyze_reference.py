#!/usr/bin/env python3
"""見本動画(reference/reference.mp4)の解析スクリプト。

出力:
  analysis/reference_metadata.json  … 動画尺・解像度・fps・音声情報
  analysis/scene_list.json          … シーン(カット)一覧
  analysis/editing_plan.json        … 編集プラン下書き(演出候補つき)
  analysis/reference_frames/        … 確認用フレーム画像
  analysis/audio_waveform.png       … 音声波形
  analysis/analysis_report.md       … 人が読むレポート(編集構成表)
  analysis/editing_sheet.csv        … 編集構成表(表計算ソフト用)

使い方:
  .venv/bin/python scripts/analyze_reference.py
  .venv/bin/python scripts/analyze_reference.py --input reference/reference.mp4 --interval 1.0
"""

import argparse
import csv
import json
import re
import subprocess
import sys
import tempfile
import wave
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import imageio_ffmpeg

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANALYSIS_DIR = PROJECT_ROOT / "analysis"
FRAMES_DIR = ANALYSIS_DIR / "reference_frames"

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

# ---- 検出しきい値(必要に応じて --help のオプションで調整可能) ----
CUT_DIFF_THRESHOLD = 28.0      # 隣接フレームの平均画素差(0-255)がこれを超えたらカット候補
CUT_HIST_THRESHOLD = 0.55      # ヒストグラム相関がこれを下回ったらカット候補
MIN_SCENE_SEC = 0.30           # これより短いシーンは前のシーンに統合
FLASH_LUMA_JUMP = 60.0         # 輝度がこれ以上跳ね上がって直後に戻ればフラッシュ候補
BLACK_LUMA = 28.0              # 平均輝度がこれ未満なら黒背景候補
MONO_SAT = 22.0                # 平均彩度(0-255)がこれ未満ならモノクロ候補
STATIC_MOTION = 1.2            # 平均モーション量がこれ未満なら静止(商品画像表示)候補
ZOOM_DIVERGENCE = 0.12         # オプティカルフローの放射成分がこれを超えたらズーム候補
ANALYSIS_WIDTH = 160           # 解析用の縮小フレーム幅(px)
GRADUAL_WINDOW_SEC = 0.75      # ディゾルブ検出: この秒数だけ離れたフレーム同士を比較
GRADUAL_DIFF = 28.0            # ディゾルブ検出: 窓越し画素差分のしきい値
GRADUAL_HIST = 0.985           # ディゾルブ検出: 窓越しヒストグラム相関のしきい値
                               # (同一店内同士の転換は相関が高く出るため緩めに設定。
                               #  相関≒1.0の純粋な手ブレ・小動作だけを除外する)


def fmt_time(sec: float) -> str:
    """秒 → MM:SS.mmm 表記"""
    m = int(sec // 60)
    s = sec - m * 60
    return f"{m:02d}:{s:06.3f}"


def run_ffmpeg(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run([FFMPEG, *args], capture_output=True, text=True)


def probe_metadata(video: Path) -> dict:
    """ffmpeg -i の出力と OpenCV から動画メタデータを取得する。

    imageio-ffmpeg には ffprobe が同梱されないため、
    ffmpeg の stderr を解析して音声情報を取り出す。
    """
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"動画を開けませんでした: {video}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    info = run_ffmpeg(["-hide_banner", "-i", str(video)]).stderr

    duration = frame_count / fps if fps > 0 else 0.0
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", info)
    if m:
        h, mi, s = int(m.group(1)), int(m.group(2)), float(m.group(3))
        duration = h * 3600 + mi * 60 + s

    audio = {"present": False}
    m = re.search(r"Stream #\d+:\d+.*?Audio:\s*([^\n]+)", info)
    if m:
        line = m.group(1).strip()
        audio = {"present": True, "raw": line}
        cm = re.match(r"([a-zA-Z0-9_]+)", line)
        if cm:
            audio["codec"] = cm.group(1)
        sm = re.search(r"(\d+)\s*Hz", line)
        if sm:
            audio["sample_rate"] = int(sm.group(1))
        chm = re.search(r"Hz,\s*([^,]+)", line)
        if chm:
            audio["channels"] = chm.group(1).strip()
        bm = re.search(r"(\d+)\s*kb/s", line)
        if bm:
            audio["bitrate_kbps"] = int(bm.group(1))

    vm = re.search(r"Stream #\d+:\d+.*?Video:\s*([^\n]+)", info)

    try:
        file_label = str(video.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        file_label = str(video)

    return {
        "file": file_label,
        "analyzed_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "duration_sec": round(duration, 3),
        "duration_str": fmt_time(duration),
        "width": width,
        "height": height,
        "fps": round(fps, 3),
        "frame_count": frame_count,
        "video_stream": vm.group(1).strip() if vm else None,
        "audio": audio,
    }


def scan_video(video: Path, fps: float, interval_sec: float) -> dict:
    """1パスで全フレームを走査し、フレームごとの統計と間隔フレーム画像を収集する。"""
    cap = cv2.VideoCapture(str(video))
    interval_dir = FRAMES_DIR / "interval"
    interval_dir.mkdir(parents=True, exist_ok=True)

    luma, sat, diff, hist_corr = [], [], [], []
    win_diff, win_hist_corr = [], []  # 約0.75秒離れたフレームとの比較(ディゾルブ検出用)
    flow_samples = []  # (prev_frame_index, frame_index, motion_mag, divergence)
    win_frames = int(max(2, round(fps * GRADUAL_WINDOW_SEC)))
    gray_buf: deque = deque(maxlen=win_frames)
    hist_buf: deque = deque(maxlen=win_frames)

    prev_gray = None
    prev_hist = None
    prev_flow_gray = None
    prev_flow_idx = -10**9
    flow_step = max(1, int(round(fps / 4)))  # 約4回/秒でオプティカルフロー
    next_capture = 0.0
    idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = idx / fps
        h, w = frame.shape[:2]
        scale = ANALYSIS_WIDTH / w
        small = cv2.resize(frame, (ANALYSIS_WIDTH, max(2, int(h * scale))))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

        luma.append(float(gray.mean()))
        sat.append(float(hsv[:, :, 1].mean()))

        # 明度(V)も含めることで、モノクロ⇔黒背景のような
        # 彩度が同じシーン間の切り替えも検出できるようにする
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [16, 16, 8],
                            [0, 180, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        if prev_gray is None:
            diff.append(0.0)
            hist_corr.append(1.0)
        else:
            diff.append(float(cv2.absdiff(gray, prev_gray).mean()))
            hist_corr.append(float(cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)))
        prev_gray, prev_hist = gray, hist

        if len(gray_buf) == win_frames:
            old_gray, old_hist = gray_buf[0], hist_buf[0]
            win_diff.append(float(cv2.absdiff(gray, old_gray).mean()))
            win_hist_corr.append(float(cv2.compareHist(old_hist, hist, cv2.HISTCMP_CORREL)))
        else:
            win_diff.append(0.0)
            win_hist_corr.append(1.0)
        gray_buf.append(gray)
        hist_buf.append(hist)

        if idx - prev_flow_idx >= flow_step:
            if prev_flow_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_flow_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                mag = float(np.linalg.norm(flow, axis=2).mean())
                # 放射成分: 中心から外向きの流れが正 → ズームイン
                gh, gw = gray.shape
                ys, xs = np.mgrid[0:gh, 0:gw].astype(np.float32)
                rx, ry = xs - gw / 2, ys - gh / 2
                rn = np.sqrt(rx**2 + ry**2) + 1e-6
                div = float(((flow[:, :, 0] * rx + flow[:, :, 1] * ry) / rn).mean())
                flow_samples.append((prev_flow_idx, idx, mag, div))
            prev_flow_gray = gray
            prev_flow_idx = idx

        if t >= next_capture:
            out = interval_dir / f"t{t:08.2f}s.jpg"
            cv2.imwrite(str(out), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            next_capture += interval_sec

        idx += 1

    cap.release()
    return {
        "n_frames": idx,
        "luma": np.array(luma),
        "sat": np.array(sat),
        "diff": np.array(diff),
        "hist_corr": np.array(hist_corr),
        "win_diff": np.array(win_diff),
        "win_hist_corr": np.array(win_hist_corr),
        "win_frames": win_frames,
        "flow": flow_samples,
    }


def detect_cuts(stats: dict, fps: float) -> list[int]:
    """フレーム差分とヒストグラム相関からカット位置(フレーム番号)を検出する。"""
    d, hc = stats["diff"], stats["hist_corr"]
    n = len(d)
    min_gap = max(1, int(MIN_SCENE_SEC * fps))
    cuts = []
    for i in range(1, n):
        # 通常: 画素差分とヒストグラム変化の両方 / 例外: 画素差分が特に大きい場合は単独で判定
        is_cut = (d[i] > CUT_DIFF_THRESHOLD and hc[i] < CUT_HIST_THRESHOLD) \
            or d[i] > CUT_DIFF_THRESHOLD * 1.8
        if is_cut and (not cuts or i - cuts[-1] >= min_gap):
            cuts.append(i)
    return cuts


def detect_gradual(stats: dict, fps: float, hard_cuts: list[int]) -> list[int]:
    """ディゾルブ(クロスフェード)などの緩やかな転換を検出する。

    約 GRADUAL_WINDOW_SEC 離れたフレーム同士の差分が大きく、かつ
    ヒストグラムも変化している区間の極大点を転換位置とみなす。
    瞬間カットの近傍は除外する。
    """
    wd, whc = stats["win_diff"], stats["win_hist_corr"]
    win = stats["win_frames"]
    n = len(wd)
    min_gap = max(1, int(MIN_SCENE_SEC * fps))
    guard = max(min_gap, win)  # 既存カットからこのフレーム数以内は無視

    candidates = []
    for i in range(1, n - 1):
        if wd[i] > GRADUAL_DIFF and whc[i] < GRADUAL_HIST \
                and wd[i] >= wd[i - 1] and wd[i] >= wd[i + 1]:
            # win_diff[i] は「フレーム i-win と i の比較」なので中間点を境界とする
            boundary = max(0, i - win // 2)
            if all(abs(boundary - c) > guard for c in hard_cuts):
                candidates.append((boundary, wd[i]))

    # 近接候補は差分が最大のものだけ残す
    candidates.sort()
    merged: list[tuple[int, float]] = []
    for b, score in candidates:
        if merged and b - merged[-1][0] <= guard:
            if score > merged[-1][1]:
                merged[-1] = (b, score)
        else:
            merged.append((b, score))
    return [b for b, _ in merged]


def detect_flashes(stats: dict, fps: float) -> list[dict]:
    """一瞬だけ輝度が跳ね上がるフレーム(フラッシュ演出候補)を検出する。"""
    luma = stats["luma"]
    flashes = []
    win = max(1, int(round(fps * 0.12)))  # 前後 約0.12秒 と比較
    for i in range(win, len(luma) - win):
        before, after = luma[i - win], luma[i + win]
        if luma[i] - before > FLASH_LUMA_JUMP and luma[i] - after > FLASH_LUMA_JUMP:
            if not flashes or i - flashes[-1]["frame"] > win * 2:
                flashes.append({"frame": i, "time": round(i / fps, 3),
                                "time_str": fmt_time(i / fps)})
    return flashes


def build_scenes(boundaries: list[tuple[int, str]], stats: dict,
                 fps: float, duration: float) -> list[dict]:
    """境界位置(フレーム, 種別)からシーン一覧を作り、演出候補を付与する。"""
    n = stats["n_frames"]
    boundaries = sorted(boundaries)
    bounds = [(0, "先頭")] + boundaries + [(n, "終端")]
    flow = stats["flow"]
    scenes = []
    for si in range(len(bounds) - 1):
        (f0, btype), (f1, _) = bounds[si], bounds[si + 1]
        start, end = f0 / fps, min(f1 / fps, duration)
        seg_luma = stats["luma"][f0:f1]
        seg_sat = stats["sat"][f0:f1]
        # 比較した2フレームが両方ともシーン内にあるサンプルだけを使う
        # (カット境界をまたぐフローはゴミ値になるため除外)
        seg_flow = [(m, dv) for (pfi, fi, m, dv) in flow if f0 <= pfi and fi < f1]
        motion = float(np.mean([m for m, _ in seg_flow])) if seg_flow else 0.0
        diverg = float(np.mean([dv for _, dv in seg_flow])) if seg_flow else 0.0

        effects = []
        if float(seg_luma.mean()) < BLACK_LUMA:
            effects.append("黒背景")
        elif float(seg_sat.mean()) < MONO_SAT:
            effects.append("モノクロ")
        if diverg > ZOOM_DIVERGENCE:
            effects.append("ズームイン候補")
        elif diverg < -ZOOM_DIVERGENCE:
            effects.append("ズームアウト候補")
        is_static = motion < STATIC_MOTION and (end - start) >= 1.5
        if is_static and "黒背景" not in effects:
            effects.append("静止(商品画像表示候補)")

        scenes.append({
            "index": si + 1,
            "start_boundary": btype,
            "start_frame": f0,
            "end_frame": f1,
            "start_sec": round(start, 3),
            "end_sec": round(end, 3),
            "duration_sec": round(end - start, 3),
            "start_str": fmt_time(start),
            "end_str": fmt_time(end),
            "mean_luma": round(float(seg_luma.mean()), 1),
            "mean_saturation": round(float(seg_sat.mean()), 1),
            "motion_level": round(motion, 3),
            "zoom_divergence": round(diverg, 4),
            "effect_candidates": effects,
            "product_image_candidate": is_static,
        })
    return scenes


def annotate_boundary_confidence(video: Path, scenes: list[dict], fps: float) -> None:
    """ディゾルブ候補の境界について、前後0.6秒のフレーム類似度を測り、
    ほぼ同一構図なら「要確認」(誤検出の可能性)とマークする。"""
    cap = cv2.VideoCapture(str(video))

    def small_hist(frame):
        small = cv2.resize(frame, (ANALYSIS_WIDTH, int(frame.shape[0] * ANALYSIS_WIDTH / frame.shape[1])))
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [16, 16, 8], [0, 180, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        return hist

    for sc in scenes:
        if sc["start_boundary"] != "ディゾルブ候補":
            continue
        b = sc["start_sec"]
        hists = []
        for dt in (-0.6, 0.6):
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int((b + dt) * fps)))
            ok, frame = cap.read()
            if ok:
                hists.append(small_hist(frame))
        if len(hists) == 2:
            corr = float(cv2.compareHist(hists[0], hists[1], cv2.HISTCMP_CORREL))
            if corr > 0.93:
                sc["start_boundary"] = "ディゾルブ候補(要確認)"
            sc["boundary_similarity"] = round(corr, 3)
    cap.release()


def extract_scene_frames(video: Path, scenes: list[dict], fps: float) -> None:
    """各シーンの開始・中間・終了フレームと、テロップ確認用の下部クロップを保存する。"""
    scene_dir = FRAMES_DIR / "scenes"
    telop_dir = FRAMES_DIR / "telop"
    scene_dir.mkdir(parents=True, exist_ok=True)
    telop_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video))
    for sc in scenes:
        f0, f1 = sc["start_frame"], sc["end_frame"]
        targets = {"start": f0, "mid": (f0 + f1) // 2, "end": max(f0, f1 - 2)}
        for label, fi in targets.items():
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if not ok:
                continue
            name = f"scene{sc['index']:03d}_{label}_{fmt_time(fi / fps).replace(':', 'm')}s.jpg"
            cv2.imwrite(str(scene_dir / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if label == "mid":
                h = frame.shape[0]
                crop = frame[int(h * 0.62):, :]  # テロップが出やすい下部38%
                cv2.imwrite(str(telop_dir / f"scene{sc['index']:03d}_bottom.jpg"),
                            crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
        sc["frames"] = {
            "start": f"analysis/reference_frames/scenes/scene{sc['index']:03d}_start_*.jpg",
            "mid": f"analysis/reference_frames/scenes/scene{sc['index']:03d}_mid_*.jpg",
            "telop_crop": f"analysis/reference_frames/telop/scene{sc['index']:03d}_bottom.jpg",
        }
    cap.release()


def render_waveform(video: Path, duration: float, has_audio: bool) -> bool:
    """音声をモノラルWAVに書き出し、波形PNGを生成する。"""
    if not has_audio:
        return False
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with tempfile.TemporaryDirectory() as td:
        wav_path = Path(td) / "audio.wav"
        r = run_ffmpeg(["-hide_banner", "-y", "-i", str(video),
                        "-vn", "-ac", "1", "-ar", "8000",
                        "-acodec", "pcm_s16le", str(wav_path)])
        if r.returncode != 0 or not wav_path.exists():
            print("  [警告] 音声抽出に失敗しました:", r.stderr.splitlines()[-1] if r.stderr else "")
            return False
        with wave.open(str(wav_path), "rb") as wf:
            sr = wf.getframerate()
            data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)

    t = np.arange(len(data)) / sr
    fig, ax = plt.subplots(figsize=(max(10, duration * 0.35), 3.2), dpi=110)
    ax.plot(t, data / 32768.0, linewidth=0.3, color="#2563eb")
    ax.set_xlim(0, duration)
    ax.set_ylim(-1, 1)
    ax.set_xlabel("time (sec)")
    ax.set_title("reference.mp4 audio waveform")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(ANALYSIS_DIR / "audio_waveform.png")
    plt.close(fig)
    return True


def write_outputs(meta: dict, scenes: list[dict], flashes: list[dict],
                  interval_sec: float, waveform_ok: bool) -> None:
    (ANALYSIS_DIR / "reference_metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    (ANALYSIS_DIR / "scene_list.json").write_text(
        json.dumps({"scene_count": len(scenes), "scenes": scenes},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    plan = {
        "source": meta["file"],
        "generated_at": meta["analyzed_at"],
        "note": "自動解析による下書きです。素材25本との照合前の段階であり、"
                "演出はすべて『候補』です。人の確認・文章での修正指示を前提とします。",
        "video": {k: meta[k] for k in ("duration_sec", "width", "height", "fps")},
        "timeline": [
            {
                "cut": sc["index"],
                "start": sc["start_str"],
                "end": sc["end_str"],
                "duration_sec": sc["duration_sec"],
                "transition_in": sc["start_boundary"],
                "effect_candidates": sc["effect_candidates"],
                "product_image_candidate": sc["product_image_candidate"],
                "material": None,   # 素材照合フェーズで割り当て予定
                "notes": "",
            }
            for sc in scenes
        ],
        "events": {"flash_candidates": flashes},
        "next_phase": ["素材動画25本との照合", "商品画像3枚の割り当て", "試作動画の生成"],
    }
    (ANALYSIS_DIR / "editing_plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")

    # 編集構成表 CSV
    with open(ANALYSIS_DIR / "editing_sheet.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["カット", "開始", "終了", "長さ(秒)", "入り方", "演出候補",
                    "商品画像候補", "モーション量", "平均輝度", "平均彩度"])
        for sc in scenes:
            w.writerow([sc["index"], sc["start_str"], sc["end_str"], sc["duration_sec"],
                        sc["start_boundary"],
                        " / ".join(sc["effect_candidates"]) or "-",
                        "◯" if sc["product_image_candidate"] else "",
                        sc["motion_level"], sc["mean_luma"], sc["mean_saturation"]])

    # レポート Markdown
    a = meta["audio"]
    audio_line = (f"{a.get('codec', '?')} / {a.get('sample_rate', '?')} Hz / "
                  f"{a.get('channels', '?')} / {a.get('bitrate_kbps', '?')} kbps"
                  if a.get("present") else "音声なし")
    lines = [
        "# 見本動画 解析レポート",
        "",
        f"- 解析日時: {meta['analyzed_at']}",
        f"- ファイル: `{meta['file']}`",
        "",
        "## 基本情報",
        "",
        "| 項目 | 値 |",
        "|---|---|",
        f"| 動画尺 | {meta['duration_str']}({meta['duration_sec']} 秒) |",
        f"| 解像度 | {meta['width']} × {meta['height']} |",
        f"| フレームレート | {meta['fps']} fps |",
        f"| 総フレーム数 | {meta['frame_count']} |",
        f"| 音声 | {audio_line} |",
        f"| カット数(検出) | {len(scenes)} |",
        f"| フラッシュ候補 | {len(flashes)} 箇所 |",
        "",
        "## 編集構成表",
        "",
        "演出はすべて自動検出による**候補**です。各カットの代表フレームは "
        "`analysis/reference_frames/scenes/` を確認してください。",
        "",
        "| カット | 開始 | 終了 | 長さ(秒) | 入り方 | 演出候補 | 商品画像候補 |",
        "|---:|---|---|---:|---|---|:---:|",
    ]
    for sc in scenes:
        lines.append(
            f"| {sc['index']} | {sc['start_str']} | {sc['end_str']} | "
            f"{sc['duration_sec']} | {sc['start_boundary']} | "
            f"{' / '.join(sc['effect_candidates']) or '-'} | "
            f"{'◯' if sc['product_image_candidate'] else ''} |")
    lines += [
        "",
        "## フラッシュ演出候補",
        "",
    ]
    if flashes:
        lines += ["| 時刻 | フレーム |", "|---|---:|"]
        lines += [f"| {fl['time_str']} | {fl['frame']} |" for fl in flashes]
    else:
        lines.append("検出されませんでした。")
    lines += [
        "",
        "## 生成物",
        "",
        "| ファイル | 内容 |",
        "|---|---|",
        "| `analysis/reference_metadata.json` | 動画メタデータ |",
        "| `analysis/scene_list.json` | シーン一覧(数値詳細つき) |",
        "| `analysis/editing_plan.json` | 編集プラン下書き |",
        "| `analysis/editing_sheet.csv` | 編集構成表(Excel等で開けます) |",
        f"| `analysis/reference_frames/interval/` | {interval_sec} 秒間隔の確認用フレーム |",
        "| `analysis/reference_frames/scenes/` | 各カットの開始・中間・終了フレーム |",
        "| `analysis/reference_frames/telop/` | テロップ確認用(各カット中間の画面下部) |",
        f"| `analysis/audio_waveform.png` | 音声波形{'' if waveform_ok else '(音声なしのため未生成)'} |",
        "",
        "## 次のフェーズ(未実施)",
        "",
        "1. 素材動画25本との照合(各カットへの素材割り当て)",
        "2. 商品画像3枚の表示タイミング確定",
        "3. 試作動画の生成 → 文章での修正指示 → 完成動画",
        "",
    ]
    (ANALYSIS_DIR / "analysis_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    global CUT_DIFF_THRESHOLD, GRADUAL_DIFF
    ap = argparse.ArgumentParser(description="見本動画の解析")
    ap.add_argument("--input", default=str(PROJECT_ROOT / "reference" / "reference.mp4"),
                    help="見本動画のパス(既定: reference/reference.mp4)")
    ap.add_argument("--interval", type=float, default=1.0,
                    help="確認用フレームの抽出間隔・秒(既定: 1.0)")
    ap.add_argument("--cut-threshold", type=float, default=CUT_DIFF_THRESHOLD,
                    help="カット検出のフレーム差分しきい値")
    ap.add_argument("--gradual-threshold", type=float, default=GRADUAL_DIFF,
                    help="ディゾルブ検出の窓越し差分しきい値(下げると敏感になる)")
    args = ap.parse_args()
    CUT_DIFF_THRESHOLD = args.cut_threshold
    GRADUAL_DIFF = args.gradual_threshold

    video = Path(args.input)
    if not video.exists():
        print("見本動画が見つかりません。")
        print(f"  期待するパス: {video}")
        print("  reference フォルダに動画を入れて reference.mp4 という名前にしてください。")
        return 1

    ANALYSIS_DIR.mkdir(exist_ok=True)
    FRAMES_DIR.mkdir(exist_ok=True)

    print(f"[1/6] メタデータ取得中: {video.name}")
    meta = probe_metadata(video)
    print(f"      {meta['duration_str']} / {meta['width']}x{meta['height']} / {meta['fps']}fps")

    print("[2/6] 全フレーム走査中(統計収集+間隔フレーム抽出)…")
    stats = scan_video(video, meta["fps"], args.interval)

    print("[3/6] シーン切り替え・演出候補を検出中…")
    cuts = detect_cuts(stats, meta["fps"])
    gradual = detect_gradual(stats, meta["fps"], cuts)
    flashes = detect_flashes(stats, meta["fps"])
    boundaries = [(c, "カット") for c in cuts] + [(g, "ディゾルブ候補") for g in gradual]
    scenes = build_scenes(boundaries, stats, meta["fps"], meta["duration_sec"])
    print(f"      シーン数: {len(scenes)}(瞬間カット {len(cuts)} / "
          f"ディゾルブ候補 {len(gradual)})/ フラッシュ候補: {len(flashes)}")

    print("[4/6] シーン代表フレーム・テロップ確認用画像を抽出中…")
    annotate_boundary_confidence(video, scenes, meta["fps"])
    extract_scene_frames(video, scenes, meta["fps"])

    print("[5/6] 音声波形を生成中…")
    waveform_ok = render_waveform(video, meta["duration_sec"], meta["audio"]["present"])

    print("[6/6] JSON・レポートを書き出し中…")
    write_outputs(meta, scenes, flashes, args.interval, waveform_ok)

    print("\n完了しました。まず analysis/analysis_report.md を確認してください。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
