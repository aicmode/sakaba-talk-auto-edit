#!/usr/bin/env python3
"""素材動画(videos/*.mov)の一括解析スクリプト。

出力:
  analysis/material_metadata.json      … 各素材のメタデータ+シーン統計+文字起こし
  analysis/material_frames/<素材名>/   … 一定間隔の確認フレーム+コンタクトシート
  analysis/material_audio/<素材名>.wav … 音声確認用データ(16kHz mono)
  analysis/material_audio/<素材名>_waveform.png … 波形

使い方:
  .venv/bin/python scripts/analyze_materials.py            # 全素材
  .venv/bin/python scripts/analyze_materials.py --only 3   # 課題素材3 のみ
"""

import argparse
import json
import re
import subprocess
import sys
import wave
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import imageio_ffmpeg

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANALYSIS_DIR = PROJECT_ROOT / "analysis"
FRAMES_DIR = ANALYSIS_DIR / "material_frames"
AUDIO_DIR = ANALYSIS_DIR / "material_audio"
VIDEOS_DIR = PROJECT_ROOT / "videos"

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

ANALYSIS_WIDTH = 160
CUT_DIFF_THRESHOLD = 28.0
CUT_HIST_THRESHOLD = 0.55
MIN_SCENE_SEC = 0.30

# whisper がよく無音区間で幻聴するフレーズ(除外用)
HALLUCINATION_PATTERNS = [
    "ご視聴ありがとうございました", "チャンネル登録", "ご覧いただき",
    "おやすみなさい", "字幕", "最後までご視聴",
]


def fmt_time(sec: float) -> str:
    m = int(sec // 60)
    s = sec - m * 60
    return f"{m:02d}:{s:06.3f}"


def run_ffmpeg(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run([FFMPEG, *args], capture_output=True, text=True)


def probe_metadata(video: Path) -> dict:
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

    vm = re.search(r"Stream #\d+:\d+.*?Video:\s*([^\n]+)", info)
    # QuickTime の回転メタデータ(displaymatrix)を確認
    rot = re.search(r"rotation of (-?\d+(?:\.\d+)?) degrees", info)

    return {
        "file": f"videos/{video.name}",
        "duration_sec": round(duration, 3),
        "duration_str": fmt_time(duration),
        "width": width,
        "height": height,
        "fps": round(fps, 3),
        "frame_count": frame_count,
        "rotation_deg": float(rot.group(1)) if rot else 0.0,
        "video_stream": vm.group(1).strip() if vm else None,
        "audio": audio,
    }


def scan_video(video: Path, name: str, fps: float, interval: float) -> dict:
    """全フレーム走査: 統計収集+間隔フレーム保存+カット検出。"""
    out_dir = FRAMES_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))

    luma, sat, diff, hist_corr = [], [], [], []
    prev_gray = None
    prev_hist = None
    next_capture = 0.0
    saved = []  # (time, path)
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

        if t >= next_capture:
            p = out_dir / f"t{t:07.2f}s.jpg"
            # 確認用は幅540に縮小(容量節約)
            fh, fw = frame.shape[:2]
            sc = 540 / fw
            view = cv2.resize(frame, (540, int(fh * sc)))
            cv2.imwrite(str(p), view, [cv2.IMWRITE_JPEG_QUALITY, 82])
            saved.append((t, p))
            next_capture += interval
        idx += 1
    cap.release()

    d = np.array(diff)
    hc = np.array(hist_corr)
    min_gap = max(1, int(MIN_SCENE_SEC * fps))
    cuts = []
    for i in range(1, len(d)):
        is_cut = (d[i] > CUT_DIFF_THRESHOLD and hc[i] < CUT_HIST_THRESHOLD) \
            or d[i] > CUT_DIFF_THRESHOLD * 1.8
        if is_cut and (not cuts or i - cuts[-1] >= min_gap):
            cuts.append(i)

    return {
        "n_frames": idx,
        "mean_luma": round(float(np.mean(luma)), 1) if luma else 0.0,
        "mean_saturation": round(float(np.mean(sat)), 1) if sat else 0.0,
        "internal_cuts": [round(c / fps, 3) for c in cuts],
        "saved_frames": saved,
    }


def make_contact_sheet(name: str, saved: list, cols: int = 8) -> str | None:
    """保存済み間隔フレームからコンタクトシート(時刻ラベル付き)を作る。"""
    if not saved:
        return None
    tiles = []
    for t, p in saved:
        img = cv2.imread(str(p))
        if img is None:
            continue
        tile = cv2.resize(img, (200, int(img.shape[0] * 200 / img.shape[1])))
        cv2.rectangle(tile, (0, 0), (92, 26), (0, 0, 0), -1)
        cv2.putText(tile, f"{t:6.1f}s", (4, 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)
    if not tiles:
        return None
    th = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, th - t.shape[0], 0, 0,
                                cv2.BORDER_CONSTANT, value=(20, 20, 20)) for t in tiles]
    rows = []
    for i in range(0, len(tiles), cols):
        row = tiles[i:i + cols]
        while len(row) < cols:
            row.append(np.full_like(tiles[0], 20))
        rows.append(cv2.hconcat(row))
    sheet = cv2.vconcat(rows)
    out = FRAMES_DIR / name / "_contact_sheet.jpg"
    cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return str(out.relative_to(PROJECT_ROOT))


def extract_audio(video: Path, name: str, duration: float, has_audio: bool) -> dict:
    """16kHz mono WAV と波形PNG、発話区間(RMSベース)を出力する。"""
    if not has_audio:
        return {"wav": None, "speech_segments": []}
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    wav_path = AUDIO_DIR / f"{name}.wav"
    r = run_ffmpeg(["-hide_banner", "-y", "-i", str(video), "-vn",
                    "-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le", str(wav_path)])
    if r.returncode != 0 or not wav_path.exists():
        return {"wav": None, "speech_segments": [], "error": "extract failed"}

    with wave.open(str(wav_path), "rb") as wf:
        sr = wf.getframerate()
        data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
    x = data.astype(np.float32) / 32768.0

    # 50msごとの RMS → 発話区間検出
    hop = int(sr * 0.05)
    n = len(x) // hop
    rms = np.array([float(np.sqrt(np.mean(x[i * hop:(i + 1) * hop] ** 2)))
                    for i in range(n)])
    thr = max(0.015, float(np.percentile(rms, 60)) * 0.8)
    active = rms > thr
    segs = []
    start = None
    for i, a in enumerate(active):
        if a and start is None:
            start = i
        elif not a and start is not None:
            if (i - start) * 0.05 >= 0.25:
                segs.append([round(start * 0.05, 2), round(i * 0.05, 2)])
            start = None
    if start is not None:
        segs.append([round(start * 0.05, 2), round(n * 0.05, 2)])
    # 0.4秒以内の隙間は結合
    merged = []
    for s in segs:
        if merged and s[0] - merged[-1][1] <= 0.4:
            merged[-1][1] = s[1]
        else:
            merged.append(s)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.arange(len(x)) / sr
    fig, ax = plt.subplots(figsize=(max(8, duration * 0.5), 2.4), dpi=100)
    ax.plot(t[::4], x[::4], linewidth=0.3, color="#2563eb")
    for s0, s1 in merged:
        ax.axvspan(s0, s1, color="#f59e0b", alpha=0.2)
    ax.set_xlim(0, duration)
    ax.set_ylim(-1, 1)
    ax.set_title(name)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(AUDIO_DIR / f"{name}_waveform.png")
    plt.close(fig)

    return {"wav": f"analysis/material_audio/{name}.wav",
            "waveform": f"analysis/material_audio/{name}_waveform.png",
            "speech_segments": merged,
            "audio_array": x}


def transcribe(x: np.ndarray | None) -> list[dict]:
    if x is None or len(x) < 1600:
        return []
    import mlx_whisper
    res = mlx_whisper.transcribe(
        x, path_or_hf_repo="mlx-community/whisper-large-v3-turbo",
        language="ja", condition_on_previous_text=False,
        no_speech_threshold=0.5, logprob_threshold=-1.2, verbose=None)
    out = []
    for s in res["segments"]:
        text = s["text"].strip()
        if not text:
            continue
        suspicious = any(p in text for p in HALLUCINATION_PATTERNS)
        out.append({
            "start": round(float(s["start"]), 2),
            "end": round(float(s["end"]), 2),
            "text": text,
            "no_speech_prob": round(float(s.get("no_speech_prob", 0)), 3),
            "avg_logprob": round(float(s.get("avg_logprob", 0)), 3),
            "suspected_hallucination": bool(
                suspicious or s.get("no_speech_prob", 0) > 0.7),
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=str, default=None,
                    help="カンマ区切りの素材番号(例: 1,3,10)")
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--no-transcribe", action="store_true")
    args = ap.parse_args()

    videos = sorted(VIDEOS_DIR.glob("*.mov"),
                    key=lambda p: int(re.search(r"(\d+)", p.stem).group(1)))
    if args.only:
        wanted = {int(x) for x in args.only.split(",")}
        videos = [v for v in videos
                  if int(re.search(r"(\d+)", v.stem).group(1)) in wanted]

    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    out_json = ANALYSIS_DIR / "material_metadata.json"
    existing = {}
    if out_json.exists():
        try:
            existing = {m["name"]: m for m in
                        json.loads(out_json.read_text(encoding="utf-8"))["materials"]}
        except Exception:
            existing = {}

    for i, v in enumerate(videos, 1):
        name = v.stem
        print(f"[{i}/{len(videos)}] {name}")
        meta = probe_metadata(v)
        print(f"    {meta['duration_str']} / {meta['width']}x{meta['height']}"
              f" / {meta['fps']}fps / rot={meta['rotation_deg']}")
        stats = scan_video(v, name, meta["fps"], args.interval)
        sheet = make_contact_sheet(name, stats.pop("saved_frames"))
        audio_info = extract_audio(v, name, meta["duration_sec"],
                                   meta["audio"]["present"])
        x = audio_info.pop("audio_array", None)
        segments = [] if args.no_transcribe else transcribe(x)
        if segments:
            for s in segments:
                flag = " [幻聴?]" if s["suspected_hallucination"] else ""
                print(f"    {s['start']:6.2f}-{s['end']:6.2f} {s['text']}{flag}")
        rec = {
            "name": name,
            **meta,
            "analyzed_at": datetime.now(timezone.utc).astimezone()
                .isoformat(timespec="seconds"),
            "scan": stats,
            "contact_sheet": sheet,
            "audio_files": {k: audio_info.get(k) for k in ("wav", "waveform")},
            "speech_segments_rms": audio_info.get("speech_segments", []),
            "transcript": segments,
            # 以下は目視確認フェーズで記入
            "content_description": existing.get(name, {}).get("content_description", ""),
            "candidate_reference_scenes": existing.get(name, {}).get(
                "candidate_reference_scenes", []),
        }
        existing[name] = rec
        # 毎素材ごとに保存(途中失敗しても進捗が残るように)
        ordered = sorted(existing.values(),
                         key=lambda m: int(re.search(r"(\d+)", m["name"]).group(1)))
        out_json.write_text(json.dumps(
            {"generated_at": datetime.now(timezone.utc).astimezone()
                .isoformat(timespec="seconds"),
             "material_count": len(ordered), "materials": ordered},
            ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n完了: analysis/material_metadata.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
