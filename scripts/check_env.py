#!/usr/bin/env python3
"""解析環境が整っているかを確認するスクリプト。

使い方:
  .venv/bin/python scripts/check_env.py
"""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REQUIRED_DIRS = ["reference", "videos", "images", "assets", "output", "scripts", "analysis"]


def main() -> int:
    ok = True
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")

    for name in ("imageio_ffmpeg", "cv2", "numpy", "PIL", "moviepy", "matplotlib"):
        try:
            mod = __import__(name)
            ver = getattr(mod, "__version__", "?")
            print(f"  OK  {name} {ver}")
        except ImportError as e:
            print(f"  NG  {name} が読み込めません: {e}")
            ok = False

    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        r = subprocess.run([exe, "-version"], capture_output=True, text=True)
        first = r.stdout.splitlines()[0] if r.stdout else "(出力なし)"
        print(f"  OK  FFmpeg: {first}")
        print(f"      パス: {exe}")
    except Exception as e:
        print(f"  NG  FFmpeg バイナリの実行に失敗: {e}")
        ok = False

    for d in REQUIRED_DIRS:
        p = PROJECT_ROOT / d
        print(f"  {'OK' if p.is_dir() else 'NG'}  フォルダ {d}/")
        ok = ok and p.is_dir()

    ref = PROJECT_ROOT / "reference" / "reference.mp4"
    if ref.exists():
        print(f"  OK  見本動画あり: {ref.relative_to(PROJECT_ROOT)}")
    else:
        print("  --  見本動画は未配置です(reference/reference.mp4)。配置後に解析できます。")

    print("\n環境チェック:", "問題ありません" if ok else "問題があります(上のNGを確認してください)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
