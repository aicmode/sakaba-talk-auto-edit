# sakaba-talk-auto-edit

見本動画を基準に、素材動画25本・商品画像3枚・黒背景を使って、
Adobe Premiere を使わず Python + FFmpeg で自動編集を行うプロジェクトです。

現在は **フェーズ1:見本動画の解析環境** まで構築済みです。
素材との照合・完成動画の生成はまだ行いません。

## フォルダ構成

| フォルダ | 用途 |
|---|---|
| `reference/` | 見本動画を入れる(ファイル名は `reference.mp4`) |
| `videos/` | 素材動画25本(後のフェーズで使用) |
| `images/` | 商品画像3枚(後のフェーズで使用) |
| `assets/` | 黒背景・BGM などその他素材 |
| `output/` | 試作動画・完成動画の出力先 |
| `scripts/` | 解析・編集スクリプト |
| `analysis/` | 解析結果の出力先 |

## セットアップ(構築済み)

Python 仮想環境 `.venv` に必要なライブラリが導入済みです。
FFmpeg は `imageio-ffmpeg` に同梱されたバイナリを使うため、
Homebrew や管理者権限は不要です。

別のPCなどで再構築する場合:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

環境の確認:

```bash
.venv/bin/python scripts/check_env.py
```

## 見本動画の解析手順

1. 見本動画を `reference/` フォルダに入れる
2. ファイル名を `reference.mp4` に変更する
3. 解析を実行する

```bash
.venv/bin/python scripts/analyze_reference.py
```

オプション:

```bash
# 確認用フレームの抽出間隔を0.5秒にする
.venv/bin/python scripts/analyze_reference.py --interval 0.5

# カット検出の感度を調整する(値を下げると細かく分割される)
.venv/bin/python scripts/analyze_reference.py --cut-threshold 22
```

## 解析結果の見方

| ファイル | 内容 |
|---|---|
| `analysis/analysis_report.md` | **まずこれを見る。** 基本情報+人が読む編集構成表 |
| `analysis/editing_sheet.csv` | 編集構成表(Excel・Numbersで開けます) |
| `analysis/reference_metadata.json` | 動画尺・解像度・fps・音声情報 |
| `analysis/scene_list.json` | シーン切り替え位置と各カットの詳細数値 |
| `analysis/editing_plan.json` | 編集プラン下書き(演出候補・商品画像タイミング候補) |
| `analysis/reference_frames/interval/` | 一定間隔(既定1秒)の確認用フレーム |
| `analysis/reference_frames/scenes/` | 各カットの開始・中間・終了フレーム |
| `analysis/reference_frames/telop/` | テロップ確認用(各カット中間の画面下部クロップ) |
| `analysis/audio_waveform.png` | 音声波形 |

検出される演出候補: ズームイン/アウト、モノクロ、フラッシュ、黒背景、
静止(商品画像表示候補)。いずれも自動検出の**候補**であり、
レポートを見ながら文章で修正を指示する前提です。

## 今後のフェーズ(未実施)

1. `videos/` の素材25本と各カットの照合・割り当て
2. `images/` の商品画像3枚の表示タイミング確定
3. 試作動画の生成(`output/`)→ 文章で修正指示 → 完成動画
