# エンコーダモデルファインチューニング

[English](README.en.md)

エンコーダのみモデル(日本語 BERT 系)を **多クラス感情分類**にファインチューニングするためのサンプル実装です。

- 主役モデル: [`sbintuitions/modernbert-ja-130m`](https://huggingface.co/sbintuitions/modernbert-ja-130m)（日本語 ModernBERT, 132M）
- データ: [WRIME ver2](https://github.com/ids-cv/wrime)（日本語感情、Plutchik の8感情 + 中立 = **9クラス**）
- 実測ベースライン: **test macro-F1 ≈ 0.465 / accuracy ≈ 0.629**（3エポック・約146秒 / RTX 5090）

---

## 必要環境

| 項目 | 内容 |
|------|------|
| OS | Windows 11（本リポジトリの検証環境）|
| Python | 3.10 |
| パッケージ管理 | [uv](https://docs.astral.sh/uv/) |
| GPU | NVIDIA GPU 推奨。RTX 5090(Blackwell / sm_120)で検証済み |

CUDA 版 PyTorch は **cu128 ビルド**を使います（Blackwell 対応）。`pyproject.toml` で torch のみ
PyTorch 公式インデックスへ向けているので、`uv sync` するだけで入ります。CPU でも動きますが学習は遅くなります。

---

## クイックスタート

```powershell
# 1. 依存をインストール(専用 .venv に cu128 torch などが入る。初回は数GB DL)
uv sync

# 2. GPU を確認(sm_120 のカーネルが入っているか等)
uv run python scripts/00_check_env.py

# 3. データ準備(WRIME を取得し 9クラスに整形して保存)
uv run python scripts/01_prepare_data.py

# 4. ファインチューニング(Trainer 版。約150秒 / RTX 5090)
uv run python scripts/02_train.py
#   生ループ版で中身を見るなら:  uv run python scripts/02_train.py --raw
#   数十件のスモークだけ:        uv run python scripts/02_train.py --tiny

# 5. test で評価(macro-F1・クラス別 F1・混同行列 → outputs/confusion_matrix.png)
uv run python scripts/03_evaluate.py

# 6. 推論(任意の文を渡せる)
uv run python scripts/04_infer.py "今日は最高の一日だった！"

# 7. (任意) モバイル配備用に ONNX へ書き出す(Unity Sentis 等)
uv sync --extra onnx
uv run python scripts/05_export_onnx.py            # model.onnx / model.sentis.onnx(int32) / model.int8.onnx
uv run python scripts/06_sentis_test_vector.py "今日は最高だ"   # Sentis検証用の id列と期待logitsを出力
```

> **Windows のコンソール文字化け対策**: 各 script は標準出力を UTF-8 に張り替えます。
> それでも化ける場合は `$env:PYTHONUTF8=1` を付けて実行してください。

---

## テスト

```powershell
# 高速な単体テスト(ネットワーク不要)
uv run python -m pytest -q

# 実データ+モデルを使う統合テスト(全経路を数十件で通す)
$env:ENC_FT_INTEGRATION=1; uv run python -m pytest -q -s
```

---

## カスタマイズ

主要な設定は `src/enc_ft/config.py` の `Config` に集約しています。CLI からも上書き可能です。

```powershell
uv run python scripts/02_train.py --epochs 5 --lr 1e-5 --batch-size 16
uv run python scripts/01_prepare_data.py --perspective writer --min-intensity 2
```

- **自分のデータで学習する**: `src/enc_ft/data.py` の `build_datasets()` を、
  `text` と `label`（0〜8）と `split` を持つ自作 CSV の読み込みに差し替えるだけ。
  トークナイズ・学習・評価・推論は無変更で動きます（詳細は
  [learning/07-inference-ops.html](learning/07-inference-ops.html)）。
- **classic BERT(tohoku)を試す**: `uv sync --extra tohoku` 後に
  `--model tohoku-nlp/bert-base-japanese-v3`。

---

## この環境で踏みやすい罠（と対策）

| 症状 | 対策 |
|------|------|
| `cuda.is_available()==False` | CPU版 torch が残っている。`.venv` を消して `uv sync` |
| GPU で `no kernel image` | cu126 以前を掴んでいる。cu128 を使う（本 `pyproject.toml` は設定済み）|
| 日本語 print で `UnicodeEncodeError` | `PYTHONUTF8=1` を付ける |
| `datasets` でスクリプト方式が失敗 | 本プロジェクトは原典 TSV を直 DL するので該当しない |

---

## ライセンス / 出典

- WRIME: [ids-cv/wrime](https://github.com/ids-cv/wrime)（利用時は原典のライセンス・規約に従うこと）
- モデル: `sbintuitions/modernbert-ja-130m`（MIT）
