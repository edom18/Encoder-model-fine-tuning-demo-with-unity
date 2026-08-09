"""WRIME ver2 を取得し、9クラス感情分類データセットに整形する。

設計上のポイント(教材の ch2 と対応):
- HuggingFace の ``datasets`` はスクリプト方式(trust_remote_code)を 4.0 で撤廃した。
  そのため原典 TSV を直接ダウンロードし、前処理を自分で書く(隠れた魔法なし)。
- ラベルは「読者(または書き手)の8感情強度の argmax」。全感情が閾値未満なら「中立」。
- 分割は WRIME 公式の ``Train/Dev/Test`` 列に従う(自前分割はリーク源になり再現性も落ちる)。
"""

from __future__ import annotations

import csv
import urllib.request
from collections import Counter

import numpy as np
import pandas as pd
from datasets import ClassLabel, Dataset, DatasetDict

from .config import ID2LABEL, NEUTRAL_ID, NUM_LABELS, Config

# 公式 split 列の値 → datasets の split 名
_SPLIT_MAP = {"train": "train", "dev": "validation", "test": "test"}


# ---------------------------------------------------------------------------
# 1) ダウンロード
# ---------------------------------------------------------------------------
def download_wrime(config: Config, force: bool = False) -> None:
    """WRIME ver2 TSV を config.raw_tsv_path に取得する(既にあれば何もしない)。"""
    dest = config.raw_tsv_path
    if dest.exists() and not force:
        print(f"[data] 既存の TSV を使用: {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[data] ダウンロード中: {config.wrime_url}")
    urllib.request.urlretrieve(config.wrime_url, dest)  # noqa: S310 (信頼できる固定URL)
    print(f"[data] 保存: {dest} ({dest.stat().st_size / 1e6:.1f} MB)")


# ---------------------------------------------------------------------------
# 2) 読み込み
# ---------------------------------------------------------------------------
def load_raw(config: Config) -> pd.DataFrame:
    """TSV を DataFrame として読む。

    tweet にはクォート記号が不規則に混じるため QUOTE_NONE で読む。
    """
    df = pd.read_csv(
        config.raw_tsv_path,
        sep="\t",
        quoting=csv.QUOTE_NONE,
        dtype=str,
        keep_default_na=False,
    )
    df.columns = [c.strip() for c in df.columns]
    return df


# ---------------------------------------------------------------------------
# 3) 9クラスラベルを作る(このプロジェクトの中心的な設計判断)
# ---------------------------------------------------------------------------
def build_labels(df: pd.DataFrame, config: Config) -> np.ndarray:
    """8感情の強度から、行ごとの単一ラベル(0..8)を返す。

    手順:
      1. 視点(reader/writer)に応じた8列の強度を取り出す(0..3)。
      2. 各行の最大強度が min_intensity 未満なら「中立」。
      3. そうでなければ argmax を採用。tie は tie_break 方針で解決。
    """
    cols = config.emotion_columns()
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"TSV に想定列がありません: {missing}")

    # 空セル等を 0 に寄せてから整数化(未アノテーション行への防御)
    inten = (
        df[cols].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(dtype=np.int64)
    )  # (N, 8)
    max_val = inten.max(axis=1)
    # argmax は「最初に出現した最大値」を返す ⇒ 列の並び順が tie の優先順位になる
    argmax = inten.argmax(axis=1)

    labels = argmax.astype(np.int64)

    # 中立: 最大強度が閾値未満(既定 min_intensity=1 なら「全て0」が中立)
    neutral_mask = max_val < config.min_intensity
    labels[neutral_mask] = NEUTRAL_ID

    # tie(同点最大が複数)の扱い
    if config.tie_break == "neutral":
        tie_count = (inten == max_val[:, None]).sum(axis=1)
        tie_mask = (tie_count > 1) & (~neutral_mask)
        labels[tie_mask] = NEUTRAL_ID
    elif config.tie_break != "priority":
        raise ValueError(f"tie_break は 'priority' か 'neutral'。got={config.tie_break!r}")

    return labels


# ---------------------------------------------------------------------------
# 4) DatasetDict を組み立てる
# ---------------------------------------------------------------------------
def build_datasets(config: Config) -> DatasetDict:
    """生 TSV から train/validation/test の DatasetDict を作る。"""
    download_wrime(config)
    df = load_raw(config)

    if "Sentence" not in df.columns:
        raise KeyError("TSV に 'Sentence' 列がありません(想定と異なる形式)")
    split_col = "Train/Dev/Test"
    if split_col not in df.columns:
        raise KeyError(f"TSV に '{split_col}' 列がありません(想定と異なる形式)")

    df = df.copy()
    # 先に公式 split に絞る。WRIME ver2 は約8800行が split 空(reader 未アノテーション)で、
    # これらは教師あり学習に使えないため、ラベル計算の前に除外する。
    df["split"] = df[split_col].str.strip().str.lower().map(_SPLIT_MAP)
    unknown = int(df["split"].isna().sum())
    if unknown:
        print(f"[data] split 未割当の {unknown} 行を除外(公式 train/dev/test のみ使用)")
        df = df[df["split"].notna()].copy()

    df["label"] = build_labels(df, config)

    class_names = [ID2LABEL[i] for i in range(NUM_LABELS)]
    class_label = ClassLabel(names=class_names)

    dsd = {}
    for split in ("train", "validation", "test"):
        sub = df[df["split"] == split][["Sentence", "label"]].rename(
            columns={"Sentence": "text"}
        )
        if config.limit_samples > 0:
            sub = sub.head(config.limit_samples)
        ds = Dataset.from_pandas(sub, preserve_index=False)
        ds = ds.cast_column("label", class_label)
        dsd[split] = ds

    return DatasetDict(dsd)


# ---------------------------------------------------------------------------
# 5) 保存 / 読み込み / 分布表示
# ---------------------------------------------------------------------------
def save_datasets(dsd: DatasetDict, config: Config) -> None:
    config.processed_dir.mkdir(parents=True, exist_ok=True)
    dsd.save_to_disk(str(config.processed_dir))
    print(f"[data] 保存: {config.processed_dir}")


def load_datasets(config: Config) -> DatasetDict:
    from datasets import load_from_disk

    return load_from_disk(str(config.processed_dir))


def label_distribution(ds: Dataset) -> Counter:
    """split 内のラベル分布(日本語名 -> 件数)を返す。"""
    names = ds.features["label"].names
    counts = Counter(ds["label"])
    return Counter({names[k]: v for k, v in sorted(counts.items())})


def print_summary(dsd: DatasetDict) -> None:
    print("\n=== データセット概要 ===")
    for split, ds in dsd.items():
        print(f"[{split}] {len(ds)} 件")
        dist = label_distribution(ds)
        total = sum(dist.values()) or 1
        for name, cnt in dist.most_common():
            print(f"    {name:<4} {cnt:>6}  ({cnt / total:5.1%})")
