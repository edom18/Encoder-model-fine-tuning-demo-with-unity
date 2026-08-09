"""学習済みモデルを test split で評価する。

    uv run python scripts/03_evaluate.py
    uv run python scripts/03_evaluate.py --limit 200   # スモーク

macro-F1・クラスごとの F1・混同行列(テキスト + PNG)を出力する。
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from enc_ft.config import Config
from enc_ft.console import enable_utf8
from enc_ft.data import load_datasets
from enc_ft.evaluate import (
    compute_metrics,
    confusion,
    full_report,
    plot_confusion_matrix,
    print_confusion,
)
from enc_ft.infer import SentimentClassifier
from enc_ft.tokenize import make_collator


@torch.no_grad()
def _predict_split(clf: SentimentClassifier, ds, config: Config):
    """test split 全体の logits と正解ラベルを集める。"""
    tokenized = ds.map(
        lambda b: clf.tokenizer(b["text"], truncation=True, max_length=config.max_length),
        batched=True,
        remove_columns=["text"],
    ).rename_column("label", "labels")
    collator = make_collator(clf.tokenizer)
    loader = DataLoader(tokenized, batch_size=config.eval_batch_size,
                        collate_fn=collator, num_workers=0)

    logits_all, labels_all = [], []
    for batch in loader:
        labels = batch.pop("labels")
        batch = {k: v.to(clf.device) for k, v in batch.items()}
        logits = clf.model(**batch).logits.float().cpu()
        logits_all.append(logits)
        labels_all.append(labels)
    return torch.cat(logits_all).numpy(), torch.cat(labels_all).numpy()


def main() -> int:
    enable_utf8()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    config = Config()
    clf = SentimentClassifier(model_dir=args.model_dir, config=config)
    test = load_datasets(config)["test"]
    if args.limit > 0:
        test = test.select(range(min(args.limit, len(test))))

    logits, labels = _predict_split(clf, test, config)
    preds = logits.argmax(axis=-1)

    print("\n=== 全体指標 (test) ===")
    for k, v in compute_metrics((logits, labels)).items():
        print(f"  {k:<12}: {v:.4f}")

    print("\n=== クラス別レポート ===")
    print(full_report(np.asarray(labels), preds))

    print("=== 混同行列(行=正解, 列=予測)===")
    cm = confusion(np.asarray(labels), preds)
    print_confusion(cm)

    out_png = config.output_dir / "confusion_matrix.png"
    plot_confusion_matrix(cm, out_png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
