"""評価指標。

不均衡データなので accuracy だけを見ると誤解する(多数派を当てるだけで高く出る)。
クラスを平等に扱う macro-F1 と、クラスごとの F1・混同行列を主役にする。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from .config import ID2LABEL, NUM_LABELS


def _label_names() -> list[str]:
    return [ID2LABEL[i] for i in range(NUM_LABELS)]


def _to_preds(logits: np.ndarray | tuple) -> np.ndarray:
    if isinstance(logits, tuple):
        logits = logits[0]
    return np.asarray(logits).argmax(axis=-1)


def compute_metrics(eval_pred) -> dict[str, float]:
    """Trainer に渡す評価関数。logits と正解から指標 dict を返す。"""
    logits, labels = eval_pred
    preds = _to_preds(logits)
    labels = np.asarray(labels)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "micro_f1": f1_score(labels, preds, average="micro", zero_division=0),
        "weighted_f1": f1_score(labels, preds, average="weighted", zero_division=0),
    }


def full_report(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    """クラスごとの precision/recall/F1 を含むテキストレポート。"""
    return classification_report(
        y_true,
        y_pred,
        labels=list(range(NUM_LABELS)),
        target_names=_label_names(),
        zero_division=0,
        digits=3,
    )


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=list(range(NUM_LABELS)))


def print_confusion(cm: np.ndarray) -> None:
    """混同行列をテキストで表示(行=正解, 列=予測)。"""
    names = _label_names()
    header = "正解\\予測".ljust(8) + "".join(n[:4].rjust(6) for n in names)
    print(header)
    for i, row in enumerate(cm):
        line = names[i][:6].ljust(8) + "".join(str(v).rjust(6) for v in row)
        print(line)


def plot_confusion_matrix(cm: np.ndarray, out_path: Path, normalize: bool = True) -> None:
    """混同行列を画像として保存する(任意)。"""
    import matplotlib

    matplotlib.use("Agg")  # GUI 不要のバックエンド
    import matplotlib.pyplot as plt

    # 日本語ラベルが豆腐(□)にならないよう、あれば日本語フォントを使う
    for jp_font in ("Yu Gothic", "Meiryo", "MS Gothic", "Noto Sans CJK JP"):
        matplotlib.rcParams["font.family"] = jp_font
        try:
            from matplotlib.font_manager import findfont

            findfont(jp_font, fallback_to_default=False)
            break
        except Exception:  # noqa: BLE001
            continue
    matplotlib.rcParams["axes.unicode_minus"] = False

    names = _label_names()
    mat = cm.astype(float)
    if normalize:
        row_sums = mat.sum(axis=1, keepdims=True)
        mat = np.divide(mat, row_sums, out=np.zeros_like(mat), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(mat, cmap="magma")
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticklabels(names)
    ax.set_xlabel("予測")
    ax.set_ylabel("正解")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[eval] 混同行列を保存: {out_path}")
