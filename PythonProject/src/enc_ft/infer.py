"""学習済みモデルによる推論。

保存済みディレクトリ(既定 outputs/model)から model と tokenizer を読み込み、
日本語文に対する感情ラベルと確信度を返す。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import Config


@dataclass
class Prediction:
    text: str
    label: str                       # 例: "喜び"
    score: float                     # そのラベルの確率
    scores: dict[str, float]         # 全クラスの確率


class SentimentClassifier:
    """感情分類の推論器。"""

    def __init__(self, model_dir: str | None = None, config: Config | None = None):
        config = config or Config()
        path = model_dir or str(config.model_dir)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path)
        self.model.to(self.device).eval()
        self.max_length = config.max_length
        # id2label は学習時に config に保存済み(数字ではなく日本語ラベルが得られる)
        self.id2label = self.model.config.id2label

    @torch.no_grad()
    def predict(self, texts: str | list[str], batch_size: int = 32) -> list[Prediction]:
        if isinstance(texts, str):
            texts = [texts]
        results: list[Prediction] = []
        for start in range(0, len(texts), batch_size):
            chunk = texts[start : start + batch_size]
            enc = self.tokenizer(
                chunk,
                truncation=True,
                max_length=self.max_length,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            logits = self.model(**enc).logits.float()
            probs = torch.softmax(logits, dim=-1).cpu()
            for text, prob in zip(chunk, probs):
                top = int(prob.argmax())
                scores = {self.id2label[i]: float(p) for i, p in enumerate(prob)}
                results.append(
                    Prediction(
                        text=text,
                        label=self.id2label[top],
                        score=float(prob[top]),
                        scores=scores,
                    )
                )
        return results
