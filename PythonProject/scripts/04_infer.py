"""学習済みモデルで日本語文の感情を推論する。

    uv run python scripts/04_infer.py                       # サンプル文で試す
    uv run python scripts/04_infer.py "今日は最高の一日だった"  # 任意の文
"""

from __future__ import annotations

import argparse

from enc_ft.config import Config
from enc_ft.console import enable_utf8
from enc_ft.infer import SentimentClassifier

SAMPLES = [
    "今日は最高の一日だった！本当に嬉しい。",
    "電車が遅れて会議に間に合わなかった。最悪だ。",
    "明日の発表、緊張するけど楽しみだな。",
    "え、そんなことある？信じられない。",
    "この料理、正直ちょっと気持ち悪い味がする。",
    "手伝ってくれてありがとう。あなたは頼りになる。",
]


def main() -> int:
    enable_utf8()
    parser = argparse.ArgumentParser()
    parser.add_argument("texts", nargs="*", help="推論する文(省略時はサンプル)")
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--topk", type=int, default=3, help="上位k感情も表示")
    args = parser.parse_args()

    config = Config()
    clf = SentimentClassifier(model_dir=args.model_dir, config=config)
    texts = args.texts if args.texts else SAMPLES

    for pred in clf.predict(texts):
        print(f"\n入力: {pred.text}")
        print(f"予測: {pred.label}  (確信度 {pred.score:.1%})")
        top = sorted(pred.scores.items(), key=lambda kv: kv[1], reverse=True)[: args.topk]
        detail = "  ".join(f"{name}:{p:.2f}" for name, p in top)
        print(f"  上位: {detail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
