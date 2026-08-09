"""ファインチューニングを実行する。

    uv run python scripts/02_train.py                 # Trainer 版(推奨)
    uv run python scripts/02_train.py --raw           # 生ループ版(教材用)
    uv run python scripts/02_train.py --tiny          # 数十件でスモーク(動作確認)
    uv run python scripts/02_train.py --epochs 3 --batch-size 32 --lr 2e-5
"""

from __future__ import annotations

import argparse

from enc_ft.config import Config, tiny_config
from enc_ft.console import enable_utf8
from enc_ft.train import train_raw_loop, train_with_trainer


def main() -> int:
    enable_utf8()
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", action="store_true", help="生ループ版で学習")
    parser.add_argument("--tiny", action="store_true", help="スモーク設定で学習")
    parser.add_argument("--model", default=None, help="モデル名を上書き")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--limit", type=int, default=None, help=">0 で各splitを間引き")
    parser.add_argument("--no-class-weights", action="store_true")
    args = parser.parse_args()

    config = tiny_config() if args.tiny else Config()
    if args.model is not None:
        config.model_name = args.model
    if args.epochs is not None:
        config.num_epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.lr is not None:
        config.learning_rate = args.lr
    if args.limit is not None:
        config.limit_samples = args.limit
    if args.no_class_weights:
        config.use_class_weights = False

    print(f"[run] model={config.model_name} epochs={config.num_epochs} "
          f"batch={config.batch_size} lr={config.learning_rate} "
          f"limit={config.limit_samples} raw={args.raw}")

    if args.raw:
        train_raw_loop(config)
    else:
        train_with_trainer(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
