"""WRIME ver2 を取得し、9クラス感情データセットを作って保存する。

    uv run python scripts/01_prepare_data.py                  # 本番(全件)
    uv run python scripts/01_prepare_data.py --perspective writer
    uv run python scripts/01_prepare_data.py --limit 200      # 動作確認用に各splitを間引き
"""

from __future__ import annotations

import argparse

from enc_ft.config import Config
from enc_ft.console import enable_utf8
from enc_ft.data import build_datasets, print_summary, save_datasets


def main() -> int:
    enable_utf8()
    parser = argparse.ArgumentParser()
    parser.add_argument("--perspective", choices=["reader", "writer"], default="reader",
                        help="ラベルを作る視点(既定 reader=読者平均)")
    parser.add_argument("--min-intensity", type=int, default=1,
                        help="中立とみなす閾値。最大強度がこれ未満なら中立")
    parser.add_argument("--tie-break", choices=["priority", "neutral"], default="priority")
    parser.add_argument("--limit", type=int, default=0, help=">0 で各splitを間引き")
    args = parser.parse_args()

    config = Config(
        label_perspective=args.perspective,
        min_intensity=args.min_intensity,
        tie_break=args.tie_break,
        limit_samples=args.limit,
    )

    dsd = build_datasets(config)
    print_summary(dsd)
    save_datasets(dsd, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
