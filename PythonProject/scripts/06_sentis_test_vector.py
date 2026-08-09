"""Sentis 推論経路の検証用ベクトルを出力する。

トークナイザを C# で用意する「前」に、Unity 側の推論(ONNX 実行)だけを
単体検証するための材料を出す。ここで出た input_ids / attention_mask を C# に貼り、
Sentis の出力(logits)が「期待 logits」と一致すれば、モデルと推論経路は正しい。

    uv run python scripts/06_sentis_test_vector.py
    uv run python scripts/06_sentis_test_vector.py "今日は最高の一日だった！"
"""

from __future__ import annotations

import argparse

from enc_ft.config import Config
from enc_ft.console import enable_utf8
from enc_ft.export_onnx import dump_test_vector


def _csharp_int_array(name: str, values: list[int]) -> str:
    body = ", ".join(str(v) for v in values)
    return f"int[] {name} = new int[] {{ {body} }};"


def main() -> int:
    enable_utf8()
    parser = argparse.ArgumentParser()
    parser.add_argument("text", nargs="?", default="今日は最高の一日だった！")
    args = parser.parse_args()

    config = Config()
    if not (config.model_dir / "config.json").exists():
        print("[NG] 学習済みモデルがありません。先に scripts/02_train.py を実行してください。")
        return 1

    v = dump_test_vector(config, args.text)

    print("=" * 64)
    print(f" 検証ベクトル: 「{v['text']}」")
    print("=" * 64)
    print(f"seq_len = {v['seq_len']}")
    print()
    print("// ---- ここから下を C# に貼る ----")
    print(_csharp_int_array("inputIds", v["input_ids"]))
    print(_csharp_int_array("attentionMask", v["attention_mask"]))
    print("// ---- ここまで ----")
    print()
    print("期待される出力(Sentis の logits がこれと一致すれば成功):")
    print(f"  pred_id    = {v['pred_id']}  ({v['pred_label']})")
    print(f"  logits     = {v['logits']}")
    print(f"  probs      = {v['probs']}")
    print()
    print("id2label:")
    for i, label in v["id2label"].items():
        print(f"  {i}: {label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
