"""学習済みモデルを ONNX に書き出す(Unity Sentis 等モバイル配備用)。

事前に学習しておくこと:  uv run python scripts/02_train.py
依存を入れる:            uv sync --extra onnx

    uv run python scripts/05_export_onnx.py                 # FP32 + INT8 を書き出し検証
    uv run python scripts/05_export_onnx.py --no-quantize   # FP32 のみ
"""

from __future__ import annotations

import argparse

from enc_ft.config import Config
from enc_ft.console import enable_utf8
from enc_ft.export_onnx import (
    export_to_onnx,
    export_to_onnx_sentis,
    quantize_onnx,
    verify_onnx,
)


def _print_verify(title: str, result: dict) -> None:
    id2label = result["id2label"]
    torch_names = [id2label[i] for i in result["torch_pred"]]
    onnx_names = [id2label[i] for i in result["onnx_pred"]]
    print(f"\n=== 検証: {title} ===")
    print(f"  logits 最大絶対差: {result['max_abs_diff']:.6f}")
    print(f"  予測一致(argmax): {result['argmax_agree']}")
    print(f"  PyTorch 予測: {torch_names}")
    print(f"  ONNX    予測: {onnx_names}")


def main() -> int:
    enable_utf8()
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None, help="出力先(既定 outputs/onnx)")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--no-quantize", action="store_true", help="INT8 量子化をしない")
    parser.add_argument("--no-sentis", action="store_true",
                        help="Sentis 用(int32入力)の書き出しをしない")
    args = parser.parse_args()

    config = Config()
    if not (config.model_dir / "config.json").exists():
        print("[NG] 学習済みモデルがありません。先に scripts/02_train.py を実行してください。")
        return 1

    out_dir = args.out or (config.output_dir / "onnx")

    # 1) FP32(int64入力)。ONNX Runtime 向けの汎用版。
    fp32_path = export_to_onnx(config, out_dir, opset=args.opset)
    _print_verify("FP32 (int64入力)", verify_onnx(config, fp32_path))

    # 2) Sentis 用(int32入力)。Unity で Tensor<int> をそのまま渡せる。
    if not args.no_sentis:
        sentis_path = export_to_onnx_sentis(config, out_dir, opset=args.opset)
        _print_verify("Sentis (int32入力)", verify_onnx(config, sentis_path))

    # 3) INT8 量子化(ONNX Runtime 用。モバイルでサイズを落とす)
    if not args.no_quantize:
        int8_path = fp32_path.with_name("model.int8.onnx")
        quantize_onnx(fp32_path, int8_path)
        _print_verify("INT8 (int64入力)", verify_onnx(config, int8_path))

    print(f"\n[done] 出力先: {out_dir}")
    print("  model.onnx(汎用) / model.sentis.onnx(Unity Sentis) / model.int8.onnx")
    print("  id2label.json / export_meta.json / tokenizer/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
