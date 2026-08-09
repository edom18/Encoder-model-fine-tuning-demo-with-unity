"""学習済みモデルを ONNX に書き出す(Unity Sentis 等でモバイル推論するため)。

ここでやること:
- PyTorch モデルの forward グラフ(分類ヘッドまで)を ONNX に変換する。
- ONNX Runtime で実行し、PyTorch と logits が一致するか検証する(壊れていないか)。
- モバイル向けに int8 量子化した ONNX も作る(サイズ削減)。
- Unity 側で必要な id2label.json と tokenizer 一式も同じフォルダへ出す。

依存は任意グループ:  uv sync --extra onnx
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import Config

# 検証や export のダミー入力に使う文
_SAMPLE_TEXTS = ["今日は最高の一日だった！", "電車が遅れて最悪だ"]


def _load_for_export(config: Config):
    """保存済みモデルを ONNX 変換に適した形で読み込む。

    ModernBERT の既定 SDPA アテンションは ONNX 変換で癖が出ることがあるので、
    素の演算に分解される eager 実装に切り替えて変換の安定性を上げる。
    """
    model_dir = str(config.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, attn_implementation="eager"
    )
    model.eval()
    return tokenizer, model


def export_to_onnx(config: Config, out_dir: Path, opset: int = 17) -> Path:
    """FP32 の ONNX を書き出し、付随ファイルも出力する。戻り値は .onnx のパス。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer, model = _load_for_export(config)

    # バッチ1・短系列のダミー入力でグラフをトレースする
    dummy = tokenizer("エクスポート用のダミー文です。", return_tensors="pt")
    onnx_path = out_dir / "model.onnx"

    torch.onnx.export(
        model,
        (dummy["input_ids"], dummy["attention_mask"]),
        str(onnx_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        # バッチ長・系列長は可変にする(推論時に文の長さが変わるため)
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "logits": {0: "batch"},
        },
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,  # 旧来(TorchScript)エクスポータを使う(onnxscript 不要で安定)
    )
    print(f"[onnx] FP32 を書き出し: {onnx_path} "
          f"({onnx_path.stat().st_size / 1e6:.1f} MB)")

    _dump_sidecar_files(config, tokenizer, model, out_dir)
    return onnx_path


def _dump_sidecar_files(config: Config, tokenizer, model, out_dir: Path) -> None:
    """Unity 側で必要になる補助ファイルを書き出す。"""
    # id2label(推論結果の数字 → 「喜び」等の変換表)
    id2label = {int(i): label for i, label in model.config.id2label.items()}
    (out_dir / "id2label.json").write_text(
        json.dumps(id2label, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # tokenizer 一式(Unity 側で「文 → id」に使う。tokenizer.json 等)
    tokenizer.save_pretrained(str(out_dir / "tokenizer"))
    # 入力仕様のメモ(max_length など)
    meta = {
        "model_name": config.model_name,
        "max_length": config.max_length,
        "num_labels": model.config.num_labels,
        "inputs": ["input_ids (int64)", "attention_mask (int64)"],
        "output": "logits (float32, shape [batch, num_labels])",
    }
    (out_dir / "export_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[onnx] 付随ファイル: id2label.json / export_meta.json / tokenizer/")


def quantize_onnx(fp32_path: Path, int8_path: Path) -> Path:
    """動的量子化で重みを int8 にした ONNX を作る(モバイルのサイズ削減)。"""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(
        str(fp32_path),
        str(int8_path),
        weight_type=QuantType.QInt8,
    )
    print(f"[onnx] INT8 を書き出し: {int8_path} "
          f"({Path(int8_path).stat().st_size / 1e6:.1f} MB)")
    return Path(int8_path)


def verify_onnx(config: Config, onnx_path: Path, texts: list[str] | None = None) -> dict:
    """PyTorch と ONNX Runtime の出力を突き合わせて、変換が正しいか確かめる。"""
    import onnxruntime as ort

    texts = texts or _SAMPLE_TEXTS
    tokenizer, model = _load_for_export(config)
    enc = tokenizer(
        texts, padding=True, truncation=True,
        max_length=config.max_length, return_tensors="pt",
    )

    with torch.no_grad():
        torch_logits = model(**enc).logits.numpy()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    # モデルが宣言する入力型(int64 版か int32 版か)に合わせてキャストする
    input_types = {i.name: i.type for i in session.get_inputs()}

    def _cast(name: str, arr):
        return arr.astype(np.int32) if "int32" in input_types.get(name, "") else arr.astype(np.int64)

    onnx_logits = session.run(
        ["logits"],
        {
            "input_ids": _cast("input_ids", enc["input_ids"].numpy()),
            "attention_mask": _cast("attention_mask", enc["attention_mask"].numpy()),
        },
    )[0]

    torch_pred = torch_logits.argmax(-1)
    onnx_pred = onnx_logits.argmax(-1)
    return {
        "max_abs_diff": float(np.abs(torch_logits - onnx_logits).max()),
        "argmax_agree": bool((torch_pred == onnx_pred).all()),
        "torch_pred": torch_pred.tolist(),
        "onnx_pred": onnx_pred.tolist(),
        "id2label": {int(i): l for i, l in model.config.id2label.items()},
    }


class _LogitsInt32Wrapper(torch.nn.Module):
    """Sentis 向けラッパ: int32 の入力を受け取り logits だけを返す。

    Unity Sentis は int64 テンソルを扱えないことが多い。そこで外側の入力を
    int32 にする。ただし PyTorch の埋め込みは long(int64)を要求するので、
    内部で long にキャストしてからモデルに渡す。
    """

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        logits = self.model(
            input_ids=input_ids.long(),
            attention_mask=attention_mask.long(),
        ).logits
        return logits


def export_to_onnx_sentis(config: Config, out_dir: Path, opset: int = 17) -> Path:
    """Sentis 向けに「int32 入力」の ONNX を書き出す(model.sentis.onnx)。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer, model = _load_for_export(config)
    wrapper = _LogitsInt32Wrapper(model).eval()

    dummy = tokenizer("エクスポート用のダミー文です。", return_tensors="pt")
    input_ids = dummy["input_ids"].to(torch.int32)
    attention_mask = dummy["attention_mask"].to(torch.int32)
    onnx_path = out_dir / "model.sentis.onnx"

    torch.onnx.export(
        wrapper,
        (input_ids, attention_mask),
        str(onnx_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "logits": {0: "batch"},
        },
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"[onnx] Sentis 用(int32入力)を書き出し: {onnx_path} "
          f"({onnx_path.stat().st_size / 1e6:.1f} MB)")
    return onnx_path


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max()
    exp = np.exp(shifted)
    return exp / exp.sum()


def dump_test_vector(config: Config, text: str) -> dict:
    """1文をトークナイズし、C# に貼れる id 列と、PyTorch の期待出力を返す。

    Sentis の推論経路(トークナイザ抜き)を単体検証するための材料。
    """
    tokenizer, model = _load_for_export(config)
    enc = tokenizer(
        text, truncation=True, max_length=config.max_length, return_tensors="pt"
    )
    with torch.no_grad():
        logits = model(**enc).logits[0].numpy()
    probs = _softmax(logits)
    pred_id = int(logits.argmax())
    id2label = {int(i): l for i, l in model.config.id2label.items()}
    return {
        "text": text,
        "seq_len": int(enc["input_ids"].shape[1]),
        "input_ids": enc["input_ids"][0].tolist(),
        "attention_mask": enc["attention_mask"][0].tolist(),
        "logits": [round(float(x), 4) for x in logits],
        "probs": [round(float(x), 4) for x in probs],
        "pred_id": pred_id,
        "pred_label": id2label[pred_id],
        "id2label": id2label,
    }
