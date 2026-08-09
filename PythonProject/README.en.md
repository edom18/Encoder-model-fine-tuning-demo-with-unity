# Encoder model fine tuning

[日本語](README.md)

A sample implementation that fine-tunes an encoder-only model (Japanese BERT family) for
**multi-class emotion classification**.

- Model: [`sbintuitions/modernbert-ja-130m`](https://huggingface.co/sbintuitions/modernbert-ja-130m) (Japanese ModernBERT, 132M)
- Data: [WRIME ver2](https://github.com/ids-cv/wrime) (Japanese emotion labels, Plutchik's 8 emotions + neutral = **9 classes**)
- Measured baseline: **test macro-F1 ≈ 0.465 / accuracy ≈ 0.629** (3 epochs, ~146s on an RTX 5090)

---

## Requirements

| Item | Details |
|------|------|
| OS | Windows 11 (this repo's verified environment) |
| Python | 3.10 |
| Package manager | [uv](https://docs.astral.sh/uv/) |
| GPU | NVIDIA GPU recommended. Verified on an RTX 5090 (Blackwell / sm_120) |

CUDA PyTorch uses the **cu128 build** (required for Blackwell). `pyproject.toml` points only the `torch`
package at the official PyTorch index, so `uv sync` alone pulls it in. It also runs on CPU, just slower.

---

## Quick Start

```powershell
# 1. Install dependencies (cu128 torch etc. go into a dedicated .venv; first run downloads a few GB)
uv sync

# 2. Check the GPU (whether sm_120 kernels are present, etc.)
uv run python scripts/00_check_env.py

# 3. Prepare data (fetch WRIME, reshape into 9 classes, and save)
uv run python scripts/01_prepare_data.py

# 4. Fine-tune (Trainer-based version. ~150s on an RTX 5090)
uv run python scripts/02_train.py
#   Raw training-loop version, to see what's inside:  uv run python scripts/02_train.py --raw
#   A tiny smoke run of a few dozen samples:           uv run python scripts/02_train.py --tiny

# 5. Evaluate on the test set (macro-F1, per-class F1, confusion matrix → outputs/confusion_matrix.png)
uv run python scripts/03_evaluate.py

# 6. Inference (pass any sentence)
uv run python scripts/04_infer.py "今日は最高の一日だった！"

# 7. (Optional) Export to ONNX for mobile/edge deployment (Unity Sentis etc.)
uv sync --extra onnx
uv run python scripts/05_export_onnx.py            # model.onnx / model.sentis.onnx (int32) / model.int8.onnx
uv run python scripts/06_sentis_test_vector.py "今日は最高だ"   # emits token ids and expected logits for Sentis verification
```

> **Windows console mojibake workaround**: each script switches stdout to UTF-8. If you still see
> garbled output, run with `$env:PYTHONUTF8=1`.

---

## Tests

```powershell
# Fast unit tests (no network required)
uv run python -m pytest -q

# Integration tests using real data + model (exercises every path on a few dozen samples)
$env:ENC_FT_INTEGRATION=1; uv run python -m pytest -q -s
```

---

## Customization

Key settings are centralized in `Config` in `src/enc_ft/config.py`, and can also be overridden from the CLI.

```powershell
uv run python scripts/02_train.py --epochs 5 --lr 1e-5 --batch-size 16
uv run python scripts/01_prepare_data.py --perspective writer --min-intensity 2
```

- **Training on your own data**: just swap out `build_datasets()` in `src/enc_ft/data.py` to load your own
  CSV with `text`, `label` (0–8), and `split` columns. Tokenization, training, evaluation, and inference all
  work unchanged (details in
  [learning/07-inference-ops.html](learning/07-inference-ops.html)).
- **Trying classic BERT (tohoku)**: after `uv sync --extra tohoku`, pass
  `--model tohoku-nlp/bert-base-japanese-v3`.

---

## Common pitfalls in this environment (and fixes)

| Symptom | Fix |
|------|------|
| `cuda.is_available()==False` | A CPU-only torch build is still installed. Delete `.venv` and run `uv sync` |
| `no kernel image` on GPU | You're on cu126 or earlier. Use cu128 (already configured in this `pyproject.toml`) |
| `UnicodeEncodeError` when printing Japanese | Add `PYTHONUTF8=1` |
| `datasets` script-based loading fails | Not applicable here — this project downloads the original TSV directly |

---

## License / Attribution

- WRIME: [ids-cv/wrime](https://github.com/ids-cv/wrime) (follow the original license/terms when using it)
- Model: `sbintuitions/modernbert-ja-130m` (MIT)
