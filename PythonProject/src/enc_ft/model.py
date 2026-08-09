"""分類モデルの構築。

事前学習済みエンコーダの上に「分類ヘッド(線形層)」を載せる。
AutoModelForSequenceClassification がこれを自動で組み立ててくれる:
  エンコーダ出力 → プーリング([CLS] 等) → dropout → Linear(hidden -> num_labels)
ヘッドの重みはランダム初期化されるので、ファインチューニングで学習する。
"""

from __future__ import annotations

from transformers import AutoConfig, AutoModelForSequenceClassification, PreTrainedModel

from .config import ID2LABEL, LABEL2ID, NUM_LABELS, Config


def build_model(config: Config) -> PreTrainedModel:
    """9クラス分類用のモデルを組み立てる。

    id2label / label2id を config に持たせることで、保存後のモデルが
    「数字」ではなく「喜び」等の日本語ラベルを返せるようになる(推論・再現で重要)。
    """
    hf_config = AutoConfig.from_pretrained(
        config.model_name,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    # ModernBERT は既定で torch.compile を使うことがある。Windows では Triton が無く
    # 詰まりやすいので明示的に無効化する(この属性を持つモデルのときだけ)。
    if hasattr(hf_config, "reference_compile"):
        hf_config.reference_compile = False

    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        config=hf_config,
    )
    return model


def freeze_encoder(model: PreTrainedModel) -> None:
    """エンコーダ本体を凍結し、分類ヘッドだけ学習する(ch4 の実験用)。

    学習パラメータが激減するので速い・省メモリだが、精度は全層学習に劣ることが多い。
    """
    base = model.base_model  # 分類ヘッドを除いたエンコーダ部分
    for param in base.parameters():
        param.requires_grad = False


def count_parameters(model: PreTrainedModel) -> tuple[int, int]:
    """(学習対象パラメータ数, 全パラメータ数) を返す。"""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
