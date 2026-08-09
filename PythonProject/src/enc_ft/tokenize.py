"""トークナイズと動的パディング。

- 事前学習済みモデルに対応する tokenizer を読み込む(モデルと必ずペア)。
- テキストをサブワード id 列に変換する。ここでは truncation のみ行い、
  padding は collator に任せる(=バッチ内の最大長にそろえる「動的パディング」で
  無駄な計算を減らす)。
"""

from __future__ import annotations

from datasets import DatasetDict
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    PreTrainedTokenizerBase,
)

from .config import Config


def load_tokenizer(config: Config) -> PreTrainedTokenizerBase:
    """モデルに対応した tokenizer を読み込む。"""
    return AutoTokenizer.from_pretrained(config.model_name)


def tokenize_datasets(
    dsd: DatasetDict,
    tokenizer: PreTrainedTokenizerBase,
    config: Config,
) -> DatasetDict:
    """text を input_ids / attention_mask に変換し、label を labels に改名する。

    - truncation=True, max_length=config.max_length で長すぎる文だけ切る。
    - padding はここでは行わない(collator が担当)。
    - モデルの forward が受け取る引数名は ``labels`` なので改名しておく
      (改名しないと Trainer が未使用列として捨て、loss が計算されない)。
    """

    def _tok(batch: dict) -> dict:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=config.max_length,
        )

    tokenized = dsd.map(_tok, batched=True, remove_columns=["text"])
    tokenized = tokenized.rename_column("label", "labels")
    return tokenized


def make_collator(tokenizer: PreTrainedTokenizerBase) -> DataCollatorWithPadding:
    """バッチごとに最大長へパディングする collator。"""
    return DataCollatorWithPadding(tokenizer, padding="longest")
