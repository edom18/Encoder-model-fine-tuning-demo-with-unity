"""ファインチューニングの学習。2通り用意する。

- train_with_trainer : HuggingFace Trainer 版(実運用ではこちらを使う)
- train_raw_loop     : PyTorch の生ループ版(ch5 で「中で何が起きているか」を学ぶ用)

どちらも同じ結果(学習済みモデル + tokenizer を config.model_dir に保存)を目指す。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    get_linear_schedule_with_warmup,
    set_seed,
)

from .config import NUM_LABELS, Config
from .data import load_datasets
from .evaluate import compute_metrics
from .model import build_model, count_parameters
from .tokenize import load_tokenizer, make_collator, tokenize_datasets


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# クラス不均衡対策: 逆頻度クラス重み
# ---------------------------------------------------------------------------
def compute_class_weights(labels: list[int]) -> torch.Tensor:
    """逆頻度に基づくクラス重みを返す(少数クラスを重く扱う)。

    weight[c] = N / (num_classes * count[c])。バランス時は概ね 1 になる。
    未出現クラス(スモーク時に起こりうる)は count=1 とみなして発散を防ぐ。
    """
    counts = np.bincount(labels, minlength=NUM_LABELS).astype(np.float64)
    counts[counts == 0] = 1.0
    total = float(len(labels))
    weights = total / (NUM_LABELS * counts)
    return torch.tensor(weights, dtype=torch.float32)


# ---------------------------------------------------------------------------
# 共通の準備
# ---------------------------------------------------------------------------
def _prepare(config: Config):
    set_seed(config.seed)
    dsd = load_datasets(config)
    # 学習時にも間引きを効かせる(--tiny / --limit でのスモーク用)。
    # データ保存済みでも、ここで各 split を先頭 N 件に切り詰められる。
    if config.limit_samples > 0:
        from datasets import DatasetDict

        dsd = DatasetDict(
            {
                name: split.select(range(min(config.limit_samples, len(split))))
                for name, split in dsd.items()
            }
        )
    tokenizer = load_tokenizer(config)
    tokenized = tokenize_datasets(dsd, tokenizer, config)
    collator = make_collator(tokenizer)
    model = build_model(config)
    trainable, total = count_parameters(model)
    print(f"[train] 学習パラメータ: {trainable:,} / {total:,} ({trainable / total:.1%})")
    class_weights = (
        compute_class_weights(tokenized["train"]["labels"])
        if config.use_class_weights
        else None
    )
    return tokenizer, tokenized, collator, model, class_weights


def _save(model, tokenizer, config: Config) -> None:
    config.model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(config.model_dir)
    tokenizer.save_pretrained(config.model_dir)
    print(f"[train] 保存: {config.model_dir}")


# ---------------------------------------------------------------------------
# (A) Trainer 版
# ---------------------------------------------------------------------------
class WeightedTrainer(Trainer):
    """CrossEntropy にクラス重みを与えられる Trainer。

    Trainer は既定で重みなし CrossEntropy を使う。少数クラスを効かせるため
    compute_loss を差し替える。**kwargs は num_items_in_batch 等を吸収する。
    """

    def __init__(self, *args, class_weights: torch.Tensor | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weight = (
            self.class_weights.to(logits.device)
            if self.class_weights is not None
            else None
        )
        loss = F.cross_entropy(logits, labels, weight=weight)
        return (loss, outputs) if return_outputs else loss


def train_with_trainer(config: Config) -> Trainer:
    tokenizer, tokenized, collator, model, class_weights = _prepare(config)

    args = TrainingArguments(
        output_dir=str(config.output_dir / "trainer"),
        eval_strategy="epoch",          # transformers 5.x は eval_strategy(旧名は廃止)
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        num_train_epochs=config.num_epochs,
        warmup_ratio=config.warmup_ratio,
        max_grad_norm=config.max_grad_norm,
        bf16=config.bf16 and get_device() == "cuda",  # Blackwell 向け(fp16 ではない)
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=1,
        seed=config.seed,
        report_to="none",               # wandb 等への送信を無効化
        dataloader_num_workers=0,        # Windows では 0 が安全
    )

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=collator,
        processing_class=tokenizer,      # 旧 tokenizer= は廃止 → processing_class=
        compute_metrics=compute_metrics,
        class_weights=class_weights,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("[train] Trainer 版で学習開始")
    trainer.train()
    _save(model, tokenizer, config)

    print("[train] validation 最終評価:")
    print(trainer.evaluate())
    return trainer


# ---------------------------------------------------------------------------
# (B) 生ループ版(教材用: 中で何が起きているかを明示)
# ---------------------------------------------------------------------------
def _build_optimizer(model, config: Config) -> torch.optim.Optimizer:
    """bias と LayerNorm には weight decay をかけない(慣例)。"""
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or name.endswith(".bias"):
            no_decay.append(param)      # 1次元(bias/LayerNorm)は減衰なし
        else:
            decay.append(param)
    groups = [
        {"params": decay, "weight_decay": config.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=config.learning_rate)


@torch.no_grad()
def _evaluate_loop(model, loader, device) -> dict[str, float]:
    model.eval()
    all_logits, all_labels = [], []
    for batch in loader:
        labels = batch.pop("labels")
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch).logits.float().cpu()
        all_logits.append(logits)
        all_labels.append(labels)
    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    return compute_metrics((logits, labels))


def train_raw_loop(config: Config) -> None:
    device = get_device()
    tokenizer, tokenized, collator, model, class_weights = _prepare(config)
    model.to(device)

    train_loader = DataLoader(
        tokenized["train"],
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
    )
    val_loader = DataLoader(
        tokenized["validation"],
        batch_size=config.eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )

    optimizer = _build_optimizer(model, config)
    num_steps = len(train_loader) * config.num_epochs
    warmup_steps = int(config.warmup_ratio * num_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, num_steps)

    weight = class_weights.to(device) if class_weights is not None else None
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight)

    # bf16 は fp16 と違い勾配のダイナミックレンジが広く、GradScaler 不要。
    use_amp = config.bf16 and device == "cuda"

    print(f"[train] 生ループ版で学習開始 (device={device}, steps={num_steps})")
    for epoch in range(config.num_epochs):
        model.train()
        running = 0.0
        for step, batch in enumerate(train_loader, 1):
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                logits = model(**batch).logits
                loss = loss_fn(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            scheduler.step()

            running += loss.item()
            if step % 50 == 0:
                print(f"  epoch {epoch + 1} step {step}/{len(train_loader)} "
                      f"loss={running / step:.4f} lr={scheduler.get_last_lr()[0]:.2e}")

        metrics = _evaluate_loop(model, val_loader, device)
        print(f"[train] epoch {epoch + 1} val: "
              f"acc={metrics['accuracy']:.3f} macroF1={metrics['macro_f1']:.3f}")

    _save(model, tokenizer, config)
