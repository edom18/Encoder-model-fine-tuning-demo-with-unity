"""設定を1箇所に集約する。

「どのモデルを使うか」「ラベルをどう作るか」「学習ハイパラは何か」を
散らさず、この Config だけを見れば実験条件が分かる状態にする。
学習ループやデータ処理は、この設定を受け取って動く(密結合を避ける)。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# ラベル定義(Plutchik の8基本感情 + 中立 = 9クラス)
# ---------------------------------------------------------------------------
# WRIME の感情列はこの順で並ぶ。id はこの順に 0..7 を割り当て、中立を 8 にする。
# 日本語名は推論結果の可読性のために持つ(model.config.id2label に保存する)。
EMOTIONS: list[tuple[str, str]] = [
    ("Joy", "喜び"),
    ("Sadness", "悲しみ"),
    ("Anticipation", "期待"),
    ("Surprise", "驚き"),
    ("Anger", "怒り"),
    ("Fear", "恐れ"),
    ("Disgust", "嫌悪"),
    ("Trust", "信頼"),
]
NEUTRAL_JA = "中立"

# id -> 日本語ラベル(中立を最後の id に置く)
ID2LABEL: dict[int, str] = {i: ja for i, (_en, ja) in enumerate(EMOTIONS)}
ID2LABEL[len(EMOTIONS)] = NEUTRAL_JA
LABEL2ID: dict[str, int] = {ja: i for i, ja in ID2LABEL.items()}
NUM_LABELS: int = len(ID2LABEL)  # = 9
NEUTRAL_ID: int = len(EMOTIONS)  # = 8

# 感情の英語キー(TSV 列名の組み立てや tie 解決の優先順位に使う)
EMOTION_KEYS: list[str] = [en for en, _ja in EMOTIONS]


# ---------------------------------------------------------------------------
# プロジェクトのパス
# ---------------------------------------------------------------------------
# このファイル: <repo>/src/enc_ft/config.py → parents[2] が <repo>
REPO_ROOT: Path = Path(__file__).resolve().parents[2]


@dataclass
class Config:
    """実験1回分の設定。"""

    # --- モデル ---
    # 既定は日本語 ModernBERT(MeCab 不要・fast tokenizer 同梱で導入が最も確実)。
    # classic BERT を試すなら "tohoku-nlp/bert-base-japanese-v3"
    # (別途 `uv sync --extra tohoku` が必要)。
    model_name: str = "sbintuitions/modernbert-ja-130m"
    max_length: int = 192  # WRIME は短文中心。長すぎると無駄に遅い。

    # --- データ / ラベル設計 ---
    wrime_url: str = (
        "https://raw.githubusercontent.com/ids-cv/wrime/master/wrime-ver2.tsv"
    )
    # ラベルを作る視点。"reader"(読者平均) か "writer"(書き手)。
    # 既定は reader=「テキストがどう受け取られるか」でアシスタント用途に合う。
    label_perspective: str = "reader"
    # 有効票とみなす最小強度。0 のままだと弱い感情も拾う。1 にすると
    # 「強度1以上が無ければ中立」と厳しめになる。
    min_intensity: int = 1
    # 同点最大(tie)のときの決め方: "priority"(EMOTION_KEYS の並び順で先勝ち)
    # か "neutral"(tie は中立に倒す)。
    tie_break: str = "priority"

    # --- 学習ハイパラ ---
    seed: int = 42
    learning_rate: float = 2e-5
    batch_size: int = 32
    eval_batch_size: int = 64
    num_epochs: int = 3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    # Blackwell は bf16 が安定して速い(fp16 ではなく)。
    bf16: bool = True
    # クラス不均衡対策。CrossEntropy に逆頻度重みを与える。
    use_class_weights: bool = True

    # --- 実行制御(スモーク用) ---
    # >0 のとき各 split をこの件数に切り詰める。動作確認・CI に使う。
    limit_samples: int = 0

    # --- 出力先 ---
    data_dir: Path = field(default_factory=lambda: REPO_ROOT / "data")
    output_dir: Path = field(default_factory=lambda: REPO_ROOT / "outputs")

    # ---- 便利プロパティ ----
    @property
    def raw_tsv_path(self) -> Path:
        return self.data_dir / "raw" / "wrime-ver2.tsv"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def model_dir(self) -> Path:
        """学習済みモデルの保存先。"""
        return self.output_dir / "model"

    def emotion_columns(self) -> list[str]:
        """設定した視点に応じた8感情の TSV 列名を返す。"""
        if self.label_perspective == "reader":
            prefix = "Avg. Readers_"
        elif self.label_perspective == "writer":
            prefix = "Writer_"
        else:  # pragma: no cover - 設定ミスを早期に知らせる
            raise ValueError(
                f"label_perspective は 'reader' か 'writer'。got={self.label_perspective!r}"
            )
        return [prefix + key for key in EMOTION_KEYS]


def tiny_config(**overrides) -> Config:
    """スモークテスト用の極小設定(数十件・1〜2エポック)。"""
    base = dict(
        limit_samples=64,
        num_epochs=1,
        batch_size=8,
        eval_batch_size=8,
        max_length=64,
    )
    base.update(overrides)
    return Config(**base)


# よく使う既定インスタンス
DEFAULT = Config()
