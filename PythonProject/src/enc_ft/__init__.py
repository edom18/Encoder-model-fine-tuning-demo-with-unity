"""enc_ft: エンコーダのみモデル(日本語)の感情分類ファインチューニング用パッケージ.

学習マテリアル(learning/)と対で使うリファレンス実装。各モジュールは
1つの関心事に対応する:

- config    : モデル名・ハイパラ・データ列名などの設定を1箇所に集約
- data      : WRIME ver2 の取得と9クラス感情ラベルへの前処理
- tokenize  : トークナイズと動的パディング collator
- model     : 事前学習済みエンコーダ + 分類ヘッドの構築
- train     : ファインチューニング(Trainer 版 / 生ループ版)
- evaluate  : macro/micro-F1・per-class・混同行列
- infer     : 学習済みモデルによる推論
"""

__version__ = "0.1.0"
