"""環境チェック: GPU(Blackwell/sm_120)で torch が動くかを最初に確かめる。

このプロジェクト最大の地雷は「CPU 版 torch のまま」「cu126 wheel(sm_120 未対応)を掴む」
の2つ。ここが緑にならない限り学習は GPU で動かないので、必ず最初に実行する。

    uv run scripts/00_check_env.py
"""

from __future__ import annotations

import sys

from enc_ft.console import enable_utf8


def main() -> int:
    enable_utf8()
    print("=" * 60)
    print(" 環境チェック (enc-ft)")
    print("=" * 60)
    print(f"Python : {sys.version.split()[0]}")

    try:
        import torch
    except ModuleNotFoundError:
        print("[NG] torch が見つかりません。`uv sync` を実行してください。")
        return 1

    print(f"torch  : {torch.__version__}")
    print(f"build CUDA : {torch.version.cuda}")  # cu128 なら '12.8'

    ok = True

    # 1) CUDA が使えるか
    if not torch.cuda.is_available():
        print("[NG] torch.cuda.is_available() == False")
        print("     → CPU 版 torch が入っている可能性大。")
        print("       .venv を消して `uv sync` で cu128 版を入れ直す。")
        ok = False
    else:
        cap = torch.cuda.get_device_capability()
        name = torch.cuda.get_device_name(0)
        print(f"[OK] CUDA 利用可能: {name}  compute capability sm_{cap[0]}{cap[1]}")

        # 2) ビルド済みアーキ一覧に sm_120 が含まれるか(最終的な地雷検知点)
        arch_list = torch.cuda.get_arch_list()
        print(f"     ビルド済みアーキ: {arch_list}")
        target = f"sm_{cap[0]}{cap[1]}"
        if target not in arch_list:
            # Blackwell(sm_120)。前方互換で動くこともあるが警告する。
            print(f"[警告] {target} が torch のビルド済みアーキ一覧に無い。")
            print("       cu128(以降)ビルドを使っているか確認する。")
            ok = False
        else:
            print(f"[OK] {target} のカーネルを同梱")

        # 3) 実際に GPU 上で行列積を1回走らせて確かめる
        try:
            x = torch.randn(512, 512, device="cuda")
            y = (x @ x).sum().item()
            print(f"[OK] GPU 行列積テスト成功 (sum={y:.1f})")
            # bf16 が使えるか(Blackwell 推奨の混合精度)
            if torch.cuda.is_bf16_supported():
                print("[OK] bf16 サポートあり(学習は bf16 AMP を使う)")
            else:
                print("[警告] bf16 未サポート。fp16 か fp32 を使う。")
        except Exception as exc:  # noqa: BLE001
            print(f"[NG] GPU 演算に失敗: {exc}")
            ok = False

    print("=" * 60)
    if ok:
        print(" 結果: OK — GPU で学習できます。")
        return 0
    print(" 結果: 要修正 — 上のメッセージに従って torch を入れ直してください。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
