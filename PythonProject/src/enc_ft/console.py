"""Windows コンソール対策。

Windows の既定コンソールは cp932(Shift-JIS)で、日本語や em-dash 等を print すると
UnicodeEncodeError で落ちることがある。標準出力を UTF-8 に張り替えてこれを防ぐ。
各 script の冒頭で ``enable_utf8()`` を呼ぶ。
"""

from __future__ import annotations

import sys


def enable_utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass
