#!/usr/bin/env python3
"""Исполняет листинги книги из listings/ и сверяет их вывод с напечатанным в книге.

Листинги одного файла выполняются по порядку в общем пространстве имён: глава часто опирается на функцию
из предыдущего листинга. Вывод листинга сравнивается с блоком «# Вывод в книге:» построчно, без учёта
пробелов в конце строк и пустых строк по краям. Листинги, которым нужны сеть, видеокарта или сервер,
в файле закомментированы и не запускаются.

    python scripts/check_listings.py listings/ch05/05-modern-models.py   # один файл
    python scripts/check_listings.py                                     # все листинги (долго)

Запускать из корня репозитория. Числа, зависящие от процессора и версии библиотек, могут отличаться в
последнем знаке (см. «Воспроизводимость» в README).
"""
from __future__ import annotations

import contextlib
import difflib
import io
import re
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CELL_RE = re.compile(r"^# %% Листинг «(?P<title>.*?)»[^\n]*\n(?P<body>.*?)(?=^# %% |\Z)", re.S | re.M)


def normalize(text: str) -> list[str]:
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").split("\n")]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return lines


def check_file(path: Path) -> int:
    namespace: dict = {"__name__": "__listing__"}
    failures = 0
    for m in CELL_RE.finditer(path.read_text(encoding="utf-8")):
        title, body = m.group("title"), m.group("body")
        if body.startswith("# Проверка этот листинг не запускает"):
            print(f"SKIP {path.name}: {title}")
            continue
        code, _, out = body.partition("# Вывод в книге:\n")
        want = [l[4:] if l.startswith("# | ") else "" for l in out.splitlines()] if out else None
        captured = io.StringIO()
        try:
            with contextlib.redirect_stdout(captured):
                exec(compile(code, f"{path}:{title}", "exec", dont_inherit=True), namespace)
        except Exception:  # noqa: BLE001
            failures += 1
            print(f"FAIL {path.name}: {title}")
            traceback.print_exc(limit=3)
            continue
        if want is not None and normalize(captured.getvalue()) != normalize("\n".join(want)):
            failures += 1
            print(f"MISMATCH {path.name}: {title}")
            for line in difflib.unified_diff(normalize("\n".join(want)), normalize(captured.getvalue()),
                                             "в книге", "у вас", lineterm=""):
                print("    " + line)
            continue
        print(f"OK   {path.name}: {title}")
    return failures


def main() -> int:
    paths = [Path(a) for a in sys.argv[1:]] or sorted((ROOT / "listings").rglob("*.py"))
    return 1 if sum(check_file(p) for p in paths) else 0


if __name__ == "__main__":
    sys.exit(main())
