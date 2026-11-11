#!/usr/bin/env python3
"""Проверяет структуру и, по запросу, выполняет учебные ноутбуки."""

from __future__ import annotations

import argparse
import ast
import contextlib
import io
import json
from pathlib import Path
import sys


def code_source(cell: dict) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else source


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    args = parser.parse_args()
    root = args.root.resolve()
    paths = sorted((root / "notebooks").glob("*.ipynb"))
    errors: list[str] = []
    code_cells = 0

    for path in paths:
        try:
            notebook = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{path.relative_to(root)}: не удалось прочитать JSON: {exc}")
            continue

        if notebook.get("nbformat") != 4 or not isinstance(notebook.get("cells"), list):
            errors.append(f"{path.relative_to(root)}: ожидается Jupyter nbformat 4")
            continue

        namespace: dict[str, object] = {"__name__": "__main__"}
        for index, cell in enumerate(notebook["cells"], start=1):
            if cell.get("cell_type") != "code":
                continue
            code_cells += 1
            source = code_source(cell)
            try:
                ast.parse(source, filename=f"{path.name}:cell-{index}")
            except SyntaxError as exc:
                errors.append(
                    f"{path.relative_to(root)}: ячейка {index}: синтаксис Python: {exc.msg}"
                )
                continue
            if args.execute:
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        exec(compile(source, f"{path.name}:cell-{index}", "exec"), namespace)
                except Exception as exc:  # noqa: BLE001 - печатаем учебную ошибку с ячейкой
                    errors.append(
                        f"{path.relative_to(root)}: ячейка {index}: выполнение: "
                        f"{type(exc).__name__}: {exc}"
                    )

    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    action = "выполнено" if args.execute else "проверено"
    print(f"Ноутбуков {len(paths)}, Python-ячеек {code_cells}: {action} без ошибок.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

