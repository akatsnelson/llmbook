"""Родословная настоящих файлов BookLM: манифесты с суммами всех родителей и проверка цепочки.

Глава 17 «Выпуск и сопровождение модели», раздел 17.1 «Родословная модели». Функции sha256_of, make_manifest, verify_parents и verify_lineage
берутся из модуля выпуска labs/ch17/release.py.

Скрипт пишет манифесты во временный каталог (файлы models/booklm/ не трогает), проверяет цепочку
токенизатор и база -> адаптер -> объединённые веса и показывает, что подмена базы ловится, хотя
адаптер цел.

Запуск из корня репозитория: python labs/ch17/manifest_lineage.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from labs.ch17.release import make_manifest, verify_lineage, verify_parents  # noqa: E402

MODELS = ROOT / "models" / "booklm"


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        files = {}
        for node, name in [("tokenizer-v1", "tokenizer.json"), ("base-step-N", "booklm-base.pt"),
                           ("lora-v1", "booklm-instruct-lora.pt"), ("merged-v1", "booklm-instruct.pt")]:
            files[node] = Path(tmp, name)
            shutil.copyfile(MODELS / name, files[node])   # копии: манифесты пишем рядом с ними
        manifests = {
            "base-step-N": make_manifest(files["base-step-N"], "base-step-N",
                                         {"tokenizer-v1": files["tokenizer-v1"]}, "pretrain"),
            "lora-v1": make_manifest(files["lora-v1"], "lora-v1",
                                     {"base-step-N": files["base-step-N"]}, "sft_lora r=16"),
            "merged-v1": make_manifest(files["merged-v1"], "merged-v1",
                                       {"base-step-N": files["base-step-N"], "lora-v1": files["lora-v1"]},
                                       "merge_lora"),
        }
        order = verify_lineage(manifests, files, "merged-v1")
        assert order[0] == "merged-v1" and set(order) == set(manifests), order
        for node in order:
            m = manifests[node]
            parents = ", ".join(f"{p} {s[:8]}" for p, s in m["parents"].items())
            print(f"{node:<12} {m['sha256'][:16]}  родители: {parents}")

        with files["base-step-N"].open("r+b") as f:      # один байт в базе
            f.seek(1000)
            byte = f.read(1)
            f.seek(1000)
            f.write(bytes([byte[0] ^ 1]))
        try:
            verify_parents(manifests["merged-v1"], files)
            raise AssertionError("подменённая база принята")
        except ValueError as error:
            print("стоп:", error)
    print("manifest_lineage: OK")


if __name__ == "__main__":
    main()
