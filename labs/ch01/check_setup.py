"""Глава 1, раздел 1.4 «Код и ноутбуки»: проверка окружения книги.

Запуск из корня репозитория: python labs/ch01/check_setup.py
Скрипт работает и из любого другого каталога, если указать путь к нему.

Скрипт проверяет, что установлены NumPy и PyTorch, что код BookLM из каталога booklm/
импортируется, что модель содержит ровно 6 622 464 параметра и что один прямой проход
выдаёт по 8192 оценки на каждую позицию текста, а в конце — что обученные веса скачаны через
Git LFS и загружаются в модель. Всё считается на процессоре за секунды.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # корень репозитория: labs/ch01/ -> ../..
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from booklm.model import BookLM, BookLMConfig  # noqa: E402

EXPECTED_PARAMS = 6_622_464


def main() -> None:
    print("Python", sys.version.split()[0], "| NumPy", np.__version__, "| PyTorch", torch.__version__)

    torch.manual_seed(0)
    cfg = BookLMConfig()
    model = BookLM(cfg)
    n = model.num_params()
    print("параметров BookLM:", n)
    assert n == EXPECTED_PARAMS, f"ожидали {EXPECTED_PARAMS}, получили {n}: проверьте версию кода booklm/"

    # Число параметров по частям: таблица токенов и шесть одинаковых блоков.
    table = cfg.vocab * cfg.d_model
    per_block = sum(p.numel() for p in model.blocks[0].parameters())
    assert table + cfg.layers * per_block + cfg.d_model == n

    # Словарь меньше вдвое -> параметров меньше на 4096 * 256 (задача 2 главы 1).
    smaller = BookLM(BookLMConfig(vocab=4096)).num_params()
    assert n - smaller == 4096 * cfg.d_model

    ids = torch.tensor([[1, 517, 2048, 33, 900]])
    with torch.no_grad():
        scores = model(ids)
    assert tuple(scores.shape) == (1, 5, cfg.vocab)
    assert torch.isfinite(scores).all()

    # Необученная модель почти не различает токены: оценки лидеров близки.
    top = scores[0, -1].topk(3).values
    assert float(top[0] - top[2]) < 0.5

    print("форма оценок:", tuple(scores.shape))
    check_weights()
    print("окружение готово")


def check_weights() -> None:
    """Обученные веса приходят через Git LFS. Без `git lfs pull` вместо них лежат текстовые указатели."""
    path = ROOT / "models" / "booklm" / "booklm-base.pt"
    head = path.read_bytes()[:64] if path.exists() else b""
    if not path.exists() or head.startswith(b"version https://git-lfs"):
        sys.exit("веса BookLM не скачаны: выполните в корне репозитория git lfs install и git lfs pull")
    state = torch.load(path, map_location="cpu")
    model = BookLM(BookLMConfig())
    model.load_state_dict(state)
    print(f"обученные веса: {path.relative_to(ROOT)}, {path.stat().st_size / 2**20:.1f} МБ")


if __name__ == "__main__":
    main()
