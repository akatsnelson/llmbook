"""Глава 1, раздел «Следующий токен и генерация текста»: обученная BookLM угадывает следующий токен
и пишет сказку по просьбе.

Опыты (время — на процессоре Apple M2 Max):
  ch01_question_next_token — рубрика «А теперь на нашей модели: следующий токен». Печатает, на какие
      токены режется вопрос «Сколько будет 17 умножить на 24?», пять самых вероятных следующих токенов
      после двух начал и разброс оценок у необученной и обученной модели. Около 2 с.
      Данные рисунка с оценками всех 8192 токенов: ch01-question-next-token.csv.
  ch01_booklm_chat — рубрика «А теперь на нашей модели: генерация текста». Инструктивная BookLM пишет
      сказку про кота, как в окне чата (температура 0,8, top-k 40, одно ядро), и печатает скорость
      в токенах в секунду. Около 2 с. Вывод только на экран: скорость зависит от машины.

Запуск из корня репозитория:
    python -m booklm.experiments.ch01                            # все опыты
    python -m booklm.experiments.ch01 ch01_question_next_token   # один опыт
    python -m booklm.experiments.ch01 --help                     # список опытов
Файлы выводов лежат в models/booklm/experiments/.
"""
import csv
import sys
import textwrap
import time
from pathlib import Path

import torch

if not __package__:  # запуск файлом (python booklm/experiments/ch01.py): корень репозитория — в путь поиска
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from booklm.experiments import run
from booklm.lab import MODELS, load, next_token_table, prompt_ids, special, tokens
from booklm.model import BookLM, BookLMConfig

OUT = MODELS / "experiments"


def ch01_booklm_chat():
    """Сказка инструктивной BookLM по просьбе, как в окне чата, и скорость генерации."""
    # То же, что делает окно чата serve_chat.py: модель «выполняет просьбы»,
    # температура 0.8, top-k 40, не больше 300 новых токенов. Зерно фиксируем.
    tok, model = load("instruct")
    torch.set_num_threads(1)                      # как на простом ноутбуке: одно ядро
    request = "Напиши сказку про кота, который боялся темноты."
    ids = prompt_ids(tok, request, chat=True)     # <bos><|user|>просьба<|assistant|>
    x = torch.tensor([ids])
    eos = special(tok)["<eos>"]
    with torch.no_grad():
        model.generate(x, max_new=20, eos_id=eos, seed=0)   # прогрев, время не считаем
        t0 = time.perf_counter()
        out = model.generate(x, max_new=300, temperature=0.8, top_k=40, eos_id=eos, seed=0)
        seconds = time.perf_counter() - t0
    new = [i for i in out[0, len(ids):].tolist() if i != eos]
    print("Просьба:", request)
    print("Ответ:")
    for line in tok.decode(new).splitlines():          # длинные строки переносим для книги
        print(textwrap.fill(line.strip(), 95) if line.strip() else "")
    print(f"{len(new)} токенов за {seconds:.2f} с, это {len(new) / seconds:.0f} токенов в секунду (CPU, одно ядро)")


def ch01_question_next_token():
    """Токены вопроса, следующий токен после двух начал и оценки до и после обучения."""
    tok, model = load("base")
    pieces = tokens(tok, "Сколько будет 17 умножить на 24?")
    print("Токены вопроса (номер и кусочек):")
    for row in range(0, len(pieces), 8):          # по восемь на строку
        print("  " + "  ".join(f"{i}:{p!r}" for i, p in pieces[row:row + 8]))
    print("Всего токенов:", len(pieces))
    left = next_token_table(tok, model, "17 умножить на 24 будет", k=5)
    right = next_token_table(tok, model, "Жила-была маленькая", k=5)
    print()
    print(f"{'после «17 умножить на 24 будет»':<34}после «Жила-была маленькая»")
    for (a, pa), (b, pb) in zip(left, right):
        print(f"  {a!r:<16}{pa:>6.2f}            {b!r:<16}{pb:>6.2f}")

    # Данные для рисунка: оценки всех 8192 токенов у необученной модели
    # (тот же листинг из главы: зерно 0 и пять номеров) и у обученной.
    torch.manual_seed(0)
    fresh = BookLM(BookLMConfig()).eval()
    with torch.no_grad():
        untrained = fresh(torch.tensor([[1, 517, 2048, 33, 900]]))[0, -1]
        trained = model(torch.tensor([prompt_ids(tok, "Жила-была маленькая")]))[0, -1]
    with open(OUT / "ch01-question-next-token.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["token_id", "untrained_score", "trained_score"])
        for i in range(len(trained)):
            w.writerow([i, round(untrained[i].item(), 4), round(trained[i].item(), 4)])
    for name, s in (("необученная", untrained), ("обученная", trained)):
        gap = s.max() - s.topk(2).values[1]
        print(f"{name}: оценки от {s.min():.2f} до {s.max():.2f}, лидер впереди второго на {gap:.2f}")


EXPERIMENTS = {"ch01_booklm_chat": ch01_booklm_chat,
               "ch01_question_next_token": ch01_question_next_token}

if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    run(EXPERIMENTS)
