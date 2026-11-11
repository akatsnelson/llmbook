"""Глава 18 «Безопасность LLM-приложений», рубрика «сигнатура пересобранной нагрузки» (подраздел «Модель
как инструмент нарушителя»): опыт ch18_regenerated_payload. Модель на стороне нарушителя пересобирает
полезную нагрузку при каждом запуске, и сигнатура по байтам перестаёт её узнавать.

Нарушитель просит инструктивную BookLM сто раз написать «сказку, где кот украл торт» (замена
полезной нагрузки: смысл держится, байты каждый раз новые). Защитник хранит сигнатуру первого образца:
SHA-256 текста и множество четвёрок слов подряд. Считаем, сколько из 99 остальных вариантов и из 100
честных сказок про кота узнаёт каждая сигнатура и в скольких есть «торт».

Запуск из корня репозитория (около 35 с на процессоре Apple M2 Max):
    python -m booklm.experiments.ch18_payload_mutation
Вывод копируется в models/booklm/experiments/ch18-regenerated-payload.txt.
"""
import hashlib
import sys
from pathlib import Path

if not __package__:  # запуск файлом (python booklm/experiments/ch18_payload_mutation.py): корень репозитория — в путь поиска
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from booklm.lab import MODELS, generate, load

OUT = MODELS / "experiments"


def words4(text):
    """Множество четвёрок слов подряд (нижний регистр, точки и запятые убраны)."""
    w = text.lower().replace(".", " ").replace(",", " ").split()
    return {tuple(w[i:i + 4]) for i in range(len(w) - 3)}


def ch18_regenerated_payload():
    """Рубрика «сигнатура пересобранной нагрузки»: хеш и четвёрки слов против ста пересказов."""
    tok, model = load("instruct")
    bad = [generate(tok, model, "Расскажи сказку, где кот украл торт.", chat=True, seed=s)
           for s in range(100)]
    good = [generate(tok, model, "Расскажи сказку про кота.", chat=True, seed=1000 + s)
            for s in range(100)]
    sig_hash = hashlib.sha256(bad[0].encode()).hexdigest()
    sig_words = words4(bad[0])
    print("образец 0:", bad[0][:160].replace("\n", " "), "...")
    for name, texts in [("варианты", bad[1:]), ("честные", good)]:
        by_hash = sum(hashlib.sha256(t.encode()).hexdigest() == sig_hash for t in texts)
        by_words = sum(bool(words4(t) & sig_words) for t in texts)
        cake = sum("торт" in t.lower() for t in texts)
        theft = sum("укра" in t.lower() or "крад" in t.lower() for t in texts)
        print(f"{name:8} {len(texts):3}: хеш {by_hash}, четыре слова {by_words:2}, "
              f"«торт» {cake:2}, кража {theft:2}")
    distinct = len({hashlib.sha256(t.encode()).hexdigest() for t in bad})
    print("разных текстов среди 100 вариантов:", distinct)
    shared = [sorted(words4(t) & sig_words) for t in good if words4(t) & sig_words]
    print("общие четвёрки у честных (первые 5):", [" ".join(g[0]) for g in shared[:5]])


EXPERIMENTS = {"ch18_regenerated_payload": ch18_regenerated_payload}

if __name__ == "__main__":
    from booklm.experiments import run
    run(EXPERIMENTS, save=True)      # копия вывода в models/booklm/experiments/<имя опыта>.txt
