"""Листинги главы 9 «Обучение с подкреплением и оценка моделей» книги «Большие языковые модели», файл 06-calibration.

Код и вывод — ровно как в книге; текст книги сюда не входит. Листинги идут по порядку и
опираются друг на друга: запускайте файл целиком или ячейками сверху вниз (разметка # %%).
Проверка, что вывод совпал с книгой: python scripts/check_listings.py listings/ch09/06-calibration.py
"""

# %% Листинг «Уверенность и точность BookLM на именах героев» — раздел «Ожидаемая ошибка калибровки»
from collections import Counter
from booklm.lab import load
from booklm.experiments.ch09_eval import (
    calibration_table, name_cuts, name_guess)

tok, base = load("base")
_, inst = load("instruct")
models = {"база": base, "после LoRA": inst}
seen, first = name_cuts(tok)
rows, tables = {}, {}
for name, m in models.items():
    rows[name] = []
    for group, cuts in (("имя было", seen), ("имя новое", first)):
        r = [name_guess(tok, m, before, hero) for before, hero in cuts]
        rows[name] += [(conf, ok) for conf, ok, guess in r]
        top = Counter(guess for _, _, guess in r).most_common(2)
        print(f"{name:10s} {group:9s} {len(r):3d} шт., увер. "
              f"{sum(x[0] for x in r) / len(r):.2f}, точн. "
              f"{sum(x[1] for x in r) / len(r):.2f}, чаще:"
              + ",".join(f"{g} {k}" for g, k in top))
    tables[name] = calibration_table(rows[name])
print("корзина    база: n  увер.  точн.   после LoRA: n  увер.  точн.")
for (b, n1, c1, a1), (_, n2, c2, a2) in zip(tables["база"][0],
                                             tables["после LoRA"][0]):
    print(f"{b:9s} {n1:8d} {c1:6.2f} {a1:6.2f} {n2:15d} {c2:6.2f} {a2:6.2f}")
print(f"ECE {tables['база'][1]:24.3f} {tables['после LoRA'][1]:29.3f}")
# Вывод в книге:
# | база       имя было  127 шт., увер. 0.67, точн. 0.72, чаще: Тим 22, Том 10
# | база       имя новое 379 шт., увер. 0.47, точн. 0.50, чаще: Тим 140, Лили 118
# | после LoRA имя было  127 шт., увер. 0.68, точн. 0.70, чаще: Тим 21, Том 10
# | после LoRA имя новое 379 шт., увер. 0.50, точн. 0.45, чаще: Тим 142, Лили 94
# | корзина    база: n  увер.  точн.   после LoRA: n  увер.  точн.
# | 0.0-0.2         27   0.15   0.30              23   0.14   0.22
# | 0.2-0.4        191   0.31   0.32             172   0.30   0.30
# | 0.4-0.6         94   0.50   0.61             123   0.51   0.50
# | 0.6-0.8         90   0.66   0.62              73   0.73   0.55
# | 0.8-1.0        104   0.90   0.92             115   0.91   0.89
# | ECE                    0.047                         0.038

# %% Листинг «Баллы BookLM при трёх правилах оценки» — раздел «Правило оценки и отказ от ответа»
def exam_score(rows, t, penalty):
    """Отвечаем при уверенности >= t, иначе «не знаю» за 0 баллов."""
    s = sum(1 if ok else -penalty for conf, ok in rows if conf >= t)
    return s / len(rows)

r = rows["база"]
print("отвечаем при   доля ответов  0/1   штраф 1   штраф 3")
for t in (0.0, 0.5, 0.75):
    share = sum(conf >= t for conf, _ in r) / len(r)
    print(f"увер. >= {t:4.2f} {share:12.2f} {exam_score(r, t, 0):6.2f} "
          f"{exam_score(r, t, 1):9.2f} {exam_score(r, t, 3):9.2f}")
# Вывод в книге:
# | отвечаем при   доля ответов  0/1   штраф 1   штраф 3
# | увер. >= 0.00         1.00   0.55      0.10     -0.79
# | увер. >= 0.50         0.48   0.37      0.26      0.04
# | увер. >= 0.75         0.23   0.21      0.18      0.14
