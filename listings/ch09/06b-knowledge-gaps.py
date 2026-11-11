"""Листинги главы 9 «Обучение с подкреплением и оценка моделей» книги «Большие языковые модели», файл 06b-knowledge-gaps.

Код и вывод — ровно как в книге; текст книги сюда не входит. Листинги идут по порядку и
опираются друг на друга: запускайте файл целиком или ячейками сверху вниз (разметка # %%).
Проверка, что вывод совпал с книгой: python scripts/check_listings.py listings/ch09/06b-knowledge-gaps.py
"""

# %% Листинг «Угаданные слова и их частота в корпусе» — раздел «Длинный хвост корпуса»
from booklm.lab import load
from booklm.experiments.ch09_knowledge_gaps import (BINS, bin_label,
    val_words, word_counts)

tok, base = load("base")
cnt = word_counts(tok)          # слово -> сколько раз оно в train.bin
v = sorted(cnt.values(), reverse=True)
top, rare = sum(v[:1000]) / sum(v), sum(x < 10 for x in v)
print(f"разных слов {len(v)}, 1000 частых дают {top:.0%} текста, "
      f"реже 10 раз: {rare}")
rows = val_words(tok, base)     # (слово, угадано ли, ошибка) по val.bin
print("частота в train   слов   угадано   ошибка на слово")
for a, b in BINS:
    r = [x for x in rows if a <= cnt.get(x[0], 0) <= b]
    acc = sum(x[1] for x in r) / len(r)
    loss = sum(x[2] for x in r) / len(r)
    print(f"{bin_label(a, b):>13s} {len(r):8d} {acc:9.2f} {loss:12.2f}")
# Вывод в книге:
# | разных слов 59072, 1000 частых дают 78% текста, реже 10 раз: 39673
# | частота в train   слов   угадано   ошибка на слово
# |             0      286      0.08        15.19
# |           1-9      928      0.06        10.36
# |         10-99     4284      0.17         6.12
# |       100-999    13180      0.29         3.84
# |     1000-9999    24025      0.43         2.48
# |   10000-99999    26618      0.58         1.62
# |       100000+    15865      0.63         1.27

# %% Листинг «Доля ответов с героем просьбы и частота героя» — раздел «Поиск пробелов»
# Проверка этот листинг не запускает: ему нужны сеть, видеокарта, сервер или внешний файл.
# from booklm.experiments.ch09_knowledge_gaps import (ANIMALS, GROUPS, RARE_A,
#     RARE_B, hero_share, stories_with)
#
# _, inst = load("instruct")
# rows = {h: (stories_with(tok, h), hero_share(tok, inst, h, n=32))
#         for h in ANIMALS}         # герой -> (историй с ним, доля ответов)
# for name, lo, hi in GROUPS:            # частые, средние, редкие, почти нет
#     r = [s for d, s in rows.values() if lo <= d <= hi]
#     print(f"{name:24s} героев {len(r):2d}, доля {sum(r) / len(r):.2f}")
# for name, group in (("A", RARE_A), ("B", RARE_B)):    # историй:доля
#     cells = [f"{h} {rows[h][0]}:{rows[h][1]:.2f}" for h in group]
#     print(f"{name}: " + ", ".join(cells[:4]))
#     print("   " + ", ".join(cells[4:]))

# %% Листинг «Заплатка для редких героев» — раздел «Целевые данные»
# Проверка этот листинг не запускает: ему нужны сеть, видеокарта, сервер или внешний файл.
# from booklm.experiments.ch09_knowledge_gaps import (HEAD, ROW, gap_report,
#     lora_patch, old_pairs, substitute_pairs)
#
# target = substitute_pairs(tok, RARE_A)        # 7 героев по 40 историй
# mixed = target + old_pairs(3 * len(target))   # и по 3 старые на каждую
# print(HEAD)
# print(ROW.format("до заплатки", *gap_report(tok, inst)))
# for name, data in (("только целевые", target), ("целевые + старые", mixed)):
#     for seed in (0, 1, 2):
#         _, m = load("instruct")
#         res = gap_report(tok, lora_patch(tok, m, data, seed=seed))
#         print(ROW.format(f"{name}, зерно {seed}", *res))
