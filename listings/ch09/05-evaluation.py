"""Листинги главы 9 «Обучение с подкреплением и оценка моделей» книги «Большие языковые модели», файл 05-evaluation.

Код и вывод — ровно как в книге; текст книги сюда не входит. Листинги идут по порядку и
опираются друг на друга: запускайте файл целиком или ячейками сверху вниз (разметка # %%).
Проверка, что вывод совпал с книгой: python scripts/check_listings.py listings/ch09/05-evaluation.py
"""

# %% Листинг «Тест «Сказки-100» для базы и инструктивной BookLM» — раздел «Тесты с вариантами ответа»
from collections import Counter
from booklm.lab import load
from booklm.experiments.ch09_eval import (
    cloze_answers, fairy_exam, generated_letter, letter_choice)

tok, base = load("base")
_, inst = load("instruct")
models = {"база": base, "после LoRA": inst}

items, rest = fairy_exam(tok)          # 100 вопросов, по 4 варианта
gold = [it["answer"] for it in items]
right = {}
for name, m in models.items():
    chat = name != "база"
    by_sum, by_mean = cloze_answers(tok, m, items)
    right[name] = [a == g for a, g in zip(by_mean, gold)]
    by_letter, picks = [], Counter()
    for shift in range(4):             # сдвигаем варианты по кругу
        n_ok = 0
        for it in items:
            order = [(j + shift) % 4 for j in range(4)]
            k, mass, ids = letter_choice(tok, m, it, order, chat)
            n_ok += order[k] == it["answer"]
            picks["АБВГ"[k]] += 1
        by_letter.append(n_ok)
    texts = []                         # ответ текстом, порядок А-Г
    for it in items:
        ids = letter_choice(tok, m, it, range(4), chat)[2]
        texts.append(generated_letter(tok, m, ids))
    (top, k), = picks.most_common(1)
    print(f"{name}: по сумме {sum(a == g for a, g in zip(by_sum, gold))}, "
          f"на токен {sum(right[name])}, по букве "
          + " ".join(map(str, by_letter)) + f", «{top}» {k} раз из 400")
    print(f"   букв в тексте: {sum(g is not None for g, _ in texts)}, "
          f"пример: {texts[0][1]!r}")
# Вывод в книге:
# | база: по сумме 59, на токен 69, по букве 26 21 31 21, «В» 397 раз из 400
# |    букв в тексте: 0, пример: 'Я никогда не забуду тебя.'
# | после LoRA: по сумме 58, на токен 68, по букве 28 29 21 26, «А» 265 раз из 400
# |    букв в тексте: 0, пример: 'Жил-был маленький мальчик по имени Флюф'

# %% Листинг «Интервалы для баллов BookLM на «Сказках-100»» — раздел «Доверительный интервал»
import random

n = len(items)
for name, r in right.items():
    f = sum(r) / n
    print(f"{name}: {f:.2f} ± {1.96 * (f * (1 - f) / n) ** 0.5:.2f}")
d = [int(a) - int(b) for a, b in zip(right["после LoRA"], right["база"])]
rng = random.Random(0)
boot = sorted(sum(rng.choice(d) for _ in range(n)) / n
              for _ in range(10_000))
print(f"разность {sum(d) / n:+.2f}, бутстреп от {boot[250]:+.2f} "
      f"до {boot[9750]:+.2f}; модели разошлись на {sum(x != 0 for x in d)}")
# Вывод в книге:
# | база: 0.69 ± 0.09
# | после LoRA: 0.68 ± 0.09
# | разность -0.01, бутстреп от -0.05 до +0.03; модели разошлись на 5

# %% Листинг «Загрязнение «Сказок-100»» — раздел «Загрязнение теста»
from booklm.lab import encode_text
from booklm.experiments.ch09_eval import (contaminate, in_train,
    mean_story_loss, ngram_hashes, other_stories, story_ids, train_ngrams)

grams = train_ngrams(13)                 # хеши всех 13-грамм train.bin
hits = [in_train(ngram_hashes(story_ids(tok, it)), grams) for it in items]
ends = [in_train(ngram_hashes(encode_text(
            tok, " " + it["options"][it["answer"]])), grams) for it in items]
print(f"общая 13-грамма с train.bin: в {sum(hits)} вопросах, "
      f"из них в концовке: {sum(ends)}")
dirty = [story_ids(tok, it) for it in items[:50]]   # эти подмешаем
others = other_stories(tok, rest)        # 100 историй не из теста
for repeats in (None, 0, 1, 4):          # None: база без дообучения
    m = base if repeats is None else contaminate(tok, base, dirty, repeats)
    ok = [a == g for a, g in zip(cloze_answers(tok, m, items)[1], gold)]
    label = "база" if repeats is None else f"подмешано {repeats} раз"
    print(f"{label:16s} подмешанные {sum(ok[:50])}/50, чистые "
          f"{sum(ok[50:])}/50, ошибка на других {mean_story_loss(m, others):.3f}")
# Вывод в книге:
# | общая 13-грамма с train.bin: в 53 вопросах, из них в концовке: 2
# | база             подмешанные 34/50, чистые 35/50, ошибка на других 1.901
# | подмешано 0 раз  подмешанные 34/50, чистые 35/50, ошибка на других 1.914
# | подмешано 1 раз  подмешанные 37/50, чистые 35/50, ошибка на других 1.910
# | подмешано 4 раз  подмешанные 45/50, чистые 35/50, ошибка на других 1.920
