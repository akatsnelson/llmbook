"""Листинги главы 9 «Обучение с подкреплением и оценка моделей» книги «Большие языковые модели», файл 03b-search-verify.

Код и вывод — ровно как в книге; текст книги сюда не входит. Листинги идут по порядку и
опираются друг на друга: запускайте файл целиком или ячейками сверху вниз (разметка # %%).
Проверка, что вывод совпал с книгой: python scripts/check_listings.py listings/ch09/03b-search-verify.py
"""

# %% Листинг «Эволюционный поиск правила с точной оценкой» — раздел «Эволюционный поиск программ»
import random
from booklm.experiments.ch09_random_reward_loop import ap_free_max

def evolve(score, mutate, start, steps=2000, keep=20, seed=0):
    rng, pool = random.Random(seed), {start: score(start)}  # база
    for _ in range(steps):
        child = mutate(rng.choice(list(pool)), rng)    # генерация
        if child not in pool:
            pool[child] = score(child)                 # проверка
            best = sorted(pool, key=pool.get, reverse=True)[:keep]
            pool = {p: pool[p] for p in best}          # отбор
    return pool

N = 27
def build(expr):                   # жадно по убыванию приоритета
    f = eval("lambda x: " + (" + ".join(expr) or "0"))
    chosen = []
    for x in sorted(range(N), key=lambda x: (-f(x), x)):
        if all(2 * y - x not in chosen and 2 * x - y not in chosen
               and (x + y) / 2 not in chosen for y in chosen):
            chosen.append(x)       # x не замыкает прогрессию
    return sorted(chosen)

TERMS = ["x", "x % {k}", "x // {k}", "x * {k} % {m}", "bin(x).count('1')"]
def mutate(expr, rng):             # в FunSearch пишет языковая модель
    term = f"{rng.choice([-3, -2, -1, 1, 2, 3])} * ({rng.choice(TERMS)})"
    expr = list(expr)
    if expr and (len(expr) == 3 or rng.random() < 0.5):
        expr.pop(rng.randrange(len(expr)))       # заменить слагаемое
    return tuple(expr + [term.format(k=rng.randint(2, 9),
                                     m=rng.randint(5, 31))])

score = lambda expr: len(build(expr))
pool = evolve(score, mutate, start=())
best = max(pool, key=pool.get)
print(f"по порядку: {score(())} чисел {build(())}")
print(f"поиск:      {pool[best]} чисел {build(best)}")
print("приоритет: ", " + ".join(best))
print("точный максимум перебором:", ap_free_max(N))
# Вывод в книге:
# | по порядку: 8 чисел [0, 1, 3, 4, 9, 10, 12, 13]
# | поиск:      11 чисел [1, 3, 4, 8, 9, 11, 16, 20, 22, 25, 26]
# | приоритет:  2 * (x // 4) + 2 * (x * 6 % 10) + -2 * (x)
# | точный максимум перебором: 11

# %% Листинг «Поиск находит дыру в слабом проверяющем» — раздел «Эволюционный поиск программ»
from collections import Counter

PIECES = ["sorted(x)", "x[::-1]", "x[1:]", "x[:1]", "[]"]
def mutate_code(code, rng):        # подставить программу в кусок
    return rng.choice(PIECES).replace("x", code)

TESTS = [[3, 1, 2], [5, 5, 1], [2, 9, 4, 4, 0], [7, 6]]
def ordered(out, x):               # выход упорядочен
    return all(a <= b for a, b in zip(out, out[1:]))
def ordered_same(out, x):          # и числа те же, что на входе
    return ordered(out, x) and Counter(out) == Counter(x)

def score_with(check):             # доля тестов минус штраф за длину
    def score(code):
        f = eval("lambda x: " + code)
        return (sum(check(f(x), x) for x in TESTS) / len(TESTS)
                - 0.001 * len(code))
    return score

for check in (ordered, ordered_same):
    pool = evolve(score_with(check), mutate_code, start="x", steps=300,
                  seed=1)
    top = sorted(pool, key=pool.get, reverse=True)[:2]
    print(f"{check.__name__:12s}",
          "   ".join(f"{c} ({pool[c]:.3f})" for c in top))
# Вывод в книге:
# | ordered      [] (0.998)   x[:1] (0.995)
# | ordered_same sorted(x) (0.991)   sorted(x[::-1]) (0.985)

# %% Листинг «Петля отбора на BookLM по раундам» — раздел «Открытые математические задачи»
import csv
from booklm.lab import MODELS

path = MODELS / "experiments" / "ch09-search-loop.csv"
print("проверяющий раунд отобрано pass@1 pass@8 длина")
for r in csv.DictReader(open(path)):
    print(f"{r['checker']:11s} {r['round']:>5s} {int(r['kept']):8d}"
          f" {float(r['strict1']):6.2f} {float(r['strict8']):6.2f}"
          f" {float(r['len']):5.0f}")
# Вывод в книге:
# | проверяющий раунд отобрано pass@1 pass@8 длина
# | слабый          0        0   0.05   0.25   181
# | слабый          1       29   0.10   0.62   190
# | слабый          2       67   0.13   0.69   177
# | слабый          3       78   0.12   0.62   192
# | строгий         0        0   0.05   0.25   181
# | строгий         1       28   0.10   0.62   181
# | строгий         2       50   0.20   0.75   189
# | строгий         3       93   0.12   0.69   185
