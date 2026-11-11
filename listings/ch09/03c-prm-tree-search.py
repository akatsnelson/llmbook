"""Листинги главы 9 «Обучение с подкреплением и оценка моделей» книги «Большие языковые модели», файл 03c-prm-tree-search.

Код и вывод — ровно как в книге; текст книги сюда не входит. Листинги идут по порядку и
опираются друг на друга: запускайте файл целиком или ячейками сверху вниз (разметка # %%).
Проверка, что вывод совпал с книгой: python scripts/check_listings.py listings/ch09/03c-prm-tree-search.py
"""

# %% Листинг «Метки шагов продолжениями на задаче из восьми операций» — раздел «Оценка шагов продолжениями»
import random

def apply(x, op):                       # одна операция по модулю 1000
    sign, a = op
    return (x + a if sign == "+" else x - a if sign == "-" else x * a) % 1000

def step(x, op, rng, e):                # шаг решающего: иногда лишний перенос
    y = apply(x, op)
    return (y + 10) % 1000 if rng.random() < e else y

def solve(x, ops, rng, e):              # довести решение до конца
    for op in ops:
        x = step(x, op, rng, e)
    return x

def mc_value(x, ops_left, answer, rng, e, n=8):   # доля верных продолжений
    return sum(solve(x, ops_left, rng, e) == answer for _ in range(n)) / n

x0, e = 37, 0.3
ops = [("+", 5), ("*", 3), ("-", 8), ("*", 2),
       ("+", 9), ("*", 4), ("-", 6), ("+", 7)]
rng = random.Random(20)
truth, xs = [x0], [x0]
for op in ops:                          # верное решение и решение модели
    truth.append(apply(truth[-1], op))
    xs.append(step(xs[-1], op, rng, e))
for i, op in enumerate(ops):
    v = mc_value(xs[i + 1], ops[i + 1:], truth[-1], rng, e)
    exact = (1 - e) ** (len(ops) - i - 1) if xs[i + 1] == truth[i + 1] else 0
    mark = "верно " if xs[i + 1] == apply(xs[i], op) else "ошибка"
    print(f"шаг {i + 1}: {xs[i]:3d} {op[0]} {op[1]} = {xs[i + 1]:3d} {mark}"
          f"  оценка {v:.3f}  точно {exact:.3f}")
# Вывод в книге:
# | шаг 1:  37 + 5 =  42 верно   оценка 0.000  точно 0.082
# | шаг 2:  42 * 3 = 126 верно   оценка 0.000  точно 0.118
# | шаг 3: 126 - 8 = 118 верно   оценка 0.375  точно 0.168
# | шаг 4: 118 * 2 = 236 верно   оценка 0.000  точно 0.240
# | шаг 5: 236 + 9 = 255 ошибка  оценка 0.000  точно 0.000
# | шаг 6: 255 * 4 =  20 верно   оценка 0.000  точно 0.000
# | шаг 7:  20 - 6 =  14 верно   оценка 0.000  точно 0.000
# | шаг 8:  14 + 7 =  21 верно   оценка 0.000  точно 0.000

# %% Листинг «Проверяющий шага и проверяющий исхода» — раздел «Выбор из N и лучевой поиск по шагам»
def prm(x, op, y, rng, ej=0.1):         # проверка шага: пересчитать его самому
    return float(step(x, op, rng, ej) == y)

def orm(x0, ops, final, rng, ej=0.1):   # проверка исхода: пересчитать всё
    return float(solve(x0, ops, rng, ej) == final)

rng = random.Random(0)
ok = sum(prm(x0, ops[0], 42, rng) for _ in range(10000)) / 10000
bad = sum(prm(x0, ops[0], 52, rng) for _ in range(10000)) / 10000
print(f"PRM: верный шаг принят в {ok:.2f}, ошибочный в {bad:.2f}")
ok = sum(orm(x0, ops, truth[-1], rng) for _ in range(10000)) / 10000
print(f"ORM: верное решение принято в {ok:.2f}")
# Вывод в книге:
# | PRM: верный шаг принят в 0.90, ошибочный в 0.10
# | ORM: верное решение принято в 0.43

# %% Листинг «Взвешенный выбор лучшего из N» — раздел «Выбор из N и лучевой поиск по шагам»
from collections import defaultdict

def pick(finals, scores, rng):          # сумма наград по каждому ответу
    total = defaultdict(float)
    for f, s in zip(finals, scores):
        total[f] += s + 1e-6 * rng.random()        # ничьи --- жребием
    return max(total, key=total.get)

def best_of_n(x0, ops, N, rng, e, judge):
    sols = []
    for _ in range(N):                  # N независимых решений
        xs = [x0]
        for op in ops:
            xs.append(step(xs[-1], op, rng, e))
        sols.append(xs)
    finals = [xs[-1] for xs in sols]
    if judge == "голоса":
        scores = [1.0] * N
    elif judge == "ORM":
        scores = [orm(x0, ops, f, rng) for f in finals]
    else:                               # PRM: приняты ли все шаги
        scores = [min(prm(xs[i], op, xs[i + 1], rng)
                      for i, op in enumerate(ops)) for xs in sols]
    return pick(finals, scores, rng)

# %% Листинг «Лучевой поиск по шагам с PRM» — раздел «Выбор из N и лучевой поиск по шагам»
def beam(x0, ops, N, rng, e, M=4):      # N/M лучей, у каждого M продолжений
    beams = [(1.0, x0)] * (N // M)      # (все ли шаги приняты, число)
    for op in ops:
        cand = []
        for ok, x in beams:
            ys = [step(x, op, rng, e) for _ in range(M)]   # M продолжений
            cand += [(min(ok, prm(x, op, y, rng)), rng.random(), y) for y in ys]
        cand.sort(reverse=True)         # принятые PRM вперёд, ничьи --- жребием
        beams = [(ok, y) for ok, _, y in cand[:N // M]]
    return pick([y for *_, y in cand], [ok for ok, *_ in cand], rng)

# %% Листинг «Выбор из N и лучевой поиск при одном бюджете» — раздел «Выбор из N и лучевой поиск по шагам»
rng = random.Random(1)
tasks = [(rng.randint(10, 99), [(rng.choice("+-*"), rng.randint(2, 9))
                                for _ in range(8)]) for _ in range(500)]
for e in (0.1, 0.3):
    print(f"e = {e}:  N  голоса   ORM   PRM  лучи PRM")
    for N in (4, 16, 64):
        hits = [0, 0, 0, 0]
        for x0, ops in tasks:
            answer = solve(x0, ops, rng, 0.0)
            for j, judge in enumerate(("голоса", "ORM", "PRM")):
                hits[j] += best_of_n(x0, ops, N, rng, e, judge) == answer
            hits[3] += beam(x0, ops, N, rng, e) == answer
        cells = "".join(f"{h / len(tasks):6.2f}" for h in hits)
        print(f"        {N:2d}" + cells)
# Вывод в книге:
# | e = 0.1:  N  голоса   ORM   PRM  лучи PRM
# |          4  0.61  0.65  0.69  0.91
# |         16  0.84  0.87  0.97  0.94
# |         64  0.95  0.97  1.00  0.99
# | e = 0.3:  N  голоса   ORM   PRM  лучи PRM
# |          4  0.06  0.13  0.13  0.65
# |         16  0.07  0.22  0.30  0.77
# |         64  0.07  0.40  0.67  0.93
