"""Глава 9, раздел 9.4 «Награда за процесс и поиск по дереву»: полная версия опыта.

Игрушечная задача с проверяемым ответом и шагами: начальное число и восемь операций вида
«+5», «*3», «-8» по модулю 1000. Решающий («модель») делает каждый шаг верно, но с
вероятностью e делает лишний перенос и пишет результат на 10 больше. Проверяющий шага (PRM) сам
пересчитывает один шаг и с вероятностью ej ошибается так же; проверяющий исхода (ORM)
пересчитывает всё решение и сравнивает итог.

Функции повторяют листинги книги (здесь у них есть параметр ej), а run_table(seed=1) печатает ту же таблицу, что
листинг «Выбор из N и лучевой поиск при одном бюджете». Файл делает:
  1. разметку шагов продолжениями (Math-Shepherd) против точного значения (1-e)^(осталось шагов);
  2. точность PRM на шаге и ORM на решении;
  3. таблицу «голоса, ORM, PRM, лучи PRM» по пяти зёрнам: среднее и разброс;
  4. ту же таблицу с безошибочным PRM (ej = 0): починка той детали, которой приписана поломка;
  5. поиск по дереву методом Монте-Карло (MCTS) по шагам при том же бюджете шагов решающего.

Запуск из корня репозитория: python labs/ch09/prm_search.py
"""

import math
import random
import statistics
from collections import defaultdict


# ----------------------- задача и решающий (листинг «Метки шагов продолжениями на задаче из восьми операций»)
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


X0, E = 37, 0.3
OPS = [("+", 5), ("*", 3), ("-", 8), ("*", 2), ("+", 9), ("*", 4), ("-", 6), ("+", 7)]


def truth_of(x0, ops):
    truth = [x0]
    for op in ops:
        truth.append(apply(truth[-1], op))
    return truth


def check_mc_labels():
    truth = truth_of(X0, OPS)
    assert truth[-1] == 981
    rng = random.Random(20)                       # то же зерно, что в листинге книги
    xs = [X0]
    for op in OPS:
        xs.append(step(xs[-1], op, rng, E))
    est = [mc_value(xs[i + 1], OPS[i + 1:], truth[-1], rng, E) for i in range(len(OPS))]
    assert xs[1:] == [42, 126, 118, 236, 255, 20, 14, 21]   # ошибка на шаге 5
    assert est == [0.0, 0.0, 0.375, 0.0, 0.0, 0.0, 0.0, 0.0]
    # Для верного префикса точное значение равно (1-e)^(осталось шагов): продолжение верно,
    # только если не ошиблось ни разу (лишние +10 последующие шаги не исправляют).
    rng = random.Random(0)
    for i in range(len(OPS) + 1):
        exact = (1 - E) ** (len(OPS) - i)
        big = mc_value(truth[i], OPS[i:], truth[-1], rng, E, n=20000)
        assert abs(big - exact) < 0.015, (i, big, exact)
    # После ошибки значение ноль, даже если дальше все шаги посчитаны верно.
    assert mc_value(255, OPS[5:], truth[-1], rng, E, n=2000) == 0.0
    # Жёсткая метка Math-Shepherd: «годен», если хоть одно из 8 продолжений верно.
    # У верного первого шага P = 1 - (1 - 0.7^7)^8 ~ 0.50: половина таких шагов — «негодные».
    v = (1 - E) ** (len(OPS) - 1)
    p_hard = 1 - (1 - v) ** 8
    hits = sum(mc_value(42, OPS[1:], truth[-1], rng, E) > 0 for _ in range(4000)) / 4000
    assert abs(hits - p_hard) < 0.03, (hits, p_hard)
    assert f"{v:.3f} {p_hard:.2f}" == "0.082 0.50"   # числа текста: 0,7^7 и половина верных шагов
    print(f"1. разметка продолжениями: точное значение первого шага {v:.3f}; "
          f"жёсткая метка «годен» в {hits:.2f} случаев (по формуле {p_hard:.2f})")


# ---------------------------------------------- проверяющие (листинг «Проверяющий шага и проверяющий исхода»)
def prm(x, op, y, rng, ej=0.1):         # проверка шага: пересчитать его самому
    return float(step(x, op, rng, ej) == y)


def orm(x0, ops, final, rng, ej=0.1):   # проверка исхода: пересчитать всё
    return float(solve(x0, ops, rng, ej) == final)


def check_judges():
    rng = random.Random(0)
    ok = sum(prm(X0, OPS[0], 42, rng) for _ in range(10000)) / 10000
    bad = sum(prm(X0, OPS[0], 52, rng) for _ in range(10000)) / 10000
    assert f"{ok:.2f} {bad:.2f}" == "0.90 0.10"
    ok_out = sum(orm(X0, OPS, 981, rng) for _ in range(10000)) / 10000
    assert f"{ok_out:.2f}" == "0.43" and abs(ok_out - 0.9 ** 8) < 0.02
    # ORM изредка принимает и неверное решение: судья ошибается так, что приходит к тому же неверному итогу
    wrong = [w for w in (solve(X0, OPS, rng, 0.3) for _ in range(20000)) if w != 981]
    fp = sum(orm(X0, OPS, w, rng) for w in wrong) / len(wrong)
    assert 0.0 < fp < 0.1
    assert f"{fp:.3f}" == "0.034"                 # текст: «неверное решение ORM принимает в 3,4 %»
    print(f"2. PRM: верный шаг принят в {ok:.2f}, ошибочный в {bad:.2f}; ORM: верное решение "
          f"принято в {ok_out:.2f}, неверное в {fp:.3f}")


def check_mode():
    """Почему голоса не помогают на трудных задачах: лишний перенос в любом из идущих подряд
    сложений и вычитаний даёт один и тот же неверный итог, и он выпадает чаще верного."""
    rng = random.Random(1)
    tasks = [(rng.randint(10, 99), [(rng.choice("+-*"), rng.randint(2, 9)) for _ in range(8)])
             for _ in range(200)]
    rng = random.Random(5)
    p_true, p_wrong, mode_ok = [], [], 0
    for x0, ops in tasks:
        answer = solve(x0, ops, rng, 0.0)
        counts = defaultdict(int)
        for _ in range(4000):
            counts[solve(x0, ops, rng, 0.3)] += 1
        top_wrong = max(v for k, v in counts.items() if k != answer)
        p_true.append(counts[answer] / 4000)
        p_wrong.append(top_wrong / 4000)
        mode_ok += counts[answer] > top_wrong
    print(f"   e = 0.3: верный итог в среднем {statistics.mean(p_true):.3f}, самый частый неверный "
          f"{statistics.mean(p_wrong):.3f}; верный --- самый частый в {mode_ok / len(tasks):.2f} задач")
    assert statistics.mean(p_wrong) > 2 * statistics.mean(p_true) and mode_ok / len(tasks) < 0.15
    assert f"{mode_ok / len(tasks):.2f}" == "0.09"  # текст: «самый частый лишь в 9 % трудных задач»


# -------------------------- поиск (листинги «Взвешенный выбор лучшего из N» и «Лучевой поиск по шагам с PRM»)
def pick(finals, scores, rng):          # сумма наград по каждому ответу
    total = defaultdict(float)
    for f, s in zip(finals, scores):
        total[f] += s + 1e-6 * rng.random()        # ничьи — жребием
    return max(total, key=total.get)


def best_of_n(x0, ops, N, rng, e, judge, ej=0.1):
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
        scores = [orm(x0, ops, f, rng, ej) for f in finals]
    else:                               # PRM: приняты ли все шаги
        scores = [min(prm(xs[i], op, xs[i + 1], rng, ej) for i, op in enumerate(ops))
                  for xs in sols]
    return pick(finals, scores, rng)


def beam(x0, ops, N, rng, e, M=4, ej=0.1):   # N/M лучей, у каждого M продолжений
    beams = [(1.0, x0)] * (N // M)      # (все ли шаги приняты, число)
    calls = 0
    for op in ops:
        cand = []
        for ok, x in beams:
            ys = [step(x, op, rng, e) for _ in range(M)]   # M продолжений
            cand += [(min(ok, prm(x, op, y, rng, ej)), rng.random(), y) for y in ys]
        calls += len(cand)
        cand.sort(reverse=True)         # принятые PRM вперёд, ничьи — жребием
        beams = [(ok, y) for ok, _, y in cand[:N // M]]
    assert calls == N * len(ops)        # столько же шагов решающего и PRM, сколько у N решений
    return pick([y for *_, y in cand], [ok for ok, *_ in cand], rng)


METHODS = ("голоса", "ORM", "PRM", "лучи PRM")


def run_table(seed=1, ej=0.1, n_tasks=500, Ns=(4, 16, 64), es=(0.1, 0.3)):
    """Тот же порядок вызовов генератора, что в листинге книги (при ej = 0.1 и seed = 1)."""
    rng = random.Random(seed)
    tasks = [(rng.randint(10, 99), [(rng.choice("+-*"), rng.randint(2, 9))
                                    for _ in range(8)]) for _ in range(n_tasks)]
    table = {}
    for e in es:
        for N in Ns:
            hits = [0, 0, 0, 0]
            for x0, ops in tasks:
                answer = solve(x0, ops, rng, 0.0)
                for j, judge in enumerate(METHODS[:3]):
                    hits[j] += best_of_n(x0, ops, N, rng, e, judge, ej) == answer
                hits[3] += beam(x0, ops, N, rng, e, ej=ej) == answer
            table[e, N] = [h / len(tasks) for h in hits]
    return table


BOOK = {(0.1, 4): "0.61 0.65 0.69 0.91", (0.1, 16): "0.84 0.87 0.97 0.94",
        (0.1, 64): "0.95 0.97 1.00 0.99", (0.3, 4): "0.06 0.13 0.13 0.65",
        (0.3, 16): "0.07 0.22 0.30 0.77", (0.3, 64): "0.07 0.40 0.67 0.93"}


def check_search():
    tables = [run_table(seed) for seed in range(1, 6)]
    for key, row in BOOK.items():                 # зерно 1 — таблица из книги
        assert " ".join(f"{v:.2f}" for v in tables[0][key]) == row, (key, tables[0][key])
    print("3. пять зёрен, по 500 задач: среднее ± разброс")
    mean, spread = {}, []
    for key in BOOK:
        cells = [(statistics.mean(t[key][j] for t in tables),
                  statistics.stdev(t[key][j] for t in tables)) for j in range(4)]
        mean[key] = [m for m, _ in cells]
        spread += [s for _, s in cells]
        print(f"   e = {key[0]}, N = {key[1]:2d}: "
              + "  ".join(f"{name} {m:.2f} ± {s:.2f}" for name, (m, s) in zip(METHODS, cells)))
    # Утверждения, на которые опирается текст:
    # трудные задачи: при N = 4 лучи с PRM намного лучше любого выбора из N
    assert all(mean[0.3, 4][3] > mean[0.3, 4][j] + 0.4 for j in range(3))
    # там же голоса не растут с N: самый частый ответ неверен
    assert max(mean[0.3, N][0] for N in (4, 16, 64)) < 0.12
    # ORM уступает PRM при одном бюджете проверки
    assert all(mean[e, N][1] <= mean[e, N][2] for e in (0.1, 0.3) for N in (16, 64))
    # лёгкие задачи: при N = 16 выбор из N по PRM лучше лучей во всех пяти зёрнах
    assert all(t[0.1, 16][2] > t[0.1, 16][3] for t in tables)
    # числа текста: разброс клеток не больше ±0,03; при e = 0,3 и N = 4 лучи 0,65 против 0,13
    # у выбора по ORM и PRM; при e = 0,1 и N = 16 выбор по PRM 0,97 против 0,94 у лучей
    assert f"{max(spread):.2f}" == "0.03", max(spread)
    t1 = tables[0]
    assert f"{t1[0.3, 4][3]:.2f} {t1[0.3, 4][1]:.2f} {t1[0.3, 4][2]:.2f}" == "0.65 0.13 0.13"
    assert f"{t1[0.1, 16][2]:.2f} {t1[0.1, 16][3]:.2f}" == "0.97 0.94"
    return tables


def check_perfect_prm():
    t = run_table(seed=1, ej=0.0)
    print("4. безошибочный PRM (ej = 0), зерно 1: PRM и лучи PRM")
    for key in BOOK:
        pm, bm = t[key][2], t[key][3]
        print(f"   e = {key[0]}, N = {key[1]:2d}: PRM {pm:.2f}, лучи PRM {bm:.2f}")
        assert bm >= pm                           # без ошибок проверяющего лучи не проигрывают
    assert t[0.1, 16][3] > 0.99 and t[0.1, 16][2] > 0.99
    assert f"{t[0.3, 4][3]:.2f} {t[0.3, 4][2]:.2f}" == "0.94 0.21"   # текст: «0,94 задач против 0,21»
    # 0,21 — это и доля задач, где среди четырёх решений есть верное: 1 - (1 - 0,7^8)^4
    assert f"{1 - (1 - 0.7 ** 8) ** 4:.2f}" == "0.21"


# ------------------------------------------------------------------ MCTS по шагам
def mcts(x0, ops, N, rng, e, M=4, c=0.5, ej=0.1):
    """MCTS с бюджетом N*K симуляций, в каждой не больше одного нового шага решающего.

    Узел — (глубина, число): решающему важно только текущее число. Новый ребёнок
    появляется, пока попыток расширения меньше min(M, 1 + sqrt(N_узла)) (прогрессивное
    расширение). Оценка нового узла — приняты ли PRM все шаги пути (в AlphaZero её даёт сеть
    ценности); обратное распространение усредняет оценки по пути. Выбор ребёнка —
    Q + c * sqrt(N_родителя) / (1 + N_ребёнка), это PUCT с равными априорными долями.
    Ответ — по числу посещений готовых решений, как ход в AlphaZero.
    """
    K = len(ops)
    tries, children = defaultdict(int), defaultdict(list)
    visits, total = defaultdict(int), defaultdict(float)
    root = (0, x0)
    ok = {root: 1.0}                              # узел -> все шаги пути приняты PRM
    gens = 0
    for _ in range(N * K):
        node, path = root, [root]
        while node[0] < K and tries[node] >= min(M, 1 + int(math.sqrt(visits[node]))):
            n_parent = visits[node]                                   # 1. выбор
            node = max(children[node], key=lambda k: total[k] / max(visits[k], 1)
                       + c * math.sqrt(n_parent) / (1 + visits[k]))
            path.append(node)
        if node[0] < K:                                               # 2. расширение
            op = ops[node[0]]
            y = step(node[1], op, rng, e)
            gens += 1
            tries[node] += 1
            child = (node[0] + 1, y)
            score = min(ok[node], prm(node[1], op, y, rng, ej))       # 3. оценка
            if child not in children[node]:
                children[node].append(child)
            ok[child] = max(ok.get(child, 0.0), score)   # то же число могло прийти другим путём
            path.append(child)
            value = ok[child]
        else:
            value = ok[node]                                          # готовое решение
        for p in path:                                                # 4. обратное распространение
            visits[p] += 1
            total[p] += value
    assert gens <= N * K                          # шагов решающего не больше, чем у N решений
    finals = [(x, s * visits[(d, x)]) for (d, x), s in ok.items() if d == K]
    if not finals:
        return None
    return pick([f for f, _ in finals], [s for _, s in finals], rng)


def check_mcts(seeds=range(1, 6), n_tasks=300):
    print("5. MCTS, N*K симуляций: среднее ± разброс по пяти зёрнам; потолок --- доля верных "
          "среди путей, все шаги которых принял PRM")
    res = {}
    for e in (0.1, 0.3):
        ej = 0.1
        # доля верных среди принятых шагов (формула точности проверяющего) в степени K
        ceiling = ((1 - e) * (1 - ej) / ((1 - e) * (1 - ej) + e * ej)) ** 8
        for N in (4, 16, 64):
            accs = []
            for seed in seeds:
                rng = random.Random(seed)
                tasks = [(rng.randint(10, 99), [(rng.choice("+-*"), rng.randint(2, 9))
                                                for _ in range(8)]) for _ in range(n_tasks)]
                accs.append(sum(mcts(x0, ops, N, rng, e) == solve(x0, ops, rng, 0.0)
                                for x0, ops in tasks) / n_tasks)
            res[e, N] = statistics.mean(accs)
            print(f"   e = {e}, N = {N:2d}: {statistics.mean(accs):.2f} ± "
                  f"{statistics.stdev(accs):.2f}   потолок {ceiling:.2f}")
            assert res[e, N] < ceiling + 0.03     # дерево упирается в точность PRM
        assert f"{ceiling:.2f}" == {0.1: "0.91", 0.3: "0.69"}[e]   # числа текста: 0,95^8 и 0,91
    assert res[0.1, 64] < 0.95 and res[0.3, 64] < 0.75   # лучи с голосованием: 0.99 и 0.92
    assert f"{res[0.1, 64]:.2f} {res[0.3, 64]:.2f}" == "0.86 0.58"   # текст: «0,86 лёгких и 0,58 трудных»
    return res


if __name__ == "__main__":
    check_mc_labels()
    check_judges()
    check_mode()
    check_search()
    check_perfect_prm()
    check_mcts()
