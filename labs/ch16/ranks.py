"""Сквозной код главы 16: ранги на обычных списках Python.

Ранг — строка таблицы, пересылка — копирование числа. Модуль растёт так же, как глава:
кольцевой all-reduce (reduce-scatter и all-gather), шаг DDP на нём, шаг ZeRO-1 и ZeRO-2,
тензорный параллелизм матрицы 2x2 и all-to-all для параллелизма экспертов. Функции совпадают с листингами книги
(глава 16), а проверки assert сверяют каждое число, которое
напечатано в главе, включая все 24 клетки рисунка 16.2 с кольцом.

Запуск из корня репозитория: python labs/ch16/ranks.py
"""


# ----------------------------------------------------------------------------- кольцо

def chunk(c, n):
    return range(c * n, (c + 1) * n)          # позиции фрагмента c


def fragment_size(buf):
    """Длина фрагмента; если массив не делится на p поровну, кольцо отказывается работать.

    Иначе хвост массива молча остался бы несложенным, а копии рангов разошлись бы. Настоящие
    библиотеки дополняют массив нулями до кратной длины; учебное кольцо требует этого от вызывающего.
    """
    p = len(buf)
    assert all(len(row) == len(buf[0]) for row in buf), "у всех рангов массивы одной длины"
    assert len(buf[0]) % p == 0, "длина массива должна делиться на p"
    return len(buf[0]) // p


def reduce_scatter(buf, show=False, history=None):
    """Первая фаза кольца: после p-1 шагов ранг r держит готовую сумму фрагмента (r+1) mod p."""
    p, n = len(buf), fragment_size(buf)
    for s in range(p - 1):
        sent = [[buf[r][i] for i in chunk((r - s) % p, n)] for r in range(p)]
        for r in range(p):                     # все ранги шлют одновременно
            c = (r - 1 - s) % p                # фрагмент, пришедший от r - 1
            got = sent[(r - 1) % p]
            for i, v in zip(chunk(c, n), got):
                buf[r][i] += v                 # получил и прибавил своё
            if history is not None:            # (шаг, ранг, фрагмент, значение, сколько чисел пришло)
                history.append((s + 1, r, c, buf[r][c * n], len(got)))
        if show:
            print(f"шаг {s + 1}:", "  ".join(
                f"c{(r - 1 - s) % p}={buf[r][(r - 1 - s) % p]}" for r in range(p)))


def all_gather(buf, show=False, history=None, broken=False):
    """Вторая фаза: готовые суммы идут по тому же кольцу, получатель только записывает.

    broken=True — нарочная ошибка из главы: получатель по привычке прибавляет своё число.
    """
    p, n = len(buf), fragment_size(buf)
    for s in range(p - 1):
        sent = [[buf[r][i] for i in chunk((r + 1 - s) % p, n)] for r in range(p)]
        for r in range(p):
            c = (r - s) % p
            got = sent[(r - 1) % p]
            for i, v in zip(chunk(c, n), got):
                buf[r][i] = buf[r][i] + v if broken else v   # только записал, без сложения
            if history is not None:
                history.append((p + s, r, c, buf[r][c * n], len(got)))
        if show:
            print(f"шаг {p + s}:", "  ".join(
                f"c{(r - s) % p}={buf[r][(r - s) % p]}" for r in range(p)))


def ring_allreduce(data, show=False):
    buf = [row[:] for row in data]             # у каждого ранга своя копия
    reduce_scatter(buf, show)
    all_gather(buf, show)
    return buf


# ----------------------------------------------------------------------------- DDP и ZeRO

def ddp_step(weights, grads_by_rank, lr):
    p = len(grads_by_rank)
    summed = ring_allreduce(grads_by_rank)     # сумма градиентов у каждого ранга
    return [[w - lr * g / p for w, g in zip(weights, row)] for row in summed]


def zero_step(weights, grads_by_rank, lr, stage, state, show=False):
    """Шаг SGD с моментом 0,9; state[r] — словарь моментов ранга r, только для его фрагмента."""
    p, n = len(grads_by_rank), len(weights) // len(grads_by_rank)
    g = [row[:] for row in grads_by_rank]
    if stage == 1:
        g = ring_allreduce(g)                  # полный градиент у всех
    else:
        reduce_scatter(g)                      # только своя готовая сумма
    w = [weights[:] for _ in range(p)]
    for r in range(p):
        mine = chunk((r + 1) % p, n)           # фрагмент, которым владеет ранг r
        for i in mine:                         # SGD с моментом 0,9
            state[r][i] = 0.9 * state[r].get(i, 0.0) + g[r][i] / p
            w[r][i] -= lr * state[r][i]        # шаг только по своему фрагменту
        if show:
            print(f"ранг {r}: градиент {[round(x, 4) for x in g[r]]},",
                  "момент для", ", ".join(f"w{i}" for i in state[r]))
    all_gather(w)                              # раздаём обновлённые фрагменты
    return w


def all_to_all(buf):
    """buf[src][dst] — порция, которую ранг src шлёт рангу dst; ранг dst получает [buf[0][dst], ...]."""
    p = len(buf)
    assert all(len(row) == p for row in buf), "у каждого ранга по порции для каждого ранга"
    return [[buf[src][dst] for src in range(p)] for dst in range(p)]


# ----------------------------------------------------------------------------- тензорный параллелизм

def matvec(x, w):
    return [sum(x[i] * w[i][j] for i in range(len(x))) for j in range(len(w[0]))]


def tp_columns(x, w):
    """GPU j хранит столбец j; куски ответа склеивают (all-gather)."""
    pieces = [matvec(x, [[row[j]] for row in w]) for j in range(len(w[0]))]
    return pieces, [v for piece in pieces for v in piece]


def tp_rows(x, w):
    """GPU i хранит строку i и координату x_i; частичные суммы складывают (all-reduce)."""
    partial = [matvec([x[i]], [w[i]]) for i in range(len(x))]
    return partial, ring_allreduce(partial)[0]


# ----------------------------------------------------------------------------- проверки

def close(a, b, tol=1e-12):
    return all(abs(x - y) < tol for x, y in zip(a, b))


if __name__ == "__main__":
    import random

    values = [[1, 2, 3, 4], [10, 20, 30, 40],
              [100, 200, 300, 400], [1000, 2000, 3000, 4000]]

    # Все клетки таблицы рисунка 16.2 «Кольцевой all-reduce»: (шаг, ранг) -> (фрагмент, значение).
    figure = {
        (1, 0): (3, 4004), (1, 1): (0, 11), (1, 2): (1, 220), (1, 3): (2, 3300),
        (2, 0): (2, 3303), (2, 1): (3, 4044), (2, 2): (0, 111), (2, 3): (1, 2220),
        (3, 0): (1, 2222), (3, 1): (2, 3333), (3, 2): (3, 4444), (3, 3): (0, 1111),
        (4, 0): (0, 1111), (4, 1): (1, 2222), (4, 2): (2, 3333), (4, 3): (3, 4444),
        (5, 0): (3, 4444), (5, 1): (0, 1111), (5, 2): (1, 2222), (5, 3): (2, 3333),
        (6, 0): (2, 3333), (6, 1): (3, 4444), (6, 2): (0, 1111), (6, 3): (1, 2222),
    }
    history = []
    buf = [row[:] for row in values]
    reduce_scatter(buf, history=history)
    # после reduce-scatter ранг r владеет готовой суммой фрагмента (r + 1) mod p
    assert [buf[r][(r + 1) % 4] for r in range(4)] == [2222, 3333, 4444, 1111]
    after_rs = [row[:] for row in buf]
    all_gather(buf, history=history)
    assert buf == [[1111, 2222, 3333, 4444]] * 4
    assert {(s, r): (c, v) for s, r, c, v, _ in history} == figure

    # Ошибка «сложение в фазе all-gather»: на шаге 4 ранг 2 прибавляет к пришедшей сумме 3333
    # своё число 300, и в итоге у рангов разные массивы.
    broken_history = []
    all_gather(after_rs, history=broken_history, broken=True)
    assert [v for s, r, c, v, _ in broken_history if (s, r) == (4, 2)] == [3633]
    assert after_rs != [[1111, 2222, 3333, 4444]] * 4

    # Сколько чисел каждый ранг получил (и, значит, отправил) за обе фазы: 2 (p-1)/p A при A = p n.
    for p in (2, 3, 4, 8):
        for n in (1, 3):
            hist = []
            data = [[1] * (p * n) for _ in range(p)]
            reduce_scatter(data, history=hist)
            all_gather(data, history=hist)
            moved = [sum(k for _, r, _, _, k in hist if r == rank) for rank in range(p)]
            assert moved == [2 * (p - 1) * n] * p, (p, n, moved)

    # Длина не делится на число рангов: кольцо отказывается, а не теряет хвост молча.
    for bad in ([[1, 2, 3, 4, 5], [10, 20, 30, 40, 50]], [[1, 2, 3]] * 4, [[1, 2], [1, 2, 3, 4]]):
        try:
            ring_allreduce(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError(f"кольцо приняло массивы неподходящей длины: {bad}")

    # Случайные кольца разного размера и фрагменты из нескольких чисел.
    rng = random.Random(0)
    for p in range(1, 9):
        for n in (1, 3):
            data = [[rng.randint(-50, 50) for _ in range(p * n)] for _ in range(p)]
            expected = [sum(data[r][i] for r in range(p)) for i in range(p * n)]
            assert ring_allreduce(data) == [expected] * p

    # DDP: пример 0,2 и 0,4 -> 0,3; обе копии весов одинаковы и совпадают с одной картой.
    weights = [1.0, 2.0, 3.0, 4.0]
    grads_by_rank = [[0.2, 0.4, 0.6, 0.8], [0.4, 0.2, 1.0, 0.6]]
    one_gpu = [w - 0.1 * g for w, g in zip(weights, [0.3, 0.3, 0.8, 0.7])]
    ddp = ddp_step(weights, grads_by_rank, 0.1)
    assert ddp[0] == ddp[1] and close(ddp[0], one_gpu)
    assert [round(x, 4) for x in ddp[0]] == [0.97, 1.97, 2.92, 3.93]

    # ZeRO-1 и ZeRO-2 дают тот же шаг (момент из нуля на первом шаге равен градиенту);
    # каждый ранг хранит момент только для своего фрагмента; у ZeRO-2 полного градиента нет ни у кого.
    for stage in (1, 2):
        state = [{} for _ in grads_by_rank]
        w = zero_step(weights, grads_by_rank, 0.1, stage, state)
        assert w[0] == w[1] and close(w[0], one_gpu)
        assert [sorted(st) for st in state] == [[2, 3], [0, 1]]
        # второй шаг с тем же градиентом: момент 0,9 * g + g = 1,9 g, как у одной карты с моментом
        w2 = zero_step(w[0], grads_by_rank, 0.1, stage, state)
        assert close(w2[0], [x - 0.1 * 1.9 * g for x, g in zip(one_gpu, [0.3, 0.3, 0.8, 0.7])])
    g = [row[:] for row in grads_by_rank]
    reduce_scatter(g)
    assert close(g[0], [0.2, 0.4, 1.6, 1.4]) and close(g[1], [0.6, 0.6, 1.0, 0.6])

    # Тензорный параллелизм 2x2: столбцы склеить, строки сложить.
    X, W = [1, 2], [[3, 4], [5, 6]]
    assert tp_columns(X, W) == ([[13], [16]], [13, 16])
    assert tp_rows(X, W) == ([[3, 4], [10, 12]], [13, 16])
    # и на случайной матрице 4x4 при четырёх «картах»
    X4 = [rng.randint(-5, 5) for _ in range(4)]
    W4 = [[rng.randint(-5, 5) for _ in range(4)] for _ in range(4)]
    assert tp_columns(X4, W4)[1] == tp_rows(X4, W4)[1] == matvec(X4, W4)

    # all-to-all: строка таблицы коллективных операций и пример с экспертами из главы.
    table = all_to_all([[10 ** r * (i + 1) for i in range(4)] for r in range(4)])
    assert table[0] == [1, 10, 100, 1000] and table[3] == [4, 40, 400, 4000]
    home = {"E1": 0, "E2": 0, "E3": 1, "E4": 1}
    routes = [["E1", "E1", "E3", "E3"], ["E4", "E4", "E4", "E2"]]
    send = [[[e for e in row if home[e] == dst] for dst in range(2)] for row in routes]
    assert [sum(len(part) for part in got) for got in all_to_all(send)] == [3, 5]

    print("ranks.py: кольцо, DDP, ZeRO-1/2, тензорный параллелизм и all-to-all сходятся с книгой")
