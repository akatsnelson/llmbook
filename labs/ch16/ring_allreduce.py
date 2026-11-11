"""Кольцевой all-reduce на обычных списках Python: две реализации и сверка с рисунком.

Сквозной код главы 16 лежит в labs/ch16/ranks.py, и функции книги берите оттуда. Этот файл
нужен ноутбуку 16_zero_by_hand.ipynb: здесь кольцо с фрагментами из одного числа, которое
повторяет таблицу рисунка 16.2 (бухгалтеры за круглым столом), и вторая реализация, которая проводит каждый фрагмент
по кольцу по очереди и записывает путь (trace). Проверки сверяют обе версии между собой и с
клетками рисунка. Обе версии требуют, чтобы у каждого из p рангов было ровно p чисел, и
отказываются работать с массивом другой длины, а не теряют хвост молча.

Запуск из корня репозитория: python labs/ch16/ring_allreduce.py
"""


# --- Версия 1: каждый фрагмент обходит кольцо по очереди
def ring_allreduce_sequential(values_by_rank):
    # values_by_rank: [ранг, фрагмент]; фрагментов столько же, сколько рангов.
    world = len(values_by_rank)
    if world == 0 or any(len(row) != world for row in values_by_rank):
        raise ValueError("ожидается квадратная таблица ранг × фрагмент")
    trace = []
    reduced_chunks = [0.0] * world

    # Reduce-scatter без параллельного расписания: каждый фрагмент обходит кольцо.
    owners = {}
    for chunk in range(world):
        rank = chunk
        accumulator = 0.0
        for hop in range(world):
            accumulator += values_by_rank[rank][chunk]
            trace.append(("reduce", chunk, rank, accumulator))
            rank = (rank + 1) % world
        owner = (chunk - 1) % world
        owners[chunk] = owner
        reduced_chunks[chunk] = accumulator

    # All-gather: владелец рассылает готовый фрагмент по тому же кольцу.
    result = [[None] * world for _ in range(world)]
    for chunk, owner in owners.items():
        rank = owner
        for hop in range(world):
            result[rank][chunk] = reduced_chunks[chunk]
            trace.append(("gather", chunk, rank, reduced_chunks[chunk]))
            rank = (rank + 1) % world
    return result, trace


# --- Версия 2: настоящее расписание кольца, все ранги шлют одновременно (как в книге)
def ring_allreduce(data, history=None):
    p = len(data)
    if p == 0 or any(len(row) != p for row in data):
        # иначе числа за позицией p - 1 остались бы несложенными, и ранги разошлись бы без сообщения
        raise ValueError("ожидается квадратная таблица ранг × фрагмент: по p чисел у каждого из p рангов")
    buf = [row[:] for row in data]          # у каждого ранга своя копия массива
    # Reduce-scatter: на шаге s ранг r шлёт фрагмент (r - s) mod p, сосед прибавляет.
    for s in range(p - 1):
        sent = [buf[r][(r - s) % p] for r in range(p)]  # все отправляют сразу
        for r in range(p):
            c = (r - 1 - s) % p
            buf[r][c] += sent[(r - 1) % p]
            if history is not None:
                history.append(("reduce", s + 1, r, c, buf[r][c]))
    # All-gather: ранг r шлёт готовый фрагмент (r + 1 - s) mod p, сосед только записывает.
    for s in range(p - 1):
        sent = [buf[r][(r + 1 - s) % p] for r in range(p)]
        for r in range(p):
            c = (r - s) % p
            buf[r][c] = sent[(r - 1) % p]
            if history is not None:
                history.append(("gather", p + s, r, c, buf[r][c]))
    return buf


values = [[1, 2, 3, 4], [10, 20, 30, 40],
          [100, 200, 300, 400], [1000, 2000, 3000, 4000]]

# Обе версии дают один ответ.
result, trace = ring_allreduce_sequential(values)
assert result == [[1111, 2222, 3333, 4444]] * 4
assert len(trace) == 2 * 4 * 4
assert ring_allreduce(values) == [[1111, 2222, 3333, 4444]] * 4

# Все клетки таблицы рисунка 16.2: (шаг, ранг) -> (фрагмент, значение).
history = []
ring_allreduce(values, history)
figure = {
    (1, 0): (3, 4004), (1, 1): (0, 11), (1, 2): (1, 220), (1, 3): (2, 3300),
    (2, 0): (2, 3303), (2, 1): (3, 4044), (2, 2): (0, 111), (2, 3): (1, 2220),
    (3, 0): (1, 2222), (3, 1): (2, 3333), (3, 2): (3, 4444), (3, 3): (0, 1111),
    (4, 0): (0, 1111), (4, 1): (1, 2222), (4, 2): (2, 3333), (4, 3): (3, 4444),
    (5, 0): (3, 4444), (5, 1): (0, 1111), (5, 2): (1, 2222), (5, 3): (2, 3333),
    (6, 0): (2, 3333), (6, 1): (3, 4444), (6, 2): (0, 1111), (6, 3): (1, 2222),
}
seen = {(step, rank): (chunk, value) for _, step, rank, chunk, value in history}
assert seen == figure

# Каждый ранг получил 2(p-1) фрагментов: при p=4 это 6 чисел, как по формуле V_ring.
for rank in range(4):
    assert sum(1 for _, _, r, _, _ in history if r == rank) == 2 * 3

# Нарочная ошибка из главы: если в фазе all-gather прибавлять, на шаге 4 ранг 2 получит
# к готовой сумме 3333 свои 300, то есть 3633, и вклад ранга 2 войдёт дважды.
def broken_gather_step4(data):
    p = len(data)
    buf = [row[:] for row in data]
    for s in range(p - 1):                  # reduce-scatter, как выше
        sent = [buf[r][(r - s) % p] for r in range(p)]
        for r in range(p):
            buf[r][(r - 1 - s) % p] += sent[(r - 1) % p]
    sent = [buf[r][(r + 1) % p] for r in range(p)]   # первый шаг all-gather
    return [buf[r][r] + sent[(r - 1) % p] for r in range(p)]   # ошибка: сложение
def broken_gather_value():
    return broken_gather_step4(values)[2]   # значение у ранга 2 после ошибочного шага 4
assert broken_gather_value() == 3633

# Длина, не равная числу рангов, вызывает отказ.
for bad in ([[1, 2, 3, 4, 5], [10, 20, 30, 40, 50]], [[1, 2, 3]] * 4):
    for fn in (ring_allreduce, ring_allreduce_sequential):
        try:
            fn(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("кольцо приняло массивы неподходящей длины")

# Случайная проверка на других размерах кольца.
import random
for p in range(1, 9):
    data = [[random.randint(-50, 50) for _ in range(p)] for _ in range(p)]
    expected = [sum(data[r][c] for r in range(p)) for c in range(p)]
    assert ring_allreduce(data) == [expected] * p
    assert ring_allreduce_sequential(data)[0] == [expected] * p

print("кольцевой all-reduce: все проверки пройдены")
