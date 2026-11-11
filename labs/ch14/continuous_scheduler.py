"""Непрерывный планировщик поверх распределителя страниц KV-кэша: журнал шагов с рисунка 14.11.

Полная версия листинга «Непрерывный планировщик на запросах с рисунка» из главы 14
(раздел 14.9 «PagedAttention и vLLM», подраздел «Непрерывное формирование пакетов»).
В книге листинг печатает журнал; здесь те же шаги проверяются assert-ами: число токенов в вызове
совпадает с нижней строкой рисунка 14.11 (2, 2, 2, 7, 2), A уходит на шаге 3, D входит на
шаге 4, страницы возвращаются в пул. Запрос принимается, только если страниц хватит на весь его вход
(функция fits): иначе append_token упал бы посреди вызова модели. Плюс сравнение с обычным пакетом на
запросах A–E с рисунка 14.10: 8 пустых мест из 24 против одного.
Распределитель BlockPool (в vLLM страницу кэша называют блоком) — тот же, что в kv_block_pool.py.

Запуск из корня репозитория: python labs/ch14/continuous_scheduler.py
"""

from collections import deque


class BlockPool:
    def __init__(self, n_blocks, block_size):
        self.block_size = block_size
        self.free = deque(range(n_blocks))
        self.tables = {}
        self.lengths = {}

    def add_request(self, request_id):
        self.tables[request_id], self.lengths[request_id] = [], 0

    def append_token(self, request_id):
        if self.lengths[request_id] % self.block_size == 0:
            if not self.free:
                raise RuntimeError("KV pool exhausted")
            self.tables[request_id].append(self.free.popleft())
        self.lengths[request_id] += 1

    def finish(self, request_id):
        self.free.extend(self.tables.pop(request_id))
        self.lengths.pop(request_id)


class Request:
    def __init__(self, id, prompt, max_new):
        self.id = id
        self.prompt = prompt
        self.max_new = max_new
        self.written = 0


def fits(request, pool):
    """Хватит ли свободных страниц на весь необработанный вход запроса."""
    return -(-request.prompt // pool.block_size) <= len(pool.free)


def scheduler_tick(waiting, running, pool, max_batch):
    """Один вызов модели. Возвращает (выжившие, работа {id: токенов}, завершённые)."""
    while waiting and len(running) < max_batch and fits(waiting[0], pool):
        request = waiting.popleft()
        pool.add_request(request.id)
        running.append(request)
    work = {r.id: r.prompt or 1 for r in running}
    survivors, done = [], []
    for r in running:
        for _ in range(work[r.id]):
            pool.append_token(r.id)
        r.prompt, r.written = 0, r.written + 1
        if r.written == r.max_new:
            pool.finish(r.id)
            done.append(r.id)
        else:
            survivors.append(r)
    return survivors, work, done


def figure_run():
    pool = BlockPool(n_blocks=8, block_size=4)
    running = [Request("A", 0, 3), Request("B", 0, 8)]
    for r in running:
        pool.add_request(r.id)
        for _ in range(4):
            pool.append_token(r.id)
    waiting = deque([Request("D", prompt=6, max_new=4)])
    log = []
    for step in range(1, 6):
        running, work, done = scheduler_tick(waiting, running, pool, max_batch=2)
        log.append((step, work, done, len(pool.free)))
        print(f"шаг {step}: {work} = {sum(work.values())}; свободно {len(pool.free)} {done}")
    return log, pool


def idle_slots(lengths, n_lines, continuous):
    """Сколько мест «линия x шаг» простаивает, пока не обслужены все ответы (по шагу на токен)."""
    queue = deque(lengths)
    lines = [queue.popleft() if queue else 0 for _ in range(n_lines)]
    idle = steps = 0
    while any(lines) or queue:
        if continuous or not any(lines):
            lines = [x if x else (queue.popleft() if queue else 0) for x in lines]
        steps += 1
        idle += sum(1 for x in lines if x == 0)
        lines = [max(x - 1, 0) for x in lines]
    return steps, idle


def main():
    log, pool = figure_run()
    assert [sum(w.values()) for _, w, _, _ in log] == [2, 2, 2, 7, 2]
    assert log[2][2] == ["A"]                      # A уходит на шаге 3
    assert log[3][1] == {"B": 1, "D": 6}           # D входит на шаге 4 вместе с токеном B
    assert [free for *_, free in log] == [4, 4, 6, 4, 3]
    assert pool.lengths == {"B": 9, "D": 7}

    # Приём по числу страниц: при одной свободной странице вход из 6 токенов (нужно две) ждёт в очереди.
    pool = BlockPool(n_blocks=1, block_size=4)
    waiting = deque([Request("D", prompt=6, max_new=4)])
    running, work, _ = scheduler_tick(waiting, [], pool, max_batch=2)
    assert running == [] and work == {} and len(waiting) == 1
    # Без проверки fits тот же вход падает громко, а не портит чужой кэш.
    pool.add_request("D")
    try:
        for _ in range(6):
            pool.append_token("D")
    except RuntimeError as err:
        assert "exhausted" in str(err)
    else:
        raise AssertionError("пул из одной страницы принял 6 токенов")

    # Рисунок 14.10: A=3, B=8, C=5 в трёх линиях, в очереди D=4, E=3.
    # Обычный пакет за первые 8 шагов теряет 8 мест из 24.
    lines, idle = [3, 8, 5], 0
    for _ in range(8):
        idle += sum(1 for x in lines if x == 0)
        lines = [max(x - 1, 0) for x in lines]
    assert idle == 8
    steps_static, _ = idle_slots([3, 8, 5, 4, 3], 3, continuous=False)
    steps_cont, _ = idle_slots([3, 8, 5, 4, 3], 3, continuous=True)
    print(f"обычный пакет: {steps_static} шагов, непрерывный: {steps_cont} шагов")
    assert steps_cont == 8 and steps_static > steps_cont
    print("continuous_scheduler: все проверки пройдены")


if __name__ == "__main__":
    main()
