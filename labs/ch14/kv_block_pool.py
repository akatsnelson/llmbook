"""Учебный распределитель страниц KV-кэша (PagedAttention).

Полная версия листинга «Распределитель страниц KV-кэша» из главы 14 книги
(раздел 14.9 «PagedAttention и vLLM»). Страницу кэша
vLLM называет блоком, отсюда имя класса BlockPool. Здесь класс целиком, с теми же проверками, что в книге,
плюс сравнение страниц по 16 токенов с резервированием максимального буфера. Непрерывный планировщик поверх
этого класса — в continuous_scheduler.py (один на главу).

Запуск из корня репозитория: python labs/ch14/kv_block_pool.py
"""

import math
import random
from collections import deque


class BlockPool:
    def __init__(self, n_blocks: int, block_size: int):
        self.block_size = block_size
        self.free = deque(range(n_blocks))
        self.tables = {}       # request_id -> [номера физических страниц]
        self.lengths = {}      # request_id -> cached tokens

    def add_request(self, request_id: str):
        self.tables[request_id] = []
        self.lengths[request_id] = 0

    def append_token(self, request_id: str):
        length = self.lengths[request_id]
        needs_block = length % self.block_size == 0
        if needs_block:
            if not self.free:
                raise RuntimeError("KV pool exhausted")
            self.tables[request_id].append(self.free.popleft())
        self.lengths[request_id] += 1

    def address(self, request_id: str, token_position: int):
        logical = token_position // self.block_size
        offset = token_position % self.block_size
        physical = self.tables[request_id][logical]
        return physical, offset

    def finish(self, request_id: str):
        self.free.extend(self.tables.pop(request_id))
        self.lengths.pop(request_id)


def main():
    # 1. Пример с рисунка: P = 4, страница 2 занята запросом B, свободные идут вразнобой.
    pool = BlockPool(n_blocks=6, block_size=4)
    pool.free = deque([4, 1, 3, 0, 5])
    pool.add_request("A")
    for _ in range(10):
        pool.append_token("A")
    print(pool.tables["A"])                          # [4, 1, 3]
    print(pool.address("A", token_position=9))       # (3, 1)
    assert pool.tables["A"] == [4, 1, 3]
    assert pool.address("A", 9) == (pool.tables["A"][2], 1)
    pool.finish("A")                                 # страницы возвращены в общий пул
    assert len(pool.free) == 5

    # 2. 100 запросов разной длины: резерв на максимум против страниц по 16 токенов.
    rng = random.Random(0)
    lengths = [rng.randint(50, 2000) for _ in range(100)]
    max_len, block = 8192, 16
    reserved = 100 * max_len
    paged = sum(math.ceil(n / block) * block for n in lengths)
    used = sum(lengths)
    assert paged - used < 100 * block                # потеря меньше одной страницы на запрос
    print(f"занято токенами: {used}, страницами: {paged}, резерв на максимум: {reserved}")

    print("kv_block_pool: все проверки пройдены")


if __name__ == "__main__":
    main()
