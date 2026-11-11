"""
Глава 10, раздел 10.5 «Цена длинного промпта», подраздел «Цепочка хешей страниц»: листинг
«Хеши страниц общего префикса» (полная версия).

Страница кэша — 16 токенов подряд (в главе 14 и в vLLM её называют блоком
KV-кэша). Строит цепочку хешей страниц для ста запросов с общей инструкцией,
считает общие страницы и токены обработки входа (4448 вместо 48 800), ломает
раскладку меткой времени и правкой 170-го токена и показывает намеренную ошибку:
хеш страницы без предыдущего даёт ложные попадания. Выполняется в ноутбуке главы
notebooks/10_constrained_decoding.ipynb.

Запуск из корня репозитория: python labs/ch10/prefix_page_hashes.py
"""

# --- Листинг «Хеши страниц общего префикса» ---
import hashlib
import random
import struct

PAGE = 16                                # токенов в странице кэша
KV_PER_TOKEN = 2 * 6 * 2 * 32 * 2  # = 1536 байт

def page_hashes(tokens, chained=True):
    hashes, prev = [], b""
    for i in range(len(tokens) // PAGE):  # неполная страница не хешируется
        page = tokens[i * PAGE:(i + 1) * PAGE]
        data = struct.pack(f"<{PAGE}H", *page)
        prev = hashlib.sha256((prev if chained else b"") + data).digest()
        hashes.append(prev)
    return hashes

def shared_pages(requests):
    tables = [page_hashes(r) for r in requests]
    n = 0
    while all(len(t) > n and t[n] == tables[0][n] for t in tables):
        n += 1
    return n

def prefill_tokens(requests):
    # Первый запрос считает всё, остальные — только промахи.
    hit = PAGE * shared_pages(requests)
    return len(requests[0]) + sum(len(r) - hit for r in requests[1:])

rng = random.Random(0)
def rand_tokens(n):
    return [rng.randrange(8, 8192) for _ in range(n)]

system = rand_tokens(448)          # инструкция и примеры
users = [rand_tokens(40) for _ in range(100)]
stable = [system + u for u in users]
assert shared_pages(stable) == 28

# минута в начале системной инструкции
stamped = [[8 + i % 60] + system + u for i, u in enumerate(users)]
assert shared_pages(stamped) == 0

edited = system.copy()
edited[169] += 1                   # токен номер 170
k = 169 // PAGE                   # индекс 10 = страница 11
old, new = page_hashes(system), page_hashes(edited)
assert old[:k] == new[:k]          # страницы 1-10 целы
assert all(a != b for a, b in zip(old[k:], new[k:]))
assert shared_pages([system, edited]) == 10
assert len(page_hashes(system[:455])) == 28   # 7 мимо

total, needed = sum(len(r) for r in stable), prefill_tokens(stable)
assert (total, needed) == (48_800, 4_448)
assert abs(needed / total - 0.091) < 1e-3
assert 448 * KV_PER_TOKEN == 688_128           # 672 КБ

# Намеренная ошибка: хеш страницы без предыдущего.
# Тот же документ после другой инструкции даёт
# ложные попадания, хотя K и V у него другие.
doc, other = rand_tokens(64), rand_tokens(32)
req_a, req_b = system[:32] + doc, other + doc
def hits(cached, request, chained):
    table = set(page_hashes(cached, chained))
    return sum(h in table for h in page_hashes(request, chained))
assert hits(req_a, req_b, chained=True) == 0
assert hits(req_a, req_b, chained=False) == 4  # ложные

print("OK: prefix_page_hashes.py")
