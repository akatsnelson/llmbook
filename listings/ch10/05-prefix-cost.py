"""Листинги главы 10 «Работа с промптом» книги «Большие языковые модели», файл 05-prefix-cost.

Код и вывод — ровно как в книге; текст книги сюда не входит. Листинги идут по порядку и
опираются друг на друга: запускайте файл целиком или ячейками сверху вниз (разметка # %%).
Проверка, что вывод совпал с книгой: python scripts/check_listings.py listings/ch10/05-prefix-cost.py
"""

# %% Листинг «Подготовка: функции раздела sec:prompt-reasoning» — раздел «Обработка входа и генерация»
from labs.ch10.reasoning_budget import generation_flops, kv_bytes

# %% Листинг «Цена обработки входа» — раздел «Обработка входа и генерация»
P_S, P_U = 448, 40               # неизменная и личная части промпта

one = generation_flops(0, P_S + P_U)        # позиции 1, ..., 488
print(f"один запрос: {one / 1e9:.2f} GFLOP,",
      kv_bytes(P_S + P_U) // 1024, "КБ кэша")
print(f"сто запросов: {100 * one / 1e9:.1f} GFLOP, из них общая часть"
      f" {generation_flops(0, P_S) / one:.0%}")
# Вывод в книге:
# | один запрос: 7.20 GFLOP, 732 КБ кэша
# | сто запросов: 719.7 GFLOP, из них общая часть 91%

# %% Листинг «Готовый префикс даёт те же логиты» — раздел «Кэширование общего префикса»
import torch
from booklm.lab import load, val_tokens

tok, model = load("base")
data = val_tokens(10_000)                       # отложенные истории подряд
prefix = data[:448][None]                       # общий префикс
tails = [data[1000 + 40 * k:1040 + 40 * k][None] for k in range(100)]

@torch.no_grad()
def read(ids, caches):                          # дочитать ids поверх кэша
    logits = model(ids, caches)
    for c in caches:
        c["len"] += ids.size(1)                 # длину кэша ведём сами
    return logits

new_cache = lambda: [{"len": 0} for _ in model.blocks]
cached = new_cache()
read(prefix, cached)                            # префикс один раз
worst, same = 0.0, 0
for tail in tails:
    full = read(torch.cat([prefix, tail], 1), new_cache())[0, 448:]
    part = read(tail, [dict(c) for c in cached])[0]   # копия кэша на запрос
    worst = max(worst, (part - full).abs().max().item())
    same += int(part[-1].argmax() == full[-1].argmax())
print(f"наибольшая разница логитов: {worst:.1e};",
      f"лидер совпал в {same} из 100")

edited = new_cache()                            # другой первый токен
read(torch.cat([data[5:6], prefix[0, 1:]])[None], edited)
for b in range(len(model.blocks)):              # ключ 32-й позиции
    k_new, k_old = edited[b]["k"][0, :, 31], cached[b]["k"][0, :, 31]
    print(b + 1, torch.equal(k_new, k_old))
# Вывод в книге:
# | наибольшая разница логитов: 0.0e+00; лидер совпал в 100 из 100
# | 1 True
# | 2 False
# | 3 False
# | 4 False
# | 5 False
# | 6 False

# %% Листинг «Сто запросов с общим префиксом» — раздел «Кэширование общего префикса»
n = 100
tokens_without, tokens_with = n * (P_S + P_U), P_S + n * P_U
flops_without = n * generation_flops(0, P_S + P_U)
flops_with = generation_flops(0, P_S) + n * generation_flops(P_S, P_U)
print(f"токены: {tokens_with} вместо {tokens_without},"
      f" {tokens_with / tokens_without:.1%}")
print(f"GFLOP:  {flops_with / 1e9:.1f} вместо {flops_without / 1e9:.1f},"
      f" {flops_with / flops_without:.1%}")
print(f"кэш префикса: {kv_bytes(P_S) / 2**10:.0f} КБ вместо"
      f" {n * kv_bytes(P_S) / 2**20:.1f} МБ")
# Вывод в книге:
# | токены: 4448 вместо 48800, 9.1%
# | GFLOP:  71.0 вместо 719.7, 9.9%
# | кэш префикса: 672 КБ вместо 65.6 МБ

# %% Листинг «Время ста запросов на BookLM» — раздел «Кэширование общего префикса»
# Проверка этот листинг не запускает: ему нужны сеть, видеокарта, сервер или внешний файл.
# import statistics
# import time
#
# def without_cache():
#     for tail in tails:
#         read(torch.cat([prefix, tail], 1), new_cache())
#
# def with_cache():
#     c0 = new_cache()
#     read(prefix, c0)                            # префикс один раз
#     for tail in tails:
#         read(tail, [dict(c) for c in c0])
#
# def seconds(f, repeats=3):                      # медиана трёх замеров
#     times = []
#     for _ in range(repeats):
#         t0 = time.perf_counter()
#         f()
#         times.append(time.perf_counter() - t0)
#     return statistics.median(times)
#
# t_without, t_with = seconds(without_cache), seconds(with_cache)
# print(f"100 запросов на CPU: без кэша {t_without:.2f} с,"
#       f" с кэшем {t_with:.2f} с, {t_with / t_without:.0%}")

# %% Листинг «Хеши страниц общего префикса» — раздел «Цепочка хешей страниц»
import hashlib
import random
import struct

PAGE = 16                                # токенов в странице кэша

def page_hashes(tokens):
    hashes, prev = [], b""
    for i in range(len(tokens) // PAGE): # неполная страница не хешируется
        data = struct.pack(f"<{PAGE}H", *tokens[i * PAGE:(i + 1) * PAGE])
        prev = hashlib.sha256(prev + data).digest()   # + предыдущий
        hashes.append(prev)
    return hashes

def shared_pages(requests):              # сколько первых страниц общие
    tables = [page_hashes(r) for r in requests]
    n = 0
    while all(len(t) > n and t[n] == tables[0][n] for t in tables):
        n += 1
    return n

rng = random.Random(0)
random_tokens = lambda n: [rng.randrange(8, 8192) for _ in range(n)]
instr, ex1, ex2 = random_tokens(160), random_tokens(144), random_tokens(144)
doc = random_tokens(64)
system_ids = instr + ex1 + ex2           # 448 токенов, 28 страниц
cached = system_ids + random_tokens(P_U)  # прошлый запрос в кэше
user = random_tokens(P_U)                # личная часть нового запроса
edited_prompt = system_ids.copy()
edited_prompt[169] += 1                  # правка 170-го токена
layouts = {"как было": system_ids + user,
           "время в начале": [8] + system_ids + user,
           "правка 170-го токена": edited_prompt + user,
           "примеры переставлены": instr + ex2 + ex1 + user,
           "лишний перевод строки": system_ids[:3] + [203]
                                    + system_ids[3:] + user,
           "документ перед инструкцией": doc + system_ids + user}
for name, request in layouts.items():
    print(f"{name:27}", shared_pages([cached, request]))
# Вывод в книге:
# | как было                    28
# | время в начале              0
# | правка 170-го токена        10
# | примеры переставлены        10
# | лишний перевод строки       0
# | документ перед инструкцией  0

# %% Листинг «Хеш страницы без предыдущего» — раздел «Цепочка хешей страниц»
def page_hashes_unchained(tokens):       # ошибка: хеш без предыдущего
    pages = [tokens[i:i + PAGE]
             for i in range(0, len(tokens) - PAGE + 1, PAGE)]
    return [hashlib.sha256(struct.pack(f"<{PAGE}H", *page)).digest()
            for page in pages]

def cache_hits(hash_fn, cached_request, request):
    table = set(hash_fn(cached_request))  # страницы, уже лежащие в кэше
    return sum(h in table for h in hash_fn(request))

doc4 = random_tokens(4 * PAGE)           # документ на четыре страницы
first, second = random_tokens(160) + doc4, instr + doc4
print("с цепочкой:  ", cache_hits(page_hashes, first, second))
print("без цепочки: ", cache_hits(page_hashes_unchained, first, second))
# Вывод в книге:
# | с цепочкой:   0
# | без цепочки:  4
