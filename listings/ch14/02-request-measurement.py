"""Листинги главы 14 «Инференс» книги «Большие языковые модели», файл 02-request-measurement.

Код и вывод — ровно как в книге; текст книги сюда не входит. Листинги идут по порядку и
опираются друг на друга: запускайте файл целиком или ячейками сверху вниз (разметка # %%).
Проверка, что вывод совпал с книгой: python scripts/check_listings.py listings/ch14/02-request-measurement.py
"""

# %% Листинг «Клиентский измеритель событий потока» — раздел «Метрики из журнала событий»
# Проверка этот листинг не запускает: ему нужны сеть, видеокарта, сервер или внешний файл.
# from time import perf_counter_ns
# from types import SimpleNamespace
# from openai import OpenAI
#
# def trace_chat(base_url, api_key, model, messages):
#     client = OpenAI(base_url=base_url, api_key=api_key)
#     started = perf_counter_ns()
#     first, events, parts = None, [], []
#     stream = client.chat.completions.create(
#         model=model, messages=messages, temperature=0, stream=True)
#     for chunk in stream:
#         now = perf_counter_ns()
#         delta = chunk.choices[0].delta.content
#         if delta:              # служебные и пустые фрагменты пропускаем
#             first = now if first is None else first
#             events.append(now)
#             parts.append(delta)
#     return SimpleNamespace(started_ns=started, first_content_ns=first,
#                            finished_ns=perf_counter_ns(),
#                            content_events_ns=events, text="".join(parts))

# %% Листинг «TTFT, интервалы, TPOT и полное время по журналу» — раздел «Метрики из журнала событий»
from types import SimpleNamespace

def summarize(trace):
    if trace.first_content_ns is None:
        raise RuntimeError("stream finished without content")
    events, ms = trace.content_events_ns, 1_000_000
    itl = [(b - a) / ms for a, b in zip(events, events[1:])]
    return {
        "ttft_ms": (trace.first_content_ns - trace.started_ns) / ms,
        "itl_ms": itl,
        # TPOT: от первого до последнего фрагмента с текстом
        "tpot_ms": (events[-1] - events[0]) / max(len(itl), 1) / ms,
        "e2e_ms": (trace.finished_ns - trace.started_ns) / ms,
    }

ms = 1_000_000
log = SimpleNamespace(            # четыре фрагмента с текстом,
    started_ns=0,                 # служебный конец потока на 510-й мс
    first_content_ns=300 * ms, finished_ns=510 * ms,
    content_events_ns=[300 * ms, 350 * ms, 400 * ms, 500 * ms])
for name, value in summarize(log).items():
    if not isinstance(value, list):
        value = round(value, 1)
    print(f"{name:8} {value}")
# Вывод в книге:
# | ttft_ms  300.0
# | itl_ms   [50.0, 50.0, 100.0]
# | tpot_ms  66.7
# | e2e_ms   510.0

# %% Листинг «TTFT и TPOT BookLM при разной длине входа» — раздел «Метрики из журнала событий»
# Проверка этот листинг не запускает: ему нужны сеть, видеокарта, сервер или внешний файл.
# import math
# import statistics
# import torch
# from booklm.lab import load
# from booklm.experiments.ch14 import long_story, timed_run
#
# torch.set_num_threads(1)                    # один поток процессора
# _, model = load("base")
# story, G, ms = long_story(448), 64, 1e6     # сказка и 64 токена ответа
# print(f"BookLM, один поток CPU, G = {G} новых токенов,",
#       "медианы по 5 повторам, мс")
# print(" вход    TTFT   TPOT  p95 ITL     E2E  TTFT+(G-1)TPOT")
# for n in (16, 128, 448):
#     ids = story[:n][None]
#     timed_run(model, ids, G)                # прогрев
#     rows = []
#     for _ in range(5):
#         start, marks = timed_run(model, ids, G)   # метки времени, нс
#         itl = sorted((b - a) / ms for a, b in zip(marks, marks[1:]))
#         rows.append(((marks[0] - start) / ms,                  # TTFT
#                      (marks[-1] - marks[0]) / (G - 1) / ms,    # TPOT
#                      itl[math.ceil(0.95 * len(itl)) - 1],      # p95 ITL
#                      (marks[-1] - start) / ms))                # E2E
#     ttft, tpot, p95, e2e = (statistics.median(col) for col in zip(*rows))
#     print(f"{n:5} {ttft:7.2f} {tpot:6.2f} {p95:8.2f} {e2e:7.1f}"
#           f" {ttft + (G - 1) * tpot:15.1f}")

# %% Листинг «Моменты прихода запросов для разомкнутой нагрузки» — раздел «Замкнутая и разомкнутая нагрузка»
import math
import random

def arrival_times(rate_per_s, n, seed=0):
    rng = random.Random(seed)
    t, times = 0.0, []
    for _ in range(n):
        t += rng.expovariate(rate_per_s)   # -ln(U) / lambda
        times.append(t)
    return times

times = arrival_times(rate_per_s=4, n=20_000)
gaps = [b - a for a, b in zip([0.0] + times, times)]
print(f"средняя пауза {sum(gaps) / len(gaps):.3f} с")
for t in (0.25, 0.5, 1.0):
    share = sum(g > t for g in gaps) / len(gaps)
    print(f"длиннее {t} с: {share:.1%}, формула {math.exp(-4 * t):.1%}")
# Вывод в книге:
# | средняя пауза 0.250 с
# | длиннее 0.25 с: 36.8%, формула 36.8%
# | длиннее 0.5 с: 13.4%, формула 13.5%
# | длиннее 1.0 с: 1.8%, формула 1.8%

# %% Листинг «Разомкнутая нагрузка» — раздел «Замкнутая и разомкнутая нагрузка»
# Проверка этот листинг не запускает: ему нужны сеть, видеокарта, сервер или внешний файл.
# import time
# from concurrent.futures import ThreadPoolExecutor
#
# URL = "http://127.0.0.1:8000/v1"
#
# def run_open_loop(rate_per_s=4.0, n_requests=200):
#     schedule = arrival_times(rate_per_s, n_requests)
#     start, futures = time.monotonic(), []
#     with ThreadPoolExecutor(max_workers=256) as pool:
#         for i, at in enumerate(schedule):
#             time.sleep(max(0.0, at - (time.monotonic() - start)))
#             messages = [{"role": "user", "content": f"Задача {i}: ..."}]
#             futures.append(pool.submit(trace_chat, URL, "local",
#                                        "org/model", messages))
#     return [f.result() for f in futures]

# %% Листинг «Среднее и перцентили задержки» — раздел «Перцентили»
import numpy as np

rng = np.random.default_rng(0)
ttft = rng.normal(200, 20, size=100)     # 100 запросов, мс
ttft[:3] = 2000                          # троим не повезло с очередью
print(f"среднее {ttft.mean():5.0f} мс")
for p in (50, 95, 99):
    print(f"p{p:<5} {np.percentile(ttft, p):5.0f} мс")
small = ttft[rng.permutation(100)[:20]]  # если бы замеров было 20
p99_small = np.percentile(small, 99)
print(f"по 20 замерам: p99 {p99_small:.0f}, максимум {small.max():.0f}")
# Вывод в книге:
# | среднее   255 мс
# | p50      202 мс
# | p95      237 мс
# | p99     2000 мс
# | по 20 замерам: p99 1666, максимум 2000
