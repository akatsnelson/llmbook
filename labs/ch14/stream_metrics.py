"""Клиентский измерительный тракт: события потока, TTFT/TPOT/ITL, нагрузка и перцентили.

Полная версия листингов раздела 14.10 «Измерение скорости ответа» из главы 14 книги. В книге оставлены сокращённые функции;
здесь всё целиком: клиент потока, расчёт метрик с той же проверкой на синтетическом журнале,
замкнутая и разомкнутая нагрузка и сводка по перцентилям.
Без сервера выполняются только проверки на синтетических данных. С запущенным OpenAI-совместимым
сервером (например, vllm serve) вызовите run_closed_loop() или run_open_loop() вручную.

Запуск из корня репозитория: python labs/ch14/stream_metrics.py
"""

from __future__ import annotations

import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from math import isclose
from time import perf_counter_ns
from types import SimpleNamespace

import numpy as np


@dataclass
class StreamTrace:
    started_ns: int
    first_content_ns: int | None
    finished_ns: int
    content_events_ns: list[int]
    text: str


def trace_chat(base_url, api_key, model, messages):
    from openai import OpenAI  # нужен только при работе с настоящим сервером

    client = OpenAI(base_url=base_url, api_key=api_key)
    started = perf_counter_ns()
    first = None
    events = []
    parts = []

    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        stream=True,
    )
    for chunk in stream:
        now = perf_counter_ns()
        delta = chunk.choices[0].delta.content
        if delta:                       # пропускаем служебные и пустые фрагменты
            first = now if first is None else first
            events.append(now)
            parts.append(delta)

    finished = perf_counter_ns()
    return StreamTrace(started, first, finished, events, "".join(parts))


def ns_to_ms(value):
    return value / 1_000_000


def summarize(trace):
    if trace.first_content_ns is None:
        raise RuntimeError("stream finished without content")

    itl_ns = [
        right - left
        for left, right in zip(
            trace.content_events_ns,
            trace.content_events_ns[1:],
        )
    ]
    # TPOT считаем от первого до последнего фрагмента с текстом:
    # служебный фрагмент в конце потока токенов не несёт.
    generation_ns = trace.content_events_ns[-1] - trace.first_content_ns
    n_intervals = max(len(trace.content_events_ns) - 1, 1)

    return {
        "ttft_ms": ns_to_ms(trace.first_content_ns - trace.started_ns),
        "e2e_ms": ns_to_ms(trace.finished_ns - trace.started_ns),
        "tpot_chunk_ms": ns_to_ms(generation_ns / n_intervals),
        "itl_chunk_ms": [ns_to_ms(x) for x in itl_ns],
        "n_content_chunks": len(trace.content_events_ns),
    }


# --- Нагрузка --------------------------------------------------------------------------------

def one_request(i, base_url="http://127.0.0.1:8000/v1", model="org/model"):
    return trace_chat(
        base_url=base_url,
        api_key="local",
        model=model,
        messages=[{"role": "user", "content": f"Задача {i}: ..."}],
    )


def run_closed_loop(n_requests=100, concurrency=16):
    traces = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(one_request, i) for i in range(n_requests)]
        for future in as_completed(futures):
            traces.append(future.result())
    return traces


def arrival_times(rate_per_s, n, seed=0):
    """Моменты прихода n запросов, в среднем rate_per_s запросов в секунду (поток Пуассона)."""
    rng = random.Random(seed)
    t, times = 0.0, []
    for _ in range(n):
        t += rng.expovariate(rate_per_s)   # то же, что -ln(U) / lambda
        times.append(t)
    return times


def run_open_loop(rate_per_s=4.0, n_requests=200, max_workers=256):
    """Отправляет запросы по расписанию, не дожидаясь ответов на предыдущие."""
    schedule = arrival_times(rate_per_s, n_requests)
    start = time.monotonic()
    traces = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = []
        for i, at in enumerate(schedule):
            delay = at - (time.monotonic() - start)
            if delay > 0:
                time.sleep(delay)
            futures.append(pool.submit(one_request, i))
        for future in as_completed(futures):
            traces.append(future.result())
    return traces


def report(traces):
    rows = [summarize(t) for t in traces]
    ttft = np.array([row["ttft_ms"] for row in rows])
    e2e = np.array([row["e2e_ms"] for row in rows])
    return {
        "ttft_p50_ms": np.percentile(ttft, 50),
        "ttft_p95_ms": np.percentile(ttft, 95),
        "ttft_p99_ms": np.percentile(ttft, 99),
        "e2e_p50_ms": np.percentile(e2e, 50),
        "e2e_p95_ms": np.percentile(e2e, 95),
    }


def main():
    # 1. Синтетический журнал: четыре фрагмента с текстом, служебный конец потока позже.
    ms = 1_000_000
    fake = SimpleNamespace(
        started_ns=0,
        first_content_ns=50 * ms,
        finished_ns=260 * ms,
        content_events_ns=[50 * ms, 100 * ms, 150 * ms, 250 * ms],
        text="",
    )
    r = summarize(fake)
    assert r["ttft_ms"] == 50 and r["e2e_ms"] == 260
    assert r["itl_chunk_ms"] == [50, 50, 100]
    assert isclose(r["tpot_chunk_ms"], 200 / 3)    # 200 мс на 3 интервала

    # 2. Разомкнутая нагрузка: промежутки в среднем 1/lambda, но бывают и очень короткие, и длинные.
    times = arrival_times(rate_per_s=4, n=20_000)
    gaps = [b - a for a, b in zip([0.0] + times, times)]
    assert abs(sum(gaps) / len(gaps) - 0.25) < 0.01
    assert min(gaps) < 0.01 and max(gaps) > 1.5

    # 3. Перцентиль: из 100 запросов 95 быстрее p95.
    ttft = np.arange(1, 101) * 10.0                # 10, 20, ..., 1000 мс
    p95 = np.percentile(ttft, 95)
    assert (ttft <= p95).sum() == 95

    # 4. p99 по 20 запросам почти совпадает с самым медленным из них.
    rng = np.random.default_rng(0)
    small = rng.exponential(200, size=20)
    assert np.percentile(small, 99) > np.sort(small)[-2]

    # 5. Сводка по синтетическим журналам.
    traces = [
        SimpleNamespace(started_ns=0, first_content_ns=(40 + i) * ms, finished_ns=(300 + i) * ms,
                        content_events_ns=[(40 + i) * ms, (90 + i) * ms], text="")
        for i in range(100)
    ]
    summary = report(traces)
    assert summary["ttft_p50_ms"] < summary["ttft_p95_ms"] < summary["ttft_p99_ms"]

    print("stream_metrics: все проверки пройдены")


if __name__ == "__main__":
    main()
