"""Глава 11, разделы 11.1–11.3: растущая реализация окна контекста одним модулем.

Здесь собрано то, что листинги первых трёх разделов главы строят шаг за шагом:

11.1 — цена состояния: weights_bytes, kv_bytes, requests_to_outweigh;
11.2 — сообщение Msg, сборщик окна assemble_context, проверка окна check_window,
       сломанная обрезка truncate_from_front с диагнозом diagnose,
       пересчёт при кэшировании общего префикса recomputed_tokens;
11.3 — запись сводки SummaryRecord, вопросы-зонды Probe, метрики measure,
       модель накопительной сводки rolling_summary.

Импорт из листинга или ноутбука (путь от корня репозитория):

    import sys; sys.path.insert(0, "labs/ch11")
    from window_kit import assemble_context, check_window, measure

Файл самодостаточен: запуск прогоняет все проверки
с числами из книги. Более подробные версии отдельных листингов лежат рядом
(lifetime_costs.py, context_assembler.py, prefix_reuse.py, summary_probes.py).

Запуск из корня репозитория: python labs/ch11/window_kit.py
"""

import random
import re
from itertools import count as id_counter
from typing import NamedTuple

# ----------------------------------------------------------------- 11.1
LAYERS, KV_HEADS, HEAD_DIM, T_MAX = 6, 2, 32, 512   # BookLM, глава 4
PARAMS, BF16 = 6_622_464, 2


def weights_bytes(params=PARAMS, b=BF16):
    return params * b


def kv_bytes(tokens, kv_heads=KV_HEADS, b=BF16):
    """Байт KV-кэша на tokens токенов: ключ и значение в каждом блоке."""
    return 2 * LAYERS * tokens * kv_heads * HEAD_DIM * b


def requests_to_outweigh(tokens=T_MAX, kv_heads=KV_HEADS):
    n = 1
    while n * kv_bytes(tokens, kv_heads=kv_heads) <= weights_bytes():
        n += 1
    return n


# ----------------------------------------------------------------- 11.2
RESERVE, S, M, U = 128, 48, 64, 32       # резерв ответа, система, память, вопрос
H = T_MAX - RESERVE - S - M - U          # доля истории: 240
_ids = id_counter()


class Msg(NamedTuple):
    role: str                # system | user | assistant | tool
    tokens: int              # длина в токенах, задана числом
    text: str = ""
    pinned: bool = False
    id: int = 0


def msg(role, tokens, text="", pinned=False):
    return Msg(role, tokens, text, pinned, next(_ids))


def count_tokens(m):
    return m.tokens          # заглушка: в рабочей системе — токенизатор модели


def assemble_context(system, memory, history, new, count=count_tokens, drop_chunk=1):
    """-> (окно, выброшенные пары). history: список пар от старых к новым."""
    fixed = count(system) + count(memory) + count(new)
    kept, dropped = list(history), []

    def size(ps):
        return sum(count(m) for p in ps for m in p)

    while fixed + size(kept) > T_MAX - RESERVE:
        free = [p for p in kept if not any(m.pinned for m in p)]
        if not free:
            raise ValueError("закреплённое не помещается в бюджет")
        for p in free[:drop_chunk]:
            kept.remove(p)
            dropped.append(p)
    window = [system, memory] + [m for p in kept for m in p] + [new]
    return window, dropped


def check_window(window, dropped, system, memory, history, new, count=count_tokens):
    """Шесть условий раздела 11.2; возвращает список нарушений (пустой — всё в порядке)."""
    bad = []
    if sum(count(m) for m in window) > T_MAX - RESERVE:
        bad.append("окно длиннее бюджета")
    if window[0] is not system or window[-1] is not new:
        bad.append("нет системы в начале или вопроса в конце")
    inside = window[2:-1]
    kept = [p for p in history if p not in dropped]
    if inside != [m for p in kept for m in p]:
        bad.append("пары разорваны или переставлены")
    if sorted(map(id, dropped + kept)) != sorted(map(id, history)):
        bad.append("пара исчезла молча")
    if any(m.pinned for p in dropped for m in p):
        bad.append("выброшено закреплённое")
    return bad


def truncate_from_front(system, memory, history, new, budget=T_MAX - RESERVE):
    """Намеренно сломанная обрезка: склеить всё в строку и оставить последние budget токенов."""
    stream = [system, memory] + [m for p in history for m in p] + [new]
    tokens = [(m, k) for m in stream for k in range(m.tokens)]
    return tokens[-budget:]


def diagnose(tail, system):
    problems = []
    if all(m is not system for m, _ in tail):
        problems.append("системная инструкция пропала")
    first, k = tail[0]
    if k > 0:
        problems.append(f"«{first.text}» начато с токена {k} из {first.tokens}")
    return problems


def recomputed_tokens(system, memory, pairs, n_turns, drop_chunk=1):
    """Сколько токенов входа пересчитывает сервер, который помнит прошлый запрос сессии."""
    cached, per_turn = [], []
    for n in range(1, n_turns + 1):
        window, _ = assemble_context(system, memory, pairs[:n - 1], pairs[n - 1][0],
                                     drop_chunk=drop_chunk)
        seq = [(m.id, k) for m in window for k in range(m.tokens)]
        common = 0
        while common < min(len(seq), len(cached)) and seq[common] == cached[common]:
            common += 1
        per_turn.append(len(seq) - common)
        answer = [(m.id, k) for m in pairs[n - 1][1:] for k in range(m.tokens)]
        cached = seq + answer
    return per_turn


# ----------------------------------------------------------------- 11.3
class SummaryRecord(NamedTuple):
    id: str
    user: str
    session: str
    turns: tuple             # (a, b): какие ходы пересказаны
    source_ids: tuple        # какие сообщения пересказаны
    compressor: str          # чем сжимали: версия программы или модели
    template: str            # под какой шаблон диалога собрана
    n_tok: int               # сколько места займёт в окне
    created_at: str


class Probe(NamedTuple):
    pattern: str             # где искать ответ (строчными буквами)
    truth: str               # верный ответ по всей истории


PROBES = [Probe(r"зовут (\w+)", "аня"), Probe(r"(коротк)", "коротк"),
          Probe(r"жив\w* в (\w{5})", "самар"),
          Probe(r"(не |)е\w* арахис", "не "),
          Probe(r"python (\d+\.\d+)", "3.12"),
          Probe(r"(\d+ декабря)", "15 декабря")]

USER = ["Меня зовут Аня, я живу в Казани.", "Пиши коротко.",
        "Пишу на Python 3.12.", "Не ем арахис.",
        "С 1 ноября живу в Самаре.", "Дедлайн диплома 15 декабря.",
        "Помоги с планом введения.", "Сократи второй пункт."]
SUMMARY = ("Пользователя зовут Аня. Живёт в Казани. Просит отвечать "
           "коротко. Пишет на Python 3.12. Не ест арахис.")


def answer(probe, text):
    found = re.findall(probe.pattern, text.lower())
    return found[-1] if found else None


def measure(probes, text):
    """-> (сохранность фактов, точность утверждений текста).

    Утверждение — предложение, которое отвечает хотя бы на один зонд; оно верно, если верны все
    его ответы. Предложения без ответов на зонды в точность не входят."""
    right = sum(answer(p, text) == p.truth for p in probes)
    claims = []
    for c in re.split(r"(?<=[.!?])\s+", text.strip()):
        said = [answer(p, c) == p.truth for p in probes if answer(p, c) is not None]
        if said:
            claims.append(all(said))
    prec = sum(claims) / len(claims) if claims else 1.0
    return right / len(probes), prec


def rolling_summary(rounds=10, k=4, new_per_round=2, decay=0.9, seed=0):
    """Сводка на k фактов; вытесненный факт не вернётся: источника нет."""
    rng = random.Random(seed)
    kept = {"имя": (0.9, 0), "арахис": (0.95, 0), "python": (0.6, 0), "коротко": (0.5, 0)}
    probes = {"имя", "город", "арахис", "python", "дедлайн", "коротко"}
    history, left = [len(probes & kept.keys())], {}
    for r in range(1, rounds + 1):
        for j in range(new_per_round):
            kept[f"факт {r}.{j}"] = (rng.random(), r)
        score = {f: imp * decay ** (r - born) for f, (imp, born) in kept.items()}
        best = sorted(kept, key=lambda f: (-score[f], f))[:k]
        for f in kept.keys() - set(best):
            left[f] = r
        kept = {f: kept[f] for f in best}
        history.append(len(probes & kept.keys()))
    return history, left


# ----------------------------------------------------------------- проверки
def _anya():
    def pair(text=""):
        return (msg("user", 32, text), msg("assistant", 28))
    system = msg("system", S, "Ты помощник BookLM.", pinned=True)
    memory = msg("system", M, "[память] данные о пользователе")
    pairs = [pair("Меня зовут Аня, я живу в Казани"), pair("Пиши коротко")]
    pairs += [pair() for _ in range(10)]
    return system, memory, pairs


if __name__ == "__main__":
    # 11.1
    assert weights_bytes() == 13_244_928
    assert kv_bytes(1) == 1536 and kv_bytes(T_MAX) == 786_432
    assert kv_bytes(240) == 368_640 and kv_bytes(64) == 98_304
    assert requests_to_outweigh() == 17 and requests_to_outweigh(kv_heads=8) == 5

    # 11.2
    system, memory, pairs = _anya()
    assert H == 240
    for n in range(1, 13):
        window, dropped = assemble_context(system, memory, pairs[:n - 1], pairs[n - 1][0])
        assert check_window(window, dropped, system, memory, pairs[:n - 1], pairs[n - 1][0]) == []
        assert sum(m.tokens for m in window) == min(144 + 60 * (n - 1), 384)
    window, dropped = assemble_context(system, memory, pairs[:5], pairs[5][0])
    assert dropped == [pairs[0]]
    tail = truncate_from_front(system, memory, pairs[:5], pairs[5][0])
    assert len(tail) == 384
    assert "системная инструкция пропала" in diagnose(tail, system)
    assert any(m is pairs[0][0] for m, _ in tail)
    slide = recomputed_tokens(system, memory, pairs, 12, drop_chunk=1)
    chunk = recomputed_tokens(system, memory, pairs, 12, drop_chunk=2)
    assert slide == [144, 32, 32, 32, 32] + [272] * 7 and sum(slide) == 2176
    assert chunk == [144, 32, 32, 32, 32] + [212, 32] * 3 + [212] and sum(chunk) == 1216
    # закрепление: «Пиши коротко» переживает обрезку
    pinned = [pairs[0], (pairs[1][0]._replace(pinned=True), pairs[1][1])] + pairs[2:]
    window, dropped = assemble_context(system, memory, pinned[:6], pinned[6][0])
    assert pinned[1] not in dropped and len(dropped) == 2

    # 11.3
    assert measure(PROBES, " ".join(USER)) == (1.0, 5 / 6)     # зонды исправны; Казань устарела
    assert measure(PROBES, " ".join(USER[-4:])) == (2 / 6, 1.0)
    assert measure(PROBES, SUMMARY) == (4 / 6, 4 / 5)
    assert measure(PROBES, "Пользователя зовут Аня.") == (1 / 6, 1.0)   # «Проверь себя», вопрос 2
    history, left = rolling_summary()
    assert history == [4, 2, 2, 2, 2, 1, 1, 0, 0, 0, 0]
    assert left["python"] == left["коротко"] == 1 and left["имя"] == 5 and left["арахис"] == 7
    print("OK: числа разделов 11.1–11.3 сходятся")
