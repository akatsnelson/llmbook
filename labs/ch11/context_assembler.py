"""Глава 11, раздел 11.2 «Бюджет окна контекста»: листинг «Сборщик контекста с бюджетом BookLM» (полная версия).

В книге оставлена сокращённая версия; этот файл самодостаточен и
выполняется в ноутбуке главы notebooks/11_memory.ipynb.

Собирает окно BookLM из системы, блока памяти, истории пар и нового
сообщения в бюджет 384 токена, проверяет все шесть обещаний сборщика
(check_contract) на каждом из двенадцати ходов, держит вызов инструмента
в одной паре с вопросом и показывает закрепление «Пиши коротко».

Запуск из корня репозитория: python labs/ch11/context_assembler.py
"""

from itertools import count as id_counter
from typing import NamedTuple

T_MAX, RESERVE = 512, 128        # окно BookLM и резерв ответа
S, M, U = 48, 64, 32             # система, память, новое сообщение
H = T_MAX - RESERVE - S - M - U  # доля истории
KV_PER_TOKEN = 1536              # байт KV-кэша на токен (глава 4)
assert H == 240
_ids = id_counter()

class Msg(NamedTuple):
    role: str                    # system | user | assistant | tool
    tokens: int                  # длина в токенах, задана числом
    text: str = ""
    pinned: bool = False
    id: int = 0

def msg(role, tokens, text="", pinned=False):
    return Msg(role, tokens, text, pinned, next(_ids))

def count_tokens(m):
    return m.tokens              # ЗАГЛУШКА: нужен токенизатор модели

def assemble_context(system, memory, history, new, count=count_tokens,
                     t_max=T_MAX, reserve=RESERVE, drop_chunk=1):
    """history: пары от старых к новым; пара — кортеж сообщений.
    Возвращает (окно, выброшенные пары)."""
    budget = t_max - reserve
    fixed = count(system) + count(memory) + count(new)
    if fixed > budget:
        raise ValueError("система, память и вопрос не помещаются")
    kept, dropped = list(history), []
    size = lambda ps: sum(count(m) for p in ps for m in p)
    while fixed + size(kept) > budget:
        free = [p for p in kept if not any(m.pinned for m in p)]
        if not free:
            raise ValueError("закреплённое не помещается в бюджет")
        for p in free[:drop_chunk]:      # самые старые незакреплённые
            kept.remove(p)
            dropped.append(p)
    window = [system, memory] + [m for p in kept for m in p] + [new]
    return window, dropped

def check_contract(window, dropped, system, new, history,
                   count=count_tokens):
    assert sum(count(m) for m in window) <= T_MAX - RESERVE  # бюджет
    assert window[0] is system and window[-1] is new        # на месте
    source = [system] + [m for p in history for m in p] + [new]
    known = source + [window[1]]                            # + память
    assert all(any(m is x for x in known) for m in window)  # целые
    order = {m.id: i for i, m in enumerate(source)}
    pos = [order[m.id] for m in window if m.id in order]
    assert pos == sorted(pos)                               # порядок
    inside = {m.id for m in window}
    for p in history:                                       # неделимы
        flags = [m.id in inside for m in p]
        assert all(flags) or not any(flags)
    kept = sum(1 for p in history if p[0].id in inside)
    assert kept + len(dropped) == len(history)              # не молча

def pair(user_text="", tool=False):      # пара = 60 токенов: 32 + 28
    if tool:     # вызов инструмента и результат — в одной паре
        return (msg("user", 32, user_text),
                msg("assistant", 8, "вызов: погода"),
                msg("tool", 12, "результат"), msg("assistant", 8))
    return (msg("user", 32, user_text), msg("assistant", 28))

system = msg("system", S, "Ты помощник BookLM.", pinned=True)
memory = msg("system", M, "[память] данные о пользователе")
pairs = [pair("Меня зовут Аня, я живу в Казани"),
         pair("Пиши коротко"),
         pair("Какая погода в субботу?", tool=True)]
pairs += [pair() for _ in range(9)]
assert all(sum(m.tokens for m in p) == 60 for p in pairs)

for n in range(1, 13):              # ход n: в истории пары 1..n-1
    new, history = pairs[n - 1][0], pairs[:n - 1]
    window, dropped = assemble_context(system, memory, history, new)
    check_contract(window, dropped, system, new, history)
    if n == 5:
        assert sum(m.tokens for m in window) == 384
        assert dropped == []
    if n == 6:
        assert dropped == [pairs[0]]           # выпала ровно пара 1
        print("ход 6, выпала пара:", dropped[0][0].text)

# закрепим «Пиши коротко»: на ходу 7 выпадут пары 1 и 3
pinned = (msg("user", 32, "Пиши коротко", pinned=True), pairs[1][1])
history = [pairs[0], pinned] + pairs[2:6]
window, dropped = assemble_context(system, memory, history, pairs[6][0])
check_contract(window, dropped, system, pairs[6][0], history)
assert dropped == [pairs[0], pairs[2]] and pinned[0] in window

assert 384 * KV_PER_TOKEN == 589_824    # 576 КБ на вход
assert 512 * KV_PER_TOKEN == 786_432    # 768 КБ с ответом: паспорт
