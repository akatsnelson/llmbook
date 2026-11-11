"""Глава 11, раздел 11.2 «Бюджет окна контекста»: листинг «Пересчитанные токены при двух политиках обрезки» (полная версия).

В книге оставлена сокращённая версия; этот файл самодостаточен и
выполняется в ноутбуке главы notebooks/11_memory.ipynb.

Моделирует сервер, который помнит предыдущий запрос сессии и берёт из кэша
общее начало. Считает пересчёт за 12 ходов при обрезке по одной паре (2176)
и блоками по две (1216), цену диалога без кэша (3240 и 37 800 токенов против
4140 и 311 400 без обрезки), а также цену неверного порядка (память после
истории, 3120) и обновления памяти на ходу 4.

Запуск из корня репозитория: python labs/ch11/prefix_reuse.py
"""

# --- сборщик контекста из раздела 11.2 (как в context_assembler.py) ---
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


# --- пересчёт токенов ---
def recomputed_tokens(n_turns, drop_chunk=1, memory_first=True,
                      memory_update_at=()):
    """Сколько токенов входа пересчитывает сервер, который помнит
    предыдущий запрос сессии (вход и ответ) и берёт общее начало."""
    cached, per_turn, mem = [], [], memory
    for n in range(1, n_turns + 1):
        if n in memory_update_at:                  # память обновилась
            mem = msg("system", M, "[память] новая версия")
        window, _ = assemble_context(system, mem, pairs[:n - 1],
                                     pairs[n - 1][0],
                                     drop_chunk=drop_chunk)
        if not memory_first:                  # память после истории
            window = [window[0], *window[2:-1], window[1], window[-1]]
        seq = [(m.id, k) for m in window for k in range(m.tokens)]
        common = 0
        while (common < min(len(seq), len(cached))
               and seq[common] == cached[common]):
            common += 1
        per_turn.append(len(seq) - common)
        answer = [(m.id, k) for m in pairs[n - 1][1:]
                  for k in range(m.tokens)]
        cached = seq + answer                      # KV ответа уже есть
    return per_turn

def input_tokens(n_turns, limit=True):         # без кэша: читаем всё
    def hist(n):
        h = 60 * (n - 1)
        return min(h, H // 60 * 60) if limit else h
    return sum(S + M + U + hist(n) for n in range(1, n_turns + 1))

slide = recomputed_tokens(12, drop_chunk=1)
chunk = recomputed_tokens(12, drop_chunk=2)
print("по одной:", slide, sum(slide))
print("по две:  ", chunk, sum(chunk))
assert slide == [144, 32, 32, 32, 32] + [272] * 7
assert sum(slide) == 2176 and sum(chunk) == 1216
assert input_tokens(10) == 3240 and input_tokens(100) == 37_800
assert input_tokens(10, limit=False) == 4140
assert input_tokens(100, limit=False) == 311_400

late = recomputed_tokens(12, memory_first=False)   # неверный порядок
assert late[1:5] == [156] * 4 and sum(late) == 3120
upd = recomputed_tokens(12, memory_update_at={4})  # память сменилась
assert upd[3] == 324 - S
