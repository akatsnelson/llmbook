"""Листинги главы 11 «Контекст и память» книги «Большие языковые модели», файл 02-context-window.

Код и вывод — ровно как в книге; текст книги сюда не входит. Листинги идут по порядку и
опираются друг на друга: запускайте файл целиком или ячейками сверху вниз (разметка # %%).
Проверка, что вывод совпал с книгой: python scripts/check_listings.py listings/ch11/02-context-window.py
"""

# %% Листинг «Сборщик контекста с бюджетом BookLM» — раздел «Сборщик контекста»
from typing import NamedTuple

T_MAX, RESERVE, S, M = 512, 128, 48, 64   # окно, резерв, система, память

class Msg(NamedTuple):
    role: str
    tokens: int            # длина сообщения в токенах, задана числом
    text: str = ""
    pinned: bool = False
    id: int = 0            # номер сообщения

last_id = 0                # счётчик номеров сообщений

def msg(role, tokens, text="", pinned=False):
    global last_id
    last_id += 1
    return Msg(role, tokens, text, pinned, last_id)

def count_tokens(m):
    return m.tokens              # ЗАГЛУШКА: нужен токенизатор модели

def pairs_tokens(pairs, count=count_tokens):
    return sum(count(m) for p in pairs for m in p)

def assemble_context(system, memory, history, new,
                     count=count_tokens, drop_chunk=1):
    fixed = count(system) + count(memory) + count(new)
    kept, dropped = list(history), []
    while fixed + pairs_tokens(kept, count) > T_MAX - RESERVE:
        free = [p for p in kept if not any(m.pinned for m in p)]
        if not free:
            raise ValueError("закреплённое не помещается в бюджет")
        for p in free[:drop_chunk]:          # самые старые незакреплённые
            kept.remove(p)
            dropped.append(p)
    window = [system, memory] + [m for p in kept for m in p] + [new]
    return window, dropped

# %% Листинг «Ход 6: выпадает первая пара» — раздел «Сборщик контекста»
pair = lambda text="": (msg("user", 32, text), msg("assistant", 28))
system = msg("system", S, "Ты помощник.", pinned=True)
memory = msg("system", M, "[память] данные о пользователе")
pairs = [pair("Меня зовут Аня, я живу в Казани"), pair("Пиши коротко")]
pairs += [pair() for _ in range(10)]

for n in (5, 6):                     # на ходу n в истории n-1 пар
    window, dropped = assemble_context(system, memory, pairs[:n - 1],
                                       pairs[n - 1][0])
    print(f"ход {n}: вход {sum(m.tokens for m in window)},",
          "выпало:", [p[0].text for p in dropped])
# Вывод в книге:
# | ход 5: вход 384, выпало: []
# | ход 6: вход 384, выпало: ['Меня зовут Аня, я живу в Казани']

# %% Листинг «Намеренно сломанная обрезка с начала строки токенов» — раздел «Обрезка строки с начала»
def truncate_from_front(system, memory, history, new,
                        budget=T_MAX - RESERVE):
    # ОШИБКА: всё склеено в одну строку, оставлены последние budget
    stream = [system, memory] + [m for p in history for m in p] + [new]
    # токен = (сообщение, номер токена внутри сообщения)
    tokens = [(m, k) for m in stream for k in range(m.tokens)]
    return tokens[-budget:]

def diagnose(tail):
    problems = []
    if all(m is not system for m, _ in tail):
        problems.append("системная инструкция пропала")
    first, k = tail[0]
    if k > 0:
        problems.append(
            f"«{first.text}» начато с токена {k} из {first.tokens}")
    return problems

tail = truncate_from_front(system, memory, pairs[:5], pairs[5][0])
print("длина входа:", len(tail))                   # это ход 6
print("город в окне:", any(m is pairs[0][0] for m, _ in tail))
print("диагноз:", *diagnose(tail), sep="\n  ")
# Вывод в книге:
# | длина входа: 384
# | город в окне: True
# | диагноз:
# |   системная инструкция пропала
# |   «[память] данные о пользователе» начато с токена 12 из 64

# %% Листинг «Реплики Ани в токенах BookLM» — раздел «Обрезка строки с начала»
from booklm.lab import load, encode_text, generate, special

tok, model = load("instruct")
USER = ["Меня зовут Аня, я живу в Казани.", "Пиши коротко.",
        "Пишу на Python 3.12.", "Не ем арахис.",
        "С 1 ноября живу в Самаре.", "Дедлайн диплома 15 декабря.",
        "Помоги с планом введения.", "Сократи второй пункт."]
for s in USER:
    words, n = len(s.split()), len(encode_text(tok, s))
    print(f"{s:<34}слов {words}, токенов {n:2}, на слово {n / words:.1f}")
for s in ("Python 3.12", "Дедлайн", "диплома", " сказку"):
    print(f"«{s}»:", "|".join(tok.decode([i]) for i in encode_text(tok, s)))
# Вывод в книге:
# | Меня зовут Аня, я живу в Казани.  слов 7, токенов 12, на слово 1.7
# | Пиши коротко.                     слов 2, токенов  6, на слово 3.0
# | Пишу на Python 3.12.              слов 4, токенов 17, на слово 4.2
# | Не ем арахис.                     слов 3, токенов  7, на слово 2.3
# | С 1 ноября живу в Самаре.         слов 6, токенов 13, на слово 2.2
# | Дедлайн диплома 15 декабря.       слов 4, токенов 17, на слово 4.2
# | Помоги с планом введения.         слов 4, токенов  8, на слово 2.0
# | Сократи второй пункт.             слов 3, токенов 12, на слово 4.0
# | «Python 3.12»: P|y|t|h|o|n| |3|.|1|2
# | «Дедлайн»: Д|ед|л|ай|н
# | «диплома»: ди|п|лом|а
# | « сказку»:  сказ|ку

# %% Листинг «Окно седьмого хода в номерах BookLM» — раздел «Обрезка строки с начала»
answers = [generate(tok, model, s, chat=True, max_new=60, temperature=0)
           for s in USER]                       # ответы сказочника
rule = msg("system", 0, "Ты сказочник BookLM. Пиши коротко.", pinned=True)
facts = msg("system", 0, "Аня живёт в Самаре. Не ест арахис.")
chat = [(msg("user", 0, u), msg("assistant", 0, a))
        for u, a in zip(USER, answers)]
count = lambda m: len(encode_text(tok, m.text)) + 1   # +1: метка роли

def render(window, sp=special(tok)):            # шаблон BookLM
    head = window[0].text + " " + window[1].text + "\n"
    ids = [sp["<bos>"]]
    for m in window[2:]:
        if m.role == "user":            # система и память текстом
            ids += [sp["<|user|>"]] + encode_text(tok, head + m.text)
            ids += [sp["<|assistant|>"]]
            head = ""
        else:
            ids += encode_text(tok, m.text) + [sp["<eos>"]]
    return ids

window, dropped = assemble_context(rule, facts, chat[:6], chat[6][0],
                                   count=count)
print("ход 7, вход по счётчику:", sum(map(count, window)))
print("выпали:", [p[0].text for p in dropped])
print("после шаблона:", len(render(window)), "токенов")
# Вывод в книге:
# | ход 7, вход по счётчику: 343
# | выпали: ['Меня зовут Аня, я живу в Казани.', 'Пиши коротко.']
# | после шаблона: 348 токенов

# %% Листинг «Пересчитанные токены при двух политиках обрезки» — раздел «Кэширование общего префикса»
def recomputed_tokens(n_turns, drop_chunk=1):
    """Сколько токенов входа пересчитывает сервер, который помнит
    предыдущий запрос сессии (вход и ответ) и берёт общее начало."""
    cached, per_turn = [], []
    for n in range(1, n_turns + 1):
        window, _ = assemble_context(system, memory, pairs[:n - 1],
                                     pairs[n - 1][0],
                                     drop_chunk=drop_chunk)
        seq = [(m.id, k) for m in window for k in range(m.tokens)]
        common = 0                           # длина совпавшего начала
        while (common < min(len(seq), len(cached))
               and seq[common] == cached[common]):
            common += 1
        per_turn.append(len(seq) - common)
        answer = [(m.id, k) for m in pairs[n - 1][1:]
                  for k in range(m.tokens)]
        cached = seq + answer                # KV ответа уже есть
    return per_turn

for chunk in (1, 2):
    turns = recomputed_tokens(12, drop_chunk=chunk)
    print(f"по {chunk}: ходы 5-12 {turns[4:]}, всего {sum(turns)}")
# Вывод в книге:
# | по 1: ходы 5-12 [32, 272, 272, 272, 272, 272, 272, 272], всего 2176
# | по 2: ходы 5-12 [32, 212, 32, 212, 32, 212, 32, 212], всего 1216

# %% Листинг «Сколько времени экономит общий префикс» — раздел «Кэширование общего префикса»
# Проверка этот листинг не запускает: ему нужны сеть, видеокарта, сервер или внешний файл.
# import statistics, time, torch
# from booklm.lab import load, val_tokens
#
# tok, model = load("base")
# ids = val_tokens()[:384][None]              # вход шестого хода: 384 токена
#
# @torch.no_grad()
# def read_time(cached, repeats=20):          # медиана в мс и логиты
#     caches = [{"len": 0} for _ in model.blocks]
#     if cached:                              # первые cached токенов в кэш
#         model(ids[:, :cached], caches)
#         caches = [dict(c, len=cached) for c in caches]
#     times = []
#     for _ in range(repeats + 1):            # первый повтор: разогрев
#         copy = [dict(c) for c in caches]    # старые тензоры не меняются
#         t0 = time.perf_counter()
#         z = model(ids[:, cached:], copy)[0, -1]
#         times.append((time.perf_counter() - t0) * 1000)
#     return statistics.median(times[1:]), z
#
# (ta, za), (tb, zb), (tc, zc) = (read_time(n) for n in (0, 352, 112))
# print(f"(а) 384 заново:        {ta:5.1f} мс")
# print(f"(б) 352 из кэша + 32:  {tb:5.1f} мс, в {ta / tb:.1f} раза быстрее")
# print(f"(в) 112 из кэша + 272: {tc:5.1f} мс, {tc / ta:.0%} от (а)")
# print(f"разница логитов: (а)-(б) {(za - zb).abs().max():.0e},"
#       f" (а)-(в) {(za - zc).abs().max():.0e}")
