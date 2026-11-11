"""Листинги главы 10 «Работа с промптом» книги «Большие языковые модели», файл 03-reasoning-budget.

Код и вывод — ровно как в книге; текст книги сюда не входит. Листинги идут по порядку и
опираются друг на друга: запускайте файл целиком или ячейками сверху вниз (разметка # %%).
Проверка, что вывод совпал с книгой: python scripts/check_listings.py listings/ch10/03-reasoning-budget.py
"""

# %% Листинг «Фиксированная глубина и записанные шаги» — раздел «Глубина вычисления на токен»
DEPTH = 6                    # переходов за один проход: по одному на блок
nxt = [6, 9, 7, 8, 2, 0, 3, 5, 1, 4]     # цикл 0-6-3-8-1-9-4-2-7-5-0

def hops(cell, n):
    for _ in range(n):
        cell = nxt[cell]
    return cell

def one_token(cell, steps_left):
    # один токен успевает не больше DEPTH последовательных переходов
    return hops(cell, min(steps_left, DEPTH))

def with_scratchpad(cell, steps):
    written = []                         # записанные промежуточные токены
    while steps > 0:
        cell = one_token(cell, steps)
        steps -= min(steps, DEPTH)
        written.append(cell)
    return written

print("верный ответ:   ", hops(0, 15))
print("сразу ответ:    ", one_token(0, 15))
print("с записью шагов:", with_scratchpad(0, 15))
# Вывод в книге:
# | верный ответ:    9
# | сразу ответ:     4
# | с записью шагов: [4, 3, 9]

# %% Листинг «Просьба считать по шагам на BookLM» — раздел «Глубина вычисления на токен»
from booklm.lab import load, prompt_ids
from booklm.experiments.ch10 import continue_ids

tok, model = load("instruct")
for q in ("Сколько будет 17 * 24?",
          "Сколько будет 17 * 24? Давай считать по шагам."):
    new = continue_ids(model, prompt_ids(tok, q, chat=True), 12)  # жадно
    print(f"{q}\n -> {tok.decode(new)}")
# Вывод в книге:
# | Сколько будет 17 * 24?
# |  -> Жил-был два друга, которые любили играть вместе. Однажды
# | Сколько будет 17 * 24? Давай считать по шагам.
# |  -> Жил-был два друга, которые любили считать. Один был

# %% Листинг «Бюджет окна и KV-кэша» — раздел «Бюджет вычислений во время ответа»
N_PARAMS = 6_622_464                     # числа BookLM из главы 4
L, H_Q, H_KV, D_H, T_MAX = 6, 8, 2, 32, 512
KV_PER_TOKEN = 2 * L * H_KV * D_H * 2    # K и V, по 2 байта на число

def kv_bytes(T):
    return T * KV_PER_TOKEN

def budget_check(P, R, n_ans, t_max=T_MAX):
    total = P + R + n_ans
    if total > t_max:
        raise ValueError(f"{P}+{R}+{n_ans}={total} > {t_max}: "
                         "окно переполнено")
    return total

print("байт KV на токен:", KV_PER_TOKEN)
total = budget_check(280, 200, 30)
kv = kv_bytes(total)
print(f"токенов: {total}, KV: {kv} байт = {kv / 1024:.0f} КБ")
try:
    budget_check(280, 210, 30)
except ValueError as err:
    print("ошибка:", err)
# Вывод в книге:
# | байт KV на токен: 1536
# | токенов: 510, KV: 783360 байт = 765 КБ
# | ошибка: 280+210+30=520 > 512: окно переполнено

# %% Листинг «Операции на ответ с рассуждением и без» — раздел «Бюджет вычислений во время ответа»
def flops_per_token(t):
    # токен на позиции t: все параметры плюс внимание к t позициям
    return 2 * N_PARAMS + 4 * L * H_Q * D_H * t

def generation_flops(P, n):
    # n токенов, написанных после промпта из P токенов
    return sum(flops_per_token(t) for t in range(P + 1, P + n + 1))

read = generation_flops(0, 280)          # чтение промпта: позиции 1..280
cot = generation_flops(280, 230)         # рассуждение 200 + ответ 30
direct = generation_flops(280, 30)       # только ответ
print(f"внимание на позицию: {flops_per_token(1) - 2 * N_PARAMS} FLOP")
print(f"чтение промпта: {read / 1e9:.2f} GFLOP")
print(f"ответ с рассуждением: {cot / 1e9:.2f},",
      f"без: {direct / 1e9:.2f} GFLOP")
ratio = (read + cot) / (read + direct)
print(f"весь запрос с рассуждением дороже в {ratio:.1f} раза")
# Вывод в книге:
# | внимание на позицию: 6144 FLOP
# | чтение промпта: 3.95 GFLOP
# | ответ с рассуждением: 3.61, без: 0.45 GFLOP
# | весь запрос с рассуждением дороже в 1.7 раза

# %% Листинг «Штраф наблюдателя в награде» — раздел «Мониторинг рассуждения»
import torch
# способы: честно, взлом с признанием, взлом молча
TESTS_PASS = torch.tensor([0.5, 1.0, 1.0])       # средняя награда тестов
SAYS_HACK = torch.tensor([0.0, 1.0, 0.0])        # наблюдатель видит взлом

def train(penalty, steps, lr=1.0):
    z = torch.tensor([0.80, 0.15, 0.05]).log()   # стартовые доли способов
    for _ in range(steps):
        p = z.softmax(0)
        r = TESTS_PASS - penalty * SAYS_HACK     # награда каждого способа
        z = z + lr * p * (r - p @ r)             # шаг градиента
    return z.softmax(0)

for penalty in (0.0, 1.0):
    print(f"штраф за слова наблюдателя: {penalty}")
    for steps in (0, 20, 200):
        p = train(penalty, steps)
        hacks = p[1] + p[2]
        print(f"  шаг {steps:>3}: честно {p[0]:.2f}, взлом {hacks:.2f},",
              f"наблюдатель видит {p[1] / hacks:.0%}")
# Вывод в книге:
# | штраф за слова наблюдателя: 0.0
# |   шаг   0: честно 0.80, взлом 0.20, наблюдатель видит 75%
# |   шаг  20: честно 0.12, взлом 0.88, наблюдатель видит 92%
# |   шаг 200: честно 0.01, взлом 0.99, наблюдатель видит 98%
# | штраф за слова наблюдателя: 1.0
# |   шаг   0: честно 0.80, взлом 0.20, наблюдатель видит 75%
# |   шаг  20: честно 0.87, взлом 0.13, наблюдатель видит 42%
# |   шаг 200: честно 0.01, взлом 0.99, наблюдатель видит 0%

# %% Листинг «Точная вероятность победы голосования» — раздел «Самосогласованность»
from itertools import combinations_with_replacement
from math import factorial

def vote_win_prob(probs, k):
    # probs[0] -- вероятность верного ответа; ничьи делятся поровну
    total, m = 0.0, len(probs)
    for votes in combinations_with_replacement(range(m), k):
        counts = [votes.count(i) for i in range(m)]  # голоса ответов
        pr = factorial(k)              # число порядков выборок раскладки
        for c, p in zip(counts, probs):
            pr *= p**c / factorial(c)
        winners = [i for i, c in enumerate(counts) if c == max(counts)]
        if 0 in winners:
            total += pr / len(winners)
    return total

print(f"{'k:':16}", *[f"{k:>5}" for k in (1, 3, 5, 9)])
for name, probs in [("p = 0.6", [0.6, 0.4]), ("p = 0.4", [0.4, 0.6]),
                    ("рассеянные", [0.4, 0.2, 0.2, 0.2]),
                    ("систематическая", [0.3, 0.45, 0.25]),
                    ("мода неверна", [0.30, 0.31] + [0.03] * 13)]:
    row = [f"{vote_win_prob(probs, k):.3f}" for k in (1, 3, 5, 9)]
    print(f"{name + ':':16}", *row)
# Вывод в книге:
# | k:                   1     3     5     9
# | p = 0.6:         0.600 0.648 0.683 0.733
# | p = 0.4:         0.400 0.352 0.317 0.267
# | рассеянные:      0.400 0.448 0.509 0.599
# | систематическая: 0.300 0.284 0.269 0.241
# | мода неверна:    0.300 0.331 0.405 0.456

# %% Листинг «Цена запроса с несколькими цепочками» — раздел «Самосогласованность»
def request_cost(P, R, n_ans, k=1, shared_prefix=True):
    budget_check(P, R, n_ans)          # у каждой цепочки своё окно
    tokens = P + k * (R + n_ans) if shared_prefix else k * (P + R + n_ans)
    reads = 1 if shared_prefix else k  # сколько раз читаем промпт
    flops = (reads * generation_flops(0, P)
             + k * generation_flops(P, R + n_ans))
    return {"токенов в KV": tokens,
            "KV, МБ": round(kv_bytes(tokens) / 2**20, 2),
            "написано": k * (R + n_ans),
            "GFLOP": round(flops / 1e9, 1)}

print(request_cost(280, 200, 30))
print(request_cost(280, 200, 30, k=5))
print(request_cost(280, 200, 30, k=5, shared_prefix=False))
# Вывод в книге:
# | {'токенов в KV': 510, 'KV, МБ': 0.75, 'написано': 230, 'GFLOP': 7.6}
# | {'токенов в KV': 1430, 'KV, МБ': 2.09, 'написано': 1150, 'GFLOP': 22.0}
# | {'токенов в KV': 2550, 'KV, МБ': 3.74, 'написано': 1150, 'GFLOP': 37.8}

# %% Листинг «Голосование имён на BookLM» — раздел «Самосогласованность»
import re
from collections import Counter
from booklm.lab import generate, next_token_table

base = load("base")[1]
prompt = "Жила-была девочка по имени"
top = next_token_table(tok, base, prompt, k=5)
print("таблица:", ", ".join(f"«{s.strip()}» {p:.2f}" for s, p in top))
for tau in (0.1, 0.8, 1.5):
    names = []
    for seed in range(9):
        text = generate(tok, base, prompt, max_new=4, temperature=tau,
                        top_k=0, seed=seed)
        names.append(re.findall(r"\w+", text[len(prompt):])[0])  # 1-е слово
    winner, n = Counter(names).most_common(1)[0]
    print(f"tau={tau}: {' '.join(names)}")
    print(f"    различных {len(set(names))}, " +
          (f"победитель {winner}: {n} из 9" if n > 1 else "победителя нет"))
# Вывод в книге:
# | таблица: «Люси» 0.12, «Джейн» 0.10, «Сью» 0.09, «Салли» 0.07, «Лили» 0.07
# | tau=0.1: Люси Люси Люси Люси Люси Люси Люси Люси Сью
# |     различных 2, победитель Люси: 8 из 9
# | tau=0.8: Лора Ами Тру Лили Джука Люси Лиззи Салли Сью
# |     различных 9, победителя нет
# | tau=1.5: Лилли Ами Тру Мисс Джука Лиза Лиз Элли Олив
# |     различных 9, победителя нет
