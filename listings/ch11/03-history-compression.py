"""Листинги главы 11 «Контекст и память» книги «Большие языковые модели», файл 03-history-compression.

Код и вывод — ровно как в книге; текст книги сюда не входит. Листинги идут по порядку и
опираются друг на друга: запускайте файл целиком или ячейками сверху вниз (разметка # %%).
Проверка, что вывод совпал с книгой: python scripts/check_listings.py listings/ch11/03-history-compression.py
"""

# %% Листинг «Запись сводки с источниками» — раздел «Запись сводки»
from typing import NamedTuple

class SummaryRecord(NamedTuple):
    id: str
    user: str
    pairs: tuple          # (a, b): какие пары разговора пересказаны
    source_ids: tuple     # из каких ходов взяты эти пары
    compressor: str       # чем сжимали: версия программы или модели
    template: str         # под какой шаблон диалога собрана
    n_tok: int            # сколько места займёт в окне
    created_at: str

sources = ("s1/t2", "s1/t3", "s1/t4", "s1/t9", "s2/t5", "s2/t6")
rec = SummaryRecord("sum-1", "u42", (1, 6), sources,
                    "compress-v1", "chat-v1", 100, "2026-11-12 19:40")
print(rec.pairs, len(rec.source_ids), "ходов:", ", ".join(rec.source_ids))
print("сжатие:", 6 * 60 / rec.n_tok)
# Вывод в книге:
# | (1, 6) 6 ходов: s1/t2, s1/t3, s1/t4, s1/t9, s2/t5, s2/t6
# | сжатие: 3.6

# %% Листинг «Вопросы-зонды и две метрики потерь» — раздел «Вычисление метрик потерь»
import re

class Probe(NamedTuple):
    pattern: str   # где в тексте искать ответ (строчными буквами)
    truth: str     # верный ответ по всей истории

PROBES = [Probe(r"зовут (\w+)", "аня"), Probe(r"(коротк)", "коротк"),
          Probe(r"жив\w* в (\w{5})", "самар"),
          Probe(r"(не |)е\w* арахис", "не "),
          Probe(r"python (\d+\.\d+)", "3.12"),
          Probe(r"(\d+ декабря)", "15 декабря")]

def answer(probe, text):
    found = re.findall(probe.pattern, text.lower())
    return found[-1] if found else None   # побеждает последнее упоминание

def measure(probes, text):
    """-> (сохранность фактов, точность утверждений текста)."""
    right = sum(answer(p, text) == p.truth for p in probes)
    claims = []                          # верно ли каждое утверждение
    for c in re.split(r"(?<=[.!?])\s+", text.strip()):    # предложения
        said = [answer(p, c) == p.truth for p in probes
                if answer(p, c) is not None]    # на какие зонды отвечает
        if said:                         # хотя бы на один: утверждение
            claims.append(all(said))
    prec = sum(claims) / len(claims) if claims else 1.0  # нечего проверять
    return right / len(probes), prec

# %% Листинг «Обрезка и сводка на шести зондах» — раздел «Вычисление метрик потерь»
USER = ["Меня зовут Аня, я живу в Казани.", "Пиши коротко.",   # пары 1-8
        "Пишу на Python 3.12.", "Не ем арахис.",
        "С 1 ноября живу в Самаре.", "Дедлайн диплома 15 декабря.",
        "Помоги с планом введения.", "Сократи второй пункт."]
SUMMARY = ("Пользователя зовут Аня. Живёт в Казани. Просит отвечать "
           "коротко. Пишет на Python 3.12. Не ест арахис.")  # пары 1-6

for name, text in [("вся история", " ".join(USER)),
                   ("обрезка", " ".join(USER[-4:])),
                   ("сводка", SUMMARY)]:
    ret, prec = measure(PROBES, text)
    print(f"{name}: сохранность {ret:.2f}, точность {prec:.2f}")
print("город по сводке:", answer(PROBES[2], SUMMARY))
# Вывод в книге:
# | вся история: сохранность 1.00, точность 0.83
# | обрезка: сохранность 0.33, точность 1.00
# | сводка: сохранность 0.67, точность 0.80
# | город по сводке: казан

# %% Листинг «Обрезка и сводка на BookLM» — раздел «Вычисление метрик потерь»
import re
import torch
from booklm.lab import load, special
from booklm.experiments.ch11 import cut_points, stem

tok, model = load("base")
bos, K = special(tok)["<bos>"], 60
hits, length, first = dict.fromkeys("абв", 0), dict.fromkeys("абв", 0), {}
tales = cut_points(tok, n=20, k=K)        # перед c: K токенов без имени
for story, name, c in tales:
    summary = tok.encode(f"Главного героя зовут {name}.").ids
    inputs = {"а": story[:c],                         # всё начало сказки
              "б": [bos] + story[c - K:c],            # обрезка: K токенов
              "в": [bos] + summary + story[c - K:c]}  # сводка + те же K
    for kind, ids in inputs.items():
        out = model.generate(torch.tensor([ids]), max_new=60,
                             temperature=0, eos_id=2)       # жадно
        text = tok.decode(out[0, len(ids):].tolist())
        hits[kind] += bool(re.search(stem(name), text))
        length[kind] += len(ids)
        first.setdefault(kind, re.split(r"(?<=[.!?»])\s+", text.strip()))
for kind in "абв":
    print(f"({kind}) вход {length[kind] / len(tales):3.0f} токенов,"
          f" имя героя в {hits[kind]:2} продолжениях из {len(tales)}")
for kind in "бв":                 # первая сказка: предложения 3 и 4
    print(f"({kind})", *first[kind][2:4])
# Вывод в книге:
# | (а) вход 226 токенов, имя героя в 15 продолжениях из 20
# | (б) вход  61 токенов, имя героя в  1 продолжениях из 20
# | (в) вход  69 токенов, имя героя в 13 продолжениях из 20
# | (б) Я буду считать до десяти, а потом посмотрим». Дети побежали к дереву.
# | (в) Я буду считать до десяти, а потом найдём тебя». Лили и её родители побежали к дереву.

# %% Листинг «Мусор в окне BookLM» — раздел «Инженерия контекста в агентных сессиях»
import math, statistics
from booklm.lab import load, special
from booklm.experiments.ch11 import cut_points
from booklm.experiments.ch11_context_rot import first_sentence, judge, junk

tok, model = load("base")
bos, K = special(tok)["<bos>"], 60
tales, res = cut_points(tok, n=40, k=K), {}
for j, (story, name, c) in enumerate(tales):
    head, recent = first_sentence(tok, story), story[c - K:c]
    trash, heroes = junk(tok, j)              # 360 токенов чужих сказок
    note = tok.encode(f"Главного героя зовут {name}.").ids
    inputs = {"чистое окно": [bos] + head + recent}
    for L in (120, 240, 360):
        inputs[f"мусор {L}"] = [bos] + head + trash[:L] + recent
    inputs["мусор 360 + заметка"] = [bos] + head + trash + note + recent
    inputs["без начала"] = [bos] + recent
    for kind, ids in inputs.items():          # свой герой, чужой, -log p
        r = judge(tok, model, ids, story, c, name, heroes)
        res.setdefault(kind, []).append((len(ids), *r))
print(f"{'вход':20}{'токенов':>8}{'свой':>6}{'чужой':>7}{'перплексия':>12}")
for kind, r in res.items():
    n, own, other, nll = zip(*r)
    ppl = math.exp(statistics.mean(nll))
    print(f"{kind:20}{sum(n) / len(r):8.0f}{sum(own):6}{sum(other):7}",
          f"{ppl:11.2f}")
# Вывод в книге:
# | вход                 токенов  свой  чужой  перплексия
# | чистое окно               72    38      0        8.57
# | мусор 120                192    10     20       11.05
# | мусор 240                312     3     22       12.09
# | мусор 360                432     3     20       12.64
# | мусор 360 + заметка      440    18     14       10.86
# | без начала                61     1      0       10.88

# %% Листинг «Просьба по частям» — раздел «Падение качества в многоходовом диалоге»
from booklm.lab import load
from booklm.experiments.ch11_multiturn import chat, requests, score

tok, model = load("instruct")
res = {}
for turns_by_kind, h_re, t_re in requests():     # 36 просьб
    for seed in (0, 1):
        for kind, turns in turns_by_kind.items():
            text, n = chat(tok, model, turns, seed)
            res.setdefault(kind, []).append((n, *score(text, h_re, t_re)))
print(f"{'способ':24}{'окно':>6}{'герой':>7}{'предмет':>9}{'оба':>5}")
for kind, r in res.items():
    n, hero, thing = zip(*r)
    both = sum(a and b for a, b in zip(hero, thing))
    print(f"{kind:24}{sum(n) / len(r):6.0f}{sum(hero):7}{sum(thing):9}{both:5}")
# Вывод в книге:
# | способ                    окно  герой  предмет  оба
# | целиком                    131     63       47   41
# | части в одной реплике      136     47       32   22
# | части в трёх репликах      346      4       59    2
# | три реплики + повтор       356     46       28   18
