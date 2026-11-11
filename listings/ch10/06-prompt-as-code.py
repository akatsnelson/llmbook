"""Листинги главы 10 «Работа с промптом» книги «Большие языковые модели», файл 06-prompt-as-code.

Код и вывод — ровно как в книге; текст книги сюда не входит. Листинги идут по порядку и
опираются друг на друга: запускайте файл целиком или ячейками сверху вниз (разметка # %%).
Проверка, что вывод совпал с книгой: python scripts/check_listings.py listings/ch10/06-prompt-as-code.py
"""

# %% Листинг «Запись версии промпта» — раздел «Состав версии промпта»
import hashlib
import json
from pathlib import Path

def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def sha256_json(obj):              # ключи по алфавиту, без пробелов
    return sha256_text(json.dumps(obj, sort_keys=True, ensure_ascii=False,
                                  separators=(",", ":")))

def sha256_file(path):             # веса, адаптер, токенизатор
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def prompt_version(template, system, examples, grammar, decoding, prefix,
                   model):
    parts = {"template": sha256_text(template),
             "system": sha256_text(system),
             "examples": sha256_json(examples),      # порядок важен
             "grammar": sha256_text(grammar),
             "decoding": sha256_json(decoding),
             "prefix": sha256_json(prefix),
             "model": sha256_json(model)}            # хеши файлов модели
    parts["version"] = sha256_json(parts)           # общий хеш версии
    return parts

template = ('<bos>{% for m in messages %}'
            '{% if m.role == "user" %}<|user|>{{ m.content }}'
            '{% else %}<|assistant|>{{ m.content }}<eos>{% endif %}'
            '{% endfor %}'
            '{% if add_generation_prompt %}<|assistant|>{% endif %}')
system = "Определи тональность отзыва. Ответь JSON."
examples = [{"input": "Отличная книга", "output": '{"label":"pos"}'},
            {"input": "Скучно и долго", "output": '{"label":"neg"}'}]
grammar = r'\{"label":"(pos|neg)"\}'
decoding = {"temperature": 0.0, "top_p": 1.0,
            "max_new_tokens": 16, "seed": 0}
prefix = {"page_size": 16, "shared_pages": 28}   # 448 токенов, раздел 10.5
model = {"weights": sha256_file("models/booklm/booklm-instruct.pt"),
         "tokenizer": sha256_file("models/booklm/tokenizer.json")}
rest = (grammar, decoding, prefix, model)

v1 = prompt_version(template, system, examples, *rest)
for part, digest in v1.items():
    print(f"{part:9} {digest[:12]}")
swapped = prompt_version(template, system, examples[::-1], *rest)
spaced = prompt_version(template, system + " ", examples, *rest)
print("примеры переставлены:", [k for k in v1 if v1[k] != swapped[k]])
print("пробел в инструкции: ", [k for k in v1 if v1[k] != spaced[k]])
shuffled = dict(reversed(list(decoding.items())))
print("ключи decoding в другом порядке:",
      sha256_json(shuffled) == v1["decoding"])
# Вывод в книге:
# | template  7a9469194052
# | system    e89d93c52bad
# | examples  c861147412b4
# | grammar   db7b370d2e00
# | decoding  688420ed73e7
# | prefix    df205f6deb11
# | model     8d6fc17ade4c
# | version   add709d36bc3
# | примеры переставлены: ['examples', 'version']
# | пробел в инструкции:  ['system', 'version']
# | ключи decoding в другом порядке: True

# %% Листинг «Проверки ответа кодом» — раздел «Набор оценки и регрессионный тест»
def check_answer(text, labels, passed_chunks, n_tokens, budget=512):
    """Список нарушений; пустой список: ответ прошёл все проверки."""
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return ["форма"]
    if not isinstance(obj, dict):
        return ["форма"]
    errors = []
    if obj.get("label") not in labels:
        errors.append("метка")
    if not set(obj.get("citations", [])) <= set(passed_chunks):
        errors.append("цитата")
    if n_tokens > budget:
        errors.append("бюджет")
    return errors

answers = [('{"label":"neg","citations":["c7"]}', 300),
           ('{"label":"neutral"}', 300),
           ('{"label":"pos","citations":["c9"]}', 300),
           ('Конечно! Вот JSON: {"label":"pos"}', 300),
           ('{"label":"pos"}', 530)]
for text, n_tokens in answers:
    print(f"{text:36}", check_answer(text, {"pos", "neg"},
                                     ["c3", "c7"], n_tokens))
# Вывод в книге:
# | {"label":"neg","citations":["c7"]}   []
# | {"label":"neutral"}                  ['метка']
# | {"label":"pos","citations":["c9"]}   ['цитата']
# | Конечно! Вот JSON: {"label":"pos"}   ['форма']
# | {"label":"pos"}                      ['бюджет']

# %% Листинг «Критерий Макнемара и правило выпуска» — раздел «Парное сравнение двух версий»
from math import comb

def mcnemar_exact(only_a, only_b):
    n = only_a + only_b
    if n == 0:
        return 1.0
    tail = sum(comb(n, j) for j in range(max(only_a, only_b), n + 1))
    return min(1.0, 2 * tail / 2 ** n)

def release_gate(form_rate, only_old, only_new, alpha=0.05):
    """Выпуск: форма 100% и новая версия не стала значимо хуже старой."""
    if form_rate < 1.0:
        return False
    worse = only_old > only_new
    return not (worse and mcnemar_exact(only_old, only_new) < alpha)

for only_a, only_b in [(3, 7), (2, 8), (1, 9), (30, 70)]:
    print(f"{only_b:2} к {only_a:2}: p = {mcnemar_exact(only_a, only_b):.2g}")
cases = [("50 задач, 7 к 3 за новую", 1.0, 3, 7),
         ("500 задач, 70 к 30 за старую", 1.0, 70, 30),
         ("форма 98%", 0.98, 0, 10),
         ("50 задач, 7 к 3 за старую", 1.0, 7, 3)]
for name, form_rate, only_old, only_new in cases:
    print(f"{name:29}", release_gate(form_rate, only_old, only_new))
# Вывод в книге:
# |  7 к  3: p = 0.34
# |  8 к  2: p = 0.11
# |  9 к  1: p = 0.021
# | 70 к 30: p = 7.9e-05
# | 50 задач, 7 к 3 за новую      True
# | 500 задач, 70 к 30 за старую  False
# | форма 98%                     False
# | 50 задач, 7 к 3 за старую     True

# %% Листинг «Две формулировки просьбы на BookLM» — раздел «Парное сравнение двух версий»
# Проверка этот листинг не запускает: ему нужны сеть, видеокарта, сервер или внешний файл.
# import hashlib
# from collections import Counter
# from booklm.lab import generate, load
# from booklm.experiments.ch10 import HEROES, mentions
#
# tok, model = load("instruct")
# heroes = [h.split() for h in HEROES]            # 100 героев, по 6 падежей
# print("набор:", len(heroes), "героев, SHA-256", hashlib.sha256(
#     json.dumps(heroes, ensure_ascii=False).encode()).hexdigest()[:16])
# versions = {"v1": "Напиши сказку про {acc}.",  # «про» + винительный
#             "v2": "Расскажи короткую историю, в которой главный герой"
#                   " — {nom}."}
# ok = {v: [] for v in versions}
# for forms in heroes:
#     for v, request in versions.items():
#         req = request.format(nom=forms[0], acc=forms[3])
#         story = generate(tok, model, req, chat=True, temperature=0,
#                          max_new=120)                  # жадно
#         ok[v].append(mentions(story, forms))    # герой назван?
# pairs = Counter(zip(ok["v1"], ok["v2"]))
# only_old, only_new = pairs[True, False], pairs[False, True]
# print("v1:", sum(ok["v1"]), "v2:", sum(ok["v2"]), "| обе верны",
#       pairs[True, True], "только v1", only_old, "только v2", only_new,
#       "обе неверны", pairs[False, False])
# print(f"p_МН = {mcnemar_exact(only_old, only_new):.2f};",
#       "release_gate:", release_gate(1.0, only_old, only_new))

# %% Листинг «Версия промпта BookLM» — раздел «Версия промпта BookLM»
booklm_v1 = prompt_version(
    template, system="", examples=[],          # инструкции у сказочника нет
    grammar="",                                # сказка: свободный текст
    decoding={"temperature": 0.0, "max_new_tokens": 120},
    prefix={"page_size": 16, "shared_pages": 0},
    model={"weights": sha256_file("models/booklm/booklm-instruct.pt"),
           "adapter": sha256_file("models/booklm/booklm-instruct-lora.pt"),
           "tokenizer": sha256_file("models/booklm/tokenizer.json")})
print("booklm-prompt-v1:", booklm_v1["version"][:16],
      "| модель:", booklm_v1["model"][:16])
# Вывод в книге:
# | booklm-prompt-v1: 97724b5a0661753d | модель: 70f384fae1a757df
