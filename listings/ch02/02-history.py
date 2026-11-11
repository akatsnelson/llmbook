"""Листинги главы 2 «История искусственного интеллекта и языковых моделей» книги «Большие языковые модели», файл 02-history.

Код и вывод — ровно как в книге; текст книги сюда не входит. Листинги идут по порядку и
опираются друг на друга: запускайте файл целиком или ячейками сверху вниз (разметка # %%).
Проверка, что вывод совпал с книгой: python scripts/check_listings.py listings/ch02/02-history.py
"""

# %% Листинг «Перцептрон: взвешенная сумма, порог и исправление после ошибки» — раздел «Перцептрон»
def predict(w, t, x):
    s = w[0] * x[0] + w[1] * x[1]       # взвешенная сумма
    return 1 if s > t else 0            # сравнение с порогом

def train(data, w, t, epochs=50):
    for _ in range(epochs):
        for x, y in data:
            err = y - predict(w, t, x)  # 0, +1 или -1
            w = [w[0] + err * x[0], w[1] + err * x[1]]
            t = t - err
    return w, t

AND = [((0, 0), 0), ((0, 1), 0), ((1, 0), 0), ((1, 1), 1)]
w, t = train(AND, w=[5, -3], t=2)
print("веса:", w, "порог:", t)
print("ответы:", [predict(w, t, x) for x, _ in AND])
# Вывод в книге:
# | веса: [3, 2] порог: 4
# | ответы: [0, 0, 0, 1]

# %% Листинг «Одна программа BookLM, два набора чисел» — раздел «Перцептрон»
import torch
from booklm.lab import load_model, load_tokenizer
from booklm.experiments.ch02 import continue_ids, show

tok = load_tokenizer()
before = load_model("init")    # класс BookLM с числами шага 0
after = load_model("base")     # тот же класс с числами после обучения
print("параметров:", before.num_params(), "и", after.num_params())
a = torch.cat([p.flatten() for p in before.parameters()])
b = torch.cat([p.flatten() for p in after.parameters()])
changed = int((a != b.bfloat16().float()).sum())  # шаг 0 записан короче
share = 100 * changed / len(a)
print(f"изменились {changed} из {len(a)} чисел ({share:.2f} %)")
print(f"средний модуль числа до обучения {a.abs().mean():.4f}, "
      f"изменения {(a - b).abs().mean():.4f}")
for name, model in (("До обучения", before), ("После обучения", after)):
    print(name + ":")
    new = continue_ids(tok, model, "Однажды", new=40)    # жадно
    show("Однажды" + tok.decode(new), width=70)
# Вывод в книге:
# | параметров: 6622464 и 6622464
# | изменились 6617863 из 6622464 чисел (99.93 %)
# | средний модуль числа до обучения 0.0165, изменения 0.0441
# | До обучения:
# |   Однаждыилоилоилоилоилоилоилоилоилоилоилоилоилоилоилоилоилоилоилоилоило
# |   илоилоилоилоилоилоилоилоилоилоилоилоилоилоилоилоилоилоило
# | После обучения:
# |   Однажды маленькая девочка по имени Лили пошла в парк. Она увидела
# |   большую дерево и захотела взобраться на него. Когда она поднималась,
# |   она увидела маленькую птичку, сидевшую на ветке. Птичка выглядела
# |   грустной.
# |   Лили спросила

# %% Листинг «Тот же перцептрон на таблице XOR» — раздел «Ограничения однослойного перцептрона»
XOR = [((0, 0), 0), ((0, 1), 1), ((1, 0), 1), ((1, 1), 0)]
for epochs in (10, 100, 1000):
    w, t = train(XOR, w=[5, -3], t=2, epochs=epochs)
    answers = [predict(w, t, x) for x, _ in XOR]
    right = sum(a == y for a, (_, y) in zip(answers, XOR))
    print(f"{epochs:>4} кругов:", w, t, answers, "верных", right)
# Вывод в книге:
# |   10 кругов: [-1, -1] -1 [1, 0, 0, 0] верных 1
# |  100 кругов: [-1, -1] -1 [1, 0, 0, 0] верных 1
# | 1000 кругов: [-1, -1] -1 [1, 0, 0, 0] верных 1

# %% Листинг «Триграммная модель: вероятность как доля в счётчике» — раздел «n-граммы»
from collections import Counter, defaultdict

corpus = [["я", "люблю", "кошек"], ["я", "люблю", "кошек"],
          ["я", "люблю", "пиццу"], ["я", "люблю", "зиму"]]

counts = defaultdict(Counter)            # (a, b) -> Counter
for words in corpus:
    for a, b, c in zip(words, words[1:], words[2:]):
        counts[(a, b)][c] += 1

def prob(a, b, c):
    seen = counts.get((a, b), Counter())
    total = sum(seen.values())
    return seen[c] / total if total else 0.0

print(counts[("я", "люблю")])
print(prob("я", "люблю", "кошек"))
print(prob("я", "обожаю", "кошек"))
# Вывод в книге:
# | Counter({'кошек': 2, 'пиццу': 1, 'зиму': 1})
# | 0.5
# | 0.0

# %% Листинг «Метод Шеннона на сказках BookLM и сама BookLM» — раздел «n-граммы»
import random
from collections import Counter, defaultdict
from booklm.lab import MODELS, load_model, load_tokenizer
from booklm.experiments.ch02 import WORD, stories, join_words, show
from booklm.experiments.ch02 import continue_ids

tok = load_tokenizer()
tales = stories(tok, MODELS / "data" / "train.bin")
print("обучающих сказок:", len(tales))
pairs = defaultdict(Counter)          # слово -> какие слова шли за ним
for tale in tales:
    seq = WORD.findall(tale)          # слова и знаки препинания
    for a, b in zip(seq, seq[1:]):
        pairs[a][b] += 1
rng = random.Random(0)
words = ["Однажды"]
while len(words) < 50:                # метод Шеннона
    seen = pairs[words[-1]]
    words.append(rng.choices(list(seen), weights=list(seen.values()))[0])
print("Счётчики пар:")
show(join_words(words), width=70)
model = load_model("base")            # BookLM тоже выбирает случайно
new = continue_ids(tok, model, "Однажды", new=200, temperature=1.0, seed=0)
print("BookLM:")
show(join_words(WORD.findall("Однажды" + tok.decode(new))[:50]), width=70)
# Вывод в книге:
# | обучающих сказок: 68269
# | Счётчики пар:
# |   Однажды кролик любил бегать и смотреть на кучу мусора. Она злится. Они
# |   научились делиться. Нам его большой кусочек и брала. Лили пошла в
# |   небе. Фин поплыли вокруг тебя любим наш волшебный кот начал говорить с
# |   восхищением, я летать. Они добрались до двери.
# | BookLM:
# |   Однажды мальчик по имени Тим с мамой пошёл на пляж. Он очень хотел
# |   играть в песке. Вдруг он услышал странные звуки. Тим испугался и
# |   широко раскрыл рот. Запущение, где он умел быть маленьким и наблюдать
# |   вокруг кана. Тим спросил маму: «Где ты

# %% Листинг «Пары слов из новых фраз, которых не было при обучении» — раздел «n-граммы»
import re

val = stories(tok, MODELS / "data" / "val.bin")  # отложенные сказки
ends = r"(?<=[.!?])\s+"                        # конец предложения
phrases = [s for tale in val for s in re.split(ends, tale)][:200]
fresh = []                                     # пары соседних слов
for p in phrases:
    seq = WORD.findall(p)
    fresh += zip(seq, seq[1:])
unseen = [(a, b) for a, b in fresh if pairs.get(a, Counter())[b] == 0]
share = 100 * len(unseen) / len(fresh)
print(f"пар слов в 200 новых фразах: {len(fresh)}")
print(f"не встречались при обучении: {len(unseen)} ({share:.1f} %)")
print("например:", ", ".join(f"«{a} {b}»" for a, b in unseen[:4]))
# Вывод в книге:
# | пар слов в 200 новых фразах: 1644
# | не встречались при обучении: 82 (5.0 %)
# | например: «дедушка жили», «жили очень», «по бабушке», «отдала телефон»

# %% Листинг «Двухслойная сеть из трёх перцептронов решает XOR» — раздел «Возвращение нейронных сетей»
def two_layer(x):
    h1 = predict([1, 1], 0, x)          # ИЛИ: хотя бы один вход
    h2 = predict([-1, -1], -2, x)       # И-НЕ: не оба сразу
    return predict([1, 1], 1, (h1, h2)), (h1, h2)   # И двух ответов

for x, y in XOR:
    out, h = two_layer(x)
    print(x, "-> скрытый слой", h, "-> ответ", out, "| нужно", y)
# Вывод в книге:
# | (0, 0) -> скрытый слой (0, 1) -> ответ 0 | нужно 0
# | (0, 1) -> скрытый слой (1, 1) -> ответ 1 | нужно 1
# | (1, 0) -> скрытый слой (1, 1) -> ответ 1 | нужно 1
# | (1, 1) -> скрытый слой (1, 0) -> ответ 0 | нужно 0

# %% Листинг «Базовая BookLM и BookLM после дообучения на одной просьбе» — раздел «Дообучение на инструкциях и оценках людей»
import re
from booklm.lab import load_model, load_tokenizer
from booklm.experiments.ch02 import continue_ids, show

tok = load_tokenizer()
ask = "Расскажи сказку про кота, который боялся темноты."
heroes = {"кота": r"\bкот(?!ор)", "ёжика": r"\b[ёе]ж(ик|ек|и|а|у|ом|ей)"}

def sample(model, text, seed):         # случайный выбор, 240 токенов
    ids = continue_ids(tok, model, text, new=240, temperature=1.0,
                       seed=seed, chat=True)
    return tok.decode(ids)

for kind in ("base", "instruct"):
    model = load_model(kind)
    print(f"{kind}, жадно:")
    greedy = continue_ids(tok, model, ask, new=40, chat=True)
    show(tok.decode(greedy), width=70)
    for hero, pattern in heroes.items():
        request = ask.replace("кота", hero)
        answers = [sample(model, request, s) for s in range(20)]
        new_tale = sum(a.lstrip()[:1].isupper() for a in answers)
        named = sum(bool(re.search(pattern, a, re.I)) for a in answers)
        print(f"  про {hero}: новых сказок {new_tale} из 20, "
              f"с героем {named}")
# Вывод в книге:
# | base, жадно:
# |   как раз, он был очень смелым и смелым. Он хотел найти способ, чтобы
# |   его жизнь была в безопасности.
# |   Расктив спросил у мамы: «Что такое волшебная сила?»
# |   про кота: новых сказок 0 из 20, с героем 10
# |   про ёжика: новых сказок 0 из 20, с героем 2
# | instruct, жадно:
# |   Жил-был маленький кот. Кот был очень любопытным. Он любил играть и
# |   бегать. Однажды кот увидел большую, страшную тень. Кот хотел играть с
# |   тенью.
# |   Кот пошёл к своему другу
# |   про кота: новых сказок 20 из 20, с героем 14
# |   про ёжика: новых сказок 20 из 20, с героем 0

# %% Листинг «Сколько обучающих сказок называют каждого героя» — раздел «Дообучение на инструкциях и оценках людей»
for hero, pattern in heroes.items():
    n = sum(bool(re.search(pattern, t, re.I)) for t in tales)
    print(f"сказок про {hero}: {n} из {len(tales)}")
# Вывод в книге:
# | сказок про кота: 6423 из 68269
# | сказок про ёжика: 43 из 68269

# %% Листинг «Базовая BookLM продолжает начало сказки» — раздел «Дообучение на инструкциях и оценках людей»
base = load_model("base")
for start in ("Жил-был кот, который боялся темноты.",
              "Жил-был ёжик, который боялся темноты."):
    new = continue_ids(tok, base, start, new=40)   # без пометок, жадно
    show(start + tok.decode(new), width=70)
# Вывод в книге:
# |   Жил-был кот, который боялся темноты. Он был очень любопытным и хотел
# |   знать, что происходит. Однажды он услышал, как кто-то говорит. Он
# |   посмотрел вокруг и увидел маленькую мышку. Мышка была испугана и не
# |   знала, как убежать.
# |   Жил-был ёжик, который боялся темноты. Однажды он услышал странный
# |   звук. Он посмотрел вокруг и увидел маленькую девочку, которая играла в
# |   парке. Она была очень тихой и не слышала, как кто-нибудь говорил.
# |   Девочка подошла к ней
