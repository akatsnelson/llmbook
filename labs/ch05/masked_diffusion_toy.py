"""Маскированная диффузия в мире из четырёх фраз: ошибка с весом 1/t и генерация открытием масок.

Полная версия листингов «Ошибка маскированной диффузии» и «Генерация открытием масок» из главы 5 (раздел 5.8
«Диффузионные языковые модели»). Модель здесь идеальная:
вероятности слов под масками считаются точно по распределению четырёх фраз, поэтому всё, что ломается,
ломается из-за самой схемы, а не из-за плохо обученных весов. Сверх книги файл:

* считает ошибку с весом 1/t точно, перебором всех масок (интеграл по t даёт бета-функцию), и
  проверяет, что она в точности равна -log p(x), а без веса — нет;
* сравнивает порядок открытия позиций: по уверенности и случайный.

Запуск из корня репозитория: python labs/ch05/masked_diffusion_toy.py
"""
import itertools
import math
import random

world = {("мама", "мыла", "раму"): 0.4, ("папа", "мыл", "машину"): 0.3,
         ("мама", "читала", "книгу"): 0.2, ("папа", "читал", "газету"): 0.1}
M = "[M]"
n = 3                                    # длина фразы, как в книге


def denoise(x):
    """Для каждой закрытой позиции — распределение слова при открытых словах x."""
    fits = {s: p for s, p in world.items()
            if all(a in (M, b) for a, b in zip(x, s))} or world   # фраза уже сломана: как без подсказки
    total = sum(fits.values())
    pred = {i: {} for i, a in enumerate(x) if a == M}
    for s, p in fits.items():
        for i in pred:
            pred[i][s[i]] = pred[i].get(s[i], 0) + p / total
    return pred


def masked_loss(x, t, weight):
    xt = tuple(M if random.random() < t else a for a in x)
    pred = denoise(xt)
    loss = -sum(math.log(pred[i][x[i]]) for i in pred)
    return loss / t if weight else loss


def exact_loss(x, weight):
    """E_t E_маски: маска из k позиций выпадает с вероятностью t^k (1-t)^(n-k), а
    интеграл по t от t^(k-1) (1-t)^(n-k) равен (k-1)! (n-k)! / n! (без веса: k! (n-k)! / (n+1)!)."""
    total = 0.0
    for hidden in itertools.product((False, True), repeat=n):
        k = sum(hidden)
        if k == 0:
            continue
        xt = tuple(M if h else a for h, a in zip(hidden, x))
        loss = -sum(math.log(p[x[i]]) for i, p in denoise(xt).items())
        if weight:
            w = math.factorial(k - 1) * math.factorial(n - k) / math.factorial(n)
        else:
            w = math.factorial(k) * math.factorial(n - k) / math.factorial(n + 1)
        total += w * loss
    return total


def unmask(x, steps, by_confidence=True):
    x = list(x)
    for s in range(steps):
        pred = denoise(tuple(x))
        if not pred:
            break
        guess = {i: random.choices(list(d), list(d.values()))[0] for i, d in pred.items()}
        if by_confidence:
            order = sorted(guess, key=lambda i: -pred[i][guess[i]])
        else:
            order = random.sample(list(guess), len(guess))
        for i in order[:math.ceil(len(pred) / (steps - s))]:
            x[i] = guess[i]
    return tuple(x)


# 1. С весом 1/t ошибка в точности равна -log p(x): идеальная модель, правило цепочки в любом порядке.
for x, p in world.items():
    assert abs(exact_loss(x, weight=True) + math.log(p)) < 1e-12, x
    assert exact_loss(x, weight=False) < -math.log(p) - 0.2          # без веса заметно меньше

# Числа листинга книги: оценка по 100 000 случайных t и масок.
random.seed(0)
book = {}
for x in (("мама", "мыла", "раму"), ("мама", "читала", "книгу")):
    book[x] = {w: sum(masked_loss(x, random.random(), w) for _ in range(100_000)) / 100_000
               for w in (True, False)}
assert abs(book[("мама", "мыла", "раму")][True] - 0.911) < 5e-4
assert abs(book[("мама", "читала", "книгу")][True] - 1.615) < 5e-4
assert abs(exact_loss(("мама", "мыла", "раму"), False) - 0.653) < 1e-3

# 2. Генерация: чем меньше шагов, тем чаще слова из разных фраз; ровно n шагов — точная выборка.
random.seed(0)
share = {}
for steps in (3, 2, 1):
    runs = [unmask([M] * n, steps) for _ in range(10_000)]
    share[steps] = sum(r in world for r in runs) / len(runs)
assert share[3] == 1.0
assert abs(share[2] - 0.544) < 1e-3 and abs(share[1] - 0.160) < 1e-3
# за один шаг слова независимы: доля настоящих фраз — сумма произведений частных вероятностей
marg = denoise((M, M, M))
one_step = sum(math.prod(marg[i][s[i]] for i in range(n)) for s in world)
assert abs(one_step - 0.16) < 1e-12

random.seed(0)
rand2 = sum(unmask([M] * n, 2, by_confidence=False) in world for _ in range(10_000)) / 10_000
assert rand2 < share[2] - 0.05                                       # случайный порядок хуже
# точно: первый шаг открывает два случайных места, слова на них независимы, третье слово верно всегда
pairs = list(itertools.combinations(range(n), 2))
exact_rand2 = sum(math.prod(marg[i][w] for i, w in zip(ij, words))
                  for ij in pairs for words in {tuple(s[i] for i in ij) for s in world}) / len(pairs)
assert abs(exact_rand2 - 0.4467) < 1e-4                              # (0,52 + 0,52 + 0,30) / 3
assert abs(rand2 - exact_rand2) < 0.01 and abs(rand2 - 0.450) < 1e-3  # 45 % в книге
# задача главы: два шага, после первого открыты «мама» и «мыла» (0,6 · 0,4) или «мама» и «раму»
# (0,6 · 0,6 · 0,4: второе слово не «мыла», иначе при равной уверенности открылось бы оно)
runs = [unmask([M] * n, 2) for _ in range(10_000)]
mama = sum(r == ("мама", "мыла", "раму") for r in runs) / len(runs)
assert abs(mama - (0.6 * 0.4 + 0.6 * 0.6 * 0.4)) < 0.01
print(f"два шага: по уверенности {share[2]:.1%}, в случайном порядке {rand2:.1%}")

# 3. Заполнение по концу фразы: модель видит последнее слово и пишет начало.
random.seed(0)
assert {unmask([M, M, "газету"], 2) for _ in range(1000)} == {("папа", "читал", "газету")}

print("masked_diffusion_toy: все проверки пройдены")
