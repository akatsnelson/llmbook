"""Глава 7, раздел 7.2 «Подготовка обучающих данных»: MinHash, полосы LSH и игрушечный корпус
через весь конвейер.

Из книги «Большие языковые модели».
В книге оставлены ручной пример MinHash и кирпичики конвейера (нормализация, ключ
точного дубля, мера фильтра); здесь лежат сигнатуры из 112 функций и полный конвейер
на десяти документах: точные дубли, LSH, фильтры, валидация и проверка загрязнения.
Запуск из корня репозитория: python labs/ch07/minhash_toy_corpus.py
"""

# --- MinHash, Жаккар и вероятность LSH ---
import hashlib
import random

def shingles(words, k):
    return {" ".join(words[i:i + k]) for i in range(len(words) - k + 1)}

def jaccard(s, t):
    return len(s & t) / len(s | t)

def minhash_signature(ids, coeffs, p):
    # ids: номера шинглов; coeffs: [(a, b)] — одна пара на функцию
    return [min((a * x + b) % p for x in ids) for a, b in coeffs]

A = "мама мыла раму а потом она пошла спать".split()
B = "мама мыла раму а потом папа пошёл спать".split()
assert jaccard(shingles(A, 2), shingles(B, 2)) == 0.4   # 4 из 10

six = [(1, 1), (3, 2), (5, 3), (7, 4), (2, 5), (4, 6)]
sig_a = minhash_signature(set(range(7)), six, 11)
sig_b = minhash_signature({0, 1, 2, 3, 7, 8, 9}, six, 11)
assert sig_a == [1, 0, 0, 0, 0, 0] and sig_b == [1, 0, 2, 0, 0, 1]

P61 = 2**61 - 1                                 # большое простое
def make_coeffs(n, seed=0):
    rng = random.Random(seed)
    return [(rng.randrange(1, P61), rng.randrange(P61))
            for _ in range(n)]

def shingle_id(s):        # hash() меняется от запуска к запуску
    digest = hashlib.sha256(s.encode()).digest()
    return int.from_bytes(digest[:8], "big") % P61

def text_signature(words, coeffs, k):
    ids = {shingle_id(s) for s in shingles(words, k)}
    return minhash_signature(ids, coeffs, P61)

def estimate(sig1, sig2):
    return sum(x == y for x, y in zip(sig1, sig2)) / len(sig1)

c112 = make_coeffs(112)
sig_a, sig_b = text_signature(A, c112, 2), text_signature(B, c112, 2)
assert abs(estimate(sig_a, sig_b) - 0.4) < 0.1

def lsh_probability(j, b, r):
    return 1 - (1 - j**r) ** b

assert round(lsh_probability(0.8, 14, 8), 4) == 0.9235
assert round(lsh_probability(0.5, 14, 8), 3) == 0.053


# --- Игрушечный корпус через конвейер ---
import re
import unicodedata

def normalize(text):          # версия из booklm-tokenizer-v1
    text = unicodedata.normalize("NFC", text).replace("\r\n", "\n")
    lines = [re.sub(r"[ \t\u00a0]+", " ", ln).strip()
             for ln in text.split("\n")]
    return "\n".join(lines).strip()

def words_of(text):           # только для ключей и шинглов
    return re.sub(r"[^\w\s]", " ", normalize(text).lower()).split()

def norm_key(text):
    canon = " ".join(words_of(text))
    return hashlib.sha256(canon.encode()).hexdigest()

assert norm_key("Мама мыла раму.") == norm_key("Мама  мыла раму!")
assert normalize("и\u0306") == "й"        # NFC: два символа -> один
assert normalize("Мама мыла.") == "Мама мыла."   # регистр на месте

mama = "Мама{}мыла раму, а потом она пошла спать. Утром рама блестела, \
и солнце било в чистое {} так ярко, что кот проснулся раньше всех{}"
corpus = [
    mama.format(" ", "стекло", "."),
    mama.format("  ", "стекло", "!"),          # точный дубль
    mama.format(" ", "окно", "."),             # почти-дубль
    "Чтобы сварить гречку, промойте крупу, залейте её водой в "
    "пропорции один к двум и варите под крышкой на медленном огне "
    "двадцать минут.",
    "Поезд опоздал на сорок минут, и пассажиры успели выпить весь чай "
    "в буфете, пересказать друг другу новости и познакомиться "
    "с проводником.",
    "Главная | Новости | Контакты | Войти",
    "12 400 13 900 15 100 17 800 21 300 24 600 26 000 29 700",
    "Скалярное произведение двух векторов велико, когда они смотрят "
    "в одну сторону, и близко к нулю, когда они перпендикулярны.",
    "Дождь шёл весь вечер; во дворе собрались лужи, и дети пускали "
    "по ним бумажные кораблики, пока их не позвали ужинать.",
    "Библиотека открывается в девять утра, но читальный зал начинает "
    "работать только в десять, после того как разложат свежие газеты.",
]
funnel = [("сбор", corpus)]
seen, docs = set(), []
for d in map(normalize, corpus):          # точные дубли
    if norm_key(d) not in seen:
        seen.add(norm_key(d))
        docs.append(d)
funnel.append(("точные дубли", docs))

coeffs, bands, rows, k = make_coeffs(112), 14, 8, 2
buckets = {}
for i, d in enumerate(docs):              # почти-дубли: LSH
    sig = text_signature(words_of(d), coeffs, k)
    for band in range(bands):
        key = (band, tuple(sig[band * rows:(band + 1) * rows]))
        buckets.setdefault(key, set()).add(i)
candidates = {(i, j) for g in buckets.values()
              for i in g for j in g if i < j}
assert candidates == {(0, 1)}             # другие пары не столкнулись
sh = lambda d: shingles(words_of(d), k)
drop = {j for i, j in candidates
        if jaccard(sh(docs[i]), sh(docs[j])) >= 0.75}
docs = [d for i, d in enumerate(docs) if i not in drop]
funnel.append(("почти-дубли", docs))

def letter_share(d):
    chars = [ch for ch in d if not ch.isspace()]
    return sum(ch.isalpha() for ch in chars) / len(chars)

filters = {"короче 5 слов": lambda d: len(words_of(d)) >= 5,
           "букв меньше 60 %": lambda d: letter_share(d) >= 0.6}
dropped = {}
for name, keep in filters.items():
    dropped[name] = [d[:16] for d in docs if not keep(d)]
    docs = [d for d in docs if keep(d)]
funnel.append(("фильтры", docs))
assert dropped == {"короче 5 слов": ["Главная | Новост"],
                   "букв меньше 60 %": ["12 400 13 900 15"]}

val_docs, train_docs = docs[-1:], docs[:-1]   # до упаковки
funnel.append(("в обучение", train_docs))
counts = [(len(ds), sum(len(words_of(d)) for d in ds))
          for _, ds in funnel]
assert counts == [(10, 192), (9, 168), (8, 144), (6, 124), (5, 105)]

def contaminated(doc, pool, threshold=0.75):   # MinHash между выборками
    sig = text_signature(words_of(doc), coeffs, k)
    return any(estimate(sig, text_signature(words_of(t), coeffs, k))
               >= threshold for t in pool)
leaked = corpus[3].replace("медленном", "тихом")
assert not contaminated(val_docs[0], train_docs)
assert contaminated(leaked, train_docs)   # уже есть в обучении


print("minhash_toy_corpus: все проверки пройдены")
