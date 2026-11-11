"""Листинги главы 7 «Предобучение» книги «Большие языковые модели», файл 02-data.

Код и вывод — ровно как в книге; текст книги сюда не входит. Листинги идут по порядку и
опираются друг на друга: запускайте файл целиком или ячейками сверху вниз (разметка # %%).
Проверка, что вывод совпал с книгой: python scripts/check_listings.py listings/ch07/02-data.py
"""

# %% Листинг «Нормализация, ключ дубля и мера фильтра» — раздел «Нормализация и точные дубли»
import hashlib
import re
import unicodedata

def normalize(text):          # NFC, переводы строк, пробелы
    text = unicodedata.normalize("NFC", text).replace("\r\n", "\n")
    lines = [re.sub(r"[ \t ]+", " ", ln).strip()
             for ln in text.split("\n")]
    return "\n".join(lines).strip()

def words_of(text):           # только для ключей и шинглов
    return re.sub(r"[^\w\s]", " ", normalize(text).lower()).split()

def norm_key(text):           # ключ точного дубля
    canon = " ".join(words_of(text))
    return hashlib.sha256(canon.encode()).hexdigest()

def letter_share(d):          # доля букв среди непробельных символов
    chars = [ch for ch in d if not ch.isspace()]
    return sum(ch.isalpha() for ch in chars) / len(chars)

two = "\u0438\u0306"                   # «и» и отдельная дужка
print("длина до и после NFC:", len(two), len(normalize(two)))
print(words_of("Мама  мыла раму!"))
print(norm_key("Мама мыла раму.")[:16], norm_key("Мама  мыла раму!")[:16])
print(f"доля букв: {letter_share('12 400 13 900 15 100'):.2f}")
# Вывод в книге:
# | длина до и после NFC: 2 1
# | ['мама', 'мыла', 'раму']
# | a5722be629f0f209 a5722be629f0f209
# | доля букв: 0.00

# %% Листинг «MinHash, Жаккар и вероятность LSH» — раздел «Поиск почти-дублей»
def shingles(words, k):
    return {" ".join(words[i:i + k]) for i in range(len(words) - k + 1)}

def jaccard(s, t):
    return len(s & t) / len(s | t)

def minhash_signature(ids, coeffs, p):
    # ids: номера шинглов; coeffs: пары (a, c), по одной на функцию
    return [min((a * x + c) % p for x in ids) for a, c in coeffs]

def lsh_probability(j, b, r):
    return 1 - (1 - j**r) ** b

A = "мама мыла раму а потом она пошла спать".split()
B = "мама мыла раму а потом папа пошёл спать".split()
print("J =", jaccard(shingles(A, 2), shingles(B, 2)))
six = [(1, 1), (3, 2), (5, 3), (7, 4), (2, 5), (4, 6)]
sig_a = minhash_signature(set(range(7)), six, 11)
sig_b = minhash_signature({0, 1, 2, 3, 7, 8, 9}, six, 11)
same = sum(x == y for x, y in zip(sig_a, sig_b))
print(sig_a, sig_b, f"совпало {same} из 6")
for j in (0.5, 0.7, 0.8, 0.9):
    prob = lsh_probability(j, 14, 8)
    print(f"J = {j}: кандидат с вероятностью {prob:.4f}")
# Вывод в книге:
# | J = 0.4
# | [1, 0, 0, 0, 0, 0] [1, 0, 2, 0, 0, 1] совпало 4 из 6
# | J = 0.5: кандидат с вероятностью 0.0533
# | J = 0.7: кандидат с вероятностью 0.5645
# | J = 0.8: кандидат с вероятностью 0.9235
# | J = 0.9: кандидат с вероятностью 0.9996

# %% Листинг «Сигнатура настоящего текста» — раздел «Поиск почти-дублей»
import random

P61 = 2**61 - 1                          # большое простое
def make_coeffs(n, seed=0):              # n пар (a, c) с зерном
    rng = random.Random(seed)
    return [(rng.randrange(1, P61), rng.randrange(P61)) for _ in range(n)]

def shingle_id(s):                       # hash() меняется от запуска
    digest = hashlib.sha256(s.encode()).digest()
    return int.from_bytes(digest[:8], "big") % P61

def text_signature(words, coeffs, k):
    ids = {shingle_id(s) for s in shingles(words, k)}
    return minhash_signature(ids, coeffs, P61)

def estimate(sig1, sig2):                # доля совпавших минимумов
    return sum(x == y for x, y in zip(sig1, sig2)) / len(sig1)

c112 = make_coeffs(112)
est = estimate(text_signature(A, c112, 2), text_signature(B, c112, 2))
print(f"оценка J по 112 функциям: {est:.2f}")
# Вывод в книге:
# | оценка J по 112 функциям: 0.43

# %% Листинг «Конвейер на игрушечном корпусе» — раздел «Очистка игрушечного корпуса»
mama = "Мама{}мыла раму, а потом она пошла спать, и кот уснул на {}{}"
corpus = [
    mama.format(" ", "окне", "."),
    mama.format("  ", "окне", "!"),        # точный дубль
    mama.format(" ", "диване", "."),       # почти-дубль
    "Чтобы сварить гречку, залейте её водой и варите двадцать минут.",
    "Поезд опоздал на сорок минут, и пассажиры выпили весь чай.",
    "Главная | Новости | Контакты | Войти",
    "12 400 13 900 15 100 17 800 21 300",
    "Дождь шёл весь вечер, и дети пускали по лужам кораблики.",
    "Скалярное произведение велико, когда векторы смотрят в одну сторону.",
    "Библиотека открывается в девять утра, а читальный зал в десять.",
]

def count(stage, docs):
    words = sum(len(words_of(d)) for d in docs)
    print(f"{stage:<13}{len(docs):>3} док. {words:>4} слов")

def lsh_pairs(docs, coeffs, k=2, bands=14, rows=8):
    buckets = {}                                 # пары, совпавшие хоть в одной полосе
    for i, d in enumerate(docs):
        sig = text_signature(words_of(d), coeffs, k)
        for band in range(bands):
            key = (band, tuple(sig[band * rows:(band + 1) * rows]))
            buckets.setdefault(key, set()).add(i)
    return {(i, j) for grp in buckets.values()
            for i in grp for j in grp if i < j}

def funnel(corpus, coeffs, k=2, j_min=0.75):
    count("сбор", corpus)
    seen, docs = set(), []
    for d in map(normalize, corpus):             # точные дубли
        if norm_key(d) not in seen:
            seen.add(norm_key(d))
            docs.append(d)
    count("точные дубли", docs)
    pairs = lsh_pairs(docs, coeffs, k)           # почти-дубли: LSH
    print("кандидаты LSH:", pairs)
    sh = lambda d: shingles(words_of(d), k)
    drop = {j for i, j in pairs
            if jaccard(sh(docs[i]), sh(docs[j])) >= j_min}
    docs = [d for i, d in enumerate(docs) if i not in drop]
    count("почти-дубли", docs)
    docs = [d for d in docs                      # фильтры качества
            if len(words_of(d)) >= 5 and letter_share(d) >= 0.6]
    count("фильтры", docs)
    return docs[:-1], docs[-1:]                  # обучение, валидация

train_docs, val_docs = funnel(corpus, c112)
count("в обучение", train_docs)
# Вывод в книге:
# | сбор          10 док.  102 слов
# | точные дубли   9 док.   89 слов
# | кандидаты LSH: {(0, 1)}
# | почти-дубли    8 док.   76 слов
# | фильтры        6 док.   62 слов
# | в обучение     5 док.   52 слов

# %% Листинг «Разнообразие набора и почти-дубли» — раздел «Синтетические и переписанные данные»
def distinct_n(texts, n=2):              # доля разных n-грамм
    grams = [tuple(w[i:i + n]) for w in map(words_of, texts)
             for i in range(len(w) - n + 1)]
    return len(set(grams)) / len(grams)

def near_duplicates(texts, coeffs, k=2, j_min=0.75):
    # номера текстов, у которых есть более ранний почти-двойник
    sets = [shingles(words_of(t), k) for t in texts]
    return {j for i, j in lsh_pairs(texts, coeffs, k)
            if jaccard(sets[i], sets[j]) >= j_min}

asks = ["Напиши сказку про кота.", "Напиши сказку про собаку.",
        "Расскажи историю про кота.", "Напиши сказку про кота!"]
print(f"distinct-2: {distinct_n(asks[:3]):.2f}, "
      f"три копии: {distinct_n([asks[0]] * 3):.2f}")
print("почти-дубли:", near_duplicates(asks, c112))
# Вывод в книге:
# | distinct-2: 0.67, три копии: 0.33
# | почти-дубли: {3}

# %% Листинг «Разнообразие просьб для дообучения BookLM» — раздел «Синтетические и переписанные данные»
from collections import Counter
from booklm.experiments.ch07_synthetic import sft_pairs

pairs = sft_pairs()                      # 9952 пары «просьба -> сказка»
sft_asks = [p["request"] for p in pairs]
stories = [p["story"] for p in pairs[:1000]]
print(f"distinct-2: просьбы {distinct_n(sft_asks):.2f},"
      f" сказки {distinct_n(stories):.2f}")
dups = near_duplicates(sft_asks, c112)
print(f"почти-дублей: {len(dups)} ({len(dups) / len(sft_asks):.1%})")
first = Counter(" ".join(words_of(sft_asks[j])[:4]) for j in dups)
print("чаще всего начинаются с:", first.most_common(1))
echo = [a for a in sft_asks if "из этой истории" in a]
print(f"просьб со словами «из этой истории»: {len(echo)}")
# Вывод в книге:
# | distinct-2: просьбы 0.34, сказки 0.48
# | почти-дублей: 475 (4.8%)
# | чаще всего начинаются с: [('попроси историю в которой', 458)]
# | просьб со словами «из этой истории»: 596

# %% Листинг «Ошибка рода в корпусе и в модели» — раздел «Синтетические и переписанные данные»
import math
import re
import torch
from booklm.lab import load, prompt_ids
from booklm.experiments.ch07_synthetic import train_texts

corpus = " ".join(train_texts())         # 68 269 обучающих сказок
tok, model = load("base")

@torch.no_grad()
def phrase_prob(text):                   # e в степени суммы ln p токенов
    ids = prompt_ids(tok, text)
    logp = model(torch.tensor([ids[:-1]]))[0].log_softmax(-1)
    return math.exp(logp[torch.arange(len(ids) - 1), ids[1:]].sum())

starts = ("Тим увидел", "Однажды Лили увидела в парке", "Они нашли")
for wrong, right in (("большую дерево", "большое дерево"),
                     ("большую мяч", "большой мяч"),
                     ("большую торт", "большой торт")):
    a, b = (len(re.findall(rf"\b{p}\b", corpus)) for p in (wrong, right))
    share = sum(phrase_prob(f"{s} {wrong}") / (phrase_prob(f"{s} {wrong}")
                + phrase_prob(f"{s} {right}")) for s in starts) / 3
    print(f"{wrong:<14} {a:>4} против {b:>3}:"
          f" в корпусе {a / (a + b):.2f}, у BookLM {share:.2f}")
# Вывод в книге:
# | большую дерево 3048 против 343: в корпусе 0.90, у BookLM 0.94
# | большую мяч     369 против 212: в корпусе 0.64, у BookLM 0.46
# | большую торт     31 против 373: в корпусе 0.08, у BookLM 0.09

# %% Листинг «Упаковка, маска документов и маска ошибки» — раздел «Упаковка последовательностей»
import torch

def pack(docs, window, eos):
    # docs: списки номеров; в каждом окне window = T+1 номеров
    stream, owner = [], []
    for n, doc in enumerate(docs, start=1):
        stream += doc + [eos]
        owner += [n] * (len(doc) + 1)    # EOS входит в свой документ
    cut = len(stream) // window * window # хвост ждёт следующих
    tokens = torch.tensor(stream[:cut]).view(-1, window)
    doc_ids = torch.tensor(owner[:cut]).view(-1, window)
    return tokens, doc_ids, stream[cut:]

def document_causal_mask(doc_ids):
    # doc_ids: [T] -> маска [T, T], True = можно смотреть
    t = doc_ids.numel()
    causal = torch.tril(torch.ones(t, t, dtype=torch.bool))
    return causal & (doc_ids[:, None] == doc_ids[None, :])

EOS = 2                                   # номер <eos> у BookLM
toy_docs = [[11, 12, 13, 14, 15], [21, 22, 23], [31, 32, 33, 34, 35, 36]]
windows, owners, rest = pack(toy_docs, window=12, eos=EOS)
x, y = windows[:, :-1], windows[:, 1:]    # сдвиг цели из главы 3
doc_x = owners[0, :-1]                    # чей токен на позиции
mask = document_causal_mask(doc_x)
loss_mask = x != EOS                      # цель после EOS не учим
print("окно: ", windows[0].tolist())
print("хвост:", rest)
print("чей:  ", doc_x.tolist())
print("видит:", mask.sum(dim=1).tolist())
print("учим: ", loss_mask[0].int().tolist())
# Вывод в книге:
# | окно:  [11, 12, 13, 14, 15, 2, 21, 22, 23, 2, 31, 32]
# | хвост: [33, 34, 35, 36, 2]
# | чей:   [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3]
# | видит: [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 1]
# | учим:  [1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1]

# %% Листинг «Тест влияния между документами» — раздел «Упаковка последовательностей»
import math

def masked_attention(q, k, v, allowed):
    # q, k, v: [T, d_h]; allowed: [T, T]; маска ставится ДО softmax
    scores = q @ k.T / math.sqrt(q.size(-1))
    scores = scores.masked_fill(~allowed, float("-inf"))
    return scores.softmax(dim=-1) @ v

torch.manual_seed(0)
emb = torch.randn(40, 8)                  # 40 номеров, d = 8
w_q, w_k, w_v = (torch.randn(8, 8) for _ in range(3))
def layer(tokens, allowed):
    h = emb[tokens]                       # [T, d]
    return masked_attention(h @ w_q, h @ w_k, h @ w_v, allowed)

changed = x[0].clone()
changed[1:5] = torch.tensor([5, 6, 7, 8]) # переписали середину d1
in_d2 = doc_x == 2
causal_only = torch.tril(torch.ones(11, 11, dtype=torch.bool))
for name, allowed in (("маска документов", mask),
                      ("только причинная", causal_only)):
    diff = layer(x[0], allowed) - layer(changed, allowed)
    print(f"{name}: выходы d2 сдвинулись на "
          f"{diff[in_d2].abs().max().item():.3g}")
size = layer(x[0], mask)[in_d2].abs().mean().item()
print(f"средний модуль выхода d2: {size:.3g}")
# Вывод в книге:
# | маска документов: выходы d2 сдвинулись на 0
# | только причинная: выходы d2 сдвинулись на 8.72
# | средний модуль выхода d2: 2.19

# %% Листинг «Читает ли BookLM соседнюю сказку» — раздел «Упаковка последовательностей»
import numpy as np
import torch
import torch.nn.functional as F
from booklm.lab import load
from booklm.experiments.ch07 import val_stories

tok, model = load("base")
stories = val_stories()          # 689 сказок: номера без <bos> и <eos>
BOS, EOS = 1, 2

@torch.no_grad()
def loss_on_b(a, b):             # окно <bos> A <eos> <bos> B <eos>
    ids = [BOS] + (a + [EOS, BOS] if a else []) + b + [EOS]
    logits = model(torch.tensor([ids[:-1]]))[0]
    nll = F.cross_entropy(logits, torch.tensor(ids[1:]), reduction="none")
    return nll[-len(b) - 1:]     # ошибки на токенах B и на его <eos>

A, A2, B = stories[249], stories[275], stories[586]   # A и A' одной длины
after_a, after_a2 = loss_on_b(A, B), loss_on_b(A2, B)
print(f"A и A' по {len(A)} токенов, B: {len(B)}")
print("начало B:", tok.decode(B[:9]))
print(f"ошибка на B после A:    {after_a.mean():.4f}")
print(f"ошибка на B после A':   {after_a2.mean():.4f}")
print(f"ошибка на B без соседа: {loss_on_b([], B).mean():.4f}")
shift = (after_a - after_a2).abs().max()
print(f"наибольший сдвиг на одном токене B: {shift:.3f}")
# Вывод в книге:
# | A и A' по 100 токенов, B: 143
# | начало B: Однажды маленький овца играла в поле. Овца
# | ошибка на B после A:    1.8112
# | ошибка на B после A':   1.8332
# | ошибка на B без соседа: 1.8206
# | наибольший сдвиг на одном токене B: 0.655

# %% Листинг «Сдвиг ошибки по позициям B» — раздел «Упаковка последовательностей»
by_len = {}                      # A и A': сказки одной длины
for i, s in enumerate(stories):
    by_len.setdefault(len(s), []).append(i)
rng, triples = np.random.default_rng(0), []
for n, same in sorted(by_len.items()):
    if len(same) >= 2 and 100 <= n <= 200:
        b = int(rng.integers(len(stories)))      # B: случайная сказка
        if b not in same[:2] and n + len(stories[b]) + 4 <= 512:
            triples.append((same[0], same[1], b))

shifts, near, alone = [], [], []
for a, a2, b in triples[:20]:
    l1, l2 = loss_on_b(stories[a], stories[b]), loss_on_b(stories[a2], stories[b])
    shifts.append((l1 - l2).abs())
    near.append((l1.mean() + l2.mean()).item() / 2)
    alone.append(loss_on_b([], stories[b]).mean().item())
T = min(len(x) for x in shifts)              # самая короткая B
mean_shift = torch.stack([x[:T] for x in shifts]).mean(0)
print("средний |сдвиг ошибки| по 20 тройкам:")
for lo, hi in ((0, 1), (1, 5), (5, 20), (20, 50), (50, T)):
    print(f"  токены B {lo + 1:3d}-{hi:3d}: {mean_shift[lo:hi].mean():.4f}")
print(f"средняя ошибка на B: с соседом {np.mean(near):.4f},"
      f" без соседа {np.mean(alone):.4f}")
# Вывод в книге:
# | средний |сдвиг ошибки| по 20 тройкам:
# |   токены B   1-  1: 0.0294
# |   токены B   2-  5: 0.0506
# |   токены B   6- 20: 0.0819
# |   токены B  21- 50: 0.0745
# |   токены B  51-118: 0.0643
# | средняя ошибка на B: с соседом 1.7830, без соседа 1.7847

# %% Листинг «Проверка валидации на загрязнение» — раздел «Загрузчик и валидационная выборка»
def contaminated(doc, train_docs, coeffs, k=2, j_min=0.5):
    sig = text_signature(words_of(doc), coeffs, k)
    return any(estimate(sig, text_signature(words_of(t), coeffs, k))
               >= j_min for t in train_docs)

leak = "Чтобы сварить гречку, залейте её кипятком и варите двадцать минут."
for name, doc in (("гречка с кипятком", leak),
                  ("валидация", val_docs[0])):
    seen = contaminated(doc, train_docs, c112)
    print(f"{name}: {'уже видели' if seen else 'новый текст'}")
# Вывод в книге:
# | гречка с кипятком: уже видели
# | валидация: новый текст
