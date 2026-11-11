"""Листинги главы 3 «Токенизация и эмбеддинги» книги «Большие языковые модели», файл 03-pre-transformer.

Код и вывод — ровно как в книге; текст книги сюда не входит. Листинги идут по порядку и
опираются друг на друга: запускайте файл целиком или ячейками сверху вниз (разметка # %%).
Проверка, что вывод совпал с книгой: python scripts/check_listings.py listings/ch03/03-pre-transformer.py
"""

# %% Листинг «Символы и байты UTF-8» — раздел «Символы, байты, токены и номера»
print("м".encode("utf-8"))                 # байты буквы м
print(len("мама"), len("мама".encode("utf-8")))
print([len(ch.encode("utf-8")) for ch in ["m", "м", "\U0001F642"]])
# Вывод в книге:
# | b'\xd0\xbc'
# | 4 8
# | [1, 2, 4]

# %% Листинг «Один цикл слияний BPE» — раздел «Ручной расчёт трёх слияний»
from collections import Counter

def merge_pair(pieces, pair):
    result, i = [], 0
    while i < len(pieces):
        if i + 1 < len(pieces) and (pieces[i], pieces[i + 1]) == pair:
            result.append(pieces[i] + pieces[i + 1])  # склеили
            i += 2                                    # и перешагнули пару
        else:
            result.append(pieces[i])
            i += 1
    return tuple(result)

def learn_merges(word_frequencies, steps):
    corpus = Counter({tuple(w): c for w, c in word_frequencies.items()})
    merges = []
    for _ in range(steps):
        counts = Counter()
        for pieces, frequency in corpus.items():
            for pair in zip(pieces, pieces[1:]):
                counts[pair] += frequency
        if not counts:
            break
        best = min(counts, key=lambda pair: (-counts[pair], pair))
        merges.append(best)
        updated = Counter()
        for pieces, frequency in corpus.items():
            updated[merge_pair(pieces, best)] += frequency
        corpus = updated
    return merges, corpus

merges, corpus = learn_merges({"мама": 2, "мамы": 1}, steps=3)
print(merges)
print(dict(corpus))
print(merge_pair(("а", "а", "а"), ("а", "а")))
# Вывод в книге:
# | [('м', 'а'), ('ма', 'ма'), ('м', 'ы')]
# | {('мама',): 2, ('ма', 'мы'): 1}
# | ('аа', 'а')

# %% Листинг «Слияния токенизатора BookLM» — раздел «Ручной расчёт трёх слияний»
import json
from booklm.lab import MODELS
from booklm.experiments.ch03 import readable   # символы файла -> текст

path = MODELS / "tokenizer.json"
merges_lm = json.load(open(path, encoding="utf-8"))["model"]["merges"]

def show(k):                                    # k-е слияние, считая с нуля
    a, b = merges_lm[k]
    return f"{k + 1}) {readable(a)} + {readable(b)} -> {readable(a + b)}"

print("слияний:", len(merges_lm))
print("в файле:", merges_lm[273], "это", show(273))
print(" | ".join(show(k) for k in range(4)))
made = {readable(a + b): k for k, (a, b) in enumerate(merges_lm)}
print(" | ".join(show(made[w]) for w in ["_Однажды", "_девочка", "_кот"]))
print(" | ".join(show(k) for k in range(len(merges_lm) - 3, len(merges_lm))))
print("самый длинный:", max(made, key=len))
# Вывод в книге:
# | слияний: 7931
# | в файле: ['ĠÐ¼Ð°Ð¼', 'Ð°'] это 274) _мам + а -> _мама
# | 1) _ + [D0] -> _[D0] | 2) [D0] + [BE] -> о | 3) [D0] + [B0] -> а | 4) [D0] + [BB] -> л
# | 354) _О + днажды -> _Однажды | 394) _дево + чка -> _девочка | 469) _к + от -> _кот
# | 7929) _ин + греди -> _ингреди | 7930) _игрушка + м -> _игрушкам | 7931) _лап + ками -> _лапками
# | самый длинный: _поприветствовать

# %% Листинг «Кодирование и декодирование тремя слияниями» — раздел «Кодирование и декодирование»
def build_vocab(alphabet, merges):
    pieces = ["<pad>", "<unk>"] + list(alphabet)
    pieces += [a + b for a, b in merges]       # склейки по порядку
    return {piece: number for number, piece in enumerate(pieces)}

def encode(text, merges, vocab):
    pieces = tuple(text)
    for pair in merges:                    # строго в выученном порядке
        pieces = merge_pair(pieces, pair)
    return [vocab.get(p, vocab["<unk>"]) for p in pieces]

def decode(ids, vocab):
    piece_of = {number: piece for piece, number in vocab.items()}
    return "".join(piece_of[number] for number in ids)

vocab = build_vocab("маы", merges)
print(list(vocab))                  # номер токена — место в списке
ids = encode("мамы", merges, vocab)
print(ids, decode(ids, vocab))
print(encode("мам", merges, vocab))
# Вывод в книге:
# | ['<pad>', '<unk>', 'м', 'а', 'ы', 'ма', 'мама', 'мы']
# | [5, 7] мамы
# | [5, 2]

# %% Листинг «Проверка туда-обратно ловит потери» — раздел «Кодирование и декодирование»
def roundtrip_failures(texts, merges, vocab):
    return [s for s in texts
            if decode(encode(s, merges, vocab), vocab) != s]

ids = encode("мамаша", merges, vocab)
print(ids, decode(ids, vocab))

texts = ["", "мама", "Мама", "ма ма", "\U0001F642"]   # последняя — эмодзи
for s in roundtrip_failures(texts, merges, vocab):
    print(decode(encode(s, merges, vocab), vocab))
# Вывод в книге:
# | [6, 1, 3] мама<unk>а
# | <unk>ама
# | ма<unk>ма
# | <unk>

# %% Листинг «Байтовый запасной путь вместо <unk>» — раздел «Байтовый запасной путь»
def build_vocab_bytes(alphabet, merges):
    vocab = build_vocab(alphabet, merges)
    for b in range(256):                     # ещё 256 токенов-байтов
        vocab[f"<0x{b:02X}>"] = len(vocab)
    return vocab

def encode_bytes(text, merges, vocab):
    pieces = tuple(text)
    for pair in merges:
        pieces = merge_pair(pieces, pair)
    ids = []
    for p in pieces:
        if p in vocab:
            ids.append(vocab[p])
        else:                                # запасной путь: байты UTF-8
            ids += [vocab[f"<0x{b:02X}>"] for b in p.encode("utf-8")]
    return ids

def decode_bytes(ids, vocab):
    piece_of = {number: piece for piece, number in vocab.items()}
    data = b""
    for number in ids:
        piece = piece_of[number]
        if piece.startswith("<0x"):          # токен-байт: берём сам байт
            data += bytes([int(piece[3:5], 16)])
        else:
            data += piece.encode("utf-8")
    return data.decode("utf-8")

vocab_b = build_vocab_bytes("маы", merges)
ids = encode_bytes("мамаша", merges, vocab_b)
print(len(vocab_b), ids, decode_bytes(ids, vocab_b))
print([s for s in texts
       if decode_bytes(encode_bytes(s, merges, vocab_b), vocab_b) != s])
print(len(encode_bytes("\U0001F642", merges, vocab_b)))
# Вывод в книге:
# | 264 [6, 217, 144, 3] мамаша
# | []
# | 4

# %% Листинг «Предварительное разбиение строки» — раздел «Байтовый запасной путь»
import re

def pre_split(text):
    # слово вместе с пробелом перед ним, знаки препинания, лишние пробелы
    return re.findall(r" ?\w+| ?[^\w\s]+|\s+", text)

print(pre_split("мама мыла раму, а потом"))
# Вывод в книге:
# | ['мама', ' мыла', ' раму', ',', ' а', ' потом']

# %% Листинг «Одна буква, две записи» — раздел «Байтовый запасной путь»
import unicodedata

one = "й"
two = unicodedata.normalize("NFD", one)   # буква и плюс отдельная кратка
print(len(one), len(two), one == two)
print(one.encode("utf-8"), two.encode("utf-8"))
print(encode_bytes(one, merges, vocab_b),
      encode_bytes(two, merges, vocab_b))
print(unicodedata.normalize("NFC", two) == one)
# Вывод в книге:
# | 1 2 False
# | b'\xd0\xb9' b'\xd0\xb8\xcc\x86'
# | [216, 193] [216, 192, 212, 142]
# | True

# %% Листинг «Токенизатор из наших функций» — раздел «Байтовый запасной путь»
def train_tokenizer(text, steps):
    words = Counter(pre_split(text))           # куски с пробелом впереди
    merges, _ = learn_merges(words, steps)
    vocab = build_vocab_bytes(sorted(set(text)), merges)
    return merges, vocab

def encode_text(text, merges, vocab):
    return [i for piece in pre_split(text)
            for i in encode_bytes(piece, merges, vocab)]

tale = ("Мама мыла раму. Потом мама мыла пол, а кот спал. "
        "Мама пошла спать, и кот пошёл спать.")
merges_t, vocab_t = train_tokenizer(tale, steps=12)
print(["".join(m) for m in merges_t])
ids = encode_text("Мама мыла кошку!", merges_t, vocab_t)
print(len(vocab_t), ids)
print([decode_bytes([i], vocab_t) for i in ids])   # куски по одному
print(decode_bytes(ids, vocab_t))
# Вывод в книге:
# | ['ам', ' м', ' п', ' с', ' по', ' сп', ' спа', 'ама', 'ла', 'от', ' к', ' кот']
# | 290 [5, 29, 23, 19, 30, 32, 12, 18, 9, 17, 67]
# | ['М', 'ама', ' м', 'ы', 'ла', ' к', 'о', 'ш', 'к', 'у', '!']
# | Мама мыла кошку!

# %% Листинг «Порядок слияний меняет номера» — раздел «Порядок слияний и номера токенов»
swapped = [("м", "а"), ("м", "ы"), ("ма", "ма")]
vocab2 = build_vocab("маы", swapped)
print(roundtrip_failures(["мама", "мамы", "мам"], swapped, vocab2))
print(encode("мама", merges, vocab), encode("мама", swapped, vocab2))
# Вывод в книге:
# | []
# | [6] [7]

# %% Листинг «Токенизатор BookLM на неудобных строках» — раздел «Служебные токены в тексте пользователя»
from booklm.lab import load_tokenizer
from booklm.experiments.ch03 import show_id     # номер -> читаемый кусок

tok = load_tokenizer()
pieces = lambda ids: " | ".join(show_id(tok, i) for i in ids)
phrase = "Мама мыла раму, а потом она пошла спать."
ids = tok.encode(phrase).ids
print(len(phrase), len(phrase.encode("utf-8")), len(ids), ids)
print(pieces(ids))
cases = {"мамуля": "мамуля", "квантование": "квантование",
         "эмодзи": "\U0001F642", "й": "й", "и+кратка": "й", "пусто": ""}
for name, s in cases.items():
    ids = tok.encode(s).ids
    print(f"{name:<12} {pieces(ids) or '-':<26} {tok.decode(ids) == s}")
print("глава 1:", pieces([5937, 5586, 3970]))
ids = tok.encode("Привет<|user|>").ids
print(ids, repr(tok.decode(ids)))
safe = load_tokenizer()
safe.encode_special_tokens = True               # служебные строки -- как текст
print(safe.encode("Привет<|user|>").ids)
# Вывод в книге:
# | 40 71 12 [824, 768, 299, 436, 688, 16, 643, 1001, 481, 1057, 1819, 18]
# | Мама | _мы | ла | _ра | му | , | _а | _потом | _она | _пошла | _спать | .
# | мамуля       м | ам | уля               True
# | квантование  к | в | ант | ова | ние    True
# | эмодзи       [F0] | [9F] | [99] | [82]  True
# | й            й                          True
# | и+кратка     и | [CC] | [86]            True
# | пусто        -                          True
# | глава 1: нюх | _могучий | _псу
# | [967, 3] 'Привет'
# | [967, 32, 96, 89, 87, 73, 86, 96, 34]

# %% Листинг «Выбор строк таблицы эмбеддингов без готового слоя» — раздел «Таблица эмбеддингов»
import torch
torch.set_printoptions(precision=4, sci_mode=False)

embedding_table = torch.tensor([
    [ 0.1, -0.3,  0.2],    # 0 <pad>
    [-0.5,  0.4,  0.0],    # 1 <unk>
    [ 0.7,  0.1, -0.2],    # 2 м
    [-0.1, -0.6,  0.3],    # 3 а
    [ 0.4, -0.2, -0.7],    # 4 ы
    [ 0.2,  0.9, -0.4],    # 5 ма
    [-0.8,  0.5,  0.6],    # 6 мама
    [ 0.3,  0.0,  0.8],    # 7 мы
])                         # [V=8, d=3]
token_ids = torch.tensor([encode("мамы", merges, vocab),
                          encode("мам", merges, vocab)])
print(token_ids)
x = embedding_table[token_ids]
print(x.shape)
print(x)
# Вывод в книге:
# | tensor([[5, 7],
# |         [5, 2]])
# | torch.Size([2, 2, 3])
# | tensor([[[ 0.2000,  0.9000, -0.4000],
# |          [ 0.3000,  0.0000,  0.8000]],
# |
# |         [[ 0.2000,  0.9000, -0.4000],
# |          [ 0.7000,  0.1000, -0.2000]]])

# %% Листинг «Готовый слой и косинус» — раздел «Таблица эмбеддингов»
import torch.nn.functional as F

layer = torch.nn.Embedding(8, 3)                # готовый слой PyTorch
layer.weight.data.copy_(embedding_table)        # с теми же числами
print(torch.equal(layer(token_ids), x))
one_hot = F.one_hot(token_ids, num_classes=8).float()   # [2, 2, 8]
print(one_hot[0, 0])
print(torch.equal(one_hot @ embedding_table, x))

cat, kitty, concrete = torch.tensor([[2.6, 1.5], [2.3, 2.0], [-0.7, 2.4]])
print(F.cosine_similarity(cat, kitty, dim=0),
      F.cosine_similarity(cat, concrete, dim=0))
# Вывод в книге:
# | True
# | tensor([0., 0., 0., 0., 0., 1., 0., 0.])
# | True
# | tensor(0.9815) tensor(0.2372)

# %% Листинг «Соседи слов в таблице эмбеддингов BookLM» — раздел «Таблица эмбеддингов»
from booklm.lab import load_model

def neighbours(table, word, k=4):
    (i,) = tok.encode(word).ids                 # слово -- один токен
    cos = F.cosine_similarity(table[i], table, dim=-1)   # [8192]
    cos[i] = -1                                 # сам токен не считаем
    top = cos.topk(k)
    return ", ".join(f"{show_id(tok, j)} {c:.2f}" for c, j
                     in zip(top.values.tolist(), top.indices.tolist()))

for kind in ["base", "init"]:                   # обученная и шаг 0
    table = load_model(kind).embed.weight.detach()
    print(kind, tuple(table.shape))
    for word in [" мама", " кот", " побежал", " красный"]:
        print(f"  {word.strip():<8}", neighbours(table, word))
    pairs = F.cosine_similarity(table[:4096], table[4096:], dim=-1)
    print(f"  средний косинус 4096 случайных пар: {pairs.mean():.2f}")
# Вывод в книге:
# | base (8192, 256)
# |   мама     _Мама 0.62, _папа 0.53, Мама 0.49, _мамы 0.44
# |   кот      _кота 0.54, _пёс 0.49, _коту 0.48, _Кот 0.48
# |   побежал  _побежала 0.71, _побежали 0.60, _бежит 0.57, _бежал 0.50
# |   красный  _синий 0.56, _красная 0.53, _красную 0.52, _жёлтый 0.52
# |   средний косинус 4096 случайных пар: 0.15
# | init (8192, 256)
# |   мама     егодня 0.23, _рождения 0.22, _бас 0.21, уна 0.21
# |   кот      _воздуш 0.26, _звали 0.25, _смея 0.22, _М 0.21
# |   побежал  _улыбкой 0.23, аны 0.22, _Скорее 0.22, об 0.21
# |   красный  аются 0.25, _камней 0.21, _позабот 0.20, _новые 0.20
# |   средний косинус 4096 случайных пар: 0.00

# %% Листинг «Сдвиг на одну позицию даёт входы и цели» — раздел «Предсказание следующего токена»
def next_token_examples(token_ids):
    # token_ids: [B, T+1] -> входы и цели, обе формы [B, T]
    return token_ids[:, :-1], token_ids[:, 1:]

stream = torch.tensor([[824, 768, 299, 436, 688]])
inputs, targets = next_token_examples(stream)
print(inputs)
print(targets)
# Вывод в книге:
# | tensor([[824, 768, 299, 436]])
# | tensor([[768, 299, 436, 688]])

# %% Листинг «Softmax для одного прогноза» — раздел «Softmax и перекрёстная энтропия»
logits = torch.tensor([2.0, 1.0, 0.0])   # токены _раму, _пол, _кота
exp = logits.exp()                       # все числа стали положительными
total = exp.sum()
probs = exp / total
print(exp, total, probs, probs.sum(), sep="\n")
print(-probs[0].log())
# Вывод в книге:
# | tensor([7.3891, 2.7183, 1.0000])
# | tensor(11.1073)
# | tensor([0.6652, 0.2447, 0.0900])
# | tensor(1.0000)
# | tensor(0.4076)

# %% Листинг «Устойчивая перекрёстная энтропия» — раздел «Softmax и перекрёстная энтропия»
def cross_entropy_one(logits, target):
    # logits: [V], target: номер правильного токена
    shifted = logits - logits.max()          # максимум стал нулём
    log_sum_exp = shifted.exp().sum().log()
    return -shifted[target] + log_sum_exp

print(f"{torch.tensor(88.0).exp().item():.2e}", torch.tensor(89.0).exp())
small = cross_entropy_one(torch.tensor([2.0, 1.0, 0.0]), target=0)
big = cross_entropy_one(torch.tensor([1002.0, 1001.0, 1000.0]), target=0)
print(small, big)
# Вывод в книге:
# | 1.65e+38 tensor(inf)
# | tensor(0.4076) tensor(0.4076)

# %% Листинг «Ошибка следующего токена у BookLM» — раздел «Перплексия»
from booklm.lab import prompt_ids, perplexity, next_token_table

ids = torch.tensor([prompt_ids(tok, phrase)])      # <bos> и фраза: [1, 13]
targets = ids[0, 1:]                               # что надо угадать
loss = {}
for kind in ["base", "init"]:                      # обученная и шаг 0
    model = load_model(kind)
    with torch.no_grad():
        logits = model(ids)[0, :-1]                # [12, 8192]
    loss[kind] = F.cross_entropy(logits, targets, reduction="none")
    m = loss[kind].mean()
    print(f"{kind}: ошибка {m:.2f}, перплексия по фразе {m.exp():.1f},",
          f"на историях {perplexity(model):.1f}")
print("токен    p      -ln p  шаг 0")
for y, a, b in zip(targets.tolist(), loss["base"], loss["init"]):
    print(f"{show_id(tok, y):<8} {(-a).exp():.3f} {a:6.2f} {b:6.2f}")
base = load_model("base")
print(next_token_table(tok, base, "Мама мыла раму, а потом она пошла", k=3))
# Вывод в книге:
# | base: ошибка 4.13, перплексия по фразе 62.5, на историях 7.0
# | init: ошибка 9.17, перплексия по фразе 9593.9, на историях 8468.5
# | токен    p      -ln p  шаг 0
# | Мама     0.004   5.64   8.97
# | _мы      0.000   7.62   9.25
# | ла       0.321   1.14   9.22
# | _ра      0.000   8.87   9.03
# | му       0.012   4.41   9.49
# | ,        0.027   3.61   9.23
# | _а       0.089   2.42   9.32
# | _потом   0.133   2.02   9.20
# | _она     0.010   4.62   9.45
# | _пошла   0.046   3.07   9.15
# | _спать   0.002   6.07   8.96
# | .        0.874   0.13   8.75
# | [(' в', 0.4427), (' к', 0.2161), (' на', 0.1667)]

# %% Листинг «Ступеньки слияний» — раздел «Строительные токены»
def scaffold_counts(alone, merges):
    built = Counter(alone)                # оставшиеся куски тоже собраны
    for a, b in reversed(merges):         # от поздних слияний к ранним
        built[a] += built[a + b]          # каждая сборка a+b начиналась
        built[b] += built[a + b]          # со сборки a и сборки b
    return built

steps, final = learn_merges({"полетели": 10}, steps=7)
print(["".join(m) for m in steps], dict(final))

corpus = {"полетели": 10, "полет": 1}
steps, final = learn_merges(corpus, steps=7)
alone = Counter()
for parts, n in final.items():            # готовая токенизация корпуса
    for p in parts:
        alone[p] += n
built = scaffold_counts(alone, steps)
for a, b in steps:
    t = a + b
    print(f"{t:<9} собран {built[t]:2}, сам {alone[t]:2}, IoS {1 - alone[t] / built[t]:.2f}")
# Вывод в книге:
# | ['ел', 'ет', 'ели', 'етели', 'летели', 'олетели', 'полетели'] {('полетели',): 10}
# | ет        собран 11, сам  0, IoS 1.00
# | лет       собран 11, сам  0, IoS 1.00
# | олет      собран 11, сам  0, IoS 1.00
# | полет     собран 11, сам  1, IoS 0.91
# | ел        собран 10, сам  0, IoS 1.00
# | ели       собран 10, сам  0, IoS 1.00
# | полетели  собран 10, сам 10, IoS 0.00

# %% Листинг «Строительные токены BookLM» — раздел «Строительные токены»
import numpy as np

vocab_lm = json.load(open(path, encoding="utf-8"))["model"]["vocab"]   # кусок -> номер
piece_of = {i: p for p, i in vocab_lm.items()}
seen = np.bincount(np.fromfile(MODELS / "data" / "train.bin", dtype=np.uint16),
                   minlength=8192)                # сколько раз номер стоит в потоке
alone = Counter({piece_of[i]: int(seen[i]) for i in range(5, 8192)})
built = scaffold_counts(alone, [tuple(m) for m in merges_lm])
total = sum(alone.values())

def kind(p):
    text = tok.decode([vocab_lm[p]]).lstrip(" ")
    if vocab_lm[p] < 261: return "байт"
    if "\ufffd" in text: return "неполная буква"   # знак замены U+FFFD
    return "буква" if len(text) <= 1 else "кусок слова"

for thr in [0.9, 0.99]:
    top = [p for p in built if built[p] and 1 - alone[p] / built[p] > thr]
    share = sum(alone[p] for p in top) / total
    print(f"IoS > {thr}: {len(top)} токенов, сами по себе {share:.2%} потока")
print(dict(Counter(kind(p) for p in top)))
words = [p for p in top if kind(p) == "кусок слова"]
print(f"куски слов: {sum(alone[p] for p in words) / total:.2%} потока")
print(show(made["_маленький"]), "|", show(made["_полетели"]))
for w in ["_мал", "_малень", "_маленький", "_пол", "_полет", "_полетели"]:
    p = "".join(merges_lm[made[w]])
    print(f"{w:<11} собран {built[p]:6}, сам {alone[p]:6}, IoS {1 - alone[p] / built[p]:.4f}")
# Вывод в книге:
# | IoS > 0.9: 1274 токенов, сами по себе 6.61% потока
# | IoS > 0.99: 338 токенов, сами по себе 1.25% потока
# | {'байт': 61, 'неполная буква': 6, 'буква': 32, 'кусок слова': 239}
# | куски слов: 0.15% потока
# | 275) _малень + кий -> _маленький | 7872) _полет + ели -> _полетели
# | _мал        собран  93235, сам    180, IoS 0.9981
# | _малень     собран  65577, сам     29, IoS 0.9996
# | _маленький  собран  30793, сам  30793, IoS 0.0000
# | _пол        собран  27847, сам   2662, IoS 0.9044
# | _полет      собран    409, сам    231, IoS 0.4352
# | _полетели   собран    178, сам    178, IoS 0.0000

# %% Листинг «Одна фраза, три токенизации» — раздел «Неканоническая токенизация»
def text_logprob(model, ids):            # ln p всей последовательности номеров
    x = torch.tensor([[1] + ids])        # <bos> и номера текста
    with torch.no_grad():
        logits = model(x)[0, :-1]
    return -F.cross_entropy(logits, x[0, 1:], reduction="sum").item()

story = "Жил-был маленький котёнок."
canon = tok.encode(story).ids
k = canon.index(tok.encode(" маленький").ids[0])
split = canon[:k] + tok.encode(" малень").ids + tok.encode("кий").ids + canon[k + 1:]
letters = [i for ch in story for i in tok.encode(ch).ids]   # каждый символ отдельно
for name, ids in [("каноническая", canon), ("малень|кий", split), ("по буквам", letters)]:
    print(f"{name:<13} {len(ids):2} токенов, текст тот же: {tok.decode(ids) == story},",
          f"ln p = {text_logprob(base, ids):7.2f}")
print(pieces(split))
# Вывод в книге:
# | каноническая   6 токенов, текст тот же: True, ln p =  -10.20
# | малень|кий     7 токенов, текст тот же: True, ln p =  -24.46
# | по буквам     26 токенов, текст тот же: True, ln p = -227.05
# | Жил | - | был | _малень | кий | _котёнок | .

# %% Листинг «Цена текста в токенах BookLM» — раздел «Модели без токенизатора»
import numpy as np

texts = {"русский": "Мама мыла раму, а потом она пошла спать.",
         "английский": "Mom washed the window, and then she went to sleep.",
         "опечатка": "Мама мыла раму, а патом она пошла спать.",
         "латинская c": "Мама мыла раму, а потом она пошла cпать.",  # c из латиницы
         "заглавные": "МАМА МЫЛА РАМУ, А ПОТОМ ОНА ПОШЛА СПАТЬ."}
print(f"{'':<12} символов байтов токенов")
for name, s in texts.items():
    print(f"{name:<12} {len(s):8} {len(s.encode('utf-8')):6} {len(tok.encode(s).ids):7}")
print(pieces(tok.encode(" патом").ids), "/", pieces(tok.encode(" cпать").ids))
print([tok.encode(w).ids for w in [" мама", " мамы", " Мама"]])

stream = np.fromfile(MODELS / "data" / "train.bin", dtype=np.uint16)
seen = np.bincount(stream, minlength=8192)      # сколько раз встретился номер
never = [i for i in range(261, 8192) if seen[i] == 0]   # 0-4 служебные, 5-260 байты
print(len(never), pieces(never[:6]))
typo = tok.encode("днажды").ids                 # «Однажды» без первой буквы
print(typo, pieces(typo), typo[0] in never)
# Вывод в книге:
# |              символов байтов токенов
# | русский            40     71      12
# | английский         50     50      39
# | опечатка           40     71      13
# | латинская c        40     70      15
# | заглавные          40     71      35
# | _па | том / _ | c | п | ать
# | [[534], [1367], [930]]
# | 65 _дру | днажды | _счаст | _друз | _игруш | _дере
# | [412] днажды True

# %% Листинг «Байтовая n-граммная модель» — раздел «Участки по энтропии»
import math
from collections import Counter, defaultdict

def stories_bytes(ids):          # поток номеров -> байты, истории через \n
    text = tok.decode(ids.tolist(), skip_special_tokens=False)
    return text.replace("<bos>", "").replace("<eos>", "\n").encode("utf-8")

K = 6                                        # байтов контекста
train = stories_bytes(stream[:200_000])
counts = defaultdict(Counter)                # контекст -> счётчик следующего байта
for n in range(K + 1):
    grams = Counter(train[i:i + n + 1] for i in range(len(train) - n))
    for gram, c in grams.items():
        counts[gram[:-1]][gram[-1]] = c
totals = {ctx: sum(c.values()) for ctx, c in counts.items()}

def byte_entropy(data, i):                   # энтропия байта data[i], в битах
    for n in range(min(K, i), -1, -1):       # откат к более короткому контексту
        ctx = data[i - n:i]
        if totals.get(ctx, 0) >= 5:
            break
    t = totals[ctx]
    return sum(c / t * math.log2(t / c) for c in counts[ctx].values())

print(len(train), "байтов,", len(counts), "контекстов")
data = "Мама мыла раму".encode("utf-8")
print(" ".join(f"{data[i]:02X}:{byte_entropy(data, i):.1f}" for i in range(8, 20)))
# Вывод в книге:
# | 1510706 байтов, 74416 контекстов
# | 20:1.3 D0:0.9 BC:3.5 D1:0.7 8B:1.4 D0:0.7 BB:0.0 D0:0.8 B0:1.3 20:0.2 D1:0.9 80:2.3

# %% Листинг «Участки по глобальному порогу» — раздел «Участки по энтропии»
from booklm.lab import val_tokens

def patch_starts(data, theta):   # BLT: участок начинается, где энтропия выше порога
    return [0] + [i for i in range(1, len(data)) if byte_entropy(data, i) > theta]

def show_patches(text, theta):   # черта перед буквой, в байтах которой начат участок
    starts, out, pos = set(patch_starts(text.encode("utf-8"), theta)), "", 0
    for ch in text:
        n = len(ch.encode("utf-8"))
        if pos > 0 and starts & set(range(pos, pos + n)):
            out += " | "
        out, pos = out + ("_" if ch == " " else ch), pos + n
    return out

print(show_patches(phrase, 2.0))
print(pieces(tok.encode(phrase).ids))
val_ids = val_tokens(10**6)                          # отложенные истории
val = stories_bytes(val_ids)
text_bytes = len(tok.decode(val_ids.tolist()).encode("utf-8"))
print(f"{len(val)} байтов, {text_bytes / int((val_ids > 4).sum()):.2f} байта на токен BookLM")
words = [(i + 1, i + 2) for i in range(len(val) - 2)     # два байта после пробела:
         if val[i] in b" \n" and val[i + 1] not in b" \n"]   # первая буква слова
first = {j for w in words for j in w}
for theta in [1.5, 2.0, 2.5]:
    starts = patch_starts(val, theta)
    s = set(starts)
    print(f"порог {theta}: участок {len(val) / len(starts):5.2f} байта,",
          f"в первой букве слова {sum(j in first for j in starts) / len(starts):.0%},",
          f"слов с началом участка {sum(a in s or b in s for a, b in words) / len(words):.0%}")
# Вывод в книге:
# | Мама_ | мыла_ | ра | му | ,_ | а_ | по | том_ | она_ | по | шла_ | с | пать.
# | Мама | _мы | ла | _ра | му | , | _а | _потом | _она | _пошла | _спать | .
# | 934308 байтов, 7.72 байта на токен BookLM
# | порог 1.5: участок  4.22 байта, в первой букве слова 36%, слов с началом участка 94%
# | порог 2.0: участок  7.94 байта, в первой букве слова 63%, слов с началом участка 88%
# | порог 2.5: участок 11.63 байта, в первой букве слова 78%, слов с началом участка 74%

# %% Листинг «Средняя ошибка без заполнителя» — раздел «Пакет, заполнитель и маски»
import torch.nn.functional as F

PAD, EOS = 0, 2
stream = torch.tensor([[8, 4, 9, 5],
                       [3, EOS, PAD, PAD]])         # [B=2, T+1=4]
inputs, targets = next_token_examples(stream)       # оба [2, 3]
print(targets)

torch.manual_seed(0)
logits = torch.randn(2, 3, 10)      # случайные числа вместо логитов модели
per_token = F.cross_entropy(logits.reshape(-1, 10), targets.reshape(-1),
                            reduction="none").reshape(2, 3)
mask = (targets != PAD).float()                     # True/False -> 1./0.
loss = (per_token * mask).sum() / mask.sum()
same = F.cross_entropy(logits.reshape(-1, 10), targets.reshape(-1),
                       ignore_index=PAD)
print(per_token)
print(mask)
print(loss, same, per_token.mean())
print(cross_entropy_one(logits[0, 0], 4))
# Вывод в книге:
# | tensor([[4, 9, 5],
# |         [2, 0, 0]])
# | tensor([[1.3389, 1.9542, 3.7094],
# |         [3.1470, 2.9171, 1.5999]])
# | tensor([[1., 1., 1.],
# |         [1., 0., 0.]])
# | tensor(2.5374) tensor(2.5374) tensor(2.4444)
# | tensor(1.3389)

# %% Листинг «Пакет BookLM из потока историй» — раздел «Пакет, заполнитель и маски»
from booklm.lab import val_tokens, special

data = val_tokens(10**6)                 # отложенные истории подряд
EOS = special(tok)["<eos>"]              # номер 2
gen = torch.Generator().manual_seed(0)

def get_batch(data, B, T):
    starts = torch.randint(len(data) - T - 1, (B,), generator=gen)
    windows = torch.stack([data[i:i + T + 1] for i in starts])   # [B, T+1]
    return next_token_examples(windows)

inputs, targets = get_batch(data, B=4, T=64)
print(inputs.shape, targets.shape)
print(inputs[0, :10].tolist())
print(targets[0, :10].tolist())
k = inputs[0].tolist().index(EOS)        # где кончилась история
piece = inputs[0, k - 6:k + 8].tolist()
print(k, tok.decode(piece, skip_special_tokens=False))

starts = torch.randint(len(data) - 513, (1000,), generator=gen)
cross = sum(bool((data[i:i + 513] == EOS).any()) for i in starts)
print(cross, "окон длины 512 из 1000 пересекают границу;",
      "средняя история", len(data) // int((data == EOS).sum()), "токенов")
# Вывод в книге:
# | torch.Size([4, 64]) torch.Size([4, 64])
# | [2425, 881, 625, 393, 1336, 278, 4161, 18, 301, 908]
# | [881, 625, 393, 1336, 278, 4161, 18, 301, 908, 16]
# | 22  он может сделать большие вещи.<eos><bos>Жил-был маленький мальчик.
# | 1000 окон длины 512 из 1000 пересекают границу; средняя история 177 токенов
