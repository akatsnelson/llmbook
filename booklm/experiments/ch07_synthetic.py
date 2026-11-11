"""Глава 7, раздел «Подготовка обучающих данных», рубрика «А теперь на нашей модели: синтетические
данные»: просьбы для дообучения BookLM и ошибка рода в корпусе.

Опыты (время — на процессоре Apple M2 Max):
  ch07_synthetic_requests — разнообразие 9752 просьб, которые сочинила Qwen3-8B (distinct-2), почти-дубли
      по MinHash и LSH и эхо подсказки «из этой истории». Около 5 с.
  ch07_gender_agreement — сколько раз в обучающих историях прилагательное женского рода стоит перед
      существительным мужского или среднего рода и как часто такую ошибку выбирает BookLM.
      Около 10 с.
Каждый опыт печатает вывод и сохраняет копию в models/booklm/experiments/<имя опыта>.txt
(ch07-synthetic-requests.txt, ch07-gender-agreement.txt).
Листинги книги импортируют отсюда только загрузчики данных: sft_pairs и train_texts.

Запуск из корня репозитория:
    python -m booklm.experiments.ch07_synthetic                            # все опыты
    python -m booklm.experiments.ch07_synthetic ch07_synthetic_requests    # один опыт
    python -m booklm.experiments.ch07_synthetic --help                     # список опытов
"""
import hashlib
import json
import math
import random
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import torch

if not __package__:  # запуск файлом (python booklm/experiments/ch07_synthetic.py): корень репозитория — в путь поиска
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from booklm.experiments import run
from booklm.lab import MODELS, load, load_tokenizer, prompt_ids

OUT = MODELS / "experiments"


# ----------------------------------------------------------------------------- данные

def sft_pairs():
    """Пары «просьба -> история» из data/sft.jsonl (их сочинила Qwen3-8B, booklm/make_sft.py)."""
    with open(MODELS / "data" / "sft.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def train_texts():
    """Обучающие истории BookLM текстом: train.bin — поток <bos> история <eos> ..."""
    data = np.fromfile(MODELS / "data" / "train.bin", dtype=np.uint16)
    ends = np.flatnonzero(data == 2)
    starts = np.r_[0, ends[:-1] + 1]
    seqs = [data[s + 1:e].tolist() for s, e in zip(starts, ends)]   # без <bos> и <eos>
    return load_tokenizer().decode_batch(seqs)


# ----------------------------------------------------------------------------- те же функции, что в листингах 7.2

def normalize(text):
    text = unicodedata.normalize("NFC", text).replace("\r\n", "\n")
    lines = [re.sub(r"[ \t ]+", " ", ln).strip() for ln in text.split("\n")]
    return "\n".join(lines).strip()


def words_of(text):
    return re.sub(r"[^\w\s]", " ", normalize(text).lower()).split()


def shingles(words, k):
    return {" ".join(words[i:i + k]) for i in range(len(words) - k + 1)}


def jaccard(s, t):
    return len(s & t) / len(s | t)


P61 = 2**61 - 1


def make_coeffs(n, seed=0):
    rng = random.Random(seed)
    return [(rng.randrange(1, P61), rng.randrange(P61)) for _ in range(n)]


def shingle_id(s):
    return int.from_bytes(hashlib.sha256(s.encode()).digest()[:8], "big") % P61


def text_signature(words, coeffs, k):
    ids = {shingle_id(s) for s in shingles(words, k)}
    return [min((a * x + c) % P61 for x in ids) for a, c in coeffs]


def distinct_n(texts, n=2):
    grams = [tuple(w[i:i + n]) for t in texts for w in [words_of(t)] for i in range(len(w) - n + 1)]
    return len(set(grams)) / len(grams), len(grams)


def near_duplicates(texts, coeffs, k=2, bands=14, rows=8, j_min=0.75):
    sets = [shingles(words_of(t), k) for t in texts]
    buckets = {}
    for i, t in enumerate(texts):
        sig = text_signature(words_of(t), coeffs, k)
        for band in range(bands):
            buckets.setdefault((band, tuple(sig[band * rows:(band + 1) * rows])), []).append(i)
    pairs = {(i, j) for grp in buckets.values() for i in grp for j in grp if i < j}
    return {j for i, j in pairs if jaccard(sets[i], sets[j]) >= j_min}, len(pairs)


# ----------------------------------------------------------------------------- опыты

def ch07_synthetic_requests():
    """Разнообразие и почти-дубли просьб для дообучения BookLM."""
    pairs = sft_pairs()
    asks = [p["request"] for p in pairs]
    stories = [p["story"] for p in pairs[:1000]]
    d_ask, g_ask = distinct_n(asks)
    d_story, g_story = distinct_n(stories)
    print(f"просьб: {len(asks)}, средняя длина {np.mean([len(words_of(a)) for a in asks]):.1f} слова")
    print(f"distinct-2 просьб: {d_ask:.2f} ({g_ask} биграмм)")
    print(f"distinct-2 первых 1000 историй: {d_story:.2f} ({g_story} биграмм)")
    dups, n_pairs = near_duplicates(asks, make_coeffs(112))
    print(f"кандидатов LSH: {n_pairs}, почти-дублей: {len(dups)} ({len(dups) / len(asks):.1%})")
    echo = [a for a in asks if "из этой истории" in a]
    print(f"просьб с «из этой истории»: {len(echo)}, из них почти-дублей: "
          f"{sum(asks[j] in echo for j in dups)}")
    first = Counter(" ".join(words_of(asks[j])[:4]) for j in dups)
    print("чаще всего почти-дубли начинаются с:", first.most_common(1))
    print("самые частые начала просьб:")
    for start, n in Counter(" ".join(a.split()[:2]) for a in asks).most_common(6):
        print(f"  {start:<22}{n}")
    print("пример эха подсказки:", " ".join(echo[0].split()))


@torch.no_grad()
def ch07_gender_agreement():
    """Ошибка рода прилагательного в обучающих историях и у BookLM."""
    texts = train_texts()
    corpus = " ".join(texts)
    print(f"обучающих историй: {len(texts)}")
    masc = "мяч дом торт шар цветок подарок замок корабль парк сад лес камень стул стол телефон самолёт поезд"
    neut = "дерево яблоко печенье окно озеро солнце мороженое платье одеяло перо гнездо"
    nouns = (masc + " " + neut).split()
    fem = re.findall(r"\b(\w+(?:ую|юю))\s+(" + "|".join(nouns) + r")\b", corpus)
    print(f"прилагательное на -ую/-юю перед существительным м. или ср. рода ({len(nouns)} слов): {len(fem)}")
    for pair, n in Counter(" ".join(x) for x in fem).most_common(6):
        print(f"  {pair:<18}{n}")
    tok, model = load("base")

    def logp(text):
        ids = prompt_ids(tok, text)
        lg = model(torch.tensor([ids[:-1]]))[0].log_softmax(-1)
        return lg[torch.arange(len(ids) - 1), torch.tensor(ids[1:])].sum().item()

    contexts = ("Тим увидел", "Однажды Лили увидела в парке", "Они нашли")
    print("доля ошибочной формы: в корпусе / у BookLM (среднее по трём началам)")
    for wrong, right in (("большую дерево", "большое дерево"), ("большую мяч", "большой мяч"),
                         ("большую торт", "большой торт")):
        a, b = (len(re.findall(rf"\b{p}\b", corpus)) for p in (wrong, right))
        model_share = np.mean([1 / (1 + math.exp(logp(f"{c} {right}") - logp(f"{c} {wrong}")))
                               for c in contexts])
        print(f"  {wrong:<15}{a:>5} против {b:>4}: {a / (a + b):.2f} / {model_share:.2f}")


EXPERIMENTS = {"ch07_synthetic_requests": ch07_synthetic_requests,
               "ch07_gender_agreement": ch07_gender_agreement}

if __name__ == "__main__":
    run(EXPERIMENTS, save=True)      # копия вывода в models/booklm/experiments/<имя опыта>.txt
