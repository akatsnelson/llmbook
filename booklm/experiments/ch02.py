"""Глава 2 на обученной BookLM: числа модели, n-граммы против нейросети, дообучение на просьбах.

Опыты (время — на процессоре Apple M2 Max):
  ch02_rule_in_numbers — раздел «Перцептрон», рубрика «А теперь на нашей модели: параметры». Печатает,
      сколько чисел изменило обучение, и продолжения «Однажды» до обучения и после. Около 2 с.
  ch02_ngram_vs_booklm — раздел «Вероятность следующего слова», рубрика «А теперь на нашей модели:
      n-граммы». Текст из счётчиков пар слов по методу Шеннона и текст BookLM, затем доля пар слов из
      новых фраз, которых не было в обучающих сказках. Около 15 с.
  ch02_embedding_neighbours — раздел «Векторы слов» (листинга с этим выводом в книге нет). Пять
      ближайших токенов по косинусу для пяти слов до обучения и после. Около 2 с.
  ch02_base_vs_instruct — раздел «Дообучение на инструкциях и оценках людей», рубрика «А теперь на
      нашей модели: обучение по инструкциям». Базовая и дообученная BookLM отвечают на одну просьбу;
      по 20 ответов считаем, сколько из них про ёжика и темноту. Около 40 с.
Все опыты печатают только на экран, txt не пишут.

Запуск из корня репозитория:
    python -m booklm.experiments.ch02                        # все опыты
    python -m booklm.experiments.ch02 ch02_ngram_vs_booklm   # один опыт
    python -m booklm.experiments.ch02 --help                 # список опытов
"""
import random
import re
import sys
import textwrap
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

if not __package__:  # запуск файлом (python booklm/experiments/ch02.py): корень репозитория — в путь поиска
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from booklm.experiments import run
from booklm.lab import MODELS, load_model, load_tokenizer, prompt_ids, special

OUT = MODELS / "experiments"
TRAIN = MODELS / "data" / "train.bin"      # обучающий поток номеров
WORD = re.compile(r"\w+|[^\w\s]")       # слово или отдельный знак препинания


def show(text, width=95, limit=None):
    """Печатает текст, перенося длинные строки; limit — сколько строк оставить."""
    lines = []
    for line in text.splitlines():
        lines += textwrap.wrap(line.strip(), width) if line.strip() else []
    for line in lines[:limit]:
        print("  " + line)


@torch.no_grad()
def continue_ids(tok, model, text, new, temperature=0.0, seed=0, chat=False):
    """Номера новых токенов после text: жадно (temperature=0) или выборкой без top-k."""
    ids = prompt_ids(tok, text, chat)
    out = model.generate(torch.tensor([ids]), max_new=new, temperature=temperature,
                         top_k=None, eos_id=special(tok)["<eos>"], seed=seed)
    return out[0, len(ids):].tolist()


def stories(tok, path):
    """Раскодировать поток номеров в список историй (граница — токен <eos>)."""
    data = np.fromfile(path, dtype=np.uint16)
    if (data == special(tok)["<pad>"]).any():      # prepare.py заполнитель не пишет
        raise RuntimeError(f"{path}: в потоке есть <pad>, файл испорчен; пересоберите его booklm/prepare.py")
    ends = np.flatnonzero(data == special(tok)["<eos>"])
    starts = np.concatenate([[0], ends[:-1] + 1])
    return tok.decode_batch([data[s:e].tolist() for s, e in zip(starts, ends)])


def join_words(words):
    """Склеить слова обратно в текст: перед знаками препинания пробел не нужен."""
    text = " ".join(words)
    return re.sub(r" ([.,!?:;»)])", r"\1", text).replace("« ", "«")


# ----------------------------------------------------------------------------- опыты

def ch02_rule_in_numbers():
    """Сколько чисел изменило обучение и продолжения «Однажды» до обучения и после."""
    tok = load_tokenizer()
    before, after = load_model("init"), load_model("base")    # одна программа, два набора чисел
    print("параметров до обучения:   ", before.num_params())
    print("параметров после обучения:", after.num_params())
    a = torch.cat([p.flatten() for p in before.parameters()])
    b = torch.cat([p.flatten() for p in after.parameters()])
    # Шаг 0 сохранён в bfloat16, поэтому и числа после обучения сравниваем с той же точностью.
    changed = int((a != b.bfloat16().float()).sum())
    print(f"изменилось чисел (с точностью bfloat16): {changed} из {len(a)} ({100 * changed / len(a):.2f} %)")
    print(f"средний модуль изменения: {(a - b).abs().mean():.4f}, средний модуль числа до обучения: {a.abs().mean():.4f}")
    for name, model in (("До обучения", before), ("После обучения", after)):
        new = continue_ids(tok, model, "Однажды", new=80)       # жадно: всегда самый вероятный токен
        print(f"\n{name}:")
        show("Однажды" + tok.decode(new), limit=6)


def ch02_ngram_vs_booklm():
    """Метод Шеннона на парах слов против BookLM и доля невиданных пар слов в новых фразах."""
    tok = load_tokenizer()
    train = stories(tok, TRAIN)
    # Счётчики пар соседних слов, как в листинге с триграммами, но по одному предыдущему слову.
    counts = defaultdict(Counter)
    for story in train:
        words = WORD.findall(story)
        for a, b in zip(words, words[1:]):
            counts[a][b] += 1

    # Метод Шеннона: следующее слово берём с той частотой, с какой оно шло за предыдущим.
    rng = random.Random(0)
    words = ["Однажды"]
    while len(words) < 80:
        seen = counts[words[-1]]
        words.append(rng.choices(list(seen), weights=list(seen.values()))[0])
    print("Счётчики пар слов, 80 слов:")
    show(join_words(words))

    # BookLM с той же случайностью: выборка с температурой 1 и без top-k, тоже 80 слов.
    model = load_model("base")
    text = "Однажды" + tok.decode(continue_ids(tok, model, "Однажды", new=300, temperature=1.0, seed=0))
    booklm_words = WORD.findall(text)[:80]
    print("BookLM, 80 слов:")
    show(join_words(booklm_words))

    # Сколько пар соседних слов из новых фраз в обучающих сказках не встречалось ни разу.
    val = stories(tok, MODELS / "data" / "val.bin")
    phrases = [s for story in val for s in re.split(r"(?<=[.!?])\s+", story) if s.strip()][:200]
    pairs = [(a, b) for p in phrases for a, b in zip(WORD.findall(p), WORD.findall(p)[1:])]
    unseen = [(a, b) for a, b in pairs if counts[a][b] == 0]
    print(f"Новые фразы: 200, пар слов {len(pairs)}, не встречались в обучении {len(unseen)}"
          f" ({100 * len(unseen) / len(pairs):.1f} %)")
    print("Например:", ", ".join(f"«{a} {b}»" for a, b in unseen[:4]))
    own = list(zip(booklm_words, booklm_words[1:]))
    print(f"В тексте BookLM таких пар {sum(counts[a][b] == 0 for a, b in own)} из {len(own)}")


def ch02_embedding_neighbours():
    """Ближайшие токены по косинусу в таблице эмбеддингов до обучения и после."""
    tok = load_tokenizer()
    words = [" кот", " мама", " девочка", " большой", " пошла"]
    print("Пять ближайших токенов по косинусу (_ — пробел перед словом)")
    for kind, title in (("base", "После обучения"), ("init", "До обучения (шаг 0)")):
        table = load_model(kind).embed.weight.detach()            # [8192, 256]
        unit = table / table.norm(dim=1, keepdim=True)            # стрелки одной длины
        print(title + ":")
        for w in words:
            (i,) = tok.encode(w).ids                              # каждое слово — один токен
            cos = unit @ unit[i]                                  # косинус со всеми токенами
            cos[i] = -1                                           # сам токен не считаем
            top = cos.topk(5)
            near = ", ".join(f"{tok.decode([j]).replace(' ', '_')} {c:.2f}"
                             for c, j in zip(top.values.tolist(), top.indices.tolist()))
            print(f"  {w.strip():<8} → {near}")


def ch02_base_vs_instruct():
    """Базовая и дообученная BookLM на одной просьбе: жадный ответ и счёт по 20 ответам."""
    tok = load_tokenizer()
    request = "Расскажи сказку про ёжика, который боялся темноты."
    lora = torch.load(MODELS / "booklm-instruct-lora.pt", map_location="cpu")
    n_lora = sum(v.numel() for v in lora["adapter"].values())
    print(f"Обучаемых чисел в LoRA: {n_lora} из 6622464 ({100 * n_lora / 6622464:.1f} %)")
    print("Просьба:", request)
    hedgehog = re.compile(r"\b[ёе]ж(ик|ек|и|а|у|ом|ей)\w*", re.IGNORECASE)
    for kind, title in (("base", "Базовая BookLM"), ("instruct", "BookLM после дообучения")):
        model = load_model(kind)
        new = continue_ids(tok, model, request, new=80, chat=True)   # жадно, 80 токенов
        print(f"\n{title} (жадно, первые 80 токенов):")
        show(tok.decode(new), limit=5)
        # Одного ответа мало: 20 ответов с выборкой (температура 1, без top-k, зёрна 0..19).
        answers = [tok.decode(continue_ids(tok, model, request, new=480, temperature=1.0, seed=s, chat=True))
                   for s in range(20)]
        fresh = sum(a.lstrip()[:1].isupper() for a in answers)
        print(f"  из 20 ответов: начинают новый текст {fresh}, упоминают ёжика "
              f"{sum(bool(hedgehog.search(a)) for a in answers)}, темноту "
              f"{sum(bool(re.search('т[её]мн', a.lower())) for a in answers)}")


EXPERIMENTS = {"ch02_rule_in_numbers": ch02_rule_in_numbers,
               "ch02_ngram_vs_booklm": ch02_ngram_vs_booklm,
               "ch02_embedding_neighbours": ch02_embedding_neighbours,
               "ch02_base_vs_instruct": ch02_base_vs_instruct}

if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    run(EXPERIMENTS)
