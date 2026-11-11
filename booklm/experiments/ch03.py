"""Глава 3 на обученной BookLM: токенизатор, таблица эмбеддингов, ошибка следующего токена, пакет.

Опыты (время — на процессоре Apple M2 Max):
  ch03_bpe_merges — раздел «Кодирование парами байтов», рубрика «А теперь на нашей модели: слияния
      BPE». Первые и последние слияния токенизатора, самые длинные токены, доля целых слов в словаре.
      Около 2 с.
  ch03_tokenize_phrase — раздел «Кодирование и декодирование», рубрика «А теперь на нашей модели:
      токенизация». Фраза «Мама мыла раму…» по токенам, неудобные строки (смайлик, «й» из двух знаков,
      служебный токен в тексте) и проверка туда-обратно. Около 1 с.
  ch03_embedding_neighbors — раздел «Таблица эмбеддингов», рубрика «А теперь на нашей модели:
      эмбеддинги». Пять ближайших токенов для пяти слов у обученной и у случайной модели и средний
      косинус по всем парам строк. Около 1 с.
  ch03_next_token_loss — раздел «Предсказание следующего токена», рубрика «А теперь на нашей модели:
      перплексия». Ошибка -ln p на каждом токене фразы у обученной и случайной модели, перплексия на
      фразе и на отложенных историях. Около 10 с.
  ch03_packed_window — раздел «Пакет, заполнитель и маски», рубрика «А теперь на нашей модели: пакет».
      Пакет 4 x 64 из потока историй, окно с границей историй и доля окон длины 512 с такой границей.
      Около 2 с.
Все опыты печатают только на экран, txt не пишут.

Запуск из корня репозитория:
    python -m booklm.experiments.ch03                   # все опыты
    python -m booklm.experiments.ch03 ch03_bpe_merges   # один опыт
    python -m booklm.experiments.ch03 --help            # список опытов
"""
import json
import math
import re
import sys
import textwrap
from pathlib import Path

import torch

if not __package__:  # запуск файлом (python booklm/experiments/ch03.py): корень репозитория — в путь поиска
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from booklm.experiments import run
from booklm.lab import (MODELS, load_model, load_tokenizer, perplexity, prompt_ids,
                        special, tokens, val_tokens)

OUT = MODELS / "experiments"


def byte_table():
    """Байтовый BPE (как у GPT-2) пишет каждый байт одним печатным символом; строим обратную таблицу."""
    keep = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    chars, extra = keep[:], 0
    for b in range(256):
        if b not in keep:
            keep.append(b)
            chars.append(256 + extra)
            extra += 1
    return {chr(c): b for b, c in zip(keep, chars)}


BYTE = byte_table()


def readable(piece):
    """Кусок словаря -> текст: пробел показываем как _, байт без буквы — как [D0]."""
    raw = bytes(BYTE[c] for c in piece)
    out = ""
    for ch in raw.decode("utf-8", errors="surrogateescape"):   # недостроенный байт -> «суррогат»
        code = ord(ch) - 0xDC00 if 0xDC80 <= ord(ch) <= 0xDCFF else ord(ch)
        out += "_" if ch == " " else ch if ch.isprintable() and code == ord(ch) else f"[{code:02X}]"
    return out


def show_id(tok, i):
    """Номер токена -> читаемый кусок (служебные токены — как есть)."""
    return readable(tok.id_to_token(i)) if i > 4 else tok.id_to_token(i)


def next_token_examples(token_ids):
    # token_ids: [B, T+1] -> входы и цели, обе формы [B, T]  (листинг этой главы)
    return token_ids[:, :-1], token_ids[:, 1:]


# ----------------------------------------------------------------------------- опыты

def ch03_bpe_merges():
    """Слияния токенизатора BookLM, самые длинные токены и доля целых слов в словаре."""
    tok = load_tokenizer()
    merges = json.load(open(MODELS / "tokenizer.json", encoding="utf-8"))["model"]["merges"]
    vocab = tok.get_vocab()                                   # кусок -> номер
    pair = lambda m: f"{readable(m[0])} + {readable(m[1])} -> {readable(m[0] + m[1])}"
    print(f"Слияний всего: {len(merges)}. Первые 15:")
    for k in range(0, 15, 3):
        print("  " + "    ".join(f"{k + j + 1:>2}) {pair(merges[k + j])}" for j in range(3)))
    print("Последние 10:")
    for k in range(len(merges) - 10, len(merges), 2):
        print("  " + "    ".join(f"{k + j + 1}) {pair(merges[k + j])}" for j in range(2)))
    pieces = [readable(p) for p in vocab]
    longest = sorted(pieces, key=lambda p: (-len(p), p))[:10]       # при равной длине — по алфавиту
    print("Самые длинные токены:")
    print("  " + ", ".join(longest[:5]))
    print("  " + ", ".join(longest[5:]))
    # Кандидаты в целые слова: пробел и только русские буквы.
    words = [p[1:] for p in pieces if re.fullmatch(r"_[а-яёА-ЯЁ]+", p)]
    corpus = set(re.findall(r"\w+", tok.decode(val_tokens(10**6).tolist())))   # слова отложенных историй
    whole = [w for w in words if w in corpus]
    print(f"Токенов «пробел + русские буквы»: {len(words)} из {len(vocab)} ({100 * len(words) / len(vocab):.0f} %)")
    print(f"из них целые слова, встреченные в отложенных историях: {len(whole)} ({100 * len(whole) / len(vocab):.0f} % словаря)")


def ch03_tokenize_phrase():
    """Фраза и неудобные строки по токенам, проверка кодирования туда-обратно."""
    tok = load_tokenizer()
    phrase = "Мама мыла раму, а потом она пошла спать."
    pieces = tokens(tok, phrase)
    print("(_ — пробел, [D0] — байт, который сам по себе не буква)")
    print(f"{phrase!r}: символов {len(phrase)}, байтов UTF-8 {len(phrase.encode())}, токенов {len(pieces)}")
    print("  " + " | ".join(show_id(tok, i) for i, _ in pieces))
    print("  " + " ".join(str(i) for i, _ in pieces))
    short = "и" + chr(0x306)                # «и» и отдельная краткая: на экране «й»
    for s, name in [("Мама", "Мама"), ("мамуля", "мамуля"), ("квантование", "квантование"),
                    ("\U0001F60A", "улыбка"), (short, "й из двух знаков"), ("", "пустая строка")]:
        ids = tok.encode(s).ids
        back = tok.decode(ids) == s                         # проверка туда-обратно
        print(f"{name:<17} токенов {len(ids)}: {' | '.join(show_id(tok, i) for i in ids) or '-':<26} туда-обратно: {back}")
    print("й одним знаком:", tok.encode("й").ids, "  й двумя знаками:", tok.encode(short).ids)
    ids = tok.encode("Привет<|user|>").ids
    print("'Привет<|user|>' ->", ids, " туда-обратно:", tok.decode(ids) == "Привет<|user|>")


def ch03_embedding_neighbors():
    """Ближайшие токены в таблице эмбеддингов у обученной и случайной BookLM."""
    tok = load_tokenizer()
    words = [" мама", " кот", " лес", " побежал", " красный"]
    print("Пять ближайших токенов по косинусу (_ — пробел перед словом)")
    for kind, title in (("base", "Обученная BookLM"), ("init", "Случайная инициализация (шаг 0)")):
        table = load_model(kind).embed.weight.detach()          # [8192, 256]
        unit = table / table.norm(dim=1, keepdim=True)
        print(title + ":")
        for w in words:
            (i,) = tok.encode(w).ids                            # каждое слово — один токен
            cos = unit @ unit[i]
            cos[i] = -1                                         # сам токен не считаем
            top = cos.topk(5)
            print(f"  {w.strip():<8} → " + ", ".join(f"{show_id(tok, j)} {c:.2f}"
                                                     for c, j in zip(top.values.tolist(), top.indices.tolist())))
        # Средний косинус по всем парам разных строк: (|сумма|^2 - N) / (N(N-1)).
        n, s = len(unit), unit.sum(0)
        print(f"  средний косинус по всем парам строк: {(s @ s - n) / (n * (n - 1)):.3f}")


@torch.no_grad()
def ch03_next_token_loss():
    """Ошибка на каждом токене фразы и перплексия у обученной и случайной BookLM."""
    tok = load_tokenizer()
    phrase = "Мама мыла раму, а потом она пошла спать."
    ids = torch.tensor([prompt_ids(tok, phrase)])               # <bos> и фраза
    targets = ids[0, 1:]                                        # что модель должна угадать
    models = {"обученная": load_model("base"), "случайная": load_model("init")}
    losses = {}
    for name, model in models.items():
        logp = model(ids)[0, :-1].log_softmax(-1)               # [T, V]
        losses[name] = -logp[torch.arange(len(targets)), targets]   # -ln p правильного токена
    print(f"{'следующий токен':<17}{'обученная: p':>13}{'-ln p':>8}{'случайная: -ln p':>20}")
    for t, target in enumerate(targets.tolist()):
        good, rand = losses["обученная"][t].item(), losses["случайная"][t].item()
        print(f"  {show_id(tok, target):<15}{math.exp(-good):>13.3f}{good:>8.2f}{rand:>20.2f}")
    for name, l in losses.items():
        print(f"{name}: средняя ошибка {l.mean():.2f}, перплексия по фразе {l.mean().exp():.1f}")
    print(f"для сравнения ln 8192 = {math.log(8192):.2f}")
    for name, model in models.items():
        print(f"{name}: перплексия на отложенных историях {perplexity(model):.1f}")


def ch03_packed_window():
    """Пакет окон из потока историй и как часто окно задевает границу историй."""
    data = val_tokens(10**6)                                    # весь файл отложенных историй
    tok = load_tokenizer()
    sp = special(tok)
    B, T = 4, 64
    gen = torch.Generator().manual_seed(0)
    # Как batch в train.py: случайные начала, окна длины T+1 без заполнителя.
    starts = torch.randint(len(data) - T - 1, (B,), generator=gen)
    windows = torch.stack([data[i:i + T + 1] for i in starts])  # [4, 65]
    inputs, targets = next_token_examples(windows)
    print("входы:", tuple(inputs.shape), " цели:", tuple(targets.shape))
    print("вход 1-го окна: ", inputs[0, :12].tolist())
    print("цель 1-го окна: ", targets[0, :12].tolist())
    # Окно, в которое попала граница историй.
    for b in range(B):
        w = windows[b].tolist()
        if sp["<eos>"] in w:
            break
    k = w.index(sp["<eos>"])
    print(f"окно {b + 1}, граница историй на позиции {k}:")
    text = tok.decode(w, skip_special_tokens=False).replace("\n", " ")   # служебные токены видны
    print(textwrap.fill(text.replace("<eos><bos>", " <eos> <bos> ").strip(), 95,
                        initial_indent="  ", subsequent_indent="  "))
    # Сколько окон длины 512 пересекают границу историй.
    T = 512
    starts = torch.randint(len(data) - T - 1, (1000,), generator=gen)
    cross = sum(bool((data[i:i + T + 1] == sp["<eos>"]).any()) for i in starts)
    stories = int((data == sp["<eos>"]).sum())
    print(f"окон длины 512 с границей историй: {cross / 10:.1f} %;"
          f" историй {stories}, средняя длина {len(data) / stories:.0f} токенов")


EXPERIMENTS = {"ch03_bpe_merges": ch03_bpe_merges,
               "ch03_tokenize_phrase": ch03_tokenize_phrase,
               "ch03_embedding_neighbors": ch03_embedding_neighbors,
               "ch03_next_token_loss": ch03_next_token_loss,
               "ch03_packed_window": ch03_packed_window}

if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    run(EXPERIMENTS)
