"""Глава 9, раздел «Пробелы в знаниях модели»: где BookLM знает мало и как это чинят целевыми данными.

Опыты (время — на процессоре Apple M2 Max):
  ch09_gaps_freq — рубрика «А теперь на нашей модели: длинный хвост». Угадывает ли база слово
      отложенной истории в зависимости от того, сколько раз это слово встречалось в train.bin.
      Около 25 с. Пишет ch09-gaps-freq.txt и ch09-gaps-freq.csv.
  ch09_gaps_heroes — рубрика «А теперь на нашей модели: пробел в навыке». Доля ответов инструктивной
      модели, где есть герой просьбы, для 60 животных разной частоты в корпусе, по 32 ответа на героя.
      Около 5 минут. Пишет ch09-gaps-heroes.txt и ch09-gaps-heroes.csv.
  ch09_gaps_patch — рубрика «А теперь на нашей модели: целевые данные». Заплатка: LoRA на синтетических
      парах «кот -> редкий герой» без старых пар и со старыми (на одну целевую три старые),
      по три зерна на вариант.
      Около 25 минут подряд или около 10 минут тремя процессами. Каждая часть пишет
      ch09-gaps-patch-<часть>.txt, сводная таблица — ch09-gaps-patch.txt.

Запуск из корня репозитория (CPU, по умолчанию 8 потоков, переменная THREADS меняет их число):
    python -m booklm.experiments.ch09_knowledge_gaps                      # все опыты, около 30 минут
    python -m booklm.experiments.ch09_knowledge_gaps ch09_gaps_freq
    python -m booklm.experiments.ch09_knowledge_gaps ch09_gaps_heroes
    python -m booklm.experiments.ch09_knowledge_gaps ch09_gaps_patch      # всё подряд, около 25 минут
    # или по частям в трёх процессах, по 4 потока:
    THREADS=4 python -m booklm.experiments.ch09_knowledge_gaps ch09_gaps_patch before
    THREADS=4 python -m booklm.experiments.ch09_knowledge_gaps ch09_gaps_patch narrow
    THREADS=4 python -m booklm.experiments.ch09_knowledge_gaps ch09_gaps_patch mix
    python -m booklm.experiments.ch09_knowledge_gaps ch09_gaps_patch join
Файлы выводов лежат в models/booklm/experiments/.
"""
import collections
import csv
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

if not __package__:  # запуск файлом (python booklm/experiments/ch09_knowledge_gaps.py): корень репозитория — в путь поиска
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from booklm.experiments import run
from booklm.lab import DEVICE, MODELS, ROOT, load, perplexity, prompt_ids, special

sys.path.insert(0, str(ROOT / "booklm"))       # sft_lora.py импортирует model без пакета
from sft_lora import add_lora, encode            # noqa: E402
from booklm.experiments.ch08 import held_out, sft_train  # noqa: E402
from booklm.eval_follow import keys              # noqa: E402

OUT = MODELS / "experiments"
WORD = re.compile(r"[а-яё]+")

# Герои-животные. Частые и две группы редких: A — для них соберём данные, B — не будем.
# В A и B только слова мужского рода на согласный: они склоняются как «кот», и подстановка работает.
FREQUENT = ["кот", "собака", "птица", "кролик", "медведь", "лягушка", "белка", "рыбка", "жук"]
RARE_A = ["бегемот", "бобр", "крот", "страус", "енот", "суслик", "дельфин"]
RARE_B = ["хомяк", "барсук", "носорог", "верблюд", "осьминог", "ёжик", "ослик"]
ANIMALS = ("кот собака птица кролик медведь мышка утка лягушка белка рыбка слон сова лиса черепаха дракон "
           "крокодил жираф пингвин дельфин осьминог бобр ящерица улитка ёжик енот хомяк барсук бегемот "
           "единорог лось волк тигр корова ослик павлин верблюд носорог страус панда коала муравей паук "
           "пчела бабочка жук ворона попугай акула кит краб медуза тюлень морж лемур ленивец скунс крот "
           "суслик кабан хорёк").split()
GROUPS = (("частые (1000+ историй)", 1000, 10**9), ("средние (100-999)", 100, 999),
          ("редкие (3-99)", 3, 99), ("почти не было (0-2)", 0, 2))      # по числу историй с героем
ASK = "Можешь рассказать историю, в которой главный герой — {}?"          # просьба при проверке
TRAIN_ASK = ["Сочини короткую сказку, где есть {}.", "Напиши сказку, в которой появляется {}.",
             "Расскажи историю, где участвует {}."]                       # просьбы в обучающих парах


# ----------------------------------------------------------------------------- корпус

_TEXTS = None


def train_texts(tok):
    """Все 68 269 обучающих историй из train.bin текстом."""
    global _TEXTS
    if _TEXTS is None:
        data = np.fromfile(MODELS / "data" / "train.bin", dtype=np.uint16)
        eos = special(tok)["<eos>"]
        cuts = np.flatnonzero(data == eos)
        starts = np.concatenate([[0], cuts[:-1] + 1])
        _TEXTS = tok.decode_batch([data[s:e].tolist() for s, e in zip(starts, cuts)])
    return _TEXTS


def word_counts(tok):
    """Сколько раз каждое слово (строчными, в этой форме) встречается в train.bin."""
    cnt = collections.Counter()
    for t in train_texts(tok):
        cnt.update(WORD.findall(t.lower()))
    return cnt


def norm(text):
    return text.lower().replace("ё", "е")


def hero_pattern(hero):
    """Слово героя в любом падеже: основа и до трёх букв окончания («бобр», «бобра», «бобрами»)."""
    s = norm(hero)
    s = s[:-1] if s[-1] in "аяоьйы" else s
    k = 3 if len(s) >= 5 else 2
    return re.compile(r"(?<![а-я])" + s + r"[а-я]{0,%d}(?![а-я])" % k)


# Слова, которые подходят под шаблон героя, но героем не являются: кошка Китти, лист, совет, котёл.
NOT_HERO = set("китти кити китай китая лист листа листе листу листы листом листов совет совок совка "
               "котел котле котла".split())


def stories_with(tok, hero):
    """В скольких обучающих историях встречается герой (без слов из NOT_HERO)."""
    p = hero_pattern(hero)
    return sum(1 for t in train_texts(tok)
               if any(m.group(0) not in NOT_HERO for m in p.finditer(norm(t))))


# ----------------------------------------------------------------------------- угадывание слов

@torch.no_grad()
def val_words(tok, model, T=512, S=256):
    """Слова отложенных историй: (слово, угадано ли целиком, ошибка на слово в натах).

    Слово угадано, если каждый его токен — самый вероятный при настоящем начале текста.
    Модель читает val.bin окнами по T токенов со сдвигом S, так что у каждого токена есть контекст.
    """
    val = np.fromfile(MODELS / "data" / "val.bin", dtype=np.uint16).astype(np.int64)
    n, eos = len(val), special(tok)["<eos>"]
    pred, loss = np.full(n, -1), np.zeros(n)
    pos = 0
    while pos < n - 1:
        lo = max(0, pos - (T - S)) if pos else 0
        hi = min(n, lo + T + 1)
        logits = model(torch.tensor(val[lo:hi - 1])[None].to(DEVICE))[0].float()
        y = torch.tensor(val[lo + 1:hi])
        lp = -logits.log_softmax(-1)[torch.arange(len(y)), y]
        k0 = pos - lo
        pred[lo + 1 + k0:hi] = logits.argmax(-1)[k0:].numpy()
        loss[lo + 1 + k0:hi] = lp[k0:].numpy()
        pos = hi - 1
    pieces = [tok.decode([i]) for i in range(tok.get_vocab_size())]
    spans, cur = [], None
    for j in range(1, n):                      # слово: токены подряд, первый начинается с буквы
        p = pieces[val[j]]
        starts_word = p[:1] in " \n«—(\"'" or cur is None
        is_letter = bool(re.match(r"^[а-яёА-ЯЁ]", p.lstrip(" \n«—(\"'")))
        if val[j] == eos or not is_letter or starts_word:
            if cur:
                spans.append(cur)
            cur = [j] if (val[j] != eos and is_letter) else None
        else:
            cur.append(j)
    if cur:
        spans.append(cur)
    rows = []
    for span in spans:
        w = "".join(WORD.findall("".join(pieces[val[j]] for j in span).lower()))
        if w:
            rows.append((w, all(pred[j] == val[j] for j in span), float(loss[span].sum())))
    return rows


BINS = [(0, 0), (1, 9), (10, 99), (100, 999), (1000, 9999), (10000, 99999), (100000, 10**9)]


def bin_label(a, b):
    return "0" if b == 0 else (f"{a}+" if b >= 10**9 else f"{a}-{b}")


def ch09_gaps_freq():
    """Угадывает ли база слово отложенной истории в зависимости от его частоты в train.bin."""
    tok, base = load("base")
    cnt = word_counts(tok)
    rows = val_words(tok, base)
    lines = [f"слов в train: {sum(cnt.values())}, разных: {len(cnt)}; слов в val: {len(rows)}",
             "частота в train   слов   угадано   ошибка на слово"]
    table = []
    for a, b in BINS:
        r = [x for x in rows if a <= cnt.get(x[0], 0) <= b]
        acc, ls = sum(x[1] for x in r) / len(r), sum(x[2] for x in r) / len(r)
        table.append((bin_label(a, b), len(r), acc, ls))
        lines.append(f"{bin_label(a, b):>13s} {len(r):8d} {acc:9.2f} {ls:12.2f}")
    text = "\n".join(lines)
    print(text)
    (OUT / "ch09-gaps-freq.txt").write_text(text + "\n", encoding="utf-8")
    with open(OUT / "ch09-gaps-freq.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["bin", "lo", "words", "acc", "loss"])
        for (lab, nw, acc, ls), (a, _) in zip(table, BINS):
            w.writerow([lab, a, nw, f"{acc:.4f}", f"{ls:.4f}"])


# ----------------------------------------------------------------------------- герой просьбы

@torch.no_grad()
def sample_batch(model, ids, n, max_new=150, temperature=0.8, top_k=40, seed=0, eos_id=2):
    """n ответов на одну просьбу сразу, пакетом: те же temperature и top_k, что у model.generate."""
    gen = torch.Generator().manual_seed(seed)
    x = torch.tensor([ids] * n, device=DEVICE)
    caches = [{"len": 0} for _ in model.blocks]
    logits = model(x, caches)
    for c in caches:
        c["len"] = x.size(1)
    out, done = [], torch.zeros(n, dtype=torch.bool)
    for _ in range(max_new):
        last = logits[:, -1, :].float() / temperature
        kth = last.topk(top_k).values[:, -1:]
        last = last.masked_fill(last < kth, float("-inf"))
        nxt = torch.multinomial(last.softmax(-1), 1, generator=gen)
        out.append(nxt)
        done |= nxt[:, 0] == eos_id
        if done.all() or len(ids) + len(out) >= model.cfg.max_len:
            break
        logits = model(nxt, caches)
        for c in caches:
            c["len"] += 1
    rows = torch.cat(out, 1).tolist()
    return [r[:r.index(eos_id)] if eos_id in r else r for r in rows]


def hero_share(tok, model, hero, n=32, seed=0, ask=ASK):
    """Доля из n ответов на просьбу о герое, где этот герой назван хотя бы раз."""
    ids = prompt_ids(tok, ask.format(hero), chat=True)
    eos = special(tok)["<eos>"]
    answers = tok.decode_batch(sample_batch(model, ids, n, seed=seed, eos_id=eos))
    p = hero_pattern(hero)
    return sum(bool(p.search(norm(a))) for a in answers) / n


def ch09_gaps_heroes(n=32):
    """Доля ответов с героем просьбы для 60 животных разной частоты в корпусе."""
    tok, inst = load("instruct")
    t0 = time.time()
    rows = []
    for hero in ANIMALS:
        rows.append((hero, stories_with(tok, hero), hero_share(tok, inst, hero, n)))
    rows.sort(key=lambda r: -r[1])
    lines = [f"{len(ANIMALS)} героев, по {n} ответов на каждого"]
    for name, lo, hi in GROUPS:
        r = [x for x in rows if lo <= x[1] <= hi]
        lines.append(f"{name:24s} героев {len(r):2d}, доля ответов с героем "
                     f"{sum(x[2] for x in r) / len(r):.2f}")
    lines.append("по героям: " + ", ".join(f"{h} {d}:{s:.2f}" for h, d, s in rows))
    lines.append(f"[{time.time() - t0:.0f} с]")
    text = "\n".join(lines)
    print(text)
    (OUT / "ch09-gaps-heroes.txt").write_text(text + "\n", encoding="utf-8")
    with open(OUT / "ch09-gaps-heroes.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["hero", "stories", "share"])
        for h, d, s in rows:
            w.writerow([h, d, f"{s:.4f}"])


# ----------------------------------------------------------------------------- заплатка

KOT = re.compile(r"(?<![А-Яа-яЁё])([Кк])от(|а|у|ом|е|ы|ов|ам|ами|ах)(?![А-Яа-яЁё])")


def substitute_pairs(tok, heroes, per_hero=40, seed=0):
    """Синтетика подстановкой: истории, где кот назван не меньше трёх раз, с котом, заменённым на героя."""
    texts = train_texts(tok)
    cat = hero_pattern("кот")
    idx = [i for i, t in enumerate(texts)
           if len(cat.findall(norm(t))) >= 3 and "кошк" not in norm(t) and "котен" not in norm(t)]
    random.Random(seed).shuffle(idx)
    pairs = []
    for j, hero in enumerate(heroes):
        for i in idx[j * per_hero:(j + 1) * per_hero]:
            story = KOT.sub(lambda m: (hero[0].upper() if m.group(1) == "К" else hero[0])
                            + hero[1:] + m.group(2), texts[i])
            pairs.append((random.Random(i).choice(TRAIN_ASK).format(hero), story))
    return pairs


def old_pairs(n):
    """Первые n пар из обучающих данных инструктивной модели (те же, что видел sft_lora.py)."""
    return [(r["request"], r["story"]) for r in sft_train()[:n]]


def lora_patch(tok, model, pairs, steps=150, lr=5e-4, seed=0):
    """LoRA r=16 поверх инструктивной модели: steps шагов по 8 пар, ошибка только на ответе."""
    torch.manual_seed(seed)
    add_lora(model, 16, 32)
    enc = [encode(tok, q, s, 512) for q, s in pairs]
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    gen = torch.Generator().manual_seed(seed)
    model.train()
    for _ in range(steps):
        batch = [enc[i] for i in torch.randint(len(enc), (8,), generator=gen).tolist()]
        L = max(len(ids) for ids, _ in batch)
        x = torch.tensor([ids[:-1] + [0] * (L - len(ids)) for ids, _ in batch])
        y = torch.tensor([lab[1:] + [-100] * (L - len(lab)) for _, lab in batch])
        loss = F.cross_entropy(model(x).view(-1, model.cfg.vocab), y.view(-1), ignore_index=-100)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model.eval()


@torch.no_grad()
def request_words(tok, model, n=50):
    """Доля главных слов просьбы в ответах на n отложенных просьб (как booklm/eval_follow.py)."""
    eos, hit, total = special(tok)["<eos>"], 0, 0
    for i, row in enumerate(held_out()[:n]):
        ids = prompt_ids(tok, row["request"], chat=True)
        out = model.generate(torch.tensor([ids], device=DEVICE), max_new=200, temperature=0.8,
                             top_k=40, eos_id=eos, seed=i)
        text, k = tok.decode(out[0, len(ids):].tolist()).lower(), keys(row["request"])
        hit += sum(w in text for w in k)
        total += len(k)
    return hit / total


def gap_report(tok, model, n=32):
    """Пять чисел: доля героя у частых, у редких A и B, перплексия и слова просьбы."""
    share = {g: sum(hero_share(tok, model, h, n) for h in hs) / len(hs)
             for g, hs in (("частые", FREQUENT), ("A", RARE_A), ("B", RARE_B))}
    return share["частые"], share["A"], share["B"], perplexity(model), request_words(tok, model)


ROW = "{:25s}{:7.2f}{:10.2f}{:10.2f}{:12.2f}{:9.2f}"
HEAD = f"{'данные':25s}{'частые':>7s}{'редкие A':>10s}{'редкие B':>10s}{'перплексия':>12s}{'просьба':>9s}"


def patch_rows(tok, variant, seeds=(0, 1, 2)):
    """Строки таблицы для одного варианта данных: по строке на зерно и среднее."""
    target = substitute_pairs(tok, RARE_A)
    data = target if variant == "только целевые" else target + old_pairs(3 * len(target))
    runs, lines = [], []
    for s in seeds:
        _, m = load("instruct")
        runs.append(gap_report(tok, lora_patch(tok, m, data, seed=s)))
        lines.append(ROW.format(f"{variant}, зерно {s}", *runs[-1]))
        print(lines[-1], flush=True)
    lines.append(ROW.format(f"{variant}, среднее", *[sum(r[k] for r in runs) / len(runs) for k in range(5)]))
    return lines


def ch09_gaps_patch(part="all"):
    """Заплатка LoRA для редких героев; часть: before, narrow, mix, join или all (всё подряд)."""
    tok, inst = load("instruct")
    t0 = time.time()
    parts = {"before": None, "narrow": "только целевые", "mix": "целевые + старые"}
    todo = list(parts) if part == "all" else [part]
    for p in todo:
        if p == "join":
            continue
        if p == "before":
            target = substitute_pairs(tok, RARE_A)
            lines = [f"целевых пар {len(target)}, пример: {target[0][0]} | {target[0][1][:100]}",
                     HEAD, ROW.format("до заплатки", *gap_report(tok, inst))]
        else:
            lines = patch_rows(tok, parts[p])
        lines.append(f"[{time.time() - t0:.0f} с]")
        (OUT / f"ch09-gaps-patch-{p}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("\n".join(lines), flush=True)
    if part in ("all", "join"):
        text = "".join((OUT / f"ch09-gaps-patch-{p}.txt").read_text(encoding="utf-8") for p in parts)
        (OUT / "ch09-gaps-patch.txt").write_text(text, encoding="utf-8")


EXPERIMENTS = {f.__name__: f for f in (ch09_gaps_freq, ch09_gaps_heroes, ch09_gaps_patch)}

if __name__ == "__main__":
    torch.set_num_threads(int(__import__("os").environ.get("THREADS", 8)))
    run(EXPERIMENTS, args=True)          # ch09_gaps_patch before|narrow|mix|join — часть опыта
