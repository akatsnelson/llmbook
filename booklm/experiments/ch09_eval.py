"""Глава 9 на обученной BookLM: экзамен «Сказки-100», загрязнение теста, калибровка уверенности на именах
героев.

Опыты (время — на процессоре Apple M2 Max):
  ch09_fairy_exam — раздел «Оценка моделей», рубрика «А теперь на нашей модели: тест с вариантами
      ответа». 100 вопросов по отложенным историям: начало истории и четыре концовки. Точность базы
      и инструктивной модели по сумме log p (acc), по среднему на токен (acc_norm) и по букве ответа.
      Около 2 минут.
  ch09_contamination — подраздел «Загрязнение теста». Сколько вопросов теста делят 13-грамму с train.bin
      и как растёт балл, если подмешать половину вопросов в дообучение. Около 2 минут.
  ch09_name_calibration — раздел «Выдумки и калибровка», рубрика «А теперь на нашей модели:
      калибровка». Уверенность и точность BookLM, когда она угадывает имя героя, по корзинам
      уверенности и ECE. Около 10 с.
Все опыты печатают только на экран, txt не пишут.

Запуск из корня репозитория:
    python -m booklm.experiments.ch09_eval                    # все опыты
    python -m booklm.experiments.ch09_eval ch09_fairy_exam    # один опыт
    python -m booklm.experiments.ch09_eval --help             # список опытов
"""
import copy
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

if not __package__:  # запуск файлом (python booklm/experiments/ch09_eval.py): корень репозитория — в путь поиска
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from booklm.experiments import run
from booklm.lab import MODELS, encode_text, load, load_model, special
from booklm.experiments.ch07 import val_stories

OUT = MODELS / "experiments"
LETTERS = "АБВГ"


# ----------------------------------------------------------------------------- экзамен «Сказки-100»

def split_last(text):
    """(всё до последней фразы, последняя фраза) или None, если фраза одна."""
    t = text.strip()
    cuts = list(re.finditer(r"(?<=[.!?»])\s+(?=[А-ЯЁ«])", t[:-1]))
    if not cuts:
        return None
    i = cuts[-1].end()
    return t[:i].rstrip(), t[i:]


def fairy_exam(tok, n=100, seed=0):
    """n вопросов по отложенным историям: начало истории и четыре концовки, одна настоящая.

    Неверные концовки — последние фразы других отложенных историй, не попавших в экзамен.
    Возвращает (вопросы, номера историй, которые в экзамен не попали).
    """
    stories = [tok.decode(s) for s in val_stories()]
    cands = []
    for i, s in enumerate(stories):
        parts = split_last(s)
        if parts and 4 <= len(parts[1].split()) <= 20 and len(encode_text(tok, parts[0])) < 400:
            cands.append((i, *parts))
    rng = random.Random(seed)
    rng.shuffle(cands)
    test, pool = cands[:n], [c[2] for c in cands[n:]]
    items = []
    for i, context, ending in test:
        options = [ending] + rng.sample(pool, 3)
        rng.shuffle(options)
        items.append({"story": i, "context": context, "options": options, "answer": options.index(ending)})
    used = {it["story"] for it in items}
    return items, [i for i in range(len(stories)) if i not in used]


@torch.no_grad()
def option_logp(tok, model, context, option):
    """Сумма log p токенов варианта после начала истории и число этих токенов."""
    pre = [special(tok)["<bos>"]] + encode_text(tok, context)
    ids = pre + encode_text(tok, " " + option)
    x, y = torch.tensor([ids[:-1]]), torch.tensor(ids[1:])
    logp = model(x)[0].log_softmax(-1).gather(-1, y[:, None])[:, 0]
    return logp[len(pre) - 1:].sum().item(), len(ids) - len(pre)


def cloze_answers(tok, model, items):
    """Выбор по сумме log p (acc) и по среднему на токен (acc_norm): списки номеров вариантов."""
    by_sum, by_mean = [], []
    for it in items:
        s = [option_logp(tok, model, it["context"], o) for o in it["options"]]
        by_sum.append(max(range(4), key=lambda j: s[j][0]))
        by_mean.append(max(range(4), key=lambda j: s[j][0] / s[j][1]))
    return by_sum, by_mean


def letter_prompt(item, order):
    text = item["context"] + "\n\nКакой фразой кончается сказка?\n"
    text += "\n".join(f"{LETTERS[k]}) {item['options'][j]}" for k, j in enumerate(order))
    return text + "\nОтвет:"


@torch.no_grad()
def letter_choice(tok, model, item, order, chat):
    """Буква с наибольшей вероятностью среди « А», « Б», « В», « Г» и доля вероятности на этих четырёх."""
    sp = special(tok)
    body = encode_text(tok, letter_prompt(item, order))
    ids = [sp["<bos>"], sp["<|user|>"]] + body + [sp["<|assistant|>"]] if chat else [sp["<bos>"]] + body
    p = model(torch.tensor([ids]))[0, -1].softmax(-1)
    letter_ids = [encode_text(tok, " " + c)[0] for c in LETTERS]
    q = p[letter_ids]
    return int(q.argmax()), q.sum().item(), ids


@torch.no_grad()
def generated_letter(tok, model, ids):
    """Жадно генерирует 8 токенов и ищет в начале ответа букву варианта; None — буквы нет."""
    out = model.generate(torch.tensor([ids]), max_new=8, temperature=0, eos_id=special(tok)["<eos>"])
    text = tok.decode(out[0, len(ids):].tolist()).strip()
    m = re.match(r"[«(]?([АБВГ])\)", text)
    return (LETTERS.index(m.group(1)) if m else None), text


def ch09_fairy_exam():
    """Тест «Сказки-100» для базы и инструктивной BookLM: acc, acc_norm и буква ответа."""
    tok, base = load("base")
    _, inst = load("instruct")
    items, _ = fairy_exam(tok)
    gold = [it["answer"] for it in items]
    for name, m, chat in (("база", base, False), ("instruct", inst, True)):
        by_sum, by_mean = cloze_answers(tok, m, items)
        acc = sum(a == g for a, g in zip(by_sum, gold))
        acc_norm = sum(a == g for a, g in zip(by_mean, gold))
        shifts, picks, gen_ok, gen_any = [], [], 0, 0
        for shift in range(4):
            right = 0
            for it in items:
                order = [(j + shift) % 4 for j in range(4)]
                k, mass, ids = letter_choice(tok, m, it, order, chat)
                right += order[k] == it["answer"]
                picks.append(LETTERS[k])
                if shift == 0:
                    g, text = generated_letter(tok, m, ids)
                    gen_any += g is not None
                    gen_ok += g is not None and order[g] == it["answer"]
            shifts.append(right)
        top = max(set(picks), key=picks.count)
        print(f"{name}: acc {acc}, acc_norm {acc_norm}; буква: {shifts} при четырёх сдвигах, "
              f"«{top}» в {picks.count(top)} из {len(picks)}; текст: буква в {gen_any}, верно {gen_ok}")
    for step in (0, 100, 762, 1525, 3051):
        m = load_model(f"step:{step}")
        _, by_mean = cloze_answers(tok, m, items)
        print(f"шаг {step}: acc_norm {sum(a == g for a, g in zip(by_mean, gold))}")


# ----------------------------------------------------------------------------- загрязнение

def train_ngrams(n=13):
    """Хеши всех n-грамм токенов train.bin (numpy, без Python-множества на 12 млн элементов)."""
    data = np.fromfile(MODELS / "data" / "train.bin", dtype=np.uint16).astype(np.uint64)
    h = np.zeros(len(data) - n + 1, dtype=np.uint64)
    for k in range(n):
        h = h * np.uint64(8209) + data[k:len(data) - n + 1 + k]
    return np.unique(h)


def in_train(hashes, grams):
    """Есть ли хоть один из hashes среди отсортированных grams (поиск делением пополам)."""
    if len(hashes) == 0:
        return False
    pos = np.minimum(np.searchsorted(grams, hashes), len(grams) - 1)
    return bool((grams[pos] == hashes).any())


def ngram_hashes(ids, n=13):
    a = np.array(ids, dtype=np.uint64)
    if len(a) < n:
        return np.zeros(0, dtype=np.uint64)
    h = np.zeros(len(a) - n + 1, dtype=np.uint64)
    for k in range(n):
        h = h * np.uint64(8209) + a[k:len(a) - n + 1 + k]
    return h


def story_ids(tok, item):
    return encode_text(tok, item["context"] + " " + item["options"][item["answer"]])


def contaminate(tok, model, stories, repeats, steps=50, lr=1e-4, seed=0):
    """Копия модели, дообученная steps шагов на обычных окнах train.bin (8 окон по 256 токенов на шаг),
    куда подмешаны stories, каждая repeats раз."""
    m = copy.deepcopy(model).train()
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.0)
    data = torch.from_numpy(np.fromfile(MODELS / "data" / "train.bin", dtype=np.uint16).astype(np.int64))
    g = torch.Generator().manual_seed(seed)
    queue = [s for _ in range(repeats) for s in stories]
    random.Random(seed).shuffle(queue)
    per_step = -(-len(queue) // steps)                     # сколько подмешанных историй на шаг
    for step in range(steps):
        starts = torch.randint(0, len(data) - 257, (8,), generator=g)
        x = torch.stack([data[s:s + 257] for s in starts])
        loss = F.cross_entropy(m(x[:, :-1]).flatten(0, 1), x[:, 1:].flatten(), reduction="sum")
        count = x[:, 1:].numel()
        batch = [[1] + s + [2] for s in queue[step * per_step:(step + 1) * per_step]]
        if batch:
            T = max(map(len, batch))
            y = torch.tensor([b + [0] * (T - len(b)) for b in batch])
            labels = torch.tensor([b[1:] + [-100] * (T - len(b)) for b in batch])
            loss = loss + F.cross_entropy(m(y[:, :-1]).flatten(0, 1), labels.flatten(),
                                          ignore_index=-100, reduction="sum")
            count += int((labels != -100).sum())
        opt.zero_grad()
        (loss / count).backward()
        opt.step()
    return m.eval()


@torch.no_grad()
def mean_story_loss(model, stories):
    total, count = 0.0, 0
    for s in stories:
        x = torch.tensor([[1] + s + [2]])
        total += F.cross_entropy(model(x[:, :-1])[0], x[0, 1:], reduction="sum").item()
        count += len(s) + 1
    return total / count


def other_stories(tok, rest, n=100):
    """Первые n отложенных историй вне экзамена, короче 500 токенов: на них следим за общей ошибкой."""
    stories = [encode_text(tok, tok.decode(s)) for s in val_stories()]
    return [stories[i] for i in rest if len(stories[i]) < 500][:n]


def ch09_contamination():
    """Общие 13-граммы теста с train.bin и балл после нарочного загрязнения."""
    tok, base = load("base")
    items, rest = fairy_exam(tok)
    grams = train_ngrams()
    hits = [in_train(ngram_hashes(story_ids(tok, it)), grams) for it in items]
    ending_hits = [in_train(ngram_hashes(encode_text(tok, " " + it["options"][it["answer"]])), grams)
                   for it in items]
    print(f"вопросов с общей 13-граммой: {sum(hits)} из {len(items)}; "
          f"из них в самой концовке: {sum(ending_hits)}")
    gold = [it["answer"] for it in items]
    dirty, clean = items[:50], items[50:]
    others = other_stories(tok, rest)
    for repeats in (None, 0, 1, 4):                      # None — база без дообучения
        m = base if repeats is None else contaminate(tok, base, [story_ids(tok, it) for it in dirty], repeats)
        _, by_mean = cloze_answers(tok, m, items)
        right = [a == g for a, g in zip(by_mean, gold)]
        print(f"{'база' if repeats is None else f'подмешано {repeats} раз'}: "
              f"acc_norm {sum(right[:50])}/50 на подмешанных, "
              f"{sum(right[50:])}/50 на чистых, ошибка на других историях {mean_story_loss(m, others):.3f}")


# ----------------------------------------------------------------------------- калибровка на именах

NAME = re.compile(r"(?<=[а-яё,] )([А-ЯЁ][а-яё]+)")


def name_cuts(tok):
    """Две группы обрезанных историй: перед именем в последней фразе, где имя уже звучало,
    и перед первым упоминанием имени после «по имени», где угадать его не из чего."""
    seen, first = [], []
    for s in val_stories():
        t = tok.decode(s).strip()
        parts = split_last(t)
        if parts:
            i = len(t) - len(parts[1])
            m = NAME.search(t, i)
            if m and m.group(1)[:3] in t[:i]:            # то же имя, возможно в другом падеже
                seen.append((t[:m.start()].rstrip(), m.group(1)))
        f = re.search(r"по имени ([А-ЯЁ][а-яё]+)", t)
        if f:
            first.append((t[:f.start(1)].rstrip(), f.group(1)))
    return seen, first


@torch.no_grad()
def name_guess(tok, model, before, name):
    """Уверенность (вероятность самого вероятного токена), угадала ли модель первый токен имени, и её догадка."""
    p = model(torch.tensor([[special(tok)["<bos>"]] + encode_text(tok, before)]))[0, -1].softmax(-1)
    conf, k = p.max(0)
    return conf.item(), int(k) == encode_text(tok, " " + name)[0], tok.decode([int(k)])


def calibration_table(rows, edges=(0, 0.2, 0.4, 0.6, 0.8, 1.0)):
    """Строки (корзина, число, средняя уверенность, точность) и ECE."""
    out, ece = [], 0.0
    for lo, hi in zip(edges, edges[1:]):
        b = [(c, ok) for c, ok in rows if lo <= c < hi or (hi == 1.0 and c == 1.0)]
        if b:
            conf = sum(c for c, _ in b) / len(b)
            acc = sum(ok for _, ok in b) / len(b)
            ece += len(b) / len(rows) * abs(conf - acc)
            out.append((f"{lo:.1f}-{hi:.1f}", len(b), conf, acc))
    return out, ece


def ch09_name_calibration():
    """Уверенность и точность BookLM на именах героев: корзины и ECE."""
    tok, base = load("base")
    _, inst = load("instruct")
    seen, first = name_cuts(tok)
    for mname, m in (("база", base), ("instruct", inst)):
        rows = []
        for gname, cuts in (("имя уже было", seen), ("имя ещё не звучало", first)):
            r = [name_guess(tok, m, b, n) for b, n in cuts]
            rows += [(c, ok) for c, ok, _ in r]
            print(f"{mname}, {gname}: {len(r)} обрезков, уверенность {sum(c for c, _, _ in r) / len(r):.2f}, "
                  f"точность {sum(ok for _, ok, _ in r) / len(r):.2f}")
        table, ece = calibration_table(rows)
        for b, n, c, a in table:
            print(f"   {b}: {n:3d} обрезков, уверенность {c:.2f}, точность {a:.2f}")
        print(f"   ECE = {ece:.3f}")


EXPERIMENTS = {f.__name__: f for f in (ch09_fairy_exam, ch09_contamination, ch09_name_calibration)}

if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    torch.manual_seed(0)
    run(EXPERIMENTS)
