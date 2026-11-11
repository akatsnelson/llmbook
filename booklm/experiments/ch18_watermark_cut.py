"""Глава 18 «Безопасность LLM-приложений»: водяной знак в выборке BookLM и вырезание направления из весов.

Опыты (время — на процессоре ноутбука Apple M2 Max):
  ch18_watermark      рубрика «водяной знак», подраздел «Водяные знаки в тексте»; около 2 мин.
                      Водяной знак «зелёного списка» (Kirchenbauer и др., ICML 2023, arXiv:2301.10226): ключ
                      и предыдущий токен задают зелёную половину словаря, к логитам зелёных токенов прибавляется
                      delta; детектор считает долю зелёных и z-оценку. Печатает z по длине текста со знаком
                      и без, ошибку на токен, z после переписывания части токенов и долю зелёных.
                      Ещё пишет ch18-watermark.csv: средний z по длине для рисунка.
  ch18_direction_cut  рубрика «вырезание направления», подраздел «Выпуск открытых весов»; несколько секунд.
                      Как abliteration у Arditi и др. (arXiv:2406.11717): направление «девочка минус мальчик»
                      после первого блока убирается из всех матриц, которые пишут в остаточный поток.
                      Печатает долю «она» после имён мальчиков и девочек до и после вырезания и пример.

Запуск из корня репозитория:
    python -m booklm.experiments.ch18_watermark_cut                   # оба опыта
    python -m booklm.experiments.ch18_watermark_cut ch18_watermark    # один опыт
Вывод каждого опыта копируется в models/booklm/experiments/<имя опыта через дефис>.txt.
"""
import contextlib
import copy
import csv
import math
import sys
from pathlib import Path

import numpy as np
import torch

if not __package__:  # запуск файлом (python booklm/experiments/ch18_watermark_cut.py): корень репозитория — в путь поиска
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from booklm.experiments.ch10_representations import BOYS, GIRLS
from booklm.experiments.ch18 import val_stories
from booklm.lab import DEVICE, MODELS, generate, load, perplexity, prompt_ids, special

OUT = MODELS / "experiments"
KEY = 20261029            # секретный ключ владельца модели
GAMMA = 0.5               # доля зелёных токенов в словаре


@contextlib.contextmanager
def one_thread():
    """Генерация по одному токену быстрее и повторяемее в один поток."""
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(old)


# ----------------------------------------------------------------------------- ch18-watermark

def green_list(prev, key=KEY, gamma=GAMMA, vocab=8192):
    """Зелёная часть словаря: перестановка номеров, которую задают ключ и предыдущий токен."""
    g = torch.Generator().manual_seed(key * 100_003 + prev)
    mask = torch.zeros(vocab, dtype=torch.bool)
    mask[torch.randperm(vocab, generator=g)[:int(gamma * vocab)]] = True
    return mask


def z_score(ids, key=KEY, gamma=GAMMA):
    """ids: токены текста вместе с одним предыдущим. Возвращает z-оценку доли зелёных."""
    green = sum(green_list(a, key, gamma)[b].item() for a, b in zip(ids, ids[1:]))
    T = len(ids) - 1
    return (green - gamma * T) / math.sqrt(T * gamma * (1 - gamma))


def green_share(ids, key=KEY, gamma=GAMMA):
    return sum(green_list(a, key, gamma)[b].item() for a, b in zip(ids, ids[1:])) / (len(ids) - 1)


@torch.no_grad()
def generate_marked(tok, model, text="Однажды", delta=2.0, max_new=200, seed=0, key=KEY, gamma=GAMMA,
                    temperature=0.8, top_k=40, rewrite=None, p_rewrite=0.0):
    """Выборка с водяным знаком; delta=0 — обычная выборка, как generate. Конец истории запрещён, чтобы
    все тексты были одной длины. rewrite: готовые токены, которые мы «переписываем» своей моделью без
    знака: на каждом месте с вероятностью p_rewrite токен выбирается заново, иначе остаётся прежним.
    Возвращает токены вместе с последним токеном затравки."""
    gen = torch.Generator().manual_seed(seed)
    ids = prompt_ids(tok, text)
    caches = [{"len": 0} for _ in model.blocks]
    logits = model(torch.tensor([ids], device=DEVICE), caches)[0, -1].float()
    for c in caches:
        c["len"] = len(ids)
    out = [ids[-1]]
    for step in range(max_new):
        last = logits / temperature
        last[special(tok)["<eos>"]] = float("-inf")
        if rewrite is None and delta:
            last = last + delta * green_list(out[-1], key, gamma)      # вся суть водяного знака
        last = last.masked_fill(last < last.topk(top_k).values[-1], float("-inf"))
        nxt = torch.multinomial(last.softmax(-1), 1, generator=gen).item()
        if rewrite is not None and torch.rand(1, generator=gen).item() >= p_rewrite:
            nxt = rewrite[step + 1]
        out.append(nxt)
        logits = model(torch.tensor([[nxt]], device=DEVICE), caches)[0, -1].float()
        for c in caches:
            c["len"] += 1
    return out


def token_texts(tok, model, n=100, length=200):
    """Три набора по n текстов длиной length токенов (с одним токеном перед текстом):
    BookLM без знака, BookLM со знаком (delta=2) и начала отложенных историй, написанных не BookLM."""
    with one_thread():
        plain = [generate_marked(tok, model, delta=0, seed=s, max_new=length) for s in range(n)]
        marked = [generate_marked(tok, model, delta=2.0, seed=s, max_new=length) for s in range(n)]
    human = [[special(tok)["<bos>"]] + tok.encode(s).ids for s in val_stories(tok)]
    human = [h[:length + 1] for h in human if len(h) > length][:n]
    return plain, marked, human


def rewrite_texts(tok, model, texts, p):
    """Переписать долю p токенов каждого текста той же моделью без знака."""
    with one_thread():
        return [generate_marked(tok, model, seed=1000 + s, max_new=len(t) - 1, rewrite=t, p_rewrite=p)
                for s, t in enumerate(texts)]


@torch.no_grad()
def mean_loss(model, texts):
    """Средняя ошибка BookLM на токен по текстам: насколько текст «естествен» для модели."""
    losses = []
    for ids in texts:
        x = torch.tensor([ids], device=DEVICE)
        losses.append(torch.nn.functional.cross_entropy(model(x)[0, :-1].float(), x[0, 1:]).item())
    return float(np.mean(losses))


def ch18_watermark():
    """Рубрика «водяной знак»: зелёный список, z-оценка и переписывание."""
    tok, model = load("base")
    plain, marked, human = token_texts(tok, model)
    print("со знаком:", tok.decode(marked[0][1:31]).strip())
    rows = [["T", "plain", "human", "marked"]]
    for T in (25, 50, 100, 200):
        z = [np.array([z_score(t[:T + 1]) for t in texts]) for texts in (plain, human, marked)]
        rows.append([T] + [f"{a.mean():.3f}" for a in z])
        print(f"T={T:3d}: со знаком z {z[2].mean():4.2f}, z>4 у {np.mean(z[2] > 4):4.0%} | "
              f"z>4 без знака {np.mean(z[0] > 4):.0%}, у людей {np.mean(z[1] > 4):.0%}")
    print(f"ошибка на токен: без знака {mean_loss(model, plain):.2f}, "
          f"со знаком {mean_loss(model, marked):.2f}")
    for p in (0.25, 0.5):
        z = np.array([z_score(t) for t in rewrite_texts(tok, model, marked, p)])
        print(f"переписано {p:.0%} токенов: z {z.mean():4.2f}, z>4 у {np.mean(z > 4):.0%}")
    print(f"доля зелёных на 200 токенах: {np.mean([green_share(t) for t in marked]):.3f}")
    with open(OUT / "ch18-watermark.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)


# ----------------------------------------------------------------------------- ch18-direction-cut

GIRL = "Жила-была девочка по имени {}. Однажды"
BOY = "Жил-был мальчик по имени {}. Однажды"


@torch.no_grad()
def after_block1(tok, model, text):
    """Средний по позициям вектор остаточного потока после первого блока."""
    h = model.embed(torch.tensor([prompt_ids(tok, text)]))
    return model.blocks[0](h, model.rope_cos, model.rope_sin)[0][0].mean(0)


@torch.no_grad()
def she_share(tok, model, names, template):
    """Средняя доля « она» среди « она» и « он» в следующем токене."""
    she, he = tok.encode(" она").ids[0], tok.encode(" он").ids[0]
    p = torch.stack([model(torch.tensor([prompt_ids(tok, template.format(n))]))[0, -1].softmax(-1)
                     for n in names])
    return (p[:, she] / (p[:, she] + p[:, he])).mean().item()


@torch.no_grad()
def cut_direction(model, r):
    """Копия модели, в которой ни одна матрица не пишет в остаточный поток вдоль единичного r."""
    P = torch.eye(r.numel()) - torch.outer(r, r)
    cut = copy.deepcopy(model)
    cut.embed.weight.copy_(cut.embed.weight @ P)
    for b in cut.blocks:
        b.attention.out.weight.copy_(P @ b.attention.out.weight)
        b.ffn.down.weight.copy_(P @ b.ffn.down.weight)
    return cut


def ch18_direction_cut():
    """Рубрика «вырезание направления»: «она» после имён до и после вырезания."""
    tok, model = load("base")
    v = (torch.stack([after_block1(tok, model, GIRL.format(n)) for n in GIRLS[:10]]).mean(0)
         - torch.stack([after_block1(tok, model, BOY.format(n)) for n in BOYS[:10]]).mean(0))
    cut = cut_direction(model, v / v.norm())
    for name, m in (("исходная", model), ("без направления", cut)):
        print(f"{name:15} «она»: мальчики {she_share(tok, m, BOYS[10:], BOY):.3f}, "
              f"девочки {she_share(tok, m, GIRLS[10:], GIRL):.3f} | перплексия {perplexity(m, n=20_000):.2f}")
    print(generate(tok, cut, BOY.format("Тим"), max_new=14, temperature=0))


EXPERIMENTS = {"ch18_watermark": ch18_watermark, "ch18_direction_cut": ch18_direction_cut}
OUT_ID = {"ch18_watermark": "ch18-watermark", "ch18_direction_cut": "ch18-direction-cut"}

if __name__ == "__main__":
    from booklm.experiments import run
    run(EXPERIMENTS, save={name: out_id + ".txt" for name, out_id in OUT_ID.items()})
