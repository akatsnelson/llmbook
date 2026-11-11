"""Опыты главы 4 о деталях блока: зачем головы, полносвязная ветвь и остаточный путь.

В книге описаны постановка и итоговые числа; здесь опыты целиком.
Как устроены шаг обучения, Adam и перекрёстная энтропия, разбирает глава 7 «Предобучение».

    accuracy                 — общий цикл: учит маленькую модель с пяти зёрен и проверяет на 10 000 новых примеров;
    facts_batch, facts_model — раздел 4.4 «Multi-head attention»: назвать два слова, одна голова против двух;
    repeat_batch, repeat_model — раздел 4.5 «Feed-forward network»: заметить повтор слова;
    stack, row               — подраздел 4.5.2 «Стопка слоёв» и раздел 4.6 «Residual connection и LayerNorm»:
                               насколько различаются позиции после каждого слоя стопки со случайными весами;
    deep_stack, show         — подразделы 4.6.1 «Затухание и взрыв градиента» и 4.6.2 «LayerNorm»: размах
                               выхода и градиент первого слоя в глубокой стопке полносвязных ветвей.

Запуск из корня репозитория: python labs/ch04/block_parts.py
Считается на CPU около полуминуты.
"""

import math
import statistics

import torch
import torch.nn as nn


# Функции внимания из листингов главы 4.
def attention_scores(q, k):
    return q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))


def attention(q, k, v):
    weights = attention_scores(q, k).softmax(dim=-1)
    return weights @ v, weights


def causal_attention(q, k, v):
    scores = attention_scores(q, k)
    t = q.size(-2)
    future = torch.triu(torch.ones(t, t, dtype=torch.bool, device=q.device), diagonal=1)
    weights = scores.masked_fill(future, float("-inf")).softmax(dim=-1)
    return weights @ v, weights


# --- Общий цикл обучения ---------------------------------------------------------------------
def accuracy(make, batch, steps):
    runs = []
    for seed in range(5):                         # пять стартов
        torch.manual_seed(seed)                   # начальные веса
        gen = torch.Generator().manual_seed(seed) # свой поток для данных
        parts, answer = make()                    # слои и правило ответа
        opt = torch.optim.Adam(nn.ModuleList(parts).parameters(), lr=0.01)
        for _ in range(steps):
            x, ys = batch(128, gen)               # ys: список ответов
            loss = sum(nn.functional.cross_entropy(o, y)
                       for o, y in zip(answer(x), ys))
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():                     # проверка на новых
            x, ys = batch(10_000, gen)
            ok = [o.argmax(dim=-1) == y for o, y in zip(answer(x), ys)]
        runs.append(torch.stack(ok).all(dim=0).float().mean().item())
    return f"{statistics.mean(runs):.2f} ± {statistics.stdev(runs):.2f}"


# --- Одна голова против двух: назвать два слова ------------------------------------------------
def facts_batch(n, gen, words=32, T=8):
    # роли: 0 подлежащее, 1 сосед, 2 прочее слово, 3 вопрос (последний)
    pos = torch.rand(n, T - 1, generator=gen).argsort(dim=-1)  # места
    i, role = torch.arange(n), torch.full((n, T), 2)
    role[i, pos[:, 0]], role[i, pos[:, 1]], role[:, -1] = 0, 1, 3
    word = torch.randint(0, words, (n, T), generator=gen)
    word[:, -1] = 0                                # у вопроса слова нет
    return (role, word), (word[i, pos[:, 0]], word[i, pos[:, 1]])
def facts_model(heads, d=32, words=32):
    roles, vocab = nn.Embedding(4, d), nn.Embedding(words, d)
    qkv = nn.Linear(d, 3 * d, bias=False)          # W_Q, W_K, W_V разом
    out = nn.Linear(d, d, bias=False)              # W_O
    say = [nn.Linear(d, words), nn.Linear(d, words)]  # подлежащее, сосед
    def answer(x):
        h = roles(x[0]) + vocab(x[1])              # роль + слово
        q, k, v = (t.unflatten(-1, (heads, -1)).transpose(1, 2)  # головы
                   for t in qkv(h).chunk(3, dim=-1))
        mixed, _ = attention(q, k, v)              # [B, H, T, d/H]
        o = out(mixed.transpose(1, 2).flatten(2))[:, -1]   # склейка
        return [s(o) for s in say]
    return [roles, vocab, qkv, out, *say], answer


# --- Заметить повтор: одно внимание и внимание с ветвью -----------------------------------------
def repeat_batch(n, gen, words=8):
    a = torch.randint(0, words, (n,), generator=gen)
    same = torch.rand(n, generator=gen) < 0.5      # половина — повторы
    other = (a + torch.randint(1, words, (n,), generator=gen)) % words
    pairs = torch.stack([a, torch.where(same, a, other)], dim=1)
    return pairs, [same.long()]                    # ответ: 1, если повтор

def repeat_model(ffn, words=8, d=16):
    embed = nn.Embedding(words, d)
    qkv = nn.Linear(d, 3 * d, bias=False)
    branch = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(),     # ветвь
                           nn.Linear(4 * d, d)) if ffn else nn.Identity()
    head = nn.Linear(d, 2)                         # «повтор» или «нет»
    def answer(pairs):
        q, k, v = qkv(embed(pairs)).chunk(3, dim=-1)
        mixed, _ = causal_attention(q, k, v)
        return [head(branch(mixed[:, -1]))]        # вторая видит оба
    return [embed, qkv, branch, head], answer


# --- Стопка со случайными весами -------------------------------------------------------------
def stack(seed, ffn=False, residual=False, attn=True, layers=8):
    torch.manual_seed(seed)
    h, parts = torch.randn(6, 16), []              # 6 позиций, ширина 16
    for _ in range(layers):
        if attn:                                   # случайные веса
            w_q, w_k, w_v = (torch.randn(16, 16) / 4 for _ in range(3))
            a, _ = attention(h @ w_q, h @ w_k, h @ w_v)
            h = h + a if residual else a
        if ffn:                                    # ветвь без смещений
            f = nn.Sequential(nn.Linear(16, 64, bias=False), nn.GELU(),
                              nn.Linear(64, 16, bias=False))
            h = h + f(h) if residual else f(h)
        # длина разброса позиций вокруг их среднего и длина всего выхода
        parts.append(((h - h.mean(dim=0)).norm().item(), h.norm().item()))
    return parts


def row(name, how, cell=lambda apart, length: apart / length,
        fmt="9.4f", agg=statistics.mean):        # по 5 зёрнам на слоях
    runs = [stack(seed, **how) for seed in range(5)]   # 1, 2, 4 и 8
    cols = [[cell(*r[n - 1]) for r in runs] for n in (1, 2, 4, 8)]
    print(f"{name:26}" + "".join(f"{agg(c):{fmt}}" for c in cols))

header = " " * 26 + "".join(f"   слой {n}" for n in (1, 2, 4, 8))


# --- Глубокая стопка полносвязных ветвей ------------------------------------------------------
def deep_stack(depth, residual=False, norm=False, he=False, d=64):
    torch.manual_seed(0)
    layers = [nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
              for _ in range(depth)]                 # старт PyTorch
    if he:                                           # старт Хе: крупнее
        for lin in (f[i] for f in layers for i in (0, 2)):
            nn.init.kaiming_normal_(lin.weight)
            nn.init.zeros_(lin.bias)
    h = torch.randn(32, d)                           # 32 позиции
    for f in layers:                                 # norm — чуть ниже
        z = nn.functional.layer_norm(h, (d,)) if norm else h
        h = h + f(z) if residual else f(z)
    h.pow(2).mean().backward()                       # условная ошибка
    grad = layers[0][0].weight.grad.norm().item()    # у первого слоя
    return h.pow(2).mean().sqrt().item(), grad       # размах, градиент

def show(name, **how):
    runs = [deep_stack(depth, **how) for depth in (2, 8, 32, 64)]
    print(f"{name:17}размах  ", "  ".join(f"{r[0]:7.2g}" for r in runs))
    print(f"{'':17}градиент", "  ".join(f"{r[1]:7.1e}" for r in runs))

if __name__ == "__main__":
    print("== Одна голова против двух")
    for heads in (1, 2):
        acc = accuracy(lambda: facts_model(heads), facts_batch, steps=1000)
        print(f"голов {heads} по {32 // heads}: оба слова верны в {acc} фраз")
    print("\n== Заметить повтор")
    for ffn in (False, True):
        acc = accuracy(lambda: repeat_model(ffn), repeat_batch, steps=600)
        name = "внимание и ветвь" if ffn else "одно внимание"
        print(f"{name:16}: верно в {acc} пар")
    print("\n== Различие позиций в стопке")
    print(header)
    row("только ветви", dict(ffn=True, attn=False))
    row("только внимание", {})
    row("внимание и ветвь", dict(ffn=True))
    row("внимание с плюсом", dict(residual=True))
    both = dict(ffn=True, residual=True)
    row("внимание и ветвь с плюсом", both)
    row("  разброс ±", both, fmt="9.2f", agg=statistics.stdev)   # по зёрнам
    print("\n== Числитель и знаменатель меры")
    print(header)
    for name, how in (("только внимание", {}),
                      ("внимание с плюсом", dict(residual=True))):
        print(name)
        row("  разброс позиций", how, lambda apart, length: apart, "9.2f")
        row("  длина выхода", how, lambda apart, length: length, "9.2f")

    print("\n== Глубокая стопка")
    print(f"{'':17}глубина ", "  ".join(f"{n:7d}" for n in (2, 8, 32, 64)))
    show("обычный старт")
    show("старт Хе", he=True)
    show("плюс", residual=True)
    show("плюс, старт Хе", residual=True, he=True)
    show("плюс и норма", residual=True, norm=True)
    show("плюс и норма, Хе", residual=True, norm=True, he=True)

