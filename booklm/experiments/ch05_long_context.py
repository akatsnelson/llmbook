"""Глава 5 на обученной BookLM: контекст длиннее обучающего окна и внимание по k ключам, без дообучения
модели из книги (короткое дообучение внутри опыта идёт на копии).

Опыты (время — на процессоре Apple M2 Max):
  ch05_long_context — раздел «Паспорт модели», подраздел «Контекст длиннее обучающего окна», рубрика
      «А теперь на нашей модели: интерполяция позиций и YaRN». Перплексия на окне 1024 по половинам
      окна: как есть, с интерполяцией позиций, NTK, YaRN и после 60 шагов дообучения на окнах 1024.
      Около 2 минут. Времена дообучения в выводе зависят от машины.
  ch05_topk_attention — раздел «Скользящее окно и разреженное внимание», рубрика «А теперь на нашей
      модели: разреженное внимание». Перплексия, если каждый запрос смотрит только на k ключей
      с наибольшими оценками, против окна из k последних ключей. Около 50 с.
Каждый опыт печатает вывод и сохраняет копию в models/booklm/experiments/<имя опыта>.txt
(ch05-long-context.txt, ch05-topk-attention.txt); таблицы для рисунков — в одноимённых .csv.

Запуск из корня репозитория:
    python -m booklm.experiments.ch05_long_context                       # все опыты
    python -m booklm.experiments.ch05_long_context ch05_topk_attention   # один опыт
    python -m booklm.experiments.ch05_long_context --help                # список опытов
"""
import math
import sys
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F

if not __package__:  # запуск файлом (python booklm/experiments/ch05_long_context.py): корень репозитория — в путь поиска
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from booklm.experiments import run
from booklm.lab import MODELS, load, perplexity, val_tokens
from booklm.model import apply_rope

OUT = MODELS / "experiments"


# ----------------------------------------------------------------------------- 5.2: позиции за окном

def rope_angles(T, d_h=32, base=10000.0, scale=1.0, yarn=None):
    """Углы RoPE [T, d_h/2]. scale — делитель позиций (интерполяция позиций).
    yarn=(s, L): деление частот по YaRN с множителем s и исходным окном L."""
    theta = base ** (-torch.arange(0, d_h, 2).float() / d_h)        # [d_h/2]
    if yarn is not None:
        s, L = yarn
        turns = L / (2 * math.pi / theta)        # сколько оборотов делает пара за окно L
        keep = ((turns - 1) / (32 - 1)).clamp(0, 1)   # 0: делим частоту на s, 1: не трогаем
        theta = theta / s * (1 - keep) + theta * keep
    pos = torch.arange(T).float() / scale
    return pos[:, None] * theta[None, :]


def set_rope(model, angles, temperature=1.0):
    """Подменить таблицы RoPE модели; temperature > 1 умножает и q, и k (YaRN)."""
    model.rope_cos = angles.cos() * temperature
    model.rope_sin = angles.sin() * temperature


@torch.no_grad()
def position_losses(model, T=1024, n=49_152):
    """Средняя ошибка на каждой позиции окна T по отложенным историям: [T]."""
    data = val_tokens(n + 1)
    total, count = torch.zeros(T), 0
    for s in range(0, n - T, T):
        x, y = data[s:s + T][None], data[s + 1:s + T + 1]
        total += F.cross_entropy(model(x)[0].float(), y, reduction="none")
        count += 1
    return total / count


@torch.no_grad()
def short_halves(model, n=49_152):
    """Перплексия тех же токенов, если каждую половину окна 1024 читать отдельным окном 512."""
    data = val_tokens(n + 1)
    sums = [0.0, 0.0]
    count = 0
    for s in range(0, n - 1024, 1024):
        for h in (0, 1):
            a = s + 512 * h
            x, y = data[a:a + 512][None], data[a + 1:a + 513]
            sums[h] += F.cross_entropy(model(x)[0].float(), y).item()
        count += 1
    return [math.exp(v / count) for v in sums]


def train_windows(model, steps=60, T=1024, batch=8, lr=3e-4, seed=0):
    """Короткое дообучение на окнах длины T из обучающих историй (CPU, минуты)."""
    import numpy as np
    data = torch.from_numpy(np.fromfile(MODELS / "data" / "train.bin", dtype=np.uint16).astype("int64"))
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    model.train()
    for step in range(steps):
        starts = torch.randint(0, len(data) - T - 1, (batch,), generator=g)
        x = torch.stack([data[a:a + T] for a in starts])
        y = torch.stack([data[a + 1:a + T + 1] for a in starts])
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    return model


def ch05_long_context():
    """Перплексия BookLM на окне 1024: интерполяция позиций, NTK, YaRN и короткое дообучение."""
    tok, model = load("base")
    print(f"обучающее окно 512, читаем окна по 1024 токена; отрезков {len(range(0, 49_152 - 1024, 1024))}")
    s = 2.0
    yarn_m = 0.1 * math.log(s) + 1            # множитель q и k из статьи YaRN: sqrt(1/t)
    ntk_base = 10000 * s ** (32 / 30)          # NTK: основание растёт в s^(d/(d-2)) раз
    schemes = {
        "как есть": dict(angles=rope_angles(1024)),
        "интерполяция /2": dict(angles=rope_angles(1024, scale=s)),
        f"NTK, база {ntk_base:,.0f}": dict(angles=rope_angles(1024, base=ntk_base)),
        "YaRN, s=2": dict(angles=rope_angles(1024, yarn=(s, 512)), temperature=yarn_m),
    }
    curves = {}
    # эталон: те же токены, нарезанные окнами по 512, как модель училась:
    # вторая половина окна 1024 читается с нуля, без первой половины перед ней
    half = short_halves(model)
    print(f"{'схема':22}{'0-511':>8}{'512-1023':>10}")
    print(f"{'окна по 512 (эталон)':22}{half[0]:8.2f}{half[1]:10.2f}")
    for name, kw in schemes.items():
        set_rope(model, kw["angles"], kw.get("temperature", 1.0))
        losses = position_losses(model)
        curves[name] = losses
        print(f"{name:22}{math.exp(losses[:512].mean()):8.2f}{math.exp(losses[512:].mean()):10.2f}")
    # отдельный этап: 60 шагов дообучения на окнах 1024 поверх интерполяции и YaRN
    import time
    for name, kw in (("интерполяция + 60 шагов", schemes["интерполяция /2"]),
                     ("YaRN + 60 шагов", schemes["YaRN, s=2"])):
        tok, model = load("base")
        set_rope(model, kw["angles"], kw.get("temperature", 1.0))
        t0 = time.time()
        train_windows(model)
        losses = position_losses(model)
        curves[name] = losses
        print(f"{name:22}{math.exp(losses[:512].mean()):8.2f}{math.exp(losses[512:].mean()):10.2f}"
              f"   ({time.time() - t0:.0f} с на CPU)")
    with open(OUT / "ch05-long-context.csv", "w") as f:
        f.write("pos,asis,pi,ntk,yarn,pi_ft,yarn_ft\n")
        for b in range(0, 1024, 32):
            row = [math.exp(curves[n][b:b + 32].mean()) for n in curves]
            f.write(f"{b + 16}," + ",".join(f"{v:.3f}" for v in row) + "\n")


# ----------------------------------------------------------------------------- 5.3: top-k ключей

def topk_forward(self, x, cos, sin, cache=None, return_weights=False, k=16, shared=False, stats=None):
    """GQASelfAttention.forward, где каждый запрос смотрит только на k ключей с наибольшими оценками.
    shared=True: один набор ключей на запрос для всех голов слоя (по сумме весов голов, как цель
    индексатора DSA)."""
    b, t, d = x.shape
    q = self.q(x).view(b, t, self.heads, self.d_head).transpose(1, 2)
    kk = self.k(x).view(b, t, self.kv_heads, self.d_head).transpose(1, 2)
    v = self.v(x).view(b, t, self.kv_heads, self.d_head).transpose(1, 2)
    q, kk = apply_rope(q, cos, sin), apply_rope(kk, cos, sin)
    group = self.heads // self.kv_heads
    kk, v = kk.repeat_interleave(group, dim=1), v.repeat_interleave(group, dim=1)
    scores = q @ kk.transpose(-2, -1) / math.sqrt(self.d_head)       # [B, H, T, T]
    causal = torch.ones(t, t, dtype=torch.bool).tril()
    scores = scores.masked_fill(~causal, float("-inf"))
    full = scores.softmax(-1)
    rank = full.sum(1, keepdim=True) if shared else scores           # по чему выбираем ключи
    idx = rank.topk(min(k, t), dim=-1).indices
    keep = torch.zeros_like(rank, dtype=torch.bool).scatter_(-1, idx, True) & causal
    keep = keep.expand_as(scores)
    if stats is not None:
        stats.append((full * keep).sum(-1).mean().item())             # доля веса в выбранных
    w = scores.masked_fill(~keep, float("-inf")).softmax(-1)
    y = w @ v
    return self.out(y.transpose(1, 2).reshape(b, t, d)), None


def set_topk(model, k, shared=False, stats=None):
    for block in model.blocks:
        block.attention.forward = partial(topk_forward, block.attention, k=k, shared=shared,
                                          stats=stats)


def ch05_topk_attention():
    """Перплексия BookLM, когда запрос смотрит только на k ключей с наибольшими оценками."""
    from booklm.experiments.ch05 import set_windows
    tok, model = load("base")
    print(f"полное внимание: перплексия {perplexity(model):.2f}")
    print(f"{'k':>3}{'окно k':>8}{'top-k':>7}{'вес':>6}{'общий top-k':>13}{'вес':>6}")
    rows = []
    for k in (8, 16, 32, 64):
        set_windows(model, [k] * 6)                  # k ближайших ключей
        row = [perplexity(model)]
        for shared in (False, True):                 # свой выбор у головы / общий
            mass = []
            set_topk(model, k, shared, mass)
            row += [perplexity(model), sum(mass) / len(mass)]
        rows.append((k, *row))
        print(f"{k:3}{row[0]:8.2f}{row[1]:7.2f}{row[2]:6.2f}{row[3]:13.2f}{row[4]:6.2f}")
    with open(OUT / "ch05-topk-attention.csv", "w") as f:
        f.write("k,ppl_window,ppl_head,mass_head,ppl_shared,mass_shared\n")
        for r in rows:
            f.write(",".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in r) + "\n")


EXPERIMENTS = {f.__name__: f for f in (ch05_long_context, ch05_topk_attention)}


if __name__ == "__main__":
    run(EXPERIMENTS, save=True)      # копия вывода в models/booklm/experiments/<имя опыта>.txt
