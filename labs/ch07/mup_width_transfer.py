"""Глава 7, подраздел 7.6.9 «Перенос гиперпараметров по ширине».

Опыт: сдвигается ли лучшая скорость обучения AdamW, когда модель становится шире, и держит ли её на
месте параметризация μP. Модель устроена как BookLM (RMSNorm, внимание с RoPE, SwiGLU, словарь 8192),
но в ней два блока, а выходная матрица отделена от таблицы токенов: в μP у входа и выхода разные правила.
Ширины d = 64, 128, 256, 512; ширина головы постоянна, 32, растёт число голов.

Две параметризации, база d0 = 64 (на ней они совпадают):
  стандартная (SP): все матрицы N(0; 0,02^2), одна скорость eta на все параметры;
  μP (Yang и др., 2022, табл. 8): матрицы блоков N(0; 0,02^2 * d0/d) и скорость eta * d0/d,
      таблица токенов и выход N(0; 0,02^2) со скоростью eta, логиты умножаются на d0/d,
      усиления RMSNorm со скоростью eta.

Каждая точка сетки: 300 шагов AdamW (beta = 0,9 и 0,95, без затухания, без ограничения нормы), пакет
16 окон по 128 токенов из models/booklm/data/train.bin, разогрев 30 шагов и косинус до 10 %; ошибка
меряется на 64 фиксированных окнах val.bin. Скорости eta = 2^-12 ... 2^-5, зёрна 0, 1, 2.

Полная сетка: 2 x 4 x 8 x 3 = 192 прогона, около полутора часов на 6 процессах CPU (зависит от загрузки машины). Результаты пишутся в
labs/ch07/mup_width_transfer.json; повторный запуск берёт готовые точки из файла и только печатает
таблицу и проверяет утверждения книги.

Запуск из корня репозитория: python labs/ch07/mup_width_transfer.py
Число параллельных процессов задаёт ключ --workers (по умолчанию 6).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "models" / "booklm" / "data"
OUT = Path(__file__).with_suffix(".json")

V, T, B, LAYERS, D_HEAD, D0 = 8192, 128, 16, 2, 32, 64
STEPS, WARMUP = 300, 30
WIDTHS = (64, 128, 256, 512)
LOG2_LRS = tuple(range(-12, -4))           # eta = 2^-12 ... 2^-5
SEEDS = (0, 1, 2)
STD = 0.02


def rope(x, cos, sin):                     # x: [B, H, T, d_h]
    x0, x1 = x[..., 0::2], x[..., 1::2]
    return torch.stack((x0 * cos - x1 * sin, x0 * sin + x1 * cos), -1).flatten(-2)


class Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.h = d // D_HEAD
        self.norm_1, self.norm_2 = nn.RMSNorm(d, eps=1e-6), nn.RMSNorm(d, eps=1e-6)
        self.q, self.k, self.v, self.o = (nn.Linear(d, d, bias=False) for _ in range(4))
        self.gate, self.up = nn.Linear(d, 3 * d, bias=False), nn.Linear(d, 3 * d, bias=False)
        self.down = nn.Linear(3 * d, d, bias=False)

    def forward(self, x, cos, sin):
        b, t, d = x.shape
        z = self.norm_1(x)
        q, k, v = (f(z).view(b, t, self.h, D_HEAD).transpose(1, 2) for f in (self.q, self.k, self.v))
        y = F.scaled_dot_product_attention(rope(q, cos, sin), rope(k, cos, sin), v, is_causal=True)
        x = x + self.o(y.transpose(1, 2).reshape(b, t, d))
        z = self.norm_2(x)
        return x + self.down(F.silu(self.gate(z)) * self.up(z))


class WideLM(nn.Module):
    def __init__(self, d, mup):
        super().__init__()
        m = d / D0                                      # во сколько раз шире базы
        self.embed = nn.Embedding(V, d)
        self.blocks = nn.ModuleList(Block(d) for _ in range(LAYERS))
        self.norm = nn.RMSNorm(d, eps=1e-6)
        self.head = nn.Linear(d, V, bias=False)         # выход отдельно от таблицы
        self.out_mult = 1 / m if mup else 1.0           # множитель логитов
        for name, p in self.named_parameters():
            if p.dim() == 2:
                hidden = name.startswith("blocks.")
                std = STD / math.sqrt(m) if (mup and hidden) else STD
                nn.init.normal_(p, std=std)
        theta = 10000.0 ** (-torch.arange(0, D_HEAD, 2) / D_HEAD)
        ang = torch.arange(T)[:, None] * theta[None]
        self.register_buffer("cos", ang.cos(), persistent=False)
        self.register_buffer("sin", ang.sin(), persistent=False)

    def forward(self, ids):
        h = self.embed(ids)
        for blk in self.blocks:
            h = blk(h, self.cos, self.sin)
        return self.head(self.norm(h)) * self.out_mult


def param_groups(model, d, lr, mup):
    m = d / D0
    hidden = [p for n, p in model.named_parameters() if n.startswith("blocks.") and p.dim() == 2]
    other = [p for n, p in model.named_parameters() if not (n.startswith("blocks.") and p.dim() == 2)]
    return [{"params": hidden, "lr": lr / m if mup else lr, "base_lr": lr / m if mup else lr},
            {"params": other, "lr": lr, "base_lr": lr}]


def lr_mult(s):                                     # s = 0 ... STEPS-1
    if s < WARMUP:
        return (s + 1) / WARMUP
    p = (s - WARMUP) / (STEPS - WARMUP)
    return 0.1 + 0.45 * (1 + math.cos(math.pi * p))


def windows(data, n, gen):
    ix = torch.randint(len(data) - T - 1, (n,), generator=gen).tolist()
    x = torch.stack([torch.from_numpy(data[i:i + T].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + T + 1].astype(np.int64)) for i in ix])
    return x, y


def run(key):
    kind, d, log2_lr, seed = key
    torch.set_num_threads(int(os.environ.get("MUP_THREADS", "2")))
    train = np.fromfile(DATA / "train.bin", dtype=np.uint16)
    val = np.fromfile(DATA / "val.bin", dtype=np.uint16)
    vx, vy = windows(val, 64, torch.Generator().manual_seed(123))   # одни окна для всех
    torch.manual_seed(seed)
    mup = kind == "mup"
    model = WideLM(d, mup)
    opt = torch.optim.AdamW(param_groups(model, d, 2.0 ** log2_lr, mup),
                            betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
    gen = torch.Generator().manual_seed(1000 + seed)                 # пакеты зависят от зерна
    for s in range(STEPS):
        for g in opt.param_groups:
            g["lr"] = g["base_lr"] * lr_mult(s)
        x, y = windows(train, B, gen)
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
        if not torch.isfinite(loss):
            return key, float("nan")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        model.eval()
        v = sum(F.cross_entropy(model(vx[i:i + 16]).flatten(0, 1), vy[i:i + 16].flatten()).item()
                for i in range(0, 64, 16)) / 4
    return key, v


def load_results():
    if OUT.exists():
        return {tuple(json.loads(k)): v for k, v in json.loads(OUT.read_text()).items()}
    return {}


def save_results(res):
    OUT.write_text(json.dumps({json.dumps(list(k)): v for k, v in sorted(res.items())}, indent=0))


def table(res):
    """Средняя по зёрнам ошибка и лучшая скорость для каждой ширины."""
    out = {}
    for kind in ("sp", "mup"):
        for d in WIDTHS:
            row = {}
            for a in LOG2_LRS:
                vals = [res[(kind, d, a, s)] for s in SEEDS]
                row[a] = (float(np.mean(vals)), float(np.std(vals, ddof=1)))   # среднее и разброс по зёрнам
            out[kind, d] = row
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    res = load_results()
    todo = [(k, d, a, s) for s in SEEDS for k in ("sp", "mup") for d in WIDTHS[::-1] for a in LOG2_LRS
            if (k, d, a, s) not in res]
    if todo:
        with ProcessPoolExecutor(args.workers) as ex:
            for key, v in ex.map(run, todo):
                res[key] = v
                save_results(res)
                print("готово", key, f"{v:.4f}", flush=True)
    tab = table(res)
    best = {}
    for kind, name in (("sp", "стандартная"), ("mup", "μP")):
        print(name)
        for d in WIDTHS:
            row = tab[kind, d]
            a = min(row, key=lambda a: row[a][0] if row[a][0] == row[a][0] else 1e9)
            best[kind, d] = a
            cells = " ".join(f"{row[x][0]:5.2f}" if row[x][0] == row[x][0] else "  nan" for x in LOG2_LRS)
            print(f"  d={d:3d}: {cells} | лучшая 2^{a}, {row[a][0]:.3f} ± {row[a][1]:.3f}")
    # Цена переноса: лучшая скорость базовой ширины d0 = 64, поставленная модели шириной 512.
    cost = {}
    for kind, name in (("sp", "стандартная"), ("mup", "μP")):
        a0, a1 = best[kind, 64], best[kind, 512]
        cost[kind] = tab[kind, 512][a0][0] - tab[kind, 512][a1][0]
        print(f"{name}: скорость 2^{a0} с d=64 на d=512 -> {tab[kind, 512][a0][0]:.3f}, "
              f"лучшая 2^{a1} -> {tab[kind, 512][a1][0]:.3f}, проигрыш {cost[kind]:.3f}")

    # Утверждения раздела книги.
    for a in LOG2_LRS:                       # на базовой ширине параметризации совпадают
        for s in SEEDS:
            assert abs(res["sp", 64, a, s] - res["mup", 64, a, s]) < 1e-6
    assert best["sp", 64] - best["sp", 512] >= 2                 # SP: оптимум уехал вчетверо и больше
    assert all(abs(best["mup", d] - best["mup", 64]) <= 1 for d in WIDTHS)   # μP: не дальше шага сетки
    assert cost["sp"] > 0.5 and cost["mup"] < 0.15
    best_loss = {k: tab[k, 512][best[k, 512]][0] for k in ("sp", "mup")}
    assert best_loss["mup"] < best_loss["sp"]                     # широкая модель в μP лучше
    mup_best = [tab["mup", d][best["mup", d]][0] for d in WIDTHS]
    assert all(x > y for x, y in zip(mup_best, mup_best[1:]))     # в μP шире — лучше
    return tab, best


if __name__ == "__main__":
    main()
