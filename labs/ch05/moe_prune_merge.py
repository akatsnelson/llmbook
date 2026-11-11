"""Удаление и слияние экспертов в обученной BookLM-MoE16 без дообучения.

Полная версия опыта главы 5 (подраздел 5.4.5 «Удаление и слияние экспертов»). Модели
обучает labs/ch05/moe_train.py: в каждом из шести блоков 16 экспертов SwiGLU по
192 нейрона, маршрутизатор выбирает четыре. ch05-moe16-mix училась на смеси сказок
(3/4 окон) и технических текстов (код репозитория), ch05-moe16 — только на
сказках. Здесь мы:

1. считаем на 40 калибровочных сказках, как часто маршрутизатор выбирает каждого
   эксперта, и оценку REAP (доля маршрутизатора, умноженная на длину выхода);
2. оставляем в каждом слое m экспертов из 16 по частоте, по REAP и случайно
   (пять зёрен) и меряем перплексию на отложенных сказках и, для частоты, на
   отложенных технических текстах, которых в калибровке не было;
3. сливаем самые похожие пары экспертов (простым средним и после сопоставления
   нейронов) и сравниваем с удалением при том же числе экспертов;
4. проверяем, что на токен по-прежнему работают четыре эксперта.

Вывод сохранён в models/booklm/experiments/ch05-moe-prune.txt, кривые для
рисунка — в ch05-moe16-mix-prune.csv и ch05-moe16-prune.csv.

Запуск из корня репозитория: python labs/ch05/moe_prune_merge.py
Время: около 15 минут на процессоре M2 Max.
"""
import copy
import csv
import math
import statistics
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from booklm.experiments.ch05 import stories  # noqa: E402
from booklm.lab import perplexity  # noqa: E402
from booklm.model import BookLM, BookLMConfig, SwiGLU  # noqa: E402

OUT = ROOT / "models" / "booklm" / "experiments"


# ----------------------------------------------------------------------------- слой из книги

def moe_forward(x, router, experts, k):      # x: [N, d]
    top = router(x).topk(k, dim=-1)
    weights = top.values.softmax(dim=-1)
    y = torch.zeros_like(x)
    for e, expert in enumerate(experts):
        rows, slot = (top.indices == e).nonzero(as_tuple=True)
        if rows.numel():
            y[rows] += weights[rows, slot, None] * expert(x[rows])
    return y


class MoEFeedForward(nn.Module):
    def __init__(self, cfg, E=4, k=2):
        super().__init__()
        self.router = nn.Linear(cfg.d_model, E, bias=False)
        self.experts = nn.ModuleList(SwiGLU(cfg) for _ in range(E))
        self.k = k

    def forward(self, x):
        rows = x.reshape(-1, x.size(-1))
        return moe_forward(rows, self.router, self.experts, self.k).view_as(x)


def load_moe16(name="ch05-moe16-mix"):
    """name: ch05-moe16 (учили на сказках) или ch05-moe16-mix (сказки и технические тексты)."""
    cfg = BookLMConfig(d_ff=192)
    model = BookLM(cfg)
    for block in model.blocks:
        block.ffn = MoEFeedForward(cfg, E=16, k=4)
    model.load_state_dict(torch.load(OUT / f"{name}.pt", map_location="cpu"))
    return model.eval()


def tech_test():
    """Отложенные технические тексты (каждый десятый файл кода), см. moe_train.py."""
    import numpy as np
    data = np.fromfile(OUT / "ch05-tech-test.bin", dtype=np.uint16).astype("int64")
    return torch.from_numpy(data)


@torch.no_grad()
def perplexity_on(model, data, n=50_000, T=512):
    """Как booklm.lab.perplexity, но на заданном потоке токенов."""
    data = data[:n]
    total, count = 0.0, 0
    for s in range(0, len(data) - T - 1, T):
        x, y = data[s:s + T][None], data[s + 1:s + T + 1]
        total += F.cross_entropy(model(x)[0].float(), y, reduction="sum").item()
        count += T
    return math.exp(total / count)


# ----------------------------------------------------------------------------- статистика маршрута

@torch.no_grad()
def route_stats(model, texts):
    """Частота выбора и оценка REAP каждого эксперта; входы слоёв для сходства экспертов."""
    inputs = [[] for _ in model.blocks]
    hooks = [b.ffn.register_forward_pre_hook(
        lambda m, inp, i=i: inputs[i].append(inp[0].reshape(-1, inp[0].size(-1))))
        for i, b in enumerate(model.blocks)]
    for t in texts:
        model(t[None])
    for h in hooks:
        h.remove()
    freq, reap, rows_all = [], [], []
    for block, xs in zip(model.blocks, inputs):
        x, layer = torch.cat(xs), block.ffn
        top = layer.router(x).topk(layer.k, dim=-1)
        w = top.values.softmax(-1)
        E = len(layer.experts)
        f = torch.bincount(top.indices.flatten(), minlength=E).float()
        score = torch.zeros(E)
        for e, expert in enumerate(layer.experts):
            rows, slot = (top.indices == e).nonzero(as_tuple=True)
            if rows.numel():                  # REAP: среднее p_e * |F_e(x)| по своим токенам
                score[e] = (w[rows, slot] * expert(x[rows]).norm(dim=-1)).mean()
        freq.append(f / f.sum())
        reap.append(score)
        rows_all.append(x)
    return freq, reap, rows_all


# ----------------------------------------------------------------------------- удаление

def prune(layer, keep):
    """Оставить экспертов keep: строки маршрутизатора и сами эксперты."""
    keep = sorted(keep)
    layer.router.weight = nn.Parameter(layer.router.weight[keep].clone())
    layer.experts = nn.ModuleList(layer.experts[i] for i in keep)


def pruned_copy(model, scores, m):
    small = copy.deepcopy(model)
    for block, s in zip(small.blocks, scores):
        prune(block.ffn, s.topk(m).indices.tolist())
    return small


def coverage(freq, scores, m):
    """Доля калибровочных выборов маршрутизатора, попавших в оставленных экспертов."""
    return statistics.mean(f[s.topk(m).indices].sum().item() for f, s in zip(freq, scores))


# ----------------------------------------------------------------------------- слияние

def expert_matrix(expert):                    # нейрон = строка gate, строка up, столбец down
    return torch.cat([expert.gate.weight, expert.up.weight, expert.down.weight.T], dim=1)


def greedy_match(a, b):
    """perm: нейрону i эксперта a ставит в пару нейрон perm[i] эксперта b (жадно по косинусу)."""
    sim = F.normalize(a, dim=1) @ F.normalize(b, dim=1).T
    perm, used_a, used_b = torch.full((len(a),), -1), set(), set()
    for flat in sim.flatten().argsort(descending=True).tolist():
        i, j = divmod(flat, len(b))
        if i not in used_a and j not in used_b:
            perm[i] = j
            used_a.add(i)
            used_b.add(j)
            if len(used_a) == len(a):
                break
    return perm


@torch.no_grad()
def merge_pair(layer, i, j, fi, fj, align):
    a, b = layer.experts[i], layer.experts[j]
    perm = greedy_match(expert_matrix(a), expert_matrix(b)) if align else torch.arange(
        a.gate.weight.size(0))
    merged = copy.deepcopy(a)
    wi, wj = fi / (fi + fj), fj / (fi + fj)  # средние с весами по частоте, как в MC-SMoE
    merged.gate.weight.copy_(wi * a.gate.weight + wj * b.gate.weight[perm])
    merged.up.weight.copy_(wi * a.up.weight + wj * b.up.weight[perm])
    merged.down.weight.copy_(wi * a.down.weight + wj * b.down.weight[:, perm])
    # обе строки маршрутизатора ведут к одному эксперту: если токен выбрал оба номера,
    # их доли складываются (так слияние M-SMoE и HC-SMoE описывают Ласби и соавторы, REAP)
    layer.experts[i] = merged
    layer.experts[j] = merged


@torch.no_grad()
def output_similarity(layer, x, n=2000):
    """Косинус выходов экспертов на одних и тех же входах слоя: [E, E]."""
    x = x[torch.randperm(len(x), generator=torch.Generator().manual_seed(0))[:n]]
    outs = F.normalize(torch.stack([e(x).flatten() for e in layer.experts]), dim=1)
    return outs @ outs.T


def merged_copy(model, freq, rows, m, align):
    """Сливать непересекающиеся пары самых похожих экспертов, пока в слое не останется m."""
    small = copy.deepcopy(model)
    for block, f, x in zip(small.blocks, freq, rows):
        layer = block.ffn
        sim = output_similarity(layer, x).fill_diagonal_(-2)
        taken = set()
        pairs = 0
        for flat in sim.flatten().argsort(descending=True).tolist():
            i, j = divmod(flat, len(sim))
            if i < j and i not in taken and j not in taken:
                merge_pair(layer, i, j, f[i].item(), f[j].item(), align)
                taken |= {i, j}
                pairs += 1
                if pairs == len(sim) - m:
                    break
    return small


def unique_params(model):                     # общие (слитые) модули считаются один раз
    return sum(p.numel() for p in model.parameters())


def active_params(model):
    """Параметры, которые работают на одном токене: всё, кроме неизбранных экспертов."""
    total = 0
    for name, p in model.named_parameters():
        if ".experts." not in name:
            total += p.numel()
    expert = sum(p.numel() for p in model.blocks[0].ffn.experts[0].parameters())
    return total + sum(b.ffn.k * expert for b in model.blocks)


# ----------------------------------------------------------------------------- опыт

def prune_curve(model, freq, reap, tech, ms=(16, 14, 12, 10, 8, 6, 5, 4)):
    base = perplexity(model)
    table = []
    print(f"{'m':>3}{'параметров':>12}{'покрытие':>10}{'частота':>9}{'REAP':>8}"
          f"{'случайно, 5 зёрен':>22}{'техн. тексты':>14}")
    for m in ms:
        small = pruned_copy(model, freq, m)
        p_freq = perplexity(small)
        p_reap = perplexity(pruned_copy(model, reap, m))
        rand = []
        for seed in range(5):
            g = torch.Generator().manual_seed(seed)
            rnd = [torch.rand(16, generator=g) for _ in freq]
            rand.append(perplexity(pruned_copy(model, rnd, m)))
        p_tech = perplexity_on(small, tech)
        cov = coverage(freq, freq, m)
        row = dict(m=m, keep=m / 16, params=unique_params(small), active=active_params(small),
                   coverage=cov, freq=p_freq, reap=p_reap, rand_mean=statistics.mean(rand),
                   rand_min=min(rand), rand_max=max(rand), tech=p_tech)
        table.append(row)
        print(f"{m:3}{row['params']:12,}{cov:10.3f}{p_freq:9.2f}{p_reap:8.2f}"
              f"{row['rand_mean']:10.2f} ({min(rand):.2f}-{max(rand):.2f}){p_tech:12.1f}")
        if m == 16:
            assert abs(p_freq - base) < 1e-6
        # на токен работают те же 4 эксперта; уходят только строки маршрутизатора
        assert row["active"] == active_params(model) - 6 * (16 - m) * 256
    return table


def main():
    torch.manual_seed(0)
    tech = tech_test()
    calib = stories(40)                       # калибровка: 40 отложенных сказок
    print(f"калибровка: {sum(len(s) for s in calib)} токенов из {len(calib)} сказок")
    results = {}
    for name in ("ch05-moe16-mix", "ch05-moe16"):
        model = load_moe16(name)
        freq, reap, rows = route_stats(model, calib)
        print(f"\n{name}: {unique_params(model):,} параметров, активных {active_params(model):,}, "
              f"перплексия на сказках {perplexity(model):.2f}, на технических текстах "
              f"{perplexity_on(model, tech):.1f}")
        print("доля выборов на калибровке, самый редкий и самый частый эксперт слоя:")
        for i, f in enumerate(freq):
            print(f"  слой {i}: {f.min():.3f} .. {f.max():.3f}")
        table = prune_curve(model, freq, reap, tech)
        results[name] = (model, freq, rows, table)
        with open(OUT / f"{name}-prune.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(table[0]))
            w.writeheader()
            for r in table:
                w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})

    model, freq, rows, _ = results["ch05-moe16-mix"]
    print("\nch05-moe16-mix: удаление и слияние при одинаковом числе экспертов в слое")
    merge_rows = []
    for m in (15, 12, 8):
        r = dict(m=m,
                 prune=perplexity(pruned_copy(model, freq, m)),
                 naive=perplexity(merged_copy(model, freq, rows, m, align=False)),
                 aligned=perplexity(merged_copy(model, freq, rows, m, align=True)))
        merge_rows.append(r)
        print(f"  m={m:2}: удаление {r['prune']:6.2f}, слияние без сопоставления "
              f"{r['naive']:7.2f}, после сопоставления нейронов {r['aligned']:6.2f}")
        assert unique_params(merged_copy(model, freq, rows, m, align=True)) == \
            unique_params(pruned_copy(model, freq, m)) + 6 * (16 - m) * 256
    # то, что сказано в книге: до десяти экспертов сказки почти не страдают, дальше перплексия
    # растёт всё быстрее; REAP точнее частоты, частота намного лучше случайного выбора; код
    # ломается раньше сказок; модель на смеси терпит удаление лучше; слияние проигрывает удалению
    mix = {r["m"]: r for r in results["ch05-moe16-mix"][3]}
    mono = {r["m"]: r for r in results["ch05-moe16"][3]}
    assert mix[10]["freq"] < 1.5 * mix[16]["freq"] < 2 * mix[16]["freq"] < mix[8]["freq"]
    assert mix[4]["freq"] > 25 * mix[16]["freq"]
    assert all(mix[m]["reap"] < mix[m]["freq"] for m in (14, 12, 10, 8, 6, 5, 4))
    assert all(mix[m]["freq"] < mix[m]["rand_min"] for m in (14, 12, 10, 8))
    assert mix[14]["tech"] > 4 * mix[16]["tech"] and mix[10]["tech"] > 40 * mix[16]["tech"]
    assert all(mix[m]["freq"] < mono[m]["freq"] for m in (14, 12, 10, 8))
    for r in merge_rows:
        assert r["prune"] < r["aligned"] < r["naive"]
    return results, merge_rows


if __name__ == "__main__":
    main()
