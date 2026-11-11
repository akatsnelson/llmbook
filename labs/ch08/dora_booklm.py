"""LoRA и DoRA одного ранга на BookLM: ошибка, число параметров, время шага, слияние (глава 8, подраздел 8.2.8
«Адаптер с разложением весов (DoRA)»).

Задача та же, что в подразделе 8.2.7 «Дообучение BookLM с адаптером»:
пары «просьба -> история» из models/booklm/data/sft.jsonl, шаблон и маска ошибки из booklm/sft_lora.py,
адаптер на всех 42 матрицах, r = 16 (alpha = 32, как в sft_lora.py) и r = 4 (alpha = 8). Обе версии учатся одинаково: AdamW 2e-3 без затухания весов,
разогрев 15 шагов и косинусный спад, 150 шагов по 16 пар. Зерно задаёт и случайную A, и порядок пар, так что
при одном зерне LoRA и DoRA видят одни и те же пакеты и стартуют с одной A: сравнение парное.
Ошибка считается на всех 200 отложенных парах (story_loss), время шага — медиана по шагам 20..149.

Запуск из корня репозитория: python labs/ch08/dora_booklm.py
Время: около часа на занятом CPU M2 Max (5 зёрен x 2 способа x 2 ранга).

Печать сохраняется в models/booklm/experiments/ch08-dora-table.txt.
"""
import contextlib
import io
import math
import random
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "booklm"))
from booklm.lab import MODELS, load                                # noqa: E402
from booklm.experiments.ch08 import held_out, sft_train, story_loss  # noqa: E402
from sft_lora import encode                                         # noqa: E402

TYPES = ("attention.q", "attention.k", "attention.v", "attention.out", "ffn.gate", "ffn.up", "ffn.down")
SEEDS, STEPS, BSZ, LR = (0, 1, 2, 3, 4), 150, 16, 2e-3
RANKS = (16, 4)                   # alpha = 2r: масштаб alpha/r = 2, как в sft_lora.py
OUT = MODELS / "experiments" / "ch08-dora-table.txt"


class LoRALinear(nn.Module):
    """y = x W^T + s (x A^T) B^T, как в листинге раздела; W заморожена."""

    def __init__(self, base, r, alpha, seed):
        super().__init__()
        self.base, self.scale = base, alpha / r
        d_out, d_in = base.weight.shape
        g = torch.Generator().manual_seed(seed)
        self.A = nn.Parameter(torch.randn(r, d_in, generator=g) / d_in**0.5)
        self.B = nn.Parameter(torch.zeros(d_out, r))

    def forward(self, x):
        return self.base(x) + self.scale * (x @ self.A.T) @ self.B.T

    def merged_weight(self):
        return self.base.weight + self.scale * self.B @ self.A


class DoRALinear(LoRALinear):
    """y = (m / n) * (x W^T + s (x A^T) B^T), n = ||W + s B A|| по строкам, без градиента (раздел 4.3 статьи).

    Так же считает PEFT (use_dora=True): матрицу W' целиком не собирают, и градиента размером с W нет.
    """

    def __init__(self, base, r, alpha, seed):
        super().__init__(base, r, alpha, seed)
        self.m = nn.Parameter(base.weight.norm(dim=1))          # [d_out]: длины строк W

    def row_norm(self):                                         # [d_out]
        return (self.base.weight + self.scale * self.B @ self.A).norm(dim=1)

    def forward(self, x):
        return (self.m / self.row_norm().detach()) * super().forward(x)

    def merged_weight(self):
        return (self.m / self.row_norm())[:, None] * super().merged_weight()


def wrap(model, cls, r, seed):
    model.requires_grad_(False)
    for i, (name, layer) in enumerate(list(model.named_modules())):
        if name.endswith(TYPES) and isinstance(layer, nn.Linear):
            parent, attr = name.rsplit(".", 1)
            setattr(model.get_submodule(parent), attr, cls(layer, r, 2 * r, seed=1000 * seed + i))
    return model


def batch_of(pairs):
    T = max(len(ids) for ids, _ in pairs) - 1
    x = torch.tensor([ids[:-1] + [0] * (T + 1 - len(ids)) for ids, _ in pairs])
    y = torch.tensor([lab[1:] + [-100] * (T + 1 - len(lab)) for _, lab in pairs])
    return x, y


def train(cls, r, seed, data):
    torch.manual_seed(seed)
    _, model = load("base")
    wrap(model, cls, r, seed)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)
    rng, times = random.Random(seed), []
    model.train()
    for step in range(STEPS):
        lr = LR * min(1.0, (step + 1) / 15) * 0.5 * (1 + math.cos(math.pi * step / STEPS))
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = batch_of(rng.sample(data, BSZ))
        t0 = time.perf_counter()
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten(), ignore_index=-100)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        times.append(time.perf_counter() - t0)
    model.eval()
    return model, sum(p.numel() for p in params), statistics.median(times[20:])


def merged_copy(model):
    """Обычная BookLM, в которую записаны W' всех 42 обёрток: ни A, ни B, ни m в ней нет."""
    _, plain = load("base")
    state = {n.replace(".base.", "."): p for n, p in model.state_dict().items()
             if not n.endswith((".A", ".B", ".m"))}
    for name, layer in model.named_modules():
        if isinstance(layer, LoRALinear):
            state[name + ".weight"] = layer.merged_weight().detach()
    plain.load_state_dict(state)
    return plain.eval()


def m_shift(model):
    """Насколько DoRA сдвинула длины строк: среднее |m - ||W|| | / ||W|| по всем строкам 42 матриц, %."""
    rel = [((l.m - l.base.weight.norm(dim=1)).abs() / l.base.weight.norm(dim=1)).detach()
           for l in model.modules() if isinstance(l, DoRALinear)]
    return 100 * torch.cat(rel).mean().item()


def timed_steps(data, r=16, reps=30):
    """Время шага вперемешку: LoRA, DoRA, LoRA, ... на одном пакете из 16 пар, чтобы фон машины делился поровну."""
    models = {}
    for name, cls in (("LoRA", LoRALinear), ("DoRA", DoRALinear)):
        _, model = load("base")
        models[name] = wrap(model, cls, r, seed=0).train()
    x, y = batch_of(data[:BSZ])
    times = {name: [] for name in models}
    for _ in range(reps):
        for name, model in models.items():
            t0 = time.perf_counter()
            F.cross_entropy(model(x).flatten(0, 1), y.flatten(), ignore_index=-100).backward()
            times[name].append(time.perf_counter() - t0)
            model.zero_grad(set_to_none=True)
    return {name: 1000 * statistics.median(t[5:]) for name, t in times.items()}


def main():
    tok, base = load("base")
    data = [encode(tok, r["request"], r["story"], 512) for r in sft_train()]
    val = [(r["request"], r["story"]) for r in held_out()]
    base_loss = story_loss(base, tok, val)
    lines = [f"база: ошибка на 200 отложенных парах {base_loss:.4f}"]
    res = {}
    for r in RANKS:
        for seed in SEEDS:
            for name, cls in (("LoRA", LoRALinear), ("DoRA", DoRALinear)):
                model, n_params, t_step = train(cls, r, seed, data)
                loss = story_loss(model, tok, val)
                res[name, r, seed] = (loss, n_params, t_step)
                print(f"  r={r} зерно {seed} {name}: ошибка {loss:.4f}, шаг {t_step * 1000:.0f} мс", flush=True)
                if seed == SEEDS[0]:
                    x, _ = batch_of(data[:4])
                    plain = merged_copy(model)
                    with torch.no_grad():
                        gap = (plain(x) - model(x)).abs().max().item()
                    res[name, r, "merge"] = (gap, story_loss(plain, tok, val), loss)
                    if cls is DoRALinear:
                        res[name, r, "m"] = m_shift(model)

    lines.append(f"{'':10s} {'обучаемых':>9s}  " + "  ".join(f"зерно {s}" for s in SEEDS)
                 + "   среднее ± разброс   шаг, мс")
    for r in RANKS:
        for name in ("LoRA", "DoRA"):
            losses = [res[name, r, s][0] for s in SEEDS]
            t_ms = 1000 * statistics.median(res[name, r, s][2] for s in SEEDS)
            lines.append(f"{name + f' r={r}':10s} {res[name, r, 0][1]:9,d}  "
                         + "  ".join(f"{v:7.4f}" for v in losses)
                         + f"   {statistics.mean(losses):.4f} ± {statistics.stdev(losses):.4f}   {t_ms:7.0f}")
        diffs = [res["DoRA", r, s][0] - res["LoRA", r, s][0] for s in SEEDS]
        lines.append(f"DoRA - LoRA, r={r}: " + "  ".join(f"{d:+.4f}" for d in diffs)
                     + f"   среднее {statistics.mean(diffs):+.4f} ± {statistics.stdev(diffs):.4f}")
    t = timed_steps(data)
    lines.append(f"прямой и обратный проход вперемешку, r=16: LoRA {t['LoRA']:.0f} мс, DoRA {t['DoRA']:.0f} мс, "
                 f"DoRA / LoRA = {t['DoRA'] / t['LoRA']:.2f}")
    for r in RANKS:
        lines.append(f"DoRA r={r}, зерно 0: длины строк сдвинулись в среднем на {res['DoRA', r, 'm']:.2f}%")
        for name in ("LoRA", "DoRA"):
            gap, merged_loss, loss = res[name, r, "merge"]
            lines.append(f"слияние {name} r={r} (зерно 0): наибольшее расхождение логитов {gap:.1e}, "
                         f"ошибка {loss:.6f} -> {merged_loss:.6f}")
    text = "\n".join(lines)
    print(text)
    OUT.write_text(text + "\n", encoding="utf-8")

    # Числа, на которые опирается текст раздела
    assert res["LoRA", 16, 0][1] == 454_656 and res["DoRA", 16, 0][1] == 454_656 + 6 * 2432
    assert all(res[n, r, s][0] < base_loss - 0.05 for n in ("LoRA", "DoRA") for r in RANKS for s in SEEDS)
    for r in RANKS:
        for name in ("LoRA", "DoRA"):
            gap, merged_loss, loss = res[name, r, "merge"]
            assert gap < 1e-4 and abs(merged_loss - loss) < 1e-5     # слитая = адаптерная до 6 знака
    book = {("LoRA", 16): 1.5947, ("DoRA", 16): 1.5955, ("LoRA", 4): 1.5763, ("DoRA", 4): 1.5770}
    for (name, r), value in book.items():                            # таблица LoRA и DoRA в подразделе 8.2.8
        assert abs(statistics.mean(res[name, r, s][0] for s in SEEDS) - value) < 3e-4
    assert all(res["DoRA", r, s][0] > res["LoRA", r, s][0] for r in RANKS for s in SEEDS)  # на всех 10 парах
    assert 0.5 < res["DoRA", 16, "m"] < 2.0 and 0.5 < res["DoRA", 4, "m"] < 2.0          # m сдвинулся на ~1 %


if __name__ == "__main__":
    main()
