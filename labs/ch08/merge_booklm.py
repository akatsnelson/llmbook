"""Слияние моделей на BookLM: арифметика задач, TIES и DARE на двух адаптерах LoRA
(глава 8, раздел 8.7 «Слияние моделей»).

Две задачи на одних и тех же парах из models/booklm/data/sft.jsonl, обе в формате диалога:
  A «сказка»:  <|user|>просьба<|assistant|>история<eos>  — это готовый адаптер booklm-instruct-lora.pt;
  B «просьба»: <|user|>история<|assistant|>просьба<eos>  — обратная задача: по сказке написать просьбу.
Адаптер B и адаптер «обе задачи сразу» (A и B вперемешку, вдвое больше шагов) обучаются здесь на CPU тем же
рецептом, что sft_lora.py: r = 16, alpha = 32, AdamW 2e-3 без затухания весов, разогрев 50 шагов, косинус,
пакеты по 32 пары, две эпохи. Затем веса сливаются разными способами и меряются три числа:
ошибка на 200 отложенных парах задачи A, то же для задачи B и перплексия на отложенных историях (perplexity()).

Запуск из корня репозитория: python labs/ch08/merge_booklm.py
Без аргумента делается всё (то же, что аргумент all): около 35 минут на свободном CPU M2 Max.
Части можно запускать по отдельности:
    python labs/ch08/merge_booklm.py train     # обучить B и A+B, около 30 минут
    python labs/ch08/merge_booklm.py table     # таблица слияний, около 4 минут

Адаптеры: models/booklm/experiments/ch08-merge-request-lora.pt и ch08-merge-multitask-lora.pt,
печать: models/booklm/experiments/ch08-merge-table.txt.
"""
import contextlib
import io
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "booklm"))
from booklm.lab import MODELS, generate, load, perplexity            # noqa: E402
from booklm.experiments.ch08 import held_out, sft_train, story_loss  # noqa: E402
from sft_lora import add_lora, encode                                 # noqa: E402

OUT = MODELS / "experiments"
TYPES = ("attention.q", "attention.k", "attention.v", "attention.out", "ffn.gate", "ffn.up", "ffn.down")
FILES = {"A": MODELS / "booklm-instruct-lora.pt",
         "B": OUT / "ch08-merge-request-lora.pt",
         "AB": OUT / "ch08-merge-multitask-lora.pt"}


# ----------------------------------------------------------------------------- данные двух задач

def pairs_A(rows):
    return [(r["request"], r["story"]) for r in rows]          # просьба -> история


def pairs_B(rows):
    return [(r["story"], r["request"]) for r in rows]          # история -> просьба


# ----------------------------------------------------------------------------- обучение адаптера (как sft_lora.py)

def train_lora(pairs, epochs=2, bsz=32, lr=2e-3, r=16, alpha=32, seed=0):
    torch.manual_seed(seed)
    tok, model = load("base")
    data = [encode(tok, u, a, 512) for u, a in pairs]
    add_lora(model, r, alpha)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    steps = int(len(data) * epochs / bsz)
    model.train()
    t0 = time.perf_counter()
    for step in range(steps):
        lr_now = lr * min(1.0, (step + 1) / 50) * 0.5 * (1 + math.cos(math.pi * step / steps))
        for g in opt.param_groups:
            g["lr"] = lr_now
        batch = random.Random(step).sample(data, bsz)
        L = max(len(i) for i, _ in batch) - 1
        x = torch.tensor([i[:-1] + [0] * (L - len(i) + 1) for i, _ in batch])
        y = torch.tensor([l[1:] + [-100] * (L - len(l) + 1) for _, l in batch])
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten(), ignore_index=-100)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 50 == 0 or step == steps - 1:
            print(f"  шаг {step}/{steps}: ошибка {loss.item():.3f}, {time.perf_counter() - t0:.0f} с", flush=True)
    adapter = {n: p.detach().clone() for n, p in model.state_dict().items() if n.endswith((".A", ".B"))}
    return {"adapter": adapter, "r": r, "alpha": alpha, "targets": TYPES, "steps": steps}


def train_all():
    rows = sft_train()
    print("адаптер B (история -> просьба):")
    torch.save(train_lora(pairs_B(rows)), FILES["B"])
    print("адаптер A+B (обе задачи вперемешку):")
    torch.save(train_lora(pairs_A(rows) + pairs_B(rows)), FILES["AB"])


# ----------------------------------------------------------------------------- векторы задач и слияния

def task_vector(file):
    """tau = (alpha/r) B A для каждой из 42 матриц; остальные веса адаптер не трогает."""
    f = torch.load(file, map_location="cpu")
    s, ad = f["alpha"] / f["r"], f["adapter"]
    return {k[:-2] + ".weight": s * ad[k[:-2] + ".B"] @ ad[k] for k in ad if k.endswith(".A")}


def apply(base_state, taus, lam=1.0):
    """theta = theta_base + lam * сумма tau."""
    new = {k: v.clone() for k, v in base_state.items()}
    for tau in taus:
        for k, d in tau.items():
            new[k] += lam * d
    return new


def ties(taus, k=0.2):
    """TIES: в каждой матрице оставить k самых крупных по модулю, выбрать знак по сумме, усреднить согласных."""
    out = {}
    for name in taus[0]:
        T = torch.stack([t[name] for t in taus])                       # [задачи, ...]
        flat = T.abs().flatten(1)
        thr = flat.kthvalue(int((1 - k) * flat.size(1)), dim=1).values  # порог для каждой задачи
        T = T * (T.abs() >= thr.view(-1, *[1] * (T.dim() - 1)))
        sign = torch.sign(T.sum(0))                                     # знак с наибольшей массой
        agree = (torch.sign(T) == sign) & (T != 0)
        out[name] = (T * agree).sum(0) / agree.sum(0).clamp(min=1)
    return out


def dare(tau, p=0.9, seed=0):
    """DARE: выбросить долю p чисел поправки, остальные умножить на 1/(1-p)."""
    g = torch.Generator().manual_seed(seed)
    return {k: d * (torch.rand(d.shape, generator=g) >= p) / (1 - p) for k, d in tau.items()}


def model_from(state):
    _, m = load("base")
    m.load_state_dict(state)
    return m.eval()


def scores(model, tok, rows):
    return (story_loss(model, tok, pairs_A(rows)), story_loss(model, tok, pairs_B(rows)), perplexity(model))


def table():
    tok, base = load("base")
    rows = held_out()
    W0 = {k: v.clone() for k, v in base.state_dict().items()}
    tA, tB = task_vector(FILES["A"]), task_vector(FILES["B"])
    cos = sum((tA[k] * tB[k]).sum() for k in tA) / math.sqrt(
        sum(tA[k].pow(2).sum() for k in tA) * sum(tB[k].pow(2).sum() for k in tB))
    print(f"косинус между векторами задач A и B: {cos:.3f}")
    norm = lambda t: math.sqrt(sum(t[k].pow(2).sum() for k in t))
    w_norm = norm({k: W0[k] for k in tA})                       # те же 42 матрицы основы
    for name, t in (("A", tA), ("B", tB)):
        mean_abs = torch.cat([d.abs().flatten() for d in t.values()]).mean()
        print(f"вектор {name}: |tau| / |W| = {norm(t) / w_norm:.3f}, средний модуль числа {mean_abs:.4f}")
    noise = {k: v - tA[k] for k, v in dare(tA, seed=1).items()}
    print(f"DARE p=0,9 добавляет к A шум с |шум| / |W| = {norm(noise) / w_norm:.3f}")
    variants = {
        "база": W0,
        "только A": apply(W0, [tA]),
        "только B": apply(W0, [tB]),
        "A+B, lam=1": apply(W0, [tA, tB], 1.0),
        "A+B, lam=0,75": apply(W0, [tA, tB], 0.75),
        "среднее, lam=0,5": apply(W0, [tA, tB], 0.5),
        "A + 0,5 B": apply(W0, [tA, {k: 0.5 * v for k, v in tB.items()}]),
        "TIES k=0,2": apply(W0, [ties([tA, tB])]),
        "DARE p=0,5": apply(W0, [dare(tA, p=0.5, seed=1), dare(tB, p=0.5, seed=2)]),
        "DARE p=0,9": apply(W0, [dare(tA, seed=1), dare(tB, seed=2)]),
        "обучены вместе": apply(W0, [task_vector(FILES["AB"])]),
    }
    results = {}
    print(f"{'модель':18s} {'ошибка A':>9s} {'ошибка B':>9s} {'перплексия':>11s}")
    for name, state in variants.items():
        results[name] = scores(model_from(state), tok, rows)
        a, b, ppl = results[name]
        print(f"{name:18s} {a:9.4f} {b:9.4f} {ppl:11.3f}")

    # Проверки: каждый адаптер учит свою задачу; слияние двух не лучше обученной вместе модели по обеим.
    assert results["только A"][0] < results["база"][0] and results["только B"][1] < results["база"][1]
    assert results["обучены вместе"][1] <= results["A+B, lam=1"][1] + 1e-6
    merges = [n for n in variants if n not in ("база", "только A", "только B", "обучены вместе")]
    # Совместный адаптер лучше любого слияния по задаче A и по перплексии (по задаче B lam=0,75 чуть лучше).
    assert results["обучены вместе"][0] < min(results[n][0] for n in merges)
    assert results["обучены вместе"][2] < min(results[n][2] for n in merges)
    assert all(results[n][0] > results["база"][0] for n in merges)  # слияния отдают весь выигрыш A
    assert results["DARE p=0,9"][2] > 100 * results["база"][2]     # крупные поправки DARE не переносят

    # Образцы: жадные ответы трёх моделей на одну просьбу и на одну историю.
    req, story = rows[0]["request"], rows[1]["story"]
    for name in ("только A", "только B", "A+B, lam=1", "обучены вместе"):
        m = model_from(variants[name])
        print(f"[{name}] на просьбу: {' '.join(generate(tok, m, req, chat=True, max_new=30, temperature=0).split())}")
        print(f"[{name}] на историю: {' '.join(generate(tok, m, story, chat=True, max_new=30, temperature=0).split())}")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("train", "all"):
        train_all()
    if what in ("table", "all"):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            table()
        print(buf.getvalue(), end="")
        (OUT / "ch08-merge-table.txt").write_text(buf.getvalue(), encoding="utf-8")
