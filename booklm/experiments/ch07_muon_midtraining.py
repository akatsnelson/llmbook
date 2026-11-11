"""Глава 7: оптимизатор Muon и промежуточный этап обучения на BookLM.

Оба опыта обучают настоящую BookLM на процессоре короткими прогонами: пакеты 16 x 256 токенов из
models/booklm/data/train.bin, ошибка на отложенных историях val.bin. Больше пары минут каждый.

Опыты (время — на процессоре Apple M2 Max, 8 потоков):
  ch07_muon — раздел «Оптимизаторы и расписание скорости обучения», рубрика «А теперь на нашей модели:
      оптимизатор Muon». Шесть прогонов по 200 шагов: AdamW и Muon при трёх скоростях обучения.
      Около 45 минут: torch.optim.Muon считает Ньютона–Шульца в BF16, на CPU это 3–4 с на шаг против
      0,5–0,7 с у AdamW. Копия вывода: ch07-muon.txt, журнал: ch07-muon.csv.
  ch07_midtraining — раздел «Длинный прогон и бюджет обучения», рубрика «А теперь на нашей модели:
      промежуточный этап обучения». Два прогона по 200 шагов: с парами «просьба -> сказка» в смеси
      и без них. Около 10–15 минут. Копия вывода: ch07-midtraining.txt.
Времена шагов в выводе зависят от машины, остальные числа совпадают с книгой.

Запуск из корня репозитория:
    python -m booklm.experiments.ch07_muon_midtraining ch07_muon
    python -m booklm.experiments.ch07_muon_midtraining ch07_midtraining
    python -m booklm.experiments.ch07_muon_midtraining --help        # список опытов
Файлы выводов лежат в models/booklm/experiments/.
"""
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

if not __package__:  # запуск файлом (python booklm/experiments/ch07_muon_midtraining.py): корень репозитория — в путь поиска
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from booklm.experiments import run
from booklm.lab import MODELS, encode_text, load_model, load_tokenizer

OUT = MODELS / "experiments"
B, T = 16, 256                                   # пакет: 16 окон по 256 токенов


def train_stream():
    return np.fromfile(MODELS / "data" / "train.bin", dtype=np.uint16)


def split_stories(data):
    # поток <bos> история <eos> <bos> ... -> список массивов, каждый с <bos> и <eos>
    starts = np.where(data == 1)[0]
    return np.split(data, starts[1:])


def window_sampler(data, seed):
    gen = torch.Generator().manual_seed(seed)

    def sample(n=B):
        ix = torch.randint(len(data) - T - 1, (n,), generator=gen).tolist()
        x = torch.stack([torch.from_numpy(data[i:i + T].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(data[i + 1:i + T + 1].astype(np.int64)) for i in ix])
        return x, y
    return sample


def train_steps(model, opts, lr_fn, steps, sample, log=None, every=25, eval_fn=None):
    # opts: оптимизаторы, у каждой группы параметров в "base_lr" её пиковая скорость
    model.train()
    for s in range(steps):
        frac = lr_fn(s)
        for opt in opts:
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * frac
        x, y = sample()
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
        for opt in opts:
            opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for opt in opts:
            opt.step()
        if log is not None and ((s + 1) % every == 0 or s == 0):
            log.append((s + 1, loss.item(), eval_fn() if eval_fn else float("nan")))
    return model


def cosine(warmup, steps, floor=0.1):
    def f(s):
        if s < warmup:
            return (s + 1) / warmup
        p = (s - warmup) / max(1, steps - warmup)
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * p))
    return f


# ----------------------------------------------------------------------------- 7.4 Muon

def make_optimizers(model, kind, lr):
    """kind = "adamw": AdamW на всём, как в booklm/train.py.
    kind = "muon": Muon на матрицах блоков, AdamW на таблице токенов и усилениях RMSNorm."""
    mats = [p for n, p in model.named_parameters() if p.dim() == 2 and n.startswith("blocks.")]
    emb = [model.embed.weight]
    gains = [p for p in model.parameters() if p.dim() < 2]
    if kind == "adamw":
        opt = torch.optim.AdamW([{"params": mats + emb, "weight_decay": 0.1},
                                 {"params": gains, "weight_decay": 0.0}],
                                lr=lr, betas=(0.9, 0.95), eps=1e-8)
        opts = [opt]
    else:
        muon = torch.optim.Muon(mats, lr=lr, weight_decay=0.1, momentum=0.95,
                                adjust_lr_fn="match_rms_adamw")
        adam = torch.optim.AdamW([{"params": emb, "weight_decay": 0.1},
                                  {"params": gains, "weight_decay": 0.0}],
                                 lr=lr, betas=(0.9, 0.95), eps=1e-8)
        opts = [muon, adam]
    for opt in opts:
        for g in opt.param_groups:
            g["base_lr"] = g["lr"]
    return opts


def ch07_muon(steps=200, lrs=(1e-3, 2e-3, 4e-3)):
    """AdamW и Muon на первых 200 шагах BookLM при трёх скоростях обучения (около 45 минут)."""
    torch.manual_seed(0)
    data = train_stream()
    val = np.fromfile(MODELS / "data" / "val.bin", dtype=np.uint16)
    gen = torch.Generator().manual_seed(1)
    vs = torch.randint(len(val) - T - 1, (64,), generator=gen).tolist()
    vx = torch.stack([torch.from_numpy(val[i:i + T].astype(np.int64)) for i in vs])
    vy = torch.stack([torch.from_numpy(val[i + 1:i + T + 1].astype(np.int64)) for i in vs])

    @torch.no_grad()
    def val_loss(model):
        model.eval()
        l = sum(F.cross_entropy(model(vx[i:i + 16]).flatten(0, 1), vy[i:i + 16].flatten()).item()
                for i in range(0, 64, 16)) / 4
        model.train()
        return l

    n_mats = sum(p.numel() for n, p in load_model("init").named_parameters()
                 if p.dim() == 2 and n.startswith("blocks."))
    print(f"матрицы блоков: {n_mats:,} параметров из 6,622,464 ({n_mats / 6_622_464:.0%})")
    rows, curves = [], {}
    for kind in ("adamw", "muon"):
        for lr in lrs:
            model = load_model("init").train()
            opts = make_optimizers(model, kind, lr)
            log, t0 = [], time.time()
            train_steps(model, opts, cosine(20, steps), steps, window_sampler(data, seed=7), log,
                every=25, eval_fn=lambda: val_loss(model))
            sec = time.time() - t0
            rows.append((kind, lr, log[-1][2], sec))
            curves[(kind, lr)] = log
            print(f"{kind:5s} lr {lr:.0e}: валидация после {steps} шагов {log[-1][2]:.3f}, "
                  f"шаг {sec / steps:.2f} с", flush=True)
    best = {k: min((r for r in rows if r[0] == k), key=lambda r: r[2]) for k in ("adamw", "muon")}
    print("лучшие:", ", ".join(f"{k} lr {r[1]:.0e} -> {r[2]:.3f}" for k, r in best.items()))
    a, m = curves[("adamw", best["adamw"][1])], curves[("muon", best["muon"][1])]
    target = a[-1][2]
    reach = next(s for s, _, v in m if v <= target)
    print(f"Muon догоняет итог AdamW ({target:.3f}) к шагу {reach}")
    with open(OUT / "ch07-muon.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["step", "adamw_val", "muon_val"])
        for (s, _, va), (_, _, vm) in zip(a, m):
            w.writerow([s, f"{va:.4f}", f"{vm:.4f}"])


# ----------------------------------------------------------------------------- 7.6 промежуточный этап

def request_ids(tok, row):
    # «Просьба: …» и «Сказка: …» обычным текстом, без служебных токенов чата; вернёт (номера, длина начала)
    head = [1] + encode_text(tok, f"Просьба: {row['request']}\nСказка:")
    return head + encode_text(tok, " " + row["story"]) + [2], len(head)


@torch.no_grad()
def story_part_loss(model, items):
    # средняя ошибка на токенах сказки при известной просьбе, по каждой паре отдельно
    model.eval()
    out = []
    for ids, n_head in items:
        ids = ids[:513]                            # окно BookLM: 512 токенов
        x = torch.tensor([ids[:-1]])
        nll = F.cross_entropy(model(x)[0], torch.tensor(ids[1:]), reduction="none")
        out.append(nll[n_head - 1:].mean().item())
    model.train()
    return torch.tensor(out)


def request_data(tok):
    """9752 пары «просьба -> сказка» одним потоком токенов и 200 отложенных пар (как в главе 8)."""
    rows = [json.loads(l) for l in open(MODELS / "data" / "sft.jsonl", encoding="utf-8")]
    held, pool = rows[:200], rows[200:]
    stream = np.array([t for r in pool for t in request_ids(tok, r)[0]], dtype=np.uint16)
    return stream, [request_ids(tok, r) for r in held]


def val_items(n=200):
    # первые n отложенных историй: окно <bos> история <eos>, ошибка на всех токенах
    stories = split_stories(np.fromfile(MODELS / "data" / "val.bin", dtype=np.uint16))[:n]
    return [(s.astype(np.int64).tolist(), 1) for s in stories]


def greedy(model, tok, text, n=20):
    # то же, что greedy из раздела 7.6
    ids = [1] + tok.encode(text).ids
    with torch.no_grad():
        for _ in range(n):
            ids.append(int(model(torch.tensor([ids]))[0, -1].argmax()))
    return tok.decode(ids[1:])


def ch07_midtraining():
    """Промежуточный этап BookLM: 200 шагов с парами «просьба -> сказка» и без них."""
    # тот же код, что в листинге «Промежуточный этап BookLM» раздела 7.6
    import textwrap
    tok = load_tokenizer()
    train, (req, held) = train_stream(), request_data(tok)
    stories, requests = window_sampler(train, seed=11), window_sampler(req, seed=12)

    def mixed():                                  # 12 окон сказок + 4 окна с просьбами
        x1, y1 = stories(12)
        x2, y2 = requests(4)
        return torch.cat([x1, x2]), torch.cat([y1, y2])

    prompt, base = "Просьба: Напиши сказку про ёжика и яблоко.\nСказка:", None
    sem = lambda d: (d.std() / len(d) ** 0.5).item()
    for name, sample in (("base", None), ("без пар", stories),
                         ("с парами", mixed)):
        model = load_model("base").train()
        if sample is not None:                    # 200 шагов, скорость 2e-4 -> 0
            # веса выпущены без состояния оптимизатора: моменты пустые, поэтому две группы,
            # как в разделе 7.4, и прогрев 20 шагов вместо полного шага с первого же раза
            mats = [p for p in model.parameters() if p.dim() >= 2]
            gains = [p for p in model.parameters() if p.dim() < 2]
            opt = torch.optim.AdamW([{"params": mats, "weight_decay": 0.1},
                                     {"params": gains, "weight_decay": 0.0}],
                                    lr=2e-4, betas=(0.9, 0.95))
            for g in opt.param_groups:
                g["base_lr"] = 2e-4
            warm = lambda s: min((s + 1) / 20, 1 - s / 200)
            train_steps(model, [opt], warm, 200, sample)
        lq, lg = story_part_loss(model, held), story_part_loss(model, val_items())
        if base is None:
            base = (lq, lg)
            print(f"{name:8s}: просьбы {lq.mean():.3f}, сказки {lg.mean():.3f}")
        else:
            dq, dg = lq - base[0], lg - base[1]
            print(f"{name:8s}: просьбы {lq.mean():.3f} ({dq.mean():+.3f} ± {sem(dq):.3f}), "
                  f"сказки {lg.mean():.3f} ({dg.mean():+.3f} ± {sem(dg):.3f})")
        if name != "без пар":
            text = greedy(model.eval(), tok, prompt, n=40).split("Сказка:")[1]
            print(textwrap.fill(f"{name}: " + " ".join(text.split()), width=74))


EXPERIMENTS = {"ch07_muon": ch07_muon, "ch07_midtraining": ch07_midtraining}

if __name__ == "__main__":
    torch.set_num_threads(8)
    run(EXPERIMENTS, save=True)      # копия вывода в models/booklm/experiments/<имя опыта>.txt
