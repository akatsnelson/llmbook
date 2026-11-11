"""Линейная связность и усреднение весов на BookLM-mini (глава 8, раздел 8.7 «Слияние моделей»).

BookLM-mini — тот же класс BookLM, 2 блока ширины 128 (1,44 млн параметров), как у учеников из раздела
8.6 «Дистилляция». Пакеты 16 x 256 из train.bin, перплексия — booklm.lab.perplexity (50 000 токенов val.bin).

1. Барьер на прямой между двумя моделями. Перплексия модели (1 - t) theta_1 + t theta_2 при t = 0, 0.1, ..., 1
   для трёх пар:
     (а) разные инициализации, одни и те же 600 пакетов;
     (б) одна инициализация, разный порядок пакетов;
     (в) общий родитель: модель после 600 шагов, от неё два «дообучения» по 300 шагов со скоростью 1e-3
         на разных пакетах, то есть две модели от одной базы.
2. Усреднение по одному прогону. Тот же старт, 600 шагов с постоянной скоростью 3e-3 после разогрева:
   последняя точка, EMA с коэффициентом 0,99 и среднее последних пяти контрольных точек (шаги 400, 450, ..., 600)
   против прогона с косинусным спадом.

Всё повторяется для пяти зёрен s = 0..4 (зерно задаёт инициализацию и порядок пакетов); печатаются среднее
и разброс, а кривые в CSV — средние по зёрнам.

Запуск из корня репозитория: python labs/ch08/merge_mini_barrier.py
Время: около 20 минут на CPU M2 Max.

Печать: models/booklm/experiments/ch08-merge-barrier.txt, кривые: ch08-merge-barrier.csv.
"""
import contextlib
import copy
import csv
import io
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from booklm.lab import MODELS, perplexity                 # noqa: E402
from booklm.model import BookLM                           # noqa: E402
from booklm.experiments.ch08_distill import MINI, batches  # noqa: E402

OUT = MODELS / "experiments"
TS = [i / 10 for i in range(11)]


def train(model, steps, data_seed, lr=3e-3, warmup=30, cosine=True, ema_decay=None, keep_every=None):
    """Обучает model на месте. Возвращает EMA-копию (если просили) и список контрольных точек."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    ema = copy.deepcopy(model) if ema_decay else None
    kept = []
    model.train()
    for step, (x, y) in enumerate(batches(steps, seed=data_seed), start=1):
        if step <= warmup:
            lr_now = lr * step / warmup
        elif cosine:
            frac = (step - 1 - warmup) / max(1, steps - warmup)
            lr_now = lr * (0.1 + 0.45 * (1 + math.cos(math.pi * frac)))
        else:
            lr_now = lr
        for g in opt.param_groups:
            g["lr"] = lr_now
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if ema is not None:
            with torch.no_grad():
                for pe, p in zip(ema.parameters(), model.parameters()):
                    pe.mul_(ema_decay).add_(p, alpha=1 - ema_decay)
        if keep_every and step % keep_every == 0:
            kept.append({k: v.clone() for k, v in model.state_dict().items()})
    model.eval()
    return ema, kept


def fresh(seed):
    torch.manual_seed(seed)
    return BookLM(MINI)


def mix(sa, sb, t):
    return {k: (1 - t) * sa[k] + t * sb[k] for k in sa}


def ppl_of(state):
    m = BookLM(MINI)
    m.load_state_dict(state)
    return perplexity(m.eval())


def one_seed(s):
    """Все модели опыта для одного зерна s; возвращает три кривые и четыре перплексии второй части."""
    m = fresh(s); train(m, 600, data_seed=s)                       # косинус, общий конец всех пар
    other_init = fresh(100 + s); train(other_init, 600, data_seed=s)
    other_data = fresh(s); train(other_data, 600, data_seed=100 + s)
    kids = []
    for ds in (200 + s, 300 + s):
        kid = copy.deepcopy(m)
        train(kid, 300, data_seed=ds, lr=1e-3, warmup=10)
        kids.append(kid)
    pairs = {"(а) разные инициализации": (m, other_init),
             "(б) одна инициализация, разные пакеты": (m, other_data),
             "(в) два дообучения одного родителя": tuple(kids)}
    curves = {name: [ppl_of(mix(p.state_dict(), q.state_dict(), t)) for t in TS]
              for name, (p, q) in pairs.items()}
    m_const = fresh(s)
    ema, kept = train(m_const, 600, data_seed=s, cosine=False, ema_decay=0.99, keep_every=50)
    last5 = kept[-5:]
    avg = {k: sum(c[k] for c in last5) / len(last5) for k in last5[0]}
    rows = {"постоянная скорость, последняя точка": perplexity(m_const),
            "EMA весов, beta = 0,99": perplexity(ema.eval()),
            "среднее 5 точек, шаги 400-600": ppl_of(avg),
            "косинусный спад, последняя точка": perplexity(m)}
    return curves, rows


def main(seeds=range(5)):
    runs = [one_seed(s) for s in seeds]
    mean = lambda xs: sum(xs) / len(xs)
    span = lambda xs: f"{mean(xs):7.2f} ({min(xs):.1f}-{max(xs):.1f})"
    print(f"{len(runs)} зёрен; среднее и в скобках разброс от наименьшего до наибольшего")
    names = list(runs[0][0])
    curves = {n: [mean([r[0][n][i] for r in runs]) for i in range(len(TS))] for n in names}
    for n in names:
        ends = [(r[0][n][0] + r[0][n][-1]) / 2 for r in runs]
        mids = [r[0][n][5] for r in runs]
        bars = [m - e for m, e in zip(mids, ends)]
        print(f"{n}:")
        print(f"   концы {span(ends)}, середина {span(mids)}, барьер {span(bars)}")
    print("t:   " + " ".join(f"{t:6.1f}" for t in TS))
    for n, c in curves.items():
        print(n[:3] + "  " + " ".join(f"{v:6.1f}" for v in c))
    for name in runs[0][1]:
        print(f"{name:38s} перплексия {span([r[1][name] for r in runs])}")

    # Проверки на каждом зерне: барьер (а) велик, у (в) середина не хуже худшего конца,
    # среднее точек и EMA лучше последней точки прогона с постоянной скоростью.
    for c, rows in runs:
        a, v = c["(а) разные инициализации"], c["(в) два дообучения одного родителя"]
        assert a[5] > 2 * max(a[0], a[-1])
        assert v[5] <= max(v[0], v[-1])
        last = rows["постоянная скорость, последняя точка"]
        assert rows["среднее 5 точек, шаги 400-600"] < last and rows["EMA весов, beta = 0,99"] < last

    with open(OUT / "ch08-merge-barrier.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t", "diff_init", "same_init", "same_parent"])
        for i, t in enumerate(TS):
            w.writerow([t] + [round(c[i], 3) for c in curves.values()])


if __name__ == "__main__":
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        main()
    print(buf.getvalue(), end="")
    (OUT / "ch08-merge-barrier.txt").write_text(buf.getvalue(), encoding="utf-8")
