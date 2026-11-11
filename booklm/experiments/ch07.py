"""Глава 7 на обученной BookLM: настоящий прогон обучения в числах.

Опыты (время — на процессоре Apple M2 Max):
  ch07_first_steps — раздел «Один шаг обучения», рубрика «А теперь на нашей модели: шаг обучения».
      Ошибка на первых 100 шагах прогона и продолжения «Однажды» на шагах 0, 25 и 100 из журнала
      обучения (ничего не обучает). Около 1 с.
  ch07_doc_leak — раздел «Подготовка обучающих данных», рубрика «А теперь на нашей модели: маска
      документов». Читает ли BookLM соседнюю сказку в одном окне: ошибка на истории B после истории A,
      после другой истории A' и без соседа. Около 2 с. Данные рисунка: ch07-doc-leak.csv.
  ch07_lr_gradnorm — раздел «Оптимизаторы и расписание скорости обучения», рубрика «А теперь на нашей
      модели: расписание скорости и норма градиента». Расписание скорости из журнала и как часто
      срабатывало ограничение нормы градиента. Около 1 с. Данные рисунка: ch07-lr-gradnorm.csv.
  ch07_bf16 — раздел «Обучение в 16-битных числах», рубрика «А теперь на нашей модели: смешанная
      точность». Ошибка BookLM в FP32 и в BF16 и доля весов, у которых шаг обучения тонет при округлении
      до BF16. Около 5 с. Данные рисунка: ch07-bf16.csv.
  ch07_training_run — раздел «Длинный прогон и бюджет обучения», рубрика «А теперь на нашей модели:
      прогон обучения». Токены, эпохи, D/N, MFU, кривые ошибки и ошибка на сохранениях прогона.
      Около 35 с. Данные рисунка: ch07-training-run.csv.
txt опыты не пишут: вывод на экран, таблицы для рисунков — в csv.
Синтетические данные — в ch07_synthetic.py, Muon и промежуточный этап — в ch07_muon_midtraining.py.

Запуск из корня репозитория:
    python -m booklm.experiments.ch07                   # все опыты
    python -m booklm.experiments.ch07 ch07_doc_leak     # один опыт
    python -m booklm.experiments.ch07 --help            # список опытов
Файлы выводов лежат в models/booklm/experiments/.
"""
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

if not __package__:  # запуск файлом (python booklm/experiments/ch07.py): корень репозитория — в путь поиска
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from booklm.experiments import run
from booklm.lab import MODELS, load, load_model, training_log, val_tokens

OUT = MODELS / "experiments"
RUN = json.load(open(MODELS / "train_run.json"))   # это run.json прогона
N_PARAMS = 6_622_464


def write_csv(name, header, rows):
    with open(OUT / name, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def short(text, n=90):
    # одна строка для книги: без переносов, не длиннее n символов
    text = " ".join(text.split())
    return text if len(text) <= n else text[:n] + "…"


# ----------------------------------------------------------------------------- 7.1 первые шаги

def ch07_first_steps():
    """Ошибка и продолжения «Однажды» на первых 100 шагах прогона (из журнала обучения)."""
    log = training_log()
    loss = {r["step"]: r["loss"] for r in log if "loss" in r}        # обучающая, каждые 10 шагов
    samples = {r["step"]: r["samples"][0] for r in log if "samples" in r}
    val0 = next(r["val_loss"] for r in log if r["step"] == 0 and "val_loss" in r)
    print(f"валидация на шаге 0: {val0:.3f}, ln 8192 = {math.log(8192):.3f}")
    print("обучающая ошибка:")
    for s in range(0, 101, 10):
        print(f"  шаг {s:3d}: {loss[s]:.3f}")
    print("продолжения «Однажды» (из журнала, температура 0.8, top-k 40):")
    for s in (0, 25, 100):
        print(f"  шаг {s:3d}: {short(samples[s])}")


# ----------------------------------------------------------------------------- 7.2 маска документов

def val_stories():
    # val.bin — поток <bos> история <eos> <bos> история <eos> ...
    data = val_tokens(10**9).tolist()
    stories, cur = [], None
    for t in data:
        if t == 1:
            cur = []
        elif t == 2:
            stories.append(cur)
        else:
            cur.append(t)
    return stories


@torch.no_grad()
def losses_on_b(model, prefix, b):
    # окно <bos> [prefix <eos> <bos>] B <eos>; ошибка и логиты на каждом токене B и на его <eos>
    ids = [1] + (prefix + [2, 1] if prefix else []) + b + [2]
    x = torch.tensor([ids[:-1]])
    y = torch.tensor(ids[1:])
    logits = model(x)[0]
    start = len(ids) - len(b) - 2          # позиция, которая предсказывает первый токен B
    nll = F.cross_entropy(logits, y, reduction="none")
    return nll[start:], logits[start:]


def ch07_doc_leak():
    """Читает ли BookLM соседнюю сказку в окне: ошибка на истории B с соседом и без."""
    tok, model = load("base")
    stories = val_stories()
    by_len = {}
    for i, s in enumerate(stories):
        by_len.setdefault(len(s), []).append(i)
    rng = np.random.default_rng(0)
    # тройки (A, A', B): A и A' одной длины, B — другая история; всё влезает в 512
    pairs = [v[:2] for L, v in sorted(by_len.items()) if len(v) >= 2 and 100 <= L <= 200]
    triples = []
    for a, a2 in pairs:
        b = int(rng.integers(len(stories)))
        if b not in (a, a2) and len(stories[a]) + len(stories[b]) + 4 <= 512:
            triples.append((a, a2, b))
    triples = triples[:20]

    shifts, means = [], []
    for n, (a, a2, b) in enumerate(triples):
        A, A2, B = stories[a], stories[a2], stories[b]
        l1, z1 = losses_on_b(model, A, B)
        l2, z2 = losses_on_b(model, A2, B)
        l0, _ = losses_on_b(model, [], B)
        shifts.append((l1 - l2).abs())
        means.append((l1.mean().item(), l2.mean().item(), l0.mean().item()))
        if n == 0:                                     # подробно печатаем первую тройку
            print(f"A, A' — по {len(A)} токенов, B — {len(B)} токенов (+<eos>)")
            print(f"  B: {short(tok.decode(B[:25]), 70)}")
            print(f"ошибка на B после A : {l1.mean():.4f}")
            print(f"ошибка на B после A': {l2.mean():.4f}")
            print(f"ошибка на B без соседа: {l0.mean():.4f}")
            print(f"наибольший сдвиг логитов на позициях B: {(z1 - z2).abs().max():.3f}")
            rows = [[j, round(l1[j].item(), 4), round(l2[j].item(), 4), round(l0[j].item(), 4)]
                    for j in range(len(l1))]
    # средний |сдвиг ошибки| по позициям B, по всем тройкам
    T = min(len(s) for s in shifts)
    mean_shift = torch.stack([s[:T] for s in shifts]).mean(0)
    print(f"\n|сдвиг ошибки| при замене A на A', среднее по {len(triples)} тройкам:")
    for lo, hi in ((0, 1), (1, 5), (5, 20), (20, 50), (50, T)):
        print(f"  токены B {lo:3d}–{hi - 1:3d}: {mean_shift[lo:hi].mean():.4f}")
    m = np.array(means)
    print(f"средняя ошибка на B по {len(triples)} тройкам: после соседа {m[:, :2].mean():.4f},"
          f" без соседа {m[:, 2].mean():.4f}")
    for r in rows:
        r.append(round(mean_shift[r[0]].item(), 4) if r[0] < T else "")
    write_csv("ch07-doc-leak.csv", ["pos_in_B", "loss_after_A", "loss_after_A2", "loss_alone",
                                    "mean_abs_shift_20"], rows)


# ----------------------------------------------------------------------------- 7.4 скорость и норма

def ch07_lr_gradnorm():
    """Расписание скорости обучения и срабатывания ограничения нормы градиента в прогоне."""
    a, steps = RUN["args"], RUN["steps"]
    log = [r for r in training_log() if "loss" in r]
    print(f"пиковая скорость {a['lr']:.0e}, разогрев {a['warmup']} шагов, всего {steps} шагов,")
    print(f"в конце спада {a['lr'] * a['min_lr_frac']:.0e} ({a['min_lr_frac']:.0%} пика)")
    middle = (a["warmup"] + steps) // 2
    phases = (("разогрев", 0, a["warmup"]), ("первая половина спада", a["warmup"], middle),
              ("вторая половина спада", middle, steps))
    print("доля записей с нормой градиента > 1 (ограничение сработало):")
    for name, lo, hi in phases:
        part = [r["grad_norm"] for r in log if lo <= r["step"] < hi]
        clipped = sum(g > 1 for g in part)
        print(f"  {name:<22} шаги {lo:4d}–{hi - 1:4d}: {clipped:3d} из {len(part):3d}"
              f" = {clipped / len(part):.0%}")
    last = max(r["step"] for r in log if r["grad_norm"] > 1)
    print(f"последний раз норма больше 1 на шаге {last}")
    g = [r["grad_norm"] for r in log]
    print(f"норма: шаг 0 — {g[0]:.2f}, медиана шагов 300–1000 — "
          f"{np.median([r['grad_norm'] for r in log if 300 <= r['step'] < 1000]):.2f}, "
          f"последняя — {g[-1]:.2f}")
    write_csv("ch07-lr-gradnorm.csv", ["step", "lr", "grad_norm"],
              [[r["step"], f"{r['lr']:.4e}", round(r["grad_norm"], 4)] for r in log])


# ----------------------------------------------------------------------------- 7.5 BF16

def drowned(w, eta):
    # доля весов, у которых в BF16 шаг w - eta*sign(w) округляется обратно к w
    wb = w.to(torch.bfloat16)
    step = (wb.float() - eta * wb.float().sign()).to(torch.bfloat16)
    return (step == wb).float().mean().item()


@torch.no_grad()
def ch07_bf16():
    """BookLM в FP32 и в BF16 и доля весов, у которых шаг тонет в BF16."""
    torch.manual_seed(0)
    _, model = load("base")
    data = val_tokens(32 * 512 + 1)
    x = data[:-1].view(32, 512)
    y = data[1:].view(32, 512)
    z32 = model(x).float()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        z16 = model(x).float()
    l32 = F.cross_entropy(z32.flatten(0, 1), y.flatten()).item()
    l16 = F.cross_entropy(z16.flatten(0, 1), y.flatten()).item()
    print(f"ошибка на 32 окнах по 512: FP32 {l32:.5f}, BF16 {l16:.5f}, разность {l16 - l32:+.5f}")
    print(f"наибольшее расхождение логитов {(z16 - z32).abs().max():.3f}"
          f" (логиты от {z32.min():.1f} до {z32.max():.1f})")

    groups = {"таблица токенов": [], "матрицы блоков": [], "усиления RMSNorm": []}
    for name, p in model.named_parameters():
        key = "таблица токенов" if name == "embed.weight" else (
            "матрицы блоков" if p.dim() == 2 else "усиления RMSNorm")
        groups[key].append(p.detach().flatten())
    eta_real = RUN["args"]["lr"] * RUN["args"]["min_lr_frac"]
    etas = (("настоящий", eta_real), ("учебный", 3e-5))
    print(f"\nдоля весов, где шаг eta тонет в BF16:   eta = {eta_real:.0e}   eta = 3e-05")
    for key, ws in groups.items():
        w = torch.cat(ws)
        cells = "   ".join(f"{drowned(w, e):11.1%}" for _, e in etas)
        print(f"  {key:<17} (медиана |w| {w.abs().median():.3f}): {cells}")
    for label, e in etas:
        print(f"порог 256*eta ({label}): {256 * e:.4f}")

    # гистограмма |w| в логарифмических корзинах для pgfplots
    edges = np.logspace(-6, 1, 71)
    rows = []
    for key, ws in groups.items():
        counts, _ = np.histogram(torch.cat(ws).abs().numpy(), bins=edges)
        rows.append(counts)
    write_csv("ch07-bf16.csv", ["abs_w_lo", "abs_w_hi", "embed", "blocks", "rmsnorm"],
              [[f"{edges[i]:.3e}", f"{edges[i + 1]:.3e}", rows[0][i], rows[1][i], rows[2][i]]
               for i in range(len(edges) - 1)])


# ----------------------------------------------------------------------------- 7.6 прогон

@torch.no_grad()
def window_losses(model, data, starts, T=512):
    # ошибка каждого окна отдельно — нужна для стандартной ошибки среднего
    x = torch.stack([data[s:s + T] for s in starts])
    y = torch.stack([data[s + 1:s + T + 1] for s in starts])
    nll = F.cross_entropy(model(x).flatten(0, 1), y.flatten(), reduction="none")
    return nll.view(len(starts), T).mean(1)


def ch07_training_run():
    """Прогон BookLM в числах: токены, эпохи, MFU, кривые ошибки, ошибка на сохранениях."""
    log = training_log()
    steps, per_step = RUN["steps"], RUN["tokens_per_step"]
    train_tokens = RUN["train_tokens"]
    D = steps * per_step
    last = [r for r in log if "loss" in r][-1]
    seconds = last["tokens"] / last["tok_per_s"]          # время с начала прогона, вместе с оценками
    peak = 125e12        # GB10, BF16 без разреженности: 1 PFLOP FP4 с разреженностью / 8
    mamf = 99.8e12       # измеренный предел умножения матриц BF16 на GB10 (StorageReview, mamf-finder)
    useful = 6 * N_PARAMS * last["tok_per_s"]
    print(f"токенов в обучающей части: {train_tokens:,}".replace(",", " "))
    print(f"увидено токенов D = {steps} x {per_step} = {D:,}".replace(",", " "))
    print(f"эпох: {D / train_tokens:.2f}")
    print(f"D/N = {D / N_PARAMS:.1f} (правило Chinchilla: 20)")
    speed = f"{last['tok_per_s']:,.0f}".replace(",", " ")
    print(f"время прогона: {seconds / 60:.1f} мин, {speed} ток./с")
    print(f"6N x ток./с = {useful / 1e12:.2f} TFLOP/s; MFU = {useful / peak:.1%} от 125 TFLOP/s,"
          f" {useful / mamf:.1%} от измеренных 99.8")

    # сглаженная обучающая ошибка: среднее по соседним записям, до 5 с каждой стороны
    # (у краёв окно сужается, чтобы не тянуть начало кривой вниз), и валидация
    tr = [(r["step"], r["loss"]) for r in log if "loss" in r]
    val = {r["step"]: r["val_loss"] for r in log if "val_loss" in r}
    rows = []
    for i, (s, l) in enumerate(tr):
        k = min(5, i, len(tr) - 1 - i)
        win = [v for _, v in tr[i - k:i + k + 1]]
        rows.append([s, round(l, 4), round(sum(win) / len(win), 4), round(val.get(s, math.nan), 4)])
    for s in val:
        if s not in dict(tr):
            rows.append([s, "nan", "nan", round(val[s], 4)])
    write_csv("ch07-training-run.csv", ["step", "train_loss", "train_smooth", "val_loss"],
              sorted(rows, key=lambda r: r[0]))

    print("\nвалидационная и сглаженная обучающая ошибка из журнала:")
    smooth = {r[0]: r[2] for r in rows if r[2] != ""}
    for s in (250, 500, 1000, 2000, 3000):
        print(f"  шаг {s:4d}: валидация {val[s]:.3f}, обучение {smooth[s]:.3f},"
              f" разрыв {val[s] - smooth[s]:+.3f}")

    # разрыв на сохранениях: одни и те же n случайных окон обучения и n окон валидации.
    # n = 256: при 32 окнах стандартная ошибка ~0,04 и случайный выбор окон сдвигает
    # обучающую ошибку на 0,1 — «сколько сигм» тогда зависит от везения с окнами.
    n = 256
    train = torch.from_numpy(np.fromfile(MODELS / "data" / "train.bin", dtype=np.uint16).astype("int64"))
    train = train[:int(train.nonzero().max())]   # в копии репозитория хвост файла занулён: берём начало
    vdata = val_tokens(10**9)
    gen = torch.Generator().manual_seed(0)
    tstarts = torch.randint(len(train) - 513, (n,), generator=gen).tolist()
    vstarts = torch.randint(len(vdata) - 513, (n,), generator=gen).tolist()
    print(f"на сохранениях (CPU, {n} + {n} окон по 512):")
    for kind in ("step:100", "step:762", "step:1000", "step:1525", "base"):
        model = load_model(kind)
        lt = torch.cat([window_losses(model, train, tstarts[i:i + 32]) for i in range(0, n, 32)])
        lv = torch.cat([window_losses(model, vdata, vstarts[i:i + 32]) for i in range(0, n, 32)])
        sigma = math.sqrt(lt.var() / n + lv.var() / n)
        print(f"  {kind:<9}: обучение {lt.mean():.3f}, валидация {lv.mean():.3f},"
              f" разрыв {lv.mean() - lt.mean():.3f}, сигма {sigma:.3f}, {(lv.mean() - lt.mean()) / sigma:.1f} сигмы")

    samples = {r["step"]: r["samples"][0] for r in log if "samples" in r}
    print("\n«Однажды» (температура 0.8, top-k 40):")
    print(f"  шаг  250: {short(samples[250], 95)}")
    print(f"  шаг {steps}: {short(samples[steps], 95)}")


EXPERIMENTS = {"ch07_first_steps": ch07_first_steps,
               "ch07_doc_leak": ch07_doc_leak,
               "ch07_lr_gradnorm": ch07_lr_gradnorm,
               "ch07_bf16": ch07_bf16,
               "ch07_training_run": ch07_training_run}

if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    run(EXPERIMENTS)
