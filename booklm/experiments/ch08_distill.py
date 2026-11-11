"""Глава 8, раздел «Дистилляция» на обученной BookLM: ученики BookLM-mini и обрезка с долечиванием.

Опыты (время — на процессоре Apple M2 Max, до 8 потоков):
  ch08_distill_students (старое имя students) — рубрика «А теперь на нашей модели: дистилляция».
      Ученик — тот же класс BookLM из booklm/model.py, только 2 блока и ширина 128 (1,44 млн
      параметров). Оба ученика стартуют с одних весов и видят одни и те же пакеты 16 x 256 из train.bin:
        (а) «текст»: обычная перекрёстная энтропия по следующему токену;
        (б) «учитель»: 0,9 * tau^2 * KL(p_T^tau || p_S^tau) + 0,1 * перекрёстная энтропия, tau = 2.
      Шаг (б) дороже: учитель делает прямой проход по тому же пакету. Поэтому в конце ученик (а) учится
      ещё раз столько шагов, сколько успевает за время (б), — сравнение при равных минутах CPU.
      Перплексию на отложенных историях считает booklm.lab.perplexity (50 000 токенов val.bin, окна 512).
      Печатает перплексию по шагам, совпадение жадного токена с учителем и жадные сказки трёх моделей.
      Около 6 минут; steps=20 — быстрая проверка.
      С save=1 пишет ch08-distill-run.txt и данные рисунка ch08-distill-curves.csv.
  ch08_distill_prune (старое имя prune) — подраздел «Обрезка и дистилляция». Из каждого блока BookLM
      убираем половину нейронов SwiGLU (768 -> 384), оставляя самые важные по среднему |silu(gate) * up|
      на 32 окнах по 256 токенов из train.bin (как в Minitron, только прямые проходы). Затем
      долечиваем обрезанную модель 200 шагами: по тексту и дистилляцией от необрезанной BookLM.
      Около 3 минут. С save=1 пишет ch08-distill-prune.txt.
Без save=1 оба опыта печатают только на экран.

Запуск из корня репозитория:
    python -m booklm.experiments.ch08_distill ch08_distill_students
    python -m booklm.experiments.ch08_distill ch08_distill_students steps=20      # быстрая проверка
    python -m booklm.experiments.ch08_distill ch08_distill_prune save=1          # с записью txt
    python -m booklm.experiments.ch08_distill --help                             # список опытов
Старый вид команды тоже работает: students --steps 20 --save.
"""
import contextlib
import copy
import csv
import io
import math
import sys
import textwrap
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

if not __package__:  # запуск файлом (python booklm/experiments/ch08_distill.py): корень репозитория — в путь поиска
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from booklm.experiments import run as run_experiments   # здесь run — прогон учеников
from booklm.lab import MODELS, generate, load, perplexity, val_tokens
from booklm.model import BookLM, BookLMConfig

OUT = MODELS / "experiments"
MINI = BookLMConfig(d_model=128, layers=2, heads=4, kv_heads=2, d_ff=384)
PROMPT = "Жила-была маленькая девочка по имени Лили."


def batches(steps, bsz=16, T=256, seed=0):
    """Одна и та же последовательность пакетов (x, y) для обоих учеников."""
    data = np.memmap(MODELS / "data" / "train.bin", dtype=np.uint16, mode="r")
    gen = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        ix = torch.randint(len(data) - T - 1, (bsz,), generator=gen).tolist()
        x = torch.stack([torch.from_numpy(data[i:i + T].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(data[i + 1:i + T + 1].astype(np.int64)) for i in ix])
        yield x, y


def student_loss(z_s, z_t, y, tau=2.0, alpha=0.1):
    """(1 - alpha) * tau^2 * KL(softmax(z_t/tau) || softmax(z_s/tau)) + alpha * CE(z_s, y)."""
    log_p_s = (z_s / tau).log_softmax(-1)
    log_p_t = (z_t / tau).log_softmax(-1)
    kl = (log_p_t.exp() * (log_p_t - log_p_s)).sum(-1).mean()
    ce = F.cross_entropy(z_s.flatten(0, 1), y.flatten())
    return (1 - alpha) * tau**2 * kl + alpha * ce


@torch.no_grad()
def top1_agreement(student, teacher, n=50_000, T=512):
    """Доля позиций отложенных историй, где жадный токен ученика совпал с жадным токеном учителя."""
    data = val_tokens(n)
    same, count = 0, 0
    for s in range(0, len(data) - T - 1, T):
        x = data[s:s + T][None]
        same += (student(x).argmax(-1) == teacher(x).argmax(-1)).sum().item()
        count += T
    return same / count


def train(kind, teacher, steps, lr=3e-3, warmup=30, eval_every=50):
    torch.manual_seed(0)                                   # одна инициализация у обоих учеников
    student = BookLM(MINI)
    opt = torch.optim.AdamW(student.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    curve = [(0, perplexity(student.eval()))]
    student.train()
    t0 = time.perf_counter()
    for step, (x, y) in enumerate(batches(steps), start=1):
        frac = (step - 1 - warmup) / max(1, steps - warmup)
        lr_now = lr * step / warmup if step <= warmup else lr * (0.1 + 0.45 * (1 + math.cos(math.pi * frac)))
        for g in opt.param_groups:
            g["lr"] = lr_now
        z_s = student(x)
        if kind == "text":
            loss = F.cross_entropy(z_s.flatten(0, 1), y.flatten())
        else:
            with torch.no_grad():
                z_t = teacher(x)
            loss = student_loss(z_s, z_t, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()
        if step % eval_every == 0 or step == steps:
            curve.append((step, perplexity(student.eval())))
            student.train()
    return student.eval(), curve, time.perf_counter() - t0


def run(steps):
    tok, teacher = load("base")
    teacher.requires_grad_(False)
    torch.manual_seed(0)
    n_mini = BookLM(MINI).num_params()
    n_embed = MINI.vocab * MINI.d_model
    sp = lambda n: f"{n:_}".replace("_", " ")
    print(f"учитель BookLM: {sp(teacher.num_params())} параметров")
    print(f"ученик BookLM-mini: {sp(n_mini)} параметров, из них таблица эмбеддингов {sp(n_embed)}")
    ppl_t = perplexity(teacher)
    print(f"перплексия учителя: {ppl_t:.2f}")
    students, curves, secs = {}, {}, {}
    for kind in ("text", "teacher"):
        students[kind], curves[kind], secs[kind] = train(kind, teacher, steps)
    print("шаг    (а) текст   (б) учитель")
    for (s, a), (_, b) in zip(curves["text"], curves["teacher"]):
        print(f"{s:4d}  {a:10.2f}  {b:12.2f}")
    for kind, name in (("text", "(а) текст"), ("teacher", "(б) учитель")):
        agree = top1_agreement(students[kind], teacher)
        print(f"{name}: {secs[kind] / 60:.1f} мин, жадный токен совпал с учителем в {agree:.1%} позиций")
    long_steps = round(steps * secs["teacher"] / secs["text"], -2)     # столько же минут, сколько у (б)
    _, long_curve, long_secs = train("text", teacher, int(long_steps), eval_every=200)
    print(f"(а) текст, {int(long_steps)} шагов за {long_secs / 60:.1f} мин: перплексия {long_curve[-1][1]:.2f}")
    print("\nЖадные продолжения (temperature=0, 40 новых токенов):")
    for name, model in (("учитель", teacher), ("(а) текст", students["text"]),
                        ("(б) учитель", students["teacher"])):
        text = generate(tok, model, PROMPT, max_new=40, temperature=0)
        print(textwrap.fill(f"{name}: {text}", width=72, subsequent_indent="  "))
    return ppl_t, curves


# ----------------------------------------------------------------------------- обрезка и долечивание

@torch.no_grad()
def ffn_importance(model, calib):
    """Важность каждого нейрона SwiGLU в каждом блоке: среднее |silu(gate) * up| по калибровочным токенам."""
    scores, hooks = [], []
    for block in model.blocks:
        def hook(ffn, inp, out):
            x = inp[0]
            scores.append((F.silu(ffn.gate(x)) * ffn.up(x)).abs().mean((0, 1)))
        hooks.append(block.ffn.register_forward_hook(hook))
    model(calib)
    for h in hooks:
        h.remove()
    return scores                                           # список [d_ff] на каждый блок


def prune_ffn(model, keep):
    """Копия BookLM, где в блоке i оставлены только нейроны keep[i] (строки gate, up и столбцы down)."""
    d_ff = len(keep[0])
    small = BookLM(BookLMConfig(**{**model.cfg.to_dict(), "d_ff": d_ff}))
    state = model.state_dict()
    for i, idx in enumerate(keep):
        name = f"blocks.{i}.ffn."
        state[name + "gate.weight"] = state[name + "gate.weight"][idx]
        state[name + "up.weight"] = state[name + "up.weight"][idx]
        state[name + "down.weight"] = state[name + "down.weight"][:, idx]
    small.load_state_dict(state)
    return small.eval()


def calibration(n=32, T=256):
    data = np.memmap(MODELS / "data" / "train.bin", dtype=np.uint16, mode="r")
    return torch.from_numpy(data[:n * T].astype(np.int64)).view(n, T)


def heal(model, teacher, kind, steps=200, lr=5e-4, eval_every=50):
    """Долечивание обрезанной модели: kind="text" — по тексту, kind="teacher" — у необрезанной BookLM."""
    model = copy.deepcopy(model).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    curve, t0 = [], time.perf_counter()
    for step, (x, y) in enumerate(batches(steps, seed=1), start=1):
        z_s = model(x)
        if kind == "text":
            loss = F.cross_entropy(z_s.flatten(0, 1), y.flatten())
        else:
            with torch.no_grad():
                z_t = teacher(x)
            loss = student_loss(z_s, z_t, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % eval_every == 0:
            curve.append((step, perplexity(model.eval()), time.perf_counter() - t0))
            model.train()
    return curve


def run_prune():
    _, big = load("base")
    big.requires_grad_(False)
    scores = ffn_importance(big, calibration())
    pruned = prune_ffn(big, [s.topk(384).indices.sort().values for s in scores])
    print(f"BookLM: {big.num_params():_} параметров, перплексия {perplexity(big):.2f}".replace("_", " "))
    print(f"обрезанная: {pruned.num_params():_} параметров, перплексия {perplexity(pruned):.2f}".replace("_", " "))
    curves = {kind: heal(pruned, big, kind) for kind in ("text", "teacher")}
    print("шаг   по тексту   у учителя")
    for (s, a, ta), (_, b, tb) in zip(curves["text"], curves["teacher"]):
        print(f"{s:4d}  {a:9.2f}  {b:10.2f}    ({ta:.0f} с и {tb:.0f} с)")


def ch08_distill_students(steps=600, save=0):
    """Два ученика BookLM-mini: (а) учится по тексту, (б) у учителя BookLM; save=1 — записать txt и csv."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ppl_t, curves = run(steps)
    sys.stdout.write(buf.getvalue())
    if save:
        (OUT / "ch08-distill-run.txt").write_text(buf.getvalue(), encoding="utf-8")
        with open(OUT / "ch08-distill-curves.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step", "ppl_text", "ppl_teacher", "ppl_big"])
            for (s, a_), (_, b_) in zip(curves["text"], curves["teacher"]):
                w.writerow([s, f"{a_:.3f}", f"{b_:.3f}", f"{ppl_t:.3f}"])


def ch08_distill_prune(save=0):
    """Обрезка половины нейронов SwiGLU и долечивание по тексту и у учителя; save=1 — записать txt."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        run_prune()
    sys.stdout.write(buf.getvalue())
    if save:
        (OUT / "ch08-distill-prune.txt").write_text(buf.getvalue(), encoding="utf-8")


EXPERIMENTS = {"ch08_distill_students": ch08_distill_students, "students": ch08_distill_students,
               "ch08_distill_prune": ch08_distill_prune, "prune": ch08_distill_prune}

if __name__ == "__main__":
    torch.set_num_threads(min(8, torch.get_num_threads()))
    # Старый вид команды «students --steps 20 --save» переводим в «students steps=20 save=1».
    argv = " ".join(sys.argv[1:]).replace("--steps=", "steps=").replace("--steps ", "steps=")
    run_experiments(EXPERIMENTS, argv.replace("--save", "save=1").split(), args=True)
