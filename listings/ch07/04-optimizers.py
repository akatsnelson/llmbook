"""Листинги главы 7 «Предобучение» книги «Большие языковые модели», файл 04-optimizers.

Код и вывод — ровно как в книге; текст книги сюда не входит. Листинги идут по порядку и
опираются друг на друга: запускайте файл целиком или ячейками сверху вниз (разметка # %%).
Проверка, что вывод совпал с книгой: python scripts/check_listings.py listings/ch07/04-optimizers.py
"""

# %% Листинг «Градиентный спуск по оврагу» — раздел «Градиентный спуск в овраге»
import math
import torch

def valley_grad(theta):          # f = (x^2 + 25 y^2) / 2;  theta: [2]
    return theta * torch.tensor([1.0, 25.0], dtype=theta.dtype)

def run(lr, beta=0.0, steps=30):  # beta = 0: обычный спуск
    theta = torch.tensor([1.0, 1.0], dtype=torch.float64)
    u, path = torch.zeros(2, dtype=torch.float64), [theta]
    for _ in range(steps):
        u = beta * u + valley_grad(theta)
        theta = theta - lr * u
        path.append(theta)
    return path

def show(path, steps=(1, 2, 3, 30)):
    for s in steps:
        x, y = path[s].tolist()
        print(f"шаг {s:2d}: x = {x:7.4f}, y = {y:8.4f}")

show(run(0.07))
print("lr = 0.085, шаг 30: y =", round(run(0.085)[30][1].item(), 1))
# Вывод в книге:
# | шаг  1: x =  0.9300, y =  -0.7500
# | шаг  2: x =  0.8649, y =   0.5625
# | шаг  3: x =  0.8044, y =  -0.4219
# | шаг 30: x =  0.1134, y =   0.0002
# | lr = 0.085, шаг 30: y = 34.2

# %% Листинг «Метод момента на том же овраге» — раздел «Метод момента»
show(run(0.07, beta=0.5), steps=(1, 2, 3, 10, 20))
x_mom, x_sgd = run(0.07, beta=0.5)[30][0].item(), run(0.07)[30][0].item()
print(f"шаг 30, x: с моментом {x_mom:.4f}, без момента {x_sgd:.4f}")
# Вывод в книге:
# | шаг  1: x =  0.9300, y =  -0.7500
# | шаг  2: x =  0.8299, y =  -0.3125
# | шаг  3: x =  0.7218, y =   0.4531
# | шаг 10: x =  0.2070, y =   0.0339
# | шаг 20: x =  0.0293, y =  -0.0005
# | шаг 30, x: с моментом 0.0041, без момента 0.1134

# %% Листинг «Первый шаг Adam с поправкой смещения и без неё» — раздел «Оптимизатор Adam»
def adamw_step(theta, g, m, v, s, lr, beta1=0.9, beta2=0.95, eps=1e-8,
               wd=0.0, decoupled=True, bias_correction=True):
    # theta, g, m, v: одной формы; s: номер шага, начиная с 1
    if not decoupled:
        g = g + wd * theta            # затухание: следующий подраздел
    m = beta1 * m + (1 - beta1) * g
    v = beta2 * v + (1 - beta2) * g * g
    m_hat, v_hat = m, v
    if bias_correction:
        m_hat, v_hat = m / (1 - beta1 ** s), v / (1 - beta2 ** s)
    update = m_hat / (v_hat.sqrt() + eps)
    if decoupled:
        update = update + wd * theta
    return theta - lr * update, m, v

f64 = dict(dtype=torch.float64)
zero = torch.zeros(2, **f64)
for scale in (1000.0, 0.001):
    g = torch.tensor([scale, -scale], **f64)
    new, _, _ = adamw_step(zero, g, zero, zero, 1, lr=1e-3)
    print(f"g = +-{scale:g}: шаг", [f"{x:+.5f}" for x in new.tolist()])
for beta2 in (0.999, 0.95):
    new, _, _ = adamw_step(zero, g, zero, zero, 1, lr=1e-3, beta2=beta2,
                           bias_correction=False)
    print(f"без поправки, beta2 = {beta2}: "
          f"шаг {abs(new[0].item()) / 1e-3:.2f} eta")
# Вывод в книге:
# | g = +-1000: шаг ['-0.00100', '+0.00100']
# | g = +-0.001: шаг ['-0.00100', '+0.00100']
# | без поправки, beta2 = 0.999: шаг 3.16 eta
# | без поправки, beta2 = 0.95: шаг 0.45 eta

# %% Листинг «Затухание весов двумя способами» — раздел «Затухание весов и AdamW»
two, nothing = torch.tensor([2.0], **f64), torch.zeros(1, **f64)
for wd, decoupled in [(0.1, True), (0.1, False), (0.01, False)]:
    new, _, _ = adamw_step(two, nothing, nothing, nothing, 1, lr=0.01,
                           wd=wd, decoupled=decoupled)
    how = "отдельно    " if decoupled else "через градиент"
    print(f"{how} wd = {wd:<4}: theta = {new.item():.4f}")
# Вывод в книге:
# | отдельно     wd = 0.1 : theta = 1.9980
# | через градиент wd = 0.1 : theta = 1.9900
# | через градиент wd = 0.01: theta = 1.9900

# %% Листинг «Группы параметров BookLM» — раздел «Затухание весов и AdamW»
import sys
sys.path.insert(0, ".")                  # корень репозитория
from booklm.model import BookLM, BookLMConfig

def param_groups(model, weight_decay):
    decay, no_decay = [], []
    for _, p in model.named_parameters():  # общий вес: один раз
        (decay if p.ndim >= 2 else no_decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]

torch.manual_seed(0)
booklm = BookLM(BookLMConfig())
groups = param_groups(booklm, weight_decay=0.1)
for group in groups:
    n = sum(p.numel() for p in group["params"])
    print(f"weight_decay = {group['weight_decay']}: "
          f"{len(group['params']):2d} тензоров, {n:,} чисел")
# Вывод в книге:
# | weight_decay = 0.1: 43 тензоров, 6,619,136 чисел
# | weight_decay = 0.0: 13 тензоров, 3,328 чисел

# %% Листинг «Состояние AdamW у BookLM» — раздел «Состояние оптимизатора»
opt = torch.optim.AdamW(groups, lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
for p in booklm.parameters():
    p.grad = torch.zeros_like(p)
opt.step()

state = opt.state[booklm.embed.weight]
print(sorted(state))
print(tuple(state["exp_avg"].shape), state["exp_avg"].dtype,
      "шаг", int(state["step"]))
nbytes = sum(s[k].numel() * s[k].element_size()
             for s in opt.state.values() for k in ("exp_avg", "exp_avg_sq"))
print(f"m и v: {nbytes:,} байт = {nbytes / 2**20:.1f} МБ")
# Вывод в книге:
# | ['exp_avg', 'exp_avg_sq', 'step']
# | (8192, 256) torch.float32 шаг 1
# | m и v: 52,979,712 байт = 50.5 МБ

# %% Листинг «Выброс градиента в AdamW» — раздел «Ограничение нормы градиента»
def adam_moves(grads, beta2, beta1=0.9, m=1.0, v=1.0):
    # одна координата спокойного прогона: m = v = 1, поправки
    # смещения уже равны 1; возвращает шаги в единицах eta
    moves = []
    for g in grads:
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g * g
        moves.append(m / math.sqrt(v))
    return moves

cases = [("выброс, beta2 = 0.999", [-100.0] + [1.0] * 99, 0.999),
         ("выброс, beta2 = 0.95 ", [-100.0] + [1.0] * 99, 0.95),
         ("укорочен до 1, 0.95  ", [-1.0] + [1.0] * 99, 0.95)]
for name, grads, beta2 in cases:
    moves = adam_moves(grads, beta2)
    print(f"{name}: за 100 шагов прошли {sum(moves):4.1f} из 100")
# Вывод в книге:
# | выброс, beta2 = 0.999: за 100 шагов прошли  0.3 из 100
# | выброс, beta2 = 0.95 : за 100 шагов прошли 13.7 из 100
# | укорочен до 1, 0.95  : за 100 шагов прошли 98.0 из 100

# %% Листинг «Ограничение глобальной нормы» — раздел «Ограничение нормы градиента»
def clip_by_global_norm(grads, c):
    norm = math.sqrt(sum((g.double() ** 2).sum().item() for g in grads))
    return [g * min(1.0, c / norm) for g in grads], norm

for grads in ([torch.tensor([3.0, 4.0])],
              [torch.tensor([0.3]), torch.tensor([0.4, 0.5])]):
    clipped, norm = clip_by_global_norm(grads, c=1.0)
    print(f"норма {norm:.3f} ->", [[round(x, 3) for x in g.tolist()]
                                  for g in clipped])
# Вывод в книге:
# | норма 5.000 -> [[0.6, 0.8]]
# | норма 0.707 -> [[0.3], [0.4, 0.5]]

# %% Листинг «Разогрев и косинус» — раздел «Расписание скорости обучения»
def lr_at(s, warmup, total, lr_max, lr_min):
    # s: номер шага оптимизатора, начиная с 1
    if s <= warmup:
        return lr_max * s / warmup
    progress = (s - warmup) / (total - warmup)
    cos = math.cos(math.pi * progress)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + cos)

sched = dict(warmup=200, total=2000, lr_max=3e-4, lr_min=3e-5)
for s in (1, 100, 200, 1100, 2000):
    print(f"учебный, шаг {s:4d}: {lr_at(s, **sched):.3e}")
long_sched = dict(sched, total=4043)
for s in (1235, 2121, 4043):
    print(f"4043 шага, шаг {s:4d}: {lr_at(s, **long_sched):.3e}")
# Вывод в книге:
# | учебный, шаг    1: 1.500e-06
# | учебный, шаг  100: 1.500e-04
# | учебный, шаг  200: 3.000e-04
# | учебный, шаг 1100: 1.650e-04
# | учебный, шаг 2000: 3.000e-05
# | 4043 шага, шаг 1235: 2.545e-04
# | 4043 шага, шаг 2121: 1.651e-04
# | 4043 шага, шаг 4043: 3.000e-05

# %% Листинг «batch_loss из раздела 7.1» — раздел «Шаг обучения с AdamW»
import torch.nn.functional as F

def batch_loss(model, inputs, targets, loss_mask=None):
    logits = model(inputs)
    per_pos = F.cross_entropy(logits.flatten(0, 1), targets.flatten(),
                              reduction="none")
    if loss_mask is None:
        return per_pos.mean()
    m = loss_mask.flatten().to(per_pos.dtype)
    return (per_pos * m).sum() / m.sum()

# %% Листинг «Шаг обучения с расписанием и ограничением нормы» — раздел «Шаг обучения с AdamW»
def train_step(model, batch, opt, sched=None, clip=None):
    opt.zero_grad(set_to_none=True)
    loss = batch_loss(model, *batch)
    loss.backward()
    row = {"loss": loss.item(), "lr": opt.param_groups[0]["lr"]}
    if clip is not None:                          # норма до ограничения
        row["grad_norm"] = torch.nn.utils.clip_grad_norm_(
            model.parameters(), clip).item()
    opt.step()
    if sched is not None:
        sched.step()                              # скорость следующего шага
    return row

torch.manual_seed(0)
booklm = BookLM(BookLMConfig())
opt = torch.optim.AdamW(param_groups(booklm, 0.1), lr=3e-4,
                        betas=(0.9, 0.95))
lr_lambda = lambda i: lr_at(i + 1, **sched) / sched["lr_max"]
scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
ids = torch.randint(0, 8192, (4, 65))
for s in range(1, 5):
    batch = (ids[:, :-1], ids[:, 1:])
    row = train_step(booklm, batch, opt, scheduler, clip=1.0)
    print(f"шаг {s}: eta = {row['lr']:.2e}, ошибка {row['loss']:.3f}, "
          f"норма до ограничения {row['grad_norm']:.3f}")
# Вывод в книге:
# | шаг 1: eta = 1.50e-06, ошибка 9.062, норма до ограничения 3.763
# | шаг 2: eta = 3.00e-06, ошибка 9.057, норма до ограничения 3.768
# | шаг 3: eta = 4.50e-06, ошибка 9.046, норма до ограничения 3.772
# | шаг 4: eta = 6.00e-06, ошибка 9.031, норма до ограничения 3.765

# %% Листинг «Игрушечный язык с известной лучшей ошибкой» — раздел «Сравнение оптимизаторов»
import torch.nn as nn

inner = torch.tensor([1 / k for k in range(1, 9)])
inner = inner / inner.sum()                   # частоты внутри восьмёрки
gen = torch.Generator().manual_seed(0)
batches = []
for _ in range(200):                          # 200 пакетов 16 x 32
    ids = torch.empty(16, 33, dtype=torch.long)
    ids[:, 0] = torch.randint(0, 64, (16,), generator=gen)
    for t in range(32):
        pick = torch.multinomial(inner, 16, replacement=True, generator=gen)
        ids[:, t + 1] = ids[:, t] // 8 * 8 + pick
    batches.append((ids[:, :-1], ids[:, 1:]))
best = -(inner * inner.log()).sum().item()
print(f"лучшая возможная ошибка {best:.3f}, стартовая около ln 64 = "
      f"{math.log(64):.3f}")
# Вывод в книге:
# | лучшая возможная ошибка 1.816, стартовая около ln 64 = 4.159

# %% Листинг «Три оптимизатора при трёх масштабах ошибки» — раздел «Сравнение оптимизаторов»
def measure(make_opt, scale):
    torch.manual_seed(0)                      # одна инициализация
    tiny = nn.Sequential(nn.Embedding(64, 32), nn.RMSNorm(32, eps=1e-6),
                         nn.Linear(32, 64, bias=False))
    tiny[2].weight = tiny[0].weight           # связанные веса
    nn.init.normal_(tiny[0].weight, std=0.02)
    opt, losses = make_opt(tiny.parameters()), []
    for batch in batches:
        opt.zero_grad()
        loss = batch_loss(tiny, *batch)
        (scale * loss).backward()             # масштаб градиента
        opt.step()
        losses.append(loss.item())
    return sum(losses[-20:]) / 20             # последние 20 шагов

makers = {"SGD    ": lambda p: torch.optim.SGD(p, lr=0.3),
          "момент ": lambda p: torch.optim.SGD(p, lr=0.03, momentum=0.9),
          "AdamW  ": lambda p: torch.optim.AdamW(p, lr=3e-3)}
for name, make in makers.items():
    row = [f"{measure(make, scale):.3f}" for scale in (1.0, 100.0, 0.01)]
    print(f"{name}: x1 {row[0]}, x100 {row[1]}, x0.01 {row[2]}")
# Вывод в книге:
# | SGD    : x1 1.844, x100 nan, x0.01 3.755
# | момент : x1 1.837, x100 nan, x0.01 3.780
# | AdamW  : x1 1.842, x100 1.842, x0.01 1.842

# %% Листинг «Расписание и ограничение нормы в настоящем прогоне» — раздел «Сравнение оптимизаторов»
import json
from booklm.lab import MODELS, training_log

run_info = json.load(open(MODELS / "train_run.json"))     # run.json прогона
a, steps = run_info["args"], run_info["steps"]
ours = (a["lr"], a["lr"] * a["min_lr_frac"], a["warmup"], steps)
for name, (top, low, warm, total) in (
        ("учебный ", (sched["lr_max"], sched["lr_min"],
                      sched["warmup"], sched["total"])),
        ("настоящий", ours)):
    print(f"{name}: eta_max {top:.0e}, eta_min {low:.0e}, "
          f"разогрев {warm}, шагов {total}")

log = [r for r in training_log() if "loss" in r]     # запись на 10 шагов
middle = (a["warmup"] + steps) // 2
for name, lo, hi in (("разогрев       ", 0, a["warmup"]),
                     ("начало спада   ", a["warmup"], middle),
                     ("вторая половина", middle, steps)):
    norms = [r["grad_norm"] for r in log if lo <= r["step"] < hi]
    print(f"{name} (шаги {lo}-{hi - 1}): норма > 1 в "
          f"{sum(g > 1.0 for g in norms)} записях из {len(norms)}")
print("последний раз норма > 1 на шаге",
      max(r["step"] for r in log if r["grad_norm"] > 1.0))
print(f"норма на шаге 0: {log[0]['grad_norm']:.2f}, "
      f"в конце: {log[-1]['grad_norm']:.2f}")
# Вывод в книге:
# | учебный : eta_max 3e-04, eta_min 3e-05, разогрев 200, шагов 2000
# | настоящий: eta_max 2e-03, eta_min 2e-04, разогрев 300, шагов 3051
# | разогрев        (шаги 0-299): норма > 1 в 8 записях из 30
# | начало спада    (шаги 300-1674): норма > 1 в 1 записях из 138
# | вторая половина (шаги 1675-3050): норма > 1 в 0 записях из 138
# | последний раз норма > 1 на шаге 310
# | норма на шаге 0: 2.72, в конце: 0.41

# %% Листинг «Итерация Ньютона–Шульца» — раздел «Оптимизатор Muon»
from booklm.lab import load_model, val_tokens

def newton_schulz(G, steps=5):
    a, b, c = 3.4445, -4.7750, 2.0315
    if G.size(0) > G.size(1):          # считаем на «лежачей» матрице
        return newton_schulz(G.T, steps).T
    X = G / G.norm()                   # все растяжения не больше 1
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    return X

fresh = load_model("init").train()
data = val_tokens(10**9)
starts = torch.randint(len(data) - 257, (8,),
                       generator=torch.Generator().manual_seed(0))
x = torch.stack([data[i:i + 256] for i in starts])
y = torch.stack([data[i + 1:i + 257] for i in starts])
batch_loss(fresh, x, y).backward()
G = fresh.blocks[0].ffn.up.weight.grad         # 768 x 256
s = torch.linalg.svdvals(G)                    # растяжения по убыванию
s = s / s[0]
print(f"градиент, доли наибольшего: 2-е {s[1]:.2f}, 10-е {s[9]:.2f}, "
      f"медиана {s.median():.3f}, наименьшее {s[-1]:.3f}")
s = torch.linalg.svdvals(newton_schulz(G))
print(f"после 5 шагов: наибольшее {s[0]:.2f}, медиана {s.median():.2f}, "
      f"наименьшее {s[-1]:.2f}")
# Вывод в книге:
# | градиент, доли наибольшего: 2-е 0.34, 10-е 0.13, медиана 0.011, наименьшее 0.003
# | после 5 шагов: наибольшее 1.20, медиана 0.93, наименьшее 0.68

# %% Листинг «AdamW и Muon на первых 200 шагах BookLM» — раздел «Оптимизатор Muon»
# Проверка этот листинг не запускает: ему нужны сеть, видеокарта, сервер или внешний файл.
# import time
# from booklm.experiments.ch07_muon_midtraining import (train_stream, window_sampler,
#     train_steps, cosine, make_optimizers)   # Muon: матрицы блоков
#
# fresh = load_model("init")
# n = sum(p.numel() for name, p in fresh.named_parameters()
#         if p.dim() == 2 and name.startswith("blocks."))
# print(f"матрицы блоков: {n:,} параметров из 6,622,464 "
#       f"({n / 6_622_464:.0%})")
# data, val = train_stream(), val_tokens(10**9)
# starts = torch.randint(len(val) - 257, (64,),
#                        generator=torch.Generator().manual_seed(1))
# vx = torch.stack([val[i:i + 256] for i in starts])
# vy = torch.stack([val[i + 1:i + 257] for i in starts])
#
# @torch.no_grad()
# def val_loss(model):                          # 64 окна, по 16 за раз
#     model.eval()
#     l = sum(batch_loss(model, vx[i:i + 16], vy[i:i + 16]).item()
#             for i in range(0, 64, 16)) / 4
#     model.train()
#     return l
#
# runs = {}
# for kind in ("adamw", "muon"):
#     for lr in (1e-3, 2e-3, 4e-3):
#         model = load_model("init").train()
#         opts, log, t0 = make_optimizers(model, kind, lr), [], time.time()
#         train_steps(model, opts, cosine(20, 200), 200,
#                     window_sampler(data, seed=7), log, every=25,
#                     eval_fn=lambda: val_loss(model))
#         runs[kind, lr] = log          # (шаг, ошибка пакета, валидация)
#         print(f"{kind:5s} lr {lr:.0e}: валидация после 200 шагов "
#               f"{log[-1][2]:.3f}, шаг {(time.time() - t0) / 200:.2f} с")
# best = {k: min((lr for kk, lr in runs if kk == k),
#                key=lambda lr: runs[k, lr][-1][2])
#         for k in ("adamw", "muon")}
# print("лучшие:", ", ".join(f"{k} lr {lr:.0e} -> {runs[k, lr][-1][2]:.3f}"
#                            for k, lr in best.items()))
# target = runs["adamw", best["adamw"]][-1][2]
# reach = next(s for s, _, v in runs["muon", best["muon"]] if v <= target)
# print(f"Muon догоняет итог AdamW ({target:.3f}) к шагу {reach}")

# %% Листинг «Диагональное и полное предобусловливание на повёрнутом овраге» — раздел «Предобусловливание»
def rotated_valley(deg):                 # H и старт (1, 1), повёрнутые на deg
    a = math.radians(deg)
    R = torch.tensor([[math.cos(a), -math.sin(a)],
                      [math.sin(a), math.cos(a)]], dtype=torch.float64)
    H = R @ torch.diag(torch.tensor([1.0, 25.0], dtype=torch.float64)) @ R.T
    return H, R @ torch.tensor([1.0, 1.0], dtype=torch.float64)

def steps_to_bottom(H, P, theta, lr=1.0, tol=1e-8):
    P_inv = torch.linalg.inv(P)
    for s in range(1, 10_000):
        theta = theta - lr * P_inv @ (H @ theta)      # формула pt-precond
        if 0.5 * theta @ H @ theta < tol:              # f < 1e-8
            return s

for deg in (0, 45):
    H, start = rotated_valley(deg)
    diag = torch.diag(H.diagonal())                    # только диагональ
    print(f"поворот {deg:2d}: диагональ H {H.diagonal().round().tolist()}, "
          f"шагов с диагональю {steps_to_bottom(H, diag, start)}, "
          f"с полной H {steps_to_bottom(H, H, start)}")
# Вывод в книге:
# | поворот  0: диагональ H [1.0, 25.0], шагов с диагональю 1, с полной H 1
# | поворот 45: диагональ H [13.0, 13.0], шагов с диагональю 132, с полной H 1

# %% Листинг «Shampoo без накопления и с ним» — раздел «Оптимизаторы Shampoo и SOAP»
def inv_root(A, p):                     # A^(-1/p) для симметричной A
    w, Q = torch.linalg.eigh(A)         # собственные числа и векторы
    return Q @ torch.diag(w ** (-1 / p)) @ Q.T

def shampoo_dir(L, G, R):               # формула pt-shampoo без eta
    return inv_root(L, 4) @ G @ inv_root(R, 4)

gen = torch.Generator().manual_seed(0)
G1, G2 = torch.randn(2, 4, 4, dtype=torch.float64, generator=gen)
U, _, Vh = torch.linalg.svd(G1)
step = shampoo_dir(G1 @ G1.T, G1, G1.T @ G1)      # без накопления
print("растяжения шага:",
      [round(x, 4) for x in torch.linalg.svdvals(step).tolist()])
print("совпал с U V^T:", torch.allclose(step, U @ Vh, atol=1e-12))
step10 = shampoo_dir(100 * G1 @ G1.T, 10 * G1, 100 * G1.T @ G1)
print("градиент в 10 раз больше, шаг тот же:",
      torch.allclose(step10, step, atol=1e-12))
L, R = G1 @ G1.T + G2 @ G2.T, G1.T @ G1 + G2.T @ G2  # память о двух шагах
print("с накоплением, растяжения:",
      [round(x, 2) for x in torch.linalg.svdvals(shampoo_dir(L, G2, R)).tolist()])
# Вывод в книге:
# | растяжения шага: [1.0, 1.0, 1.0, 1.0]
# | совпал с U V^T: True
# | градиент в 10 раз больше, шаг тот же: True
# | с накоплением, растяжения: [0.88, 0.68, 0.53, 0.17]

# %% Листинг «Оптимизатор Lion» — раздел «Оптимизаторы Adafactor и Lion»
class Lion(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas,
                                      weight_decay=weight_decay))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            (beta1, beta2), lr = group["betas"], group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                if not self.state[p]:                 # один момент
                    self.state[p]["exp_avg"] = torch.zeros_like(p)
                m = self.state[p]["exp_avg"]
                c = beta1 * m + (1 - beta1) * p.grad  # смесь для шага
                p.mul_(1 - lr * group["weight_decay"])
                p.add_(c.sign(), alpha=-lr)           # каждая координата на lr
                m.mul_(beta2).add_(p.grad, alpha=1 - beta2)

row = [f"{measure(lambda p: Lion(p, lr=1e-3), s):.3f}" for s in (1.0, 100.0, 0.01)]
print(f"Lion   : x1 {row[0]}, x100 {row[1]}, x0.01 {row[2]}")
# Вывод в книге:
# | Lion   : x1 1.843, x100 1.843, x0.01 1.843

# %% Листинг «Память состояния оптимизаторов на BookLM» — раздел «Оптимизаторы Adafactor и Lion»
net = BookLM(BookLMConfig())
mats = [p for n, p in net.named_parameters()       # матрицы блоков
        if n.startswith("blocks.") and p.ndim == 2]
rest = [p for n, p in net.named_parameters()       # таблица и усиления
        if not (n.startswith("blocks.") and p.ndim == 2)]

def state_mib(*opts):                              # один шаг, потом счёт
    for p in net.parameters():
        p.grad = torch.zeros_like(p)
    for o in opts:
        o.step()
    return sum(t.numel() * t.element_size() for o in opts
               for s in o.state.values() for t in s.values()
               if torch.is_tensor(t) and t.numel() > 1) / 2**20

params = lambda: net.parameters()
measured = {
    "SGD с моментом": state_mib(torch.optim.SGD(params(), 1e-3, momentum=0.9)),
    "AdamW": state_mib(torch.optim.AdamW(params())),
    "Lion": state_mib(Lion(params())),
    "Adafactor": state_mib(torch.optim.Adafactor(params())),
    "Muon + AdamW": state_mib(torch.optim.Muon(mats), torch.optim.AdamW(rest))}
rest_adamw = 2 * sum(p.numel() for p in rest)      # AdamW на остальном
shampoo = sum(2 * (a * a + b * b) for a, b in (p.shape for p in mats))
soap = shampoo + 2 * sum(p.numel() for p in mats)  # + m и v в повёрнутых осях
for name, mib in measured.items():
    print(f"{name:15s}: {mib:6.2f} МБ")
for name, n in (("Shampoo + AdamW", shampoo), ("SOAP + AdamW", soap)):
    print(f"{name:15s}: {(n + rest_adamw) * 4 / 2**20:6.2f} МБ по формуле")
# Вывод в книге:
# | SGD с моментом :  25.26 МБ
# | AdamW          :  50.53 МБ
# | Lion           :  25.26 МБ
# | Adafactor      :   0.15 МБ
# | Muon + AdamW   :  33.28 МБ
# | Shampoo + AdamW: 124.40 МБ по формуле
# | SOAP + AdamW   : 158.90 МБ по формуле
