"""Листинги главы 7 «Предобучение» книги «Большие языковые модели», файл 05-precision-stability.

Код и вывод — ровно как в книге; текст книги сюда не входит. Листинги идут по порядку и
опираются друг на друга: запускайте файл целиком или ячейками сверху вниз (разметка # %%).
Проверка, что вывод совпал с книгой: python scripts/check_listings.py listings/ch07/05-precision-stability.py
"""

# %% Листинг «Три формата в PyTorch» — раздел «Числа с плавающей точкой»
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

for name, dt in [("FP32", torch.float32), ("FP16", torch.float16),
                 ("BF16", torch.bfloat16)]:
    f = torch.finfo(dt)
    print(f"{name}: eps = {f.eps:.2e}, max = {f.max:.6g}, "
          f"min = {f.tiny:.3g}")

def bf16_bits(x):                     # знак | порядок | мантисса
    t = torch.tensor(x, dtype=torch.bfloat16)
    s = format(t.view(torch.int16).item() & 0xFFFF, "016b")
    return f"{s[0]} {s[1:9]} {s[9:]}"

print("0.1 в BF16:", bf16_bits(0.1), "=",
      torch.tensor(0.1, dtype=torch.bfloat16).item())
# Вывод в книге:
# | FP32: eps = 1.19e-07, max = 3.40282e+38, min = 1.18e-38
# | FP16: eps = 9.77e-04, max = 65504, min = 6.1e-05
# | BF16: eps = 7.81e-03, max = 3.38953e+38, min = 1.18e-38
# | 0.1 в BF16: 0 01111011 1001101 = 0.10009765625

# %% Листинг «Две поломки 16-битных чисел» — раздел «Числа с плавающей точкой»
one = torch.tensor(1.0)
for name, dt in [("FP32", torch.float32), ("FP16", torch.float16),
                 ("BF16", torch.bfloat16)]:
    print(f"{name}: 1 + 0.001 = {(one + 1e-3).to(dt).item():.6f}")
to16 = lambda v, dt: torch.tensor(v).to(dt).item()
print("FP16: 70000 ->", to16(70000.0, torch.float16),
      "| 1e-8 ->", to16(1e-8, torch.float16))
print(f"BF16: 1e-8 -> {to16(1e-8, torch.bfloat16):.4e}")

w16 = torch.tensor(0.02, dtype=torch.bfloat16)
w32 = torch.tensor(0.02)
for _ in range(100):                  # сто шагов по eta_min = 3e-5
    w16 = w16 - torch.tensor(3e-5, dtype=torch.bfloat16)
    w32 = w32 - 3e-5
print(f"вес после 100 шагов: BF16 {w16.item():.5f}, "
      f"FP32 {w32.item():.5f}")
# Вывод в книге:
# | FP32: 1 + 0.001 = 1.001000
# | FP16: 1 + 0.001 = 1.000977
# | BF16: 1 + 0.001 = 1.000000
# | FP16: 70000 -> inf | 1e-8 -> 0.0
# | BF16: 1e-8 -> 1.0012e-08
# | вес после 100 шагов: BF16 0.02002, FP32 0.01700

# %% Листинг «Сумма в BF16 и умножение матриц в BF16» — раздел «Смешанная точность»
x = torch.full((1000,), 0.01, dtype=torch.bfloat16)
total = torch.tensor(0.0, dtype=torch.bfloat16)
for v in x:                                # копим сумму в BF16
    total = total + v
row = torch.ones(1, 1000, dtype=torch.bfloat16)
print(f"сумма в BF16: {total.item()}, "
      f"сумма во FP32: {x.float().sum().item():.4f}")
print(f"умножение матриц в BF16: {(row @ x.view(1000, 1)).item()}")
# Вывод в книге:
# | сумма в BF16: 4.0, сумма во FP32: 10.0098
# | умножение матриц в BF16: 10.0

# %% Листинг «BookLM во FP32 и в BF16» — раздел «Смешанная точность»
from booklm.lab import load, val_tokens

_, model = load("base")
data = val_tokens(32 * 512 + 1)          # 32 окна отложенных историй
x, y = data[:-1].view(32, 512), data[1:].view(32, 512)
with torch.no_grad():                            # градиент не нужен
    z32 = model(x)                               # всё во FP32
    with torch.autocast("cpu", dtype=torch.bfloat16):
        z16 = model(x).float()                   # умножения в BF16
for name, z in (("FP32", z32), ("BF16", z16)):
    loss = F.cross_entropy(z.flatten(0, 1), y.flatten())
    print(f"{name}: ошибка {loss:.5f}")
print(f"наибольшее расхождение логитов {(z16 - z32).abs().max():.3f}, "
      f"логиты от {z32.min():.1f} до {z32.max():.1f}")
# Вывод в книге:
# | FP32: ошибка 1.94460
# | BF16: ошибка 1.94484
# | наибольшее расхождение логитов 0.476, логиты от -18.6 до 27.7

# %% Листинг «Сколько шагов утонуло бы в BF16» — раздел «Смешанная точность»
def drowned(w, eta):              # доля весов, где шаг eta тонет в BF16
    wb = w.to(torch.bfloat16)
    moved = (wb.float() - eta * wb.float().sign()).to(torch.bfloat16)
    return (moved == wb).float().mean().item()

groups = {"таблица токенов": [], "матрицы блоков": [],
          "усиления RMSNorm": []}
for name, p in model.named_parameters():
    kind = ("таблица токенов" if name == "embed.weight" else
            "матрицы блоков" if p.dim() == 2 else "усиления RMSNorm")
    groups[kind].append(p.detach().flatten())
print(f"{'доля утонувших шагов':<30}{'2e-4':>9} {'3e-5':>11}")
for kind, parts in groups.items():
    w = torch.cat(parts)
    print(f"{kind:<16} |w| ~ {w.abs().median():.3f}: "
          f"{drowned(w, 2e-4):9.1%} {drowned(w, 3e-5):11.1%}")
# Вывод в книге:
# | доля утонувших шагов               2e-4        3e-5
# | таблица токенов  |w| ~ 0.047:     36.6%       91.0%
# | матрицы блоков   |w| ~ 0.035:     23.3%       87.8%
# | усиления RMSNorm |w| ~ 0.974:    100.0%      100.0%

# %% Листинг «Масштабировщик ошибки» — раздел «Масштабирование ошибки»
class DynamicScaler:         # те же поля, что у torch.amp.GradScaler
    def __init__(self, init_scale=2.0**16, growth_interval=2000):
        self.S, self.interval, self.good = init_scale, growth_interval, 0

    def unscale_and_check(self, params):   # True: есть inf или NaN
        found_inf = False
        for p in params:
            p.grad.mul_(1.0 / self.S)      # снимаем масштаб во FP32
            found_inf |= not torch.isfinite(p.grad).all().item()
        return found_inf

    def update(self, found_inf):           # правило масштабировщика
        self.good = 0 if found_inf else self.good + 1
        if found_inf:
            self.S /= 2
        elif self.good == self.interval:
            self.S, self.good = self.S * 2, 0

scaler, p = DynamicScaler(), nn.Parameter(torch.zeros(2))
for packet in (1, 2, 3):
    S = scaler.S
    p.grad = (torch.tensor([2.0, 1e-8]) * S).half().float()  # через FP16
    found_inf = scaler.unscale_and_check([p])
    scaler.update(found_inf)
    print(f"пакет {packet}: S = {S:5.0f}, inf {found_inf!s:5}, "
          f"малый g {p.grad[1].item():.1e}, новый S {scaler.S:.0f}")
# Вывод в книге:
# | пакет 1: S = 65536, inf True , малый g 1.0e-08, новый S 32768
# | пакет 2: S = 32768, inf True , малый g 1.0e-08, новый S 16384
# | пакет 3: S = 16384, inf False, малый g 1.0e-08, новый S 16384

# %% Листинг «batch_loss из раздела 7.1» — раздел «Масштабирование ошибки»
def batch_loss(model, inputs, targets, loss_mask=None):
    logits = model(inputs)
    per_pos = F.cross_entropy(logits.flatten(0, 1), targets.flatten(),
                              reduction="none")
    if loss_mask is None:
        return per_pos.mean()
    m = loss_mask.flatten().to(per_pos.dtype)
    return (per_pos * m).sum() / m.sum()

# %% Листинг «Итоговый шаг обучения» — раздел «Масштабирование ошибки»
def train_step(model, batch, opt, sched=None, clip=None, scaler=None,
               amp=None):
    opt.zero_grad(set_to_none=True)
    device = next(model.parameters()).device.type
    with torch.autocast(device, dtype=amp, enabled=amp is not None):
        loss = batch_loss(model, *batch)          # умножения в amp
    S = 1.0 if scaler is None else scaler.S
    (loss * S).backward()                         # от S * L
    params = [p for p in model.parameters() if p.grad is not None]
    found_inf = scaler is not None and scaler.unscale_and_check(params)
    row = {"loss": loss.item(), "lr": opt.param_groups[0]["lr"],
           "skipped": found_inf}
    if not found_inf:                             # шаг и счётчики тут
        if clip is not None:                      # после снятия масштаба
            row["grad_norm"] = torch.nn.utils.clip_grad_norm_(
                params, clip).item()
        opt.step()
        if sched is not None:
            sched.step()
    if scaler is not None:
        scaler.update(found_inf)
    return row

# %% Листинг «Ограничение нормы до и после снятия масштаба» — раздел «Порядок действий в шаге»
g, S, c = torch.tensor([3.0, 4.0]), 1024.0, 1.0
clip = lambda v: v * min(1.0, c / v.norm().item())
right = clip(g * S / S)                  # снять масштаб, потом ограничить
wrong = clip(g * S) / S                  # ограничить, потом снять масштаб
print(f"правильный порядок: длина {right.norm():.5f}, "
      f"неправильный: {wrong.norm():.5f}")
for name, make in [("SGD  ", lambda w: torch.optim.SGD(w, lr=0.1)),
                   ("AdamW", lambda w: torch.optim.AdamW(w, lr=0.1))]:
    moves = []
    for grad in (right, wrong):
        w = nn.Parameter(torch.zeros(2))
        w.grad = grad.clone()
        make([w]).step()
        moves.append(w.norm().item())
    print(f"{name}: шаг {moves[0]:.5f} при правильном, "
          f"{moves[1]:.5f} при неправильном")
# Вывод в книге:
# | правильный порядок: длина 1.00000, неправильный: 0.00098
# | SGD  : шаг 0.10000 при правильном, 0.00010 при неправильном
# | AdamW: шаг 0.14142 при правильном, 0.14142 при неправильном

# %% Листинг «Накопление градиента и полный пакет» — раздел «Накопление градиента»
IGNORE = -100                          # цель заполнителя: маска 0
torch.manual_seed(1)
ids = torch.randint(0, 16, (2, 4))     # 2 окна по T+1 = 4
inputs, targets = ids[:, :-1].clone(), ids[:, 1:].clone()
targets[1, 1:] = IGNORE                # во втором окне один токен
counts = (targets != IGNORE).sum(1)

def accumulate(model, denom):          # denom(j): на что делить сумму
    model.zero_grad(set_to_none=True)
    for j in range(2):                 # k = 2 микропакета
        logits = model(inputs[j:j + 1]).flatten(0, 1)
        loss = F.cross_entropy(logits, targets[j],
                               ignore_index=IGNORE, reduction="sum")
        (loss / denom(j)).backward()
    return [p.grad.clone() for p in model.parameters()]

torch.manual_seed(0)                   # любая модель: таблица + выход
model = nn.Sequential(nn.Embedding(16, 8), nn.Linear(8, 16))
model.zero_grad(set_to_none=True)
F.cross_entropy(model(inputs).flatten(0, 1), targets.flatten(),
                ignore_index=IGNORE).backward()      # целый пакет
full = [p.grad.clone() for p in model.parameters()]
gap = lambda grads: max((a - b).abs().max().item()
                        for a, b in zip(full, grads))
total = counts.sum()                   # знаменатель известен заранее
print("токенов с целью в микропакетах:", counts.tolist())
print(f"общий знаменатель: расхождение "
      f"{gap(accumulate(model, lambda j: total)):.0e}")
print(f"среднее средних:   расхождение "
      f"{gap(accumulate(model, lambda j: counts[j] * 2)):.0e}")
# Вывод в книге:
# | токенов с целью в микропакетах: [3, 1]
# | общий знаменатель: расхождение 1e-08
# | среднее средних:   расхождение 4e-01
