"""Листинги главы 8 «Дообучение» книги «Большие языковые модели», файл 05-merging.

Код и вывод — ровно как в книге; текст книги сюда не входит. Листинги идут по порядку и
опираются друг на друга: запускайте файл целиком или ячейками сверху вниз (разметка # %%).
Проверка, что вывод совпал с книгой: python scripts/check_listings.py listings/ch08/05-merging.py
"""

# %% Листинг «Последняя точка, среднее и EMA на шумном спуске» — раздел «Усреднение весов одного прогона»
import torch

torch.manual_seed(0)
d, lr = 100, 0.1
w = torch.full((d,), 3.0)                # старт далеко от минимума w* = 0
ema, tail = w.clone(), torch.zeros(d)
loss = lambda w: 0.5 * w.pow(2).sum().item()
for step in range(1, 1001):
    grad = w + torch.randn(d)            # точный градиент плюс шум пакета
    w = w - lr * grad
    ema = 0.99 * ema + 0.01 * w          # EMA: обновляется на каждом шаге
    if step > 800:
        tail += w / 200                  # среднее последних 200 точек
print(f"последняя точка:          ошибка {loss(w):.3f}")
print(f"EMA с коэффициентом 0,99: ошибка {loss(ema):.3f}")
print(f"среднее 200 последних:    ошибка {loss(tail):.3f}")
# Вывод в книге:
# | последняя точка:          ошибка 2.666
# | EMA с коэффициентом 0,99: ошибка 0.279
# | среднее 200 последних:    ошибка 0.302

# %% Листинг «Перестановка нейронов и среднее двух близнецов» — раздел «Суп моделей и линейная связность»
import copy
import torch.nn as nn

torch.manual_seed(0)
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
perm = torch.randperm(8)                     # новый порядок скрытых нейронов
twin = copy.deepcopy(net)
with torch.no_grad():
    twin[0].weight.copy_(net[0].weight[perm])      # строки первого слоя
    twin[0].bias.copy_(net[0].bias[perm])
    twin[2].weight.copy_(net[2].weight[:, perm])   # столбцы второго
x = torch.randn(1000, 4)
print("близнец считает ту же функцию:", torch.allclose(net(x), twin(x), atol=1e-6))
avg = copy.deepcopy(net)
with torch.no_grad():
    for pa, p1, p2 in zip(avg.parameters(), net.parameters(), twin.parameters()):
        pa.copy_((p1 + p2) / 2)
gap = (avg(x) - net(x)).norm() / net(x).norm()
print(f"среднее двух близнецов отличается от них на {gap:.0%}")
# Вывод в книге:
# | близнец считает ту же функцию: True
# | среднее двух близнецов отличается от них на 52%

# %% Листинг «Прямая от базы BookLM к инструктивной модели» — раздел «Арифметика задач»
from booklm.lab import load, perplexity
from booklm.experiments.ch08 import held_out, story_loss

tok, model = load("base")
W0 = {k: v.clone() for k, v in model.state_dict().items()}
W1 = torch.load("models/booklm/booklm-instruct.pt")      # слитая после LoRA
tau = {k: W1[k].float() - W0[k] for k in W0}              # вектор задачи
pairs = [(r["request"], r["story"]) for r in held_out()]
print("lam   ошибка на ответах   перплексия")
for lam in (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5):
    model.load_state_dict({k: W0[k] + lam * tau[k] for k in W0})
    print(f"{lam:4.2f}  {story_loss(model, tok, pairs):18.4f}  {perplexity(model):11.3f}")
# Вывод в книге:
# | lam   ошибка на ответах   перплексия
# | 0.00              1.7025        6.958
# | 0.25              1.6293        6.951
# | 0.50              1.5842        6.994
# | 0.75              1.5803        7.092
# | 1.00              1.5990        7.258
# | 1.25              1.6343        7.509
# | 1.50              1.6845        7.857

# %% Листинг «TIES и DARE на трёх векторах задач» — раздел «Интерференция векторов задач»
taus = torch.tensor([[0.6, -0.1,  0.0, 0.3, -0.05,  0.4],    # задача 1
                     [-0.4, 0.5,  0.1, 0.1,  0.02, -0.6],    # задача 2
                     [0.3,  0.0, -0.1, 0.2,  0.0,   0.5]])   # задача 3

def ties(taus, k=0.5):
    keep = int(k * taus.size(1))                   # сколько чисел оставить
    thr = taus.abs().topk(keep, dim=1).values[:, -1:]
    trimmed = taus * (taus.abs() >= thr)           # 1) обрезка
    sign = torch.sign(trimmed.sum(0))              # 2) выбор знака по массе
    agree = (torch.sign(trimmed) == sign) & (trimmed != 0)
    return (trimmed * agree).sum(0) / agree.sum(0).clamp(min=1)  # 3) слияние

def dare(tau, p, gen):
    keep = torch.rand(tau.shape, generator=gen) >= p   # выжившие числа
    return tau * keep / (1 - p)

show = lambda v: " ".join(f"{x + 0.0:+.2f}" for x in v)
print("среднее:", show(taus.mean(0)))
print("TIES:   ", show(ties(taus)))
g = torch.Generator().manual_seed(0)
print("DARE p=0,5, одна выборка:", show(dare(taus[0], 0.5, g)))
mean = sum(dare(taus[0], 0.5, g) for _ in range(10000)) / 10000
print("DARE p=0,5, среднее 10000:", show(mean))
# Вывод в книге:
# | среднее: +0.17 +0.13 +0.00 +0.20 -0.01 +0.10
# | TIES:    +0.45 +0.50 +0.00 +0.25 +0.00 +0.45
# | DARE p=0,5, одна выборка: +0.00 -0.20 +0.00 +0.00 +0.00 +0.80
# | DARE p=0,5, среднее 10000: +0.61 -0.10 +0.00 +0.30 -0.05 +0.40

# %% Листинг «Хорда и дуга между двумя матрицами» — раздел «Сферическая интерполяция»
def slerp(a, b, t):
    cos = (a @ b) / (a.norm() * b.norm())
    omega = torch.arccos(cos.clamp(-1, 1))           # угол между векторами
    return (torch.sin((1 - t) * omega) * a + torch.sin(t * omega) * b) / torch.sin(omega)

torch.manual_seed(0)
d = 65_536                                           # одна матрица 256 x 256
base = torch.randn(d) * 0.05
pairs = {"чужие модели": (torch.randn(d) * 0.05, torch.randn(d) * 0.05),
         "две от одной базы": (base + 0.01 * torch.randn(d), base + 0.01 * torch.randn(d))}
for name, (a, b) in pairs.items():
    angle = torch.rad2deg(torch.arccos((a @ b) / (a.norm() * b.norm())))
    lerp = (a + b) / 2
    print(f"{name}: угол {angle:.1f}°, норма середины / норма концов: "
          f"LERP {lerp.norm() / a.norm():.3f}, SLERP {slerp(a, b, 0.5).norm() / a.norm():.3f}")
# Вывод в книге:
# | чужие модели: угол 90.1°, норма середины / норма концов: LERP 0.704, SLERP 0.997
# | две от одной базы: угол 16.0°, норма середины / норма концов: LERP 0.991, SLERP 1.000

# %% Листинг «Среднее двух адаптеров: множители и склейка» — раздел «Слияние адаптеров LoRA»
torch.manual_seed(0)
d, r = 32, 4
B1, A1, B2, A2 = (torch.randn(d, r), torch.randn(r, d),
                  torch.randn(d, r), torch.randn(r, d))
want = (B1 @ A1 + B2 @ A2) / 2                  # среднее двух поправок
naive = ((B1 + B2) / 2) @ ((A1 + A2) / 2)       # среднее множителей
B_cat = torch.cat([B1, B2], dim=1) / 2          # [d, 2r]
A_cat = torch.cat([A1, A2], dim=0)              # [2r, d]
err = lambda M: ((M - want).norm() / want.norm()).item()
print(f"средние B и A по отдельности: ошибка {err(naive):.0%}")
print(f"склейка (cat), ранг {torch.linalg.matrix_rank(B_cat @ A_cat)}: "
      f"ошибка {err(B_cat @ A_cat):.0e}")
# Вывод в книге:
# | средние B и A по отдельности: ошибка 76%
# | склейка (cat), ранг 8: ошибка 6e-08
