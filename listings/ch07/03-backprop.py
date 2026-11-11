"""Листинги главы 7 «Предобучение» книги «Большие языковые модели», файл 03-backprop.

Код и вывод — ровно как в книге; текст книги сюда не входит. Листинги идут по порядку и
опираются друг на друга: запускайте файл целиком или ячейками сверху вниз (разметка # %%).
Проверка, что вывод совпал с книгой: python scripts/check_listings.py listings/ch07/03-backprop.py
"""

# %% Листинг «Обратный проход руками» — раздел «Обратный проход двухслойной сети»
import numpy as np
np.set_printoptions(formatter={"float": lambda v: f"{v + 0.0:7.4f}"})

x = np.array([1.0, 2.0])                         # [2]
W = np.array([[1.0, -1.0], [0.5, 0.5]])          # [2, 2]
U = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 1.0]])  # [3, 2]
y = 0                                            # правильный номер

def forward(W, U, x):
    h = W @ x                                    # [2]
    r = np.maximum(h, 0)                         # ReLU
    a = U @ r                                    # [3] логиты
    return h, r, a, np.log(np.exp(a).sum()) - a[y]

h, r, a, L = forward(W, U, x)
p = np.exp(a) / np.exp(a).sum()
g_a = p - np.eye(3)[y]                           # p - [j = y]
dU = np.outer(g_a, r)                            # [3, 2]: форма U
g_r = U.T @ g_a                                  # [2]
g_h = g_r * (h > 0)                              # ReLU: только h > 0
dW = np.outer(g_h, x)                            # [2, 2]: форма W
print(f"L = {L:.4f}")
print("g_a =", g_a)
print("dU =", dU, sep="\n")
print("dW =", dW, sep="\n")
# Вывод в книге:
# | L = 2.2989
# | g_a = [-0.8996  0.4498  0.4498]
# | dU =
# | [[ 0.0000 -1.3494]
# |  [ 0.0000  0.6747]
# |  [ 0.0000  0.6747]]
# | dW =
# | [[ 0.0000  0.0000]
# |  [ 0.8996  1.7993]]

# %% Листинг «Проверка центральной разностью» — раздел «Обратный проход двухслойной сети»
def numeric_grad(P, loss, eps):                  # повороты ручек
    G = np.zeros_like(P)
    for i in np.ndindex(P.shape):
        old = P[i]
        P[i] = old + eps; up = loss()
        P[i] = old - eps; down = loss()
        P[i] = old
        G[i] = (up - down) / (2 * eps)
    return G

def check(dtype):
    W_, U_, x_ = W.astype(dtype), U.astype(dtype), x.astype(dtype)
    loss = lambda: forward(W_, U_, x_)[3]
    nW = numeric_grad(W_, loss, dtype(1e-6))
    nU = numeric_grad(U_, loss, dtype(1e-6))
    err = max(np.abs(nW - dW).max() / np.abs(dW).max(),
              np.abs(nU - dU).max() / np.abs(dU).max())
    return err, nW

for dtype in (np.float64, np.float32):
    err, nW = check(dtype)
    print(f"{dtype.__name__}: отн. ошибка {err:.1e}, "
          f"dL/dW21 = {nW[1, 0]:.4f} (обратный проход: {dW[1, 0]:.4f})")
# Вывод в книге:
# | float64: отн. ошибка 5.7e-11, dL/dW21 = 0.8996 (обратный проход: 0.8996)
# | float32: отн. ошибка 3.6e-02, dL/dW21 = 0.8345 (обратный проход: 0.8996)

# %% Листинг «Autograd как запись графа: класс Value» — раздел «Автоматическое дифференцирование»
import math

class Value:
    """Скаляр, который помнит, как он вычислен."""
    def __init__(self, data, parents=(), local=()):
        self.data, self.grad = data, 0.0
        self.parents = parents  # входы операции
        self.local = local      # d(выход)/d(вход)

    def __add__(self, o):
        return Value(self.data + o.data, (self, o), (1.0, 1.0))

    def __mul__(self, o):
        o = o if isinstance(o, Value) else Value(o)
        return Value(self.data * o.data, (self, o), (o.data, self.data))

    def exp(self):
        e = math.exp(self.data)
        return Value(e, (self,), (e,))

    def log(self):
        return Value(math.log(self.data), (self,), (1.0 / self.data,))

    def backward(self):
        order, seen = [], set()
        def visit(node):            # входы раньше выходов
            if id(node) not in seen:
                seen.add(id(node))
                for parent in node.parents:
                    visit(parent)
                order.append(node)
        visit(self)
        self.grad = 1.0
        for node in reversed(order):  # от ошибки к листьям
            for parent, d in zip(node.parents, node.local):
                parent.grad += d * node.grad  # сигнал x производная

# %% Листинг «Value на трёх логитах» — раздел «Автоматическое дифференцирование»
logits = [Value(2.0), Value(1.0), Value(0.0)]
total = logits[0].exp() + logits[1].exp() + logits[2].exp()
Lv = logits[0] * -1.0 + total.log()
Lv.backward()
print(f"ошибка {Lv.data:.4f}")
print("градиенты", " ".join(f"{v.grad:.4f}" for v in logits))
# Вывод в книге:
# | ошибка 0.4076
# | градиенты -0.3348 0.2447 0.0900

# %% Листинг «Остаточная связь пропускает сигнал» — раздел «Остаточные связи и связанные веса»
import torch

def input_signal(residual, n_add=12, d=16, seed=0):
    gen = torch.Generator().manual_seed(seed)
    h0 = torch.randn(d, generator=gen, dtype=torch.float64,
                     requires_grad=True)
    h = h0
    for _ in range(n_add):              # 12 прибавлений, как в BookLM
        W_l = torch.randn(d, d, generator=gen,
                          dtype=torch.float64) * 0.5 / math.sqrt(d)
        branch = torch.tanh(W_l @ h)    # tanh сжимает в (-1, 1)
        h = h + branch if residual else branch
    h.sum().backward()                  # сигнал на выходе: норма 4
    return h0.grad.norm().item()        # дошло до первого блока

with_res, without = input_signal(True), input_signal(False)
print(f"с остаточной связью: {with_res:.2f}")
print(f"без неё:             {without:.5f}, доля {without / 4:.1e}")
print(f"оценка 0,5 ** 12:    {0.5**12:.1e}")
# Вывод в книге:
# | с остаточной связью: 4.06
# | без неё:             0.00068, доля 1.7e-04
# | оценка 0,5 ** 12:    2.4e-04

# %% Листинг «Градиент связанной таблицы» — раздел «Остаточные связи и связанные веса»
import torch.nn.functional as F

torch.manual_seed(0)
table = torch.randn(16, 4, requires_grad=True)   # [V, d]: V = 16, d = 4
ids = torch.tensor([3, 7, 7, 1])                 # токены пакета
targets = torch.tensor([7, 7, 1, 5])             # следующие токены

def tied_loss(t_in, t_out):                      # таблица на входе и выходе
    return F.cross_entropy(t_in[ids] @ t_out.T, targets)

tied_loss(table, table).backward()               # одна таблица, две роли
copy = lambda: table.detach().clone().requires_grad_(True)
e_in, e_out = copy(), copy()
tied_loss(e_in, e_out).backward()                # две отдельные копии
rows = lambda g: (g.abs().sum(dim=1) > 0).nonzero().flatten().tolist()
print("ненулевые строки входной копии:", rows(e_in.grad))
print("ненулевых строк выходной копии:", len(rows(e_out.grad)), "из 16")
print("связанная = вход + выход:",
      torch.allclose(table.grad, e_in.grad + e_out.grad))
# Вывод в книге:
# | ненулевые строки входной копии: [1, 3, 7]
# | ненулевых строк выходной копии: 16 из 16
# | связанная = вход + выход: True

# %% Листинг «Сколько операций на вес делают проходы» — раздел «Стоимость обратного прохода»
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode

layer = nn.Linear(256, 768, bias=False)          # матрица SwiGLU BookLM
tokens = torch.randn(512, 256, requires_grad=True)   # окно из 512
with FlopCounterMode(display=False) as fwd:
    out = layer(tokens)
with FlopCounterMode(display=False) as bwd:
    out.sum().backward()
per = 512 * 256 * 768                            # токены x веса
print(f"вперёд: {fwd.get_total_flops() / per:.0f} операции на вес и токен")
print(f"назад:  {bwd.get_total_flops() / per:.0f} операции на вес и токен")
# Вывод в книге:
# | вперёд: 2 операции на вес и токен
# | назад:  4 операции на вес и токен

# %% Листинг «Операции шага BookLM на токен» — раздел «Стоимость обратного прохода»
import sys
sys.path.insert(0, ".")                  # корень репозитория
from torch.nn.attention import sdpa_kernel, SDPBackend
from booklm.model import BookLM, BookLMConfig

booklm = BookLM(BookLMConfig())          # цена от значений весов не зависит
six_n = 6 * booklm.num_params()

def per_token(T):                        # вперёд и назад, на один токен
    ids = torch.randint(0, 8192, (1, T))
    with sdpa_kernel(SDPBackend.MATH), FlopCounterMode(display=False) as fc:
        booklm(ids).sum().backward()     # внимание считаем в лоб
    return fc.get_total_flops() / T

print(f"6N = {six_n}; дальше всё в долях 6N")
for T in (128, 256, 512):
    half = 1 + 36 * T * 256 / six_n      # внимание по T/2 ключам
    print(f"T = {T}: счётчик {per_token(T) / six_n:.3f}, "
          f"оценка по T/2 ключам {half:.3f}")
# Вывод в книге:
# | 6N = 39734784; дальше всё в долях 6N
# | T = 128: счётчик 1.059, оценка по T/2 ключам 1.030
# | T = 256: счётчик 1.118, оценка по T/2 ключам 1.059
# | T = 512: счётчик 1.237, оценка по T/2 ключам 1.119
