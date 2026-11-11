"""Глава 7, раздел 7.3 «Обратное распространение ошибки»: полная версия листингов раздела.

Из книги «Большие языковые модели».
Здесь: обратный проход сети из двух матриц руками и проверка центральной разностью
в float64 и float32; класс Value на той же сети и накопление .grad в PyTorch;
сигнал через двенадцать прибавлений с остаточной связью и без, два вклада в градиент
связанной таблицы и подсчёт операций прямого и обратного прохода (6N).
Запуск из корня репозитория: python labs/ch07/backprop_autograd.py
"""

# --- Обратный проход руками и проверка разностью ---
import numpy as np

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
g_a = p - np.eye(3)[y]                           # формула pt-ce-grad
dU = np.outer(g_a, r)                            # [3, 2]: форма U
g_r = U.T @ g_a                                  # [2]
g_h = g_r * (h > 0)                              # ReLU: только h > 0
dW = np.outer(g_h, x)                            # [2, 2]: форма W

assert abs(L - 2.2989) < 1e-4 and abs(g_a.sum()) < 1e-12
assert np.allclose(g_a, [-0.8996, 0.4498, 0.4498], atol=1e-4)
assert np.all(dU[:, 0] == 0)                     # r_1 = 0
assert np.allclose(dW, [[0, 0], [0.8996, 1.7993]], atol=1e-4)

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

err64, _ = check(np.float64)
assert err64 < 1e-7                              # измерено 5.7e-11
# Намеренная поломка: та же проверка во float32.
err32, nW32 = check(np.float32)
assert err32 > 1e-4                              # измерено 0.036
assert abs(nW32[1, 0] - 0.8345) < 1e-3           # вместо 0.8996


# --- Autograd как запись графа: класс Value ---
import math
import torch

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
        return Value(self.data * o.data, (self, o),
                     (o.data, self.data))

    def relu(self):
        return Value(max(self.data, 0.0), (self,),
                     (float(self.data > 0),))

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
        for node in order:          # промежуточные заново,
            if node.parents:        # копится только в листьях
                node.grad = 0.0
        self.grad = 1.0
        for node in reversed(order):  # от ошибки к листьям
            for parent, d in zip(node.parents, node.local):
                parent.grad += d * node.grad  # сигнал x производная

def matvec(M, v):
    return [sum((m * vi for m, vi in zip(row, v)), Value(0.0))
            for row in M]

Wv = [[Value(w) for w in row] for row in W.tolist()]
Uv = [[Value(u) for u in row] for row in U.tolist()]
av = matvec(Uv, [hi.relu() for hi in matvec(Wv, x.tolist())])
Lv = av[0] * -1.0 + sum((ai.exp() for ai in av), Value(0.0)).log()
Lv.backward()
grads = lambda M: np.array([[v.grad for v in row] for row in M])
assert abs(Lv.data - L) < 1e-12
assert np.allclose(grads(Wv), dW) and np.allclose(grads(Uv), dU)

Lv.backward()                                # намеренно: без обнуления
assert np.allclose(grads(Wv), 2 * dW)        # градиент удвоился

# В PyTorch так же: .grad копит сумму, пока его не обнулят.
W_t = torch.tensor(W, requires_grad=True)
loss_t = lambda: -torch.log_softmax(
    torch.tensor(U) @ torch.relu(W_t @ torch.tensor(x)), dim=0)[y]
loss_t().backward()
assert torch.allclose(W_t.grad, torch.tensor(dW))
loss_t().backward()
assert torch.allclose(W_t.grad, 2 * torch.tensor(dW))


# --- Остаточная связь, связанная таблица и цена 6N ---
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode

def input_signal(residual, n_add=12, d=16, seed=0):
    gen = torch.Generator().manual_seed(seed)
    h0 = torch.randn(d, generator=gen, dtype=torch.float64,
                     requires_grad=True)
    h = h0
    for _ in range(n_add):              # 12 прибавлений, как в BookLM
        W_l = torch.randn(d, d, generator=gen,
                          dtype=torch.float64) * 0.5 / math.sqrt(d)
        branch = torch.tanh(W_l @ h)    # уменьшает вектор вдвое
        h = h + branch if residual else branch
    h.sum().backward()                  # сигнал на выходе: норма 4
    return h0.grad.norm().item()        # дошло до первого блока

assert input_signal(residual=True) > 1.0         # измерено 4.06
weak = input_signal(residual=False) / 4          # измерено 1.7e-4
assert 0.5**13 < weak < 0.5**12                  # около 0,5^12 = 2,4e-4

# Связанная таблица: BookLM без блоков, V=16, d=8.
E = torch.randn(16, 8, dtype=torch.float64,
                generator=torch.Generator().manual_seed(0))
inputs, targets = torch.tensor([1, 2, 3]), torch.tensor([2, 3, 4])
def lm_loss(E_in, E_out):
    z = F.rms_norm(E_in[inputs], (8,))  # выбор строк и RMSNorm
    return F.cross_entropy(z @ E_out.T, targets) # логиты [3, 16]
tied = E.clone().requires_grad_()
lm_loss(tied, tied).backward()
E_in, E_out = E.clone().requires_grad_(), E.clone().requires_grad_()
lm_loss(E_in, E_out).backward()
assert torch.allclose(tied.grad, E_in.grad + E_out.grad)
rows = lambda g: set(torch.nonzero(g.norm(dim=1)).flatten().tolist())
assert rows(E_in.grad) == {1, 2, 3}              # только строки пакета
assert rows(E_out.grad) == set(range(16))        # все строки

# Цена: матрица SwiGLU BookLM [256 -> 768] на окне из 512 токенов.
layer = nn.Linear(256, 768, bias=False)
tokens = torch.randn(512, 256, requires_grad=True)
with FlopCounterMode(display=False) as fwd:
    out = layer(tokens)
with FlopCounterMode(display=False) as bwd:
    out.sum().backward()
assert fwd.get_total_flops() == 2 * 512 * 256 * 768  # 2 на вес
assert bwd.get_total_flops() == 2 * fwd.get_total_flops()  # ещё 4
N = 6_622_464                                    # паспорт BookLM
assert 6 * N == 39_734_784
attn = 2 * 6 * 512 * 256                         # внимание, T/2 ключей
assert abs(attn / (2 * N) - 0.119) < 1e-3


print("backprop_autograd: все проверки пройдены")
