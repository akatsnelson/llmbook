"""Глава 7, раздел 7.1 «Один шаг обучения»: полная версия листинга «Один шаг обучения как функция».

Из книги «Большие языковые модели».
В книге оставлены функция шага и проверка, что ошибка падает за пять шагов. Здесь, кроме того,
проверяются формы градиентов, совпадение torch.optim.SGD с формулой theta - eta * g и опыт
с забытым zero_grad: норма градиента второго шага удваивается (1,00 и 2,00).
Тот же опыт на полной BookLM лежит в booklm_init_and_step.py.
Запуск из корня репозитория: python labs/ch07/train_step_tinylm.py
Считается на CPU примерно за секунду.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# --- TinyLM и правило инициализации (как в книге) ---
class RMSNorm(nn.Module):                     # как в главе 4: счёт во float
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight, self.eps = nn.Parameter(torch.ones(d)), eps

    def forward(self, x):
        ms = x.float().pow(2).mean(dim=-1, keepdim=True)
        x32 = x.float() * torch.rsqrt(ms + self.eps)
        return x32.to(x.dtype) * self.weight


class TinyLM(nn.Module):                      # BookLM с layers=0
    def __init__(self, vocab_size=8192, d=256):
        super().__init__()
        self.tokens = nn.Embedding(vocab_size, d)
        self.blocks = nn.ModuleList()         # у BookLM здесь шесть блоков
        self.norm = RMSNorm(d)
        self.output = nn.Linear(d, vocab_size, bias=False)
        self.output.weight = self.tokens.weight      # связанные веса

    def forward(self, ids):                  # [B, T] -> [B, T, V]
        return self.output(self.norm(self.tokens(ids)))


@torch.no_grad()
def init_booklm_(model, std=0.02):
    branch_std = std / math.sqrt(2 * max(len(model.blocks), 1))
    for name, p in model.named_parameters():  # связанная матрица - однажды
        if p.dim() == 1:
            nn.init.ones_(p)                  # усиления RMSNorm
        elif name.endswith(("out.weight", "down.weight")):
            nn.init.normal_(p, std=branch_std)  # W_O, W_down: 0,00577
        else:
            nn.init.normal_(p, std=std)


# --- Один шаг обучения как функция (полная версия листинга) ---
def train_step(model, batch, opt, zero_grad=True):
    inputs, targets = batch                   # 1. пакет [B, T]
    if zero_grad:
        opt.zero_grad(set_to_none=True)       # прошлого g больше нет
    logits = model(inputs)                    # 2. прямой проход [B, T, V]
    loss = F.cross_entropy(logits.flatten(0, 1),   # 3. сравнение с Y
                           targets.flatten())      # 4. одно число
    loss.backward()                           # 5. обратный проход
    opt.step()                                # 6. theta - eta * g
    return loss.item()


def grad_norm(model):
    flat = [p.grad.flatten() for p in model.parameters()]
    return torch.cat(flat).norm().item()


def fresh(lr=0.02):                           # TinyLM из booklm-base-init
    torch.manual_seed(0)
    model = TinyLM()
    init_booklm_(model)
    return model, torch.optim.SGD(model.parameters(), lr=lr)


ids = torch.randint(0, 8192, (4, 129),
                    generator=torch.Generator().manual_seed(1))
batch = (ids[:, :-1], ids[:, 1:])             # X и Y со сдвигом

model, opt = fresh()
losses = [train_step(model, batch, opt) for _ in range(5)]
assert all(p.grad.shape == p.shape for p in model.parameters())
assert all(b < a for a, b in zip(losses, losses[1:]))   # 9,065 -> 8,985
w, before = model.output.weight, model.output.weight.detach().clone()
train_step(model, batch, opt)
assert torch.allclose(w, before - 0.02 * w.grad)        # step = формула

norms = []
for zero in (True, False):                    # второй шаг с обнулением и без
    model, opt = fresh()
    train_step(model, batch, opt)
    train_step(model, batch, opt, zero_grad=zero)
    norms.append(grad_norm(model))            # 1,00 и 2,00
assert 1.8 < norms[1] / norms[0] < 2.3

print(f"ошибки за пять шагов: {losses[0]:.3f} -> {losses[-1]:.3f}")
print(f"норма градиента второго шага: {norms[0]:.2f} с zero_grad, {norms[1]:.2f} без")
print("train_step_tinylm: все проверки пройдены")
