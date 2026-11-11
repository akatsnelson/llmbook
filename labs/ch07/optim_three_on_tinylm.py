"""Глава 7, подраздел 7.4.9 «Сравнение оптимизаторов».

Полная версия листинга «Три оптимизатора при трёх масштабах ошибки»: SGD, момент и AdamW на TinyLM,
200 одинаковых пакетов 16 x 32, ошибка умножена на 1, 100 и 0,01. Утверждения проверяют
числа, которые этот листинг печатает в книге. Работает на CPU за несколько секунд.
Запуск из корня репозитория: python labs/ch07/optim_three_on_tinylm.py
"""

import math
import torch
import torch.nn as nn

# --- Три оптимизатора на TinyLM при трёх масштабах ошибки ------------------
import torch.nn.functional as F

class TinyLM(nn.Module):                 # BookLM без блоков
    def __init__(self, V=64, d=32):
        super().__init__()
        self.tokens = nn.Embedding(V, d)
        self.norm = nn.RMSNorm(d, eps=1e-6)
        self.output = nn.Linear(d, V, bias=False)
        self.output.weight = self.tokens.weight   # связанные веса
        nn.init.normal_(self.tokens.weight, std=0.02)

    def forward(self, ids):                       # [B,T] -> [B,T,V]
        return self.output(self.norm(self.tokens(ids)))

inner = torch.tensor([1 / i for i in range(1, 9)])
inner = inner / inner.sum()                       # частоты 1, 1/2, ..., 1/8
best = -(inner * inner.log()).sum().item()
assert abs(best - 1.816) < 1e-3                   # лучшая возможная ошибка

gen = torch.Generator().manual_seed(0)
batches = []
for _ in range(200):                              # 200 пакетов 16 x 32
    ids = torch.empty(16, 33, dtype=torch.long)
    ids[:, 0] = torch.randint(0, 64, (16,), generator=gen)
    for t in range(32):                           # из группы текущего
        pick = torch.multinomial(inner, 16, replacement=True,
                                 generator=gen)
        ids[:, t + 1] = ids[:, t] // 8 * 8 + pick
    batches.append((ids[:, :-1], ids[:, 1:]))

def measure(make_opt, scale):
    torch.manual_seed(0)                          # одна инициализация
    model = TinyLM()
    opt = make_opt(model.parameters())
    losses = []
    for x, y in batches:
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
        opt.zero_grad()
        (scale * loss).backward()                 # масштаб градиента
        opt.step()
        losses.append(loss.item())
    return sum(losses[-20:]) / 20                 # последние 20 шагов

makers = {"SGD": lambda p: torch.optim.SGD(p, lr=0.3),
          "момент": lambda p: torch.optim.SGD(p, lr=0.03, momentum=0.9),
          "AdamW": lambda p: torch.optim.AdamW(p, lr=3e-3)}
table = {name: [measure(make, s) for s in (1.0, 100.0, 0.01)]
         for name, make in makers.items()}
for name in ("SGD", "момент"):
    x1, x100, x001 = table[name]
    assert x1 < best + 0.05                       # 1,844 и 1,837
    assert math.isnan(x100) and x001 > 3.5        # NaN; 3,75 и 3,78
adamw = table["AdamW"]
assert adamw[0] < best + 0.05                     # 1,842
assert max(adamw) - min(adamw) < 1e-3             # масштаб не важен


if __name__ == "__main__":
    print("OK: optim_three_on_tinylm.py")
