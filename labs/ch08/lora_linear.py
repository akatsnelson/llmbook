"""LoRALinear: нулевая поправка, путь градиента и ручной пример на двух числах.

Полная версия листингов «Слой LoRALinear» и «Шаг LoRA, слияние и отключение на двух числах» из главы 8,
подраздел 8.2.4 «Шаг обучения на двух числах».
Выполняется в ноутбуке главы 08_lora_sft.ipynb и отдельно.
Запуск из корня репозитория: python labs/ch08/lora_linear.py
"""
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """y = x W^T + (alpha/r)(x A^T) B^T; W [d_out, d_in] заморожена"""

    def __init__(self, base: nn.Linear, r: int, alpha: float, seed=0):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)       # W не учится
        d_out, d_in = base.weight.shape
        g = torch.Generator().manual_seed(seed)
        A = torch.randn(r, d_in, generator=g) / d_in**0.5
        self.A = nn.Parameter(A)                       # [r, d_in]
        self.B = nn.Parameter(torch.zeros(d_out, r))   # [d_out, r]
        self.scale = alpha / r
        self.enabled = True

    def forward(self, x):                            # x: [..., d_in]
        y = self.base(x)                             # [..., d_out]
        if self.enabled:
            y = y + self.scale * (x @ self.A.T) @ self.B.T
        return y

    def merged_weight(self):                         # W + (alpha/r) B A
        return self.base.weight + self.scale * self.B @ self.A


torch.manual_seed(0)
base = nn.Linear(256, 64, bias=False)                # форма W_K BookLM
lora = LoRALinear(base, r=8, alpha=16)
x = torch.randn(5, 256)                              # [N, d_in]

assert torch.equal(lora(x), base(x))                 # шаг 0: бит в бит
lora(x).pow(2).sum().backward()
assert base.weight.grad is None                      # у W градиента нет
assert torch.count_nonzero(lora.A.grad) == 0         # dA = 0, ведь B = 0
assert torch.count_nonzero(lora.B.grad) > 0          # B учится первой

# Ручной пример: W = I, r = alpha = 1, A = [1 1], B = 0
toy = LoRALinear(nn.Linear(2, 2, bias=False), r=1, alpha=1)
with torch.no_grad():
    toy.base.weight.copy_(torch.eye(2))
    toy.A.copy_(torch.tensor([[1.0, 1.0]]))
x, t = torch.tensor([[1.0, 2.0]]), torch.tensor([[1.0, 3.0]])
opt = torch.optim.SGD([toy.A, toy.B], lr=0.1)


def close(a, b):
    torch.testing.assert_close(a, b, atol=1e-6, rtol=0)


loss = 0.5 * (toy(x) - t).pow(2).sum()
loss.backward()
assert abs(loss.item() - 0.5) < 1e-6
assert torch.equal(toy.B.grad, torch.tensor([[0.0], [-3.0]]))
assert torch.equal(toy.A.grad, torch.zeros(1, 2))
opt.step()
opt.zero_grad()

y = toy(x)
close(toy.B, torch.tensor([[0.0], [0.3]]))
close(y, torch.tensor([[1.0, 2.9]]))
assert abs(0.5 * (y - t).pow(2).sum().item() - 0.005) < 1e-6
close(x @ toy.merged_weight().T, y)                  # слияние
toy.enabled = False
assert torch.equal(toy(x), torch.tensor([[1.0, 2.0]]))   # отключили
toy.enabled = True

(0.5 * (toy(x) - t).pow(2).sum()).backward()         # второй шаг
assert torch.count_nonzero(toy.A.grad) == 2          # A ожила
print("LoRALinear: шаг 0 совпал с основой, ручной пример сошёлся")
