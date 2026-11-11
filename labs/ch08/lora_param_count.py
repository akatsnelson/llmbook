"""Подсчёт параметров и байтов адаптера BookLM.

Полная версия листинга «Подсчёт параметров и байтов адаптера BookLM» из главы 8, подраздел 8.2.5 «Размер
адаптера BookLM»: счёт по формуле и по флагу requires_grad на заглушке с формами BookLM.
Выполняется в ноутбуке главы 08_lora_sft.ipynb и отдельно.
Запуск из корня репозитория: python labs/ch08/lora_param_count.py
"""
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """y = x W^T + (alpha/r)(x A^T) B^T; W [d_out, d_in] заморожена"""

    def __init__(self, base: nn.Linear, r: int, alpha: float, seed=0):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        d_out, d_in = base.weight.shape
        g = torch.Generator().manual_seed(seed)
        self.A = nn.Parameter(torch.randn(r, d_in, generator=g) / d_in**0.5)
        self.B = nn.Parameter(torch.zeros(d_out, r))
        self.scale = alpha / r
        self.enabled = True

    def forward(self, x):
        y = self.base(x)
        if self.enabled:
            y = y + self.scale * (x @ self.A.T) @ self.B.T
        return y


V, D, N_BLOCKS = 8192, 256, 6
SHAPES = {"q": (256, 256), "k": (64, 256), "v": (64, 256),  # (out, in)
          "o": (256, 256), "gate": (768, 256), "up": (768, 256),
          "down": (256, 768)}


def lora_params(r, targets=tuple(SHAPES)):
    return N_BLOCKS * sum(r * sum(SHAPES[n]) for n in targets)


assert lora_params(8) == 227_328
assert lora_params(8, ("q", "v")) == 39_936
assert lora_params(4) == 113_664 and lora_params(16) == 454_656

# Заглушка с формами BookLM: веса случайные, контрольная точка
# главы 7 для подсчёта не нужно
torch.manual_seed(0)
model = nn.ModuleDict({
    "emb": nn.Embedding(V, D),             # связана с выходной матрицей
    "blocks": nn.ModuleList(
        nn.ModuleDict({n: nn.Linear(i, o, bias=False)
                       for n, (o, i) in SHAPES.items()})
        for _ in range(N_BLOCKS)),
})
norms = nn.ParameterList(nn.Parameter(torch.ones(D))
                         for _ in range(2 * N_BLOCKS + 1))
params = list(model.parameters()) + list(norms)
P = sum(p.numel() for p in params)
assert P == 6_622_464

for p in params:
    p.requires_grad_(False)                # вся основа заморожена
for i, block in enumerate(model["blocks"]):
    for j, n in enumerate(SHAPES):
        block[n] = LoRALinear(block[n], r=16, alpha=32, seed=10 * i + j)


def count_trainable(module):
    return sum(p.numel() for p in module.parameters()
               if p.requires_grad)


assert sum(isinstance(m, LoRALinear) for m in model.modules()) == 42
assert count_trainable(model) == 454_656     # таблица эмбеддингов не учится, как в sft_lora.py

full_sft = 16 * P                          # 2 + 4 + 2 + 4 + 4 байта
adapter = lora_params(16)
lora_sft = 2 * P + 16 * adapter            # основа BF16 + адаптер
assert full_sft == 105_959_424 and lora_sft == 20_519_424
print(f"полное SFT {full_sft / 2**20:.2f} МБ, "
      f"LoRA {lora_sft / 2**20:.2f} МБ, "
      f"адаптер в BF16 {2 * adapter / 2**10:.0f} КБ")
