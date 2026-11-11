"""Глава 8, раздел 8.8 «Поломки дообучения и их измерение»:
листинг «Забывание на двух линейных задачах» (полная версия).

В книге оставлена сокращённая версия; этот файл самодостаточен и
выполняется в ноутбуке главы notebooks/08_lora_sft.ipynb.
Запуск из корня репозитория: python labs/ch08/forgetting_linear.py
"""

import torch
from torch import nn

torch.manual_seed(0)
d, n = 16, 512                    # ширина и число примеров на задачу
W_A = torch.randn(d, d) / d**0.5  # правило задачи A, [d_out, d_in]
u, v = torch.randn(d, 1), torch.randn(1, d)
W_B = W_A + 0.5 * u @ v / d**0.5 + 0.05 * torch.randn(d, d)  # ~ранг 1
X_A, X_B = torch.randn(n, d), torch.randn(n, d)              # [n, d]
Y_A, Y_B = X_A @ W_A.T, X_B @ W_B.T                          # y = x W^T
X_0 = torch.randn(n, d)           # «нейтральные» входы для L_keep

def mse(model, X, Y):
    with torch.no_grad():
        return ((model(X) - Y) ** 2).mean().item()

def train(model, X, Y, steps=400, lr=1e-2, keep=0.0, ref=None):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = ((model(X) - Y) ** 2).mean()
        if keep > 0:              # штраф за уход от выходов базы
            loss = loss + keep * ((model(X_0) - ref(X_0)) ** 2).mean()
        loss.backward()
        opt.step()
    return model

class LoRALinear(nn.Module):      # коротко, как в разделе о LoRA
    def __init__(self, base, r=1, alpha=1.0):
        super().__init__()
        self.base, self.scale, self.enabled = base, alpha / r, True
        self.base.weight.requires_grad_(False)
        self.A = nn.Parameter(0.1 * torch.randn(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
    def forward(self, x):
        y = self.base(x)
        if self.enabled:
            y = y + self.scale * (x @ self.A.T) @ self.B.T
        return y

base = train(nn.Linear(d, d, bias=False), X_A, Y_A, steps=600)
base.weight.requires_grad_(False)
snapshot = base.weight.detach().clone()
def copy_base():
    m = nn.Linear(d, d, bias=False)
    m.weight.data.copy_(snapshot)
    return m

k = int(0.2 * n)                  # 20 % пакета: примеры задачи A
models = {
    "полное": train(copy_base(), X_B, Y_B),
    "LoRA r=1": train(LoRALinear(copy_base(), r=1), X_B, Y_B),
    "добавка": train(copy_base(), torch.cat([X_B[k:], X_A[:k]]),
                     torch.cat([Y_B[k:], Y_A[:k]])),
    "штраф": train(copy_base(), X_B, Y_B, keep=0.25, ref=base),
}
loss_A0 = mse(base, X_A, Y_A)
forget, err_B = {}, {}
for name, m in models.items():
    forget[name] = mse(m, X_A, Y_A) - loss_A0
    err_B[name] = mse(m, X_B, Y_B)
    print(f"{name:9s} забывание A: {forget[name]:.4f}  "
          f"ошибка B: {err_B[name]:.4f}")

others = [nm for nm in models if nm != "полное"]
assert all(forget["полное"] > forget[nm] > 0 for nm in others)
assert all(err_B["полное"] < err_B[nm] for nm in others)
assert err_B["LoRA r=1"] == max(err_B.values())           # выучила меньше всех
assert forget["LoRA r=1"] > max(forget["добавка"], forget["штраф"])
lora = models["LoRA r=1"]
lora.enabled = False
assert mse(lora, X_A, Y_A) == loss_A0             # база вернулась точно
assert torch.equal(lora.base.weight, snapshot)
