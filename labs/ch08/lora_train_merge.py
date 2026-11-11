"""Обучение, слияние и отключение адаптера.

Полная версия листинга «Слияние, отключение и «разлияние» в BF16» из главы 8, подраздел 8.2.6 «Слияние
и отключение адаптера»: двухслойная сеть ширины 32 учится находить поправку ранга 2 адаптером ранга 4,
затем адаптер сливают в FP32 и в BF16, выключают и пробуют «разлить» обратно.
Выполняется в ноутбуке главы 08_lora_sft.ipynb и отдельно.
Запуск из корня репозитория: python labs/ch08/lora_train_merge.py
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

    def merged_weight(self):
        return self.base.weight + self.scale * self.B @ self.A


def count_trainable(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


torch.manual_seed(1)
d, r_true = 32, 2


def lin():
    return nn.Linear(d, d, bias=False)


net = nn.Sequential(LoRALinear(lin(), r=4, alpha=8, seed=1), nn.SiLU(),
                    LoRALinear(lin(), r=4, alpha=8, seed=2))
layers = [net[0], net[2]]
W_before = [m.base.weight.clone() for m in layers]
# Учитель: та же основа плюс поправка ранга 2 в каждом слое
teacher = [m.base.weight + 0.3 * torch.randn(d, r_true)
           @ torch.randn(r_true, d) / d**0.5 for m in layers]


def target(x):
    return nn.functional.silu(x @ teacher[0].T) @ teacher[1].T


def plain(W1, W2):                         # сеть с весами W1, W2
    m = nn.Sequential(lin(), nn.SiLU(), lin())
    with torch.no_grad():
        m[0].weight.copy_(W1)
        m[2].weight.copy_(W2)
    return m


X_train, X_test = torch.randn(512, d), torch.randn(256, d)
trainable = [p for p in net.parameters() if p.requires_grad]
assert sum(p.numel() for p in trainable) == 2 * 4 * (d + d)  # A и B
opt = torch.optim.AdamW(trainable, lr=1e-2, weight_decay=0.0)


def mse(m, X):
    return (m(X) - target(X)).pow(2).mean()


loss_0 = mse(net, X_test).item()
for step in range(200):
    loss = mse(net, X_train)
    opt.zero_grad()
    loss.backward()
    opt.step()
loss_1 = mse(net, X_test).item()
print(f"ошибка на отложенных: {loss_0:.4f} -> {loss_1:.2e}")
assert loss_1 < loss_0 / 10
assert all(torch.equal(m.base.weight, w)
           for m, w in zip(layers, W_before))   # W не сдвинулась

with torch.no_grad():
    y_adapter = net(X_test)
    merged = plain(net[0].merged_weight(), net[2].merged_weight())
    torch.testing.assert_close(merged(X_test), y_adapter,
                               atol=1e-5, rtol=0)   # слияние в FP32

    for m in layers:
        m.enabled = False                       # выключили адаптер
    y_off = net(X_test)
    for m in layers:
        m.enabled = True
    assert torch.equal(y_off, plain(*W_before)(X_test))  # бит в бит

    merged_bf16 = merged.to(torch.bfloat16)    # слили и округлили
    y_bf16 = merged_bf16(X_test.bfloat16()).float()
    gap = (y_bf16 - y_adapter).abs().max().item()
    print(f"слитая BF16 против адаптерного режима: {gap:.4f}")
    assert gap < 2e-2

    W = net[0].base.weight.bfloat16()          # основа в BF16
    dW = (net[0].scale * net[0].B @ net[0].A).bfloat16()
    W_back = (W + dW) - dW                     # «разлили»
    share = (W_back != W).float().mean().item()
    print(f"после разлияния не совпало {share:.0%} весов")
    assert not torch.equal(W_back, W)

# Намеренно сломанный вариант: забыли заморозить W
broken = LoRALinear(lin(), r=4, alpha=8)
broken.base.weight.requires_grad_(True)
n_broken = count_trainable(broken)
assert n_broken != 4 * (d + d), "счётчик ловит размороженную основу"
print(f"ожидали {4 * (d + d)} обучаемых, а учится {n_broken}")
