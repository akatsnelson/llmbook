"""Глава 8, раздел 8.4 «Предпочтения и RLHF»: листинг «Ошибка Брэдли–Терри» и модель награды (полная версия).

В книге оставлена сокращённая версия; этот файл самодостаточен и
выполняется в ноутбуке главы notebooks/08_lora_sft.ipynb.
Запуск из корня репозитория: python labs/ch08/reward_model_bt.py
"""

import math
import torch
import torch.nn.functional as F

def bt_loss(r_w, r_l):
    # r_w, r_l: [N] оценки предпочтённого и отвергнутого ответов
    return -F.logsigmoid(r_w - r_l).mean()

f64 = lambda *v: torch.tensor(v, dtype=torch.float64)
assert abs(bt_loss(f64(1.2), f64(0.2)).item() - 0.3133) < 1e-4
assert abs(bt_loss(f64(0.2), f64(1.2)).item() - 1.3133) < 1e-4

delta = f64(1.0).requires_grad_()
(-F.logsigmoid(delta)).sum().backward()
assert abs(delta.grad.item() - (-0.2689)) < 1e-4

# свой сдвиг c(x) для каждого запроса не меняет ошибку
r_w, r_l, c = f64(1.2, -0.3, 2.0), f64(0.2, 0.5, 1.9), f64(10.0, -3.0, 7.0)
assert torch.allclose(bt_loss(r_w, r_l), bt_loss(r_w + c, r_l + c))

# оценку читаем в позиции последнего <|end|> (номер 5 в шаблоне v1)
END = 5
def last_end_index(ids):
    # ids: [B, T] -> [B]
    pos = torch.arange(ids.size(1)).expand_as(ids)
    return torch.where(ids == END, pos, -1).max(dim=1).values

ids = torch.tensor([[3, 40, 41, 5, 4, 70, 71, 5, 0, 0],    # 0: заполнитель
                    [3, 50, 5, 4, 80, 81, 82, 83, 5, 0]])
idx = last_end_index(ids)
assert idx.tolist() == [7, 8]
torch.manual_seed(0)
hidden = torch.randn(2, 10, 256)            # выход остова BookLM: [B, T, d]
head = torch.nn.Linear(256, 1, bias=False)  # голова вместо выходной матрицы
reward = head(hidden[torch.arange(2), idx]).squeeze(-1)    # [B]
assert reward.shape == (2,)
assert 6_622_464 + sum(p.numel() for p in head.parameters()) == 6_622_720

# линейная модель награды на синтетических признаках ответов
torch.manual_seed(8)
def make_pairs(n, d=16):
    f_w, f_l = torch.randn(n, d), torch.randn(n, d)
    f_w[:, 0] = 1.0 + 0.3 * torch.randn(n)  # «есть проверяемое число»
    f_l[:, 0] = 0.0 + 0.3 * torch.randn(n)
    return f_w, f_l

train, held = make_pairs(512), make_pairs(256)
w = torch.zeros(16, requires_grad=True)
assert abs(bt_loss(train[0] @ w, train[1] @ w).item() - math.log(2)) < 1e-6
opt = torch.optim.SGD([w], lr=0.5)
for step in range(300):
    loss = bt_loss(train[0] @ w, train[1] @ w)
    opt.zero_grad()
    loss.backward()
    opt.step()
with torch.no_grad():
    acc = ((held[0] @ w) > (held[1] @ w)).float().mean().item()
print(f"ошибка {loss.item():.3f}, точность на отложенных {acc:.3f}")
assert acc > 0.9
assert w.abs().argmax().item() == 0         # голова нашла нужный признак
