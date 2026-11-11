"""Глава 8, разделы 8.4–8.8 (предпочтения и RLHF, DPO, дистилляция, поломки): проверки к листингам книги.
Проверки к GRPO и остальным разделам главы 9 лежат в labs/ch09/rl_checks.py.

В книге листинги печатают результат; здесь те же вычисления закреплены утверждениями.
Файл самодостаточен.
Запуск из корня репозитория: python labs/ch08/prefs_checks.py
"""

import math

import torch
import torch.nn.functional as F

f64 = lambda *v: torch.tensor(v, dtype=torch.float64)


# --------------------------------------------------------------------- Брэдли–Терри
def bt_loss(r_w, r_l):
    return -F.logsigmoid(r_w - r_l).mean()

assert abs(bt_loss(f64(1.2), f64(0.2)).item() - 0.3133) < 1e-4
assert abs(bt_loss(f64(0.2), f64(1.2)).item() - 1.3133) < 1e-4
assert abs(bt_loss(f64(0.0), f64(0.0)).item() - math.log(2)) < 1e-12
r_w, r_l, c = f64(1.2, -0.3, 2.0), f64(0.2, 0.5, 1.9), f64(10.0, -3.0, 7.0)
assert torch.allclose(bt_loss(r_w, r_l), bt_loss(r_w + c, r_l + c))   # сдвиг c(x) не виден


# --------------------------------------------------------------------- память PPO
P, head = 6_622_464, 256
ppo = 16 * P + 16 * (P + head) + 2 * P + 2 * (P + head)
assert ppo == 238_413_312 and 16 * P == 105_959_424
assert abs(ppo / 2**20 - 227.37) < 0.005
assert 2 * P + 16 * 454_656 + 16 * (P + head) + 2 * (P + head) == 139_728_384   # 133,26 МБ
assert 64 * 512 * (2 * 2 * 32 * 2 * 6) == 48 * 2**20                            # KV-кэш пакета
assert 16 * P + 2 * P == 119_204_352                                            # DPO, 113,68 МБ


# --------------------------------------------------------------------- DPO
def dpo_loss(pw, pl, rw, rl, beta):
    margin = beta * ((pw - rw) - (pl - rl))
    return -F.logsigmoid(margin).mean(), margin

pw, pl = torch.tensor([-12.0], requires_grad=True), torch.tensor([-14.0], requires_grad=True)
loss, margin = dpo_loss(pw, pl, torch.tensor([-12.5]), torch.tensor([-13.5]), beta=0.1)
loss.backward()
assert abs(margin.item() - 0.1) < 1e-6 and abs(loss.item() - 0.6444) < 1e-4
assert pw.grad.item() < 0 < pl.grad.item()
assert abs(pw.grad.item() + 0.1 * torch.sigmoid(torch.tensor(-0.1)).item()) < 1e-6
same = torch.tensor([-12.5])
assert dpo_loss(same, same, same, same, 0.1)[0].item() == F.softplus(torch.tensor(0.0)).item()


# --------------------------------------------------------------------- дистилляция
def distill_loss(z_t, z_s, tau):
    p_t = (z_t / tau).softmax(dim=-1)
    log_p_s = (z_s / tau).log_softmax(dim=-1)
    kl = (p_t * (p_t.log() - log_p_s)).sum(dim=-1)
    return tau ** 2 * kl, kl

teacher, student = f64(2.0, 1.0, 0.0), f64(1.0, 1.0, 0.0)
assert abs(teacher.softmax(-1)[0].item() - 0.6652) < 1e-4
assert abs(distill_loss(teacher, student, 1.0)[1].item() - 0.1196) < 1e-4
loss2, kl2 = distill_loss(teacher, student, 2.0)
assert abs(kl2.item() - 0.0310) < 1e-4 and abs(loss2.item() - 0.1240) < 1e-4
assert distill_loss(teacher, teacher, 2.0)[0].item() == 0.0
lib = F.kl_div((student / 2).log_softmax(-1), (teacher / 2).softmax(-1), reduction="sum")
torch.testing.assert_close(lib, kl2)
norms = []
for tau in (2.0, 4.0, 8.0):
    z_s = student.clone().requires_grad_(True)
    distill_loss(teacher, z_s, tau)[0].backward()
    norms.append(z_s.grad.norm().item())
assert max(norms) / min(norms) < 1.1


# --------------------------------------------------------------------- забывание: четыре варианта
torch.manual_seed(0)
d, n = 16, 512
W_A = torch.randn(d, d) / d**0.5
u, v = torch.randn(d, 1), torch.randn(1, d)
W_B = W_A + 0.5 * u @ v / d**0.5 + 0.05 * torch.randn(d, d)
X_A, X_B, X_0 = torch.randn(n, d), torch.randn(n, d), torch.randn(n, d)
Y_A, Y_B = X_A @ W_A.T, X_B @ W_B.T
mse = lambda W, X, Y: ((X @ W.T - Y) ** 2).mean()

def finetune(X, Y, keep=0.0, rank=0):
    if rank:
        A = (0.1 * torch.randn(rank, d)).requires_grad_()
        B = torch.zeros(d, rank, requires_grad=True)
        params, weight = [A, B], lambda: W_A + B @ A
    else:
        W = W_A.clone().requires_grad_()
        params, weight = [W], lambda: W
    opt = torch.optim.Adam(params, lr=1e-2)
    for _ in range(400):
        loss = mse(weight(), X, Y) + keep * mse(weight(), X_0, X_0 @ W_A.T)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return weight().detach()

k = n // 5
variants = {"полное": finetune(X_B, Y_B),
            "LoRA r=1": finetune(X_B, Y_B, rank=1),
            "добавка": finetune(torch.cat([X_B[k:], X_A[:k]]), torch.cat([Y_B[k:], Y_A[:k]])),
            "штраф": finetune(X_B, Y_B, keep=0.25)}
forget = {nm: mse(W, X_A, Y_A).item() for nm, W in variants.items()}
err_B = {nm: mse(W, X_B, Y_B).item() for nm, W in variants.items()}
others = [nm for nm in variants if nm != "полное"]
assert all(forget["полное"] > forget[nm] > 0 for nm in others)
assert all(err_B["полное"] < err_B[nm] for nm in others)
assert err_B["LoRA r=1"] == max(err_B.values())
assert forget["LoRA r=1"] > max(forget["добавка"], forget["штраф"])


# --------------------------------------------------------------------- доля побед
def win_rate(wins, losses, draws):
    return (wins + draws / 2) / (wins + losses + draws)

first, second = win_rate(58, 30, 12), win_rate(47, 41, 12)
both = win_rate(58 + 47, 30 + 41, 12 + 12)
assert abs(first - 0.64) < 1e-12 and abs(second - 0.53) < 1e-12
assert abs(both - 0.585) < 1e-12 and abs(both - (first + second) / 2) < 1e-12

print("все проверки prefs_checks прошли")
