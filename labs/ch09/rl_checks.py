"""Глава 9: проверки к листингам и ручным расчётам главы (GRPO, агентное RL, дистилляция на ответах
ученика, оценка моделей, калибровка).

В книге листинги печатают результат; здесь те же вычисления закреплены утверждениями.
Файл самодостаточен.
Запуск из корня репозитория: python labs/ch09/rl_checks.py
"""

import math
import random

import torch

f64 = lambda *v: torch.tensor(v, dtype=torch.float64)


# --------------------------------------------------------------------- GRPO: длина растёт сама
def grpo_advantages(rewards, eps=1e-6):
    mean = rewards.mean(dim=-1, keepdim=True)
    std = rewards.std(dim=-1, correction=0, keepdim=True)
    return (rewards - mean) / (std + eps)

for group in ([1., 0, 0, 1], [1., 0, 0, 0], [0.3, 0.9, 0.1, 0.5]):
    assert abs(grpo_advantages(f64(*group)).sum().item()) < 1e-12
assert torch.equal(grpo_advantages(f64(0, 0, 0, 0)), torch.zeros(4, dtype=torch.float64))

torch.manual_seed(4)
lengths = torch.tensor([20.0, 60.0, 120.0, 240.0])
p_correct = torch.tensor([0.3, 0.5, 0.7, 0.8])
strategy = torch.zeros(4, requires_grad=True)
ref_logp = torch.zeros(4).log_softmax(-1)
opt = torch.optim.Adam([strategy], lr=0.05)
mean_len = []
for step in range(200):
    probs = strategy.detach().softmax(-1)
    mean_len.append((probs * lengths).sum().item())
    picks = torch.multinomial(probs.expand(16, 4), 8, replacement=True)
    rewards = torch.bernoulli(p_correct[picks])
    adv = grpo_advantages(rewards)
    logp = strategy.log_softmax(-1)
    kl = (logp.exp() * (logp - ref_logp)).sum()
    loss = -(adv * logp[picks]).mean() + 0.02 * kl
    opt.zero_grad()
    loss.backward()
    opt.step()
solved = (strategy.detach().softmax(-1) * p_correct).sum().item()
assert abs(mean_len[0] - 110.0) < 1e-4
assert mean_len[-1] > mean_len[50] > mean_len[0] and mean_len[-1] > 200
assert solved > 0.75


# --------------------------------------------------------------------- агентное RL: маска, рубрика, TIS
model_tokens = 30 + 20 + 10                       # ходы модели в траектории из 150 токенов
assert model_tokens == 60 and 150 - model_tokens == 90
assert abs((3 + 1) / (3 + 2 + 1 + 1) - 4 / 7) < 1e-12           # награда по рубрике
tis = lambda p_train, p_gen, C=2.0: min(p_train / p_gen, C)
assert abs(tis(0.30, 0.25) - 1.2) < 1e-12 and tis(0.9, 0.3) == 2.0


# --------------------------------------------------------------------- дистилляция на ответах ученика
teacher, student = f64(2.0, 1.0, 0.0), f64(1.0, 1.0, 0.0)
lp_s, lp_t = student.log_softmax(-1), teacher.log_softmax(-1)
per_token = lp_s - lp_t
assert abs(per_token[0].item() + 0.454) < 1e-3 and abs(per_token[1].item() - 0.546) < 1e-3
reverse = (lp_s.exp() * per_token).sum().item()
forward = (lp_t.exp() * (lp_t - lp_s)).sum().item()
assert abs(reverse - 0.1233) < 1e-4 and abs(forward - 0.1196) < 1e-4 and reverse > forward


# --------------------------------------------------------------------- доверительный интервал
se = math.sqrt(0.69 * 0.31 / 100)
assert abs(se - 0.046) < 1e-3 and abs(1.96 * se - 0.09) < 0.005
assert abs(1.96 * math.sqrt(0.69 * 0.31 / 14_000) - 0.008) < 0.0005
d = [0] * 95 + [1, 1, -1, -1, -1]                # парная разность на ста вопросах
rng = random.Random(0)
boot = sorted(sum(rng.choice(d) for _ in range(100)) / 100 for _ in range(2000))
assert boot[50] < sum(d) / 100 < boot[1950]


# --------------------------------------------------------------------- калибровка и штраф за выдумку
ece = 0.5 * abs(0.9 - 0.6) + 0.5 * abs(0.3 - 0.3)
assert abs(ece - 0.15) < 1e-12
ce = lambda q: -0.3 * math.log(q) - 0.7 * math.log(1 - q)
assert abs(ce(0.3) - 0.611) < 1e-3 and abs(ce(0.5) - 0.693) < 1e-3 and abs(ce(0.9) - 1.643) < 1e-3
assert min((ce(q / 1000), q) for q in range(1, 1000))[1] == 300
penalty = lambda t: t / (1 - t)
assert penalty(0.75) == 3.0
assert abs(0.6 - 0.4 * 3 + 0.6) < 1e-12 and abs(0.9 - 0.1 * 3 - 0.6) < 1e-12

print("все проверки rl_checks прошли")
