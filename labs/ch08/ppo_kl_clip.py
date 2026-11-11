"""Глава 8, раздел 8.4 «Предпочтения и RLHF»: листинг «Награда по токенам и отсечение PPO» (полная версия).
В начале файла — проверки дивергенции KL из листинга «Дивергенция Кульбака–Лейблера» (подраздел 8.3.5).

В книге оставлена сокращённая версия; этот файл самодостаточен и
выполняется в ноутбуке главы notebooks/08_lora_sft.ipynb.
Запуск из корня репозитория: python labs/ch08/ppo_kl_clip.py
"""

import torch

f64 = lambda *v: torch.tensor(v, dtype=torch.float64)

import torch

def kl(p, q):
    p = torch.as_tensor(p, dtype=torch.float64)
    q = torch.as_tensor(q, dtype=torch.float64)
    return (p * (p / q).log()).sum().item()

p_pol, q_ref = (0.3, 0.3, 0.4), (0.6, 0.3, 0.1)   # политика и эталон: «вода» выросла
assert abs(kl(p_pol, q_ref) - 0.3466) < 1e-4
assert abs(kl(q_ref, p_pol) - 0.2773) < 1e-4
assert kl(q_ref, q_ref) == 0.0
g = torch.Generator().manual_seed(1)
for _ in range(1000):                              # KL >= 0 на любых парах
    a, b = torch.rand(2, 3, generator=g, dtype=torch.float64) + 1e-3
    assert kl(a / a.sum(), b / b.sum()) >= -1e-15

def kl_shaped_rewards(logp, logp_ref, r_rm, beta):
    # logp, logp_ref: [T] логарифмы вероятностей выбранных токенов ответа
    rewards = -beta * (logp - logp_ref)       # поводок на каждом токене
    rewards[-1] += r_rm                       # оценка судьи: только в конце
    return rewards

logp     = torch.tensor([-1.0, -2.1, -0.5], dtype=torch.float64)
logp_ref = torch.tensor([-1.2, -2.0, -0.9], dtype=torch.float64)
r = kl_shaped_rewards(logp, logp_ref, r_rm=1.0, beta=0.05)
penalties = r - torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
torch.testing.assert_close(penalties, f64(-0.01, 0.005, -0.02))
assert abs(r.sum().item() - 0.975) < 1e-12
assert abs((r.sum() - 0.9).item() - 0.075) < 1e-12   # преимущество при V = 0,9

def ppo_clip_objective(ratio, adv, eps=0.2):
    return torch.minimum(ratio * adv, ratio.clamp(1 - eps, 1 + eps) * adv)

def objective_and_grad(rho, a):
    ratio = torch.tensor(rho, dtype=torch.float64, requires_grad=True)
    obj = ppo_clip_objective(ratio, torch.tensor(a, dtype=torch.float64))
    obj.backward()
    return obj.item(), ratio.grad.item()

for rho, a, want, want_grad in [(1.3, +2.0, 2.4, 0.0),    # выгода обрезана
                                (0.7, -2.0, -1.6, 0.0),   # худшая из двух
                                (1.3, -2.0, -2.6, -2.0),  # штраф целиком
                                (1.0, +2.0, 2.0, 2.0)]:   # первый шаг: rho = 1
    obj, grad = objective_and_grad(rho, a)
    assert abs(obj - want) < 1e-12 and grad == want_grad, (rho, a, obj, grad)


if __name__ == "__main__":
    print("OK: ppo_kl_clip.py")
