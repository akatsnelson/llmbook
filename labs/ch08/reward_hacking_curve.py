"""Глава 8, раздел 8.8 «Поломки дообучения и их измерение»:
листинг «Взлом награды в замкнутом виде» (полная версия).

В книге оставлена сокращённая версия; этот файл самодостаточен и
выполняется в ноутбуке главы notebooks/08_lora_sft.ipynb.
Запуск из корня репозитория: python labs/ch08/reward_hacking_curve.py
"""

import torch

lengths = torch.arange(1, 13, dtype=torch.float64)    # длины 1..12
q = torch.tensor([0.10, 0.35, 0.60, 0.80, 0.95, 1.00,  # качество:
                  0.90, 0.70, 0.50, 0.35, 0.20, 0.10], # кратко: мало,
                 dtype=torch.float64)                   # длинно: вода
r = q + 0.3 * lengths                            # судья любит длину
pi_ref = 0.6 ** lengths
pi_ref = pi_ref / pi_ref.sum()                # эталон пишет коротко

def optimal_policy(beta):             # pi* ~ pi_ref * exp(r / beta)
    return torch.softmax(pi_ref.log() + r / beta, dim=0)

def kl(p, p_ref):
    return (p * (p / p_ref).log()).sum().item()

betas = [8, 4, 2, 1.4, 1, 0.7, 0.5, 0.35, 0.25, 0.18, 0.12, 0.08, 0.05]
print("эталон     KL= 0.00  E[r]={:5.3f}  E[q]={:5.3f}  длина={:5.2f}".format(
    (pi_ref * r).sum().item(), (pi_ref * q).sum().item(),
    (pi_ref * lengths).sum().item()))
rows = []
for beta in betas:
    p = optimal_policy(beta)
    rows.append((beta, kl(p, pi_ref), (p * r).sum().item(),
                 (p * q).sum().item(), (p * lengths).sum().item()))
    print("beta={:<5} KL={:5.2f}  E[r]={:5.3f}  E[q]={:5.3f}  "
          "длина={:5.2f}".format(*rows[-1]))

E_r = [row[2] for row in rows]
E_q = [row[3] for row in rows]
best = max(range(len(betas)), key=E_q.__getitem__)
q_ref = (pi_ref * q).sum().item()
assert all(a < b for a, b in zip(E_r, E_r[1:]))   # оценка растёт всегда
assert 0 < best < len(betas) - 1                    # горб внутри сетки
assert E_q[-1] < E_q[best] and E_q[-1] < q_ref  # в конце хуже эталона
assert torch.allclose(optimal_policy(1e6), pi_ref, atol=1e-5)  # beta -> inf
print("лучшее beta:", betas[best], " E[q] эталона:", round(q_ref, 3))
