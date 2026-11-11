"""Глава 8, подраздел 8.5.2 «Ошибка DPO на одной паре»: DPO на табличной политике и поломка, при которой
эталон считается тем же проходом, что и политика (ошибка застывает на ln 2 = 0,6931, таблица не меняется).

В книге этот опыт описан словами; здесь код целиком. Файл самодостаточен и
выполняется в ноутбуке главы notebooks/08_lora_sft.ipynb.
Запуск из корня репозитория: python labs/ch08/dpo_tabular.py
"""

import torch
import torch.nn.functional as F


def dpo_loss(pw, pl, rw, rl, beta):
    # pw, pl: log pi_theta(y_w|x), log pi_theta(y_l|x); rw, rl: то же у эталона
    margin = beta * ((pw - rw) - (pl - rl))
    return -F.logsigmoid(margin).mean(), margin

import math
import torch

torch.manual_seed(8)
Q, K = 8, 4                                   # 8 запросов, 4 варианта ответа
quality = torch.randn(Q, K)                   # скрытое качество (его DPO не видит)
pairs = torch.tensor([(q, i, j) for q in range(Q) for i in range(K)
                      for j in range(K) if quality[q, i] > quality[q, j]])
qi, wi, li = pairs.T                          # 48 пар (x, y_w, y_l)
ref_logits = torch.randn(Q, K)                # эталон: модель после SFT

def train_dpo(steps=200, beta=0.5, frozen_ref=True):
    theta = ref_logits.clone().requires_grad_(True)
    opt = torch.optim.Adam([theta], lr=0.05)
    history = []
    for _ in range(steps):
        logp = theta.log_softmax(dim=-1)
        ref = ref_logits if frozen_ref else theta   # ошибка: эталон обучается
        logp_ref = ref.log_softmax(dim=-1)
        loss, margin = dpo_loss(logp[qi, wi], logp[qi, li],
                                logp_ref[qi, wi], logp_ref[qi, li], beta)
        history.append((loss.item(), margin.mean().item(),
                        (margin > 0).float().mean().item()))
        opt.zero_grad()
        loss.backward()
        opt.step()
    return theta.detach(), history

theta, hist = train_dpo()
pi, pi_ref = theta.softmax(-1), ref_logits.softmax(-1)
kl = (pi * (pi.log() - pi_ref.log())).sum(-1).mean().item()
print(f"ошибка {hist[0][0]:.4f} -> {hist[-1][0]:.4f}, "
      f"отступ {hist[0][1]:.3f} -> {hist[-1][1]:.3f}, KL к эталону {kl:.3f}")
assert abs(hist[0][0] - math.log(2)) < 1e-6   # первый шаг: ln 2
assert hist[-1][2] == 1.0                     # все пары ранжированы верно
assert hist[-1][1] > hist[50][1] > hist[0][1] # средний отступ растёт
assert kl > 0.0

theta_broken, broken = train_dpo(frozen_ref=False)
assert torch.equal(theta_broken, ref_logits)  # таблица не сдвинулась ни на бит
assert all(abs(loss - math.log(2)) < 1e-6 for loss, _, _ in broken)
