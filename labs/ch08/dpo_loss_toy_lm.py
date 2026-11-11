"""Глава 8, раздел 8.5 «Прямая оптимизация предпочтений (DPO)»:
листинг «Логарифм вероятности ответа и ошибка DPO» (полная версия).

В книге оставлена сокращённая версия; этот файл самодостаточен и
выполняется в ноутбуке главы notebooks/08_lora_sft.ipynb.
Запуск из корня репозитория: python labs/ch08/dpo_loss_toy_lm.py
"""

import math
import torch
import torch.nn.functional as F

def sequence_logprob(logits, targets, mask):
    # logits: [B, T, V], targets: [B, T], mask: [B, T] (1, если цель из ответа)
    logp = logits.log_softmax(dim=-1)
    token_logp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # [B, T]
    return (token_logp * mask).sum(dim=-1)                            # [B]

def dpo_loss(pw, pl, rw, rl, beta):
    # pw, pl: log pi_theta(y_w|x), log pi_theta(y_l|x); rw, rl: то же у эталона
    margin = beta * ((pw - rw) - (pl - rl))
    return -F.logsigmoid(margin).mean(), margin

# ручной пример
loss, margin = dpo_loss(torch.tensor([-12.0]), torch.tensor([-14.0]),
                        torch.tensor([-12.5]), torch.tensor([-13.5]), beta=0.1)
torch.testing.assert_close(margin, torch.tensor([0.1]))
assert abs(loss.item() - 0.6444) < 1e-4

# знак градиента: шаг спуска поднимает log pi(y_w) и опускает log pi(y_l)
pw, pl = torch.tensor([-12.0], requires_grad=True), torch.tensor([-14.0], requires_grad=True)
dpo_loss(pw, pl, torch.tensor([-12.5]), torch.tensor([-13.5]), beta=0.1)[0].backward()
assert pw.grad.item() < 0 < pl.grad.item()
assert abs(pw.grad.item() + 0.1 * 0.4750) < 1e-4     # -beta * sigma(-u)

# Крошечная причинная модель: состояние позиции = среднее эмбеддингов слева,
# выходная матрица W с LoRA-поправкой scale * B @ A.
torch.manual_seed(0)
V, d, rank = 16, 8, 2
E = torch.randn(V, d, dtype=torch.float64)
W = torch.randn(V, d, dtype=torch.float64)
A = torch.randn(rank, d, dtype=torch.float64) * 0.1
B = torch.zeros(V, rank, dtype=torch.float64)       # LoRA стартует с B = 0

def toy_logits(ids, adapter=True):
    h = E[ids].cumsum(dim=1) / torch.arange(1, ids.size(1) + 1).view(1, -1, 1)
    out = h @ W.T
    if adapter:
        out = out + 2.0 * (h @ A.T) @ B.T
    return out                                       # [B, T, V]

prompt = [3, 7, 9, 4]                               # <|user|> ... <|end|><|assistant|>
y_w, y_l = [11, 12, 5], [13, 5]                     # ответы с <|end|> = 5
seqs = [prompt + y_w, prompt + y_l]
T = max(map(len, seqs))
ids = torch.tensor([s + [0] * (T - len(s)) for s in seqs])        # <|pad|> = 0
inputs, targets = ids[:, :-1], ids[:, 1:]
answer_mask = torch.zeros_like(targets, dtype=torch.float64)
for row, s in enumerate(seqs):
    answer_mask[row, len(prompt) - 1:len(s) - 1] = 1.0  # цели = токены ответа

policy = sequence_logprob(toy_logits(inputs, adapter=True), targets, answer_mask)
ref = sequence_logprob(toy_logits(inputs, adapter=False), targets, answer_mask)
assert torch.equal(policy, ref)                      # B = 0: бит в бит
loss0, _ = dpo_loss(policy[0], policy[1], ref[0], ref[1], beta=0.1)
assert loss0.item() == math.log(2)                   # ровно ln 2, без допуска

# Вклад запроса одинаков в обеих строках и сокращается в отступе:
# позиции запроса не видят ответа (причинность).
full_mask = (targets != 0).to(torch.float64)
B.normal_()                                          # как будто адаптер обучили
pol_a = sequence_logprob(toy_logits(inputs), targets, answer_mask)
pol_f = sequence_logprob(toy_logits(inputs), targets, full_mask)
_, m_answer = dpo_loss(pol_a[0], pol_a[1], ref[0], ref[1], beta=0.1)
ref_f = sequence_logprob(toy_logits(inputs, adapter=False), targets, full_mask)
_, m_full = dpo_loss(pol_f[0], pol_f[1], ref_f[0], ref_f[1], beta=0.1)
torch.testing.assert_close(m_answer, m_full, atol=1e-12, rtol=0)


if __name__ == "__main__":
    print("OK: dpo_loss_toy_lm.py")
