"""Глава 9, раздел 9.1 «Проверяемая награда и GRPO»: листинги «Групповые преимущества GRPO и
проверяемая награда» и «Мини-цикл GRPO с проверяющей функцией» (полная версия).

В книге оставлена сокращённая версия; этот файл самодостаточен.
Запуск из корня репозитория: python labs/ch09/grpo_verifier.py
"""

import re
import torch

ANSWER = re.compile(r"Ответ: (-?\d+)")

def verify(text, a, b):
    # Правило: ответ целиком имеет вид Ответ: N, и N == a + b.
    match = ANSWER.fullmatch(text.strip())
    return 1.0 if match and int(match.group(1)) == a + b else 0.0

def grpo_advantages(rewards, eps=1e-6):
    # rewards: [..., G], награды G ответов на один запрос
    mean = rewards.mean(dim=-1, keepdim=True)
    std = rewards.std(dim=-1, correction=0, keepdim=True)   # делим на G
    return (rewards - mean) / (std + eps)

def adv(values):
    return grpo_advantages(torch.tensor(values, dtype=torch.float64))

torch.testing.assert_close(adv([1., 0, 0, 1]), torch.tensor(
    [1., -1, -1, 1], dtype=torch.float64), atol=1e-5, rtol=0)
torch.testing.assert_close(adv([1., 0, 0, 0]), torch.tensor(
    [1.732, -0.577, -0.577, -0.577], dtype=torch.float64), atol=1e-3, rtol=0)
assert torch.equal(adv([0., 0, 0, 0]), torch.zeros(4, dtype=torch.float64))
# преимущества группы в сумме дают ноль: среднее группы и есть «ожидание»
for group in ([1., 0, 0, 1], [1., 0, 0, 0], [0.3, 0.9, 0.1, 0.5]):
    assert abs(adv(group).sum().item()) < 1e-12
assert verify("Ответ: 7", 3, 4) == 1.0
assert verify("Ответ: 7.", 3, 4) == 0.0 and verify("7", 3, 4) == 0.0
assert verify("Ответ: 8", 3, 4) == 0.0

# Мини-цикл GRPO на табличной политике
questions = [(a, b) for a in range(5) for b in range(5)]
def candidates(a, b):
    return [f"Ответ: {a + b + delta}" for delta in (-2, -1, 0, 1, 2)] + [str(a + b)]
cand = [candidates(a, b) for a, b in questions]
reward_table = torch.tensor([[verify(t, a, b) for t in row]
                             for row, (a, b) in zip(cand, questions)])

def pass_rate(logits):
    return (logits.softmax(-1) * reward_table).sum(-1).mean().item()

torch.manual_seed(3)
logits = torch.zeros(len(questions), 6, requires_grad=True)
ref_logp = torch.zeros(len(questions), 6).log_softmax(-1)      # эталон заморожен
opt = torch.optim.Adam([logits], lr=0.1)
G, beta = 4, 0.02
start = pass_rate(logits)
for step in range(150):
    probs = logits.detach().softmax(-1)
    picks = torch.multinomial(probs, G, replacement=True)       # [25, G] ответы
    rewards = torch.tensor([[verify(cand[q][k], *questions[q]) for k in row]
                            for q, row in enumerate(picks.tolist())])
    adv = grpo_advantages(rewards)                              # [25, G]
    logp = logits.log_softmax(-1)
    pg = -(adv * logp.gather(1, picks)).mean()                  # rho = 1: один шаг
    kl = (logp.exp() * (logp - ref_logp)).sum(-1).mean()
    loss = pg + beta * kl
    opt.zero_grad()
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    opt.step()
print(f"доля решённых: {start:.3f} -> {pass_rate(logits):.3f}")
assert abs(start - 1 / 6) < 1e-6 and pass_rate(logits) > 0.9

# Группа из одних нулей не даёт градиента и не даёт NaN
zero_adv = grpo_advantages(torch.zeros(1, G))
probe = torch.zeros(1, 6, requires_grad=True)
(-(zero_adv * probe.log_softmax(-1)[:, :G]).mean()).backward()
assert torch.equal(probe.grad, torch.zeros(1, 6))
