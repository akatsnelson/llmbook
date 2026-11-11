"""Листинги главы 9 «Обучение с подкреплением и оценка моделей» книги «Большие языковые модели», файл 01-grpo.

Код и вывод — ровно как в книге; текст книги сюда не входит. Листинги идут по порядку и
опираются друг на друга: запускайте файл целиком или ячейками сверху вниз (разметка # %%).
Проверка, что вывод совпал с книгой: python scripts/check_listings.py listings/ch09/01-grpo.py
"""

# %% Листинг «Групповые преимущества GRPO и проверяемая награда» — раздел «Групповое преимущество»
import re
import torch

ANSWER = re.compile(r"Ответ: (-?\d+)")

def verify(text, a, b):
    # ответ целиком имеет вид "Ответ: N", и N == a + b
    match = ANSWER.fullmatch(text.strip())
    return 1.0 if match and int(match.group(1)) == a + b else 0.0

def grpo_advantages(rewards, eps=1e-6):
    # rewards: [..., G], награды G ответов на один запрос
    mean = rewards.mean(dim=-1, keepdim=True)
    std = rewards.std(dim=-1, correction=0, keepdim=True)   # делим на G
    return (rewards - mean) / (std + eps)

group = ["Ответ: 5", "Ответ: 6", "5", "Ответ: 5"]            # 2 + 3 = ?
print("награды:", [verify(s, 2, 3) for s in group])
for g in ([1, 0, 0, 1], [1, 0, 0, 0], [0, 0, 0, 0]):
    adv = grpo_advantages(torch.tensor(g, dtype=torch.float64))
    print(f"{g} -> " + " ".join(f"{a:+.3f}" for a in adv.tolist()))
# Вывод в книге:
# | награды: [1.0, 0.0, 0.0, 1.0]
# | [1, 0, 0, 1] -> +1.000 -1.000 -1.000 +1.000
# | [1, 0, 0, 0] -> +1.732 -0.577 -0.577 -0.577
# | [0, 0, 0, 0] -> +0.000 +0.000 +0.000 +0.000

# %% Листинг «Мини-цикл GRPO с проверяющей функцией» — раздел «Цикл GRPO на табличной политике»
questions = [(a, b) for a in range(5) for b in range(5)]
cand = [[f"Ответ: {a + b + d}" for d in (-2, -1, 0, 1, 2)] + [str(a + b)]
        for a, b in questions]                        # 6 вариантов
reward_table = torch.tensor([[verify(text, a, b) for text in row]
                             for row, (a, b) in zip(cand, questions)])

def pass_rate(lg):                    # доля решённых, только для журнала
    return (lg.softmax(-1) * reward_table).sum(-1).mean().item()

torch.manual_seed(3)
logits = torch.zeros(25, 6, requires_grad=True)       # политика
ref_logp = torch.zeros(25, 6).log_softmax(-1)         # эталон заморожен
opt = torch.optim.Adam([logits], lr=0.1)
G, beta = 4, 0.02
for step in range(150):
    if step % 50 == 0:
        print(f"шаг {step:3d}: доля решённых {pass_rate(logits):.3f}")
    probs = logits.detach().softmax(-1)
    picks = torch.multinomial(probs, G, replacement=True)     # [25, G]
    rewards = torch.tensor([[verify(cand[q][k], *questions[q]) for k in ks]
                            for q, ks in enumerate(picks.tolist())])
    adv = grpo_advantages(rewards)
    logp = logits.log_softmax(-1)
    pg = -(adv * logp.gather(1, picks)).mean()        # rho = 1: один шаг
    kl = (logp.exp() * (logp - ref_logp)).sum(-1).mean()
    loss = pg + beta * kl
    opt.zero_grad()
    loss.backward()
    opt.step()
print(f"шаг 150: доля решённых {pass_rate(logits):.3f}")
# Вывод в книге:
# | шаг   0: доля решённых 0.167
# | шаг  50: доля решённых 0.971
# | шаг 100: доля решённых 0.990
# | шаг 150: доля решённых 0.994

# %% Листинг «Длина ответа при GRPO растёт без награды за длину» — раздел «Длина ответа при GRPO»
torch.manual_seed(4)
lengths = torch.tensor([20.0, 60.0, 120.0, 240.0])  # длины стратегий
p_correct = torch.tensor([0.3, 0.5, 0.7, 0.8])      # как часто верна
strategy = torch.zeros(4, requires_grad=True)        # логиты политики
ref_logp = torch.zeros(4).log_softmax(-1)
opt = torch.optim.Adam([strategy], lr=0.05)
for step in range(201):
    probs = strategy.detach().softmax(-1)
    if step % 50 == 0:                                # только журнал
        mean_len = (probs * lengths).sum()
        print(f"шаг {step:3d}: средняя длина {mean_len:5.1f}, "
              f"доля верных {(probs * p_correct).sum():.3f}")
    # 16 запросов, на каждый группа из G = 8 ответов
    picks = torch.multinomial(probs.expand(16, 4), 8, replacement=True)
    rewards = torch.bernoulli(p_correct[picks])      # проверка: 1 или 0
    adv = grpo_advantages(rewards)                   # [16, 8]
    logp = strategy.log_softmax(-1)
    kl_value = (logp.exp() * (logp - ref_logp)).sum()
    loss = -(adv * logp[picks]).mean() + 0.02 * kl_value
    opt.zero_grad()
    loss.backward()
    opt.step()
# Вывод в книге:
# | шаг   0: средняя длина 110.0, доля верных 0.575
# | шаг  50: средняя длина 215.7, доля верных 0.771
# | шаг 100: средняя длина 232.1, доля верных 0.791
# | шаг 150: средняя длина 236.7, доля верных 0.796
# | шаг 200: средняя длина 238.1, доля верных 0.798
