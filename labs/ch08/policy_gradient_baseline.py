"""Глава 8, раздел 8.4 «Предпочтения и RLHF»:
листинг «Градиент политики с вычитанием ожидания и без него» (полная версия).

В книге оставлена сокращённая версия; этот файл самодостаточен и
выполняется в ноутбуке главы notebooks/08_lora_sft.ipynb.
Запуск из корня репозитория: python labs/ch08/policy_gradient_baseline.py
"""

import torch
import torch.nn.functional as F

r = torch.tensor([3.0, 2.5, 2.0], dtype=torch.float64)   # оценки судьи
theta = torch.zeros(3, dtype=torch.float64, requires_grad=True)
pi = theta.softmax(-1)
(pi * r).sum().backward()                                # точный градиент E[r]
exact = theta.grad.clone()
torch.testing.assert_close(exact, torch.tensor([1 / 6, 0, -1 / 6], dtype=torch.float64))


def per_answer_grads(y, b):
    # y: [N] номера сгенерированных ответов; вклад (r_y - b) * (e_y - pi)
    probs = theta.detach().softmax(-1)
    grad_logp = F.one_hot(y, 3).double() - probs           # [N, 3]
    return (r[y] - b)[:, None] * grad_logp


torch.manual_seed(0)
y = torch.multinomial(pi.detach(), 50_000, replacement=True)   # генерация
expected_r = (pi.detach() * r).sum()                           # 2,5
plain, with_b = per_answer_grads(y, 0.0), per_answer_grads(y, expected_r)
for g in (plain, with_b):                                      # обе оценки верны
    torch.testing.assert_close(g.mean(0), exact, atol=0.03, rtol=0)
var_plain, var_b = plain.var(0).sum().item(), with_b.var(0).sum().item()
print(f"разброс одного вклада: без вычитания {var_plain:.3f}, с вычитанием {var_b:.3f}")
assert var_b < var_plain / 20

# То же через autograd: градиент суммы adv * log pi(y) и есть оценка
theta.grad = None
adv = r[y] - expected_r                                        # преимущество
(adv * theta.log_softmax(-1)[y]).mean().backward()
torch.testing.assert_close(theta.grad, with_b.mean(0))
print("градиент политики сходится к точному:", [round(v, 4) for v in theta.grad.tolist()])
