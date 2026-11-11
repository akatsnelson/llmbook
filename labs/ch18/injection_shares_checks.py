"""Глава 18, раздел 18.2 «Инъекция инструкций»: проверки чисел из листингов про доли внимания и стену.

Книга печатает доли внимания четырёх источников, отношение «документ : система» при разных
ключах разделителя и таблицу теста влияния (разделитель против маски). Здесь те же расчёты
переписаны утверждениями: если правка кода сдвинет число из книги, файл скажет какое.

Запуск из корня репозитория: python labs/ch18/injection_shares_checks.py   (нужен только PyTorch, CPU)
"""

import math

import torch

# --- доли внимания на четырёх источниках, d_h = 2 ---
query = torch.tensor([1.0, 1.0])
keys = torch.tensor([[1.0, 0.0],    # системная инструкция
                     [0.0, 0.0],    # разделитель EVIDENCE:
                     [2.0, 1.0],    # документ с инъекцией
                     [0.0, 1.0]])   # сама позиция


def shares_for(keys):
    scores = keys @ query / math.sqrt(query.numel())
    return scores, scores.softmax(dim=-1)


scores, shares = shares_for(keys)
assert torch.allclose(scores, torch.tensor([0.7071, 0.0, 2.1213, 0.7071]), atol=1e-4)
assert torch.allclose(shares, torch.tensor([0.151, 0.075, 0.623, 0.151]), atol=1e-3)
assert int(shares.argmin()) == 1 and int(shares.argmax()) == 2
for delimiter_key in ([0.0, 0.0], [3.0, 3.0], [-5.0, -5.0], [9.0, -9.0]):
    other = keys.clone()
    other[1] = torch.tensor(delimiter_key)
    _, a = shares_for(other)
    assert torch.allclose(a[2] / a[0], torch.tensor(math.exp(math.sqrt(2))))  # 4,11
loud = shares_for(torch.tensor([[1.0, 0.0], [3.0, 3.0], [2.0, 1.0], [0.0, 1.0]]))[1]
assert torch.allclose(loud, torch.tensor([0.025, 0.849, 0.102, 0.025]), atol=1e-3)


# --- тест влияния: разделитель с разными ключами против маски на документе ---
def last_position(q, k, v, blocked=()):
    scores = k @ q[-1] / math.sqrt(q.size(-1))
    if blocked:
        scores[list(blocked)] = float("-inf")
    a = scores.softmax(dim=-1)
    return a, a @ v


torch.manual_seed(17)
T, d_h = 8, 32
DELIM, DOC = 2, [3, 4, 5]
q, k, v = (torch.randn(T, d_h) for _ in range(3))
k2, v2 = k.clone(), v.clone()
k2[DOC], v2[DOC] = torch.randn(3, d_h), torch.randn(3, d_h)


def shift(delimiter, blocked=()):
    kd, kd2 = k.clone(), k2.clone()
    kd[DELIM] = kd2[DELIM] = delimiter
    a, out = last_position(q, kd, v, blocked)
    return a, (out - last_position(q, kd2, v2, blocked)[1]).abs().max().item()


for key in (k[DELIM], torch.zeros(d_h), q[-1]):
    assert shift(key)[1] > 0.05                       # документ влияет при любом ключе
a, s = shift(3 * q[-1])
assert 0 < s < 1e-3 and a[DOC].sum() > 0              # громче некуда, но не ноль
a, s = shift(k[DELIM], blocked=DOC)
assert s == 0.0 and a[DOC].sum() == 0                 # маска: ровно ноль
keep = [i for i in range(T) if i not in DOC]
walled = last_position(q, k, v, blocked=DOC)[1]
assert torch.allclose(walled, last_position(q[keep], k[keep], v[keep])[1])

print("injection_shares_checks.py: все числа раздела 18.2 сошлись")
