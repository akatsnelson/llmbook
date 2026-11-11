"""Глава 10, раздел 10.2 «Обучение в контексте», подраздел «Индукционные головы»: листинг «Индукционный поиск при
замороженных весах» (полная версия).

В книге оставлена сокращённая версия; этот файл самодостаточен и
выполняется в ноутбуке главы notebooks/10_constrained_decoding.ipynb.

Запуск из корня репозитория: python labs/ch10/induction_head.py
"""

import torch

VOCAB = ["кот", "cat", "дом", "house", "кошка", "здание", "kitty"]
ID = {w: i for i, w in enumerate(VOCAB)}
# Веса головы. Ключ строится из предыдущего токена (см. keys ниже).
W_Q = torch.eye(len(VOCAB))   # [V, V]
W_K = torch.eye(len(VOCAB))   # [V, V]
W_V = torch.eye(len(VOCAB))   # [V, V]

def induction_step(tokens, beta=10.0, lam=0.0):
    x = torch.eye(len(VOCAB))[[ID[w] for w in tokens]]  # [T, V]
    t = len(tokens) - 1               # запрос стоит последним
    q = beta * (x[t] @ W_Q)           # [V]
    # Кандидаты j = 1..t (с нуля): ключ x[j-1], значение x[j].
    keys = x[:-1] @ W_K               # [t, V]
    values = x[1:] @ W_V              # [t, V]
    # Штраф давности lam * (t - j); при lam = 0 его нет.
    dist = torch.arange(t - 1, -1, -1, dtype=torch.float)
    scores = keys @ q - lam * dist    # [t]
    shares = scores.softmax(dim=-1)   # [t]
    out = shares @ values             # [V]
    return VOCAB[int(out.argmax())], shares

answer, shares = induction_step(["кот", "cat", "дом", "house", "кот"])
assert answer == "cat"
torch.testing.assert_close(shares.sum(), torch.tensor(1.0))
assert abs(shares[0].item() - 0.99986) < 1e-5

answer, shares = induction_step(["кот", "кошка", "дом", "здание", "кот"])
assert answer == "кошка"
torch.testing.assert_close(shares.sum(), torch.tensor(1.0))
# Те же веса, другие примеры — другой ответ при тех же долях внимания.
assert abs(shares[0].item() - 0.99986) < 1e-5


if __name__ == "__main__":
    print("OK: induction_head.py")
