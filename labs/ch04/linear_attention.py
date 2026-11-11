"""Линейное внимание с phi(x) = elu(x) + 1: параллельная и рекуррентная формы и сравнение с softmax.

Полная версия листинга «Линейное внимание: рекуррентно и параллельно» из главы 4 книги
(раздел 4.11 «Линейное внимание»); сверх книги — сравнение с обычным вниманием через softmax.
В книге оставлена сокращённая версия; здесь весь код целиком, с теми же проверками.
Запуск из корня репозитория: python labs/ch04/linear_attention.py
"""

import math

def dot(a, b):
    return sum(x * y for x, y in zip(a, b))

def phi(vector):
    # elu(x) + 1: всегда положительно, как в оригинальной работе
    return [x + 1.0 if x > 0 else math.exp(x) for x in vector]

def linear_attention_parallel(queries, keys, values):
    outputs = []
    for t, q in enumerate(queries):
        fq = phi(q)
        weights = [dot(fq, phi(keys[j])) for j in range(t + 1)]      # причинно
        total = sum(weights)
        outputs.append([sum(w * values[j][c] for j, w in enumerate(weights)) / total
                        for c in range(len(values[0]))])
    return outputs

def linear_attention_recurrent(queries, keys, values):
    d = len(keys[0])
    S = [[0.0] * d for _ in range(d)]    # состояние d x d
    z = [0.0] * d
    outputs = []
    for q, k, v in zip(queries, keys, values):
        fk = phi(k)
        for a in range(d):
            z[a] += fk[a]
            for b in range(d):
                S[a][b] += fk[a] * v[b]  # S += phi(k) v^T
        fq = phi(q)
        numerator = [sum(fq[a] * S[a][b] for a in range(d)) for b in range(d)]
        outputs.append([n / dot(fq, z) for n in numerator])
    return outputs

def softmax_attention(queries, keys, values):
    outputs = []
    for t, q in enumerate(queries):
        scores = [dot(q, keys[j]) / math.sqrt(len(q)) for j in range(t + 1)]
        m = max(scores)
        weights = [math.exp(s - m) for s in scores]
        total = sum(weights)
        outputs.append([sum(w * values[j][c] for j, w in enumerate(weights)) / total
                        for c in range(len(values[0]))])
    return outputs

keys = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
values = [[2.0, 0.0], [0.0, 2.0], [1.0, 1.0]]
queries = [[0.5, 0.5], [1.0, 0.0], [1.0, 2.0]]

parallel = linear_attention_parallel(queries, keys, values)
recurrent = linear_attention_recurrent(queries, keys, values)
for p, r in zip(parallel, recurrent):
    assert all(abs(a - b) < 1e-12 for a, b in zip(p, r))   # две формы совпадают

ordinary = softmax_attention(queries, keys, values)
gap = max(abs(a - b) for p, o in zip(parallel, ordinary) for a, b in zip(p, o))
assert gap > 1e-2                                          # но это не softmax

print("OK: linear_attention.py")
