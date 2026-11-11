"""Глава 18, раздел 18.2 «Инъекция инструкций», подраздел «Маскирование документа»:
листинг «Тест влияния документа: разделитель и маска» (полная версия).

В книге оставлена сокращённая версия; этот файл самодостаточен.

Одна голова BookLM (d_h = 32), окно из восьми позиций: системная инструкция,
разделитель, документ и вопрос. Подменяем текст документа и смотрим, сдвинулся
ли выход последней позиции. Разделитель с любым ключом, даже «громкий»
(ключ равен запросу), влияние документа не убирает. Маска -inf на столбцах
документа убирает его полностью, но выход совпадает с выходом окна, в котором
документа нет вовсе: модель перестаёт подчиняться, потому что перестаёт читать.

Запуск из корня репозитория: python labs/ch18/attention_wall_test.py
"""

import math

import torch


def last_position(q, k, v, blocked=()):
    # q, k, v: [T, d_h]; выход последней позиции, которая видит всех слева.
    scores = k @ q[-1] / math.sqrt(q.size(-1))     # [T]
    if blocked:
        scores[list(blocked)] = float("-inf")       # стена: маска до softmax
    return scores.softmax(dim=-1) @ v               # [d_h]


torch.manual_seed(17)
T, d_h = 8, 32                     # одна голова BookLM, учебная длина окна
SYSTEM, DELIM, DOC, ASK = [0, 1], 2, [3, 4, 5], [6, 7]
q, k, v = (torch.randn(T, d_h) for _ in range(3))

k2, v2 = k.clone(), v.clone()      # тот же контекст, другой текст документа
k2[DOC] = torch.randn(len(DOC), d_h)
v2[DOC] = torch.randn(len(DOC), d_h)


def doc_shift(delimiter_key):
    """Насколько сдвигается выход, если подменить документ при данном ключе разделителя."""
    kd, kd2 = k.clone(), k2.clone()
    kd[DELIM] = kd2[DELIM] = delimiter_key
    return (last_position(q, kd, v) - last_position(q, kd2, v2)).abs().max().item()


# Разделитель на месте, какой бы ключ у него ни был: документ влияет.
shifts = [doc_shift(key) for key in (k[DELIM], torch.zeros(d_h), q[-1])]
for shift in shifts:
    assert shift > 0.05
print("сдвиг от подмены документа при трёх ключах разделителя:",
      [round(s, 3) for s in shifts])

# Самый громкий из трёх: ключ равен запросу, разделитель забирает ~90% внимания.
loud_keys = torch.cat([k[:DELIM], q[-1:], k[DELIM + 1:]])
loud = (loud_keys @ q[-1] / math.sqrt(d_h)).softmax(dim=-1)  # [T]
assert loud[DELIM] > 0.85 and loud[DOC].sum() > 0     # разделителю ~90%, документу не ноль
assert 0.1 < shifts[2] < 0.2                           # сдвиг падает, но остаётся ~0,14
print(f"громкий разделитель: доля {loud[DELIM].item():.3f}, сдвиг {shifts[2]:.3f}")

# Втрое громче: сдвиг порядка 1e-5, но не ноль — softmax без -inf никого не обнуляет.
louder = doc_shift(3 * q[-1])
assert 0 < louder < 1e-3
print(f"втрое громче: сдвиг {louder:.1e}")

# Маска на столбцах документа: влияния нет ни на бит...
walled = last_position(q, k, v, blocked=DOC)
assert torch.equal(walled, last_position(q, k2, v2, blocked=DOC))
# ...но и читать нечего: выход как у окна без документа.
keep = [i for i in range(T) if i not in DOC]
assert torch.allclose(walled, last_position(q[keep], k[keep], v[keep]))
print("маска: влияние документа 0, выход совпадает с окном без документа")
