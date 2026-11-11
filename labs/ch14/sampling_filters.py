"""Три фильтра хвоста распределения: top-k, top-p (ядерная выборка) и min-p.

Дополнение к разделу 14.3 «От логитов к токену» главы 14. В книге правила описаны формулами на строке
логитов с рисунка 14.2; здесь все три правила отдельными функциями, общая перенормировка и проверка
частот на 20 000 выборках.

Запуск из корня репозитория: python labs/ch14/sampling_filters.py
"""

import torch

# последняя строка логитов с рисунка 14.2: позиция T, форма [V]
z = torch.tensor([1.03, -0.22, 2.08, 0.19, -0.91, 0.59, 1.49, -0.51])
probs = torch.softmax(z, dim=-1)        # [V], сумма 1


def top_k_mask(p, k):
    keep = torch.zeros_like(p, dtype=torch.bool)
    keep[torch.topk(p, k).indices] = True
    return keep


def top_p_mask(p, top_p):
    sorted_p, order = torch.sort(p, descending=True)
    # сколько вероятности набрано до этого токена
    before = torch.cumsum(sorted_p, dim=0) - sorted_p
    keep = torch.zeros_like(p, dtype=torch.bool)
    # минимальный набор с суммой не меньше top_p
    keep[order] = before < top_p
    return keep


def min_p_mask(p, p_min):
    return p >= p_min * p.max()


def renormalize(p, keep):
    q = torch.where(keep, p, torch.zeros_like(p))
    return q / q.sum()


masks = {"top_k": top_k_mask(probs, 3),
         "top_p": top_p_mask(probs, 0.9),
         "min_p": min_p_mask(probs, 0.2)}
assert [int(m.sum()) for m in masks.values()] == [3, 5, 4]
for keep in masks.values():
    q = renormalize(probs, keep)
    assert torch.isclose(q.sum(), torch.tensor(1.0))
    assert (q[~keep] == 0).all()
q_k = renormalize(probs, masks["top_k"])
expected = torch.tensor([0.53, 0.29, 0.18])
assert torch.allclose(q_k[[2, 6, 0]], expected, atol=0.01)

torch.manual_seed(0)
q_p = renormalize(probs, masks["top_p"])
n = 20_000
draws = torch.multinomial(q_p, n, replacement=True)
freq = torch.bincount(draws, minlength=len(probs)).float() / n
# частоты выборки совпадают с перенормированными вероятностями в пределах четырёх сигм;
# с исходными probs (без отсечения хвоста) та же проверка падает
sigma = (q_p * (1 - q_p) / n).sqrt().clamp(min=1e-4)
assert ((freq - q_p).abs() <= 4 * sigma).all(), (freq, q_p)
assert not ((freq - probs).abs() <= 4 * (probs * (1 - probs) / n).sqrt()).all()

# Задание «Остановись и ответь»: ядерная выборка с p = 0,8 на том же распределении
assert int(top_p_mask(probs, 0.8).sum()) == 4
print("ok: top-k 3, top-p 5, min-p 4 токена")
