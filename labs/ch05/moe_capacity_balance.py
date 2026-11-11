"""Ёмкость эксперта и балансирующая ошибка смеси экспертов.

Полная версия листингов «Ёмкость эксперта» и «Балансирующая ошибка» из главы 5 «Архитектуры современных
LLM» (подраздел 5.4.4 «Перегрузка экспертов и балансировка нагрузки»). В книге оставлены функции и проверки
на пакете из восьми токенов; здесь вдобавок разобран коэффициент ёмкости 1,25.
Запуск из корня репозитория: python labs/ch05/moe_capacity_balance.py
"""
import math

import torch


def capacity(tokens, k, experts, factor):
    return math.ceil(factor * tokens * k / experts)


def dispatch(chosen, experts, cap):
    load, kept = [0] * experts, []
    for token, pair in enumerate(chosen.tolist()):    # порядок токенов в пакете
        for e in pair:
            if load[e] < cap:
                load[e] += 1
                kept.append((token, e))
    return kept, load


chosen = torch.tensor([[0, 2], [0, 1], [0, 2], [0, 3],
                       [0, 1], [0, 2], [0, 2], [1, 3]])          # [N=8, k=2]
assert torch.bincount(chosen.flatten(), minlength=4).tolist() == [7, 3, 4, 2]
assert capacity(8, 2, 4, factor=1.0) == 4
kept, load = dispatch(chosen, 4, cap=4)
assert load == [4, 3, 4, 2] and 16 - len(kept) == 3               # три отброшены
kept, _ = dispatch(chosen, 4, cap=capacity(8, 2, 4, factor=1.25))
assert 16 - len(kept) == 2


def balance_loss(probs, chosen, experts):
    f = torch.bincount(chosen.flatten(), minlength=experts).float() / chosen.numel()
    return experts * (f * probs.mean(dim=0)).sum()


even_probs = torch.full((8, 4), 0.25)
even_chosen = torch.tensor([[0, 1], [2, 3]] * 4)
assert abs(balance_loss(even_probs, even_chosen, 4).item() - 1.0) < 1e-6
skewed_probs = torch.tensor([[0.97, 0.01, 0.01, 0.01]] * 8)
skewed_chosen = torch.tensor([[0, 1]] * 8)
assert balance_loss(skewed_probs, skewed_chosen, 4).item() > 1.9   # перекос наказан
# руками: f = (0,5; 0,5; 0; 0), P = (0,97; 0,01; 0,01; 0,01)
assert abs(balance_loss(skewed_probs, skewed_chosen, 4).item() - 1.96) < 1e-5

# градиент идёт только через P: подсчёт f не дифференцируем
probs = skewed_probs.clone().requires_grad_(True)
balance_loss(probs, skewed_chosen, 4).backward()
grad = probs.grad.sum(dim=0)
assert grad[0] > 0 and grad[2] == 0       # давит на перегруженного, свободных не трогает

print("moe_capacity_balance: все проверки пройдены")
