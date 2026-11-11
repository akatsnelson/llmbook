"""Глава 10, раздел 10.3 «Цепочка рассуждений и бюджет вычислений»: листинги «Бюджет окна и KV-кэша»,
«Операции на ответ с рассуждением и без» и опыт с голосованием, которого в книге нет.

Считает FLOP на токен, байты KV-кэша, проверяет бюджет окна и сравнивает цену ответа с рассуждением и без
него (3,61 против 0,45 GFLOP). Во второй части сверяет точную вероятность победы голосования
(vote_win_prob из книги) с десятью тысячами случайных голосований и показывает, что при очень низкой
температуре голосование вырождается в один жадный ответ. Выполняется и в ноутбуке главы
notebooks/10_constrained_decoding.ipynb.

Запуск из корня репозитория: python labs/ch10/reasoning_budget.py
"""

N_PARAMS = 6_622_464                     # паспорт BookLM, глава 4
L, H_Q, H_KV, D_H = 6, 8, 2, 32
T_MAX = 512
KV_PER_TOKEN = 2 * L * H_KV * D_H * 2    # K и V, блоки, головы, числа, байты BF16

def flops_per_token(t):
    # один токен на позиции t (видит t позиций, считая себя)
    return 2 * N_PARAMS + 4 * L * H_Q * D_H * t

def kv_bytes(T):
    return T * KV_PER_TOKEN

def budget_check(P, R, n_ans, t_max=T_MAX):
    total = P + R + n_ans
    if total > t_max:
        raise ValueError(f"{P}+{R}+{n_ans}={total} > {t_max}: окно переполнено")
    return total

def generation_flops(P, n):
    # n написанных токенов после промпта длины P
    return sum(flops_per_token(t) for t in range(P + 1, P + n + 1))

assert 2 * N_PARAMS == 13_244_928
assert flops_per_token(1) - 2 * N_PARAMS == 6144
assert KV_PER_TOKEN == 1536 and kv_bytes(512) == 786_432
assert budget_check(280, 200, 30) == 510 and kv_bytes(510) == 783_360
try:
    budget_check(280, 250, 30)
    raise AssertionError("переполнение не поймано")
except ValueError:
    pass

cot, direct = generation_flops(280, 230), generation_flops(280, 30)
assert cot == 3_605_222_400 and direct == 451_814_400
assert abs(cot / direct - 7.98) < 0.01
prompt = generation_flops(0, 280)        # чтение промпта: позиции 1..280
assert abs(prompt / 1e9 - 3.95) < 0.01
assert abs((prompt + cot) / (prompt + direct) - 1.72) < 0.01

# ----------------------------------------------------------------------------- голосование: опыт против формулы
from itertools import combinations_with_replacement
from math import factorial

import torch


def vote_win_prob(probs, k):
    # точный перебор раскладок голосов, как в книге; probs[0] — верный ответ
    total, m = 0.0, len(probs)
    for votes in combinations_with_replacement(range(m), k):
        counts = [votes.count(i) for i in range(m)]
        pr = factorial(k)
        for c, p in zip(counts, probs):
            pr *= p**c / factorial(c)
        winners = [i for i, c in enumerate(counts) if c == max(counts)]
        if 0 in winners:
            total += pr / len(winners)
    return total


def simulate_vote(probs, k, trials=10_000, temperature=1.0):
    # k окончательных ответов из softmax(log p / tau) в каждом испытании, ничью решает жребий
    logits = torch.log(torch.tensor(probs))
    p = torch.softmax(logits / temperature, dim=-1)
    picks = torch.multinomial(p.expand(trials, -1), k, replacement=True)     # [trials, k]
    counts = torch.nn.functional.one_hot(picks, len(probs)).sum(dim=1)      # [trials, m]
    is_best = (counts == counts.max(dim=1, keepdim=True).values).float()
    winner = (is_best + 0.5 * torch.rand_like(is_best)).argmax(dim=1)
    return (winner == 0).float().mean().item()


torch.manual_seed(0)
for probs in ([0.6, 0.4], [0.4, 0.2, 0.2, 0.2], [0.3, 0.45, 0.25]):
    experiment, formula = simulate_vote(probs, k=5), vote_win_prob(probs, 5)
    assert abs(experiment - formula) < 0.01, (probs, experiment, formula)
    print(f"опыт {experiment:.3f}  формула {formula:.3f}")
# очень низкая температура: все пять выборок совпадают, голосование = жадный ответ
assert simulate_vote([0.4, 0.2, 0.2, 0.2], k=5, temperature=0.1) == 1.0
assert simulate_vote([0.3, 0.45, 0.25], k=5, temperature=0.1) == 0.0
