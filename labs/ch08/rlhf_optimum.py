"""Глава 8, раздел 8.5 «Прямая оптимизация предпочтений (DPO)»:
листинг «Оптимум цели RLHF в замкнутом виде» (полная версия).

В книге оставлена сокращённая версия; этот файл самодостаточен и
выполняется в ноутбуке главы notebooks/08_lora_sft.ipynb.
Запуск из корня репозитория: python labs/ch08/rlhf_optimum.py
"""

import torch


def rlhf_objective(pi, pi_ref, r, beta):
    # pi, pi_ref, r: [K], по числу ответов на один запрос
    return (pi * r).sum() - beta * (pi * (pi / pi_ref).log()).sum()


def rlhf_optimum(pi_ref, r, beta):
    weights = pi_ref * torch.exp(r / beta)
    return weights / weights.sum(), weights.sum()       # pi_star, Z


pi_ref = torch.full((3,), 1 / 3, dtype=torch.float64)
r = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
pi_star, Z = rlhf_optimum(pi_ref, r, beta=1.0)
torch.testing.assert_close(
    pi_star, torch.tensor([0.5761, 0.2119, 0.2119], dtype=torch.float64),
    atol=1e-4, rtol=0)
# в оптимуме цель равна beta * log Z
torch.testing.assert_close(rlhf_objective(pi_star, pi_ref, r, 1.0), Z.log())
# ручка громкости: малое beta уводит к награде, большое держит у эталона
assert rlhf_optimum(pi_ref, r, beta=0.1)[0][0] > 0.9999
torch.testing.assert_close(rlhf_optimum(pi_ref, r, beta=10.0)[0], torch.tensor(
    [0.3559, 0.3220, 0.3220], dtype=torch.float64), atol=1e-4, rtol=0)
torch.testing.assert_close(rlhf_optimum(pi_ref, r, beta=1e6)[0], pi_ref,
                           atol=1e-6, rtol=0)        # beta -> бесконечность

# политика «помнит» награду: log(pi*/pi_ref) восстанавливает r до сдвига
implied = (pi_star / pi_ref).log()                   # beta = 1
torch.testing.assert_close(implied[0] - implied[1], r[0] - r[1])
assert abs(-implied[1].item() - Z.log().item()) < 1e-12  # сдвиг = beta log Z

torch.manual_seed(5)
for _ in range(1000):
    pi = torch.randn(3, dtype=torch.float64).softmax(dim=-1)
    kl_to_opt = (pi * (pi / pi_star).log()).sum()
    # запись через KL совпадает с исходной целью
    torch.testing.assert_close(rlhf_objective(pi, pi_ref, r, 1.0),
                               -kl_to_opt + Z.log())
    assert rlhf_objective(pi, pi_ref, r, 1.0) <= Z.log() + 1e-12

print(f"pi* = {[round(v, 4) for v in pi_star.tolist()]}, beta log Z = {Z.log().item():.4f}")
