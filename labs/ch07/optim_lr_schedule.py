"""Глава 7, раздел 7.4 «Оптимизаторы и расписание скорости обучения».

Полная версия листинга «Разогрев и косинус»: формула расписания из подраздела 7.4.7 «Расписание
скорости обучения», учебный пример s_max = 2000,
настоящий прогон BookLM s_max = 4043, монотонность на разогреве и спаде и сверка с
torch.optim.lr_scheduler.LambdaLR на 300 шагах, включая поломку «формула без сдвига на единицу».
Запуск из корня репозитория: python labs/ch07/optim_lr_schedule.py
"""

import math
import torch

# --- Разогрев и косинус ----------------------------------------------------
def lr_at(s, warmup, total, lr_max, lr_min):
    # s — номер шага оптимизатора, начиная с 1
    if s <= warmup:
        return lr_max * s / warmup
    progress = (s - warmup) / (total - warmup)
    cos = math.cos(math.pi * progress)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + cos)

# учебный пример: s_max = 2000
sched = dict(warmup=200, total=2000, lr_max=3e-4, lr_min=3e-5)
for s, expected in [(100, 1.5e-4), (200, 3e-4),
                    (1100, 1.65e-4), (2000, 3e-5)]:
    assert math.isclose(lr_at(s, **sched), expected, rel_tol=1e-9)
values = [lr_at(s, **sched) for s in range(1, 2001)]
up, down = values[:200], values[199:]
assert all(a < b for a, b in zip(up, up[1:]))      # растёт до s_w
assert all(a > b for a, b in zip(down, down[1:]))  # убывает после

bk = dict(sched, total=4043)          # настоящий прогон BookLM
assert math.isclose(lr_at(4043, **bk), 3e-5, rel_tol=1e-9)
assert abs(lr_at(2121, **bk) - 1.65e-4) < 1e-7
assert abs(lr_at(1235, **bk) - 2.545e-4) < 1e-7    # после падения
assert math.isclose(lr_at(1, **bk), 1.5e-6)        # планировщик заново

def lrs_seen_by_optimizer(lr_lambda, steps):
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([p], lr=sched["lr_max"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    seen = []
    for _ in range(steps):
        seen.append(opt.param_groups[0]["lr"])  # lr этого шага
        opt.step()
        scheduler.step()
    return seen

# LambdaLR считает с 0: шаг s получает lr_lambda(s - 1)
right = lrs_seen_by_optimizer(
    lambda k: lr_at(k + 1, **sched) / sched["lr_max"], 300)
assert all(math.isclose(a, b, rel_tol=1e-12)
           for a, b in zip(right, values))
wrong = lrs_seen_by_optimizer(        # поломка: формула без сдвига
    lambda k: lr_at(k, **sched) / sched["lr_max"], 300)
assert wrong[0] == 0.0                # первый шаг сделан с lr = 0


if __name__ == "__main__":
    print("OK: optim_lr_schedule.py")
