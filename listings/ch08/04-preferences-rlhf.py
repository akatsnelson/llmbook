"""Листинги главы 8 «Дообучение» книги «Большие языковые модели», файл 04-preferences-rlhf.

Код и вывод — ровно как в книге; текст книги сюда не входит. Листинги идут по порядку и
опираются друг на друга: запускайте файл целиком или ячейками сверху вниз (разметка # %%).
Проверка, что вывод совпал с книгой: python scripts/check_listings.py listings/ch08/04-preferences-rlhf.py
"""

# %% Листинг «Ошибка Брэдли–Терри» — раздел «Модель награды»
import torch
import torch.nn.functional as F

def bt_loss(r_w, r_l):
    # r_w, r_l: [N] награды предпочтённого и отвергнутого ответов
    return -F.logsigmoid(r_w - r_l).mean()

t = lambda *v: torch.tensor(v, dtype=torch.float64)
print(f"верный порядок   {bt_loss(t(1.2), t(0.2)).item():.4f}")
print(f"перепутанный     {bt_loss(t(0.2), t(1.2)).item():.4f}")
print(f"обе награды ноль {bt_loss(t(0.0), t(0.0)).item():.4f}")
print(f"сдвиг обеих +10  {bt_loss(t(11.2), t(10.2)).item():.4f}")
# Вывод в книге:
# | верный порядок   0.3133
# | перепутанный     1.3133
# | обе награды ноль 0.6931
# | сдвиг обеих +10  0.3133

# %% Листинг «Градиент политики с вычитанием ожидания и без него» — раздел «Градиент политики»
r = t(3.0, 2.5, 2.0)                      # точный, короткий, вода
theta = torch.zeros(3, dtype=torch.float64, requires_grad=True)
(theta.softmax(-1) * r).sum().backward()   # точный градиент E[r]
fmt = lambda v: " ".join(f"{x + 0.0:+.3f}" for x in v.tolist())
print("точный градиент:   ", fmt(theta.grad))

torch.manual_seed(0)
pi = theta.detach().softmax(-1)
y = torch.multinomial(pi, 50_000, replacement=True)  # сгенерированные ответы
grad_logp = F.one_hot(y, 3).double() - pi            # e_y - pi
for name, b in [("без вычитания", 0.0), ("с вычитанием 2,5", 2.5)]:
    g = (r[y] - b)[:, None] * grad_logp          # вклад каждого ответа
    print(f"{name:17s}  среднее {fmt(g.mean(0))}, "
          f"разброс {g.var(0).sum().item():.3f}")
    print(f"  толчки воде, ответы 1-5: {fmt(g[:5, 2])}")
# Вывод в книге:
# | точный градиент:    +0.167 +0.000 -0.167
# | без вычитания      среднее +0.171 -0.009 -0.162, разброс 4.222
# |   толчки воде, ответы 1-5: +1.333 +1.333 -0.833 +1.333 -0.833
# | с вычитанием 2,5   среднее +0.168 +0.000 -0.168, разброс 0.056
# |   толчки воде, ответы 1-5: -0.333 -0.333 +0.000 -0.333 +0.000

# %% Листинг «Логарифм отношения инструктивной BookLM к базе» — раздел «Градиент политики»
import re
from booklm.lab import load, prompt_ids, special
from booklm.experiments.ch08 import sample

tok, base = load("base")
_, inst = load("instruct")
ids = prompt_ids(tok, "Расскажи сказку про ёжика, который боялся темноты.",
                 chat=True)                # <bos><|user|>...<|assistant|>

def log_ratio(y):            # log pi_instruct - log pi_base на токенах y
    x, idx = torch.tensor([ids + y]), torch.tensor(y)[:, None]
    with torch.no_grad():
        lp = [m(x)[0, len(ids) - 1:-1].log_softmax(-1).gather(-1, idx)
              for m in (inst, base)]
    return (lp[0] - lp[1])[:, 0]

runs = [sample(tok, inst, ids, 400, seed) for seed in range(16)]
answers = [new + [special(tok)["<eos>"]] * ended for new, ended in runs]
sums = torch.tensor([log_ratio(y).sum().item() for y in answers])
hedgehog = sum(bool(re.search("[её]жик", tok.decode(y).lower()))
               for y in answers)
print(f"KL: {sums.mean():.1f} ната (от {sums.min():.1f} "
      f"до {sums.max():.1f}), ёжик в {hedgehog} ответах из 16")
lr = log_ratio(answers[0])
top, k = lr.sort(descending=True).values, len(lr) // 10   # 10% токенов
print(f"ответ 0: длина {len(lr)}, сумма {lr.sum():.1f}, первый токен "
      f"«{tok.decode(answers[0][:1])}» {lr[0]:+.2f}, <eos> {lr[-1]:+.2f}")
print(f"самые дорогие 10% токенов {top[:k].sum():+.1f}, "
      f"остальные 90% {top[k:].sum():+.1f}")
# Вывод в книге:
# | KL: 35.3 ната (от 25.2 до 54.2), ёжик в 0 ответах из 16
# | ответ 0: длина 151, сумма 31.7, первый токен «Жил» +12.34, <eos> -0.00
# | самые дорогие 10% токенов +27.8, остальные 90% +3.9

# %% Листинг «Награда по токенам и отсечение PPO» — раздел «Оптимизация политики с отсечением (PPO)»
def kl_shaped_rewards(logp, logp_ref, r_rm, beta):
    # logp, logp_ref: [T] логарифмы вероятностей выбранных токенов ответа
    rewards = -beta * (logp - logp_ref)      # поводок на каждом токене
    rewards[-1] += r_rm                      # награда судьи только в конце
    return rewards

r_t = kl_shaped_rewards(t(-1.0, -2.1, -0.5), t(-1.2, -2.0, -0.9),
                        r_rm=1.0, beta=0.05)
print("награды по токенам:", fmt(r_t), f" сумма {r_t.sum().item():.3f}")

def ppo_clip_objective(ratio, adv, eps=0.2):
    return torch.minimum(ratio * adv, ratio.clamp(1 - eps, 1 + eps) * adv)

for rho, a in [(1.3, 2.0), (0.7, -2.0), (1.3, -2.0), (1.0, 2.0)]:
    ratio = t(rho).requires_grad_()
    obj = ppo_clip_objective(ratio, t(a)).sum()
    obj.backward()
    print(f"rho = {rho}, A = {a:+.0f}:  цель {obj.item():+.1f}, "
          f"градиент по rho {ratio.grad.item() + 0.0:+.1f}")
# Вывод в книге:
# | награды по токенам: -0.010 +0.005 +0.980  сумма 0.975
# | rho = 1.3, A = +2:  цель +2.4, градиент по rho +0.0
# | rho = 0.7, A = -2:  цель -1.6, градиент по rho +0.0
# | rho = 1.3, A = -2:  цель -2.6, градиент по rho -2.0
# | rho = 1.0, A = +2:  цель +2.0, градиент по rho +2.0

# %% Листинг «Память четырёх моделей PPO для BookLM» — раздел «Память конвейера PPO»
P, head = 6_622_464, 256              # параметры BookLM; голова 256 -> 1
MiB = 2**20
ppo = {"политика, учится": 16 * P, "критик, учится": 16 * (P + head),
       "эталон, заморожен": 2 * P, "судья, заморожен": 2 * (P + head)}
for name, b in ppo.items():
    print(f"{name:18s} {b / MiB:7.2f} МБ")
total = sum(ppo.values())
print(f"{'всего':18s} {total / MiB:7.2f} МБ, "
      f"в {total / (16 * P):.2f} раза больше SFT")
adapter = 454_656                     # адаптер LoRA r=16
with_lora = (2 * P + 16 * adapter                    # основа + адаптер
             + ppo["критик, учится"] + ppo["судья, заморожен"])
print(f"{'с адаптером':18s} {with_lora / MiB:7.2f} МБ")
# Вывод в книге:
# | политика, учится    101.05 МБ
# | критик, учится      101.05 МБ
# | эталон, заморожен    12.63 МБ
# | судья, заморожен     12.63 МБ
# | всего               227.37 МБ, в 2.25 раза больше SFT
# | с адаптером         133.26 МБ
