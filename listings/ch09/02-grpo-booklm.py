"""Листинги главы 9 «Обучение с подкреплением и оценка моделей» книги «Большие языковые модели», файл 02-grpo-booklm.

Код и вывод — ровно как в книге; текст книги сюда не входит. Листинги идут по порядку и
опираются друг на друга: запускайте файл целиком или ячейками сверху вниз (разметка # %%).
Проверка, что вывод совпал с книгой: python scripts/check_listings.py listings/ch09/02-grpo-booklm.py
"""

# %% Листинг «Группа ответов BookLM и её преимущества» — раздел «Проверяющая функция для сказки»
import re
import torch
from booklm.lab import load, prompt_ids
from booklm.experiments.ch08 import sample           # генерация с зерном
from booklm.experiments.ch09 import KOT, EZH, grpo_advantages

tok, inst = load("instruct")

def story_reward(new, ended, words):    # проверяющая функция для сказки
    text = tok.decode(new).lower()
    found = all(re.search(w, text) for w in words)
    return float(found and ended and len(new) + 1 <= 300)

asks = {"Напиши сказку, в которой есть кот и мяч.": [KOT, r"\bмяч"],
        "Напиши сказку, в которой есть ёжик и фонарик.": [EZH, r"\bфонар"]}
for request, words in asks.items():
    ids = prompt_ids(tok, request, chat=True)
    group = [sample(tok, inst, ids, 300, seed) for seed in range(8)]
    rewards = torch.tensor([story_reward(n, e, words) for n, e in group],
                           dtype=torch.float64)
    adv = grpo_advantages(rewards)
    print(request)
    print("  награды:     ", " ".join(f"{v:5.0f}" for v in rewards))
    print("  преимущества:", " ".join(f"{v:+5.2f}" for v in adv))
    if rewards.sum() > 0:
        best = tok.decode(group[int(rewards.argmax())][0]).split()
        print("  удачный ответ:", " ".join(best[:10]), "…")
# Вывод в книге:
# | Напиши сказку, в которой есть кот и мяч.
# |   награды:          0     0     0     0     1     0     0     0
# |   преимущества: -0.38 -0.38 -0.38 -0.38 +2.65 -0.38 -0.38 -0.38
# |   удачный ответ: Жил-был маленький кот и собака. Они любили играть вместе. Однажды …
# | Напиши сказку, в которой есть ёжик и фонарик.
# |   награды:          0     0     0     0     0     0     0     0
# |   преимущества: +0.00 +0.00 +0.00 +0.00 +0.00 +0.00 +0.00 +0.00

# %% Листинг «Шаг GRPO на BookLM» — раздел «Шаг GRPO с адаптером LoRA»
from booklm.experiments.ch09 import (REQUESTS, answer_batch,
                                              new_policy, sample_group)
from sft_lora import set_lora

def hero_reward(new, ended, hero, limit=150):   # герой есть, конец вовремя
    return story_reward(new, ended, [hero]) * (len(new) + 1 <= limit)

def grpo_step(policy, opt, batch, gen, G=8, beta=0.04):
    log = []
    opt.zero_grad()
    for request, hero in batch:
        ids = prompt_ids(tok, request, chat=True)
        group = sample_group(policy, ids, G, 300, gen)   # без градиента
        r = torch.tensor([hero_reward(a, e, hero) for a, e in group])
        adv = grpo_advantages(r)                         # [G]
        x, mask = answer_batch(ids, group)    # [G, T]; маска ошибки по ответу
        logp = policy(x[:, :-1]).log_softmax(-1)         # [G, T-1, V]
        with torch.no_grad():                            # эталон
            set_lora(policy, False)
            ref = policy(x[:, :-1]).log_softmax(-1)
            set_lora(policy, True)
        tok_logp = logp.gather(-1, x[:, 1:, None])[..., 0]
        kl = (logp.exp() * (logp - ref)).sum(-1)         # [G, T-1]
        n = mask.sum(1)                                  # |y_i| с <eos>
        pg = -(adv * (tok_logp * mask).sum(1) / n).mean()
        kl_mean = ((kl * mask).sum(1) / n).mean()
        ((pg + beta * kl_mean) / len(batch)).backward()
        log.append((r.mean().item(), kl_mean.item(), n.mean().item()))
    opt.step()
    return [sum(col) / len(col) for col in zip(*log)]

_, policy, params = new_policy()            # instruct + LoRA r=8, B = 0
opt = torch.optim.Adam(params, lr=2e-3)
gen = torch.Generator().manual_seed(0)
for step in range(3):
    r, kl, n = grpo_step(policy, opt, REQUESTS[4 * step:4 * step + 4], gen)
    print(f"шаг {step}: награда {r:.3f}, KL {kl:.4f}, длина {n:.1f}")
# Вывод в книге:
# | шаг 0: награда 0.219, KL 0.0000, длина 162.7
# | шаг 1: награда 0.094, KL 0.0045, длина 162.4
# | шаг 2: награда 0.312, KL 0.0098, длина 178.5

# %% Листинг «Журнал прогона GRPO по десяти шагам» — раздел «Журнал прогона»
import csv
from booklm.lab import MODELS

log = list(csv.DictReader(open(MODELS / "experiments" / "ch09-grpo-run.csv")))
cols = ("reward", "hero", "eos", "kl", "length", "dead")
print("шаги    награда  герой  <eos>     KL  длина  мёртвых групп")
for start in range(0, len(log), 10):
    part = log[start:start + 10]                       # 10 шагов = 40 групп
    m = {c: sum(float(r[c]) for r in part) / len(part) for c in cols}
    print(f"{start:2d}-{start + 9:2d}   {m['reward']:7.2f} {m['hero']:6.2f}"
          f" {m['eos']:6.2f} {m['kl']:6.3f} {m['length']:6.1f}"
          f" {m['dead']:14.2f}")
print(f"время прогона: {float(log[-1]['seconds']) / 60:.0f} мин")
# Вывод в книге:
# | шаги    награда  герой  <eos>     KL  длина  мёртвых групп
# |  0- 9      0.38   0.60   0.98  0.039  146.4           0.20
# | 10-19      0.80   0.86   1.00  0.189  106.6           0.25
# | 20-29      0.82   0.86   1.00  0.208  104.4           0.55
# | 30-39      0.82   0.89   1.00  0.199  118.0           0.40
# | 40-49      0.87   0.95   1.00  0.251  113.6           0.40
# | 50-59      0.91   0.93   1.00  0.241  107.5           0.60
# | 60-69      0.82   0.89   1.00  0.265  117.5           0.25
# | 70-79      0.90   0.93   1.00  0.274  112.6           0.45
# | время прогона: 10 мин

# %% Листинг «BookLM до и после прогона GRPO» — раздел «Сказки до и после прогона»
from booklm.experiments.ch09 import NEW_HEROES

adapter = torch.load(MODELS / "experiments" / "ch09-grpo-adapter.pt")
policy.load_state_dict(adapter, strict=False)      # веса после 80 шагов

def measure(requests):
    gen = torch.Generator().manual_seed(1)
    reward, names, length = [], [], []
    for request, hero in requests:
        ids = prompt_ids(tok, request, chat=True)
        for a, e in sample_group(policy, ids, 8, 300, gen):
            reward.append(hero_reward(a, e, hero))
            names.append(len(re.findall(hero, tok.decode(a).lower())))
            length.append(len(a) + e)
    return [sum(v) / len(v) for v in (reward, names, length)]

for title, requests in (("16 просьб прогона", REQUESTS),
                        ("4 новых героя", NEW_HEROES)):
    for name, on in (("до GRPO", False), ("после", True)):
        set_lora(policy, on)
        r, k, n = measure(requests)
        print(f"{title}, {name}: награда {r:.2f}, "
              f"герой назван {k:.1f} раза, длина {n:.0f}")
gen = torch.Generator().manual_seed(1)
ids = prompt_ids(tok, REQUESTS[0][0], chat=True)     # сказка про кота
for a, _ in sample_group(policy, ids, 8, 300, gen)[:3]:
    print("  " + " ".join(tok.decode(a).split()[:12]) + " …")
# Вывод в книге:
# | 16 просьб прогона, до GRPO: награда 0.24, герой назван 3.3 раза, длина 166
# | 16 просьб прогона, после: награда 0.92, герой назван 6.7 раза, длина 112
# | 4 новых героя, до GRPO: награда 0.28, герой назван 3.6 раза, длина 168
# | 4 новых героя, после: награда 0.91, герой назван 7.5 раза, длина 116
# |   Жил-был маленький кот. Кот любил пробовать новые вещи. Однажды кот увидел большую …
# |   Однажды кот почувствовал страх, и кот очень злился. Он хотел передвинуть ящик …
# |   Жил-был милый кот. Кот увидел бутылку черреда на земле. Он не понял, …

# %% Листинг «GRPO на BookLM с настоящей и случайной наградой» — раздел «Контроль случайной наградой»
import csv
from booklm.lab import MODELS

def blocks(name):                        # средние по 20 шагов
    log = list(csv.DictReader(open(MODELS / "experiments" / name)))
    return [{c: sum(float(r[c]) for r in log[i:i + 20]) / 20
             for c in ("reward", "hero", "length", "kl")}
            for i in range(0, len(log), 20)]

print("шаги   награда      герой        длина      KL")
real = blocks("ch09-grpo-run.csv")
coin = blocks("ch09-grpo-random.csv")       # монетка вместо проверки
for i, (a, b) in enumerate(zip(real, coin)):
    cells = [f"{a[c]:.2f} / {b[c]:.2f}" for c in ("reward", "hero")]
    cells.append(f"{a['length']:3.0f} / {b['length']:3.0f}")
    cells.append(f"{a['kl']:.2f} / {b['kl']:.2f}")
    print(f"{20 * i:2d}-{20 * i + 19:2d}  " + "  ".join(cells))
# Вывод в книге:
# | шаги   награда      герой        длина      KL
# |  0-19  0.59 / 0.26  0.73 / 0.58  126 / 164  0.11 / 0.06
# | 20-39  0.82 / 0.16  0.88 / 0.41  111 / 175  0.20 / 0.15
# | 40-59  0.89 / 0.13  0.94 / 0.37  111 / 170  0.25 / 0.20
# | 60-79  0.86 / 0.20  0.91 / 0.48  115 / 169  0.27 / 0.25
