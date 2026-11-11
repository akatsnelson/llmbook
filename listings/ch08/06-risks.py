"""Листинги главы 8 «Дообучение» книги «Большие языковые модели», файл 06-risks.

Код и вывод — ровно как в книге; текст книги сюда не входит. Листинги идут по порядку и
опираются друг на друга: запускайте файл целиком или ячейками сверху вниз (разметка # %%).
Проверка, что вывод совпал с книгой: python scripts/check_listings.py listings/ch08/06-risks.py
"""

# %% Листинг «Забывание на двух линейных задачах» — раздел «Катастрофическое забывание»
import torch

torch.manual_seed(0)
d, n = 16, 512                     # ширина и число примеров на задачу
W_A = torch.randn(d, d) / d**0.5   # правило A; база знает его точно
u, v = torch.randn(d, 1), torch.randn(1, d)
W_B = W_A + 0.5 * u @ v / d**0.5 + 0.05 * torch.randn(d, d)  # ~ранг 1
X_A, X_B, X_0 = torch.randn(n, d), torch.randn(n, d), torch.randn(n, d)
Y_A, Y_B = X_A @ W_A.T, X_B @ W_B.T
mse = lambda W, X, Y: ((X @ W.T - Y) ** 2).mean()

def finetune(X, Y, keep=0.0, rank=0):
    if rank:                       # LoRA: W = W_A + B A, учим A и B
        A = (0.1 * torch.randn(rank, d)).requires_grad_()
        B = torch.zeros(d, rank, requires_grad=True)
        params, weight = [A, B], lambda: W_A + B @ A
    else:                          # учим все d * d чисел
        W = W_A.clone().requires_grad_()
        params, weight = [W], lambda: W
    opt = torch.optim.Adam(params, lr=1e-2)
    for _ in range(400):
        W = weight()
        loss = mse(W, X, Y) + keep * mse(W, X_0, X_0 @ W_A.T)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return weight().detach()

k = n // 5                         # добавка: 20 % пакета из задачи A
variants = {"полное": finetune(X_B, Y_B),
            "LoRA r=1": finetune(X_B, Y_B, rank=1),
            "добавка": finetune(torch.cat([X_B[k:], X_A[:k]]),
                                torch.cat([Y_B[k:], Y_A[:k]])),
            "штраф": finetune(X_B, Y_B, keep=0.25)}
for name, W in variants.items():
    print(f"{name:9s} забывание A {mse(W, X_A, Y_A):.3f}   "
          f"ошибка B {mse(W, X_B, Y_B):.3f}")
# Вывод в книге:
# | полное    забывание A 0.196   ошибка B 0.000
# | LoRA r=1  забывание A 0.153   ошибка B 0.045
# | добавка   забывание A 0.129   ошибка B 0.007
# | штраф     забывание A 0.126   ошибка B 0.008

# %% Листинг «Забывание BookLM после SFT с LoRA» — раздел «Катастрофическое забывание»
import math
from booklm.lab import load, perplexity, prompt_ids
from booklm.experiments.ch08 import sample

tok, base = load("base")
_, inst = load("instruct")
models = {"база": base, "после LoRA": inst}
L_pre = {name: math.log(perplexity(m)) for name, m in models.items()}
for name, value in L_pre.items():
    print(f"{name:11s} L_pre = {value:.4f}, "
          f"перплексия {math.exp(value):.3f}")
print(f"Delta_forget = {L_pre['после LoRA'] - L_pre['база']:+.4f}")

ids = prompt_ids(tok, "Однажды летом маленькая девочка")    # без <|user|>
for name, m in models.items():
    runs = [sample(tok, m, ids, 400, seed) for seed in range(20)]
    done = [len(new) for new, ended in runs if ended]
    print(f"{name}: <eos> в {len(done)} продолжениях из 20, "
          f"средняя длина в токенах {sum(done) / len(done):.0f}")
# Вывод в книге:
# | база        L_pre = 1.9400, перплексия 6.958
# | после LoRA  L_pre = 1.9822, перплексия 7.258
# | Delta_forget = +0.0422
# | база: <eos> в 20 продолжениях из 20, средняя длина в токенах 181
# | после LoRA: <eos> в 20 продолжениях из 20, средняя длина в токенах 151

# %% Листинг «Взлом награды в замкнутом виде» — раздел «Взлом награды»
lengths = torch.arange(1, 13, dtype=torch.float64)    # длины 1..12
Q = torch.tensor([0.10, 0.35, 0.60, 0.80, 0.95, 1.00,  # качество:
                  0.90, 0.70, 0.50, 0.35, 0.20, 0.10], # горб на 6
                 dtype=torch.float64)
r = Q + 0.3 * lengths                                  # судья любит длину
pi_ref = 0.6 ** lengths
pi_ref = pi_ref / pi_ref.sum()         # эталон пишет коротко

def optimal_policy(beta):             # pi* ~ pi_ref * exp(r / beta)
    return torch.softmax(pi_ref.log() + r / beta, dim=0)

for beta in [float("inf"), 4, 1, 0.5, 0.25, 0.12, 0.05]:
    p = pi_ref if beta == float("inf") else optimal_policy(beta)
    kl_value = (p * (p / pi_ref).log()).sum()
    print(f"beta={beta:<4}  KL={kl_value:4.2f}  E[r]={(p * r).sum():.2f}  "
          f"E[Q]={(p * Q).sum():.2f}  длина={(p * lengths).sum():5.2f}")
# Вывод в книге:
# | beta=inf   KL=0.00  E[r]=1.13  E[Q]=0.39  длина= 2.47
# | beta=4     KL=0.02  E[r]=1.32  E[Q]=0.45  длина= 2.89
# | beta=1     KL=0.45  E[r]=1.99  E[Q]=0.63  длина= 4.56
# | beta=0.5   KL=1.49  E[r]=2.71  E[Q]=0.70  длина= 6.70
# | beta=0.25  KL=2.89  E[r]=3.21  E[Q]=0.53  длина= 8.95
# | beta=0.12  KL=4.92  E[r]=3.57  E[Q]=0.20  длина=11.20
# | beta=0.05  KL=6.36  E[r]=3.69  E[Q]=0.10  длина=11.96

# %% Листинг «Измерения BookLM до и после SFT с LoRA» — раздел «Измерения после каждой стадии»
# Проверка этот листинг не запускает: ему нужны сеть, видеокарта, сервер или внешний файл.
# from booklm.experiments.ch08 import (adapter_model, answer_logp, encode,
#                                      eos_stats, held_out, story_loss)
#
# models = {"база": base, "адаптер": adapter_model(), "слитая": inst}
# pairs = [(r["request"], r["story"]) for r in held_out()]   # 200 пар
# requests = [r["request"] for r in held_out()[:50]]
# base_logits = [answer_logp(base, *encode(tok, *p, 512))[1] for p in pairs]
# rows = {k: [] for k in ("ошибка на ответах", "L_pre", "Δ_forget",
#                         "доля <eos> ≤ 300", "средняя длина",
#                         "KL к базе на ответах")}
# for name, m in models.items():
#     rows["ошибка на ответах"].append(story_loss(m, tok, pairs))
#     rows["L_pre"].append(math.log(perplexity(m)))
#     rows["Δ_forget"].append(rows["L_pre"][-1] - rows["L_pre"][0])
#     share, mean_len = eos_stats(tok, m, requests, 300)
#     rows["доля <eos> ≤ 300"].append(share)
#     rows["средняя длина"].append(mean_len)
#     kl, n = 0.0, 0
#     for p, lb in zip(pairs, base_logits):     # KL(pi_модели || pi_базы)
#         lm = answer_logp(m, *encode(tok, *p, 512))[1].log_softmax(-1)
#         kl += (lm.exp() * (lm - lb.log_softmax(-1))).sum().item()
#         n += len(lm)
#     rows["KL к базе на ответах"].append(kl / n)
# print(f"{'метрика':22s}" + "".join(f"{k:>10s}" for k in models))
# for k, v in rows.items():
#     fmt = ("{:10.0%}" if "доля" in k else
#            "{:10.1f}" if "длина" in k else "{:10.4f}")
#     print(f"{k:22s}" + "".join(fmt.format(x) for x in v))

# %% Листинг «Настройки дообучения BookLM в PEFT, transformers и TRL» — раздел «Дообучение в настоящих системах»
# Проверка этот листинг не запускает: ему нужны сеть, видеокарта, сервер или внешний файл.
# # Сверено по исходникам выпусков: peft 0.21.2, transformers 5.21,
# # bitsandbytes 0.50.2, trl 1.16.0 (октябрь 2026).
# import torch
# from peft import LoraConfig
# from transformers import BitsAndBytesConfig
# from trl import SFTConfig, DPOConfig
#
# lora = LoraConfig(
#     r=16,                                 # ранг адаптера BookLM
#     lora_alpha=32,                        # alpha/r = 2
#     lora_dropout=0.0,
#     bias="none",
#     target_modules=["q_proj", "k_proj", "v_proj", "o_proj",  # матрицы
#                     "gate_proj", "up_proj", "down_proj"],     # в Llama
# )
# nf4 = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",            # раздел о QLoRA
#     bnb_4bit_use_double_quant=True,       # 4,127 бита на вес
#     bnb_4bit_compute_dtype=torch.bfloat16,
# )
# # шаблон диалога должен размечать ответ метками {% generation %}
# sft = SFTConfig(assistant_only_loss=True, max_length=512,
#                 packing=False)  # упаковке нужен FlashAttention, см. ниже
# dpo = DPOConfig(beta=0.1)
