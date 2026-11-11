"""Листинги главы 14 «Инференс» книги «Большие языковые модели», файл 04-quantization-formats.

Код и вывод — ровно как в книге; текст книги сюда не входит. Листинги идут по порядку и
опираются друг на друга: запускайте файл целиком или ячейками сверху вниз (разметка # %%).
Проверка, что вывод совпал с книгой: python scripts/check_listings.py listings/ch14/04-quantization-formats.py
"""

# %% Листинг «Симметричное квантование одного тензора» — раздел «Равномерное квантование»
import torch

def quantize_symmetric(x, bits=8):
    qmax = 2 ** (bits - 1) - 1              # 127 для INT8, 7 для INT4
    scale = x.abs().amax().clamp(min=1e-8) / qmax
    q = torch.round(x / scale).clamp(-qmax, qmax).to(torch.int8)
    return q, scale

def dequantize(q, scale):
    return q.float() * scale

torch.manual_seed(1)
w = torch.randn(4096)
for bits in (8, 4):
    q, scale = quantize_symmetric(w, bits)
    err = (w - dequantize(q, scale)).abs()
    print(f"INT{bits}: шаг {scale:.4f}, средняя ошибка {err.mean():.4f}, "
          f"худшая {err.max():.4f}")
# Вывод в книге:
# | INT8: шаг 0.0311, средняя ошибка 0.0078, худшая 0.0155
# | INT4: шаг 0.5636, средняя ошибка 0.1398, худшая 0.2818

# %% Листинг «Групповое квантование и цена масштабов» — раздел «Группы весов»
def groupwise_fake_quant(w, bits=4, group_size=128):
    qmax = 2 ** (bits - 1) - 1
    groups = w.reshape(w.size(0), -1, group_size)    # [out, групп, g]
    amax = groups.abs().amax(dim=-1, keepdim=True)  # [out, групп, 1]
    scales = amax.clamp(min=1e-8) / qmax
    q = torch.round(groups / scales).clamp(-qmax, qmax)
    return (q * scales).reshape_as(w), q.to(torch.int8), scales

torch.manual_seed(0)
w = torch.randn(1024, 4096)
for g in (32, 128, 512, 4096):
    w_hat, _, _ = groupwise_fake_quant(w, bits=4, group_size=g)
    mse = ((w - w_hat) ** 2).mean()
    print(f"g={g:>4}: ошибка {mse:.4f}, бит на вес {4 + 16 / g:.3f}")
# Вывод в книге:
# | g=  32: ошибка 0.0094, бит на вес 4.500
# | g= 128: ошибка 0.0138, бит на вес 4.125
# | g= 512: ошибка 0.0182, бит на вес 4.031
# | g=4096: ошибка 0.0248, бит на вес 4.004

# %% Листинг «Упаковка двух знаковых INT4 в один байт» — раздел «Группы весов»
def pack_int4(q):                     # коды из [-8, 7], два в байт
    u = (q.to(torch.int16) + 8).to(torch.uint8)      # сдвиг в [0, 15]
    if u.numel() % 2:
        u = torch.cat([u, torch.zeros(1, dtype=torch.uint8)])
    return u[0::2] | (u[1::2] << 4)   # первый код в младших битах

def unpack_int4(packed, n):
    low, high = packed & 0x0F, packed >> 4
    out = torch.stack([low, high], dim=1).flatten().to(torch.int8) - 8
    return out[:n]

q = torch.tensor([-1, 3, -8, 7, 0], dtype=torch.int8)
packed = pack_int4(q)
print([hex(b) for b in packed.tolist()])
print(unpack_int4(packed, len(q)).tolist())
# Вывод в книге:
# | ['0xb7', '0xf0', '0x8']
# | [-1, 3, -8, 7, 0]

# %% Листинг «GPTQ на двух весах: ошибку первого гасит второй» — раздел «GPTQ»
X = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])  # токены, x1 = x2
w = torch.tensor([1.4, 1.4])
G = X.T @ X                    # G[0,1]: сумма x1*x2, G[1,1]: сумма x2*x2

rtn = torch.round(w)                   # каждый вес сам по себе
q1 = torch.round(w[0])                 # округлили первый вес до 1
loss = w[0] - q1                       # потеряли 0.4
w2 = w[1] + loss * G[0, 1] / G[1, 1]   # вернули потерю через второй
gptq = torch.stack([q1, torch.round(w2)])

def out_err(v):                        # ошибка выхода слоя
    return ((X @ w - X @ v) ** 2).mean().item()

for name, v in [("RTN", rtn), ("GPTQ", gptq)]:
    print(f"{name}: веса {v.tolist()}, ошибка весов "
          f"{(w - v).abs().max():.1f}, ошибка выхода {out_err(v):.2f}")
# Вывод в книге:
# | RTN: веса [1.0, 1.0], ошибка весов 0.4, ошибка выхода 2.99
# | GPTQ: веса [1.0, 2.0], ошибка весов 0.6, ошибка выхода 0.19

# %% Листинг «Учебный поиск масштабов AWQ» — раздел «AWQ»
def awq_search(w, x, steps=11):
    # w: [out, in], x: [tokens, in]
    reference = x @ w.T                     # что слой считает сейчас
    a = x.abs().mean(dim=0)                       # типичный модуль канала
    results = []
    for alpha in torch.linspace(0, 1, steps).tolist():
        lam = a ** alpha
        lam = lam / (lam.max() * lam.min()).sqrt()   # держим около 1
        w_hat = groupwise_fake_quant(w * lam, bits=4, group_size=128)[0]
        loss = ((reference - (x / lam) @ w_hat.T) ** 2).mean().item()
        results.append((loss, alpha))
    return results

torch.manual_seed(0)
w, x = torch.randn(64, 256), torch.randn(512, 256)
x[:, 7] *= 50                                     # один канал-выброс
results = awq_search(w, x)
for loss, alpha in results[::2]:
    print(f"alpha={alpha:.1f}: ошибка выхода {loss:.2f}")
best_loss, best_alpha = min(results)
print(f"лучшее alpha={best_alpha:.1f}: {best_loss:.2f}")
# Вывод в книге:
# | alpha=0.0: ошибка выхода 42.11
# | alpha=0.2: ошибка выхода 10.24
# | alpha=0.4: ошибка выхода 9.65
# | alpha=0.6: ошибка выхода 28.66
# | alpha=0.8: ошибка выхода 64.06
# | alpha=1.0: ошибка выхода 97.87
# | лучшее alpha=0.3: 7.22

# %% Листинг «Сжатие обученной BookLM» — раздел «Число бит на вес с учётом масштабов»
import os
from booklm.lab import load_tokenizer
from booklm.experiments.ch14 import (bf16_copy, linear_layers, greedy,
    size_mib, story_perplexity, mxfp4_fake_quant, ternary_fake_quant)

def by_rows(bits):                       # один масштаб FP16 на строку
    return lambda w: groupwise_fake_quant(w, bits, w.size(1))[0]

def by_groups(g):                        # INT4, масштаб на g весов
    return lambda w: groupwise_fake_quant(w, 4, g)[0]

variants = [  # имя, приближение матрицы, бит на вес при длине строки d
    ("BF16", lambda w: w, lambda d: 16),
    ("INT8, строка", by_rows(8), lambda d: 8 + 16 / d),
    ("INT4, g=128", by_groups(128), lambda d: 4 + 16 / 128),
    ("INT4, g=32", by_groups(32), lambda d: 4 + 16 / 32),
    ("INT4, строка", by_rows(4), lambda d: 4 + 16 / d),
    ("MXFP4", mxfp4_fake_quant, lambda d: 4.25),
    ("три уровня", ternary_fake_quant, lambda d: 1.6 + 16 / 128),
]
tok, start = load_tokenizer(), "Жила-была маленькая лиса. Однажды она"
for name, quant, bits in variants:
    model = bf16_copy("base")                # веса, как их хранят: BF16
    for lin in linear_layers(model):         # 42 матрицы шести блоков
        lin.weight.data = quant(lin.weight.data)
    words = greedy(tok, model, start, 40).split()
    reference = words if name == "BF16" else reference
    same = len(os.path.commonprefix([words, reference]))
    print(f"{name:<13}{size_mib(model, bits):6.2f} МБ  перплексия "
          f"{story_perplexity(model):6.2f}  слов как у BF16: {same}")
print("три уровня:", " ".join(words[:12]))
# Вывод в книге:
# | BF16          12.63 МБ  перплексия   6.66  слов как у BF16: 31
# | INT8, строка   8.35 МБ  перплексия   6.66  слов как у BF16: 31
# | INT4, g=128    6.23 МБ  перплексия   7.06  слов как у BF16: 31
# | INT4, g=32     6.43 МБ  перплексия   6.88  слов как у BF16: 7
# | INT4, строка   6.19 МБ  перплексия   7.16  слов как у BF16: 17
# | MXFP4          6.30 МБ  перплексия   6.95  слов как у BF16: 12
# | три уровня     4.94 МБ  перплексия 136.23  слов как у BF16: 0
# | три уровня: была очень запуталась Эм Эм Эм Эм Эм Эм Эм Эм Эм

# %% Листинг «Три уровня на вес: дообучение с прямым оценивателем градиента» — раздел «Число бит на вес с учётом масштабов»
# Проверка этот листинг не запускает: ему нужны сеть, видеокарта, сервер или внешний файл.
# from booklm.experiments.ch14 import train_ternary
#
# model, seen = train_ternary(steps=300)    # 300 шагов, 16 окон по 256
# mib = size_mib(model, lambda d: 1.6 + 16 / 128)
# print(f"три уровня + STE {mib:5.2f} МБ  перплексия "
#       f"{story_perplexity(model):6.2f}  токенов дообучения: {seen}")
# print(" ".join(greedy(tok, model, start, 40).split()[:12]))
