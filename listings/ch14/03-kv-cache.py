"""Листинги главы 14 «Инференс» книги «Большие языковые модели», файл 03-kv-cache.

Код и вывод — ровно как в книге; текст книги сюда не входит. Листинги идут по порядку и
опираются друг на друга: запускайте файл целиком или ячейками сверху вниз (разметка # %%).
Проверка, что вывод совпал с книгой: python scripts/check_listings.py listings/ch14/03-kv-cache.py
"""

# %% Листинг «Учебный стенд (копия для проверки)» — раздел «Кэш в коде»
import torch
torch.manual_seed(0)
V, d = 8, 4
E = torch.randn(V, d)
W_out = torch.randn(d, V)
seen = []
prompt = torch.tensor([[3, 1, 4, 1]])

# %% Листинг «Учебный стенд с кэшем» — раздел «Кэш в коде»
def toy_model(ids, cache=None):        # cache: сумма и счётчик позиций
    seen.append(ids.size(1))
    if cache is None:                  # без кэша --- с нуля, как раньше
        cache = {"sum": torch.zeros(ids.size(0), d), "n": 0}
    h = cache["sum"][:, None] + E[ids].cumsum(dim=1)     # [B, T, d]
    n = cache["n"] + torch.arange(1, ids.size(1) + 1)    # номера позиций
    cache["sum"], cache["n"] = h[:, -1], cache["n"] + ids.size(1)
    return (h / n[None, :, None]) @ W_out

def generate_cached(model, ids, steps):
    cache = {"sum": torch.zeros(ids.size(0), d), "n": 0}
    new = ids                                   # сначала весь запрос
    for _ in range(steps):
        logits = model(new, cache)              # [B, len(new), V]
        new = logits[:, -1].argmax(dim=-1, keepdim=True)
        ids = torch.cat([ids, new], dim=1)      # дальше подаём только new
    return ids

seen.clear()
print(generate_cached(toy_model, prompt, steps=3))
print(seen, sum(seen))
# Вывод в книге:
# | tensor([[3, 1, 4, 1, 5, 4, 7]])
# | [4, 1, 1] 6

# %% Листинг «Одна голова причинного внимания: полный проход и пошаговый с кэшем» — раздел «Кэш в коде»
import math
import torch

torch.manual_seed(17)
D = 8                                         # длина вектора головы
W_q, W_k, W_v = (torch.randn(D, D) for _ in range(3))
x = torch.randn(5, D)                         # 5 токенов по D чисел

def causal_attention(q, k, v):                # q: [n, D], k и v: [m, D]
    scores = q @ k.T / math.sqrt(q.size(-1))  # [n, m]
    n, m = len(q), len(k)
    future = torch.ones(n, m, dtype=torch.bool).triu(m - n + 1)
    scores = scores.masked_fill(future, float("-inf"))
    return scores.softmax(dim=-1) @ v         # [n, D]

full = causal_attention(x @ W_q, x @ W_k, x @ W_v)   # все токены сразу

K_cache, V_cache = [], []
for t in range(len(x)):                       # токены приходят по одному
    x_t = x[t:t + 1]                          # [1, D]
    K_cache.append(x_t @ W_k)                 # ключ нового токена
    V_cache.append(x_t @ W_v)                 # и его значение
    K, V = torch.cat(K_cache), torch.cat(V_cache)
    out = causal_attention(x_t @ W_q, K, V)   # запрос нужен только сейчас
    same = torch.allclose(out, full[t:t + 1], atol=1e-5)
    print(f"шаг {t}: кэш {tuple(K.shape)}, совпало с полным: {same}")
# Вывод в книге:
# | шаг 0: кэш (1, 8), совпало с полным: True
# | шаг 1: кэш (2, 8), совпало с полным: True
# | шаг 2: кэш (3, 8), совпало с полным: True
# | шаг 3: кэш (4, 8), совпало с полным: True
# | шаг 4: кэш (5, 8), совпало с полным: True

# %% Листинг «Позиция RoPE в пошаговом расчёте» — раздел «Кэш в коде»
def rope(x, pos):                     # поворот соседних пар координат
    theta = 10000 ** (-torch.arange(0, D, 2) / D)
    angle = pos[:, None] * theta              # [T, D/2]
    cos, sin = angle.cos(), angle.sin()
    x1, x2 = x[:, 0::2], x[:, 1::2]
    out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1)
    return out.flatten(1)                     # пары снова рядом

pos = torch.arange(5.0)
full = causal_attention(rope(x @ W_q, pos), rope(x @ W_k, pos), x @ W_v)
print(f"полный проход: наибольший модуль выхода {full.abs().max():.2f}")

def cached_run(position_of):                  # какой номер позиции подаём
    K_cache, V_cache, outs = [], [], []
    for t in range(len(x)):
        x_t, p = x[t:t + 1], torch.tensor([float(position_of(t))])
        K_cache.append(rope(x_t @ W_k, p))    # поворот ДО записи в кэш
        V_cache.append(x_t @ W_v)
        K, V = torch.cat(K_cache), torch.cat(V_cache)
        outs.append(causal_attention(rope(x_t @ W_q, p), K, V))
    return torch.cat(outs)

for name, pos_of in [("позиция t", lambda t: t),
                     ("позиция 0", lambda t: 0)]:
    out = cached_run(pos_of)
    err = (out - full).abs().max()
    print(f"{name}: форма {tuple(out.shape)}, расхождение {err:.2f}")
# Вывод в книге:
# | полный проход: наибольший модуль выхода 5.71
# | позиция t: форма (5, 8), расхождение 0.00
# | позиция 0: форма (5, 8), расхождение 4.66

# %% Листинг «Калькулятор весов и KV-кэша» — раздел «KV-кэш BookLM»
def gib(n_bytes):
    return n_bytes / 2**30

def weight_bytes(n_params, bits):
    return n_params * bits / 8

def kv_bytes(layers, batch, tokens, kv_heads, head_dim, dtype_bytes=2):
    return 2 * layers * batch * tokens * kv_heads * head_dim * dtype_bytes

cases = [  # имя, параметров, бит на вес, L, H_KV, D_h, токенов, запросов
    ("8B, ноутбук", 8e9, 4.9, 32, 8, 128, 8192, 1),
    ("8B, сервер", 8e9, 4.9, 32, 8, 128, 8192, 16),
    ("3B, телефон", 3e9, 4.5, 28, 8, 128, 4096, 1),
    ("70B, ноутбук", 70e9, 4.9, 80, 8, 128, 8192, 1),
]
for name, n, bits, L, h_kv, d_h, tokens, batch in cases:
    w = gib(weight_bytes(n, bits))
    kv = gib(kv_bytes(L, batch, tokens, h_kv, d_h))
    print(f"{name:<13} веса {w:5.2f} ГБ   кэш {kv:5.2f} ГБ")
# Вывод в книге:
# | 8B, ноутбук   веса  4.56 ГБ   кэш  1.00 ГБ
# | 8B, сервер    веса  4.56 ГБ   кэш 16.00 ГБ
# | 3B, телефон   веса  1.57 ГБ   кэш  0.44 ГБ
# | 70B, ноутбук  веса 39.93 ГБ   кэш  2.50 ГБ

# %% Листинг «Кэш в GQASelfAttention (отрывок booklm/model.py)» — раздел «KV-кэш BookLM»
# Проверка этот листинг не запускает: ему нужны сеть, видеокарта, сервер или внешний файл.
# start = 0 if cache is None else cache["len"]   # позиций уже в кэше
# q = apply_rope(q, cos[start:], sin[start:])    # углы с позиции start
# k = apply_rope(k, cos[start:], sin[start:])
# if cache is not None:                          # дописываем и читаем всё
#     if cache.get("k") is not None:
#         k = torch.cat([cache["k"], k], dim=2)  # [B, H_KV, len + t, D_h]
#         v = torch.cat([cache["v"], v], dim=2)
#     cache["k"], cache["v"] = k, v
# group = self.heads // self.kv_heads            # 4 головы запросов на пару
# kk = k.repeat_interleave(group, dim=1)
# vv = v.repeat_interleave(group, dim=1)
# total = kk.size(2)
# if t == 1:                                     # один токен видит всё
#     y = F.scaled_dot_product_attention(q, kk, vv)
# elif total == t:                               # весь текст сразу
#     y = F.scaled_dot_product_attention(q, kk, vv, is_causal=True)
# else:                                          # кусок поверх кэша
#     allowed = torch.ones(t, total, dtype=torch.bool).tril(total - t)
#     y = F.scaled_dot_product_attention(q, kk, vv, attn_mask=allowed)

# %% Листинг «KV-кэш обученной BookLM: сверка с полным проходом» — раздел «KV-кэш BookLM»
from booklm.lab import load, load_model, prompt_ids

tok, booklm = load("base")                       # обученная BookLM, FP32
text = "Жила-была маленькая лиса. Однажды она"
ids = torch.tensor([prompt_ids(tok, text)])      # <bos> и восемь токенов

@torch.no_grad()
def run_in_pieces(model, ids, size):             # кусками по size токенов
    caches = [{"len": 0} for _ in model.blocks]  # по кэшу на блок
    pieces = []
    for s in range(0, ids.size(1), size):
        pieces.append(model(ids[:, s:s + size], caches))
        for c in caches:
            c["len"] += pieces[-1].size(1)       # длину кэша ведём сами
    return torch.cat(pieces, dim=1), caches

with torch.no_grad():
    full = booklm(ids)                           # весь текст разом, без кэша
for size in (1, 4):
    err = (run_in_pieces(booklm, ids, size)[0] - full).abs().max()
    print(f"кусками по {size}: расхождение с полным {err:.0e}")
for model in (booklm, load_model("base").to(torch.bfloat16)):
    caches = run_in_pieces(model, ids, 1)[1]
    k = caches[0]["k"]                           # ключи блока 0
    total = sum(c["k"].nbytes + c["v"].nbytes for c in caches)
    print(f"ключи блока 0: {list(k.shape)} {k.dtype},"
          f" байт на токен {total // k.size(2)}")
# Вывод в книге:
# | кусками по 1: расхождение с полным 1e-05
# | кусками по 4: расхождение с полным 9e-06
# | ключи блока 0: [1, 2, 9, 32] torch.float32, байт на токен 3072
# | ключи блока 0: [1, 2, 9, 32] torch.bfloat16, байт на токен 1536

# %% Листинг «Шаг генерации BookLM с кэшем и без него» — раздел «KV-кэш BookLM»
# Проверка этот листинг не запускает: ему нужны сеть, видеокарта, сервер или внешний файл.
# import statistics, time
# torch.set_num_threads(1)                         # одно ядро процессора
# start = torch.tensor([prompt_ids(tok, "Жила-была маленькая лиса.")])
#
# @torch.no_grad()
# def greedy_timed(use_cache, n=400):        # время каждого шага
#     caches = [{"len": 0} for _ in booklm.blocks] if use_cache else None
#     x, new, times = start, start, []
#     for _ in range(n):
#         t0 = time.perf_counter()
#         logits = booklm(new, caches) if use_cache else booklm(x)
#         for c in caches or []:
#             c["len"] += new.size(1)
#         new = logits[:, -1:].argmax(dim=-1)      # [1, 1]
#         x = torch.cat([x, new], dim=1)
#         times.append((time.perf_counter() - t0) * 1e3)   # мс
#     return x, times
#
# greedy_timed(True), greedy_timed(False)          # прогрев
# (a, fast), (b, slow) = greedy_timed(True), greedy_timed(False)
# print(f"тексты совпали: {torch.equal(a, b)}; весь ответ: "
#       f"{sum(fast) / 1e3:.2f} с с кэшем, {sum(slow) / 1e3:.2f} с без")
# for pos in (50, 200, 400):
#     i = pos - start.size(1) - 1            # шаг, где текст дорос до pos
#     f = statistics.median(fast[i - 5:i + 5])
#     s = statistics.median(slow[i - 5:i + 5])
#     print(f"позиция {pos}: шаг {f:.2f} мс с кэшем, {s:.2f} мс без,"
#           f" в {s / f:.1f} раза")

# %% Листинг «KV-кэш Llama 3.1 8B при длинном контексте» — раздел «Квантование KV-кэша»
llama = dict(layers=32, kv_heads=8, head_dim=128)     # Llama 3.1 8B
formats = [("BF16", 2), ("FP8", 1), ("4 бита", 5 / 8)]  # байт на число
print(f"веса в BF16: {gib(16_060_522_496):.2f} ГБ")
for batch, tokens in [(1, 32768), (1, 131072), (8, 32768)]:
    sizes = [gib(kv_bytes(batch=batch, tokens=tokens, dtype_bytes=b, **llama))
             for _, b in formats]
    row = "   ".join(f"{name} {s:5.2f}" for (name, _), s in zip(formats, sizes))
    print(f"{batch} x {tokens:>6} токенов, ГБ:  {row}")
# Вывод в книге:
# | веса в BF16: 14.96 ГБ
# | 1 x  32768 токенов, ГБ:  BF16  4.00   FP8  2.00   4 бита  1.25
# | 1 x 131072 токенов, ГБ:  BF16 16.00   FP8  8.00   4 бита  5.00
# | 8 x  32768 токенов, ГБ:  BF16 32.00   FP8 16.00   4 бита 10.00

# %% Листинг «FP8 в PyTorch: пределы и масштаб» — раздел «FP8 в KV-кэше»
for name, dt in [("E4M3", torch.float8_e4m3fn), ("E5M2", torch.float8_e5m2)]:
    f = torch.finfo(dt)
    print(f"{name}: max = {f.max:g}, eps = {f.eps}, min = {f.tiny:g}")

def to_e4m3(x, scale=1.0):          # хранится x / scale, читается обратно
    return (x / scale).to(torch.float8_e4m3fn).float() * scale

x = torch.tensor([600.0, 15.1, 1.3, 0.13, 0.004])
for scale in (1.0, x.abs().max().item() / 448):  # 600 встанет на 448
    print(f"масштаб {scale:.3f}:", [round(v, 4) for v in to_e4m3(x, scale).tolist()])
# Вывод в книге:
# | E4M3: max = 448, eps = 0.125, min = 0.015625
# | E5M2: max = 57344, eps = 0.25, min = 6.10352e-05
# | масштаб 1.000: [448.0, 15.0, 1.25, 0.125, 0.0039]
# | масштаб 1.339: [600.0, 14.7321, 1.3393, 0.1256, 0.0052]

# %% Листинг «Группы по токенам и по каналам» — раздел «Ключи по каналам, значения по токенам»
def quantize_groups(x, bits, dim, group=32):   # формула min-max по группам
    x = x.movedim(dim, -1)                      # группы идут вдоль dim
    n, pad = x.size(-1), -x.size(-1) % group    # хвост добиваем последним числом
    g = torch.cat([x, x[..., -1:].expand(*x.shape[:-1], pad)], -1)
    g = g.unflatten(-1, (-1, group))            # [..., групп, group]
    lo, hi = g.amin(-1, keepdim=True), g.amax(-1, keepdim=True)
    step = (hi - lo).clamp(min=1e-8) / (2 ** bits - 1)
    q = ((g - lo) / step).round()               # коды 0 .. 2^bits - 1
    return (q * step + lo).flatten(-2)[..., :n].movedim(-1, dim)

row = torch.tensor([[0.3, -0.5, 12.0, 0.8]])    # ручной пример
print(quantize_groups(row, bits=2, dim=-1, group=4))

torch.manual_seed(0)
K, V = torch.randn(64, 32), torch.randn(64, 32) # 64 токена по 32 канала
K[:, 5] *= 10                                   # канал-выброс у ключей
normal = [c for c in range(32) if c != 5]
def err(x, y):                                  # относительная ошибка
    return ((x - y).norm() / x.norm()).item()
for name, X in (("K", K), ("V", V)):
    for how, dim in (("по токенам", 1), ("по каналам", 0)):
        Y = quantize_groups(X, bits=2, dim=dim)
        print(f"{name} {how}: обычные каналы {err(X[:, normal], Y[:, normal]):.2f},"
              f" канал 5 {err(X[:, 5], Y[:, 5]):.2f}")
# Вывод в книге:
# | tensor([[-0.5000, -0.5000, 12.0000, -0.5000]])
# | K по токенам: обычные каналы 1.26, канал 5 0.01
# | K по каналам: обычные каналы 0.40, канал 5 0.41
# | V по токенам: обычные каналы 0.40, канал 5 0.44
# | V по каналам: обычные каналы 0.41, канал 5 0.41

# %% Листинг «Каналы ключей и значений BookLM» — раздел «Ключи по каналам, значения по токенам»
from booklm.lab import val_tokens

data = val_tokens()                             # отложенные истории
seen, att = {}, booklm.blocks[1].attention      # блок 1, модель FP32
hooks = [att.k.register_forward_hook(lambda m, i, out: seen.update(k=out[0])),
         att.v.register_forward_hook(lambda m, i, out: seen.update(v=out[0]))]
with torch.no_grad():
    booklm(data[:512][None])                    # одно окно из 512 токенов
for h in hooks:
    h.remove()
for name in ("k", "v"):
    top = seen[name].abs().amax(dim=0)          # [64]: наибольший модуль канала
    big = top.topk(4)
    print(f"{name}: каналы {big.indices.tolist()}, модуль "
          f"{[round(v, 1) for v in big.values.tolist()]}, медиана {top.median():.1f}")
# Вывод в книге:
# | k: каналы [19, 50, 21, 55], модуль [15.1, 14.2, 5.4, 5.0], медиана 3.1
# | v: каналы [31, 51, 18, 14], модуль [3.3, 3.3, 3.2, 3.2], медиана 2.5

# %% Листинг «Квантованный KV-кэш BookLM: перплексия» — раздел «Ключи по каналам, значения по токенам»
from booklm.lab import perplexity

def kv_quantized(model, k_fn, v_fn):           # подменяем выходы k и v
    hooks = []
    for block in model.blocks:
        a = block.attention
        hooks.append(a.k.register_forward_hook(lambda m, i, out: k_fn(out)))
        hooks.append(a.v.register_forward_hook(lambda m, i, out: v_fn(out)))
    return hooks

def fp8(dt):                                   # масштаб 1, как в vLLM без калибровки
    return lambda x: x.to(dt).float()
def by_token(bits):                            # out: [B, T, 64], группа --- голова
    return lambda x: quantize_groups(x, bits, dim=-1)
def by_channel(bits):                          # группа --- 32 токена одного канала
    return lambda x: quantize_groups(x, bits, dim=1)

e4m3, e5m2 = fp8(torch.float8_e4m3fn), fp8(torch.float8_e5m2)
variants = [("без сжатия (FP32)", 32, None, None),
            ("FP8 E4M3", 8, e4m3, e4m3), ("FP8 E5M2", 8, e5m2, e5m2),
            ("INT4 по токенам", 5, by_token(4), by_token(4)),
            ("INT4, K по каналам", 5, by_channel(4), by_token(4)),
            ("INT2 по токенам", 3, by_token(2), by_token(2)),
            ("INT2, K по каналам", 3, by_channel(2), by_token(2)),
            ("INT2, K и V по каналам", 3, by_channel(2), by_channel(2))]
for name, bits, k_fn, v_fn in variants:
    hooks = kv_quantized(booklm, k_fn, v_fn) if k_fn else []
    print(f"{name:<23} бит: {bits:>2}   перплексия {perplexity(booklm):6.3f}")
    for h in hooks:
        h.remove()
# Вывод в книге:
# | без сжатия (FP32)       бит: 32   перплексия  6.958
# | FP8 E4M3                бит:  8   перплексия  6.965
# | FP8 E5M2                бит:  8   перплексия  6.996
# | INT4 по токенам         бит:  5   перплексия  7.172
# | INT4, K по каналам      бит:  5   перплексия  6.998
# | INT2 по токенам         бит:  3   перплексия 16.082
# | INT2, K по каналам      бит:  3   перплексия  7.934
# | INT2, K и V по каналам  бит:  3   перплексия  7.812
