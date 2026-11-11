"""Листинги главы 5 «Архитектуры современных LLM» книги «Большие языковые модели», файл 05-modern-models.

Код и вывод — ровно как в книге; текст книги сюда не входит. Листинги идут по порядку и
опираются друг на друга: запускайте файл целиком или ячейками сверху вниз (разметка # %%).
Проверка, что вывод совпал с книгой: python scripts/check_listings.py listings/ch05/05-modern-models.py
"""

# %% Листинг «Чтение config.json» — раздел «Паспорт модели»
def architecture_passport(config, default_tied=None):
    d, q_heads = config["hidden_size"], config["num_attention_heads"]
    tied = config.get("tie_word_embeddings", default_tied)
    if tied is None:              # умолчание задаёт класс модели
        raise KeyError("tie_word_embeddings")
    return {
        "V": config["vocab_size"], "d": d, "L": config["num_hidden_layers"],
        "H_q": q_heads, "tied": tied,
        "d_ff": (config.get("moe_intermediate_size")   # у части MoE
                 or config["intermediate_size"]),
        "H_kv": config.get("num_key_value_heads") or q_heads,  # нет: MHA
        "d_h": config.get("head_dim") or d // q_heads,  # нет: d/H_q
        "window": config.get("sliding_window"),
        "E": (config.get("num_local_experts")
              or config.get("num_experts") or 1),
        "k": config.get("num_experts_per_tok") or 1,
    }

mistral = architecture_passport({                      # Mistral 7B v0.1
    "vocab_size": 32000, "hidden_size": 4096, "num_hidden_layers": 32,
    "num_attention_heads": 32, "num_key_value_heads": 8,
    "intermediate_size": 14336, "tie_word_embeddings": False,
    "sliding_window": 4096})
print({key: mistral[key] for key in ("d_h", "H_kv", "window", "E", "k")})
# Вывод в книге:
# | {'d_h': 128, 'H_kv': 8, 'window': 4096, 'E': 1, 'k': 1}

# %% Листинг «Число параметров и KV-кэш» — раздел «Паспорт модели»
def parameter_count(p):                 # плотная модель: E = k = 1
    d, d_h = p["d"], p["d_h"]
    attention = (d * p["H_q"] * d_h + 2 * d * p["H_kv"] * d_h
                 + p["H_q"] * d_h * d)        # W_Q, W_K и W_V, W_O
    block = attention + 3 * d * p["d_ff"] + 2 * d  # + SwiGLU + две нормы
    return p["V"] * d * (1 if p["tied"] else 2) + p["L"] * block + d

def kv_cache_bytes(p, tokens, bytes_per_number=2):
    return 2 * p["L"] * tokens * p["H_kv"] * p["d_h"] * bytes_per_number

booklm = architecture_passport({                       # BookLM, глава 4
    "vocab_size": 8192, "hidden_size": 256, "num_hidden_layers": 6,
    "num_attention_heads": 8, "num_key_value_heads": 2,
    "intermediate_size": 768, "tie_word_embeddings": True})
for name, p in (("Mistral 7B", mistral), ("BookLM", booklm)):
    print(f"{name:10} d_h={p['d_h']:<4} H_kv={p['H_kv']:<2} "
          f"{parameter_count(p):>13,} параметров  "
          f"{kv_cache_bytes(p, tokens=1) / 1024} КБ/токен")
# Вывод в книге:
# | Mistral 7B d_h=128  H_kv=8  7,241,732,096 параметров  128.0 КБ/токен
# | BookLM     d_h=32   H_kv=2      6,622,464 параметров  1.5 КБ/токен

# %% Листинг «Конфигурация без двух полей» — раздел «Паспорт модели»
llama = {"vocab_size": 32000, "hidden_size": 4096,     # размеры LLaMA 7B
         "num_hidden_layers": 32, "num_attention_heads": 32,
         "intermediate_size": 11008}                   # двух полей нет
try:
    architecture_passport(llama)
except KeyError as err:
    print("нет поля", err)
p = architecture_passport(llama, default_tied=False)
print(f"H_kv={p['H_kv']}, параметров {parameter_count(p):,}, "
      f"кэш на токен {kv_cache_bytes(p, tokens=1) / 1024} КБ")
# Вывод в книге:
# | нет поля 'tie_word_embeddings'
# | H_kv=32, параметров 6,738,415,616, кэш на токен 512.0 КБ

# %% Листинг «BookLM в формате Hugging Face» — раздел «Паспорт модели»
import json, torch
from pathlib import Path
from transformers import AutoModelForCausalLM
from booklm.lab import load, prompt_ids

folder = Path("models/booklm/hf/base")      # экспорт booklm/export_hf.py
config = json.loads((folder / "config.json").read_text())
print(config["model_type"], config["architectures"], config["rope_parameters"])
p = architecture_passport(config)
tok, model = load("base")
hf = AutoModelForCausalLM.from_pretrained(folder, dtype=torch.float32)
print(f"параметров: по полям {parameter_count(p):,}, BookLM {model.num_params():,},"
      f" LlamaForCausalLM {hf.num_parameters():,}")
print(f"кэш на 512 токенов {kv_cache_bytes(p, 512) // 1024} КБ, файл весов "
      f"{(folder / 'model.safetensors').stat().st_size:,} байт")
# Вывод в книге:
# | llama ['LlamaForCausalLM'] {'rope_theta': 10000.0, 'rope_type': 'default'}
# | параметров: по полям 6,622,464, BookLM 6,622,464, LlamaForCausalLM 6,622,464
# | кэш на 512 токенов 768 КБ, файл весов 13,250,992 байт

# %% Листинг «Логиты и жадное продолжение двух классов» — раздел «Паспорт модели»
@torch.no_grad()
def greedy(f, ids, n):                  # f(ids) возвращает логиты [1, T, V]
    for _ in range(n):
        ids = torch.cat([ids, f(ids)[:, -1:].argmax(-1)], dim=1)
    return ids[0, -n:]

llama = lambda ids: hf(ids).logits
ids = torch.tensor([prompt_ids(tok, "Мама мыла раму, а потом она пошла спать.")])
theirs = greedy(llama, ids, 30)
for weights in ("FP32", "BF16"):
    if weights == "BF16":               # округлим и наши веса, как в файле
        for w in model.parameters():
            w.data = w.data.bfloat16().float()
    with torch.no_grad():
        gap = (llama(ids) - model(ids)).abs().max()
    same = int((greedy(model, ids, 30) == theirs).cumprod(0).sum())
    print(f"наши веса в {weights}: логиты расходятся на {gap:.1e}, "
          f"совпали первые {same} токенов из 30")
print(tok.decode(theirs.tolist()).strip())
# Вывод в книге:
# | наши веса в FP32: логиты расходятся на 9.0e-02, совпали первые 29 токенов из 30
# | наши веса в BF16: логиты расходятся на 1.3e-05, совпали первые 30 токенов из 30
# | Она была очень тихой и не хотела спать.
# | Мама сказала: «Давай, давай, давай, пойдём спать».
# | Маленькая девочка была так

# %% Листинг «Углы RoPE за обучающим окном» — раздел «Контекст длиннее обучающего окна»
import math

def rope_angles(T, d_h=32, base=10000.0, scale=1.0, yarn=None):
    theta = base ** (-torch.arange(0, d_h, 2).float() / d_h)
    if yarn is not None:              # yarn = (s, старое окно L)
        s, L = yarn
        turns = L * theta / (2 * math.pi)       # оборотов за L
        keep = ((turns - 1) / 31).clamp(0, 1)   # 1: не трогать
        theta = theta * keep + theta / s * (1 - keep)
    pos = torch.arange(T).float() / scale       # интерполяция
    return pos[:, None] * theta[None, :]        # [T, d_h/2]

ntk = 10000 * 2 ** (32 / 30)          # основание NTK при s = 2
print(f"{'':15}{'пара 0, шаг':>12}{'пара 8':>8}{'пара 15':>9}")
for name, a, p in (
        ("обучение, 511", rope_angles(512), 511),
        ("как есть, 1023", rope_angles(1024), 1023),
        ("интерполяция", rope_angles(1024, scale=2), 1023),
        ("NTK", rope_angles(1024, base=ntk), 1023),
        ("YaRN", rope_angles(1024, yarn=(2, 512)), 1023)):
    print(f"{name:15}{a[1, 0]:12.3f}{a[p, 8]:8.2f}{a[p, 15]:9.3f}")
# Вывод в книге:
# |                 пара 0, шаг  пара 8  пара 15
# | обучение, 511         1.000    5.11    0.091
# | как есть, 1023        1.000   10.23    0.182
# | интерполяция          0.500    5.11    0.091
# | NTK                   1.000    7.07    0.091
# | YaRN                  1.000    5.11    0.091

# %% Листинг «Перплексия BookLM на окне 1024» — раздел «Контекст длиннее обучающего окна»
from booklm.experiments.ch05_long_context import (
    position_losses, set_rope, short_halves, train_windows)

tok, model = load("base")
yarn = (rope_angles(1024, yarn=(2, 512)), 0.1 * math.log(2) + 1)
schemes = {"как есть": (rope_angles(1024), 1.0),
           "интерполяция": (rope_angles(1024, scale=2), 1.0),
           "NTK": (rope_angles(1024, base=ntk), 1.0), "YaRN": yarn}
show = lambda name, a, b: print(f"{name:24}{a:7.2f}{b:10.2f}")
print(f"{'':24}{'0-511':>7}{'512-1023':>10}")
show("окна по 512", *short_halves(model))
for name, (angles, mult) in schemes.items():
    set_rope(model, angles, mult)
    loss = position_losses(model)              # [1024]
    show(name, loss[:512].mean().exp(), loss[512:].mean().exp())
for name in ("интерполяция", "YaRN"):          # отдельный этап
    tok, model = load("base")
    set_rope(model, *schemes[name])
    loss = position_losses(train_windows(model, steps=60))
    show(name + " + 60 шагов", loss[:512].mean().exp(),
         loss[512:].mean().exp())
# Вывод в книге:
# |                           0-511  512-1023
# | окна по 512                7.20      6.70
# | как есть                   7.20    114.33
# | интерполяция               9.96      9.28
# | NTK                        7.34     12.40
# | YaRN                       7.84      7.05
# | интерполяция + 60 шагов    7.51      6.78
# | YaRN + 60 шагов            7.49      6.73

# %% Листинг «Маска скользящего окна» — раздел «Скользящее окно и разреженное внимание»
import torch
torch.set_printoptions(precision=4, sci_mode=False)

def window_mask(T, W):
    t = torch.arange(T)[:, None]          # запросы идут по строкам
    j = torch.arange(T)[None, :]          # ключи идут по столбцам
    return (j <= t) & (j > t - W)         # True = ключ виден

m = window_mask(6, 3)
print("строка 5:", m[5].int().tolist())
print("видно в строке:", m.sum(dim=1).tolist(), "всего", int(m.sum()))
print("W = T совпадает с причинной:",
      torch.equal(window_mask(6, 6), torch.ones(6, 6).tril().bool()))
# Вывод в книге:
# | строка 5: [0, 0, 0, 1, 1, 1]
# | видно в строке: [1, 2, 3, 3, 3, 3] всего 15
# | W = T совпадает с причинной: True

# %% Листинг «Дальность передачи сведений через слои» — раздел «Дальность передачи через слои»
def reach(T, W, layers):
    step = window_mask(T, W).int()
    seen = step.clone()
    for _ in range(layers - 1):
        seen = ((seen @ step) > 0).int()   # кого видели те, кого вижу я
    return seen.bool()

for layers in (1, 2, 3, 6):
    farthest = int(reach(64, 3, layers)[63].nonzero().min())
    print(f"слоёв {layers}: позиция 63 знает о позиции {farthest}, "
          f"на {63 - farthest} назад")
# Вывод в книге:
# | слоёв 1: позиция 63 знает о позиции 61, на 2 назад
# | слоёв 2: позиция 63 знает о позиции 59, на 4 назад
# | слоёв 3: позиция 63 знает о позиции 57, на 6 назад
# | слоёв 6: позиция 63 знает о позиции 51, на 12 назад

# %% Листинг «KV-кэш при чередовании локальных и глобальных слоёв» — раздел «Локальные и глобальные слои»
def layer_types(config):
    kinds, L = config.get("layer_types"), config["num_hidden_layers"]
    pattern = config.get("sliding_window_pattern")      # Gemma 3
    if kinds is None and pattern:
        kinds = ["full_attention" if (i + 1) % pattern == 0
                 else "sliding_attention" for i in range(L)]
    if kinds is None:      # нет списка: окно во всех слоях или ни в одном
        on = (config.get("sliding_window")
              and config.get("use_sliding_window", True))
        kinds = ["sliding_attention" if on else "full_attention"] * L
    return kinds

def interleaved_kv_bytes(p, kinds, tokens, bytes_per_number=2):
    kept = sum(min(tokens, p["window"]) if kind == "sliding_attention"
               else tokens for kind in kinds)          # токено-слои в кэше
    return 2 * kept * p["H_kv"] * p["d_h"] * bytes_per_number

local = dict(booklm, window=128)     # копия booklm, заменено только window
five_to_one = ["sliding_attention"] * 5 + ["full_attention"]
print("все слои полные, КБ:",
      interleaved_kv_bytes(local, ["full_attention"] * 6, 512) // 1024)
print("пять к одному, КБ:  ",
      interleaved_kv_bytes(local, five_to_one, 512) // 1024)
long = dict(booklm, window=1024)
print("доля при T = 32768:   ",
      round(interleaved_kv_bytes(long, five_to_one, 32768)
            / kv_cache_bytes(long, 32768), 3))
# Вывод в книге:
# | все слои полные, КБ: 768
# | пять к одному, КБ:   288
# | доля при T = 32768:    0.193

# %% Листинг «Как далеко смотрят слои BookLM» — раздел «Локальные и глобальные слои»
from booklm.lab import load, val_tokens

tok, model = load("base")
data = val_tokens()                               # отложенные истории
t = torch.arange(128, 512)[:, None]               # запросы с позиции 128
dist = t - torch.arange(512)[None, :]             # расстояние t - j
share = torch.zeros(6, 4)
with torch.no_grad():
    for s in range(0, 8 * 512, 512):              # 8 отрезков по 512
        ids = data[s:s + 512][None]
        _, maps = model(ids, return_weights=True)
        for layer, w in enumerate(maps):          # w: [1, головы, T, T]
            w = w[0, :, 128:]                     # [головы, 384, 512]
            for c, far in enumerate((16, 64, 128)):
                share[layer, c] += (w * (dist >= far)).sum(-1).mean()
            share[layer, 3] += (w * (ids[0] == 1)).sum(-1).mean()
share /= 8                                        # среднее по отрезкам
print("слой   >16   >64  >128 <bos>")
for layer, row in enumerate(share.tolist()):
    print(f"{layer:4}" + "".join(f"{x:6.2f}" for x in row))
# Вывод в книге:
# | слой   >16   >64  >128 <bos>
# |    0  0.83  0.64  0.44  0.01
# |    1  0.26  0.06  0.01  0.02
# |    2  0.47  0.16  0.04  0.01
# |    3  0.70  0.33  0.14  0.02
# |    4  0.31  0.11  0.04  0.02
# |    5  0.52  0.22  0.09  0.03

# %% Листинг «Перплексия BookLM со скользящим окном» — раздел «Локальные и глобальные слои»
from functools import partial
from booklm.lab import perplexity
from booklm.experiments.ch05 import masked_forward  # внимание с маской mask

def set_windows(model, windows, T=512):           # None: полная маска
    for block, W in zip(model.blocks, windows):
        block.attention.forward = partial(masked_forward, block.attention,
                                          mask=window_mask(T, W or T))

schemes = {"полная маска": [None] * 6, "окно 128 везде": [128] * 6,
           "окно 64 везде": [64] * 6, "окно 16 везде": [16] * 6,
           "пять к одному, 64": [64] * 5 + [None]}
for name, windows in schemes.items():
    set_windows(model, windows)
    kinds = ["full_attention" if W is None else "sliding_attention"
             for W in windows]
    p = dict(booklm, window=windows[0] or 512)
    print(f"{name:18} перплексия {perplexity(model):5.2f}  "
          f"кэш {interleaved_kv_bytes(p, kinds, 512) // 1024:3} КБ")
# Вывод в книге:
# | полная маска       перплексия  6.96  кэш 768 КБ
# | окно 128 везде     перплексия  7.00  кэш 192 КБ
# | окно 64 везде      перплексия  7.33  кэш  96 КБ
# | окно 16 везде      перплексия 10.57  кэш  24 КБ
# | пять к одному, 64  перплексия  7.18  кэш 208 КБ

# %% Листинг «Внимание по k лучшим ключам» — раздел «Разреженное внимание»
def topk_attention(scores, values, k):  # scores, values: [T]
    keep = scores.topk(k).indices       # номера k лучших ключей
    w = scores[keep].softmax(dim=0)     # softmax только по ним
    return keep, (w * values[keep]).sum()

scores = torch.tensor([2.0, 0.0, 1.0, -1.0, 3.0, 0.5])
values = torch.tensor([1.0, 2.0, 3.0, 4.0, 0.0, 2.0])
full = scores.softmax(dim=0)
print([round(x, 3) for x in full.tolist()],
      f"выход {(full * values).sum():.3f}")
for k in (2, 3):
    keep, y = topk_attention(scores, values, k)
    print(f"k={k}: ключи {keep.tolist()}, их вес "
          f"{full[keep].sum():.3f}, выход {y:.3f}")
# Вывод в книге:
# | [0.222, 0.03, 0.082, 0.011, 0.605, 0.05] выход 0.672
# | k=2: ключи [4, 0], их вес 0.827, выход 0.269
# | k=3: ключи [4, 0, 2], их вес 0.909, выход 0.515

# %% Листинг «Перплексия BookLM при внимании по k ключам» — раздел «Разреженное внимание»
from booklm.experiments.ch05_long_context import topk_forward

tok, model = load("base")                 # свежая модель, полная маска
print(f"полное внимание: перплексия {perplexity(model):.2f}")
print(f"{'k':>3}{'окно k':>8}{'top-k':>7}{'вес':>6}"
      f"{'общий top-k':>13}{'вес':>6}")
for k in (8, 16, 32, 64):
    set_windows(model, [k] * 6)           # k ближайших ключей
    row = [perplexity(model)]
    for shared in (False, True):          # свой выбор у головы / общий
        mass = []                         # доли веса по слоям
        for b in model.blocks:
            b.attention.forward = partial(topk_forward, b.attention,
                                          k=k, shared=shared, stats=mass)
        row += [perplexity(model), sum(mass) / len(mass)]
    print(f"{k:3}{row[0]:8.2f}{row[1]:7.2f}{row[2]:6.2f}"
          f"{row[3]:13.2f}{row[4]:6.2f}")
# Вывод в книге:
# | полное внимание: перплексия 6.96
# |   k  окно k  top-k   вес  общий top-k   вес
# |   8   13.80   8.70  0.60         8.49  0.50
# |  16   10.57   7.71  0.71         7.55  0.63
# |  32    8.34   7.31  0.81         7.17  0.75
# |  64    7.33   7.11  0.89         7.02  0.85

# %% Листинг «Маршрут одного токена» — раздел «Маршрут одного токена»
def route(scores, k):
    chosen = scores.topk(k).indices          # [k]
    weights = scores[chosen].softmax(dim=0)  # softmax только по выбранным
    return chosen, weights

r = torch.tensor([2.0, 0.5, 1.0, -1.0])
chosen, weights = route(r, k=2)
print(chosen, weights)
p_all = r.softmax(dim=0)                     # вторая запись маршрута
print(p_all, p_all[chosen].sum())
print(p_all[chosen] / p_all[chosen].sum())
# Вывод в книге:
# | tensor([0, 2]) tensor([0.7311, 0.2689])
# | tensor([0.6095, 0.1360, 0.2242, 0.0303]) tensor(0.8337)
# | tensor([0.7311, 0.2689])

# %% Листинг «Какие токены пришли к эксперту 0» — раздел «Групповой проход по экспертам»
indices = torch.tensor([[0, 2],     # токен 0: эксперты 0 и 2
                        [0, 1],     # токен 1: эксперты 0 и 1
                        [2, 0]])    # токен 2: сначала 2, потом 0
hit = indices == 0                  # та же форма [N, k]
print(hit)
rows, slot = hit.nonzero(as_tuple=True)
print(rows, slot)
# Вывод в книге:
# | tensor([[ True, False],
# |         [ True, False],
# |         [False,  True]])
# | tensor([0, 1, 2]) tensor([0, 0, 1])

# %% Листинг «Групповой проход по экспертам» — раздел «Групповой проход по экспертам»
def moe_forward(x, router, experts, k):      # x: [N, d], N = B * T
    top = router(x).topk(k, dim=-1)  # поля .values и .indices: [N, k]
    weights = top.values.softmax(dim=-1)     # [N, k]
    y = torch.zeros_like(x)
    for e, expert in enumerate(experts):
        rows, slot = (top.indices == e).nonzero(as_tuple=True)
        if rows.numel():                     # один вызов на эксперта
            # None добавляет ось: [n] -> [n, 1], доля множит всю строку
            y[rows] += weights[rows, slot, None] * expert(x[rows])
    return y

torch.manual_seed(5)
router = torch.nn.Linear(8, 4, bias=False)
experts = [torch.nn.Linear(8, 8, bias=False)       # вместо SwiGLU
           for _ in range(4)]
x = torch.randn(10, 8)
with torch.no_grad():
    y = moe_forward(x, router, experts, k=2)
    naive = []
    for row in x:                            # эталон: токен за токеном
        c, w = route(router(row), k=2)
        naive.append(w[0] * experts[int(c[0])](row)
                     + w[1] * experts[int(c[1])](row))
    # bincount: сколько раз встретился каждый номер эксперта
    load = torch.bincount(router(x).topk(2).indices.flatten(), minlength=4)
print("строк у экспертов:", load.tolist())
print("совпало с циклом:", torch.allclose(y, torch.stack(naive), atol=1e-6))
# Вывод в книге:
# | строк у экспертов: [5, 5, 3, 7]
# | совпало с циклом: True

# %% Листинг «Общее и активное число параметров» — раздел «Общее и активное число параметров»
def moe_parameter_count(p, active=False):
    extra = (p["k"] if active else p["E"]) - 1   # сверх одной SwiGLU
    router = p["d"] * p["E"] if p["E"] > 1 else 0
    added = extra * 3 * p["d"] * p["d_ff"] + router   # на один блок
    return parameter_count(p) + p["L"] * added

for name, p in (("BookLM", booklm), ("BookLM-MoE", dict(booklm, E=4, k=2)),
                ("Mixtral 8x7B", dict(mistral, E=8, k=2, window=None))):
    print(f"{name:12} всего {moe_parameter_count(p):>14,}"
          f"  активных {moe_parameter_count(p, active=True):>14,}")
# Вывод в книге:
# | BookLM       всего      6,622,464  активных      6,622,464
# | BookLM-MoE   всего     17,245,440  активных     10,167,552
# | Mixtral 8x7B всего 46,702,792,704  активных 12,879,925,248

# %% Листинг «Смесь экспертов внутри BookLM» — раздел «Общее и активное число параметров»
import torch.nn as nn
from booklm.model import BookLM, SwiGLU

class MoEFeedForward(nn.Module):           # смесь вместо одной SwiGLU
    def __init__(self, cfg, E=4, k=2):
        super().__init__()
        self.router = nn.Linear(cfg.d_model, E, bias=False)
        self.experts = nn.ModuleList(SwiGLU(cfg) for _ in range(E))
        self.k = k

    def forward(self, x):                   # x: [B, T, d]
        rows = x.reshape(-1, x.size(-1))    # [B * T, d]
        y = moe_forward(rows, self.router, self.experts, self.k)
        return y.view_as(x)                 # обратно в [B, T, d]

moe = BookLM()
for block in moe.blocks:
    block.ffn = MoEFeedForward(moe.cfg)
print(f"в модели {moe.num_params():,} параметров, по формуле "
      f"{moe_parameter_count(dict(booklm, E=4, k=2)):,}")
print("логиты:", tuple(moe(torch.randint(8192, (2, 16))).shape))
# Вывод в книге:
# | в модели 17,245,440 параметров, по формуле 17,245,440
# | логиты: (2, 16, 8192)

# %% Листинг «Сколько нейронов SwiGLU работает на токене» — раздел «Общее и активное число параметров»
from booklm.lab import load
from booklm.experiments.ch05 import stories   # отложенные истории

tok, model = load("base")                    # свежая модель, без окон
acts = [[] for _ in model.blocks]            # входы down по слоям
hooks = [b.ffn.down.register_forward_pre_hook(
             lambda m, inp, i=i: acts[i].append(inp[0][0]))
         for i, b in enumerate(model.blocks)]
with torch.no_grad():
    for story in stories(20):
        model(story[None])
for h in hooks:
    h.remove()
acts = [torch.cat(a) for a in acts]          # [токены, 768]
need = []
for a in acts:                               # сколько нейронов несут 90 %
    mass = a.abs().sort(dim=1, descending=True).values.cumsum(1)
    count = (mass < 0.9 * mass[:, -1:]).sum(1) + 1   # у каждого токена
    need.append(round(count.float().mean().item() / 768, 2))
print("токенов:", len(acts[0]))
print("доля нейронов на 90 % активации:", need)
# Вывод в книге:
# | токенов: 3315
# | доля нейронов на 90 % активации: [0.47, 0.54, 0.54, 0.53, 0.53, 0.51]

# %% Листинг «Группы нейронов как эксперты» — раздел «Общее и активное число параметров»
import torch.nn.functional as F
from booklm.experiments.ch05 import balanced_kmeans          # группы по 192
from booklm.experiments.ch05 import oracle_route, top_neurons  # крючки

torch.manual_seed(0)
groups = [balanced_kmeans(F.normalize(a.abs().T, dim=1)) for a in acts]
shuffled = [torch.randperm(768) % 4 for _ in acts]   # случайные группы

def ppl_with(hook):                  # hook(i): крючок перед down в блоке i
    hs = [b.ffn.down.register_forward_pre_hook(hook(i))
          for i, b in enumerate(model.blocks)]
    value = perplexity(model)
    for h in hs:
        h.remove()
    return value

print(f"{'все 768 нейронов':27}{perplexity(model):6.2f}")
for name, g in (("группы по сходству", groups),
                ("случайные группы", shuffled)):
    route2 = lambda i: partial(oracle_route, label=g[i], k=2)
    print(f"{name + ', k=2':27}{ppl_with(route2):6.2f}")
for n in (384, 192):
    best = lambda i: partial(top_neurons, n=n)
    print(f"{f'свои {n} нейрона':27}{ppl_with(best):6.2f}")
# Вывод в книге:
# | все 768 нейронов             6.96
# | группы по сходству, k=2     13.92
# | случайные группы, k=2       17.72
# | свои 384 нейрона             7.04
# | свои 192 нейрона             7.51

# %% Листинг «Градиент маршрутизатора при k = 2 и k = 1» — раздел «Перегрузка экспертов и балансировка нагрузки»
torch.manual_seed(0)
x = torch.randn(2, 16, 256)
for k in (2, 1):
    layer = MoEFeedForward(moe.cfg, k=k)
    layer(x).sum().backward()               # градиенты всех весов слоя
    print(f"k={k}: сумма |градиента| маршрутизатора "
          f"{layer.router.weight.grad.abs().sum():.1f}")
# Вывод в книге:
# | k=2: сумма |градиента| маршрутизатора 1909.6
# | k=1: сумма |градиента| маршрутизатора 0.0

# %% Листинг «Ёмкость эксперта» — раздел «Перегрузка экспертов и балансировка нагрузки»
import math

def capacity(tokens, k, experts, factor):
    return math.ceil(factor * tokens * k / experts)

def dispatch(chosen, experts, cap):
    load, kept = [0] * experts, []
    for token, pair in enumerate(chosen.tolist()):  # по порядку
        for e in pair:
            if load[e] < cap:
                load[e] += 1
                kept.append((token, e))
    return kept, load

# по два эксперта у каждого из восьми токенов: [N=8, k=2]
chosen = torch.tensor([[0, 2], [0, 1], [0, 2], [0, 3],
                       [0, 1], [0, 2], [0, 2], [1, 3]])
for factor in (1.0, 1.25):
    cap = capacity(8, 2, 4, factor)
    kept, load = dispatch(chosen, 4, cap)
    print(f"c={factor}: C={cap}, принято {load},",
          f"отброшено {16 - len(kept)}")
# Вывод в книге:
# | c=1.0: C=4, принято [4, 3, 4, 2], отброшено 3
# | c=1.25: C=5, принято [5, 3, 4, 2], отброшено 2

# %% Листинг «Балансирующая ошибка» — раздел «Перегрузка экспертов и балансировка нагрузки»
def balance_loss(probs, chosen, experts):
    counts = torch.bincount(chosen.flatten(), minlength=experts)
    f = counts.float() / chosen.numel()        # доли назначений
    return experts * (f * probs.mean(dim=0)).sum()

even = balance_loss(torch.full((8, 4), 0.25),
                    torch.tensor([[0, 1], [2, 3]] * 4), 4)
skewed = balance_loss(torch.tensor([[0.97, 0.01, 0.01, 0.01]] * 8),
                      torch.tensor([[0, 1]] * 8), 4)
print(f"ровный пакет {even.item():.2f}, перекошенный {skewed.item():.2f}")
# Вывод в книге:
# | ровный пакет 1.00, перекошенный 1.96

# %% Листинг «Частота выбора экспертов» — раздел «Удаление и слияние экспертов»
import numpy as np
from booklm.lab import MODELS
from booklm.model import BookLMConfig

cfg16 = BookLMConfig(d_ff=192)                # эксперт: 192 нейрона, а не 768
moe16 = BookLM(cfg16)
for block in moe16.blocks:
    block.ffn = MoEFeedForward(cfg16, E=16, k=4)
moe16.load_state_dict(torch.load(MODELS / "experiments/ch05-moe16-mix.pt"))
moe16.eval()

def expert_shares(model, texts):              # доля выборов эксперта по слоям
    counts = [torch.zeros(16) for _ in model.blocks]
    def counter(i):
        def hook(layer, inp):                 # вход слоя смеси: [1, T, d]
            top = layer.router(inp[0]).topk(layer.k, dim=-1).indices
            counts[i] += torch.bincount(top.flatten(), minlength=16)
        return hook
    hs = [b.ffn.register_forward_pre_hook(counter(i))
          for i, b in enumerate(model.blocks)]
    with torch.no_grad():
        for t in texts:
            model(t[None])
    for h in hs:
        h.remove()
    return [c / c.sum() for c in counts]

code = np.fromfile(MODELS / "experiments/ch05-tech-test.bin", np.uint16)
code = torch.from_numpy(code.astype(np.int64))[:14 * 512].view(14, 512)
tales = expert_shares(moe16, stories(40))     # калибровка: 40 сказок
techs = expert_shares(moe16, code)            # 14 отрезков кода
order = tales[0].argsort(descending=True)     # слой 0, от частых к редким
print("сказки, %:", (100 * tales[0][order]).round().int().tolist())
print("код,    %:", (100 * techs[0][order]).round().int().tolist())
print(f"перплексия на сказках {perplexity(moe16):.2f}")
# Вывод в книге:
# | сказки, %: [17, 15, 14, 13, 10, 7, 5, 5, 4, 4, 3, 1, 1, 1, 1, 0]
# | код,    %: [0, 1, 2, 2, 6, 9, 1, 11, 6, 15, 6, 6, 6, 9, 11, 9]
# | перплексия на сказках 7.17

# %% Листинг «Удаление редких экспертов» — раздел «Удаление и слияние экспертов»
import copy

def prune(layer, keep):                       # keep: номера оставленных экспертов
    keep = sorted(keep)
    layer.router.weight = nn.Parameter(layer.router.weight[keep].clone())
    layer.experts = nn.ModuleList(layer.experts[i] for i in keep)

for m in (16, 12, 8, 6, 4):
    small = copy.deepcopy(moe16)
    for block, share in zip(small.blocks, tales):
        prune(block.ffn, share.topk(m).indices.tolist())   # m самых частых
    cover = sum(s.topk(m).values.sum() for s in tales) / len(tales)
    print(f"m={m:2}  параметров {small.num_params():>10,}  "
          f"покрытие {cover:.3f}  перплексия {perplexity(small):6.2f}")
# Вывод в книге:
# | m=16  параметров 17,263,872  покрытие 1.000  перплексия   7.17
# | m=12  параметров 13,718,784  покрытие 0.866  перплексия   8.10
# | m= 8  параметров 10,173,696  покрытие 0.658  перплексия  16.56
# | m= 6  параметров  8,401,152  покрытие 0.531  перплексия  26.38
# | m= 4  параметров  6,628,608  покрытие 0.383  перплексия 201.14

# %% Листинг «Рост весов и энтропия внимания» — раздел «Рост оценок внимания»
torch.manual_seed(0)
x = torch.randn(64, 256)                          # 64 токена, d = 256
W_q, W_k = torch.randn(256, 64) / 16, torch.randn(256, 64) / 16
rms = lambda z: z / z.pow(2).mean(-1, keepdim=True).sqrt()   # RMSNorm без g

def report(name, scores):                         # scores: [64, 64]
    a = scores.softmax(dim=-1)
    entropy = -(a * a.clamp_min(1e-12).log()).sum(-1).mean()
    print(f"  {name:13}макс. оценка {scores.max():6.1f}  энтропия {entropy:4.2f}"
          f"  доля лучшего ключа {a.max(-1).values.mean():.2f}")

for s in (1, 3, 10):                              # во сколько раз выросли W_Q и W_K
    q, k = x @ (s * W_q), x @ (s * W_k)
    plain = q @ k.T / 8                           # делим на корень из d_h = 64
    print(f"W_Q и W_K умножены на {s}:")
    report("как есть", plain)
    report("tanh, c = 50", 50 * torch.tanh(plain / 50))
    report("QK-norm", rms(q) @ rms(k).T / 8)
print(f"при равных долях энтропия ln 64 = {math.log(64):.2f}")
# Вывод в книге:
# | W_Q и W_K умножены на 1:
# |   как есть     макс. оценка    3.5  энтропия 3.72  доля лучшего ключа 0.10
# |   tanh, c = 50 макс. оценка    3.5  энтропия 3.72  доля лучшего ключа 0.10
# |   QK-norm      макс. оценка    3.3  энтропия 3.71  доля лучшего ключа 0.10
# | W_Q и W_K умножены на 3:
# |   как есть     макс. оценка   31.4  энтропия 0.53  доля лучшего ключа 0.81
# |   tanh, c = 50 макс. оценка   27.8  энтропия 0.61  доля лучшего ключа 0.78
# |   QK-norm      макс. оценка    3.3  энтропия 3.71  доля лучшего ключа 0.10
# | W_Q и W_K умножены на 10:
# |   как есть     макс. оценка  348.4  энтропия 0.05  доля лучшего ключа 0.98
# |   tanh, c = 50 макс. оценка   50.0  энтропия 2.30  доля лучшего ключа 0.17
# |   QK-norm      макс. оценка    3.3  энтропия 3.71  доля лучшего ключа 0.10
# | при равных долях энтропия ln 64 = 4.16

# %% Листинг «QK-clip на одной голове» — раздел «Рост оценок внимания»
def qk_clip(W_q, W_k, x, tau=100.0):
    s_max = ((x @ W_q) @ (x @ W_k).T / 8).max()  # наибольшая оценка головы
    gamma = min(1.0, tau / s_max.item())
    return W_q * gamma ** 0.5, W_k * gamma ** 0.5   # каждой матрице корень

clipped_q, clipped_k = qk_clip(10 * W_q, 10 * W_k, x)
report("QK-clip", (x @ clipped_q) @ (x @ clipped_k).T / 8)
# Вывод в книге:
# |   QK-clip      макс. оценка  100.0  энтропия 0.17  доля лучшего ключа 0.93

# %% Листинг «Сток в знаменателе и выходные ворота» — раздел «Стоки внимания и выходные ворота»
scores = torch.zeros(3)                       # ни один ключ не подходит
values = torch.tensor([[2.0, 0.0], [0.0, 2.0], [-1.0, 1.0]])
plain = scores.softmax(dim=0)
print("обычный softmax: доли", [round(v, 3) for v in plain.tolist()],
      "выход", [round(v, 3) for v in (plain @ values).tolist()])
for b in (0.0, 4.0):                          # обучаемый сток головы
    shares = scores.exp() / (math.exp(b) + scores.exp().sum())
    print(f"сток b = {b}: сумма долей {shares.sum():.3f},",
          "выход", [round(v, 3) for v in (shares @ values).tolist()])
gate = torch.sigmoid(torch.tensor(-4.0))      # ворота по самому токену
print(f"ворота {gate:.3f}: выход", [round(v, 3) for v in (gate * plain @ values).tolist()])
# Вывод в книге:
# | обычный softmax: доли [0.333, 0.333, 0.333] выход [0.333, 1.0]
# | сток b = 0.0: сумма долей 0.750, выход [0.25, 0.75]
# | сток b = 4.0: сумма долей 0.052, выход [0.017, 0.052]
# | ворота 0.018: выход [0.006, 0.018]

# %% Листинг «Доля внимания первому токену у голов BookLM» — раздел «Стоки внимания и выходные ворота»
from booklm.lab import load, val_tokens           # имя load занято в разделе о MoE
tok, model = load("base")
data, first = val_tokens(), torch.zeros(6, 8)       # [слои, головы]
with torch.no_grad():
    for s in range(0, 8 * 512, 512):               # 8 окон по 512
        _, maps = model(data[s:s + 512][None], return_weights=True)
        for layer, w in enumerate(maps):           # w: [1, головы, T, T]
            first[layer] += w[0, :, 64:, 0].mean(-1) / 8
even = sum(1 / (t + 1) for t in range(64, 512)) / 448
print(f"при равных долях {even:.4f}")
print("среднее по головам:  ", [round(v, 4) for v in first.mean(1).tolist()])
print("самая сильная голова:", [round(v, 4) for v in first.max(1).values.tolist()])
# Вывод в книге:
# | при равных долях 0.0046
# | среднее по головам:   [0.0041, 0.0004, 0.0008, 0.0019, 0.0015, 0.0016]
# | самая сильная голова: [0.0051, 0.0023, 0.0014, 0.0029, 0.0036, 0.0037]

# %% Листинг «Окно с первыми токенами и без них» — раздел «Стоки внимания и выходные ворота»
def sink_window_mask(T, W, sinks):            # окно плюс sinks первых позиций
    mask = window_mask(T, W)
    mask[:, :sinks] = torch.ones(T, T, dtype=torch.bool).tril()[:, :sinks]
    return mask

for W in (16, 64):
    for sinks in (0, 1, 4):
        for block in model.blocks:
            block.attention.forward = partial(masked_forward, block.attention,
                                              mask=sink_window_mask(512, W, sinks))
        print(f"окно {W:2}, первых токенов {sinks}: перплексия {perplexity(model):.2f}")
# Вывод в книге:
# | окно 16, первых токенов 0: перплексия 10.57
# | окно 16, первых токенов 1: перплексия 10.54
# | окно 16, первых токенов 4: перплексия 10.52
# | окно 64, первых токенов 0: перплексия 7.33
# | окно 64, первых токенов 1: перплексия 7.32
# | окно 64, первых токенов 4: перплексия 7.32

# %% Листинг «Синкхорн и усиление через слои» — раздел «Гиперсвязи»
def sinkhorn(logits, iters=20):               # проходы Синкхорна-Кноппа, как в mHC
    m = logits.exp()                          # все числа положительны
    for _ in range(iters):
        m = m / m.sum(dim=1, keepdim=True)    # строки в сумме 1
        m = m / m.sum(dim=0, keepdim=True)    # столбцы в сумме 1
    return m

torch.manual_seed(0)
free = [torch.eye(4) + 0.3 * torch.randn(4, 4) for _ in range(60)]
mixed = [sinkhorn(torch.randn(4, 4)) for _ in range(60)]
print("суммы строк", [round(v, 3) for v in mixed[0].sum(1).tolist()],
      "столбцов", [round(v, 3) for v in mixed[0].sum(0).tolist()])
for name, mats in (("без ограничений", free), ("дважды стохастические", mixed)):
    total, gains = torch.eye(4), []
    for l, m in enumerate(mats, 1):
        total = m @ total                     # смешивание за l слоёв
        if l in (1, 10, 30, 60):
            gains.append(round(total.abs().sum(dim=1).max().item(), 2))
    print(f"{name:22} усиление за 1/10/30/60 слоёв: {gains}")
# Вывод в книге:
# | суммы строк [1.0, 1.0, 1.0, 1.0] столбцов [1.0, 1.0, 1.0, 1.0]
# | без ограничений        усиление за 1/10/30/60 слоёв: [2.19, 2.09, 40.05, 854.63]
# | дважды стохастические  усиление за 1/10/30/60 слоёв: [1.0, 1.0, 1.0, 1.0]

# %% Листинг «KV-кэш Gemma 3n с общими слоями» — раздел «Общий KV-кэш слоёв и Gemma 3n»
gemma3n = {"H_kv": 2, "d_h": 256, "window": 512}           # Gemma 3n E4B
kinds35 = (["sliding_attention"] * 4 + ["full_attention"]) * 7   # 35 слоёв
for name, stored in (("свой кэш у всех 35 слоёв", kinds35),
                     ("кэш только у первых 20", kinds35[:20])):
    mb = interleaved_kv_bytes(gemma3n, stored, 32768) / 2**20
    print(f"{name:25} {mb:6.1f} МБ")
# Вывод в книге:
# | свой кэш у всех 35 слоёв   476.0 МБ
# | кэш только у первых 20     272.0 МБ

# %% Листинг «Первые 384 нейрона SwiGLU у BookLM» — раздел «Общий KV-кэш слоёв и Gemma 3n»
def first_neurons(module, inputs, m):         # крючок перед down: нейроны с m-го молчат
    a = inputs[0].clone()
    a[..., m:] = 0
    return (a,)

from booklm.lab import load
tok, model = load("base")
print(f"первые 384 нейрона: перплексия {ppl_with(lambda i: partial(first_neurons, m=384)):.2f}")
# Вывод в книге:
# | первые 384 нейрона: перплексия 57.46

# %% Листинг «Повтор блоков BookLM без обучения» — раздел «Петлевые модели»
from booklm.lab import load
tok, model = load("base")
blocks = list(model.blocks)
for name, order in (("обычный проход", range(6)), ("блок 3 дважды", [0, 1, 2, 3, 3, 4, 5]),
                    ("все блоки дважды", list(range(6)) * 2)):
    model.blocks = nn.Sequential(*[blocks[i] for i in order])   # те же веса
    print(f"{name:17} перплексия {perplexity(model):.2f}")
# Вывод в книге:
# | обычный проход    перплексия 6.96
# | блок 3 дважды     перплексия 8.02
# | все блоки дважды  перплексия 8.74

# %% Листинг «Поля config.json Qwen3.5-35B-A3B» — раздел «Блок Qwen3.5»
qwen35 = {                                     # text_config Qwen3.5-35B-A3B
    "hidden_size": 2048, "num_hidden_layers": 40,
    "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 10,
    "num_attention_heads": 16, "num_key_value_heads": 2, "head_dim": 256,
    "attn_output_gate": True, "partial_rotary_factor": 0.25,
    "linear_num_key_heads": 16, "linear_num_value_heads": 32,
    "linear_key_head_dim": 128, "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4, "num_experts": 256, "num_experts_per_tok": 8,
    "moe_intermediate_size": 512, "shared_expert_intermediate_size": 512}
cfg, n = qwen35, 262144                       # n токенов в контексте
n_full = cfg["layer_types"].count("full_attention")
n_linear = cfg["num_hidden_layers"] - n_full
kv = 2 * n_full * n * cfg["num_key_value_heads"] * cfg["head_dim"] * 2   # BF16
state = n_linear * cfg["linear_num_value_heads"] * 128 * 128 * 4        # FP32
rotated = int(cfg["head_dim"] * cfg["partial_rotary_factor"])
print(f"слоёв: {n_linear} Gated DeltaNet и {n_full} внимания")
print(f"поворачивает RoPE {rotated} координат из {cfg['head_dim']}")
print(f"кэш на {n} токенов {kv / 2**30:.2f} ГБ, если бы внимание было везде "
      f"{kv * 4 / 2**30:.2f} ГБ; состояния {state / 2**20:.1f} МБ")
# Вывод в книге:
# | слоёв: 30 Gated DeltaNet и 10 внимания
# | поворачивает RoPE 64 координат из 256
# | кэш на 262144 токенов 5.00 ГБ, если бы внимание было везде 20.00 ГБ; состояния 60.0 МБ

# %% Листинг «Слой Gated DeltaNet» — раздел «Блок Qwen3.5»
def gated_delta_net(x, p, l2=True):               # x: [T, d], одна голова
    qkv = x @ p["W_qkv"]                          # [T, 3 d_k]: q, k, v рядом
    c = qkv.size(1)                               # свёртка по времени, ядро 4
    qkv = F.conv1d(F.pad(qkv.T, (3, 0)), p["conv"][:, None], groups=c).T  # у канала свои веса
    q, k, v = F.silu(qkv).chunk(3, dim=-1)
    if l2:                                        # запрос и ключ длины 1
        q, k = F.normalize(q, dim=-1, eps=1e-6), F.normalize(k, dim=-1, eps=1e-6)
    alpha = torch.exp(-p["A"] * F.softplus(x @ p["w_a"] + p["dt"]))   # (0, 1)
    beta = torch.sigmoid(x @ p["w_b"])            # сила записи, (0, 1)
    S, out, sizes = torch.zeros(k.size(1), v.size(1)), [], []
    for t in range(len(x)):                       # правило дельты с затуханием
        S = alpha[t] * S
        S = S + beta[t] * torch.outer(k[t], v[t] - k[t] @ S)
        out.append(q[t] @ S / math.sqrt(k.size(1)))
        sizes.append(S.norm())
    o = F.rms_norm(torch.stack(out), (v.size(1),), eps=1e-6)   # норма выхода
    return (o * F.silu(x @ p["W_z"])) @ p["W_out"], torch.stack(sizes)

torch.manual_seed(0)
d, d_k = 32, 16
p = {"W_qkv": torch.randn(d, 3 * d_k) / d ** 0.5, "conv": torch.randn(3 * d_k, 4) / 2,
     "w_a": torch.randn(d) / d ** 0.5, "dt": torch.tensor(0.0), "A": torch.tensor(0.1),
     "w_b": torch.randn(d) / d ** 0.5, "W_z": torch.randn(d, d_k) / d ** 0.5,
     "W_out": torch.randn(d_k, d) / d_k ** 0.5}
x = torch.randn(64, d)
for l2 in (True, False):
    y, sizes = gated_delta_net(x, p, l2)
    print(f"L2={l2}: длина S на шагах 8/32/64:",
          [f"{sizes[i]:.3g}" for i in (7, 31, 63)], tuple(y.shape))
# Вывод в книге:
# | L2=True: длина S на шагах 8/32/64: ['2.84', '3.14', '2.77'] (64, 32)
# | L2=False: длина S на шагах 8/32/64: ['19.4', '483', '6.07e+03'] (64, 32)

# %% Листинг «Разбор config.json модели gpt-oss-120b» — раздел «Разбор незнакомой модели»
gpt_oss = {                                     # config.json gpt-oss-120b
    "vocab_size": 201088, "hidden_size": 2880, "num_hidden_layers": 36,
    "num_attention_heads": 64, "num_key_value_heads": 8, "head_dim": 64,
    "intermediate_size": 2880, "tie_word_embeddings": False,
    "num_local_experts": 128, "num_experts_per_tok": 4,
    "sliding_window": 128, "attention_bias": True,
    "layer_types": ["sliding_attention", "full_attention"] * 18}
p = architecture_passport(gpt_oss)
total = moe_parameter_count(p)
active = moe_parameter_count(p, active=True)
embedding = p["V"] * p["d"]
print(f"всего {total / 1e9:.2f} млрд, активных {active / 1e9:.2f} млрд, "
      f"без эмбеддингов {(active - embedding) / 1e9:.2f} млрд")
kv = interleaved_kv_bytes(p, layer_types(gpt_oss), 131072)
print(f"KV-кэш на 131072 токена: {kv / 2**30:.2f} ГБ, "
      f"без окон {kv_cache_bytes(p, 131072) / 2**30:.1f} ГБ")
in_experts = p["L"] * p["E"] * 3 * p["d"] * p["d_ff"]
# байты: 4,25 бита / 8 в MXFP4, по 2 байта в BF16
size = in_experts * 4.25 / 8 + (total - in_experts) * 2
print(f"в экспертах {in_experts / total:.1%} весов, "
      f"файл {size / 2**30:.1f} ГБ")
# Вывод в книге:
# | всего 116.79 млрд, активных 5.71 млрд, без эмбеддингов 5.13 млрд
# | KV-кэш на 131072 токена: 4.50 ГБ, без окон 9.0 ГБ
# | в экспертах 98.2% весов, файл 60.7 ГБ

# %% Листинг «Семейства моделей в числах» — раздел «Семейства открытых моделей»
def summary(name, config, tokens=8192):
    p = architecture_passport(config, default_tied=False)  # умолчание LLaMA
    kv = interleaved_kv_bytes(p, layer_types(config), tokens) / 2**30
    total = moe_parameter_count(p) / 1e9
    active = moe_parameter_count(p, active=True) / 1e9
    # :5 - столбец шириной 5 знаков, :<2 - прижать к левому краю
    print(f"{name:13}{p['d']:5}{p['L']:3}{p['H_q']:4}/{p['H_kv']:<2}"
          f"{p['d_ff']:6}{p['window'] or '-':>5}{p['E']:4}/{p['k']:<2}"
          f"{total:8.2f}{active:7.2f}{kv:8.2f}")

dense = {"vocab_size": 128256, "hidden_size": 4096, "num_hidden_layers": 32,
         "num_attention_heads": 32, "num_key_value_heads": 8,
         "intermediate_size": 14336}
print(f"{'модель':13}{'d':>5}{'L':>3}{'Hq':>4}/{'kv':<2}{'d_ff':>6}{'W':>5}"
      f"{'E':>4}/{'k':<2}{'всего':>8}{'актив':>7}{'KV,ГБ':>8}")
summary("Llama 3 8B", dense)
summary("Mistral 7B", dict(dense, vocab_size=32000, sliding_window=4096))
summary("Mixtral 8x7B", dict(dense, vocab_size=32000,
                             num_local_experts=8, num_experts_per_tok=2))
summary("Gemma 2 9B", {
    "vocab_size": 256000, "hidden_size": 3584, "num_hidden_layers": 42,
    "num_attention_heads": 16, "num_key_value_heads": 8, "head_dim": 256,
    "intermediate_size": 14336, "tie_word_embeddings": True,
    "sliding_window": 4096,
    "layer_types": ["sliding_attention", "full_attention"] * 21})
summary("Qwen3-235B", {
    "vocab_size": 151936, "hidden_size": 4096, "num_hidden_layers": 94,
    "num_attention_heads": 64, "num_key_value_heads": 4, "head_dim": 128,
    "intermediate_size": 12288, "moe_intermediate_size": 1536,
    "num_experts": 128, "num_experts_per_tok": 8})
summary("gpt-oss-120b", gpt_oss)
# Вывод в книге:
# | модель           d  L  Hq/kv  d_ff    W   E/k    всего  актив   KV,ГБ
# | Llama 3 8B    4096 32  32/8  14336    -   1/1     8.03   8.03    1.00
# | Mistral 7B    4096 32  32/8  14336 4096   1/1     7.24   7.24    0.50
# | Mixtral 8x7B  4096 32  32/8  14336    -   8/2    46.70  12.88    1.00
# | Gemma 2 9B    3584 42  16/8  14336 4096   1/1     9.24   9.24    1.97
# | Qwen3-235B    4096 94  64/4   1536    - 128/8   235.09  22.19    1.47
# | gpt-oss-120b  2880 36  64/8   2880  128 128/4   116.79   5.71    0.29

# %% Листинг «Три варианта BookLM» — раздел «Семейства открытых моделей»
variants = {"BookLM": booklm, "одна KV-голова": dict(booklm, H_kv=1),
            "окно 128, 5:1": dict(booklm, window=128),
            "4 эксперта, k=2": dict(booklm, E=4, k=2)}
for name, p in variants.items():
    kinds = five_to_one if p["window"] else ["full_attention"] * 6
    print(f"{name:16} всего {moe_parameter_count(p):>10,}"
          f"  активных {moe_parameter_count(p, active=True):>10,}"
          f"  кэш {interleaved_kv_bytes(p, kinds, 512) // 1024} КБ")
# Вывод в книге:
# | BookLM           всего  6,622,464  активных  6,622,464  кэш 768 КБ
# | одна KV-голова   всего  6,524,160  активных  6,524,160  кэш 384 КБ
# | окно 128, 5:1    всего  6,622,464  активных  6,622,464  кэш 288 КБ
# | 4 эксперта, k=2  всего 17,245,440  активных 10,167,552  кэш 768 КБ

# %% Листинг «Прирост KV-кэша на токен» — раздел «Архитектуры открытых семейств»
def growing_kv_bytes(layers, bytes_per_number=2):   # прирост кэша на токен
    return bytes_per_number * sum(n * numbers for n, numbers in layers)

gqa = lambda h_kv, d_h: 2 * h_kv * d_h      # K и V всех KV-голов слоя
mla = 512 + 64      # выжимка c_t и общий ключ (у K3 без поворота)
families = {   # слои: (сколько, чисел на токен); окно и линейные --- 0
    "gpt-oss-120b": [(18, gqa(8, 64)), (18, 0)],
    "Gemma 4 31B": [(10, 4 * 512), (50, 0)],                 # K = V
    "Qwen3.8-Flash-Next": [(12, gqa(2, 256) + 128 / 4), (36, 0)],
    "Kimi K3": [(24, mla), (69, 0)],
    "DeepSeek-V4-Flash": [(21, (512 + 128) / 4), (20, 512 / 128), (2, 0)]}
for name, layers in families.items():
    print(f"{name:19}{growing_kv_bytes(layers) / 1024:7.2f} КБ на токен")

p = architecture_passport(gpt_oss)          # сверка с функцией главы
kinds = layer_types(gpt_oss)
step = (interleaved_kv_bytes(p, kinds, 10**6 + 1)
        - interleaved_kv_bytes(p, kinds, 10**6))
print("gpt-oss по interleaved_kv_bytes:", step / 1024, "КБ на токен")
ds67b = {"L": 95, "H_kv": 8, "d_h": 128}    # первая модель DeepSeek, 67B
v1 = kv_cache_bytes(ds67b, tokens=1)
print(f"DeepSeek 67B: {v1 / 1024:.0f} КБ,",
      f"в {v1 / 890:.0f} раз больше 890 байт")
# Вывод в книге:
# | gpt-oss-120b         36.00 КБ на токен
# | Gemma 4 31B          40.00 КБ на токен
# | Qwen3.8-Flash-Next   24.75 КБ на токен
# | Kimi K3              27.00 КБ на токен
# | DeepSeek-V4-Flash     6.72 КБ на токен
# | gpt-oss по interleaved_kv_bytes: 36.0 КБ на токен
# | DeepSeek 67B: 380 КБ, в 437 раз больше 890 байт

# %% Листинг «Доля экспертов в параметрах» — раздел «Архитектуры открытых семейств»
def expert_params(layers, experts, width, d_ff, matrices=3):
    return layers * experts * matrices * width * d_ff   # SwiGLU: 3 матрицы

cards = {  # слоёв смеси, E, k, вход эксперта, d_ff, матриц; всего, активных
    "Qwen3.8-2.4T-A95B": (92, 512, 10, 8192, 2048, 3, 2446e9, 95e9),
    "DeepSeek-V4-Pro": (61, 384, 6, 7168, 3072, 3, 1600e9, 49e9),
    "Kimi K3": (92, 896, 16, 3584, 3072, 3, 2800e9, 104e9),
    "Nemotron 3 Ultra": (48, 512, 22, 2048, 5120, 2, 550e9, 55e9)}  # ReLU^2
print(f"{'':18}{'в экспертах':>12}{'активных':>10}{'k экспертов':>13}")
for name, (L, E, k, w, f, m, total, active) in cards.items():
    pool = expert_params(L, E, w, f, m)
    routed = expert_params(L, k, w, f, m)
    print(f"{name:18}{pool / total:12.1%}{active / total:10.1%}"
          f"{routed / 1e9:10.1f} млрд")
# Вывод в книге:
# |                    в экспертах  активных  k экспертов
# | Qwen3.8-2.4T-A95B        96.9%      3.9%      46.3 млрд
# | DeepSeek-V4-Pro          96.7%      3.1%      24.2 млрд
# | Kimi K3                  97.2%      3.7%      48.6 млрд
# | Nemotron 3 Ultra         93.7%     10.0%      22.1 млрд

# %% Листинг «Ошибка маскированной диффузии» — раздел «Маскированная диффузия»
import math, random

world = {("мама", "мыла", "раму"): 0.4, ("папа", "мыл", "машину"): 0.3,
         ("мама", "читала", "книгу"): 0.2, ("папа", "читал", "газету"): 0.1}
M = "[M]"                                # закрытая позиция

def denoise(x):                          # идеальная модель: слово под каждой маской
    fits = {s: p for s, p in world.items()
            if all(a in (M, b) for a, b in zip(x, s))} or world
    total = sum(fits.values())           # нет подходящих фраз: как без подсказки
    pred = {i: {} for i, a in enumerate(x) if a == M}
    for s, p in fits.items():
        for i in pred:
            pred[i][s[i]] = pred[i].get(s[i], 0) + p / total
    return pred                          # {позиция: {слово: вероятность}}

def masked_loss(x, t, weight):           # одна доля масок t, одна случайная маска
    xt = tuple(M if random.random() < t else a for a in x)
    pred = denoise(xt)
    loss = -sum(math.log(pred[i][x[i]]) for i in pred)
    return loss / t if weight else loss

random.seed(0)
for x in (("мама", "мыла", "раму"), ("мама", "читала", "книгу")):
    est = {w: sum(masked_loss(x, random.random(), w) for _ in range(100_000))
           / 100_000 for w in (True, False)}
    print(" ".join(x), f"-log p = {-math.log(world[x]):.3f},",
          f"с весом 1/t {est[True]:.3f}, без веса {est[False]:.3f}")
# Вывод в книге:
# | мама мыла раму -log p = 0.916, с весом 1/t 0.911, без веса 0.648
# | мама читала книгу -log p = 1.609, с весом 1/t 1.615, без веса 1.109

# %% Листинг «Генерация открытием масок» — раздел «Генерация открытием масок»
def unmask(x, steps):                    # x: список слов и масок
    x = list(x)
    for s in range(steps):
        pred = denoise(tuple(x))
        if not pred:
            break
        guess = {i: random.choices(list(d), list(d.values()))[0]
                 for i, d in pred.items()}            # слово для каждой маски
        sure = sorted(guess, key=lambda i: -pred[i][guess[i]])
        for i in sure[:math.ceil(len(pred) / (steps - s))]:
            x[i] = guess[i]              # открываем самые уверенные
    return tuple(x)

random.seed(0)
for steps in (3, 2, 1):
    runs = [unmask([M, M, M], steps) for _ in range(10_000)]
    ok = sum(r in world for r in runs) / len(runs)
    print(f"шагов {steps}: настоящих фраз {ok:.1%},",
          f"например «{' '.join(runs[0])}»")
runs = [unmask([M, M, "газету"], 2) for _ in range(1000)]
print("конец «газету»:", {" ".join(r) for r in runs})
# Вывод в книге:
# | шагов 3: настоящих фраз 100.0%, например «папа мыл машину»
# | шагов 2: настоящих фраз 54.4%, например «папа мыла книгу»
# | шагов 1: настоящих фраз 16.0%, например «мама мыл книгу»
# | конец «газету»: {'папа читал газету'}

# %% Листинг «Ключи второго слоя после открытия токена» — раздел «KV-кэш и блочная диффузия»
import torch
import torch.nn.functional as F

torch.manual_seed(0)
d, T = 8, 6
W1, W2 = torch.randn(d, 3 * d) / d**0.5, torch.randn(d, 2 * d) / d**0.5

def keys_of_layer2(x, causal):           # x: [T, d] -> ключи второго слоя [T, d]
    q, k, v = (x @ W1).chunk(3, dim=-1)  # первый слой внимания
    h = x + F.scaled_dot_product_attention(q[None], k[None], v[None],
                                           is_causal=causal)[0]
    return (h @ W2).chunk(2, dim=-1)[0]  # их и кладут в KV-кэш

x = torch.randn(T, d)
x_new = x.clone()
x_new[4] = torch.randn(d)                # открыли токен на позиции 4
for causal in (True, False):
    change = (keys_of_layer2(x_new, causal) - keys_of_layer2(x, causal)).abs()
    print("причинное " if causal else "двустороннее",
          [round(c, 2) for c in change.max(dim=-1).values.tolist()])
# Вывод в книге:
# | причинное  [0.0, 0.0, 0.0, 0.0, 1.46, 0.47]
# | двустороннее [0.14, 0.13, 0.68, 0.11, 1.47, 0.47]
