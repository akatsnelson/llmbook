"""Листинги главы 4 «Архитектура трансформера» книги «Большие языковые модели», файл 04-innovations.

Код и вывод — ровно как в книге; текст книги сюда не входит. Листинги идут по порядку и
опираются друг на друга: запускайте файл целиком или ячейками сверху вниз (разметка # %%).
Проверка, что вывод совпал с книгой: python scripts/check_listings.py listings/ch04/04-innovations.py
"""

# %% Листинг «Бегущее среднее вместо внимания» — раздел «Эмбеддинги и контекст»
import math
import torch
import torch.nn as nn
torch.set_printoptions(precision=4, sci_mode=False)

words = ["Мама", "мыла", "раму", "а", "потом", "она"]
vec = dict(zip(words, torch.eye(6)))       # единица на месте своего слова

def running_mean(phrase):
    x = torch.stack([vec[w] for w in phrase.split()])           # [T, 6]
    return x.cumsum(dim=0) / torch.arange(1, x.size(0) + 1)[:, None]

said = running_mean("Мама мыла раму а потом она")
print(said[-1])                          # что «она» взяла у каждого слова
# Вывод в книге:
# | tensor([0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.1667])

# %% Листинг «Доли по сходству слов и по поиску» — раздел «Self-attention и тройка query, key, value»
x = torch.stack([vec[w] for w in words])     # 6 слов фразы: [6, 6]
seek = torch.zeros(6, 6)
seek[5, 0] = 3.0                             # «она» ищет «Маму»
for name, s, row in (("сходство, «она»:", x @ x.T, -1),
                     ("поиск, «она»:   ", (x @ seek) @ x.T, -1),
                     ("поиск, «Мама»:  ", (x @ seek) @ x.T, 0)):
    print(name, [round(a, 2) for a in s.softmax(dim=-1)[row].tolist()])
# Вывод в книге:
# | сходство, «она»: [0.13, 0.13, 0.13, 0.13, 0.13, 0.35]
# | поиск, «она»:    [0.8, 0.04, 0.04, 0.04, 0.04, 0.04]
# | поиск, «Мама»:   [0.17, 0.17, 0.17, 0.17, 0.17, 0.17]

# %% Листинг «Внимание для всех позиций сразу» — раздел «Softmax и смесь значений»
def attention_scores(q, k):
    # q, k: [..., T, d_h] -> оценки [..., T, T]
    return q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))

def attention(q, k, v):
    weights = attention_scores(q, k).softmax(dim=-1)   # softmax по строке
    return weights @ v, weights                        # [..., T, d_h]

x = torch.tensor([[1.0, 0.0],     # токен 0
                  [0.0, 1.0],     # токен 1
                  [1.0, 1.0]])    # токен 2
output, weights = attention(x, x, x)
print(weights)
print(weights.sum(dim=-1))
# Вывод в книге:
# | tensor([[0.4011, 0.1978, 0.4011],
# |         [0.1978, 0.4011, 0.4011],
# |         [0.2483, 0.2483, 0.5035]])
# | tensor([1., 1., 1.])

# %% Листинг «Причинное внимание для всей последовательности» — раздел «Причинная маска»
def causal_attention(q, k, v):
    scores = attention_scores(q, k)                          # [..., T, T]
    t = q.size(-2)
    future = torch.triu(
        torch.ones(t, t, dtype=torch.bool, device=q.device), diagonal=1)
    scores = scores.masked_fill(future, float("-inf"))       # до softmax!
    weights = scores.softmax(dim=-1)
    return weights @ v, weights

output, weights = causal_attention(x, x, x)
print(weights)
print(output)
# Вывод в книге:
# | tensor([[1.0000, 0.0000, 0.0000],
# |         [0.3302, 0.6698, 0.0000],
# |         [0.2483, 0.2483, 0.5035]])
# | tensor([[1.0000, 0.0000],
# |         [0.3302, 0.6698],
# |         [0.7517, 0.7517]])

# %% Листинг «Разрезание на головы и склейка обратно» — раздел «Multi-head attention»
def split_heads(x, heads):
    # [B, T, d] -> [B, H, T, d_h]
    b, t, d = x.shape
    return x.reshape(b, t, heads, d // heads).transpose(1, 2)
def join_heads(x):
    # [B, H, T, d_h] -> [B, T, d]
    b, heads, t, d_h = x.shape
    return x.transpose(1, 2).contiguous().reshape(b, t, heads * d_h)
demo = torch.arange(8).reshape(1, 2, 4)     # B=1, T=2 токена, d=4
by_heads = split_heads(demo, heads=2)
print(demo)
print(by_heads)
print(by_heads.shape)
print(torch.equal(join_heads(by_heads), demo))
# Вывод в книге:
# | tensor([[[0, 1, 2, 3],
# |          [4, 5, 6, 7]]])
# | tensor([[[[0, 1],
# |           [4, 5]],
# |
# |          [[2, 3],
# |           [6, 7]]]])
# | torch.Size([1, 2, 2, 2])
# | True

# %% Листинг «SwiGLU и сверка с ручным счётом» — раздел «SwiGLU»
import torch.nn.functional as F

class SwiGLU(nn.Module):
    def __init__(self, d, d_ff):
        super().__init__()
        self.gate = nn.Linear(d, d_ff, bias=False)    # W_gate
        self.up = nn.Linear(d, d_ff, bias=False)      # W_up
        self.down = nn.Linear(d_ff, d, bias=False)    # W_down
    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))

print(F.silu(torch.tensor([2.0, -4.0])) * torch.tensor([3.0, 10.0]))
# Вывод в книге:
# | tensor([ 5.2848, -0.7194])

# %% Листинг «RMSNorm и сверка с ручным счётом» — раздел «RMSNorm»
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))       # g
        self.eps = eps
    def forward(self, x):
        rms = (x.pow(2).mean(dim=-1, keepdim=True) + self.eps).sqrt()
        return self.weight * x / rms

with torch.no_grad():
    x4 = torch.tensor([1.0, 3.0, 5.0, 7.0])
    print(RMSNorm(4)(x4))
    print(nn.LayerNorm(4)(x4))
# Вывод в книге:
# | tensor([0.2182, 0.6547, 1.0911, 1.5275])
# | tensor([-1.3416, -0.4472,  0.4472,  1.3416])

# %% Листинг «Многоголовое причинное внимание как слой» — раздел «Декодерный блок и модель»
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, heads):
        super().__init__()                       # учёт параметров
        self.heads = heads
        self.q, self.k, self.v, self.out = (     # W_Q, W_K, W_V, W_O
            nn.Linear(d_model, d_model, bias=False) for _ in range(4))
    def forward(self, x):                        # x: [B, T, d]
        q, k, v = (split_heads(w(x), self.heads)
                   for w in (self.q, self.k, self.v))
        mixed, _ = causal_attention(q, k, v)     # [B, H, T, d_h]
        return self.out(join_heads(mixed))       # [B, T, d]

# %% Листинг «Декодерный блок из уже написанных частей» — раздел «Декодерный блок и модель»
class DecoderBlock(nn.Module):
    def __init__(self, d_model, heads, d_ff):
        super().__init__()
        self.norm_1 = nn.LayerNorm(d_model)
        self.attention = CausalSelfAttention(d_model, heads)
        self.norm_2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
    def forward(self, x):
        x = x + self.attention(self.norm_1(x))   # собрать
        return x + self.ffn(self.norm_2(x))      # подумать

# %% Листинг «Тест влияния для блока» — раздел «Декодерный блок и модель»
torch.manual_seed(7)
block = DecoderBlock(d_model=16, heads=4, d_ff=48).eval()
x = torch.randn(1, 4, 16)
y = x.clone()
y[:, 2:] = torch.randn(1, 2, 16)                 # новые позиции 2 и 3
with torch.no_grad():
    out_x, out_y = block(x), block(y)
print(out_x.shape)
print((out_x - out_y).abs().amax(dim=-1))        # сдвиг каждой позиции
# Вывод в книге:
# | torch.Size([1, 4, 16])
# | tensor([[0.0000, 0.0000, 1.7417, 2.7631]])

# %% Листинг «Формы тензоров на пути через блок» — раздел «Декодерный блок и модель»
torch.manual_seed(0)
B, T, d, H, V = 2, 16, 256, 8, 8192              # размеры BookLM
embed = nn.Embedding(V, d)
block = DecoderBlock(d, H, d_ff=1024)
att = block.attention
with torch.no_grad():
    ids = torch.randint(0, V, (B, T))
    h = embed(ids)
    n = block.norm_1(h)
    q, k, v = (split_heads(w(n), H) for w in (att.q, att.k, att.v))
    mixed, weights = causal_attention(q, k, v)
    u = h + att.out(join_heads(mixed))            # лента после внимания
    out = block(h)
    print("номера токенов    ", ids.shape)
    print("эмбеддинги        ", h.shape)
    print("q по головам      ", q.shape)
    print("доли              ", weights.shape)
    print("смесь по головам  ", mixed.shape)
    print("после склейки, W_O", att.out(join_heads(mixed)).shape)
    print("внутри FFN        ", block.ffn[1](block.ffn[0](block.norm_2(u))).shape)
    print("выход блока       ", out.shape)
    print("логиты            ", (out @ embed.weight.T).shape)
# Вывод в книге:
# | номера токенов     torch.Size([2, 16])
# | эмбеддинги         torch.Size([2, 16, 256])
# | q по головам       torch.Size([2, 8, 16, 32])
# | доли               torch.Size([2, 8, 16, 16])
# | смесь по головам   torch.Size([2, 8, 16, 32])
# | после склейки, W_O torch.Size([2, 16, 256])
# | внутри FFN         torch.Size([2, 16, 1024])
# | выход блока        torch.Size([2, 16, 256])
# | логиты             torch.Size([2, 16, 8192])

# %% Листинг «Один блок не различает порядок прошлых токенов» — раздел «Порядок слов»
said = running_mean("Мама мыла раму а потом она")
swapped = running_mean("раму мыла Мама а потом она")
print("среднее, слово «она»:", torch.equal(said[-1], swapped[-1]))
torch.manual_seed(7)
embed = nn.Embedding(32, 16)
first = DecoderBlock(16, heads=4, d_ff=48).eval()
second = DecoderBlock(16, heads=4, d_ff=48).eval()
def order_effect(model):
    # Сдвиг выхода каждой позиции, когда токены 0 и 1 меняются местами.
    with torch.no_grad():
        straight = model(torch.tensor([[2, 5, 7, 11]]))
        swapped = model(torch.tensor([[5, 2, 7, 11]]))
    return (straight - swapped).abs().amax(dim=-1)
print(order_effect(nn.Sequential(embed, first)))
print(order_effect(nn.Sequential(embed, first, second)))
# Вывод в книге:
# | среднее, слово «она»: True
# | tensor([[2.8544, 2.7300, 0.0000, 0.0000]])
# | tensor([[2.9227, 2.5180, 0.0915, 0.0590]])

# %% Листинг «Прибавленная позиция в оценке внимания» — раздел «Абсолютные позицииПозиционное кодирование»
def clock(t):                          # стрелка позиции t: угол t радиан
    return torch.tensor([math.sin(t), math.cos(t)])

q, k = torch.tensor([3.0, 4.0]), torch.tensor([2.0, -1.0])
for p, s in ((1, 4), (11, 14), (300, 303)):
    score = (q + clock(p)) @ (k + clock(s))
    print(f"позиции {p:3d} и {s:3d}: оценка {score.item():.4f}")
# Вывод в книге:
# | позиции   1 и   4: оценка -2.7323
# | позиции  11 и  14: оценка 2.5244
# | позиции 300 и 303: оценка 2.6443

# %% Листинг «Одна пара RoPE руками» — раздел «RoPE»
def rotate_pair(x0, x1, angle):
    c, s = torch.cos(angle), torch.sin(angle)
    return c * x0 - s * x1, s * x0 + c * x1

q, k = torch.tensor([3.0, 4.0]), torch.tensor([2.0, -1.0])
for q_angle, k_angle in ((0.3, 1.1), (2.3, 3.1)):     # радианы
    q_turned = torch.stack(rotate_pair(*q, torch.tensor(q_angle)))
    k_turned = torch.stack(rotate_pair(*k, torch.tensor(k_angle)))
    print(f"углы {q_angle} и {k_angle}: длина q {q_turned.norm():.1f},",
          f"оценка {q_turned @ k_turned:.4f}")
print("без поворота: длина q", q.norm().item(), "оценка", (q @ k).item())
# Вывод в книге:
# | углы 0.3 и 1.1: длина q 5.0, оценка 9.2843
# | углы 2.3 и 3.1: длина q 5.0, оценка 9.2843
# | без поворота: длина q 5.0 оценка 2.0

# %% Листинг «RoPE для целого тензора» — раздел «RoPE»
def apply_rope(x, base=10000.0):
    # x: [..., T, d_h]; пара k на позиции p крутится на p * theta_k
    t, d_h = x.shape[-2:]
    theta = base ** (-torch.arange(0, d_h, 2) / d_h)   # [d_h/2]
    angles = torch.arange(t)[:, None] * theta          # [T, d_h/2]
    x0, x1 = rotate_pair(x[..., 0::2], x[..., 1::2], angles)
    return torch.stack((x0, x1), dim=-1).flatten(-2)

q = torch.tensor([3.0, 4.0, 3.0, 4.0]).expand(16, 4)   # d_h=4: две пары
k = torch.tensor([2.0, -1.0, 2.0, -1.0]).expand(16, 4)
print(apply_rope(q)[:3])                       # позиции 0, 1, 2
scores = apply_rope(q) @ apply_rope(k).T       # [16, 16]: q на p, k на s
print(round(scores[1, 4].item(), 4), round(scores[11, 14].item(), 4))
print(round(scores[1, 5].item(), 4))
# Вывод в книге:
# | tensor([[ 3.0000,  4.0000,  3.0000,  4.0000],
# |         [-1.7450,  4.6856,  2.9599,  4.0298],
# |         [-4.8856,  1.0633,  2.9194,  4.0592]])
# | 1.9014 1.9014
# | -7.1938

# %% Листинг «Граница RoFormer и настроенный ключ» — раздел «Частоты пар»
def turn(x, r, base=10000.0):           # все пары вектора x на r позиций
    theta = base ** (-torch.arange(0, x.size(-1), 2) / x.size(-1))
    x0, x1 = rotate_pair(x[0::2], x[1::2], r * theta)
    return torch.stack((x0, x1), dim=-1).flatten()

def decay_bound(r, d_h=32, base=10000.0):         # формула B(r)
    theta = base ** (-torch.arange(0, d_h, 2) / d_h)
    arrows = torch.polar(torch.ones(d_h // 2), r * theta)   # e^{i r theta_k}
    return arrows.cumsum(0).abs().mean().item()

torch.manual_seed(0)
ones, q = torch.ones(32), torch.randn(32)
k_far = turn(q, 100)                    # ключ, настроенный на расстояние 100
print("   r  граница  единицы  ключ на 100")
for r in (0, 1, 3, 10, 100, 1000):
    print(f"{r:4d} {decay_bound(r):8.2f} {ones @ turn(ones, -r):8.2f}"
          f" {q @ turn(k_far, -r):12.2f}")
# Вывод в книге:
# |    r  граница  единицы  ключ на 100
# |    0     8.50    32.00        19.29
# |    1     8.15    30.63        16.32
# |    3     6.14    24.54        11.65
# |   10     4.15    20.12         7.47
# |  100     4.38    18.67        32.40
# | 1000     2.25     8.21        -8.49

# %% Листинг «Остановленные пары RoPE в BookLM» — раздел «Частоты пар»
from booklm.lab import load, perplexity
tok, model = load("base")
cos, sin = model.rope_cos.clone(), model.rope_sin.clone()   # [512, 16]
theta = 10000.0 ** (-torch.arange(0, 32, 2) / 32)
print(f"все пары крутятся: перплексия {perplexity(model):.2f}")
for first in (0, 4, 8, 12):
    model.rope_cos, model.rope_sin = cos.clone(), sin.clone()
    model.rope_cos[:, first:first + 4] = 1.0         # угол 0 у четырёх пар
    model.rope_sin[:, first:first + 4] = 0.0
    turned = 511 * theta[first:first + 4]          # углы за окно, рад
    print(f"стоят пары {first:2d}-{first + 3:2d} (за окно {turned[0]:5.2f}"
          f"-{turned[-1]:4.2f} рад): {perplexity(model):5.2f}")
model.rope_cos, model.rope_sin = cos, sin
# Вывод в книге:
# | все пары крутятся: перплексия 6.96
# | стоят пары  0- 3 (за окно 511.00-90.87 рад): 15.11
# | стоят пары  4- 7 (за окно 51.10-9.09 рад):  8.60
# | стоят пары  8-11 (за окно  5.11-0.91 рад): 24.53
# | стоят пары 12-15 (за окно  0.51-0.09 рад):  6.96

# %% Листинг «Две раскладки пар на BookLM» — раздел «Раскладка пар»
import booklm.model as booklm_model
from booklm.lab import generate

def rope_halves(x, cos, sin):           # пары (i, i + d_h/2), как rotate_half
    t, half = x.size(-2), x.size(-1) // 2
    cos, sin = cos[:t], sin[:t]
    x0, x1 = x[..., :half], x[..., half:]
    return torch.cat((x0 * cos - x1 * sin, x0 * sin + x1 * cos), dim=-1)

def show(name):
    tale = generate(tok, model, "Жил-был кот, и однажды он", max_new=10,
                    temperature=0)
    print(f"{name}: перплексия {perplexity(model):.5g}\n  {tale}")

neighbours = booklm_model.apply_rope
show("соседние пары")
booklm_model.apply_rope = rope_halves              # веса те же
show("половинки")
perm = torch.cat([torch.arange(0, 32, 2), torch.arange(1, 32, 2)])
with torch.no_grad():                              # строки внутри каждой головы
    for block in model.blocks:
        att = block.attention
        for w, heads in ((att.q.weight, 8), (att.k.weight, 2)):
            w.copy_(w.view(heads, 32, 256)[:, perm].reshape(heads * 32, 256))
show("половинки, W_Q и W_K переставлены")
booklm_model.apply_rope = neighbours
# Вывод в книге:
# | соседние пары: перплексия 6.9584
# |   Жил-был кот, и однажды он вышел гулять. Он был очень рад, потому что
# | половинки: перплексия 186.91
# |   Жил-был кот, и однажды он был очень особенный человек. Он был очень любил играть
# | половинки, W_Q и W_K переставлены: перплексия 6.9584
# |   Жил-был кот, и однажды он вышел гулять. Он был очень рад, потому что

# %% Листинг «Наклоны и штрафы ALiBi» — раздел «ALiBi»
def alibi_slopes(heads):                   # m_h = 2^(-8h/H), h = 1..H
    return 2.0 ** (-8 * torch.arange(1, heads + 1) / heads)

def alibi_bias(t, heads):                  # [H, T, T]: -m_h (t - j) слева
    back = torch.arange(t)[None, :] - torch.arange(t)[:, None]  # j - t
    return alibi_slopes(heads)[:, None, None] * back.clamp(max=0)

m = alibi_slopes(8)
print(m)
print("штраф 10 на расстоянии:", (10 / m).int().tolist())
print(alibi_bias(4, 8)[0])                 # первая голова, T = 4
# Вывод в книге:
# | tensor([0.5000, 0.2500, 0.1250, 0.0625, 0.0312, 0.0156, 0.0078, 0.0039])
# | штраф 10 на расстоянии: [20, 40, 80, 160, 320, 640, 1280, 2560]
# | tensor([[ 0.0000,  0.0000,  0.0000,  0.0000],
# |         [-0.5000,  0.0000,  0.0000,  0.0000],
# |         [-1.0000, -0.5000,  0.0000,  0.0000],
# |         [-1.5000, -1.0000, -0.5000,  0.0000]])

# %% Листинг «Назначение GQA-голов и размер KV-кэша» — раздел «MHA, MQA и GQA»
def kv_cache_bytes(layers, tokens, kv_heads, head_dim, bytes_per_number=2):
    return 2 * layers * tokens * kv_heads * head_dim * bytes_per_number

def kv_head_for_query(query_head, query_heads, kv_heads):
    return query_head * kv_heads // query_heads

print([kv_head_for_query(h, query_heads=8, kv_heads=2) for h in range(8)])
for name, kv_heads in (("MHA", 8), ("GQA", 2), ("MQA", 1)):
    size = kv_cache_bytes(6, 512, kv_heads, head_dim=32)      # L=6, T=512
    print(f"{name}: {size:8d} байт = {size // 1024:5d} КБ")
for kv_heads in (64, 8):
    size = kv_cache_bytes(80, 8192, kv_heads, head_dim=128)   # L=80
    print(f"80 слоёв, {kv_heads:2d} KV-голов: {size / 2**30:4.1f} ГБ")
# Вывод в книге:
# | [0, 0, 0, 0, 1, 1, 1, 1]
# | MHA:  3145728 байт =  3072 КБ
# | GQA:   786432 байт =   768 КБ
# | MQA:   393216 байт =   384 КБ
# | 80 слоёв, 64 KV-голов: 20.0 ГБ
# | 80 слоёв,  8 KV-голов:  2.5 ГБ

# %% Листинг «Число параметров BookLM» — раздел «Параметры, вычисления и память»
from booklm.model import BookLM
model = BookLM()                    # размеры по умолчанию --- как выше
count = lambda m: sum(p.numel() for p in m.parameters())
print(count(model), count(model.blocks[0]))
# Вывод в книге:
# | 6622464 754176

# %% Листинг «Линейное внимание: рекуррентно и параллельно» — раздел «Линейное внимание как рекуррентная сеть»
def linear_recurrent(queries, keys, values, phi=lambda x: x):
    S = torch.zeros(keys.size(-1), values.size(-1))   # матрица d_h x d_h
    z = torch.zeros(keys.size(-1))
    outputs = []
    for q, k, v in zip(queries, keys, values):
        S = S + torch.outer(phi(k), v)                # S += phi(k) v^T
        z = z + phi(k)
        outputs.append(phi(q) @ S / (phi(q) @ z))
    return torch.stack(outputs)
keys = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
values = torch.tensor([[2.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
queries = torch.tensor([[0.5, 0.5], [1.0, 0.0], [1.0, 2.0]])
print(linear_recurrent(queries, keys, values))
scores = torch.tril(queries @ keys.T)            # будущее = 0
print(scores / scores.sum(dim=-1, keepdim=True) @ values)
print((queries[2] @ keys.T).softmax(dim=-1) @ values)  # softmax, токен 3
# Вывод в книге:
# | tensor([[2.0000, 0.0000],
# |         [2.0000, 0.0000],
# |         [0.8333, 1.1667]])
# | tensor([[2.0000, 0.0000],
# |         [2.0000, 0.0000],
# |         [0.8333, 1.1667]])
# | tensor([0.8453, 1.1547])

# %% Листинг «Селективная SSM: рекуррентно и параллельно» — раздел «Селективность в Mamba»
torch.manual_seed(0)
T, N = 6, 4                                     # 6 токенов, состояние из 4 чисел
x = torch.randn(T)                              # один канал входа
A = -torch.arange(1.0, N + 1)                   # A < 0: без входа состояние гаснет
w_dt, w_B, w_C = torch.randn(1), torch.randn(N), torch.randn(N)
dt = F.softplus(w_dt * x + 1.0)                 # шаг Delta_t зависит от токена
B, C = torch.outer(x, w_B), torch.outer(x, w_C)  # B_t и C_t тоже: [T, N]
decay = torch.exp(dt[:, None] * A)              # A с чертой = exp(Delta_t A): [T, N]

def ssm_recurrent(decay, B, C, x, dt):
    h, ys = torch.zeros(N), []
    for t in range(T):                          # по одному токену, память --- только h
        h = decay[t] * h + dt[t] * B[t] * x[t]  # h_t = A_t h_{t-1} + B_t x_t
        ys.append(C[t] @ h)                     # y_t = C_t . h_t
    return torch.stack(ys)

def ssm_parallel(decay, B, C, x, dt):
    P = torch.log(decay).cumsum(dim=0)          # логарифм A_1 A_2 ... A_t
    past = torch.tril(torch.ones(T, T, dtype=torch.bool))[..., None]
    L = torch.exp((P[:, None] - P[None]).masked_fill(~past, -torch.inf))  # [t, s, N]
    return torch.einsum("tn,tsn,sn,s->t", C, L, B, dt * x)

print(ssm_recurrent(decay, B, C, x, dt))
print(ssm_parallel(decay, B, C, x, dt))
# Вывод в книге:
# | tensor([ 6.3557, -0.1442, -7.8730,  0.4051, -1.3852, -2.7582])
# | tensor([ 6.3557, -0.1442, -7.8730,  0.4051, -1.3852, -2.7582])

# %% Листинг «Параллельное сканирование удвоением» — раздел «Параллельное сканирование»
def scan_sequential(a, b):                      # h_t = a_t h_{t-1} + b_t, h_0 = 0
    h, out = torch.tensor(0.0), []
    for a_t, b_t in zip(a, b):
        h = a_t * h + b_t
        out.append(h)
    return torch.stack(out)

def scan_doubling(a, b):                        # log2(T) ходов, в каждом все позиции сразу
    a, b, shift = a.clone(), b.clone(), 1
    while shift < len(a):
        # позиция t приклеивает слева отрезок, который кончается на t - shift
        b[shift:] = a[shift:] * b[:-shift] + b[shift:]
        a[shift:] = a[shift:] * a[:-shift]
        shift *= 2
    return b

a = torch.tensor([0.5, 1.0, 0.5, 0.0, 1.0, 0.5, 0.5, 1.0])   # a_t = 0: сброс
b = torch.tensor([4.0, 0.0, 2.0, 1.0, 3.0, 0.0, 2.0, 1.0])
print(scan_sequential(a, b))
print(scan_doubling(a, b))
# Вывод в книге:
# | tensor([4., 4., 4., 1., 4., 2., 3., 4.])
# | tensor([4., 4., 4., 1., 4., 2., 3., 4.])

# %% Листинг «Mamba-2 как маскированное линейное внимание» — раздел «Дуальность пространства состояний и внимания»
def ssd_attention(queries, keys, values, a):    # a: [T], затухание a_t из (0, 1]
    P = torch.log(a).cumsum(dim=0)
    L = torch.exp(P[:, None] - P[None]).tril()  # L[t, s] = a_{s+1} ... a_t, будущее = 0
    return (L * (queries @ keys.T)) @ values    # маскированное линейное внимание

def ssd_recurrent(queries, keys, values, a):
    S, outputs = torch.zeros(keys.size(-1), values.size(-1)), []
    for q, k, v, a_t in zip(queries, keys, values, a):
        S = a_t * S + torch.outer(k, v)         # затухание, потом запись
        outputs.append(q @ S)
    return torch.stack(outputs)

ones = torch.ones(3)
num = ssd_attention(queries, keys, values, ones)
den = ssd_attention(queries, keys, torch.ones(3, 1), ones)
print(num / den)                                # a_t = 1: линейное внимание
a = torch.tensor([1.0, 0.5, 0.5])
print(ssd_attention(queries, keys, values, a))
print(ssd_recurrent(queries, keys, values, a))
# Вывод в книге:
# | tensor([[2.0000, 0.0000],
# |         [2.0000, 0.0000],
# |         [0.8333, 1.1667]])
# | tensor([[1.0000, 0.0000],
# |         [1.0000, 0.0000],
# |         [3.5000, 5.0000]])
# | tensor([[1.0000, 0.0000],
# |         [1.0000, 0.0000],
# |         [3.5000, 5.0000]])

# %% Листинг «Сумма, затухание и правило дельты» — раздел «Правило дельты»
k1, k2 = torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])
v1, v2, v1_new = torch.tensor([2.0, 0.0]), torch.tensor([0.0, 2.0]), torch.tensor([5.0, 5.0])
writes = [(k1, v1), (k2, v2), (k1, v1_new)]     # третья запись меняет значение ключа k1

def remember(rule, alpha=0.5, beta=1.0):
    S = torch.zeros(2, 2)
    for k, v in writes:
        if rule == "сумма":                      # линейное внимание
            S = S + torch.outer(k, v)
        elif rule == "затухание":                # Mamba-2: старое слабеет целиком
            S = alpha * S + torch.outer(k, v)
        elif rule == "дельта":                   # DeltaNet: меняем то, что лежит по ключу
            S = S + beta * torch.outer(k, v - k @ S)
    return k1 @ S, k2 @ S                        # читаем по обоим ключам

for rule in ("сумма", "затухание", "дельта"):
    by_k1, by_k2 = remember(rule)
    print(f"{rule:10s} k1 -> {by_k1.tolist()}, k2 -> {by_k2.tolist()}")
# Вывод в книге:
# | сумма      k1 -> [7.0, 5.0], k2 -> [0.0, 2.0]
# | затухание  k1 -> [5.5, 5.0], k2 -> [0.0, 1.0]
# | дельта     k1 -> [5.0, 5.0], k2 -> [0.0, 2.0]

# %% Листинг «Состояние и KV-кэш гибрида Qwen3-Next» — раздел «Ограниченное состояние и гибридные модели»
layers, T = 48, 262_144                         # Qwen3-Next-80B-A3B, config.json
kv_per_token = 2 * 2 * 256                      # K и V, 2 KV-головы по 256 чисел
state = 32 * 128 * 128                          # 32 головы Gated DeltaNet, матрица 128 x 128
gib = lambda numbers: numbers * 2 / 2**30       # BF16: 2 байта на число
full = layers * kv_per_token * T                # если бы все слои были вниманием
hybrid = 12 * kv_per_token * T + 36 * state     # каждый четвёртый слой --- внимание
print(f"все 48 слоёв --- внимание: {gib(full):.1f} ГБ")
print(f"гибрид 12 + 36:            {gib(hybrid):.2f} ГБ")
print(f"состояние одного слоя = кэш {state // kv_per_token} токенов слоя внимания")
# Вывод в книге:
# | все 48 слоёв --- внимание: 24.0 ГБ
# | гибрид 12 + 36:            6.04 ГБ
# | состояние одного слоя = кэш 512 токенов слоя внимания
