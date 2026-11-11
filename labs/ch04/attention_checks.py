"""Проверки к главе 4: все утверждения листингов главы в виде assert.

В книге листинги печатают числа и формы, а читатель
сверяет их глазами. Здесь те же функции собраны в один файл и проверены утверждениями,
чтобы любую правку кода главы можно было проверить одной командой.
Запуск из корня репозитория: python labs/ch04/attention_checks.py
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# --- Внимание: одна позиция, все позиции, маска -------------------------------------------------

def attention_scores(q, k):
    return q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))


def attention(q, k, v):
    weights = attention_scores(q, k).softmax(dim=-1)
    return weights @ v, weights


def causal_attention(q, k, v):
    scores = attention_scores(q, k)
    t = q.size(-2)
    future = torch.triu(torch.ones(t, t, dtype=torch.bool, device=q.device), diagonal=1)
    weights = scores.masked_fill(future, float("-inf")).softmax(dim=-1)
    return weights @ v, weights


def influence_leak(attention_fn, q, k, v):
    q2, k2, v2 = q.clone(), k.clone(), v.clone()
    q2[..., -1, :] += 5.0
    k2[..., -1, :] -= 5.0
    v2[..., -1, :] *= -3.0
    before, _ = attention_fn(q, k, v)
    after, _ = attention_fn(q2, k2, v2)
    return (before[..., :-1, :] - after[..., :-1, :]).abs().max().item()


def broken_attention(q, k, v):
    t = q.size(-2)
    past = torch.tril(torch.ones(t, t, dtype=torch.bool), diagonal=-1)
    weights = attention_scores(q, k).masked_fill(past, float("-inf")).softmax(dim=-1)
    return weights @ v, weights


# Одна позиция руками: q_2 = (0, 1), два ключа и два значения.
query = torch.tensor([0.0, 1.0])
keys = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
values = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
weights = (query @ keys.T / math.sqrt(2)).softmax(dim=-1)
torch.testing.assert_close(weights.sum(), torch.tensor(1.0))
torch.testing.assert_close(weights @ values, torch.tensor([0.6605, 1.3395]), atol=1e-4, rtol=0)

# Доли по сходству самих слов: «она» больше всего смотрит на себя; с W_Q, W_K — на «Маму».
x6 = torch.eye(6)
naive = (x6 @ x6.T).softmax(dim=-1)[-1]
assert naive[-1] > 0.35 and abs(naive[0] - naive[3]) < 1e-6
W_Q = torch.zeros(6, 6)
W_Q[5, 0] = 3.0
fixed = ((x6 @ W_Q) @ x6.T).softmax(dim=-1)
assert fixed[-1, 0] > 0.8 and torch.allclose(fixed[0], torch.full((6,), 1 / 6))

# Матричная форма без маски: доли каждой строки в сумме 1, но строка 0 смотрит в будущее.
x = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
_, weights = attention(x, x, x)
torch.testing.assert_close(weights.sum(dim=-1), torch.ones(3))
assert weights[0, 2] > 0.3

# Причинная маска: верхний треугольник нулевой, строки складываются в 1.
output, weights = causal_attention(x[None], x[None], x[None])
assert output.shape == (1, 3, 2) and weights.shape == (1, 3, 3)
assert torch.count_nonzero(torch.triu(weights[0], diagonal=1)) == 0
torch.testing.assert_close(weights.sum(dim=-1), torch.ones(1, 3))
torch.testing.assert_close(weights[0, 1], torch.tensor([0.3302, 0.6698, 0.0]), atol=1e-4, rtol=0)

# Тест влияния: правильная маска не протекает ни на бит, сломанная протекает.
torch.manual_seed(4)
q, k, v = (torch.randn(1, 6, 8) for _ in range(3))
assert influence_leak(causal_attention, q, k, v) == 0.0
assert influence_leak(broken_attention, q, k, v) > 1e-2
torch.testing.assert_close(broken_attention(q, k, v)[1].sum(dim=-1), torch.ones(1, 6))
# Нулей у обеих масок по 15 (T(T-1)/2), только у сломанной они под диагональю.
assert int((causal_attention(q, k, v)[1] == 0).sum()) == 15
assert int((broken_attention(q, k, v)[1] == 0).sum()) == 15

# Масштаб: размах оценок растёт как корень из ширины.
torch.manual_seed(0)
for d_h in (4, 64, 1024):
    s = (torch.randn(2000, 8, d_h) @ torch.randn(2000, d_h, 1)).squeeze(-1)
    spread = s.pow(2).mean().sqrt().item()
    assert abs(spread / math.sqrt(d_h) - 1) < 0.1, (d_h, spread)


# --- Головы, блок, порядок слов -----------------------------------------------------------------

def split_heads(x, heads):
    b, t, d = x.shape
    assert d % heads == 0
    return x.reshape(b, t, heads, d // heads).transpose(1, 2)


def join_heads(x):
    b, heads, t, d_h = x.shape
    return x.transpose(1, 2).contiguous().reshape(b, t, heads * d_h)


demo = torch.arange(2 * 3 * 8).reshape(2, 3, 8)
assert split_heads(demo, 4).shape == (2, 4, 3, 2)
torch.testing.assert_close(join_heads(split_heads(demo, heads=4)), demo)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, heads):
        super().__init__()
        self.heads = heads
        self.q, self.k, self.v, self.out = (
            nn.Linear(d_model, d_model, bias=False) for _ in range(4))

    def forward(self, x):
        q, k, v = (split_heads(w(x), self.heads) for w in (self.q, self.k, self.v))
        mixed, _ = causal_attention(q, k, v)
        return self.out(join_heads(mixed))


class DecoderBlock(nn.Module):
    def __init__(self, d_model, heads, d_ff):
        super().__init__()
        self.norm_1 = nn.LayerNorm(d_model)
        self.attention = CausalSelfAttention(d_model, heads)
        self.norm_2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))

    def forward(self, x):
        x = x + self.attention(self.norm_1(x))
        return x + self.ffn(self.norm_2(x))


assert sum(p.numel() for p in CausalSelfAttention(16, heads=4).parameters()) == 4 * 16 * 16

torch.manual_seed(7)
block = DecoderBlock(d_model=16, heads=4, d_ff=48).eval()
x = torch.randn(1, 4, 16)
y = x.clone()
y[:, 2:] = torch.randn(1, 2, 16)
with torch.no_grad():
    out_x, out_y = block(x), block(y)
assert out_x.shape == x.shape
torch.testing.assert_close(out_x[:, :2], out_y[:, :2])          # прошлое на месте
assert not torch.allclose(out_x[:, 2:], out_y[:, 2:])


def order_effect(model):
    with torch.no_grad():
        straight = model(torch.tensor([[2, 5, 7, 11]]))
        swapped = model(torch.tensor([[5, 2, 7, 11]]))
    return (straight - swapped).abs().amax(dim=-1)[0]


torch.manual_seed(7)
embed = nn.Embedding(32, 16)
first = DecoderBlock(16, heads=4, d_ff=48).eval()
second = DecoderBlock(16, heads=4, d_ff=48).eval()
one = order_effect(nn.Sequential(embed, first))
two = order_effect(nn.Sequential(embed, first, second))
assert one[2:].max() < 1e-6                     # один слой видит мешок
assert two[2:].min() > 1e-3                     # второй слой узнаёт порядок от маски


# --- RoPE -----------------------------------------------------------------------------------------

def rotate_pair(x0, x1, angle):
    c, s = torch.cos(angle), torch.sin(angle)
    return c * x0 - s * x1, s * x0 + c * x1


def apply_rope(x, base=10000.0):
    t, d_h = x.shape[-2:]
    theta = base ** (-torch.arange(0, d_h, 2) / d_h)
    angles = torch.arange(t)[:, None] * theta
    x0, x1 = rotate_pair(x[..., 0::2], x[..., 1::2], angles)
    return torch.stack((x0, x1), dim=-1).flatten(-2)


def rope_score(q, k, q_angle, k_angle):
    q_turned = torch.stack(rotate_pair(*q, torch.tensor(q_angle)))
    k_turned = torch.stack(rotate_pair(*k, torch.tensor(k_angle)))
    return (q_turned @ k_turned).item()


q, k = torch.tensor([3.0, 4.0]), torch.tensor([2.0, -1.0])
turned = torch.stack(rotate_pair(*q, torch.tensor(2.0)))
assert abs(turned.norm().item() - 5.0) < 1e-5                 # поворот не растягивает
assert abs(rope_score(q, k, 0.3, 1.1) - rope_score(q, k, 2.3, 3.1)) < 1e-4

q4 = torch.tensor([3.0, 4.0, 3.0, 4.0]).expand(16, 4)         # d_h=4: две пары, две скорости
k4 = torch.tensor([2.0, -1.0, 2.0, -1.0]).expand(16, 4)
turned = apply_rope(q4)
assert (turned[1, :2] - q4[1, :2]).abs().max() > 1.0           # theta_0 = 1: быстрая стрелка
assert (turned[2, 2:] - q4[2, 2:]).abs().max() < 0.1           # theta_1 = 0,01: медленная
scores = turned @ apply_rope(k4).T
assert abs(scores[1, 4] - scores[11, 14]) < 1e-4              # важна только разность позиций
assert abs(scores[1, 4] - scores[1, 5]) > 1e-2
wide = torch.randn(1, 2, 16, 32)
torch.testing.assert_close(apply_rope(wide).norm(dim=-1), wide.norm(dim=-1))


# --- GQA и KV-кэш ---------------------------------------------------------------------------------

def kv_head_for_query(query_head, query_heads, kv_heads):
    assert query_heads % kv_heads == 0
    return query_head * kv_heads // query_heads


def kv_cache_bytes(layers, tokens, kv_heads, head_dim, bytes_per_number=2):
    return 2 * layers * tokens * kv_heads * head_dim * bytes_per_number


assert [kv_head_for_query(h, 8, 2) for h in range(8)] == [0, 0, 0, 0, 1, 1, 1, 1]
mha = kv_cache_bytes(6, 512, 8, 32)
gqa = kv_cache_bytes(6, 512, 2, 32)
assert mha == 3 * 1024 ** 2                      # 3 МБ
assert gqa == 768 * 1024                         # 768 КБ, как в паспорте BookLM
assert kv_cache_bytes(6, 512, 1, 32) == 384 * 1024
assert kv_cache_bytes(80, 8192, 8, 128) == 2.5 * 2 ** 30
assert kv_cache_bytes(80, 8192, 64, 128) == 20 * 2 ** 30


# --- BookLM ---------------------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        rms = (x.pow(2).mean(dim=-1, keepdim=True) + self.eps).sqrt()
        return self.weight * x / rms


class SwiGLU(nn.Module):
    def __init__(self, d, d_ff):
        super().__init__()
        self.gate = nn.Linear(d, d_ff, bias=False)
        self.up = nn.Linear(d, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class GQASelfAttention(CausalSelfAttention):
    def __init__(self, d_model, heads, kv_heads):
        super().__init__(d_model, heads)
        self.kv_heads = kv_heads
        self.use_rope = True
        kv_width = kv_heads * (d_model // heads)
        self.k = nn.Linear(d_model, kv_width, bias=False)
        self.v = nn.Linear(d_model, kv_width, bias=False)

    def forward(self, x):
        q = split_heads(self.q(x), self.heads)
        k = split_heads(self.k(x), self.kv_heads)
        v = split_heads(self.v(x), self.kv_heads)
        if self.use_rope:
            q, k = apply_rope(q), apply_rope(k)
        group = self.heads // self.kv_heads
        k = k.repeat_interleave(group, dim=1)
        v = v.repeat_interleave(group, dim=1)
        mixed, _ = causal_attention(q, k, v)
        return self.out(join_heads(mixed))


class BookLMBlock(DecoderBlock):
    def __init__(self, d_model, heads, kv_heads, d_ff):
        super().__init__(d_model, heads, d_ff)
        self.norm_1 = RMSNorm(d_model)
        self.norm_2 = RMSNorm(d_model)
        self.attention = GQASelfAttention(d_model, heads, kv_heads)
        self.ffn = SwiGLU(d_model, d_ff)


class BookLM(nn.Module):
    def __init__(self, vocab, d_model, layers, heads, kv_heads, d_ff):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        blocks = [BookLMBlock(d_model, heads, kv_heads, d_ff) for _ in range(layers)]
        self.blocks = nn.Sequential(*blocks)
        self.norm = RMSNorm(d_model)

    def forward(self, ids):
        h = self.norm(self.blocks(self.embed(ids)))
        return h @ self.embed.weight.T


with torch.no_grad():
    x4 = torch.tensor([1.0, 3.0, 5.0, 7.0])
    torch.testing.assert_close(RMSNorm(4)(x4), torch.tensor([0.2182, 0.6547, 1.0911, 1.5275]),
                               atol=1e-4, rtol=0)
    torch.testing.assert_close(nn.LayerNorm(4)(x4), torch.tensor([-1.3416, -0.4472, 0.4472, 1.3416]),
                               atol=1e-4, rtol=0)
    torch.testing.assert_close(F.silu(torch.tensor([2.0, -4.0])) * torch.tensor([3.0, 10.0]),
                               torch.tensor([5.2848, -0.7194]), atol=1e-4, rtol=0)

assert sum(p.numel() for p in GQASelfAttention(256, 8, 2).parameters()) == 163_840

model = BookLM(vocab=8192, d_model=256, layers=6, heads=8, kv_heads=2, d_ff=768)
n_params = sum(p.numel() for p in model.parameters())
assert n_params == 6_622_464                                   # паспорт BookLM
assert n_params == (8192 * 256
                    + 6 * (256 * 256 + 2 * 256 * 64 + 256 * 256 + 3 * 256 * 768 + 2 * 256)
                    + 256)
with torch.no_grad():
    assert model(torch.randint(0, 8192, (2, 16))).shape == (2, 16, 8192)
for block in model.blocks:                                     # [B, T, 256] -> [B, T, 256]
    assert block(torch.randn(1, 5, 256)).shape == (1, 5, 256)

torch.manual_seed(7)
tiny = BookLM(vocab=32, d_model=16, layers=1, heads=4, kv_heads=2, d_ff=48).eval()
assert order_effect(tiny)[2:].min() > 1e-2                     # RoPE: один блок видит порядок
tiny.blocks[0].attention.use_rope = False
assert order_effect(tiny)[2:].max() < 1e-5                      # без поворота снова мешок
tiny.blocks[0].attention.use_rope = True
with torch.no_grad():
    a = tiny(torch.tensor([[2, 5, 7, 11]]))
    c = tiny(torch.tensor([[2, 5, 19, 23]]))
torch.testing.assert_close(a[:, :2], c[:, :2], atol=0, rtol=0)  # будущее закрыто ни на бит

# Умножения в слое BookLM: веса растут как T d^2, пары как T^2 d; равны при T = 1472.
def layer_mults(T, d=256, kv_width=64, d_ff=768):
    return T * (2 * d * d + 2 * d * kv_width + 3 * d * d_ff), 2 * T * T * d


assert layer_mults(1472)[0] == layer_mults(1472)[1]
assert layer_mults(512)[0] > layer_mults(512)[1] and layer_mults(4096)[0] < layer_mults(4096)[1]

# Блок 2017 года: 4d^2 + 2 d d_ff плюс смещения FFN (1024 + 256) и две LayerNorm (2 * 2 * 256).
block_2017_params = sum(p.numel() for p in DecoderBlock(256, 8, 1024).parameters())
assert block_2017_params == 4 * 256 * 256 + 2 * 256 * 1024 + 1024 + 256 + 4 * 256 == 788_736

# Память BookLM в BF16.
assert round(n_params * 2 / 2 ** 20, 1) == 12.6
assert 8 * 512 ** 2 * 2 == 4 * 2 ** 20
assert 8 * 131072 ** 2 * 2 == 256 * 2 ** 30


# --- Линейное внимание ----------------------------------------------------------------------------

def linear_recurrent(queries, keys, values, phi=lambda x: x):
    S = torch.zeros(keys.size(-1), values.size(-1))
    z = torch.zeros(keys.size(-1))
    outputs = []
    for q, k, v in zip(queries, keys, values):
        S = S + torch.outer(phi(k), v)
        z = z + phi(k)
        outputs.append(phi(q) @ S / (phi(q) @ z))
    return torch.stack(outputs)


keys = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
values = torch.tensor([[2.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
queries = torch.tensor([[0.5, 0.5], [1.0, 0.0], [1.0, 2.0]])
recurrent = linear_recurrent(queries, keys, values)
torch.testing.assert_close(recurrent[2], torch.tensor([5 / 6, 7 / 6]))
scores = torch.tril(queries @ keys.T)
torch.testing.assert_close(recurrent, scores / scores.sum(dim=-1, keepdim=True) @ values)
softmax_answer = (queries[2] @ keys.T).softmax(dim=-1) @ values
assert not torch.allclose(softmax_answer, recurrent[2], atol=1e-3)

# --- Опыты разделов о полносвязной ветви и остаточном пути -----------------------------------

# Стопка из одних вниманий сливает позиции: к четвёртому слою различие меньше 1e-3.
torch.manual_seed(0)
h = torch.randn(6, 16)
for _ in range(4):
    w_q, w_k, w_v = (torch.randn(16, 16) / 4 for _ in range(3))
    h, _ = attention(h @ w_q, h @ w_k, h @ w_v)
assert ((h - h.mean(dim=0)).norm() / h.norm()).item() < 1e-3


def deep_stack(depth, residual=False, norm=False, init=None, d=64):
    torch.manual_seed(0)
    layers = [nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d)) for _ in range(depth)]
    if init == "he":
        for lin in (f[i] for f in layers for i in (0, 2)):
            nn.init.kaiming_normal_(lin.weight)
            nn.init.zeros_(lin.bias)
    h = torch.randn(32, d)
    for f in layers:
        z = F.layer_norm(h, (d,)) if norm else h
        h = h + f(z) if residual else f(z)
    h.pow(2).mean().backward()
    return h.pow(2).mean().sqrt().item(), layers[0][0].weight.grad.norm().item()


assert deep_stack(32)[1] == 0.0                              # обычный старт: градиент погас
assert deep_stack(32, init="he")[1] > 1e6                    # старт Хе: взрыв
assert deep_stack(32, residual=True)[1] > 0.1                # плюс оживил первый слой
assert deep_stack(32, residual=True, init="he")[0] > 1e6     # но от взрыва не спас
for init in (None, "he"):                                    # плюс и норма держат оба старта
    spread, grad = deep_stack(64, residual=True, norm=True, init=init)
    assert spread < 20 and 0.1 < grad < 100

print("OK: attention_checks.py")
