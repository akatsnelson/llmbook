"""Глава 7, разделы 7.1 «Один шаг обучения», 7.2 «Подготовка обучающих данных» и 7.3 «Обратное
распространение ошибки»: проверки к листингам этих разделов.

Из книги «Большие языковые модели». В книге листинги печатают числа, а здесь те же вычисления закреплены утверждениями assert:
запустите файл после любой правки кода главы, и он скажет, какое из чисел перестало сходиться.
Запуск из корня репозитория: python labs/ch07/step_data_backprop_checks.py
Считается на CPU за несколько секунд.
"""

import hashlib
import math
import random
import re
import unicodedata

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode


# =============================================================================
# 7.1. Один шаг обучения
# =============================================================================
def row_loss(row, targets):
    return F.cross_entropy(row.expand(len(targets), -1), torch.tensor(targets))


def gd_step(row, targets, eta):
    row = row.detach().clone().requires_grad_(True)
    row_loss(row, targets).backward()
    return (row - eta * row.grad).detach()


f64 = dict(dtype=torch.float64)
row = torch.tensor([2.0, 1.0, 0.0], **f64, requires_grad=True)
loss = row_loss(row, [0])
loss.backward()
p = row.detach().softmax(dim=-1)
g = p - torch.tensor([1.0, 0.0, 0.0], **f64)
assert torch.allclose(p, torch.tensor([0.6652, 0.2447, 0.0900], **f64), atol=1e-4)
assert torch.allclose(g, row.grad, atol=1e-12, rtol=0)          # формула = autograd
assert abs(g.sum().item()) < 1e-12                              # сумма по логитам = 0
assert abs(row_loss(gd_step(row, [0], 1.0), [0]).item() - 0.2582) < 1e-4
assert abs(row_loss(gd_step(row, [0], 0.5), [0]).item() - 0.3255) < 1e-4
assert row_loss(gd_step(row, [0], 10.0), [0]).item() < 0.01      # один пример: отлично
both = [0, 1]
start = row_loss(row, both).item()
assert abs(start - 0.9076) < 1e-4
assert row_loss(gd_step(row, both, 1.0), both).item() < start    # 0,8239
assert row_loss(gd_step(row, both, 10.0), both).item() > start   # 1,6534: проскочили


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight, self.eps = nn.Parameter(torch.ones(d)), eps

    def forward(self, x):
        ms = x.float().pow(2).mean(dim=-1, keepdim=True)
        return (x.float() * torch.rsqrt(ms + self.eps)).to(x.dtype) * self.weight


class TinyLM(nn.Module):
    def __init__(self, vocab_size=8192, d=256):
        super().__init__()
        self.tokens = nn.Embedding(vocab_size, d)
        self.blocks = nn.ModuleList()
        self.norm = RMSNorm(d)
        self.output = nn.Linear(d, vocab_size, bias=False)
        self.output.weight = self.tokens.weight

    def forward(self, ids):
        return self.output(self.norm(self.tokens(ids)))


@torch.no_grad()
def init_booklm_(model, std=0.02):
    branch_std = std / math.sqrt(2 * max(len(model.blocks), 1))
    for name, prm in model.named_parameters():
        if prm.dim() == 1:
            nn.init.ones_(prm)
        elif name.endswith(("out.weight", "down.weight")):
            nn.init.normal_(prm, std=branch_std)
        else:
            nn.init.normal_(prm, std=std)


def start_loss(model, V=8192, batch=4, length=128, seed=0):
    ids = torch.randint(0, V, (batch, length + 1), generator=torch.Generator().manual_seed(seed))
    with torch.no_grad():
        logits = model(ids[:, :-1])
    return F.cross_entropy(logits.flatten(0, 1), ids[:, 1:].flatten()).item()


torch.manual_seed(0)
tiny = TinyLM()
assert tiny.tokens.weight is tiny.output.weight
assert sum(prm.numel() for prm in tiny.parameters()) == 8192 * 256 + 256
assert start_loss(tiny) > 100                                    # 255,75
init_booklm_(tiny)
assert abs(start_loss(tiny) - math.log(8192)) < 0.1              # 9,08
nn.init.normal_(tiny.tokens.weight, std=1.0)
assert start_loss(tiny) > 100                                    # поломка снова видна


def batch_loss(model, inputs, targets, loss_mask=None):
    logits = model(inputs)
    per_pos = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), reduction="none")
    if loss_mask is None:
        return per_pos.mean()
    m = loss_mask.flatten().to(per_pos.dtype)
    return (per_pos * m).sum() / m.sum()


def train_step(model, batch, opt):
    opt.zero_grad(set_to_none=True)
    loss = batch_loss(model, *batch)
    loss.backward()
    opt.step()
    return loss.item()


def grad_norm(model):
    return math.sqrt(sum(prm.grad.pow(2).sum().item() for prm in model.parameters() if prm.grad is not None))


torch.manual_seed(0)
model = TinyLM()
init_booklm_(model)
opt = torch.optim.SGD(model.parameters(), lr=0.02)
ids = torch.randint(0, 8192, (4, 129), generator=torch.Generator().manual_seed(1))
batch = (ids[:, :-1], ids[:, 1:])
losses = [train_step(model, batch, opt) for _ in range(5)]
assert all(prm.grad.shape == prm.shape for prm in model.parameters())
assert all(b < a for a, b in zip(losses, losses[1:]))
assert abs(losses[0] - 9.065) < 1e-3 and abs(losses[-1] - 8.985) < 1e-3
# Маска из одних единиц даёт то же среднее, что и без маски.
ones = torch.ones_like(batch[0], dtype=torch.bool)
assert abs(batch_loss(model, *batch).item() - batch_loss(model, *batch, ones).item()) < 1e-6
model.zero_grad(set_to_none=True)
norms = []
for _ in range(2):                                              # zero_grad забыли
    batch_loss(model, *batch).backward()
    norms.append(grad_norm(model))
assert abs(norms[1] / norms[0] - 2.0) < 1e-4


# =============================================================================
# 7.2. Данные
# =============================================================================
def shingles(words, k):
    return {" ".join(words[i:i + k]) for i in range(len(words) - k + 1)}


def jaccard(s, t):
    return len(s & t) / len(s | t)


def minhash_signature(ids, coeffs, prime):
    return [min((a * x + b) % prime for x in ids) for a, b in coeffs]


def lsh_probability(j, b, r):
    return 1 - (1 - j**r) ** b


A = "мама мыла раму а потом она пошла спать".split()
B = "мама мыла раму а потом папа пошёл спать".split()
assert jaccard(shingles(A, 2), shingles(B, 2)) == 0.4
six = [(1, 1), (3, 2), (5, 3), (7, 4), (2, 5), (4, 6)]
assert minhash_signature(set(range(7)), six, 11) == [1, 0, 0, 0, 0, 0]
assert minhash_signature({0, 1, 2, 3, 7, 8, 9}, six, 11) == [1, 0, 2, 0, 0, 1]
assert sorted((5 * x + 3) % 11 for x in range(11)) == list(range(11))   # перестановка без повторов
assert round(lsh_probability(0.8, 14, 8), 4) == 0.9235
assert round(lsh_probability(0.5, 14, 8), 3) == 0.053

P61 = 2**61 - 1


def make_coeffs(n, seed=0):
    rng = random.Random(seed)
    return [(rng.randrange(1, P61), rng.randrange(P61)) for _ in range(n)]


def shingle_id(s):
    return int.from_bytes(hashlib.sha256(s.encode()).digest()[:8], "big") % P61


def text_signature(words, coeffs, k):
    return minhash_signature({shingle_id(s) for s in shingles(words, k)}, coeffs, P61)


def estimate(sig1, sig2):
    return sum(x == y for x, y in zip(sig1, sig2)) / len(sig1)


c112 = make_coeffs(112)
assert round(estimate(text_signature(A, c112, 2), text_signature(B, c112, 2)), 2) == 0.43


def normalize(text):
    text = unicodedata.normalize("NFC", text).replace("\r\n", "\n")
    lines = [re.sub(r"[ \t ]+", " ", ln).strip() for ln in text.split("\n")]
    return "\n".join(lines).strip()


def words_of(text):
    return re.sub(r"[^\w\s]", " ", normalize(text).lower()).split()


def norm_key(text):
    return hashlib.sha256(" ".join(words_of(text)).encode()).hexdigest()


def letter_share(d):
    chars = [ch for ch in d if not ch.isspace()]
    return sum(ch.isalpha() for ch in chars) / len(chars)


assert normalize("й") == "й"
assert normalize("Мама мыла.") == "Мама мыла."                 # регистр на месте
assert norm_key("Мама мыла раму.") == norm_key("Мама  мыла раму!")
assert letter_share("12 400 13 900 15 100") < 0.6

mama = "Мама{}мыла раму, а потом она пошла спать, и кот уснул на {}{}"
corpus = [
    mama.format(" ", "окне", "."),
    mama.format("  ", "окне", "!"),
    mama.format(" ", "диване", "."),
    "Чтобы сварить гречку, залейте её водой и варите двадцать минут.",
    "Поезд опоздал на сорок минут, и пассажиры выпили весь чай.",
    "Главная | Новости | Контакты | Войти",
    "12 400 13 900 15 100 17 800 21 300",
    "Дождь шёл весь вечер, и дети пускали по лужам кораблики.",
    "Скалярное произведение велико, когда векторы смотрят в одну сторону.",
    "Библиотека открывается в девять утра, а читальный зал в десять.",
]
stages = []


def count(docs):
    stages.append((len(docs), sum(len(words_of(d)) for d in docs)))


count(corpus)
seen, docs = set(), []
for d in map(normalize, corpus):
    if norm_key(d) not in seen:
        seen.add(norm_key(d))
        docs.append(d)
count(docs)
buckets = {}
for i, d in enumerate(docs):
    sig = text_signature(words_of(d), c112, 2)
    for band in range(14):
        buckets.setdefault((band, tuple(sig[band * 8:(band + 1) * 8])), set()).add(i)
pairs = {(i, j) for grp in buckets.values() for i in grp for j in grp if i < j}
assert pairs == {(0, 1)}
sh = lambda d: shingles(words_of(d), 2)
assert abs(jaccard(sh(docs[0]), sh(docs[1])) - 11 / 13) < 1e-12
docs = [d for i, d in enumerate(docs) if i != 1]
count(docs)
docs = [d for d in docs if len(words_of(d)) >= 5 and letter_share(d) >= 0.6]
count(docs)
count(docs[:-1])
assert stages == [(10, 102), (9, 89), (8, 76), (6, 62), (5, 52)]
assert lsh_probability(11 / 13, 14, 8) > 0.98


def pack(docs, window, eos):
    stream, owner = [], []
    for n, doc in enumerate(docs, start=1):
        stream += doc + [eos]
        owner += [n] * (len(doc) + 1)
    cut = len(stream) // window * window
    return (torch.tensor(stream[:cut]).view(-1, window), torch.tensor(owner[:cut]).view(-1, window),
            stream[cut:])


def document_causal_mask(doc_ids):
    t = doc_ids.numel()
    causal = torch.tril(torch.ones(t, t, dtype=torch.bool))
    return causal & (doc_ids[:, None] == doc_ids[None, :])


EOS = 0
windows, owners, rest = pack([[11, 12, 13, 14, 15], [21, 22, 23], [31, 32, 33, 34, 35, 36]], 12, EOS)
assert windows.shape == (1, 12) and rest == [33, 34, 35, 36, 0]
x, y = windows[:, :-1], windows[:, 1:]
doc_x = owners[0, :-1]
assert doc_x.tolist() == [1] * 6 + [2] * 4 + [3]
mask = document_causal_mask(doc_x)
assert mask.sum() == 21 + 10 + 1 and mask[6, 6] and not mask[6, 5]
loss_mask = x != EOS
assert loss_mask.sum() == 9 and not loss_mask[0, 5] and y[0, 5] == 21
assert y[0, 4] == EOS and loss_mask[0, 4]


def masked_attention(q, k, v, allowed):
    scores = (q @ k.T / math.sqrt(q.size(-1))).masked_fill(~allowed, float("-inf"))
    return scores.softmax(dim=-1) @ v


torch.manual_seed(0)
emb = torch.randn(40, 8)
w_q, w_k, w_v = (torch.randn(8, 8) for _ in range(3))


def layer(tokens, allowed):
    h = emb[tokens]
    return masked_attention(h @ w_q, h @ w_k, h @ w_v, allowed)


changed = x[0].clone()
changed[1:5] = torch.tensor([1, 2, 3, 4])
in_d2 = doc_x == 2
causal_only = torch.tril(torch.ones(11, 11, dtype=torch.bool))
shift = lambda allowed: (layer(x[0], allowed) - layer(changed, allowed))[in_d2].abs().max().item()
assert shift(mask) == 0.0                                        # с маской d2 не заметил
assert abs(shift(causal_only) - 5.72) < 0.01                     # без неё d2 читает d1


# =============================================================================
# 7.3. Обратное распространение
# =============================================================================
xv = np.array([1.0, 2.0])
W = np.array([[1.0, -1.0], [0.5, 0.5]])
U = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 1.0]])


def forward(W, U, x):
    h = W @ x
    r = np.maximum(h, 0)
    a = U @ r
    return h, r, a, np.log(np.exp(a).sum()) - a[0]


h, r, a, L = forward(W, U, xv)
pp = np.exp(a) / np.exp(a).sum()
g_a = pp - np.eye(3)[0]
dU = np.outer(g_a, r)
g_h = (U.T @ g_a) * (h > 0)
dW = np.outer(g_h, xv)
assert abs(L - 2.2989) < 1e-4 and abs(g_a.sum()) < 1e-12
assert np.allclose(g_a, [-0.8996, 0.4498, 0.4498], atol=1e-4)
assert np.all(dU[:, 0] == 0)
assert np.allclose(dW, [[0, 0], [0.8996, 1.7993]], atol=1e-4)


def numeric_grad(P, loss, eps):
    G = np.zeros_like(P)
    for i in np.ndindex(P.shape):
        old = P[i]
        P[i] = old + eps
        up = loss()
        P[i] = old - eps
        down = loss()
        P[i] = old
        G[i] = (up - down) / (2 * eps)
    return G


def check(dtype):
    W_, U_, x_ = W.astype(dtype), U.astype(dtype), xv.astype(dtype)
    loss = lambda: forward(W_, U_, x_)[3]
    nW = numeric_grad(W_, loss, dtype(1e-6))
    nU = numeric_grad(U_, loss, dtype(1e-6))
    err = max(np.abs(nW - dW).max() / np.abs(dW).max(), np.abs(nU - dU).max() / np.abs(dU).max())
    return err, nW


err64, _ = check(np.float64)
assert err64 < 1e-7                                              # 5,7e-11
err32, nW32 = check(np.float32)
assert err32 > 1e-4                                              # 3,6e-2
assert abs(nW32[1, 0] - 0.8345) < 1e-3                           # вместо 0,8996


class Value:
    def __init__(self, data, parents=(), local=()):
        self.data, self.grad, self.parents, self.local = data, 0.0, parents, local

    def __add__(self, o):
        return Value(self.data + o.data, (self, o), (1.0, 1.0))

    def __mul__(self, o):
        o = o if isinstance(o, Value) else Value(o)
        return Value(self.data * o.data, (self, o), (o.data, self.data))

    def exp(self):
        e = math.exp(self.data)
        return Value(e, (self,), (e,))

    def log(self):
        return Value(math.log(self.data), (self,), (1.0 / self.data,))

    def backward(self):
        order, seen_ids = [], set()

        def visit(node):
            if id(node) not in seen_ids:
                seen_ids.add(id(node))
                for parent in node.parents:
                    visit(parent)
                order.append(node)

        visit(self)
        self.grad = 1.0
        for node in reversed(order):
            for parent, d in zip(node.parents, node.local):
                parent.grad += d * node.grad


logits = [Value(2.0), Value(1.0), Value(0.0)]
Lv = logits[0] * -1.0 + (logits[0].exp() + logits[1].exp() + logits[2].exp()).log()
Lv.backward()
assert abs(Lv.data - 0.4076) < 1e-4
assert all(abs(v.grad - gg) < 1e-4 for v, gg in zip(logits, [-0.3348, 0.2447, 0.0900]))


def input_signal(residual, n_add=12, d=16, seed=0):
    gen = torch.Generator().manual_seed(seed)
    h0 = torch.randn(d, generator=gen, dtype=torch.float64, requires_grad=True)
    hh = h0
    for _ in range(n_add):
        W_l = torch.randn(d, d, generator=gen, dtype=torch.float64) * 0.5 / math.sqrt(d)
        branch = torch.tanh(W_l @ hh)
        hh = hh + branch if residual else branch
    hh.sum().backward()
    return h0.grad.norm().item()


assert input_signal(True) > 1.0                                  # 4,06
assert 0.5**13 < input_signal(False) / 4 < 0.5**12               # 1,7e-4

lin = nn.Linear(256, 768, bias=False)
tokens = torch.randn(512, 256, requires_grad=True)
with FlopCounterMode(display=False) as fwd:
    out = lin(tokens)
with FlopCounterMode(display=False) as bwd:
    out.sum().backward()
assert fwd.get_total_flops() == 2 * 512 * 256 * 768              # 2 операции на вес
assert bwd.get_total_flops() == 2 * fwd.get_total_flops()        # ещё 4

print("step_data_backprop_checks: все проверки пройдены")
