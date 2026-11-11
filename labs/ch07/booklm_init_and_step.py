"""Глава 7, раздел 7.1 «Один шаг обучения»: полная версия листингов раздела.

Из книги «Большие языковые модели».
В книге оставлены TinyLM, правило инициализации и функция шага; здесь лежат полные
классы BookLM главы 4 (RoPE, GQA, SwiGLU) с теми же именами параметров, проверка
стартовой ошибки полной модели и опыт с забытым zero_grad на полной BookLM.
Запуск из корня репозитория: python labs/ch07/booklm_init_and_step.py
Считается на CPU за несколько секунд.
"""

# --- Шаг по логитам руками и проверка autograd ---
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def row_loss(row, targets):                  # row: [V], строка таблицы
    return F.cross_entropy(row.expand(len(targets), -1),
                           torch.tensor(targets))

def gd_step(row, targets, eta):              # theta - eta * g
    row = row.detach().clone().requires_grad_(True)
    row_loss(row, targets).backward()
    return (row - eta * row.grad).detach()

f64 = dict(dtype=torch.float64)
row = torch.tensor([2.0, 1.0, 0.0], **f64, requires_grad=True)
loss = row_loss(row, [0])
loss.backward()                              # g от autograd
p = row.detach().softmax(dim=-1)
g = p - torch.tensor([1.0, 0.0, 0.0], **f64) # руками: p_j - [j = y]
hand_p = torch.tensor([0.6652, 0.2447, 0.0900], **f64)
assert torch.allclose(p, hand_p, atol=1e-4)
assert torch.allclose(g, row.grad, atol=1e-12, rtol=0)
assert abs(g.sum().item()) < 1e-12           # сумма по логитам = 0
after = row_loss(gd_step(row, [0], eta=1.0), [0]).item()
assert abs(after - 0.2582) < 1e-4 and after < loss.item()
half = row_loss(gd_step(row, [0], eta=0.5), [0]).item()
assert abs(half - 0.3255) < 1e-4

# Длинный шаг: на одном примере хорош, на пакете из двух позиций вреден.
assert row_loss(gd_step(row, [0], eta=10.0), [0]).item() < 0.01
both = [0, 1]
start = row_loss(row, both).item()                         # 0,9076
assert row_loss(gd_step(row, both, 1.0), both).item() < start   # 0,8239
assert row_loss(gd_step(row, both, 10.0), both).item() > start  # 1,6534


# --- Стартовая ошибка и правило инициализации ---
def rope(x):                                 # x: [B, H, T, d_h]
    t, d_h = x.shape[-2:]
    theta = 1e4 ** (-torch.arange(0, d_h, 2) / d_h)  # [d_h/2]
    ang = torch.arange(t)[:, None] * theta            # [T, d_h/2]
    x0, x1 = x[..., 0::2], x[..., 1::2]
    c, s = ang.cos(), ang.sin()
    pairs = (x0 * c - x1 * s, x0 * s + x1 * c)
    return torch.stack(pairs, dim=-1).flatten(-2)

class RMSNorm(nn.Module):                     # как в главе 4: счёт во float
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight, self.eps = nn.Parameter(torch.ones(d)), eps

    def forward(self, x):
        ms = x.float().pow(2).mean(dim=-1, keepdim=True)
        x32 = x.float() * torch.rsqrt(ms + self.eps)
        return x32.to(x.dtype) * self.weight

class GroupedQueryAttention(nn.Module):
    def __init__(self, d=256, hq=8, hkv=2):
        super().__init__()
        self.hq, self.hkv, self.dh = hq, hkv, d // hq
        self.q, self.k, self.v = (nn.Linear(d, h * self.dh, bias=False)
                                  for h in (hq, hkv, hkv))
        self.out = nn.Linear(d, d, bias=False)                # W_O

    def heads(self, proj, x, n):             # -> [B, n, T, d_h]
        return proj(x).view(*x.shape[:2], n, self.dh).transpose(1, 2)

    def forward(self, x):                    # [B, T, d]
        r = self.hq // self.hkv
        q = rope(self.heads(self.q, x, self.hq))
        k = rope(self.heads(self.k, x, self.hkv)).repeat_interleave(r, 1)
        v = self.heads(self.v, x, self.hkv).repeat_interleave(r, 1)  # GQA
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(a.transpose(1, 2).flatten(2))

class SwiGLU(nn.Module):
    def __init__(self, d=256, d_ff=768):
        super().__init__()
        self.gate = nn.Linear(d, d_ff, bias=False)
        self.up = nn.Linear(d, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d, bias=False)            # W_down

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))

class BookLMBlock(nn.Module):                # имена — как в главе 4
    def __init__(self, d=256):
        super().__init__()
        self.norm_1, self.attention = RMSNorm(d), GroupedQueryAttention(d)
        self.norm_2, self.ffn = RMSNorm(d), SwiGLU(d)

    def forward(self, x):
        x = x + self.attention(self.norm_1(x))
        return x + self.ffn(self.norm_2(x))

class BookLM(nn.Module):
    def __init__(self, vocab_size=8192, d=256, layers=6):
        super().__init__()
        self.tokens = nn.Embedding(vocab_size, d)
        self.blocks = nn.ModuleList(BookLMBlock(d) for _ in range(layers))
        self.norm = RMSNorm(d)
        self.output = nn.Linear(d, vocab_size, bias=False)
        self.output.weight = self.tokens.weight      # связанные веса

    def forward(self, ids):                  # [B, T] -> [B, T, V]
        h = self.tokens(ids)
        for block in self.blocks:
            h = block(h)
        return self.output(self.norm(h))

@torch.no_grad()
def init_booklm_(model, std=0.02):
    branch_std = std / math.sqrt(2 * max(len(model.blocks), 1))
    for name, p in model.named_parameters():  # связанная матрица - однажды
        if p.dim() == 1:
            nn.init.ones_(p)                  # усиления RMSNorm
        elif name.endswith(("out.weight", "down.weight")):
            nn.init.normal_(p, std=branch_std)  # W_O, W_down: 0,00577
        else:
            nn.init.normal_(p, std=std)

def start_loss(model, V=8192, batch=4, length=128, seed=0):
    ids = torch.randint(0, V, (batch, length + 1),
                        generator=torch.Generator().manual_seed(seed))
    with torch.no_grad():
        logits = model(ids[:, :-1])
    targets = ids[:, 1:].flatten()
    return F.cross_entropy(logits.flatten(0, 1), targets).item()

torch.manual_seed(0)
booklm = BookLM()
assert sum(p.numel() for p in booklm.parameters()) == 6_622_464
assert booklm.tokens.weight is booklm.output.weight     # один тензор
names = [n for n, _ in booklm.named_parameters() if ".0." in n]
assert names == ["blocks.0.norm_1.weight",              # как в главе 4
                 *(f"blocks.0.attention.{m}.weight" for m in "qkv"),
                 "blocks.0.attention.out.weight", "blocks.0.norm_2.weight",
                 *(f"blocks.0.ffn.{m}.weight" for m in ("gate", "up", "down"))]
assert start_loss(booklm) > 100                         # измерено: 243,4
torch.manual_seed(0)
init_booklm_(booklm)                                    # booklm-base-init
assert abs(start_loss(booklm) - math.log(8192)) < 0.1   # измерено: 9,058

torch.manual_seed(0)
tiny = BookLM(layers=0)                                 # TinyLM
assert start_loss(tiny) > 100                           # измерено: 255,8
init_booklm_(tiny)
assert abs(start_loss(tiny) - math.log(8192)) < 0.1     # измерено: 9,08
nn.init.normal_(tiny.tokens.weight, std=1.0)            # поломка: N(0, 1)
assert start_loss(tiny) > 100                           # снова сотни


# --- Один шаг обучения как функция ---
def train_step(model, batch, opt, zero_grad=True):
    inputs, targets = batch                   # 1. пакет [B, T]
    if zero_grad:
        opt.zero_grad(set_to_none=True)       # прошлого g больше нет
    logits = model(inputs)                    # 2. прямой проход [B, T, V]
    loss = F.cross_entropy(logits.flatten(0, 1),   # 3. сравнение с Y
                           targets.flatten())      # 4. одно число
    loss.backward()                           # 5. обратный проход
    opt.step()                                # 6. theta - eta * g
    return loss.item()

def grad_norm(model):
    flat = [p.grad.flatten() for p in model.parameters()]
    return torch.cat(flat).norm().item()

def fresh(lr=0.02):                           # booklm-base-init и SGD
    torch.manual_seed(0)
    model = BookLM()
    init_booklm_(model)
    return model, torch.optim.SGD(model.parameters(), lr=lr)

gen = torch.Generator().manual_seed(1)
ids = torch.randint(0, 8192, (4, 129), generator=gen)
batch = (ids[:, :-1], ids[:, 1:])             # X и Y со сдвигом

model, opt = fresh()
losses = [train_step(model, batch, opt) for _ in range(5)]
assert all(p.grad.shape == p.shape for p in model.parameters())
assert all(b < a for a, b in zip(losses, losses[1:]))   # 9,07 -> 8,67
w, before = model.output.weight, model.output.weight.detach().clone()
train_step(model, batch, opt)
assert torch.allclose(w, before - 0.02 * w.grad)        # step = формула

model, opt = fresh()                          # поломка: забыли обнулить
train_step(model, batch, opt)
train_step(model, batch, opt)
right = grad_norm(model)                      # g второго шага: 2,89
model, opt = fresh()
train_step(model, batch, opt)
train_step(model, batch, opt, zero_grad=False)
wrong = grad_norm(model)                      # g1 + g2: 5,95
assert 1.8 < wrong / right < 2.3


print("booklm_init_and_step: все проверки пройдены")
