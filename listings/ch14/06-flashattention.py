"""Листинги главы 14 «Инференс» книги «Большие языковые модели», файл 06-flashattention.

Код и вывод — ровно как в книге; текст книги сюда не входит. Листинги идут по порядку и
опираются друг на друга: запускайте файл целиком или ячейками сверху вниз (разметка # %%).
Проверка, что вывод совпал с книгой: python scripts/check_listings.py listings/ch14/06-flashattention.py
"""

# %% Листинг «Память матрицы оценок» — раздел «FlashAttentionFlashAttention»
for T in (512, 4096, 32768):
    mib = T * T * 2 / 2**20                   # FP16: 2 байта на оценку
    print(f"T={T:>5}: одна матрица оценок {mib:7.1f} МБ на голову")
# Вывод в книге:
# | T=  512: одна матрица оценок     0.5 МБ на голову
# | T= 4096: одна матрица оценок    32.0 МБ на голову
# | T=32768: одна матрица оценок  2048.0 МБ на голову

# %% Листинг «Потоковый softmax по плиткам» — раздел «Потоковый softmax»
import math
import torch

def online_attention_row(s, values, tiles, use_alpha=True):
    m, ell, o = float("-inf"), 0.0, 0.0       # пустое состояние
    for tile in tiles:
        new_m = max(m, s[tile].max().item())
        alpha = math.exp(m - new_m) if use_alpha else 1.0
        p = torch.exp(s[tile] - new_m)        # экспоненты плитки
        ell = alpha * ell + p.sum().item()
        o = alpha * o + (p @ values[tile]).item()
        m = new_m
    return m, ell, o

s = torch.tensor([2.0, 0.0, 3.0, 2.0])        # оценки строки
values = torch.tensor([1.0, 3.0, 0.0, 2.0])   # числа вместо векторов V
tiles = (slice(0, 2), slice(2, 4))            # две плитки по две оценки
m, ell, o = online_attention_row(s, values, tiles)
print(f"m={m}, ell={ell:.3f}, o={o:.3f}, o/ell={o / ell:.3f}")
print(f"softmax всей строки: {(s.softmax(dim=0) @ values).item():.3f}")
_, ell_bad, o_bad = online_attention_row(s, values, tiles, False)
print(f"без alpha: {o_bad / ell_bad:.3f}")
# Вывод в книге:
# | m=3.0, ell=1.786, o=1.253, o/ell=0.702
# | softmax всей строки: 0.702
# | без alpha: 0.856

# %% Листинг «causal_attention (копия для проверки)» — раздел «Внимание по плиткам»
import math
import torch

def causal_attention(q, k, v):
    scores = q @ k.T / math.sqrt(q.size(-1))
    n, m = len(q), len(k)
    future = torch.ones(n, m, dtype=torch.bool).triu(m - n + 1)
    scores = scores.masked_fill(future, float("-inf"))
    return scores.softmax(dim=-1) @ v

# %% Листинг «Точное причинное внимание с потоковым softmax» — раздел «Внимание по плиткам»
def tiled_causal_attention(q, k, v, tile=64, log=None):
    T, D = q.shape                                    # одна голова
    out = torch.empty_like(q)
    for qs in range(0, T, tile):
        q_blk = q[qs:qs + tile]                       # [Bq, D]
        q_pos = torch.arange(qs, qs + len(q_blk))
        m = torch.full((len(q_blk),), float("-inf"))  # состояние строк
        ell = torch.zeros(len(q_blk))
        acc = torch.zeros(len(q_blk), D)
        for ks in range(0, qs + len(q_blk), tile):    # будущее не берём
            if log is not None:
                log.append((qs // tile, ks // tile))
            k_blk, v_blk = k[ks:ks + tile], v[ks:ks + tile]
            k_pos = torch.arange(ks, ks + len(k_blk))
            scores = q_blk @ k_blk.T / math.sqrt(D)
            scores = scores.masked_fill(k_pos[None] > q_pos[:, None],
                                        float("-inf"))
            new_m = torch.maximum(m, scores.max(dim=1).values)
            alpha = torch.exp(m - new_m)
            p = torch.exp(scores - new_m[:, None])
            ell = alpha * ell + p.sum(dim=1)
            acc = alpha[:, None] * acc + p @ v_blk
            m = new_m
        out[qs:qs + len(q_blk)] = acc / ell[:, None]
    return out

torch.manual_seed(7)
q, k, v = (torch.randn(129, 64) for _ in range(3))   # 129 = 4 * 32 + 1
log = []
tiled = tiled_causal_attention(q, k, v, tile=32, log=log)
for i in range(5):
    print(f"Q{i}:", *[f"K{j}" for qi, j in log if qi == i])
sdpa = torch.nn.functional.scaled_dot_product_attention(
    q, k, v, is_causal=True)                         # ядро PyTorch
same = torch.allclose(tiled, causal_attention(q, k, v), atol=1e-5)
print(len(log), tiled.shape, same, torch.allclose(tiled, sdpa, atol=1e-5))
# Вывод в книге:
# | Q0: K0
# | Q1: K0 K1
# | Q2: K0 K1 K2
# | Q3: K0 K1 K2 K3
# | Q4: K0 K1 K2 K3 K4
# | 15 torch.Size([129, 64]) True True

# %% Листинг «Граф FX, записанный TorchDynamo» — раздел «Компилятор torch.compile»
import torch
import torch.nn.functional as F

def norm_swiglu(x, w, gate, up):       # RMSNorm, затем SwiGLU без down
    h = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * w
    return F.silu(h @ gate) * (h @ up)

graphs = []
def look(gm, example_inputs):          # «компилятор», который только смотрит
    graphs.append(gm)
    return gm.forward                  # и выполняет граф как есть

torch.manual_seed(0)
x, w = torch.randn(8, 256), torch.ones(256)
gate, up = torch.randn(256, 768) / 16, torch.randn(256, 768) / 16
f = torch.compile(norm_swiglu, backend=look)
print(torch.allclose(f(x, w, gate, up), norm_swiglu(x, w, gate, up)))
ops = [getattr(n.target, "__name__", n.target) for n in graphs[0].graph.nodes
       if n.op in ("call_function", "call_method")]
print(len(graphs), "граф,", len(ops), "операций:", *ops)
# Вывод в книге:
# | True
# | 1 граф, 10 операций: pow mean add rsqrt mul mul matmul silu matmul mul

# %% Листинг «Три причины разрыва графа» — раздел «Компилятор torch.compile»
def rms(x, w):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * w

def with_print(x, w):
    h = rms(x, w)
    print("отладка:", tuple(h.shape))  # печать посреди графа
    return h + x

def with_item(x, w):
    h = rms(x, w)
    return h / h.abs().max().item() + x  # число уходит в Python

def with_branch(x, w):
    h = rms(x, w)
    if h.mean() > 0:                   # ветвление по значению тензора
        return h + x
    return h - x

for f in (with_print, with_item, with_branch):
    torch._dynamo.reset()              # забыть прошлые графы
    report = torch._dynamo.explain(f)(x, w)
    print(f"{f.__name__}: графов {report.graph_count},"
          f" разрывов {report.graph_break_count}")
    print("  причина:", report.break_reasons[0].reason.splitlines()[0])
# Вывод в книге:
# | отладка: (8, 256)
# | with_print: графов 2, разрывов 1
# |   причина: Failed to trace builtin operator
# | with_item: графов 2, разрывов 1
# |   причина: Unsupported Tensor.item() call with capture_scalar_outputs=False
# | with_branch: графов 2, разрывов 1
# |   причина: generic_jump TensorVariable()

# %% Листинг «Что сгенерировал TorchInductor» — раздел «Компилятор torch.compile»
from torch._inductor.utils import run_and_get_code

torch._dynamo.reset()
y, (code,) = run_and_get_code(torch.compile(norm_swiglu), x, w, gate, up)
print(torch.allclose(y, norm_swiglu(x, w, gate, up), atol=1e-5))
call = code[code.index("def call(self, args)"):]   # что делает проход
for line in call.splitlines():
    line = line.strip()
    if line.startswith(("cpp_fused", "extern_kernels")):
        print(line.split("(")[0])
# Вывод в книге:
# | True
# | cpp_fused_add_mean_mul_pow_rsqrt_0
# | extern_kernels.mm
# | extern_kernels.mm
# | cpp_fused_mul_silu_1

# %% Листинг «Перекомпиляция при новой длине пакета» — раздел «Режимы и формы тензоров»
def compiled_shapes(dynamic):
    shapes = []
    def look(gm, example_inputs):
        x = next(n for n in gm.graph.nodes if n.name == "l_x_")
        shapes.append(list(x.meta["example_value"].shape))  # форма x
        return gm.forward
    torch._dynamo.reset()
    f = torch.compile(norm_swiglu, backend=look, dynamic=dynamic)
    for T in (1, 2, 4, 8, 16, 3):      # пакеты разного размера
        f(torch.randn(T, 256), w, gate, up)
    return shapes

for dynamic in (False, None):
    print(f"dynamic={dynamic}:", *compiled_shapes(dynamic))
# Вывод в книге:
# | dynamic=False: [1, 256] [2, 256] [4, 256] [8, 256] [16, 256] [3, 256]
# | dynamic=None: [1, 256] [s77, 256]

# %% Листинг «Правило принятия черновика: проверка частот» — раздел «Спекулятивное декодирование»
import torch
torch.manual_seed(0)
p = torch.tensor([0.2, 0.6, 0.2])    # основная: окне, подоконнике, столе
q = torch.tensor([0.5, 0.1, 0.4])    # черновик
n = 100_000
x = torch.multinomial(q, n, replacement=True)      # предложения черновика
accept = torch.rand(n) < (p[x] / q[x]).clamp(max=1)    # min(1, p/q)
rest = (p - q).clamp(min=0)                        # недобор max(0, p - q)
redraw = torch.multinomial(rest / rest.sum(), n, replacement=True)
final = torch.where(accept, x, redraw)             # отказ: из недобора
print(f"принято предложений: {accept.float().mean():.3f}")
rules = [("принимать всегда", x), ("правило min(1, p/q)", final)]
for name, tokens in rules:
    freq = torch.bincount(tokens, minlength=3) / n
    print(f"{name:>19}: {[round(f, 3) for f in freq.tolist()]}")
# Вывод в книге:
# | принято предложений: 0.501
# |    принимать всегда: [0.499, 0.1, 0.401]
# | правило min(1, p/q): [0.201, 0.599, 0.201]

# %% Листинг «Черновая голова MTP» — раздел «Многотокенное предсказание и черновые головы»
import torch
import torch.nn as nn
import torch.nn.functional as F
from booklm.model import RMSNorm

class DraftHead(nn.Module):          # h_t и токен t+1 -> логиты токена t+2
    def __init__(self, embed, d=256, d_ff=768):
        super().__init__()
        self.embed = embed                                  # у модели
        self.norm_h, self.norm_e = RMSNorm(d), RMSNorm(d)
        self.proj = nn.Linear(2 * d, d, bias=False)         # [h; e] -> d
        self.norm_f, self.norm_out = RMSNorm(d), RMSNorm(d)
        self.gate = nn.Linear(d, d_ff, bias=False)          # SwiGLU
        self.up = nn.Linear(d, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d, bias=False)
        nn.init.zeros_(self.down.weight)                    # сперва молчит

    def forward(self, h, next_ids):                         # h: [B, T, d]
        e = self.embed.weight[next_ids]                     # [B, T, d]
        u = self.proj(torch.cat([self.norm_h(h), self.norm_e(e)], dim=-1))
        f = self.norm_f(u)
        u = u + self.down(F.silu(self.gate(f)) * self.up(f))
        return self.norm_out(u) @ self.embed.weight.T       # [B, T, V]

# %% Листинг «Скрытые векторы BookLM и голова на них» — раздел «Многотокенное предсказание и черновые головы»
from booklm.lab import load, prompt_ids
tok, booklm = load("base")

def hidden(model, ids, caches=None):     # логиты и h_t
    h = model.embed(ids)
    for i, block in enumerate(model.blocks):
        h, _ = block(h, model.rope_cos, model.rope_sin,
                     None if caches is None else caches[i])
    h = model.norm(h)
    return h @ model.embed.weight.T, h

FOX = "Жила-была маленькая лиса. Однажды она"
ids = torch.tensor([prompt_ids(tok, FOX)])           # [1, T]
with torch.no_grad():
    logits, h = hidden(booklm, ids)
    head = DraftHead(booklm.embed)
    guess = head(h[:, :-1], ids[:, 1:])     # позиция t знает токен t+1
print(ids.shape, h.shape, guess.shape)
same = torch.allclose(logits, booklm(ids), atol=1e-5)
print("логиты как у BookLM:", same)
own = sum(p.numel() for n, p in head.named_parameters()
          if "embed" not in n)                    # эмбеддинги не её
print(f"своих параметров у головы: {own} ({own / booklm.num_params():.1%})")
# Вывод в книге:
# | torch.Size([1, 9]) torch.Size([1, 9, 256]) torch.Size([1, 8, 8192])
# | логиты как у BookLM: True
# | своих параметров у головы: 721920 (10.9%)

# %% Листинг «Обучение черновой головы на замороженной BookLM» — раздел «Многотокенное предсказание и черновые головы»
# Проверка этот листинг не запускает: ему нужны сеть, видеокарта, сервер или внешний файл.
# from booklm.experiments.ch14_mtp import train_head
#
# head, log = train_head(booklm, steps=1000, batch=16, T=256, log_every=250)
# for step, loss in log:
#     print(f"шаг {step:3d}: ошибка на токене t+2 {loss:.2f}")

# %% Листинг «Доля принятых черновиков и цикл с головой» — раздел «Многотокенное предсказание и черновые головы»
from booklm.lab import load_model
from booklm.experiments.ch14_mtp import (load_head, acceptance,
    bigram_table, speculative_greedy)

head = load_head(booklm)                      # обучена листингом выше
early = load_model("step:762")
rates, total = acceptance(booklm, head, early, bigram_table())
print(f"доля принятых черновиков на {total} токенах жадных сказок:")
for name, r in rates.items():
    print(f"  {name:<16} {r:.1%}")
spec, calls = speculative_greedy(booklm, head, ids, 200)
ref = booklm.generate(ids, max_new=200, temperature=0)[0].tolist()
print(f"лиса, 200 токенов: совпало с жадным: {spec == ref},",
      f"вызовов: {calls}")
# Вывод в книге:
# | доля принятых черновиков на 4729 токенах жадных сказок:
# |   таблица биграмм  32.0%
# |   BookLM шага 762  72.1%
# |   голова MTP       63.6%
# | лиса, 200 токенов: совпало с жадным: True, вызовов: 121

# %% Листинг «Время генерации с черновой головой на процессоре» — раздел «Многотокенное предсказание и черновые головы»
# Проверка этот листинг не запускает: ему нужны сеть, видеокарта, сервер или внешний файл.
# from booklm.experiments.ch14_mtp import timing
# timing(booklm, head, ids, new=200)
