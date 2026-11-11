"""Листинги главы 14 «Инференс» книги «Большие языковые модели», файл 05-scheduling-vllm.

Код и вывод — ровно как в книге; текст книги сюда не входит. Листинги идут по порядку и
опираются друг на друга: запускайте файл целиком или ячейками сверху вниз (разметка # %%).
Проверка, что вывод совпал с книгой: python scripts/check_listings.py listings/ch14/05-scheduling-vllm.py
"""

# %% Листинг «Распределитель страниц KV-кэша» — раздел «Учебный распределитель страниц»
from collections import deque

class BlockPool:
    def __init__(self, n_blocks, block_size):
        self.block_size = block_size
        self.free = deque(range(n_blocks))
        self.tables = {}    # запрос -> номера физических страниц
        self.lengths = {}   # запрос -> сколько токенов в кэше

    def add_request(self, request_id):
        self.tables[request_id], self.lengths[request_id] = [], 0

    def append_token(self, request_id):
        if self.lengths[request_id] % self.block_size == 0:  # нужна новая
            if not self.free:
                raise RuntimeError("KV pool exhausted")
            self.tables[request_id].append(self.free.popleft())
        self.lengths[request_id] += 1

    def address(self, request_id, t):
        table = self.tables[request_id]
        return table[t // self.block_size], t % self.block_size

    def finish(self, request_id):
        self.free.extend(self.tables.pop(request_id))
        self.lengths.pop(request_id)

pool = BlockPool(n_blocks=6, block_size=4)
pool.free = deque([4, 1, 3, 0, 5])   # страница 2 занята запросом B
pool.add_request("A")
for _ in range(10):
    pool.append_token("A")
print("таблица A:", pool.tables["A"])
print("токен 9 -> (страница, место):", pool.address("A", 9))
pool.finish("A")
print("свободные страницы:", list(pool.free))
# Вывод в книге:
# | таблица A: [4, 1, 3]
# | токен 9 -> (страница, место): (3, 1)
# | свободные страницы: [0, 5, 4, 1, 3]

# %% Листинг «Внимание, которое читает кэш через таблицу страниц» — раздел «Учебный распределитель страниц»
import torch
import torch.nn.functional as F
from booklm.lab import load, prompt_ids

tok, model = load("base")
text = "Жила-была маленькая лиса. Однажды она пошла"   # 10 токенов
ids = torch.tensor([prompt_ids(tok, text)])
caches = [{"len": 0} for _ in model.blocks]
with torch.no_grad():
    model(ids, caches)                       # обычный кэш
k, v = caches[0]["k"][0], caches[0]["v"][0]  # блок 0: [2, T, 32]
T, n, table = k.size(1), 4, [4, 1, 3]        # n токенов в странице
pool_k, pool_v = torch.zeros(6, 2, n, 32), torch.zeros(6, 2, n, 32)
for t in range(T):                           # кэш -> страницы пула
    pool_k[table[t // n], :, t % n] = k[:, t]
    pool_v[table[t // n], :, t % n] = v[:, t]

def paged_attention(q, pool_k, pool_v, table, length):
    k = pool_k[table].transpose(0, 1).flatten(1, 2)[:, :length]
    v = pool_v[table].transpose(0, 1).flatten(1, 2)[:, :length]
    g = q.size(0) // k.size(0)               # 4 головы Q на пару K, V
    return F.scaled_dot_product_attention(
        q, k.repeat_interleave(g, 0), v.repeat_interleave(g, 0))

torch.manual_seed(0)
q = torch.randn(8, 1, 32)                    # запрос: 8 голов
paged = paged_attention(q, pool_k, pool_v, table, T)
plain = F.scaled_dot_product_attention(
    q, k.repeat_interleave(4, 0), v.repeat_interleave(4, 0))
print("токенов:", T, "| страницы:", table, "| форма:", tuple(paged.shape))
print("расхождение с обычным кэшем:", (paged - plain).abs().max().item())
# Вывод в книге:
# | токенов: 10 | страницы: [4, 1, 3] | форма: (8, 1, 32)
# | расхождение с обычным кэшем: 0.0

# %% Листинг «Непрерывный планировщик на запросах с рисунка» — раздел «Непрерывное формирование пакетов»
class Request:
    def __init__(self, id, prompt, max_new):
        self.id = id
        self.prompt = prompt     # необработанных токенов входа
        self.max_new = max_new   # сколько токенов ответа написать
        self.written = 0

def fits(request, pool):                 # хватит ли страниц на весь вход
    return -(-request.prompt // pool.block_size) <= len(pool.free)

def scheduler_tick(waiting, running, pool, max_batch):
    while waiting and len(running) < max_batch and fits(waiting[0], pool):
        request = waiting.popleft()
        pool.add_request(request.id)
        running.append(request)
    work = {r.id: r.prompt or 1 for r in running}  # весь вход или 1
    survivors, done = [], []
    for r in running:                    # один общий вызов модели
        for _ in range(work[r.id]):
            pool.append_token(r.id)
        r.prompt, r.written = 0, r.written + 1
        if r.written == r.max_new:
            pool.finish(r.id)
            done.append(r.id)
        else:
            survivors.append(r)
    return survivors, work, done

pool = BlockPool(n_blocks=8, block_size=4)
running = [Request("A", 0, 3), Request("B", 0, 8)]
for r in running:                        # входы A и B обработаны раньше
    pool.add_request(r.id)
    for _ in range(4):
        pool.append_token(r.id)
waiting = deque([Request("D", prompt=6, max_new=4)])
for step in range(1, 6):
    running, work, done = scheduler_tick(waiting, running, pool, 2)
    parts = " ".join(f"{k}+{v}" for k, v in work.items())
    print(f"шаг {step}: {parts} = {sum(work.values())};",
          f"свободно {len(pool.free)}", *(f"[{d} готов]" for d in done))
# Вывод в книге:
# | шаг 1: A+1 B+1 = 2; свободно 4
# | шаг 2: A+1 B+1 = 2; свободно 4
# | шаг 3: A+1 B+1 = 2; свободно 6 [A готов]
# | шаг 4: B+1 D+6 = 7; свободно 4
# | шаг 5: B+1 D+1 = 2; свободно 3

# %% Листинг «Обработка входа BookLM частями» — раздел «Обработка входа частями»
from booklm.experiments.ch14 import long_story

story = long_story(448)[None, :448]          # 448 токенов сказки
with torch.no_grad():
    whole = model(story)                     # весь вход одним вызовом
    caches, parts = [{"len": 0} for _ in model.blocks], []
    for a in range(0, 448, 160):             # части по 160 токенов
        parts.append(model(story[:, a:a + 160], caches))
        for c in caches:
            c["len"] += parts[-1].size(1)
chunked = torch.cat(parts, dim=1)
print("размеры частей:", [p.size(1) for p in parts])
print("расхождение логитов:", (chunked - whole).abs().max().item())
# Вывод в книге:
# | размеры частей: [160, 160, 128]
# | расхождение логитов: 0.0

# %% Листинг «Страницы и непрерывный пакет на 32 сказках BookLM» — раздел «Повторное использование общего начала»
import statistics
from collections import deque
from booklm.lab import load
from booklm.experiments.ch14 import dialog_lengths
from labs.ch14.continuous_scheduler import (BlockPool, Request,
                                           scheduler_tick)

tok, model = load("instruct")
rows = dialog_lengths(tok, model)       # (вход, ответ) для 32 сказок
total = [p + a for p, a in rows]
answer = [a for _, a in rows]
med = statistics.median(total)
print(f"длина диалога: мин {min(total)}, медиана {med:.0f},",
      f"макс {max(total)}")
print(f"длина ответа:  мин {min(answer)}, макс {max(answer)}")
per_token = 1536 / 2**20                # МБ на токен
pages = sum(-(-n // 16) for n in total) # страниц по 16 токенов
print(f"резерв на 512:  {32 * 512 * per_token:5.2f} МБ")
print(f"страницы по 16: {pages * 16 * per_token:5.2f} МБ",
      f"({pages} страницы)")
print(f"точно:          {sum(total) * per_token:5.2f} МБ",
      f"(токенов: {sum(total)})")

static = sum(max(answer[i:i + 8]) for i in range(0, 32, 8))
waiting = deque(Request(i, p, a) for i, (p, a) in enumerate(rows))
running, pool, calls = [], BlockPool(n_blocks=10_000, block_size=16), 0
while waiting or running:
    running, _, _ = scheduler_tick(waiting, running, pool, max_batch=8)
    calls += 1
for name, n in (("обычный пакет", static), ("непрерывный", calls)):
    empty = 1 - sum(answer) / (8 * n)
    print(f"{name:14} {n} вызовов, пустых мест {empty:.0%}")
# Вывод в книге:
# | длина диалога: мин 105, медиана 159, макс 220
# | длина ответа:  мин 93, макс 208
# | резерв на 512:  24.00 МБ
# | страницы по 16:  7.78 МБ (332 страницы)
# | точно:           7.50 МБ (токенов: 5122)
# | обычный пакет  789 вызовов, пустых мест 25%
# | непрерывный    674 вызовов, пустых мест 12%

# %% Листинг «Цепочка хешей страниц и префиксное дерево на просьбах BookLM» — раздел «Префиксное дерево SGLang»
import hashlib
from booklm.experiments.ch14_disagg_radix import sft_requests

def page_hashes(ids, size=16):          # хеши полных страниц цепочкой
    h, out = b"", []
    for a in range(0, len(ids) - size + 1, size):
        h = hashlib.sha256(h + str(ids[a:a + size]).encode()).digest()
        out.append(h)
    return out

def pages_hit(known, ids, size=16):     # токены из совпавших страниц
    hashes, hit = page_hashes(ids, size), 0
    while hit < len(hashes) and hashes[hit] in known:
        hit += 1
    known.update(hashes)
    return min(hit * size, len(ids) - 1)

def tree_hit(root, ids):                # спуск по дереву, затем вставка
    node, n = root, 0
    while n < len(ids) and ids[n] in node:
        node, n = node[ids[n]], n + 1
    for t in ids[n:]:
        node = node.setdefault(t, {})
    return min(n, len(ids) - 1)         # последний токен считаем всегда

requests = sft_requests(500)
known, root = set(), {}
by_pages = sum(pages_hit(known, ids) for ids in requests)
by_tree = sum(tree_hit(root, ids) for ids in requests)
total = sum(map(len, requests))
print(f"запросов {len(requests)}, токенов {total},",
      f"средняя длина {total / len(requests):.1f}")
for name, hit in (("страницы по 16:", by_pages), ("дерево:", by_tree)):
    print(f"{name:15} {hit} токенов из кэша ({hit / total:.1%})")
# Вывод в книге:
# | запросов 500, токенов 12571, средняя длина 25.1
# | страницы по 16: 832 токенов из кэша (6.6%)
# | дерево:         5280 токенов из кэша (42.0%)

# %% Листинг «Кэш BookLM уходит в другой процесс» — раздел «Раздельная обработка входа и генерации»
import subprocess, sys
import torch
from booklm.lab import ROOT, load
from booklm.experiments.ch14 import long_story

def pack_kv(caches, dtype):
    kv = torch.stack([torch.stack([c["k"], c["v"]]) for c in caches])
    return kv.to(dtype).contiguous().view(torch.uint8).numpy().tobytes()

tok, model = load("base")
prompt = long_story(448)[None, :400]                 # 400 токенов сказки
with torch.no_grad():
    caches = [{"len": 0} for _ in model.blocks]      # машина A
    first = model(prompt, caches)[0, -1].argmax().item()
    whole = model.generate(prompt, max_new=40,       # одна машина
                           temperature=0)[0, 400:].tolist()
for dtype in (torch.float32, torch.bfloat16):
    blob = pack_kv(caches, dtype)
    b = subprocess.run(
        [sys.executable, "-m", "booklm.experiments.ch14_disagg_radix",
         "decode", str(first), "400", str(dtype), "40"],
        input=blob, capture_output=True, check=True, cwd=ROOT)
    got = [int(x) for x in b.stdout.split()]
    same = next((i for i, x in enumerate(got) if x != whole[i]), 40)
    print(f"{dtype}: {len(blob):,} байт, совпало {same} из 40")
print("продолжение:", tok.decode(whole[:whole.index(2)]))   # до <eos>
# Вывод в книге:
# | torch.float32: 1,228,800 байт, совпало 40 из 40
# | torch.bfloat16: 614,400 байт, совпало 40 из 40
# | продолжение:  будет ещё много приключений с бабушкой и дедушкой. Они будут любить друг друга».

# %% Листинг «Пакет из трёх запросов к разным адаптерам одного слоя» — раздел «Пакет с разными адаптерами»
import torch

W0 = torch.load("models/booklm/booklm-base.pt")
AB = torch.load("models/booklm/booklm-instruct-lora.pt")["adapter"]
name = "blocks.0.attention.q"
W = W0[name + ".weight"]                       # [256, 256], W_Q блока 0
g = torch.Generator().manual_seed(0)
adapters = {                                   # имя -> (A, B, alpha/r)
    "сказки": (AB[name + ".A"], AB[name + ".B"], 2.0),   # r = 16
    "другой": (torch.randn(8, 256, generator=g) / 16,    # r = 8
               torch.randn(256, 8, generator=g) / 16, 2.0)}

def sgmv(x, W, adapters, starts, names):
    y = x @ W.T                                # общий путь: весь пакет
    for lo, hi, a in zip(starts, starts[1:], names):
        if a is not None:                      # None: чистая основа
            A, B, s = adapters[a]
            y[lo:hi] += s * (x[lo:hi] @ A.T) @ B.T   # сжали, развернули
    return y

names, starts = ["сказки", None, "другой"], [0, 3, 4, 6]
x = torch.randn(6, 256, generator=g)           # 6 токенов пакета
y = sgmv(x, W, adapters, starts, names)
for lo, hi, a in zip(starts, starts[1:], names):
    alone = sgmv(x[lo:hi], W, adapters, [0, hi - lo], [a])  # поштучно
    W_a = W                                    # слитые веса запроса
    if a is not None:
        A, B, s = adapters[a]
        W_a = W + s * B @ A
    print(f"{a or 'основа':6} токены {lo}-{hi - 1}:",
          f"поштучно {(y[lo:hi] - alone).abs().max():.0e},",
          f"слитые {(y[lo:hi] - x[lo:hi] @ W_a.T).abs().max():.0e}")
# Вывод в книге:
# | сказки токены 0-2: поштучно 0e+00, слитые 2e-06
# | основа токены 3-3: поштучно 9e-07, слитые 9e-07
# | другой токены 4-5: поштучно 0e+00, слитые 2e-06

# %% Листинг «Слой со слотами адаптеров для всей BookLM» — раздел «Кэш общего начала и адаптер»
import torch.nn as nn
from booklm.lab import load, prompt_ids

class SlotLoRA(nn.Module):           # основа и адаптеры в слотах
    def __init__(self, base, A_all, B_all):
        super().__init__()           # A_all: [3 слота, 16, d_in]
        self.base, self.A_all, self.B_all = base, A_all, B_all

    def forward(self, x):            # x: [пакет, T, d_in]
        A, B = self.A_all[slot], self.B_all[slot]  # слот каждой строки
        h = torch.einsum("btd,brd->btr", x, A)      # сжатие до 16
        return self.base(x) + torch.einsum("btr,bor->bto", h, B)

def add_slots(model, r_max=16):      # слот 0: нули, 1: сказки, 2: другой
    for i, (name, layer) in enumerate(list(model.named_modules())):
        if name + ".A" not in AB:    # 42 матрицы с адаптером главы 8
            continue
        d_out, d_in = layer.weight.shape
        A_all = torch.zeros(3, r_max, d_in)
        B_all = torch.zeros(3, d_out, r_max)
        A_all[1], B_all[1] = AB[name + ".A"], 2.0 * AB[name + ".B"]
        g = torch.Generator().manual_seed(i)     # r = 8, дальше нули
        A_all[2, :8] = torch.randn(8, d_in, generator=g) / d_in**0.5
        B_all[2, :, :8] = 2.0 * torch.randn(d_out, 8, generator=g) / 16
        parent, attr = name.rsplit(".", 1)
        setattr(model.get_submodule(parent), attr,
                SlotLoRA(layer, A_all, B_all))

tok, multi = load("base")
add_slots(multi)
n = sum(isinstance(m, SlotLoRA) for m in multi.modules())
print("обёрнуто матриц:", n)
# Вывод в книге:
# | обёрнуто матриц: 42

# %% Листинг «Три адаптера в одном пакете BookLM» — раздел «Кэш общего начала и адаптер»
_, base = load("base")
_, merged = load("instruct")                  # слитая модель главы 8
ask = "Напиши сказку про кота."
ids = torch.tensor([prompt_ids(tok, ask, chat=True)])
slot = torch.tensor([0, 1, 2])                # три запроса, один вход
caches = [{"len": 0} for _ in multi.blocks]
with torch.no_grad():
    logits = multi(ids.repeat(3, 1), caches)
    d_base = (logits[0] - base(ids)[0]).abs().max()
    d_merged = (logits[1] - merged(ids)[0]).abs().max()
    print(f"основа против base: {d_base:.0e} |",
          f"сказки против слитой: {d_merged:.0e}")
    out = multi.generate(ids.repeat(3, 1), max_new=10, temperature=0)
for who, row in zip(("основа", "сказки", "другой"), out[:, ids.size(1):]):
    print(f"{who:6}", tok.decode(row.tolist()).strip())
print("блок:        ", "  ".join(f"{b:4d}" for b in range(6)))
gaps = [(c["k"][0] - c["k"][1]).norm() / c["k"][0].norm() for c in caches]
print("K: основа/сказки", "  ".join(f"{g:4.0%}" for g in gaps))
# Вывод в книге:
# | основа против base: 0e+00 | сказки против слитой: 2e-05
# | основа кот, который хотел поймать кота. Он был очень
# | сказки Жил-был маленький кот. Кот был очень голоден
# | другой особенный кот может поедет по своему лучшему другу
# | блок:            0     1     2     3     4     5
# | K: основа/сказки  33%   14%   26%   50%   36%   48%
