"""Листинги главы 16 «Распределённое обучение» книги «Большие языковые модели», файл 16-distributed.

Код и вывод — ровно как в книге; текст книги сюда не входит. Листинги идут по порядку и
опираются друг на друга: запускайте файл целиком или ячейками сверху вниз (разметка # %%).
Проверка, что вывод совпал с книгой: python scripts/check_listings.py listings/ch16/16-distributed.py
"""

# %% Листинг «Фаза reduce-scatter по шагам рисунка» — раздел «Кольцевой all-reduce»
def chunk(c, n):
    return range(c * n, (c + 1) * n)   # позиции фрагмента c

def fragment_size(buf):
    p = len(buf)
    assert len(buf[0]) % p == 0, "длина массива должна делиться на p"
    return len(buf[0]) // p

def reduce_scatter(buf, show=False):
    p, n = len(buf), fragment_size(buf)
    for s in range(p - 1):
        # ранг r шлёт соседу фрагмент (r - s) mod p
        sent = [[buf[r][i] for i in chunk((r - s) % p, n)]
                for r in range(p)]
        for r in range(p):
            c = (r - 1 - s) % p        # пришёл от ранга r - 1
            for i, v in zip(chunk(c, n), sent[(r - 1) % p]):
                buf[r][i] += v         # получил и прибавил своё
        if show:                       # для фрагментов из одного числа
            print(f"шаг {s + 1}:", "  ".join(
                f"c{(r - 1 - s) % p}={buf[r][(r - 1 - s) % p]}"
                for r in range(p)))

values = [[1, 2, 3, 4], [10, 20, 30, 40],
          [100, 200, 300, 400], [1000, 2000, 3000, 4000]]
buf = [row[:] for row in values]       # у каждого ранга своя копия
reduce_scatter(buf, show=True)
# Вывод в книге:
# | шаг 1: c3=4004  c0=11  c1=220  c2=3300
# | шаг 2: c2=3303  c3=4044  c0=111  c1=2220
# | шаг 3: c1=2222  c2=3333  c3=4444  c0=1111

# %% Листинг «Фаза all-gather и кольцевой all-reduce» — раздел «Кольцевой all-reduce»
def all_gather(buf, show=False):
    p, n = len(buf), fragment_size(buf)
    for s in range(p - 1):
        # ранг r шлёт соседу готовый фрагмент (r + 1 - s) mod p
        sent = [[buf[r][i] for i in chunk((r + 1 - s) % p, n)]
                for r in range(p)]
        for r in range(p):
            c = (r - s) % p
            for i, v in zip(chunk(c, n), sent[(r - 1) % p]):
                buf[r][i] = v          # только записал, без сложения
        if show:
            print(f"шаг {p + s}:", "  ".join(
                f"c{(r - s) % p}={buf[r][(r - s) % p]}"
                for r in range(p)))

def ring_allreduce(data, show=False):
    buf = [row[:] for row in data]     # у каждого ранга своя копия
    reduce_scatter(buf, show)
    all_gather(buf, show)
    return buf

all_gather(buf, show=True)             # buf после первой фазы
print(buf[0], buf == [buf[0]] * 4)
try:
    ring_allreduce([[1, 2, 3, 4, 5], [10, 20, 30, 40, 50]])
except AssertionError as err:
    print("пять чисел на два ранга:", err)
# Вывод в книге:
# | шаг 4: c0=1111  c1=2222  c2=3333  c3=4444
# | шаг 5: c3=4444  c0=1111  c1=2222  c2=3333
# | шаг 6: c2=3333  c3=4444  c0=1111  c1=2222
# | [1111, 2222, 3333, 4444] True
# | пять чисел на два ранга: длина массива должна делиться на p

# %% Листинг «Шаг DDP на кольцевом all-reduce» — раздел «Параллелизм данных»
def ddp_step(weights, grads_by_rank, lr):
    p = len(grads_by_rank)
    summed = ring_allreduce(grads_by_rank)  # сумма у каждого ранга
    return [[w - lr * g / p for w, g in zip(weights, row)]
            for row in summed]

weights = [1.0, 2.0, 3.0, 4.0]
grads_by_rank = [[0.2, 0.4, 0.6, 0.8],      # ранг 0
                 [0.4, 0.2, 1.0, 0.6]]      # ранг 1
for r, w in enumerate(ddp_step(weights, grads_by_rank, 0.1)):
    print(f"ранг {r}:", [round(x, 4) for x in w])
# Вывод в книге:
# | ранг 0: [0.97, 1.97, 2.92, 3.93]
# | ранг 1: [0.97, 1.97, 2.92, 3.93]

# %% Листинг «Накопление градиента в DDP» — раздел «Параллелизм данных»
# Проверка этот листинг не запускает: ему нужны сеть, видеокарта, сервер или внешний файл.
# *first, last = micro_batches           # k микропакетов одного шага
# with model.no_sync():                  # градиенты копятся локально
#     for x, y in first:
#         loss_fn(model(x), y).backward()
# x, y = last
# loss_fn(model(x), y).backward()        # здесь DDP запускает all-reduce
# optimizer.step()

# %% Листинг «Два ранга BookLM на gloo» — раздел «Параллелизм данных»
import tempfile, torch
import torch.distributed as dist
import torch.multiprocessing as mp
from booklm.lab import load_model
from booklm.experiments.ch16 import windows, loss_on

x, y = windows(8, 128)                 # 8 окон по 128 токенов
out = tempfile.mkdtemp()               # сюда ранг 0 положит итог

def worker(rank, parts, fair):         # один процесс = один ранг
    dist.init_process_group("gloo", rank=rank, world_size=2,
                            init_method="tcp://127.0.0.1:29531")
    model = load_model("base")
    mine = parts[rank]                 # номера окон этого ранга
    loss_on(model, x[mine], y[mine]).backward()
    for p in model.parameters():
        p.grad *= len(mine) / 8 if fair else 1 / 2
        dist.all_reduce(p.grad)        # сумма по двум рангам
    if rank == 0:
        torch.save([p.grad for p in model.parameters()], f"{out}/g.pt")
    dist.destroy_process_group()

def ddp_grads(parts, fair=False):
    mp.start_processes(worker, (parts, fair), nprocs=2,
                       start_method="fork")
    return torch.load(f"{out}/g.pt")

half, uneven = [[0, 1, 2, 3], [4, 5, 6, 7]], [[0, 1, 2], [3, 4, 5, 6, 7]]
runs = {"4 + 4, среднее средних": ddp_grads(half),
        "3 + 5, среднее средних": ddp_grads(uneven),
        "3 + 5, веса 3/8 и 5/8": ddp_grads(uneven, fair=True)}
ref = load_model("base")               # эталон: один процесс, 8 окон
loss_on(ref, x, y).backward()
ref_g = [p.grad for p in ref.parameters()]
print(f"градиенты: {sum(g.numel() * 4 for g in ref_g) / 2**20:.1f} МБ")
for name, grads in runs.items():
    d = max((g - r).abs().max().item() for g, r in zip(grads, ref_g))
    print(f"{name}: разница с эталоном {d:.1e}")
# Вывод в книге:
# | градиенты: 25.3 МБ
# | 4 + 4, среднее средних: разница с эталоном 6.7e-08
# | 3 + 5, среднее средних: разница с эталоном 1.7e-02
# | 3 + 5, веса 3/8 и 5/8: разница с эталоном 3.2e-07

# %% Листинг «Память обучения BookLM» — раздел «Память обучения»
from booklm.experiments.ch16 import saved_bytes

model = load_model("init").train()
opt = torch.optim.AdamW(model.parameters(), lr=2e-3, betas=(0.9, 0.95))
x, y = windows(8, 512)
loss_on(model, x, y).backward()
opt.step()                               # появились градиенты и оба момента
ps = list(model.parameters())
def mib(ts):
    return sum(t.numel() * t.element_size() for t in ts) / 2**20
for name, ts in (("веса", ps), ("градиенты", [p.grad for p in ps]),
                 ("exp_avg", [opt.state[p]["exp_avg"] for p in ps]),
                 ("exp_avg_sq", [opt.state[p]["exp_avg_sq"] for p in ps])):
    print(f"{name:<11}{mib(ts):5.1f} МБ")
model.zero_grad(set_to_none=True)
for B in (8, 64):
    x, y = windows(B, 512)
    kept = saved_bytes(model, x, y) / 2**20
    print(f"пакет {B:2d} x 512: сохранено {kept:4.0f} МБ")
# Вывод в книге:
# | веса        25.3 МБ
# | градиенты   25.3 МБ
# | exp_avg     25.3 МБ
# | exp_avg_sq  25.3 МБ
# | пакет  8 x 512: сохранено  512 МБ
# | пакет 64 x 512: сохранено 4005 МБ

# %% Листинг «Пересчёт блоков BookLM» — раздел «Пересчёт активаций»
from torch.utils.checkpoint import checkpoint

class Recompute(torch.nn.Module):        # блок, который пересчитают
    def __init__(self, block):
        super().__init__()
        self.block = block
    def forward(self, h, cos, sin, cache=None, return_weights=False):
        return checkpoint(self.block, h, cos, sin, use_reentrant=True)

x, y = windows(8, 512)
loss_on(model, x, y).backward()
before = [p.grad.clone() for p in model.parameters()]
model.zero_grad(set_to_none=True)
model.blocks = torch.nn.Sequential(*map(Recompute, model.blocks))
loss_on(model, x, y).backward()
after = [p.grad for p in model.parameters()]
d = max((a - b).abs().max().item() for a, b in zip(after, before))
print("градиенты с пересчётом и без расходятся на", d)
model.zero_grad(set_to_none=True)
for B in (8, 64):
    x, y = windows(B, 512)
    kept = saved_bytes(model, x, y) / 2**20
    print(f"пакет {B:2d} x 512: сохранено {kept:4.0f} МБ")
# Вывод в книге:
# | градиенты с пересчётом и без расходятся на 0.0
# | пакет  8 x 512: сохранено  166 МБ
# | пакет 64 x 512: сохранено 1301 МБ

# %% Листинг «Пик памяти и арифметика шага» — раздел «Пересчёт активаций»
from booklm.experiments.ch16_checkpointing import blocks_memory, step_gflop

plain = load_model("init").train()       # те же блоки без пересчёта
x, y = windows(8, 512)
for name, m in (("без пересчёта", plain), ("с пересчётом", model)):
    kept, peak = (b / 2**20 for b in blocks_memory(m, x))
    gflop = step_gflop(m, x, y)
    print(f"{name:<14}конец прямого {kept:3.0f} МБ, пик {peak:3.0f} МБ, "
          f"шаг {gflop:.0f} GFLOP")
# Вывод в книге:
# | без пересчёта конец прямого 370 МБ, пик 370 МБ, шаг 201 GFLOP
# | с пересчётом  конец прямого  24 МБ, пик  82 МБ, шаг 251 GFLOP

# %% Листинг «Шаг ZeRO-1 и ZeRO-2 на тех же рангах» — раздел «Учебный шаг ZeRO»
def zero_step(weights, grads_by_rank, lr, stage, state, show=False):
    p, n = len(grads_by_rank), len(weights) // len(grads_by_rank)
    g = [row[:] for row in grads_by_rank]
    if stage == 1:
        g = ring_allreduce(g)          # полный градиент у всех
    else:
        reduce_scatter(g)              # только своя готовая сумма
    w = [weights[:] for _ in range(p)]
    for r in range(p):
        mine = chunk((r + 1) % p, n)   # фрагмент владельца r
        for i in mine:                 # SGD с моментом 0,9
            state[r][i] = 0.9 * state[r].get(i, 0.0) + g[r][i] / p
            w[r][i] -= lr * state[r][i]
        if show:
            print(f"ранг {r}: градиент {[round(x, 4) for x in g[r]]},",
                  "момент для", ", ".join(f"w{i}" for i in state[r]))
    all_gather(w)                      # раздаём новые фрагменты
    return w

for stage in (1, 2):
    state = [{} for _ in grads_by_rank]  # у каждого ранга свои моменты
    w = zero_step(weights, grads_by_rank, 0.1, stage, state, show=True)
    print(f"ZeRO-{stage}:", [round(x, 4) for x in w[0]], w[0] == w[1])
# Вывод в книге:
# | ранг 0: градиент [0.6, 0.6, 1.6, 1.4], момент для w2, w3
# | ранг 1: градиент [0.6, 0.6, 1.6, 1.4], момент для w0, w1
# | ZeRO-1: [0.97, 1.97, 2.92, 3.93] True
# | ранг 0: градиент [0.2, 0.4, 1.6, 1.4], момент для w2, w3
# | ранг 1: градиент [0.6, 0.6, 1.0, 0.6], момент для w0, w1
# | ZeRO-2: [0.97, 1.97, 2.92, 3.93] True

# %% Листинг «ZeRO-1 на BookLM и ограничение нормы» — раздел «Учебный шаг ZeRO»
import copy
from booklm.experiments.ch16 import N_PARAMS, num

model = load_model("base")
x, y = windows(8, 128)
loss_on(model, x, y).backward()
sizes = [p.numel() for p in model.parameters()]
cut = next(i for i in range(len(sizes))
           if sum(sizes[:i + 1]) >= N_PARAMS / 2) + 1
ranks = [range(cut), range(cut, len(sizes))]  # два «ранга» ZeRO-1
for r, own in enumerate(ranks):
    n = sum(sizes[i] for i in own)
    print(f"ранг {r}: {len(own)} тензоров, {num(n)} параметров,"
          f" моменты AdamW {8 * n / 2**20:.1f} МБ")

def step(groups, clip=None):              # у каждой группы свой AdamW
    ps = list(copy.deepcopy(model).parameters())
    for p, q in zip(ps, model.parameters()):
        p.grad = q.grad.clone()           # deepcopy не копирует .grad
    norms = [torch.cat([ps[i].grad.flatten() for i in g]).norm()
             for g in groups]
    total = sum(n ** 2 for n in norms) ** 0.5  # all-reduce квадратов
    for g, n in zip(groups, norms):
        limit = {"общая": total, "своя": n}.get(clip, 0)
        for i in g:
            ps[i].grad /= max(1.0, limit) # норма не больше 1,0
        torch.optim.AdamW([ps[i] for i in g], lr=1e-4).step()
    return ps, norms

for clip in (None, "общая", "своя"):
    ref, _ = step([range(len(sizes))], clip)  # эталон: один AdamW
    zero, norms = step(ranks, clip)
    d = max((a - b).abs().max().item() for a, b in zip(ref, zero))
    print(f"норма {clip or 'не ограничена'}: разница с эталоном {d:.1e}")
print("нормы фрагментов:", [round(n.item(), 2) for n in norms],
      "общая:", round(sum(n ** 2 for n in norms).item() ** 0.5, 2))
# Вывод в книге:
# | ранг 0: 18 тензоров, 3 408 896 параметров, моменты AdamW 26.0 МБ
# | ранг 1: 38 тензоров, 3 213 568 параметров, моменты AdamW 24.5 МБ
# | норма не ограничена: разница с эталоном 0.0e+00
# | норма общая: разница с эталоном 3.0e-08
# | норма своя: разница с эталоном 1.2e-05
# | нормы фрагментов: [1.94, 1.52] общая: 2.47

# %% Листинг «Цикл обучения BookLM с DeepSpeed» — раздел «DeepSpeed ZeRO»
# Проверка этот листинг не запускает: ему нужны сеть, видеокарта, сервер или внешний файл.
# model_engine, optimizer, _, scheduler = deepspeed.initialize(
#     model=booklm, optimizer=optimizer, lr_scheduler=scheduler,
#     config="ds_zero3.json")            # новое: обёртка модели
#
# for x, y in batches:                   # x, y --- один микропакет
#     logits = model_engine(x)           # было: booklm(x)
#     loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
#     model_engine.backward(loss)        # было: loss.backward()
#     model_engine.step()                # было: optimizer.step(),
#                                        # zero_grad(), scheduler.step()

# %% Листинг «Каркас FSDP2 для блоков BookLM» — раздел «FSDP2 в PyTorchFSDP»
# Проверка этот листинг не запускает: ему нужны сеть, видеокарта, сервер или внешний файл.
# import os
# import torch
# import torch.distributed as dist
# from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
# from booklm.model import BookLM, BookLMConfig
#
# dist.init_process_group(backend="nccl")
# local_rank = int(os.environ["LOCAL_RANK"])
# torch.cuda.set_device(local_rank)
# print(f"rank {dist.get_rank()} local_rank {local_rank} "
#       f"world {dist.get_world_size()}")
#
# with torch.device("meta"):             # формы без памяти
#     booklm = BookLM(BookLMConfig())
# bf16 = MixedPrecisionPolicy(param_dtype=torch.bfloat16,
#                             reduce_dtype=torch.float32)
# for block in booklm.blocks:
#     fully_shard(block, mp_policy=bf16) # сначала каждый блок
# fully_shard(booklm, mp_policy=bf16)    # потом вся модель
# booklm.to_empty(device="cuda")         # память только под свои фрагменты
# # заполнить веса и заново посчитать буферы RoPE, затем
# optimizer = torch.optim.AdamW(booklm.parameters(), lr=3e-4)
# # и обычный forward, backward, step

# %% Листинг «Матрица 2×2 на двух картах» — раздел «Тензорный параллелизмТензорный параллелизм»
X = [1, 2]
W = [[3, 4], [5, 6]]

def matvec(x, w):
    return [sum(x[i] * w[i][j] for i in range(len(x)))
            for j in range(len(w[0]))]

cols = [[[row[j]] for row in W] for j in range(2)]  # GPU j: столбец j
pieces = [matvec(X, cols[j]) for j in range(2)]
print("по столбцам:", pieces, "-> склеить:", pieces[0] + pieces[1])

partial = [matvec([X[i]], [W[i]]) for i in range(2)]  # GPU i: строка i
print("по строкам:", partial, "-> сложить:", ring_allreduce(partial)[0])
# Вывод в книге:
# | по столбцам: [[13], [16]] -> склеить: [13, 16]
# | по строкам: [[3, 4], [10, 12]] -> сложить: [13, 16]

# %% Листинг «Обратный проход SwiGLU на двух картах» — раздел «Тензорный параллелизмТензорный параллелизм»
from booklm.lab import load, prompt_ids
from booklm.experiments.ch16 import swiglu    # down(silu(gate x) * up x)

tok, model = load("base")
block, got = model.blocks[0], {}
hook = block.norm_2.register_forward_hook(lambda m, i, o: got.update(x=o))
text = "Мама мыла раму, а потом она пошла спать."
with torch.no_grad():
    model(torch.tensor([prompt_ids(tok, text)]))
hook.remove()
torch.manual_seed(0)
r = torch.randn_like(got["x"])           # градиент ошибки по выходу блока

def backward_on(ch):                     # «карта» с каналами ch из 768
    f = block.ffn
    W = [f.gate.weight[ch], f.up.weight[ch], f.down.weight[:, ch]]
    W = [w.detach().clone().requires_grad_() for w in W]
    x = got["x"].clone().requires_grad_()
    (swiglu(x, *W) * r).sum().backward() # ошибка sum(y * r)
    return x.grad, [w.grad for w in W]

gx, gW = backward_on(slice(0, 768))      # весь блок на одной карте
halves = [slice(0, 384), slice(384, 768)]
parts = [backward_on(ch) for ch in halves]
print(f"вход {list(gx.shape)}, наибольший |dL/dx| = {gx.abs().max():.2f}")
print("на карте: gate, up, down", [list(w.shape) for w in parts[0][1]])
for c, (ch, (hx, hW)) in enumerate(zip(halves, parts)):
    full = [gW[0][ch], gW[1][ch], gW[2][:, ch]]
    dw = max((a - b).abs().max().item() for a, b in zip(hW, full))
    print(f"карта {c}: |dL/dx - полный| = {(hx - gx).abs().max():.2f},"
          f" градиенты весов - срез полного = {dw}")
s = parts[0][0] + parts[1][0]            # это и делает all-reduce
print(f"сумма двух карт: |dL/dx - полный| = {(s - gx).abs().max():.0e}")
print(f"веса SwiGLU: {3 * 256 * 768 * 4} байт,",
      f"all-reduce: {gx.numel() * 4} байт")
# Вывод в книге:
# | вход [1, 13, 256], наибольший |dL/dx| = 2.71
# | на карте: gate, up, down [[384, 256], [384, 256], [256, 384]]
# | карта 0: |dL/dx - полный| = 2.22, градиенты весов - срез полного = 0.0
# | карта 1: |dL/dx - полный| = 2.14, градиенты весов - срез полного = 0.0
# | сумма двух карт: |dL/dx - полный| = 2e-06
# | веса SwiGLU: 2359296 байт, all-reduce: 13312 байт

# %% Листинг «All-to-all для экспертов» — раздел «Параллелизм экспертов»
def all_to_all(buf):
    p = len(buf)                       # buf[src][dst]: от src для dst
    return [[buf[src][dst] for src in range(p)] for dst in range(p)]

table = [[10**r * (i + 1) for i in range(4)] for r in range(4)]
print(all_to_all(table)[0])            # как строка all-to-all в таблице
home = {"E1": 0, "E2": 0, "E3": 1, "E4": 1}  # где живёт эксперт
routes = [["E1", "E1", "E3", "E3"],         # выбор на карте 0
          ["E4", "E4", "E4", "E2"]]         # и на карте 1
send = [[[f"{e}<-к{src}" for e in row if home[e] == dst]
         for dst in range(2)] for src, row in enumerate(routes)]
for card, parts in enumerate(all_to_all(send)):
    tokens = [t for part in parts for t in part]
    print(f"карта {card} получила {len(tokens)}:", " ".join(tokens))
# Вывод в книге:
# | [1, 10, 100, 1000]
# | карта 0 получила 3: E1<-к0 E1<-к0 E2<-к1
# | карта 1 получила 5: E3<-к0 E3<-к0 E4<-к1 E4<-к1 E4<-к1

# %% Листинг «Обучение через Ray Train на двух GPU» — раздел «Планировщик и RayRay»
# Проверка этот листинг не запускает: ему нужны сеть, видеокарта, сервер или внешний файл.
# import ray.train, ray.train.torch
# from ray.train import FailureConfig, RunConfig, ScalingConfig
# from ray.train.torch import TorchTrainer
# from booklm.model import BookLM, BookLMConfig
#
# def train_booklm(config):
#     # Ray уже создал группу процессов и выдал этому процессу GPU.
#     model = ray.train.torch.prepare_model(BookLM(BookLMConfig()))  # DDP
#     start = ray.train.get_checkpoint()  # None при первом запуске
#     ...  # если start есть, загрузить из него веса, оптимизатор, шаг
#     ...  # обычный цикл; время от времени сохранение через
#     ...  # ray.train.report(metrics, checkpoint=...)
#
# trainer = TorchTrainer(
#     train_booklm,
#     scaling_config=ScalingConfig(num_workers=2, use_gpu=True),
#     run_config=RunConfig(failure_config=FailureConfig(max_failures=3)),
# )
# trainer.fit()
