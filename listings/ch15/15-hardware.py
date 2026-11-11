"""Листинги главы 15 «Устройство GPU» книги «Большие языковые модели», файл 15-hardware.

Код и вывод — ровно как в книге; текст книги сюда не входит. Листинги идут по порядку и
опираются друг на друга: запускайте файл целиком или ячейками сверху вниз (разметка # %%).
Проверка, что вывод совпал с книгой: python scripts/check_listings.py listings/ch15/15-hardware.py
"""

# %% Листинг «Машина с загрузкой и сохранением» — раздел «Инструкции и регистры»
memory = {"x_i": 2.0, "a": 3.0, "b_i": 4.0, "y_i": 0.0}
regs = {}

def ld(r, addr):            # загрузка: память -> регистр
    regs[r] = memory[addr]
def fma(d, a, b, c):        # d = a*b + c, операнды только из регистров
    regs[d] = regs[a] * regs[b] + regs[c]
def st(addr, r):            # сохранение: регистр -> память
    memory[addr] = regs[r]
program = [(ld, "r0", "x_i"), (ld, "r1", "a"), (ld, "r2", "b_i"),
           (fma, "r3", "r1", "r0", "r2"), (st, "y_i", "r3")]
for step, (op, *args) in enumerate(program, start=1):
    op(*args)
    print(step, f"{op.__name__:3s}", regs, "y_i =", memory["y_i"])
# Вывод в книге:
# | 1 ld  {'r0': 2.0} y_i = 0.0
# | 2 ld  {'r0': 2.0, 'r1': 3.0} y_i = 0.0
# | 3 ld  {'r0': 2.0, 'r1': 3.0, 'r2': 4.0} y_i = 0.0
# | 4 fma {'r0': 2.0, 'r1': 3.0, 'r2': 4.0, 'r3': 10.0} y_i = 0.0
# | 5 st  {'r0': 2.0, 'r1': 3.0, 'r2': 4.0, 'r3': 10.0} y_i = 10.0

# %% Листинг «Запуск ядра axpy без видеокарты» — раздел «Компиляция ядра CUDA»
import numpy as np

def axpy_kernel(blockIdx, blockDim, threadIdx, x, b, y, a, n):
    i = blockIdx * blockDim + threadIdx   # глобальный номер элемента
    if i < n:                             # лишние потоки простаивают
        y[i] = a * x[i] + b[i]

def launch(kernel, blocks, threads, *args):   # <<<blocks, threads>>>
    for block in range(blocks):        # на GPU параллельно,
        for thread in range(threads):  # здесь по очереди
            kernel(block, threads, thread, *args)

n, threads = 1000, 256
blocks = (n + threads - 1) // threads
x, b, y = np.full(n, 2.0), np.full(n, 4.0), np.zeros(n)
launch(axpy_kernel, blocks, threads, x, b, y, 3.0, n)
print("блоков:", blocks, "| потоков:", blocks * threads,
      "| лишних:", blocks * threads - n)
print("y[:3] =", y[:3], "| y[999] =", y[999])
# Вывод в книге:
# | блоков: 4 | потоков: 1024 | лишних: 24
# | y[:3] = [10. 10. 10.] | y[999] = 10.0

# %% Листинг «Что делает драйвер при запуске ядра» — раздел «Компиляция ядра CUDA»
def load_kernel(app, gpu):
    if gpu in app["cubin"]:
        return f"{gpu}: есть cubin, запускаю сразу"
    if app["ptx"]:
        return f"{gpu}: cubin нет, собираю из PTX (JIT)"
    return f"{gpu}: ни cubin, ни PTX, ядро не запустится"

with_ptx = {"cubin": {"sm_86"}, "ptx": True}
no_ptx = {"cubin": {"sm_86"}, "ptx": False}
print(load_kernel(with_ptx, "sm_86"))
print(load_kernel(with_ptx, "sm_90"))
print(load_kernel(no_ptx, "sm_90"))
# Вывод в книге:
# | sm_86: есть cubin, запускаю сразу
# | sm_90: cubin нет, собираю из PTX (JIT)
# | sm_90: ни cubin, ни PTX, ядро не запустится

# %% Листинг «Операции одного прохода BookLM» — раздел «Библиотеки ядер»
# Проверка этот листинг не запускает: ему нужны сеть, видеокарта, сервер или внешний файл.
# import torch
# from torch.profiler import ProfilerActivity, profile
# from booklm.lab import load_model, val_tokens
# from booklm.experiments.ch15 import prefill, decode_step, top_ops
#
# model = load_model("base")                    # FP32, на процессоре
# ids = val_tokens(512)[None]                   # 512 токенов истории
# with torch.no_grad():
#     prefill(model, ids)                       # прогрев
#     with profile(activities=[ProfilerActivity.CPU]) as p_long:
#         model(ids)                            # обработка входа, T = 512
#     logits, caches = prefill(model, ids[:, :511])
#     nxt = logits[:, -1:].argmax(-1)
#     decode_step(model, nxt, [dict(c) for c in caches])   # прогрев шага
#     with profile(activities=[ProfilerActivity.CPU]) as p_step:
#         decode_step(model, nxt, caches)       # шаг генерации, T = 1
# for title, prof in (("вход, T = 512", p_long), ("шаг, T = 1", p_step)):
#     calls, total, rows = top_ops(prof)
#     print(f"{title}: вызовов {calls}, время {total / 1e3:.2f} мс")
#     for name, n, share in rows:
#         print(f"  {n:3d} раз {share:6.1%}  {name[6:]}")   # без "aten::"

# %% Листинг «Планировщик варпов и ожидание памяти» — раздел «Устройство потокового мультипроцессора»
def sm_busy(warps, work=10, latency=400, cycles=100_000):
    ready = [0] * warps              # такт, с которого варп готов
    t = busy = 0
    while t < cycles:
        w = min(range(warps), key=lambda g: ready[g])
        if ready[w] > t:             # все ждут память: SM простаивает
            t = ready[w]
            continue
        t += work                    # 10 тактов арифметики,
        busy += work
        ready[w] = t + latency       # потом новая загрузка
    return busy / t

for warps in (1, 10, 20, 40, 48):
    print(f"варпов {warps:2d}: SM занят {sm_busy(warps):4.0%}")
# Вывод в книге:
# | варпов  1: SM занят   2%
# | варпов 10: SM занят  24%
# | варпов 20: SM занят  49%
# | варпов 40: SM занят  98%
# | варпов 48: SM занят 100%

# %% Листинг «Предел блоков по регистрам и занятость» — раздел «Давление на регистры и занятость»
def blocks_per_sm(r, T=256, smem=0, R_SM=65536, SMEM_SM=164 * 1024,
                  max_warps=64, max_blocks=32):
    warps = T // 32                          # варпов в блоке
    B_regs = R_SM // (T * r)                 # предел по регистрам
    B_shared = SMEM_SM // smem if smem else max_blocks  # по общей памяти
    B_active = min(B_regs, B_shared, max_warps // warps, max_blocks)
    return B_regs, B_active, B_active * warps / max_warps

for r in (12, 32, 33, 64, 96):
    B_regs, B_active, occ = blocks_per_sm(r)
    print(f"r = {r:2d}: B_regs = {B_regs:2d}, блоков {B_active},",
          f"занятость {occ:4.0%}")
# Вывод в книге:
# | r = 12: B_regs = 21, блоков 8, занятость 100%
# | r = 32: B_regs =  8, блоков 8, занятость 100%
# | r = 33: B_regs =  7, блоков 7, занятость  88%
# | r = 64: B_regs =  4, блоков 4, занятость  50%
# | r = 96: B_regs =  2, блоков 2, занятость  25%

# %% Листинг «Занятость и загрузка SM вместе» — раздел «Давление на регистры и занятость»
for r, work in [(12, 10), (96, 10), (96, 40)]:
    warps = blocks_per_sm(r)[1] * 8          # 8 варпов в блоке
    print(f"r = {r:2d}, работа {work:2d} тактов: варпов {warps:2d},",
          f"SM занят {sm_busy(warps, work=work):4.0%}")
# Вывод в книге:
# | r = 12, работа 10 тактов: варпов 64, SM занят 100%
# | r = 96, работа 10 тактов: варпов 16, SM занят  39%
# | r = 96, работа 40 тактов: варпов 16, SM занят 100%

# %% Листинг «Транзакции памяти и проходы банков» — раздел «Объединённый доступ и банки памяти»
def transactions(addresses, segment=32):  # задетые 32-байтные участки
    return len({a // segment for a in addresses})

lanes = range(32)
for name, start, step in [("подряд", 0, 1), ("сдвиг на 4 Б", 4, 1),
                          ("шаг 2", 0, 2), ("шаг 16", 0, 16)]:
    t = transactions([start + 4 * step * lane for lane in lanes])
    print(f"{name:12s}: участков {t:2d}, байт {32 * t:4d}")

def bank_passes(words, banks=32):         # проходы загруженного банка
    per_bank = [0] * banks
    for w in set(words):
        per_bank[w % banks] += 1
    return max(per_bank)

for width in (32, 33):
    column = [row * width for row in lanes]   # столбец 0 плитки
    print(f"строка {width} слова: проходов {bank_passes(column):2d}")
# Вывод в книге:
# | подряд      : участков  4, байт  128
# | сдвиг на 4 Б: участков  5, байт  160
# | шаг 2       : участков  8, байт  256
# | шаг 16      : участков 32, байт 1024
# | строка 32 слова: проходов 32
# | строка 33 слова: проходов  1

# %% Листинг «Наивное и плиточное умножение матриц» — раздел «Плиточное умножение матриц»
def matmul_naive(X, W):
    Y, reads = np.zeros((X.shape[0], W.shape[1])), 0
    for i in range(X.shape[0]):
        for j in range(W.shape[1]):      # поток (i, j) сам читает
            Y[i, j] = X[i, :] @ W[:, j]  # строку X и столбец W
            reads += X.shape[1] + W.shape[0]
    return Y, reads

def matmul_tiled(X, W, tile=32):
    assert X.shape[0] % tile == X.shape[1] % tile == W.shape[1] % tile == 0, \
        "размеры должны делиться на tile"
    Y, reads = np.zeros((X.shape[0], W.shape[1])), 0
    for i0 in range(0, X.shape[0], tile):
        for j0 in range(0, W.shape[1], tile):    # один блок потоков
            acc = np.zeros((tile, tile))         # суммы в регистрах
            for k0 in range(0, X.shape[1], tile):
                Xs = X[i0:i0 + tile, k0:k0 + tile]   # плитки в
                Ws = W[k0:k0 + tile, j0:j0 + tile]   # общую память
                reads += Xs.size + Ws.size
                acc += Xs @ Ws                   # по 32 FMA на поток
            Y[i0:i0 + tile, j0:j0 + tile] = acc  # одно сохранение
    return Y, reads

rng = np.random.default_rng(0)
X = rng.standard_normal((32, 256))
W = rng.standard_normal((256, 32))
Y1, naive = matmul_naive(X, W)
Y2, tiled = matmul_tiled(X, W)
print("запрошено чисел: наивно", naive, "| плитками", tiled,
      "| в", naive // tiled, "раза меньше")
print("ответы совпадают с X @ W:",
      np.allclose(Y1, X @ W) and np.allclose(Y2, X @ W))
# Вывод в книге:
# | запрошено чисел: наивно 524288 | плитками 16384 | в 32 раза меньше
# | ответы совпадают с X @ W: True

# %% Листинг «Цена большой плитки» — раздел «Плиточное умножение матриц»
print("тип     t  чисел/элемент   КБ  блоков/SM  занятость  в сетке")
for nbytes, name in ((4, "FP32"), (2, "BF16")):
    for t in (16, 32, 64, 128):
        smem = 2 * t * t * nbytes                  # плитки X и W
        _, B, occ = blocks_per_sm(r=32, smem=smem)
        grid = (512 // t) * (768 // t)             # блоков на слой BookLM
        print(f"{name} {t:4d} {2 * 256 // t:14d} {smem // 1024:4d}"
              f" {B:10d} {occ:10.0%} {grid:8d}")
# Вывод в книге:
# | тип     t  чисел/элемент   КБ  блоков/SM  занятость  в сетке
# | FP32   16             32    2          8       100%     1536
# | FP32   32             16    8          8       100%      384
# | FP32   64              8   32          5        62%       96
# | FP32  128              4  128          1        12%       24
# | BF16   16             32    1          8       100%     1536
# | BF16   32             16    4          8       100%      384
# | BF16   64              8   16          8       100%       96
# | BF16  128              4   64          2        25%       24

# %% Листинг «Плитка FlashAttention для BookLM» — раздел «Плиточное умножение матриц»
B_regs, B, occ = blocks_per_sm(r=64, T=128, smem=3 * 64 * 32 * 2)
tiles = (512 // 64) * 8                 # плитки Q: 512 токенов, 8 голов
print(f"блоков на SM: по регистрам {B_regs}, всего {B},",
      f"занятость {occ:.0%}")
print(f"блоков в сетке для одного запроса: {tiles}")
# Вывод в книге:
# | блоков на SM: по регистрам 8, всего 8, занятость 50%
# | блоков в сетке для одного запроса: 64

# %% Листинг «Модель крыши для матрицы BookLM 256768» — раздел «Модель крыши»
d, f = 256, 768                # W: [d, f]
BW, P = 1e12, 100e12           # байт/с и операций/с

def intensity(tokens, w_bytes=2):
    flops = 2 * tokens * d * f                      # FMA = 2 операции
    nbytes = w_bytes * d * f                        # веса W
    nbytes += 2 * (tokens * d + tokens * f)         # X и Y в BF16
    return flops / nbytes

def attainable(I):
    return min(P, I * BW)

for tokens in (1, 64, 512):
    I = intensity(tokens)
    print(f"T = {tokens:3d}: I = {I:6.2f} оп/Б,"
          f" не быстрее {attainable(I) / 1e12:5.1f} Тфлоп/с")
I = intensity(1, w_bytes=0.5)                       # веса INT4
print(f"T =   1, веса INT4: I = {I:.2f} оп/Б,"
      f" не быстрее {attainable(I) / 1e12:.1f} Тфлоп/с")
print(f"перелом крыши: I* = P / BW = {P / BW:.0f}")
# Вывод в книге:
# | T =   1: I =   0.99 оп/Б, не быстрее   1.0 Тфлоп/с
# | T =  64: I =  48.00 оп/Б, не быстрее  48.0 Тфлоп/с
# | T = 512: I = 139.64 оп/Б, не быстрее 100.0 Тфлоп/с
# | T =   1, веса INT4: I = 3.92 оп/Б, не быстрее 3.9 Тфлоп/с
# | перелом крыши: I* = P / BW = 100

# %% Листинг «Перевод корпуса на GB10 по модели крыши» — раздел «Характеристики ускорителя»
N, P, BW = 8e9, 100e12, 273e9        # модель; GB10: BF16 и полоса
kv_token = 36 * 2 * 8 * 128 * 2      # слои, K и V, KV-головы, d_head, 2 Б
inp, out, batch = 430, 330, 256      # токенов на историю; пакет
ctx = inp + out / 2                  # средний контекст при генерации

step_bytes = 2 * N + batch * ctx * kv_token   # веса и кэш всех историй
I = batch * 2 * N / step_bytes                # операций на байт шага
t_in = inp * 2 * N / P                        # вход упирается в P
t_out = out * step_bytes / BW / batch         # генерация упирается в BW
print(f"шаг генерации: {step_bytes / 1e9:.0f} ГБ,"
      f" I = {I:.0f} при I* = {P / BW:.0f}")
print(f"на историю: вход {t_in:.3f} с, генерация {t_out:.3f} с")
print(f"не быстрее {1 / (t_in + t_out):.1f} историй/с,"
      f" {(inp + out) * 2 * N / (t_in + t_out) / 1e12:.0f} Тфлоп/с")
# Вывод в книге:
# | шаг генерации: 38 ГБ, I = 106 при I* = 366
# | на историю: вход 0.069 с, генерация 0.182 с
# | не быстрее 4.0 историй/с, 49 Тфлоп/с

# %% Листинг «BookLM на крыше процессора» — раздел «Характеристики ускорителя»
# Проверка этот листинг не запускает: ему нужны сеть, видеокарта, сервер или внешний файл.
# from booklm.lab import load_model, val_tokens
# from booklm.experiments.ch15 import (N_PARAMS, best_time, decode_200,
#     kv_bytes, measure_bw, measure_peak, prefill)
#
# BW, P = measure_bw(), measure_peak()          # копирование, matmul
# print(f"крыша CPU: BW = {BW / 1e9:.0f} ГБ/с, P = {P / 1e12:.2f} Тфлоп/с,"
#       f" I* = {P / BW:.0f} оп/Б")
# model = load_model("base")                    # FP32: веса 26,5 МБ
# data = val_tokens(64 * 512).view(64, 512)
# rows = []
# for B in (1, 8, 64):
#     ids = data[:B]
#     t = best_time(lambda: prefill(model, ids), rounds=3)
#     rows.append((f"вход, B={B}", t, 2 * N_PARAMS * B * 512,
#                  4 * N_PARAMS + kv_bytes(512, B)))
#     t = min(decode_200(model, ids) for _ in range(3))   # 200 шагов
#     rows.append((f"шаг, B={B}", t, 2 * N_PARAMS * B,
#                  4 * N_PARAMS + kv_bytes(412, B)))
# for name, t, F, D in rows:
#     I, speed = F / D, F / t                   # оп/Б и операций в секунду
#     print(f"{name:<10}{t * 1e3:8.2f} мс  I = {I:6.1f}  {speed / 1e12:5.3f}"
#           f" Тфлоп/с  {speed / min(P, I * BW):4.0%} крыши")

# %% Листинг «Два соглашения для кодов INT4» — раздел «Матричная инструкция и смешанная точность»
lo, hi = 3, -2                          # два кода INT4
twos = (lo % 16) | ((hi % 16) << 4)     # дополнительный код: -2 -> 14
shift = (lo + 8) | ((hi + 8) << 4)      # сдвиг на 8, как в главе 14
print("дополнительный код:", twos, f"{twos:08b}")
print("сдвиг на 8:        ", shift, f"{shift:08b}")

def read_twos(byte):                    # правило этой главы
    return [c - 16 if c > 7 else c for c in (byte & 15, byte >> 4)]

def read_shift(byte):                   # правило главы 14
    return [(byte & 15) - 8, (byte >> 4) - 8]

print("байт 227 по-нашему:", read_twos(227),
      "| по правилу главы 14:", read_shift(227))
# Вывод в книге:
# | дополнительный код: 227 11100011
# | сдвиг на 8:         107 01101011
# | байт 227 по-нашему: [3, -2] | по правилу главы 14: [-5, 6]

# %% Листинг «Путь веса W4A16: упаковка, распаковка, накопитель» — раздел «Матричная инструкция и смешанная точность»
import torch
torch.manual_seed(0)
W = torch.randn(256, 768).half()          # матрица слоя в FP16
scale = W.abs().amax(dim=0) / 7           # масштаб на столбец
q = torch.round(W / scale).clamp(-8, 7).short()   # коды -8..7
u = q % 16                                # те же коды как 0..15
packed = (u[0::2] | (u[1::2] << 4)).to(torch.uint8)  # 2 кода в байте
print("в памяти:     ", packed.dtype, tuple(packed.shape),
      packed.numel(), "байт")

lo, hi = (packed & 15).short(), (packed >> 4).short()
codes = torch.stack([lo, hi], dim=1).reshape(256, 768)
codes = torch.where(codes > 7, codes - 16, codes)   # снова -8..7
W16 = codes.half() * scale                # распаковка в FP16
print("в инструкцию: ", W16.dtype, tuple(W16.shape),
      W16.numel() * 2, "байт")

x = torch.randn(1, 256).half()            # активации FP16
y = x.float() @ W16.float()               # накопитель FP32
ref = x.float() @ W.float()
print("накопитель:   ", y.dtype, tuple(y.shape))
err = ((y - ref).norm() / ref.norm()).item()
print(f"относительная ошибка от INT4: {err:.2f}")
# Вывод в книге:
# | в памяти:      torch.uint8 (128, 768) 98304 байт
# | в инструкцию:  torch.float16 (256, 768) 393216 байт
# | накопитель:    torch.float32 (1, 768)
# | относительная ошибка от INT4: 0.13

# %% Листинг «Размер весов BookLM в INT4» — раздел «Матричная инструкция и смешанная точность»
import torch
from booklm.lab import load_model
from booklm.experiments.ch15 import pack_int4, unpack_int4

model = load_model("base")
linears = [m.weight.data for m in model.modules()
           if isinstance(m, torch.nn.Linear)]
n = sum(w.numel() for w in linears)
packs = [pack_int4(w) for w in linears]             # группы по 64 веса
n_bytes = sum(p.numel() + s.numel() * 2 for p, s in packs)  # масштабы FP16
print(f"{len(linears)} линейных слоя, {n:_} весов".replace("_", " "))
print(f"BF16: {2 * n / 2**20:.2f} МБ, INT4 с масштабами: "
      f"{n_bytes / 2**20:.2f} МБ, в {2 * n / n_bytes:.2f} раза меньше")
W = model.blocks[0].ffn.gate.weight.data            # [768, 256], FP32
packed, scale = pack_int4(W)
W_q = unpack_int4(packed, scale)                    # распаковано заранее
print(f"ошибка весов gate: {((W_q - W).norm() / W.norm()).item():.3f}")
# Вывод в книге:
# | 42 линейных слоя, 4 521 984 весов
# | BF16: 8.62 МБ, INT4 с масштабами: 2.29 МБ, в 3.76 раза меньше
# | ошибка весов gate: 0.107

# %% Листинг «Время умножения с весами INT4» — раздел «Матричная инструкция и смешанная точность»
# Проверка этот листинг не запускает: ему нужны сеть, видеокарта, сервер или внешний файл.
# import torch.nn.functional as F
# from booklm.experiments.ch15 import time_us
#
# x = torch.randn(64, 256)                            # пакет активаций
# x16, W16 = x.bfloat16(), W.bfloat16()
# variants = [("FP32", lambda: F.linear(x, W)),
#             ("BF16", lambda: F.linear(x16, W16)),
#             ("INT4, распаковка на каждом вызове",
#              lambda: F.linear(x, unpack_int4(packed, scale))),
#             ("INT4, распакован заранее", lambda: F.linear(x, W_q))]
# times = [(name, time_us(fn)) for name, fn in variants]   # мкс на вызов
# for name, t in times:
#     print(f"{name:<34}{t:7.1f} мкс  x{t / times[0][1]:.2f}")

# %% Листинг «Модель на 70 млрд параметров на крыше B200» — раздел «Четырёхбитная арифметика Blackwell»
BW, N = 7.7e12, 70e9                       # B200: байт/с; параметры
formats = [("BF16", 2, 2250e12),           # имя, байт на вес, пик P
           ("NVFP4 W4A16", 4.5 / 8, 2250e12),
           ("NVFP4 W4A4", 4.5 / 8, 9000e12)]
print(f"{'формат':<12}{'веса, ГБ':>9}{'ген., ток/с':>12}"
      f"{'вход, ток/с':>12}{'I*':>6}{'пакет':>7}")
for name, w_bytes, P in formats:
    gen = BW / (N * w_bytes)               # шаг читает все веса
    pre = P / (2 * N)                      # вход упирается в P
    knee = P / BW * w_bytes / 2            # пакет, при котором I = I*
    print(f"{name:<12}{N * w_bytes / 1e9:9.0f}{gen:12.0f}{pre:12.0f}"
          f"{P / BW:6.0f}{knee:7.0f}")
# Вывод в книге:
# | формат       веса, ГБ ген., ток/с вход, ток/с    I*  пакет
# | BF16              140          55       16071   292    292
# | NVFP4 W4A16        39         196       16071   292     82
# | NVFP4 W4A4         39         196       64286  1169    329

# %% Листинг «BookLM в W4A4» — раздел «Четырёхбитная арифметика Blackwell»
from booklm.experiments.ch14 import story_perplexity
from booklm.experiments.ch15_w4a4 import VARIANTS, quantized_booklm, spikes

for name in VARIANTS:                  # BF16, FP8, NVFP4 и MXFP4
    model = quantized_booklm(name)     # вход слоя округляется на лету
    print(f"{name:<12} перплексия {story_perplexity(model):.2f}")
act, weight = spikes(quantized_booklm("BF16"))
print(f"наибольшее/типичное: входы слоёв {act:.0f}, веса {weight:.0f}")
# Вывод в книге:
# | BF16         перплексия 6.66
# | FP8 W8A8     перплексия 6.69
# | NVFP4 W4A16  перплексия 6.88
# | MXFP4 W4A16  перплексия 6.95
# | NVFP4 W4A4   перплексия 7.14
# | MXFP4 W4A4   перплексия 7.49
# | наибольшее/типичное: входы слоёв 125, веса 13

# %% Листинг «Джоули на токен по паспортной мощности» — раздел «Энергия на операцию и на токен»
GIB = 2**30

def joules_per_token(watts, seconds, tokens):
    return watts * seconds / tokens           # Дж = Вт * с

BW, power = 3.35e12, 700                      # H100: байт/с и TDP, Вт
weights, kv = 4.56 * GIB, 1 * GIB             # 8 млрд в Q4_K_M; кэш диалога
cases = [("шаг, 1 диалог", (weights + kv) / BW, 1),
         ("шаг, 16 диалогов", (weights + 16 * kv) / BW, 16),
         ("вход, 8192 токена", 0.27, 8192)]   # оценка главы 14
for name, t, n in cases:
    j = joules_per_token(power, t, n)
    print(f"{name:<18}{t * 1e3:7.2f} мс {j:7.3f} Дж/токен"
          f" {3.6e6 / j / 1e6:6.1f} млн токенов на кВт*ч")
j1, j16 = (joules_per_token(power, t, n) for _, t, n in cases[:2])
assert 4.3 < j1 / j16 < 4.4
print(f"пакет из 16 дешевле на токен в {j1 / j16:.1f} раза")
print(f"H100 на пределе: {power / 989e12 * 1e12:.2f} пДж на операцию,"
      f" {power / BW * 1e12:.0f} пДж на байт")
# Вывод в книге:
# | шаг, 1 диалог        1.78 мс   1.247 Дж/токен    2.9 млн токенов на кВт*ч
# | шаг, 16 диалогов     6.59 мс   0.288 Дж/токен   12.5 млн токенов на кВт*ч
# | вход, 8192 токена  270.00 мс   0.023 Дж/токен  156.0 млн токенов на кВт*ч
# | пакет из 16 дешевле на токен в 4.3 раза
# | H100 на пределе: 0.71 пДж на операцию, 209 пДж на байт

# %% Листинг «Энергия обучения по GPU-часам» — раздел «Энергия обучения»
def training_kwh(gpu_hours, watts, pue=1.0):
    return gpu_hours * watts / 1e3 * pue      # кВт*ч

runs = [("Llama 3.1, три модели", 39.3e6, 700, 1.1),  # H100; PUE взят нами
        ("Llama 3.1 405B", 30.84e6, 700, 1.1),
        ("BLOOM 176B", 1_082_990, 400, 1.2)]         # A100; PUE из статьи
for name, hours, w, pue in runs:
    e = training_kwh(hours, w) / 1e3                 # МВт*ч
    print(f"{name:<22}{hours / 1e6:6.2f} млн GPU-ч {e:7.0f} МВт*ч,"
          f" с PUE {pue}: {e * pue:6.0f}")
need = 3.8e25 / 400e12 / 3600                     # C / (флоп/с на GPU)
pj = training_kwh(30.84e6, 700) * 3.6e6 / 3.8e25 * 1e12
assert 26e6 < need < 27e6 and 1.9 < pj < 2.1
print(f"405B по бюджету C: {need / 1e6:.1f} млн GPU-ч из 30.84;"
      f" {pj:.2f} пДж на полезную операцию")
booklm = training_kwh(25.5 / 60, 240)             # GB10: весь DGX Spark
translate = training_kwh(68_958 / 3 / 3600, 240)  # перевод корпуса
print(f"BookLM: обучение до {booklm:.2f} кВт*ч,"
      f" перевод корпуса около {translate:.1f} кВт*ч")
# Вывод в книге:
# | Llama 3.1, три модели  39.30 млн GPU-ч   27510 МВт*ч, с PUE 1.1:  30261
# | Llama 3.1 405B         30.84 млн GPU-ч   21588 МВт*ч, с PUE 1.1:  23747
# | BLOOM 176B              1.08 млн GPU-ч     433 МВт*ч, с PUE 1.2:    520
# | 405B по бюджету C: 26.4 млн GPU-ч из 30.84; 2.05 пДж на полезную операцию
# | BookLM: обучение до 0.10 кВт*ч, перевод корпуса около 1.5 кВт*ч
