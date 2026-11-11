"""Маленькая модель GPU из главы 15 «Устройство GPU» — в одном файле и с проверками.

Функции совпадают с листингами главы: машина load/store, запуск ядра axpy сеткой блоков,
модель планировщика варпов, предел блоков по регистрам и общей памяти, занятость, транзакции памяти и банки
общей памяти, наивное и плиточное умножение со счётчиком чтений, модель крыши и путь веса W4A16.
Видеокарта не нужна.

Запуск из корня репозитория: python labs/ch15/gpu_by_hand.py
"""
from __future__ import annotations

import numpy as np
import torch


# --- 1. Машина с загрузкой и сохранением ------------------------------------------------
def run_program(memory: dict, program: list) -> list[dict]:
    """Выполняет инструкции ld/fma/st и возвращает состояние регистров после каждой."""
    regs: dict = {}
    trace = []
    for op, *args in program:
        if op == "ld":
            r, addr = args
            regs[r] = memory[addr]
        elif op == "fma":
            d, a, b, c = args
            regs[d] = regs[a] * regs[b] + regs[c]
        elif op == "st":
            addr, r = args
            memory[addr] = regs[r]
        else:
            raise ValueError(op)
        trace.append(dict(regs))
    return trace


# --- 2. Запуск ядра сеткой блоков -------------------------------------------------------
def axpy_kernel(blockIdx, blockDim, threadIdx, x, b, y, a, n):
    i = blockIdx * blockDim + threadIdx
    if i < n:
        y[i] = a * x[i] + b[i]


def launch(kernel, blocks, threads, *args):
    for block in range(blocks):
        for thread in range(threads):
            kernel(block, threads, thread, *args)


def grid_size(n: int, threads: int) -> int:
    return (n + threads - 1) // threads


# --- 3. Планировщик групп ---------------------------------------------------------------
def sm_busy(warps, work=10, latency=400, cycles=100_000):
    ready = [0] * warps
    t = busy = 0
    while t < cycles:
        w = min(range(warps), key=lambda g: ready[g])
        if ready[w] > t:
            t = ready[w]
            continue
        t += work
        busy += work
        ready[w] = t + latency
    return busy / t


# --- 4. Регистры и занятость ------------------------------------------------------------
def blocks_per_sm(r, T=256, smem=0, R_SM=65536, SMEM_SM=164 * 1024, max_warps=64, max_blocks=32):
    warps = T // 32                                       # варпов в блоке
    B_regs = R_SM // (T * r)                              # предел по регистрам
    B_shared = SMEM_SM // smem if smem else max_blocks    # предел по общей памяти
    B_active = min(B_regs, B_shared, max_warps // warps, max_blocks)
    return B_regs, B_active, B_active * warps / max_warps


# --- 5. Транзакции и банки --------------------------------------------------------------
def transactions(addresses, segment=32):
    return len({a // segment for a in addresses})


def bank_passes(words, banks=32):
    per_bank = [0] * banks
    for w in set(words):
        per_bank[w % banks] += 1
    return max(per_bank)


# --- 6. Наивное и плиточное умножение ---------------------------------------------------
def matmul_naive(X, W):
    Y, reads = np.zeros((X.shape[0], W.shape[1])), 0
    for i in range(X.shape[0]):
        for j in range(W.shape[1]):
            Y[i, j] = X[i, :] @ W[:, j]
            reads += X.shape[1] + W.shape[0]
    return Y, reads


def matmul_tiled(X, W, tile=32):
    Y, reads = np.zeros((X.shape[0], W.shape[1])), 0
    for i0 in range(0, X.shape[0], tile):
        for j0 in range(0, W.shape[1], tile):
            acc = np.zeros((min(tile, X.shape[0] - i0), min(tile, W.shape[1] - j0)))
            for k0 in range(0, X.shape[1], tile):
                Xs = X[i0:i0 + tile, k0:k0 + tile]
                Ws = W[k0:k0 + tile, j0:j0 + tile]
                reads += Xs.size + Ws.size
                acc += Xs @ Ws
            Y[i0:i0 + tile, j0:j0 + tile] = acc
    return Y, reads


# --- 7. Модель крыши --------------------------------------------------------------------
def intensity(tokens, w_bytes=2, d=256, f=768):
    flops = 2 * tokens * d * f
    nbytes = w_bytes * d * f + 2 * (tokens * d + tokens * f)   # веса W; X и Y в BF16
    return flops / nbytes


def attainable(I, BW=1e12, P=100e12):
    return min(P, I * BW)


# --- 8. Путь веса W4A16 -----------------------------------------------------------------
def pack_int4(W: torch.Tensor, group: int | None = None):
    """Коды INT4 по два в байте (чётная строка — младшие биты) и масштабы.

    group=None — один масштаб на столбец, иначе на группу из `group` строк.
    Коды хранятся в дополнительном коде (q % 16: -2 -> 14). pack_int4 главы 14 сдвигает коды на +8
    (-2 -> 6); байты одной функции, прочитанные другой, дают другие веса без ошибки,
    поэтому распаковывать только парной unpack_int4 из этого файла."""
    rows, cols = W.shape
    g = rows if group is None else group
    Wg = W.float().reshape(rows // g, g, cols)
    scale = Wg.abs().amax(dim=1, keepdim=True) / 7
    q = torch.clamp(torch.round(Wg / scale), -8, 7).reshape(rows, cols).short()
    u = q % 16
    packed = (u[0::2] | (u[1::2] << 4)).to(torch.uint8)
    return packed, scale.half()


def unpack_int4(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    lo, hi = (packed & 15).short(), (packed >> 4).short()
    rows, cols = packed.shape[0] * 2, packed.shape[1]
    codes = torch.stack([lo, hi], dim=1).reshape(rows, cols)
    codes = torch.where(codes > 7, codes - 16, codes)
    g = rows // scale.shape[0]
    return (codes.reshape(scale.shape[0], g, cols).half() * scale).reshape(rows, cols)


# --- проверки ---------------------------------------------------------------------------
def main() -> None:
    trace = run_program({"x_i": 2.0, "a": 3.0, "b_i": 4.0, "y_i": 0.0},
                        [("ld", "r0", "x_i"), ("ld", "r1", "a"), ("ld", "r2", "b_i"),
                         ("fma", "r3", "r1", "r0", "r2"), ("st", "y_i", "r3")])
    assert trace[3]["r3"] == 10.0 and len(trace) == 5
    mem = {"x_i": 1.5, "a": -2.0, "b_i": 0.25, "y_i": 0.0}
    run_program(mem, [("ld", "r0", "x_i"), ("ld", "r1", "a"), ("ld", "r2", "b_i"),
                      ("fma", "r3", "r1", "r0", "r2"), ("st", "y_i", "r3")])
    assert mem["y_i"] == -2.75

    n, threads = 1000, 256
    blocks = grid_size(n, threads)
    x, b, y = np.full(n, 2.0), np.full(n, 4.0), np.zeros(n)
    launch(axpy_kernel, blocks, threads, x, b, y, 3.0, n)
    assert blocks == 4 and blocks * threads - n == 24 and np.all(y == 10.0)
    assert grid_size(1_000_000, 256) == 3907

    assert round(sm_busy(1), 2) == 0.02 and sm_busy(48) == 1.0 and sm_busy(40) > 0.97

    assert blocks_per_sm(10) == (25, 8, 1.0)
    assert blocks_per_sm(32) == (8, 8, 1.0)
    assert blocks_per_sm(64) == (4, 4, 0.5)
    assert blocks_per_sm(96) == (2, 2, 0.25)
    assert blocks_per_sm(32, smem=2 * 128 * 128 * 4) == (8, 1, 0.125)    # плитка 128 в FP32
    assert blocks_per_sm(64, T=128, smem=3 * 64 * 32 * 2) == (8, 8, 0.5)  # плитка FlashAttention BookLM

    lanes = range(32)
    assert transactions([4 * l for l in lanes]) == 4
    assert transactions([4 + 4 * l for l in lanes]) == 5
    assert transactions([64 * l for l in lanes]) == 32
    assert bank_passes([32 * r for r in lanes]) == 32
    assert bank_passes([33 * r for r in lanes]) == 1

    rng = np.random.default_rng(0)
    X, W = rng.standard_normal((32, 256)), rng.standard_normal((256, 32))
    Y1, naive = matmul_naive(X, W)
    Y2, tiled = matmul_tiled(X, W)
    assert naive == 524_288 and tiled == 16_384
    assert np.allclose(Y1, X @ W) and np.allclose(Y2, X @ W)

    I_dec, I_pre = intensity(1), intensity(512)
    assert 0.99 < I_dec < 1.0 and 139 < I_pre < 140
    assert attainable(I_dec) < 100e12 / 50 and attainable(I_pre) == 100e12
    assert 3.9 < intensity(1, w_bytes=0.5) < 4.0                          # веса INT4 при генерации

    torch.manual_seed(0)
    Wt = torch.randn(256, 768).half()
    packed, scale = pack_int4(Wt)
    assert packed.dtype == torch.uint8 and packed.numel() == 98_304
    W16 = unpack_int4(packed, scale)
    assert W16.dtype == torch.float16 and W16.shape == (256, 768)
    xt = torch.randn(1, 256).half()
    ref = xt.float() @ Wt.float()
    err = lambda Wq: ((xt.float() @ Wq.float() - ref).norm() / ref.norm()).item()
    e_col = err(W16)
    e_g64 = err(unpack_int4(*pack_int4(Wt, group=64)))
    assert 0.10 < e_col < 0.16 and e_g64 < e_col
    print("все проверки главы 15 прошли")


if __name__ == "__main__":
    main()
