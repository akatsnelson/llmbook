"""Двойное квантование и байты основы BookLM в QLoRA.

Полная версия листинга «Байты основы BookLM в QLoRA» из главы 8, подраздел 8.3.3 «Двойное квантование».
Выполняется в ноутбуке главы 08_lora_sft.ipynb и отдельно.
Запуск из корня репозитория: python labs/ch08/qlora_double_quant.py
"""
import math

import torch

# --- блочное квантование NF4 из qlora_nf4.py --------------------------------
NF4 = torch.tensor([
    -1.0, -0.6961928, -0.5250731, -0.3949175, -0.2844414, -0.1847734,
    -0.0910500, 0.0, 0.0795803, 0.1609302, 0.2461123, 0.3379152,
    0.4407098, 0.5626170, 0.7229568, 1.0])


def quantize_nf4(w, block=64, table=NF4):
    blocks = w.reshape(-1, block)
    absmax = blocks.abs().amax(dim=1)
    normed = blocks / absmax[:, None]
    codes = (normed[..., None] - table).abs().argmin(dim=-1).to(torch.uint8)
    return codes, absmax


def dequantize_nf4(codes, absmax, shape, table=NF4):
    return (table[codes.long()] * absmax[:, None]).reshape(shape)


def rel_rmse(approx, exact):
    return ((approx - exact).pow(2).mean().sqrt() / exact.pow(2).mean().sqrt()).item()


torch.manual_seed(0)
W = torch.randn(256, 768) * 0.02                  # форма W_down BookLM
codes, absmax = quantize_nf4(W)
W_nf4 = dequantize_nf4(codes, absmax, W.shape)


# --- двойное квантование -----------------------------------------------------
def double_quant(absmax, block=256):
    groups = absmax.reshape(-1, block)                    # [n_groups, 256]
    c = groups.amax(dim=1)                                 # FP32: одно число на группу
    q8 = torch.round(groups / c[:, None] * 255).to(torch.uint8)
    return q8, c


def double_dequant(q8, c):
    return (q8.float() / 255 * c[:, None]).reshape(-1)


# Масштабы матрицы W: 3072 блока = 12 групп по 256
q8, c = double_quant(absmax)
absmax_dq = double_dequant(q8, c)
assert (absmax_dq - absmax).abs().max() <= c.max() / 510 + 1e-7
W_dq = dequantize_nf4(codes, absmax_dq, W.shape)
print(f"ошибка: один уровень {rel_rmse(W_nf4, W):.4f}, двойное {rel_rmse(W_dq, W):.4f}")
assert rel_rmse(W_dq, W) < 1.02 * rel_rmse(W_nf4, W)


def bits_per_weight(block=64, block2=256, double=True):
    if not double:
        return 4 + 32 / block
    return 4 + 8 / block + 32 / (block * block2)


assert bits_per_weight(double=False) == 4.5
assert bits_per_weight() == 4.126953125

# Формы семи матриц одного блока BookLM: (d_out, d_in), как у nn.Linear
SHAPES = {"q": (256, 256), "k": (64, 256), "v": (64, 256), "o": (256, 256),
          "gate": (768, 256), "up": (768, 256), "down": (256, 768)}


def qlora_base_bytes(n_layers=6, V=8192, d=256, block=64, block2=256):
    n_lin = n_blocks = n_groups = 0
    for d_out, d_in in SHAPES.values():
        blocks = d_out * d_in // block
        n_lin += n_layers * d_out * d_in
        n_blocks += n_layers * blocks
        n_groups += n_layers * math.ceil(blocks / block2)
    return {
        "коды NF4": n_lin // 2,                  # два 4-битных кода в байте
        "масштабы, 8 бит": n_blocks,
        "масштабы масштабов, FP32": 4 * n_groups,
        "эмбеддинги, BF16": V * d * 2,
        "RMSNorm, BF16": (2 * n_layers + 1) * d * 2,
    }


parts = qlora_base_bytes()
assert parts == {"коды NF4": 2_260_992, "масштабы, 8 бит": 70_656,
                 "масштабы масштабов, FP32": 1_104,
                 "эмбеддинги, BF16": 4_194_304, "RMSNorm, BF16": 6_656}
total = sum(parts.values())
assert total == 6_533_712
print(f"основа: {total} Б = {total / 2**20:.2f} МБ, "
      f"в {13_244_928 / total:.2f} раза меньше BF16")
assert total + 16 * 228_352 == 10_187_344     # основа + адаптер с AdamW
