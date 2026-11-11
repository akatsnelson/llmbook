"""Уровни NF4 из квантилей и квантование блоками.

Полная версия листинга «Квантование блоками в NF4» из главы 8, подраздел 8.3.2 «Сетка NF4»: таблица NF4
строится из квантилей нормального распределения и сверяется с таблицей bitsandbytes, затем
квантуются ручной блок, блок с выбросом и матрица формы W_down BookLM.
Выполняется в ноутбуке главы 08_lora_sft.ipynb и отдельно.
Запуск из корня репозитория: python labs/ch08/qlora_nf4.py
"""
import torch

# 16 уровней NF4 в том виде, в каком их хранит bitsandbytes
NF4 = torch.tensor([
    -1.0, -0.6961928, -0.5250731, -0.3949175, -0.2844414, -0.1847734,
    -0.0910500, 0.0, 0.0795803, 0.1609302, 0.2461123, 0.3379152,
    0.4407098, 0.5626170, 0.7229568, 1.0])

# Те же уровни из квантилей нормального распределения: 8 справа, 7 слева и ноль.
# В точке 1 квантиль бесконечен, поэтому крайний уровень берут в точке edge:
# среднее середин последнего из 16 и последнего из 15 равных отрезков [0, 1].
edge = (1 - 1 / 30 + 1 - 1 / 32) / 2   # 0.9677083
right = torch.special.ndtri(torch.linspace(edge, 0.5, 9, dtype=torch.float64)[:-1])
left = -torch.special.ndtri(torch.linspace(edge, 0.5, 8, dtype=torch.float64)[:-1])
levels = torch.cat([left, torch.zeros(1, dtype=torch.float64), right]).sort().values
levels = (levels / levels.abs().max()).float()
assert torch.allclose(levels, NF4, atol=1e-6)


def quantize_nf4(w, block=64, table=NF4):
    blocks = w.reshape(-1, block)                         # [n_blocks, block]
    absmax = blocks.abs().amax(dim=1)                     # [n_blocks], FP32
    normed = blocks / absmax[:, None]                     # в [-1, 1]
    codes = (normed[..., None] - table).abs().argmin(dim=-1).to(torch.uint8)
    return codes, absmax                                  # коды 0..15 и масштабы


def dequantize_nf4(codes, absmax, shape, table=NF4):
    return (table[codes.long()] * absmax[:, None]).reshape(shape)


# Ручной пример: блок из четырёх весов
w = torch.tensor([0.12, -0.40, 0.03, 0.80])
codes, absmax = quantize_nf4(w, block=4)
w_hat = dequantize_nf4(codes, absmax, w.shape)
assert codes.tolist() == [[9, 2, 7, 15]]
assert torch.allclose(w_hat, torch.tensor([0.1287, -0.4201, 0.0, 0.8]), atol=1e-4)
assert w_hat[2] == 0.0 and w_hat[3] == w[3]      # ноль и absmax восстановлены точно

# Выброс 8.0 в том же блоке
codes_out, absmax_out = quantize_nf4(torch.tensor([0.12, -0.40, 0.03, 8.0]), block=4)
assert codes_out.tolist() == [[7, 6, 7, 15]]
print("с выбросом:", codes_out.tolist(),
      dequantize_nf4(codes_out, absmax_out, (4,)).tolist())

# Граница ошибки на матрице формы W_down BookLM [d_out, d_in] = [256, 768]
torch.manual_seed(0)
W = torch.randn(256, 768) * 0.02
codes, absmax = quantize_nf4(W)
W_nf4 = dequantize_nf4(codes, absmax, W.shape)
max_gap = (NF4[1:] - NF4[:-1]).max()                     # 0.3038
err = (W_nf4 - W).reshape(-1, 64).abs()
assert (err <= absmax[:, None] * max_gap / 2 + 1e-7).all()


# NF4 против равномерной сетки из 15 уровней на той же матрице
def rel_rmse(approx, exact):
    return ((approx - exact).pow(2).mean().sqrt() / exact.pow(2).mean().sqrt()).item()


UNIFORM = torch.arange(-7, 8) / 7.0
c_u, a_u = quantize_nf4(W, table=UNIFORM)
err_nf4 = rel_rmse(W_nf4, W)
err_uniform = rel_rmse(dequantize_nf4(c_u, a_u, W.shape, table=UNIFORM), W)
print(f"относительная ошибка: NF4 {err_nf4:.3f}, равномерная {err_uniform:.3f}")
assert err_nf4 < err_uniform
