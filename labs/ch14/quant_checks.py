"""Проверки листингов о квантовании из главы 14 и то, что в книге оставлено словами.

Глава 14, раздел 14.5 «Квантование». В книге
листинги печатают числа; здесь те же функции с утверждениями assert, а также код, который в книге описан
без листинга: упаковка пяти тернарных весов в байт, скалярное произведение с тернарными весами (сложения
вместо умножений) и прямой оценщик градиента (STE) для обучения тернарной модели.

Запуск из корня репозитория: python labs/ch14/quant_checks.py
"""

import math

import torch


# --- Симметричное и групповое квантование ---------------------------------------------------------
def quantize_symmetric(x, bits=8):
    qmax = 2 ** (bits - 1) - 1
    scale = x.abs().amax().clamp(min=1e-8) / qmax
    q = torch.round(x / scale).clamp(-qmax, qmax).to(torch.int8)
    return q, scale


def dequantize(q, scale):
    return q.float() * scale


def groupwise_fake_quant(w, bits=4, group_size=128):
    qmax = 2 ** (bits - 1) - 1
    groups = w.reshape(w.size(0), -1, group_size)
    scales = groups.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / qmax
    q = torch.round(groups / scales).clamp(-qmax, qmax)
    return (q * scales).reshape_as(w), q.to(torch.int8), scales


def check_symmetric_and_groups():
    torch.manual_seed(1)
    w = torch.randn(4096)
    for bits in (8, 4):
        q, scale = quantize_symmetric(w, bits)
        err = (w - dequantize(q, scale)).abs()
        # худшая ошибка — не больше половины шага
        assert err.max() <= scale / 2 + 1e-6
    # ручной пример с рисунка: две группы по 4 веса в INT4
    w2 = torch.tensor([[0.1, -0.3, 2.8, 0.15, 0.12, -0.05, 0.3, -0.21]])
    w_hat, q, scales = groupwise_fake_quant(w2, bits=4, group_size=4)
    assert q.tolist() == [[[0, -1, 7, 0], [3, -1, 7, -5]]]
    assert torch.allclose(scales.flatten(), torch.tensor([0.4, 0.3 / 7]))
    # меньше группа — меньше ошибка
    torch.manual_seed(0)
    w3 = torch.randn(256, 1024)
    errs = [((w3 - groupwise_fake_quant(w3, 4, g)[0]) ** 2).mean().item() for g in (32, 128, 1024)]
    assert errs[0] < errs[1] < errs[2]


# --- Упаковка INT4 ----------------------------------------------------------------------------------
def pack_int4(q):
    u = (q.to(torch.int16) + 8).to(torch.uint8)
    if u.numel() % 2:
        u = torch.cat([u, torch.zeros(1, dtype=torch.uint8)])
    return u[0::2] | (u[1::2] << 4)


def unpack_int4(packed, n):
    low, high = packed & 0x0F, packed >> 4
    out = torch.stack([low, high], dim=1).flatten().to(torch.int8) - 8
    return out[:n]


def check_int4_packing():
    q = torch.tensor([-1, 3, -8, 7, 0], dtype=torch.int8)
    packed = pack_int4(q)
    assert packed.tolist() == [0xB7, 0xF0, 0x08]
    torch.testing.assert_close(unpack_int4(packed, len(q)), q)
    # перепутанные половины байта читают (3, -1) вместо (-1, 3)
    swapped = ((packed[:1] & 0x0F) << 4) | (packed[:1] >> 4)
    assert unpack_int4(swapped, 2).tolist() == [3, -1]


# --- GPTQ на двух весах -----------------------------------------------------------------------------
def check_gptq_two_weights():
    X = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    w = torch.tensor([1.4, 1.4])
    H = X.T @ X
    rtn = torch.round(w)
    q1 = torch.round(w[0])
    w2 = w[1] + (w[0] - q1) * H[0, 1] / H[1, 1]
    gptq = torch.stack([q1, torch.round(w2)])

    def out_err(v):
        return ((X @ w - X @ v) ** 2).mean()

    assert gptq.tolist() == [1.0, 2.0]
    assert torch.isclose(out_err(rtn) / out_err(gptq), torch.tensor(16.0))
    assert (w - gptq).abs().max() > (w - rtn).abs().max()   # веса хуже, выход лучше


# --- AWQ ----------------------------------------------------------------------------------------------
def awq_search(w, x, steps=11):
    reference = x @ w.T
    a = x.abs().mean(dim=0)
    results = []
    for alpha in torch.linspace(0, 1, steps).tolist():
        lam = a ** alpha
        lam = lam / (lam.max() * lam.min()).sqrt()
        w_hat, _, _ = groupwise_fake_quant(w * lam, bits=4, group_size=128)
        loss = ((reference - (x / lam) @ w_hat.T) ** 2).mean().item()
        results.append((loss, alpha))
    return results


def check_awq():
    # тождество Wx = (W Lambda)(Lambda^-1 x) на примере с рисунка
    W = torch.tensor([[0.2, 0.9]])
    x = torch.tensor([1.0, 4.0])
    lam = torch.tensor([1.0, 2.0])
    assert torch.allclose(W @ x, (W * lam) @ (x / lam))
    torch.manual_seed(0)
    w, xs = torch.randn(64, 256), torch.randn(512, 256)
    xs[:, 7] *= 50
    results = awq_search(w, xs)
    best_loss, best_alpha = min(results)
    rtn_loss = results[0][0]                       # alpha = 0 — обычное RTN
    assert best_loss < rtn_loss / 4 and 0 < best_alpha < 1
    assert results[-1][0] > rtn_loss               # перестарались — хуже RTN


# --- Тернарные веса -------------------------------------------------------------------------------------
def pack_five_trits(values):
    assert len(values) == 5 and all(v in (-1, 0, 1) for v in values)
    code = 0
    for position, value in enumerate(values):
        code += (value + 1) * 3 ** position         # -1, 0, +1 -> цифры 0, 1, 2
    assert 0 <= code < 243
    return code


def unpack_five_trits(code):
    assert 0 <= code < 243
    values = []
    for _ in range(5):
        code, digit = divmod(code, 3)
        values.append(digit - 1)
    return values


def pack_four_trits_2bit(values):
    """Двухбитная раскладка: -1 -> 00, 0 -> 01, +1 -> 10, четыре веса в байте."""
    code = 0
    for position, value in enumerate(values):
        code |= (value + 1) << (2 * position)
    return code


def ternary_absmean(w, eps=1e-8):
    alpha = w.abs().mean()
    q = torch.round(w / (alpha + eps)).clamp(-1, 1)
    return q, alpha


def ternary_dot(q, x, alpha):
    acc = 0.0
    for weight, value in zip(q, x):
        if weight == 1:
            acc += value
        elif weight == -1:
            acc -= value
        else:
            assert weight == 0
    return alpha * acc


def ternary_with_ste(weight):
    alpha = weight.abs().mean().clamp(min=1e-8)
    quantized = torch.round(weight / alpha).clamp(-1, 1) * alpha
    return weight + (quantized - weight).detach()   # вперёд quantized, назад производная 1


def check_ternary():
    assert math.isclose(math.log2(3), 1.5849625, rel_tol=1e-7)
    weights = [-1, 0, 1, -1, 1]
    byte = pack_five_trits(weights)
    assert byte == 183 == 0xB7                      # тот же байт, что у пары INT4
    assert unpack_five_trits(byte) == weights
    for code in range(243):
        assert pack_five_trits(unpack_five_trits(code)) == code
    assert 3 ** 6 > 256 >= 3 ** 5
    assert pack_four_trits_2bit([-1, 0, 1, -1]) == 0b00_10_01_00
    q, alpha = ternary_absmean(torch.tensor([-0.9, -0.1, 0.2, 1.2]))
    assert q.tolist() == [-1.0, 0.0, 0.0, 1.0] and math.isclose(alpha.item(), 0.6, rel_tol=1e-6)
    assert math.isclose(1.6 + 16 / 128, 1.725) and math.isclose(2 + 16 / 128, 2.125)
    assert ternary_dot([-1, 0, 1, -1], [0.5, 7.0, 1.5, -2.0], 0.25) == 0.75
    w = torch.tensor([-0.9, -0.1, 0.2, 1.2], requires_grad=True)
    out = ternary_with_ste(w)
    assert torch.allclose(out, torch.tensor([-0.6, 0.0, 0.0, 0.6]))
    out.sum().backward()
    assert w.grad.tolist() == [1.0, 1.0, 1.0, 1.0]


# --- Таблица бит на вес -----------------------------------------------------------------------------------
def check_bits_table():
    assert 4 + 16 / 32 == 4.5 and 4 + 16 / 128 == 4.125
    assert (32 * 4 + 8) / 32 == 4.25
    # для модели 8B размер в ГБ численно равен числу бит на вес
    for bits in (16, 4.9, 4.5, 4.25, 4.125, 2.125, 1.725):
        assert math.isclose(8e9 * bits / 8 / 1e9, bits)
    # gpt-oss-120b: эксперты в MXFP4, остальное в BF16
    gb = (115e9 * 4.25 / 8 + 2e9 * 2) / 1e9
    assert round(gb) == 65


def main():
    check_symmetric_and_groups()
    check_int4_packing()
    check_gptq_two_weights()
    check_awq()
    check_ternary()
    check_bits_table()
    print("все проверки квантования прошли")


if __name__ == "__main__":
    main()
