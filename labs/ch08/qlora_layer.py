"""Линейный слой QLoRA: буферы вместо параметров, отключение, слияние и KL логитов.

Код к разделу 8.3 «QLoRA», подразделы «Типы хранения и вычисления», «Расстояние между распределениями»
и «Отключение и слияние после QLoRA». В книге слой описан словами; здесь он собран целиком.
Выполняется в ноутбуке главы 08_lora_sft.ipynb и отдельно.
Запуск из корня репозитория: python labs/ch08/qlora_layer.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- NF4 и двойное квантование из qlora_nf4.py и qlora_double_quant.py -------
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


def double_quant(absmax, block=256):
    groups = absmax.reshape(-1, block)
    c = groups.amax(dim=1)
    q8 = torch.round(groups / c[:, None] * 255).to(torch.uint8)
    return q8, c


def double_dequant(q8, c):
    return (q8.float() / 255 * c[:, None]).reshape(-1)


# --- слой QLoRA ---------------------------------------------------------------
class QLoRALinear(nn.Module):
    """Основа в NF4 с двойным квантованием, адаптер LoRA, вычисление в BF16."""

    def __init__(self, weight, r=8, alpha=16):
        super().__init__()
        d_out, d_in = weight.shape
        codes, absmax = quantize_nf4(weight)
        q8, c = double_quant(absmax)
        # Буферы, а не параметры: оптимизатор их не видит, градиента у них нет
        self.register_buffer("codes", codes)
        self.register_buffer("q8", q8)
        self.register_buffer("c", c)
        self.shape = (d_out, d_in)
        self.A = nn.Parameter(torch.randn(r, d_in) / d_in ** 0.5)
        self.B = nn.Parameter(torch.zeros(d_out, r))
        self.scale, self.enabled = alpha / r, True

    def base_weight(self):            # W~ в BF16: восстановить, умножить, выбросить
        absmax = double_dequant(self.q8, self.c)
        return dequantize_nf4(self.codes, absmax, self.shape).bfloat16()

    def forward(self, x):
        x = x.bfloat16()
        y = x @ self.base_weight().T
        if self.enabled:
            y = y + self.scale * (x @ self.A.bfloat16().T) @ self.B.bfloat16().T
        return y


torch.manual_seed(1)
W_ref = torch.randn(256, 768) / 768 ** 0.5        # исходная W_down, [d_out, d_in]
layer = QLoRALinear(W_ref)
assert [name for name, _ in layer.named_parameters()] == ["A", "B"]

x = torch.randn(16, 768, requires_grad=True)
x0 = x.detach().bfloat16()
y = layer(x)
W_tilde = layer.base_weight()
assert torch.equal(y, x0 @ W_tilde.T)                        # шаг 0: ровно основа W~
G = torch.randn(16, 256)                                      # dL/dy
(y.float() * G).sum().backward()
assert torch.count_nonzero(layer.A.grad) == 0                 # B = 0, значит dL/dA = 0
assert torch.count_nonzero(layer.B.grad) > 0
assert torch.allclose(x.grad, (G.bfloat16() @ W_tilde).float(), rtol=1e-2, atol=1e-2)

# Шаг по B, затем выключаем адаптер: возвращается W~, а не W
with torch.no_grad():
    layer.B -= 0.1 * layer.B.grad
layer.enabled = False
assert torch.equal(layer(x0), x0 @ W_tilde.T)
assert not torch.equal(layer(x0), x0 @ W_ref.bfloat16().T)
layer.enabled = True

# Слияние поправки в исходную W вместо W~
delta = layer.scale * layer.B.detach() @ layer.A.detach()
merge_gap = (x0.float() @ (W_ref + delta).T - layer(x0).float()).abs().max().item()
print(f"слияние в W вместо W~: расхождение {merge_gap:.3f}, |y| в среднем "
      f"{y.float().abs().mean().item():.2f}")
assert merge_gap > 1e-2

# KL логитов к FP32 на заглушке: ветвь SwiGLU формы BookLM, RMSNorm, словарь 1024
torch.manual_seed(2)
d, V = 256, 1024
Ws = [torch.randn(768, d) / d ** 0.5, torch.randn(768, d) / d ** 0.5,   # gate, up
      torch.randn(d, 768) / 768 ** 0.5]                                  # down
E = torch.randn(V, d) * 0.15
xs = torch.randn(64, d)


def logits_from(weights, dtype):
    g, u, dn = (w.to(dtype) for w in weights)
    x = xs.to(dtype)
    h = (x + (F.silu(x @ g.T) * (x @ u.T)) @ dn.T).float()
    return h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6) @ E.T


def kl(logits_p, logits_q):                                  # KL(p || q) на позицию
    return F.kl_div(logits_q.log_softmax(-1), logits_p.log_softmax(-1),
                    log_target=True, reduction="batchmean").item()


p3, q3 = torch.tensor([[0.6, 0.3, 0.1]]), torch.tensor([[0.3, 0.3, 0.4]])
assert abs(kl(p3.log(), q3.log()) - 0.2773) < 1e-4             # пример из книги
assert abs(kl(q3.log(), p3.log()) - 0.3466) < 1e-4             # другой порядок


def nf4_roundtrip(w):
    codes, absmax = quantize_nf4(w)
    return dequantize_nf4(codes, double_dequant(*double_quant(absmax)), w.shape)


ref = logits_from(Ws, torch.float32)
kl_bf16 = kl(ref, logits_from(Ws, torch.bfloat16))
kl_nf4 = kl(ref, logits_from([nf4_roundtrip(w) for w in Ws], torch.bfloat16))
print(f"KL к FP32: только BF16 {kl_bf16:.5f}, NF4 + BF16 {kl_nf4:.5f}")
assert kl_bf16 < kl_nf4 < 5e-2
assert kl_nf4 > 100 * kl_bf16                    # NF4 огрубляет в сотни раз сильнее
