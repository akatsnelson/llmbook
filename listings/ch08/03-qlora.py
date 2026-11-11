"""Листинги главы 8 «Дообучение» книги «Большие языковые модели», файл 03-qlora.

Код и вывод — ровно как в книге; текст книги сюда не входит. Листинги идут по порядку и
опираются друг на друга: запускайте файл целиком или ячейками сверху вниз (разметка # %%).
Проверка, что вывод совпал с книгой: python scripts/check_listings.py listings/ch08/03-qlora.py
"""

# %% Листинг «Квантование блоками в NF4» — раздел «Сетка NF4»
import torch

NF4 = torch.tensor([                  # уровни из таблицы NF4
    -1.0, -0.6961928, -0.5250731, -0.3949175, -0.2844414,
    -0.1847734, -0.0910500, 0.0, 0.0795803, 0.1609302, 0.2461123,
    0.3379152, 0.4407098, 0.5626170, 0.7229568, 1.0])

def quantize_nf4(w, block=64, table=NF4):
    blocks = w.reshape(-1, block)            # [n_blocks, block]
    absmax = blocks.abs().amax(dim=1)        # масштаб s_b на блок
    normed = blocks / absmax[:, None]        # в [-1, 1]
    dist = (normed[..., None] - table).abs() # до каждого уровня
    codes = dist.argmin(dim=-1).to(torch.uint8)
    return codes, absmax                     # коды 0..15, масштабы

def dequantize_nf4(codes, absmax, shape, table=NF4):
    return (table[codes.long()] * absmax[:, None]).reshape(shape)

def r4(t):                         # та же печать, что в разделе 8.2
    return t.double().mul(1e4).round().div(1e4).tolist()

w = torch.tensor([0.12, -0.40, 0.03, 0.80])   # ручной пример
codes, absmax = quantize_nf4(w, block=4)
print("коды", codes.tolist(), "масштаб", r4(absmax))
print("восстановили", r4(dequantize_nf4(codes, absmax, w.shape)))
spike = torch.tensor([0.12, -0.40, 0.03, 8.0])  # выброс в блоке
codes, absmax = quantize_nf4(spike, block=4)
print("с выбросом: коды", codes.tolist(),
      "веса", r4(dequantize_nf4(codes, absmax, spike.shape)))
# Вывод в книге:
# | коды [[9, 2, 7, 15]] масштаб [0.8]
# | восстановили [0.1287, -0.4201, 0.0, 0.8]
# | с выбросом: коды [[7, 6, 7, 15]] веса [0.0, -0.7284, 0.0, 8.0]

# %% Листинг «Байты основы BookLM в QLoRA» — раздел «Двойное квантование»
import math, sys
sys.path.insert(0, "labs/ch08")
from ft_kit import SHAPES                 # формы (out, in) из 8.2

sizes = [o * i for o, i in SHAPES.values()]
n_lin = 6 * sum(sizes)                    # веса 42 матриц
n_groups = 6 * sum(math.ceil(n / 64 / 256) for n in sizes)
base_bytes = n_lin // 2 + n_lin // 64 + 4 * n_groups   # коды, масштабы
base_bytes += 8192 * 256 * 2 + 13 * 256 * 2  # эмбеддинги, RMSNorm: BF16
print(f"основа {base_bytes:,d} Б = {base_bytes / 2**20:.2f} МБ, "
      f"в {13_244_928 / base_bytes:.2f} раза меньше BF16")
# Вывод в книге:
# | основа 6,533,712 Б = 6.23 МБ, в 2.03 раза меньше BF16

# %% Листинг «Дивергенция Кульбака–Лейблера» — раздел «Расстояние между распределениями»
import torch.nn.functional as F

def kl(logits_p, logits_q):              # KL(p || q) на позицию
    return F.kl_div(logits_q.log_softmax(-1),
                    logits_p.log_softmax(-1),
                    log_target=True, reduction="batchmean").item()

p3 = torch.tensor([[0.6, 0.3, 0.1]]).log()
q3 = torch.tensor([[0.3, 0.3, 0.4]]).log()
print(f"KL(p||q) = {kl(p3, q3):.4f}, KL(q||p) = {kl(q3, p3):.4f}")
# Вывод в книге:
# | KL(p||q) = 0.2773, KL(q||p) = 0.3466

# %% Листинг «Цена четырёх бит на BookLM» — раздел «Расстояние между распределениями»
import copy
from booklm.lab import DEVICE, generate, load, perplexity, val_tokens
from ft_kit import double_dequant, double_quant

tok, model = load("base")
bf16 = copy.deepcopy(model).to(torch.bfloat16)
nf4, n, worst = copy.deepcopy(model), 0, (0.0, "")
with torch.no_grad():
    for name, layer in nf4.named_modules():
        if name.endswith(("q", "k", "v", "out", "gate", "up", "down")):
            codes, absmax = quantize_nf4(layer.weight)
            ratio = (absmax.max() / absmax.median()).item()
            worst = max(worst, (ratio, name))     # самый сильный выброс
            absmax = double_dequant(*double_quant(absmax))
            w_hat = dequantize_nf4(codes, absmax, layer.weight.shape)
            layer.weight.copy_(w_hat)
            n += 1
print(f"матриц: {n}; сильнейший выброс {worst[0]:.1f} раза, {worst[1]}")

data, kls = val_tokens(50_000), {"BF16": [], "NF4": []}
with torch.no_grad():
    for s in range(0, len(data) - 512, 512):
        x = data[s:s + 512][None].to(DEVICE)
        ref = model(x)[0]                          # исходная модель p
        for key, m in (("BF16", bf16), ("NF4", nf4)):
            kls[key].append(kl(ref, m(x)[0].float()))
mean_kl = {k: sum(v) / len(v) for k, v in kls.items()}
print(f"KL к исходной: BF16 {mean_kl['BF16']:.5f}, "
      f"NF4 {mean_kl['NF4']:.5f}")
p0, p1 = perplexity(model), perplexity(nf4)
print(f"перплексия: исходная {p0:.3f}, NF4 {p1:.3f}")
for m in (model, nf4):
    text = generate(tok, m, "Однажды", max_new=10, temperature=0)
    print(" ".join(text.split()))
# Вывод в книге:
# | матриц: 42; сильнейший выброс 6.1 раза, blocks.1.attention.k
# | KL к исходной: BF16 0.00028, NF4 0.03868
# | перплексия: исходная 6.958, NF4 7.169
# | Однажды маленькая девочка по имени Лили пошла в парк. Она
# | Однажды маленький мальчик по имени Тим пошёл в парк. Он
