"""Много адаптеров LoRA в одном пакете: сегменты (SGMV), слоты с дополнением ранга и кэш, зависящий от адаптера.

Полная версия листингов подраздела «Обслуживание многих адаптеров LoRA» главы 14
(раздел 14.9 «PagedAttention и vLLM»).

1. Один слой BookLM (W_Q первого блока) и пакет из трёх запросов к разным адаптерам: настоящий адаптер
   главы 8 (r = 16), «другой клиент» со случайным адаптером (r = 8) и чистая основа. Пакетный проход
   считается двумя способами: циклом по сегментам, как SGMV в Punica, и сбором по номеру слота с
   дополнением ранга нулями до r_max, как слоты vLLM. Оба совпадают с поштучным расчётом и со слитыми
   весами W + (alpha/r) B A.
2. Вся BookLM: 42 матрицы заменены слоем со слотами, три строки пакета с одним и тем же входом идут к
   разным слотам. Строка основы совпадает с booklm-base, строка «сказок» — со слитой booklm-instruct
   в пределах допуска слияния главы 8, строка «другого» — с тем же адаптером, посчитанным отдельно.
3. KV-кэш одного и того же входа у основы и у адаптера различается во всех шести блоках: делить страницы
   кэша между адаптерами нельзя, поэтому vLLM кладёт имя и путь адаптера в хеш страницы, а SGLang — в ключ дерева.
4. Цена адаптера для модели формы Llama 3 8B: память, доля умножений и лишнее чтение памяти на шаге
   генерации в зависимости от ранга и числа разных адаптеров в пакете; сравнение со слитыми копиями.

Запуск из корня репозитория: python labs/ch14/multi_lora_batch.py
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from booklm.lab import load, prompt_ids  # noqa: E402

torch.set_grad_enabled(False)
W0 = torch.load(ROOT / "models/booklm/booklm-base.pt")
lora_file = torch.load(ROOT / "models/booklm/booklm-instruct-lora.pt")
AB, S = lora_file["adapter"], lora_file["alpha"] / lora_file["r"]   # r = 16, alpha / r = 2
TARGETS = lora_file["targets"]                                      # 7 матриц каждого блока
R_OTHER, R_MAX = 8, 16


def other_adapter(name, d_out, d_in, seed):
    """Адаптер «другого клиента»: случайные A и B ранга 8 вместо обученных."""
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(R_OTHER, d_in, generator=g) / d_in**0.5
    B = torch.randn(d_out, R_OTHER, generator=g) / 16
    return A, B, 2.0


# ----------------------------------------------------------------------------- 1. один слой

name = "blocks.0.attention.q"
W = W0[name + ".weight"]                               # [256, 256]
g = torch.Generator().manual_seed(0)                   # те же числа, что в листинге книги
adapters = {"сказки": (AB[name + ".A"], AB[name + ".B"], S),
            "другой": (torch.randn(R_OTHER, 256, generator=g) / 16,
                       torch.randn(256, R_OTHER, generator=g) / 16, 2.0)}


def sgmv(x, W, adapters, starts, names):
    """Пакет из сегментов: токены starts[i]..starts[i+1]-1 идут к адаптеру names[i]."""
    y = x @ W.T                                        # общий путь: одно умножение на весь пакет
    for lo, hi, a in zip(starts, starts[1:], names):
        if a is not None:                              # None — запрос к чистой основе
            A, B, s = adapters[a]
            y[lo:hi] += s * (x[lo:hi] @ A.T) @ B.T     # сжатие до r, затем разворот
    return y


def stack_slots(adapters, order, r_max):
    """Слоты, как в vLLM: слот 0 — нули (основа), ранг каждого адаптера дополнен нулями до r_max."""
    A0, B0, _ = adapters[order[0]]
    d_in, d_out = A0.size(1), B0.size(0)
    A_all = torch.zeros(len(order) + 1, r_max, d_in)
    B_all = torch.zeros(len(order) + 1, d_out, r_max)
    for k, a in enumerate(order, start=1):
        A, B, s = adapters[a]
        A_all[k, :A.size(0)], B_all[k, :, :B.size(1)] = A, s * B   # масштаб вносим в B
    return A_all, B_all


def gather_slots(x, W, A_all, B_all, slot_of_token):
    """Каждый токен достаёт A и B своего слота: [N, r_max, d_in] и [N, d_out, r_max]."""
    h = torch.einsum("nd,nrd->nr", x, A_all[slot_of_token])
    return x @ W.T + torch.einsum("nr,nor->no", h, B_all[slot_of_token])


names, starts = ["сказки", None, "другой"], [0, 3, 4, 6]
x = torch.randn(6, 256, generator=g)
y = sgmv(x, W, adapters, starts, names)

A_all, B_all = stack_slots(adapters, ["сказки", "другой"], R_MAX)
slot = torch.tensor([1, 1, 1, 0, 2, 2])
y_gather = gather_slots(x, W, A_all, B_all, slot)
assert (y - y_gather).abs().max() < 1e-5

print("1. Слой W_Q блока 0, пакет из трёх запросов")
for lo, hi, a in zip(starts, starts[1:], names):
    alone = sgmv(x[lo:hi], W, adapters, [0, hi - lo], [a])
    if a is None:
        merged = x[lo:hi] @ W.T
    else:
        A, B, s = adapters[a]
        merged = x[lo:hi] @ (W + s * B @ A).T
    gap_alone = (y[lo:hi] - alone).abs().max().item()
    gap_merged = (y[lo:hi] - merged).abs().max().item()
    assert gap_alone < 1e-5 and gap_merged < 1e-4
    print(f"   {a or 'основа':6} токены {lo}-{hi - 1}: поштучно {gap_alone:.0e}, слитые {gap_merged:.0e}")
print(f"   сегменты = слоты: {(y - y_gather).abs().max().item():.0e}")

# Сбор по слотам читает N копий A и B; сегменты — по одной на адаптер. Для W_Q и пакета 6 токенов:
gather_numbers = 6 * R_MAX * (256 + 256)
segment_numbers = (16 + 8) * (256 + 256)
print(f"   чисел адаптеров прочитано: сбор {gather_numbers}, сегменты {segment_numbers}")

# ----------------------------------------------------------------------------- 2. вся BookLM


class SlotLoRA(nn.Module):
    """nn.Linear основы и адаптеры в слотах; slot_of_row задаёт слот каждой строки пакета."""

    def __init__(self, base, A_all, B_all, slots):
        super().__init__()
        self.base, self.A_all, self.B_all, self.slots = base, A_all, B_all, slots

    def forward(self, x):                                   # x: [пакет, T, d_in]
        s = self.slots["row"]
        h = torch.einsum("btd,brd->btr", x, self.A_all[s])  # сжатие до r_max
        return self.base(x) + torch.einsum("btr,bor->bto", h, self.B_all[s])


def add_slots(model, slots):
    for i, (mod_name, layer) in enumerate(list(model.named_modules())):
        if mod_name.endswith(TARGETS):
            d_out, d_in = layer.weight.shape
            ad = {"сказки": (AB[mod_name + ".A"], AB[mod_name + ".B"], S),
                  "другой": other_adapter(mod_name, d_out, d_in, seed=i)}
            parent, attr = mod_name.rsplit(".", 1)
            setattr(model.get_submodule(parent), attr,
                    SlotLoRA(layer, *stack_slots(ad, ["сказки", "другой"], R_MAX), slots))
    return model


tok, multi = load("base")
slots = {"row": torch.tensor([0, 1, 2])}
add_slots(multi, slots)
n_wrapped = sum(isinstance(m, SlotLoRA) for m in multi.modules())
assert n_wrapped == 42
_, base = load("base")
_, merged = load("instruct")

ids = torch.tensor([prompt_ids(tok, "Напиши сказку про кота.", chat=True)])
batch = ids.repeat(3, 1)                                    # один вход, три адаптера
caches = [{"len": 0} for _ in multi.blocks]
logits = multi(batch, caches)
gap_base = (logits[0] - base(ids)[0]).abs().max().item()
gap_merged = (logits[1] - merged(ids)[0]).abs().max().item()
slots["row"] = torch.tensor([2])
gap_other = (logits[2] - multi(ids)[0]).abs().max().item()  # «другой» отдельно
slots["row"] = torch.tensor([0, 1, 2])
assert gap_base < 1e-4 and gap_merged < 1e-4 and gap_other < 1e-4
print("2. BookLM, три строки пакета к разным слотам")
print(f"   основа против booklm-base {gap_base:.0e}, сказки против слитой {gap_merged:.0e},",
      f"другой против отдельного прохода {gap_other:.0e}")

out = multi.generate(batch, max_new=10, temperature=0)[:, ids.size(1):]
for label, row in zip(("основа", "сказки", "другой"), out.tolist()):
    print(f"   {label:7}", repr(tok.decode(row)))

# ----------------------------------------------------------------------------- 3. кэш зависит от адаптера

print("3. Ключи кэша одного входа: основа против адаптера «сказки»")
gaps = []
for b, c in enumerate(caches):
    k = c["k"]                                              # [пакет, 2, T, 32]
    gap = (k[0] - k[1]).abs().max().item()
    rel = ((k[0] - k[1]).norm() / k[0].norm()).item()
    gaps.append(gap)
    print(f"   блок {b}: наибольшее расхождение K {gap:.2f}, относительное {rel:.0%}")
assert min(gaps) > 0.1

# ----------------------------------------------------------------------------- 4. цена адаптера

# Llama 3 8B: 32 блока, ширина 4096, 8 KV-голов по 128, промежуточный слой 14336; (d_out, d_in)
LLAMA = {"q": (4096, 4096), "k": (1024, 4096), "v": (1024, 4096), "o": (4096, 4096),
         "gate": (14336, 4096), "up": (14336, 4096), "down": (4096, 14336)}
L, BASE_BYTES = 32, 16.06e9                                 # веса Llama 3 8B в BF16
linear = L * sum(o * i for o, i in LLAMA.values())          # умножений на токен в линейных слоях
print("4. Адаптер на всех семи матрицах Llama 3 8B, BF16")
for r in (8, 16, 64):
    p = L * r * sum(o + i for o, i in LLAMA.values())
    print(f"   r={r:2d}: {p / 1e6:5.1f} млн чисел, {2 * p / 2**20:4.0f} МБ,",
          f"{2 * p / BASE_BYTES:.2%} весов, умножений {p / linear:.2%} от основы,",
          f"8 разных в пакете: +{8 * 2 * p / BASE_BYTES:.1%} к чтению весов")
p16 = L * 16 * sum(o + i for o, i in LLAMA.values())
print(f"   100 клиентов: слитые копии {100 * BASE_BYTES / 2**40:.2f} ТБ,",
      f"основа и 100 адаптеров r=16 {(BASE_BYTES + 100 * 2 * p16) / 2**30:.1f} ГБ")
assert abs(p16 - 41_943_040) == 0

print("multi_lora_batch: все проверки пройдены")
