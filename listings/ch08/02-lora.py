"""Листинги главы 8 «Дообучение» книги «Большие языковые модели», файл 02-lora.

Код и вывод — ровно как в книге; текст книги сюда не входит. Листинги идут по порядку и
опираются друг на друга: запускайте файл целиком или ячейками сверху вниз (разметка # %%).
Проверка, что вывод совпал с книгой: python scripts/check_listings.py listings/ch08/02-lora.py
"""

# %% Листинг «Слой LoRALinear» — раздел «Шаг обучения на двух числах»
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    """y = x W^T + (alpha/r)(x A^T) B^T; W заморожена"""
    def __init__(self, base: nn.Linear, r, alpha, seed=0):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)    # W не учится
        d_out, d_in = base.weight.shape
        g = torch.Generator().manual_seed(seed)
        A = torch.randn(r, d_in, generator=g) / d_in**0.5
        self.A = nn.Parameter(A)                       # [r, d_in]
        self.B = nn.Parameter(torch.zeros(d_out, r))   # B = 0
        self.scale, self.enabled = alpha / r, True

    def forward(self, x):                         # x: [..., d_in]
        y = self.base(x)
        if self.enabled:
            y = y + self.scale * (x @ self.A.T) @ self.B.T
        return y

    def merged_weight(self):                   # W + (alpha/r) BA
        return self.base.weight + self.scale * self.B @ self.A

torch.manual_seed(0)
base = nn.Linear(256, 64, bias=False)          # форма W_K BookLM
lora = LoRALinear(base, r=16, alpha=32)
x = torch.randn(5, 256)
print("шаг 0 = основа:", torch.equal(lora(x), base(x)))
lora(x).pow(2).sum().backward()
print("градиент W:", base.weight.grad,
      "| ненулевых в dA:", torch.count_nonzero(lora.A.grad).item(),
      "| в dB:", torch.count_nonzero(lora.B.grad).item())
# Вывод в книге:
# | шаг 0 = основа: True
# | градиент W: None | ненулевых в dA: 0 | в dB: 1024

# %% Листинг «Шаг LoRA, слияние и отключение на двух числах» — раздел «Шаг обучения на двух числах»
def r4(t):                                   # печать до 4 знаков
    return t.double().mul(1e4).round().div(1e4).tolist()

base = nn.Linear(2, 2, bias=False)
lora = LoRALinear(base, r=1, alpha=1)
with torch.no_grad():
    base.weight.copy_(torch.eye(2))          # W = I
    lora.A.fill_(1.0)                        # A = [1 1], B = 0
x, t = torch.tensor([[1.0, 2.0]]), torch.tensor([[1.0, 3.0]])
loss = 0.5 * (lora(x) - t).pow(2).sum()
loss.backward()
print("шаг 0: y =", r4(lora(x)), "L =", r4(loss))
print("dB =", r4(lora.B.grad), "dA =", r4(lora.A.grad))
with torch.no_grad():
    lora.B -= 0.1 * lora.B.grad              # шаг спуска, lr = 0.1
    lora.A -= 0.1 * lora.A.grad
    y = lora(x)
    print("шаг 1: y =", r4(y), "L =", r4(0.5 * (y - t).pow(2).sum()))
    print("W' =", r4(lora.merged_weight()),
          "x W'^T =", r4(x @ lora.merged_weight().T))
    lora.enabled = False
    print("адаптер выключен: y =", r4(lora(x)))
# Вывод в книге:
# | шаг 0: y = [[1.0, 2.0]] L = 0.5
# | dB = [[0.0], [-3.0]] dA = [[0.0, 0.0]]
# | шаг 1: y = [[1.0, 2.9]] L = 0.005
# | W' = [[1.0, 0.0], [0.3, 1.3]] x W'^T = [[1.0, 2.9]]
# | адаптер выключен: y = [[1.0, 2.0]]

# %% Листинг «Подсчёт параметров и байтов адаптера BookLM» — раздел «Размер адаптера BookLM»
SHAPES = {"q": (256, 256), "k": (64, 256), "v": (64, 256),
          "out": (256, 256), "gate": (768, 256), "up": (768, 256),
          "down": (256, 768)}                    # (out, in)

def lora_params(r, targets=tuple(SHAPES), n_blocks=6):
    return n_blocks * sum(r * sum(SHAPES[n]) for n in targets)

P = 6_622_464                           # параметры BookLM
for r in (4, 8, 16):
    n = lora_params(r)
    print(f"r={r:2d}: {n:7,d} чисел, {n / P:.2%} модели")
print(f"r=16 только на q и v: {lora_params(16, ('q', 'v')):,d}")
adapter = lora_params(16)
full_sft = 16 * P                       # 2 + 4 + 2 + 4 + 4 байта
lora_sft = 2 * P + 16 * adapter         # основа BF16 + адаптер
print(f"полное SFT {full_sft / 2**20:.2f} МБ, "
      f"LoRA {lora_sft / 2**20:.2f} МБ, "
      f"адаптер в BF16 {2 * adapter / 2**10:.0f} КБ")
# Вывод в книге:
# | r= 4: 113,664 чисел, 1.72% модели
# | r= 8: 227,328 чисел, 3.43% модели
# | r=16: 454,656 чисел, 6.87% модели
# | r=16 только на q и v: 79,872
# | полное SFT 101.05 МБ, LoRA 19.57 МБ, адаптер в BF16 888 КБ

# %% Листинг «Насколько адаптер сдвинул веса BookLM» — раздел «Размер адаптера BookLM»
import torch

W0 = torch.load("models/booklm/booklm-base.pt")          # основа
lora_file = torch.load("models/booklm/booklm-instruct-lora.pt")
AB, s = lora_file["adapter"], lora_file["alpha"] / lora_file["r"]
NAMES = {"attention.q": "W_Q", "attention.k": "W_K", "attention.v": "W_V",
         "attention.out": "W_O", "ffn.gate": "W_gate", "ffn.up": "W_up",
         "ffn.down": "W_down"}

def delta(name):                                  # (alpha/r) B A
    return s * AB[name + ".B"] @ AB[name + ".A"]

print("|dW| / |W|, %")
print("блок" + "".join(f"{n:>7}" for n in NAMES.values()))
for b in range(6):
    rel = [100 * delta(f"blocks.{b}.{t}").norm()
           / W0[f"blocks.{b}.{t}.weight"].norm() for t in NAMES]
    print(f"{b:4d}" + "".join(f"{v:7.1f}" for v in rel))
# Вывод в книге:
# | |dW| / |W|, %
# | блок    W_Q    W_K    W_V    W_O W_gate   W_up W_down
# |    0   36.6   29.8   23.7   33.3   13.7   13.6    8.0
# |    1   19.4   12.6    8.9   12.0   15.5   16.9   11.3
# |    2   19.6   15.2   11.3   14.3   15.3   16.6   11.8
# |    3   24.4   19.1   10.5   14.5   15.3   16.6   11.9
# |    4   18.3   12.0    8.6   13.5   15.1   16.3   11.6
# |    5   22.5   16.9   10.7   15.5   15.0   16.3   11.4

# %% Листинг «Сингулярные числа поправки и основы» — раздел «Размер адаптера BookLM»
name = "blocks.0.attention.q"
sv_dW = torch.linalg.svdvals(delta(name).double())
sv_W = torch.linalg.svdvals(W0[name + ".weight"].double())
for label, sv in (("dW", sv_dW), ("W ", sv_W)):
    print(label, " ".join(f"{v:.2g}" for v in sv[:3]), "...",
          " ".join(f"{v:.2g}" for v in sv[14:18]))
print("ранг dW:", torch.linalg.matrix_rank(delta(name)).item())
# Вывод в книге:
# | dW 2.3 2 1.6 ... 0.46 0.42 5.2e-08 4.6e-08
# | W  4.9 3.8 3.2 ... 1.4 1.3 1.3 1.3
# | ранг dW: 16

# %% Листинг «Слияние, отключение и «разлияние» в BF16» — раздел «Слияние и отключение адаптера»
torch.manual_seed(1)
layer = LoRALinear(nn.Linear(32, 32, bias=False), r=4, alpha=8)
with torch.no_grad():
    layer.B.normal_(0, 0.1)              # будто адаптер уже обучен
    X = torch.randn(256, 32)
    y_adapter = layer(X)
    y_merged = X @ layer.merged_weight().T
    gap = (y_merged - y_adapter).abs().max().item()
    print(f"слитая FP32 против адаптерного режима: {gap:.0e}")
    layer.enabled = False
    print("выключенный адаптер = основа:",
          torch.equal(layer(X), layer.base(X)))
    layer.enabled = True

    W = layer.base.weight.bfloat16()                # основа в BF16
    dW = (layer.scale * layer.B @ layer.A).bfloat16()
    W_back = (W + dW) - dW                          # «разлили»
    share = (W_back != W).float().mean().item()
    print(f"после разлияния в BF16 не совпало {share:.0%} весов")
# Вывод в книге:
# | слитая FP32 против адаптерного режима: 7e-07
# | выключенный адаптер = основа: True
# | после разлияния в BF16 не совпало 25% весов

# %% Листинг «Адаптер на BookLM и первые шаги SFT» — раздел «Дообучение BookLM с адаптером»
import random
import torch.nn.functional as F
from booklm.lab import DEVICE, load
from booklm.experiments.ch08 import held_out, sft_train
from sft_lora import encode as encode_pair     # шаблон и маска 8.1

def add_lora(model, r=16, alpha=32):
    model.requires_grad_(False)                # основа заморожена
    for i, (name, layer) in enumerate(list(model.named_modules())):
        if name.endswith(tuple(NAMES)):        # 7 матриц каждого блока
            parent, attr = name.rsplit(".", 1)
            setattr(model.get_submodule(parent), attr,
                    LoRALinear(layer, r, alpha, seed=i))
    return model.to(DEVICE)

def batch_of(rows):              # пары -> входы и цели одной длины
    pairs = [encode_pair(tok, r["request"], r["story"], 512) for r in rows]
    T = max(len(ids) for ids, _ in pairs) - 1
    x = [ids[:-1] + [0] * (T + 1 - len(ids)) for ids, _ in pairs]
    y = [lab[1:] + [-100] * (T + 1 - len(lab)) for _, lab in pairs]
    return torch.tensor(x, device=DEVICE), torch.tensor(y, device=DEVICE)

tok, model = load("base")
add_lora(model)
params = [p for p in model.parameters() if p.requires_grad]
n_wrapped = sum(isinstance(m, LoRALinear) for m in model.modules())
print("обёрнуто матриц:", n_wrapped,
      "| обучаемых чисел:", sum(p.numel() for p in params))
loss_on = lambda x, y: F.cross_entropy(model(x).flatten(0, 1), y.flatten())
x_val, y_val = batch_of(held_out()[:32])
train, rng = sft_train(), random.Random(0)
opt = torch.optim.AdamW(params, lr=2e-3, weight_decay=0.0)
for step in range(41):
    if step in (0, 1, 10, 40):
        with torch.no_grad():
            val = loss_on(x_val, y_val)
        print(f"шаг {step:2d}: ошибка на отложенных {val:.4f}")
    loss = loss_on(*batch_of(rng.sample(train, 8)))
    opt.zero_grad()
    loss.backward()
    opt.step()
# Вывод в книге:
# | обёрнуто матриц: 42 | обучаемых чисел: 454656
# | шаг  0: ошибка на отложенных 1.7219
# | шаг  1: ошибка на отложенных 1.6622
# | шаг 10: ошибка на отложенных 1.6557
# | шаг 40: ошибка на отложенных 1.6641

# %% Листинг «Четыре режима BookLM» — раздел «Дообучение BookLM с адаптером»
@torch.no_grad()
def nll(model, row):                     # ошибка на каждой цели истории
    ids, labels = encode_pair(tok, row["request"], row["story"], 512)
    y = torch.tensor(labels[1:], device=DEVICE)
    logits = model(torch.tensor([ids[:-1]], device=DEVICE))[0]
    return F.cross_entropy(logits, y, reduction="none")[y != -100]

def set_lora(model, on):
    for layer in model.modules():
        if isinstance(layer, LoRALinear):
            layer.enabled = on

_, base = load("base")
_, tuned = load("base")
add_lora(tuned).load_state_dict(lora_file["adapter"], strict=False)
_, merged = load("instruct")
rows, seen = held_out(), {}
print("режим              все цели  первые 5  остальные")
MODES = (("база", base, True), ("адаптер", tuned, True),
         ("адаптер выключен", tuned, False), ("слитая", merged, True))
for name, m, on in MODES:
    set_lora(m, on)
    v = seen[name] = [nll(m, r) for r in rows]
    full = torch.cat(v).mean()
    head = torch.cat([t[:5] for t in v]).mean()     # первые 5 целей
    rest = torch.cat([t[5:] for t in v]).mean()
    print(f"{name:17s}  {full:.6f}  {head:8.3f}  {rest:9.3f}")
off = zip(seen["база"], seen["адаптер выключен"])
print("выключенный адаптер = база бит в бит:",
      all(torch.equal(a, b) for a, b in off))
# Вывод в книге:
# | режим              все цели  первые 5  остальные
# | база               1.702523     4.356      1.631
# | адаптер            1.598963     1.369      1.605
# | адаптер выключен   1.702523     4.356      1.631
# | слитая             1.598963     1.369      1.605
# | выключенный адаптер = база бит в бит: True

# %% Листинг «Берёт ли BookLM героя из просьбы» — раздел «Дообучение BookLM с адаптером»
import re, textwrap
from booklm.lab import generate

def reply(model, request, n=120):        # жадный ответ на просьбу
    return generate(tok, model, request, chat=True, max_new=n,
                    temperature=0)

ask = "Расскажи сказку про ёжика, который боялся темноты."
text = " ".join(reply(merged, ask, 40).split())
print(textwrap.fill("после LoRA: " + text, 66))

HEROES = {"девочку": r"девоч", "кота": r"\bкот(?!ор)",
          "собаку": r"собак|\bпёс", "медведя": r"медвед",
          "черепаху": r"черепах", "дракона": r"дракон",
          "жирафа": r"жираф", "ёжика": r"\bёж|\bежик"}
FORMS = ("Напиши сказку про {}.", "Расскажи историю про {}.",
         "Сочини короткую сказку про {}.")
start = lambda text: " ".join(text.split()[:3])
tales = 0                                # ответы, начатые с «Жил-был»
print(f"{'герой':9s} {'база начинает':23s} {'LoRA начинает':28s} герой")
for hero, pat in HEROES.items():
    answers = [reply(merged, f.format(hero)) for f in FORMS]
    found = sum(bool(re.search(pat, a.lower())) for a in answers)
    tales += sum(a.split()[0] == "Жил-был" for a in answers)
    first = start(reply(base, FORMS[0].format(hero), 12))
    print(f"{hero:9s} {first:23.23s} {start(answers[0]):28.28s} {found}/3")
print("начинают с «Жил-был»:", tales, "из", 3 * len(HEROES))
# Вывод в книге:
# | после LoRA: Жил-был большой, страшный монстр. Он жил в маленьком
# | домике. Монстр любил спать весь день. Он был очень тихий и не
# | любил много спать. Однажды монстр увидел маленькую птичку. Птичка
# | была
# | герой     база начинает           LoRA начинает                герой
# | девочку   Джуби не могла          Жил-был маленький девочка    3/3
# | кота      кот, который хотел      Жил-был маленький кот.       3/3
# | собаку    младший брат, который   Жил-был маленький пёс        3/3
# | медведя   Медведь, который хотел  Жил-был маленький медведь.   3/3
# | черепаху  га была очень           Жил-был маленький козлёнок.  0/3
# | дракона   ерно, но он             Жил-был маленький дракон.    3/3
# | жирафа    жирафа, который хотел   Жил-был жираф. Он            3/3
# | ёжика     томерни — это           Жил-был большой ёж.          2/3
# | начинают с «Жил-был»: 24 из 24

# %% Листинг «Растяжение строк и лучший адаптер LoRA» — раздел «Адаптер с разложением весов (DoRA)»
torch.manual_seed(0)
W = torch.randn(64, 256) / 16                   # форма W_K BookLM
c = 1 + 0.1 * torch.randn(64)                   # длины строк +-10%
target = c[:, None] * W                         # направления те же
dW = target - W
S = torch.linalg.svdvals(dW)
for r in (4, 16, 32):                           # лучшее: первые r чисел
    left = (S[r:].pow(2).sum() / S.pow(2).sum()).sqrt()
    print(f"LoRA r={r:2d} в лучшем случае упустит {left:.0%} поправки")
# Вывод в книге:
# | LoRA r= 4 в лучшем случае упустит 77% поправки
# | LoRA r=16 в лучшем случае упустит 47% поправки
# | LoRA r=32 в лучшем случае упустит 24% поправки

# %% Листинг «Слой DoRALinear» — раздел «Адаптер с разложением весов (DoRA)»
class DoRALinear(LoRALinear):
    def __init__(self, base: nn.Linear, r, alpha, seed=0):
        super().__init__(base, r, alpha, seed)       # W, A, B, s как у LoRA
        self.m = nn.Parameter(base.weight.norm(dim=1))   # [d_out]

    def row_norm(self):                          # n: [d_out]
        return (self.base.weight + self.scale * self.B @ self.A).norm(dim=1)

    def forward(self, x):
        if not self.enabled:
            return self.base(x)
        k = self.m / self.row_norm().detach()    # n без градиента
        return k * super().forward(x)            # выход LoRA, растянутый

    def merged_weight(self):                     # m * V / ||V||
        return (self.m / self.row_norm())[:, None] * super().merged_weight()

# %% Листинг «Проверка DoRALinear» — раздел «Адаптер с разложением весов (DoRA)»
torch.manual_seed(0)
base = nn.Linear(256, 64, bias=False)            # форма W_K BookLM
dora = DoRALinear(base, r=16, alpha=32)
x = torch.randn(5, 256)
print("шаг 0 = основа:", torch.equal(dora(x), base(x)))
with torch.no_grad():                            # будто адаптер обучен
    dora.B.normal_(0, 0.1)
    dora.m.mul_(1 + 0.1 * torch.randn(64))
    W1 = dora.merged_weight()
    print(f"длины строк W' минус m: {(W1.norm(dim=1) - dora.m).abs().max():.0e}")
    print(f"слитая против адаптерной: {(x @ W1.T - dora(x)).abs().max():.0e}")
    stretch = DoRALinear(nn.Linear(256, 64, bias=False), r=4, alpha=8)
    stretch.base.weight.copy_(W)                 # W и c из первого листинга
    stretch.m.copy_(c * W.norm(dim=1))           # B = 0: правим только m
    gap = (stretch.merged_weight() - target).norm() / dW.norm()
    print(f"растяжение строк, DoRA r=4: осталось {gap:.0e} поправки")
# Вывод в книге:
# | шаг 0 = основа: True
# | длины строк W' минус m: 2e-07
# | слитая против адаптерной: 1e-06
# | растяжение строк, DoRA r=4: осталось 2e-07 поправки
