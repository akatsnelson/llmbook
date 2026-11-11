"""Листинги главы 7 «Предобучение» книги «Большие языковые модели», файл 01-training-step.

Код и вывод — ровно как в книге; текст книги сюда не входит. Листинги идут по порядку и
опираются друг на друга: запускайте файл целиком или ячейками сверху вниз (разметка # %%).
Проверка, что вывод совпал с книгой: python scripts/check_listings.py listings/ch07/01-training-step.py
"""

# %% Листинг «Шаг по логитам руками и в PyTorch» — раздел «Ручной расчёт шага»
import torch
import torch.nn.functional as F
torch.set_printoptions(precision=4, sci_mode=False)

def row_loss(row, targets):                  # row: [V], строка таблицы
    # expand: та же строка для каждой цели, без копирования
    return F.cross_entropy(row.expand(len(targets), -1),
                           torch.tensor(targets))

def gd_step(row, targets, eta):              # theta - eta * g
    row = row.detach().clone().requires_grad_(True)
    row_loss(row, targets).backward()
    return (row - eta * row.grad).detach()

row = torch.tensor([2.0, 1.0, 0.0], requires_grad=True)
loss = row_loss(row, [0])
loss.backward()                              # g от autograd
p = row.detach().softmax(dim=-1)
print("p          ", p)
print("p - [j=y]  ", p - torch.tensor([1.0, 0.0, 0.0]))
print("row.grad   ", row.grad)
for eta in (0.5, 1.0):
    new = gd_step(row, [0], eta)
    print(f"eta={eta}: {new}, ошибка {row_loss(new, [0]).item():.4f}")
# Вывод в книге:
# | p           tensor([0.6652, 0.2447, 0.0900])
# | p - [j=y]   tensor([-0.3348,  0.2447,  0.0900])
# | row.grad    tensor([-0.3348,  0.2447,  0.0900])
# | eta=0.5: tensor([ 2.1674,  0.8776, -0.0450]), ошибка 0.3255
# | eta=1.0: tensor([ 2.3348,  0.7553, -0.0900]), ошибка 0.2582

# %% Листинг «Длинный шаг на пакете из двух позиций» — раздел «Ручной расчёт шага»
both = [0, 1]                        # «мама» дважды: дальше токены 0 и 1
print(f"старт:    ошибка пакета {row_loss(row, both).item():.4f}")
for eta in (1.0, 10.0):
    new = gd_step(row, both, eta)
    print(f"eta={eta:>4}: ошибка пакета {row_loss(new, both).item():.4f},"
          f" p = {new.softmax(dim=-1)}")
# Вывод в книге:
# | старт:    ошибка пакета 0.9076
# | eta= 1.0: ошибка пакета 0.8239, p = tensor([0.5861, 0.3283, 0.0855])
# | eta=10.0: ошибка пакета 1.6534, p = tensor([0.0385, 0.9504, 0.0111])

# %% Листинг «Логиты при таблице из N(0; 1)» — раздел «Проверка начальной ошибки»
torch.manual_seed(0)
table = torch.randn(8192, 256)           # как nn.Embedding: N(0; 1)
e = table[5]                             # на входе токен 5
logits = table @ e                       # логиты всех 8192 токенов
print(f"длина строки e:         {e.norm():.1f}")
print(f"свой токен, e @ e:      {logits[5]:.0f}")
print(f"токен 7:                {logits[7]:.1f}")
print(f"самый крупный из чужих: {logits[torch.arange(8192) != 5].max():.0f}")
# Вывод в книге:
# | длина строки e:         16.5
# | свой токен, e @ e:      271
# | токен 7:                -8.6
# | самый крупный из чужих: 72

# %% Листинг «TinyLM: BookLM без блоков» — раздел «Проверка начальной ошибки»
import math
import torch.nn as nn

class RMSNorm(nn.Module):           # как в booklm/model.py: счёт в float32
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight, self.eps = nn.Parameter(torch.ones(d)), eps

    def forward(self, x):
        ms = x.float().pow(2).mean(dim=-1, keepdim=True)
        x32 = x.float() * torch.rsqrt(ms + self.eps)
        return x32.to(x.dtype) * self.weight

class TinyLM(nn.Module):                      # BookLM с layers=0
    def __init__(self, vocab_size=8192, d=256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d)
        self.blocks = nn.ModuleList()         # у BookLM здесь шесть блоков
        self.norm = RMSNorm(d)
        self.output = nn.Linear(d, vocab_size, bias=False)
        self.output.weight = self.embed.weight       # связанные веса

    def forward(self, ids):                  # [B, T] -> [B, T, V]
        return self.output(self.norm(self.embed(ids)))

tiny = TinyLM()
print("один тензор:", tiny.embed.weight is tiny.output.weight)
print("параметров:", sum(p.numel() for p in tiny.parameters()))
# Вывод в книге:
# | один тензор: True
# | параметров: 2097408

# %% Листинг «Правило инициализации и стартовая ошибка» — раздел «Проверка начальной ошибки»
@torch.no_grad()
def init_booklm_(model, std=0.02):            # правило GPT
    for p in model.parameters():              # связанная матрица - однажды
        if p.dim() == 1:
            nn.init.ones_(p)                  # усиления RMSNorm
        else:
            nn.init.normal_(p, std=std)       # все матрицы и таблица

def start_loss(model, V=8192, batch=4, length=128, seed=0):
    ids = torch.randint(0, V, (batch, length + 1),
                        generator=torch.Generator().manual_seed(seed))
    with torch.no_grad():
        logits = model(ids[:, :-1])
    return F.cross_entropy(logits.flatten(0, 1),
                           ids[:, 1:].flatten()).item()

torch.manual_seed(0)
tiny = TinyLM()
print(f"по умолчанию PyTorch: {start_loss(tiny):6.2f}")
init_booklm_(tiny)                                  # правило GPT
print(f"правило GPT:          {start_loss(tiny):6.2f}")
print(f"ln V:                 {math.log(8192):6.2f}")
nn.init.normal_(tiny.embed.weight, std=1.0)         # поломка: N(0, 1)
print(f"таблица снова N(0,1): {start_loss(tiny):6.2f}")
# Вывод в книге:
# | по умолчанию PyTorch: 255.75
# | правило GPT:            9.08
# | ln V:                   9.01
# | таблица снова N(0,1): 257.38

# %% Листинг «Один шаг обучения как функция» — раздел «Функция шага обучения»
def batch_loss(model, inputs, targets, loss_mask=None):
    logits = model(inputs)                        # [B, T, V]
    per_pos = F.cross_entropy(logits.flatten(0, 1), targets.flatten(),
                              reduction="none")   # [B*T]
    if loss_mask is None:
        return per_pos.mean()
    m = loss_mask.flatten().to(per_pos.dtype)     # маска ошибки M
    return (per_pos * m).sum() / m.sum()

def train_step(model, batch, opt):
    opt.zero_grad(set_to_none=True)   # прошлого g больше нет
    loss = batch_loss(model, *batch)  # 1-4: пакет, проход, одно число
    loss.backward()                   # 5: обратный проход
    opt.step()                        # 6: theta - eta * g
    return {"loss": loss.item()}      # строка журнала

def grad_norm(model):                 # корень из суммы квадратов всех g
    return math.sqrt(sum(p.grad.pow(2).sum().item()
                         for p in model.parameters() if p.grad is not None))

torch.manual_seed(0)
model = TinyLM()
init_booklm_(model)
opt = torch.optim.SGD(model.parameters(), lr=0.02)
ids = torch.randint(0, 8192, (4, 129),
                    generator=torch.Generator().manual_seed(1))
batch = (ids[:, :-1], ids[:, 1:])             # X и Y со сдвигом
for s in range(5):
    row = train_step(model, batch, opt)
    print(f"шаг {s}: ошибка {row['loss']:.3f}, норма g {grad_norm(model):.3f}")
# Вывод в книге:
# | шаг 0: ошибка 9.065, норма g 1.000
# | шаг 1: ошибка 9.045, норма g 1.000
# | шаг 2: ошибка 9.025, норма g 1.000
# | шаг 3: ошибка 9.005, норма g 1.000
# | шаг 4: ошибка 8.985, норма g 1.000

# %% Листинг «Забытый zero_grad» — раздел «Функция шага обучения»
model.zero_grad(set_to_none=True)
for call in (1, 2):                           # zero_grad забыли
    batch_loss(model, *batch).backward()
    print(f"после backward №{call}: норма .grad {grad_norm(model):.3f}")
# Вывод в книге:
# | после backward №1: норма .grad 0.999
# | после backward №2: норма .grad 1.999

# %% Листинг «Первые шаги настоящего прогона» — раздел «Функция шага обучения»
import math
from booklm.lab import training_log

log = training_log()                      # строки журнала, словари
loss = {r["step"]: r["loss"] for r in log if "loss" in r}
val = {r["step"]: r["val_loss"] for r in log if "val_loss" in r}
print(f"ln 8192 = {math.log(8192):.3f}, отложенные до шага 0: {val[0]:.3f}")
for s in (0, 10, 20, 50, 100, 200, 300):
    print(f"шаг {s:3d}: ошибка пакета {loss[s]:.3f}")
print(f"шаг 100: отложенные {val[100]:.3f}")
# Вывод в книге:
# | ln 8192 = 9.011, отложенные до шага 0: 9.038
# | шаг   0: ошибка пакета 9.038
# | шаг  10: ошибка пакета 8.650
# | шаг  20: ошибка пакета 8.293
# | шаг  50: ошибка пакета 6.820
# | шаг 100: ошибка пакета 5.071
# | шаг 200: ошибка пакета 3.696
# | шаг 300: ошибка пакета 3.007
# | шаг 100: отложенные 5.104

# %% Листинг «Продолжения до и после ста шагов» — раздел «Функция шага обучения»
import textwrap
from booklm.lab import load_tokenizer, load_model, generate

tok = load_tokenizer()
for step in (0, 100):
    model = load_model(f"step:{step}")     # веса после step шагов
    text = generate(tok, model, "Жила-была девочка", max_new=16,
                    temperature=0)          # жадный выбор
    print(textwrap.fill(f"шаг {step}: " + " ".join(text.split()), 66))
# Вывод в книге:
# | шаг 0: Жила-была девочкаенегенегенегенегенегенегенегенегенегенеген
# | егенегенегенегенегенег
# | шаг 100: Жила-была девочка по имени Тим. Он был очень счастлив. Он
# | был очень счастлив. Он был
