"""Листинги главы 7 «Предобучение» книги «Большие языковые модели», файл 06-run-and-budget.

Код и вывод — ровно как в книге; текст книги сюда не входит. Листинги идут по порядку и
опираются друг на друга: запускайте файл целиком или ячейками сверху вниз (разметка # %%).
Проверка, что вывод совпал с книгой: python scripts/check_listings.py listings/ch07/06-run-and-budget.py
"""

# %% Листинг «Три шага AdamW: сохранённые моменты и пустые» — раздел «Возобновление без моментов»
import hashlib, io, math, os, sys, tempfile
import torch
import torch.nn as nn
import torch.nn.functional as F

def adam_trace(grads, m, v, s0, beta1=0.9, beta2=0.95):
    # одна координата; s0: сколько шагов уже сделано
    steps = []
    for s, g in enumerate(grads, start=s0 + 1):
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g * g
        m_hat, v_hat = m / (1 - beta1 ** s), v / (1 - beta2 ** s)
        steps.append(m_hat / math.sqrt(v_hat))   # в единицах eta
    return steps

grads = [0.010, -0.004, 0.008]
kept = adam_trace(grads, m=0.002, v=1e-4, s0=1234)   # непрерывно
fresh = adam_trace(grads, m=0.0, v=0.0, s0=0)        # новый AdamW
for s, g, a, b in zip((1235, 1236, 1237), grads, kept, fresh):
    print(f"шаг {s}: g = {g:+.3f}, непрерывно {a:.2f}, заново {b:.2f}")
print(f"сумма трёх шагов:     непрерывно {sum(kept):.2f}, "
      f"заново {sum(fresh):.2f}")
# Вывод в книге:
# | шаг 1235: g = +0.010, непрерывно 0.28, заново 1.00
# | шаг 1236: g = -0.004, непрерывно 0.22, заново 0.35
# | шаг 1237: g = +0.008, непрерывно 0.28, заново 0.60
# | сумма трёх шагов:     непрерывно 0.78, заново 1.95

# %% Листинг «Атомарная запись и полное сохранение» — раздел «Атомарная запись»
def write_atomic(path, data):
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
    with os.fdopen(fd, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())                     # байты на диске
    os.replace(tmp, path)                        # одно переименование

def save_state(run, path):     # run: имя -> объект со state_dict()
    state = {name: obj.state_dict() for name, obj in run.items()}
    state["rng"] = torch.get_rng_state()         # numpy, random: так же
    buffer = io.BytesIO()
    torch.save(state, buffer)
    write_atomic(path, buffer.getvalue())

def load_state(run, path, skip=()):
    state = torch.load(path, weights_only=True)  # без чужого кода
    for name, obj in run.items():
        if name not in skip:
            obj.load_state_dict(state[name])
    if "rng" not in skip:
        torch.set_rng_state(state["rng"])

model = nn.Linear(2, 2)
opt = torch.optim.AdamW(model.parameters())
with tempfile.TemporaryDirectory() as folder:
    path = os.path.join(folder, "state.pt")
    save_state({"model": model, "optimizer": opt}, path)
    print("в каталоге:", os.listdir(folder))
    state = torch.load(path, weights_only=True)
print("в сохранении:", sorted(state))
print("у оптимизатора:", sorted(state["optimizer"]))
# Вывод в книге:
# | в каталоге: ['state.pt']
# | в сохранении: ['model', 'optimizer', 'rng']
# | у оптимизатора: ['param_groups', 'state']

# %% Листинг «batch_loss и train_step из разделов 7.1 и 7.5» — раздел «Цикл обучения»
def batch_loss(model, inputs, targets, loss_mask=None):
    logits = model(inputs)
    per_pos = F.cross_entropy(logits.flatten(0, 1), targets.flatten(),
                              reduction="none")
    if loss_mask is None:
        return per_pos.mean()
    m = loss_mask.flatten().to(per_pos.dtype)
    return (per_pos * m).sum() / m.sum()

def train_step(model, batch, opt, sched=None, clip=None, scaler=None,
               amp=None):
    opt.zero_grad(set_to_none=True)
    device = next(model.parameters()).device.type
    with torch.autocast(device, dtype=amp, enabled=amp is not None):
        loss = batch_loss(model, *batch)
    S = 1.0 if scaler is None else scaler.S
    (loss * S).backward()
    params = [p for p in model.parameters() if p.grad is not None]
    found_inf = scaler is not None and scaler.unscale_and_check(params)
    row = {"loss": loss.item(), "lr": opt.param_groups[0]["lr"],
           "skipped": found_inf}
    if not found_inf:
        if clip is not None:
            row["grad_norm"] = torch.nn.utils.clip_grad_norm_(
                params, clip).item()
        opt.step()
        if sched is not None:
            sched.step()
    if scaler is not None:
        scaler.update(found_inf)
    return row

# %% Листинг «Загрузчик и цикл обучения» — раздел «Цикл обучения»
import numpy as np
sys.path.insert(0, ".")                          # корень репозитория
from booklm.model import BookLM, BookLMConfig
from booklm.lab import load_tokenizer

data = np.memmap("models/booklm/data/train.bin", dtype=np.uint16, mode="r")

def get_batch(B=8, T=128):                       # как batch() в train.py
    starts = torch.randint(len(data) - T - 1, (B,)).tolist()
    w = torch.stack([torch.from_numpy(data[i:i + T + 1].astype(np.int64))
                     for i in starts])
    return w[:, :-1], w[:, 1:]                   # X и Y со сдвигом

def make_run(seed):
    torch.manual_seed(seed)
    model = BookLM(BookLMConfig())
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, betas=(0.9, 0.95))
    warmup = lambda i: min(1.0, (i + 1) / 4)    # 4 шага разогрева
    sched = torch.optim.lr_scheduler.LambdaLR(opt, warmup)
    return {"model": model, "optimizer": opt, "scheduler": sched}

def train(run, n, folder=None, save_every=None):
    log = []
    for _ in range(n):
        row = train_step(run["model"], get_batch(), run["optimizer"],
                         run["scheduler"], clip=1.0)
        row["step"] = run["scheduler"].last_epoch
        log.append(row)
        if save_every and row["step"] % save_every == 0:
            name = f"step{row['step']:06d}.pt"
            save_state(run, os.path.join(folder, name))
    return log

# %% Листинг «Двести шагов на сказках» — раздел «Цикл обучения»
@torch.no_grad()                                 # граф не записываем
def greedy(model, tok, text, n=20):
    ids = [1] + tok.encode(text).ids             # 1 = <bos>
    for _ in range(n):
        logits = model(torch.tensor([ids]))[0, -1]   # последняя позиция
        ids.append(int(logits.argmax()))         # самый вероятный токен
    return tok.decode(ids[1:])

tok = load_tokenizer()
run = make_run(seed=0)
print("шаг   0:", greedy(run["model"], tok, "Жила-была девочка", n=8))
with tempfile.TemporaryDirectory() as folder:
    log = train(run, 200, folder, save_every=100)
    print("сохранения:", sorted(os.listdir(folder)))
for row in log[:1] + log[49::50]:
    print(f"шаг {row['step']:3d}: ошибка {row['loss']:.3f}, "
          f"норма до ограничения {row['grad_norm']:.2f}")
print("шаг 200:", greedy(run["model"], tok, "Жила-была девочка", n=11))
# Вывод в книге:
# | шаг   0: Жила-была девочка име име име име име име име име
# | сохранения: ['step000100.pt', 'step000200.pt']
# | шаг   1: ошибка 9.080, норма до ограничения 3.04
# | шаг  50: ошибка 6.428, норма до ограничения 0.82
# | шаг 100: ошибка 5.722, норма до ограничения 0.79
# | шаг 150: ошибка 5.535, норма до ограничения 0.71
# | шаг 200: ошибка 5.104, норма до ограничения 0.79
# | шаг 200: Жила-была девочка. Они счастливы, что не не играли. Они играли

# %% Листинг «Побитное возобновление» — раздел «Побитное возобновление»
def resumed(skip=()):
    first = make_run(seed=0)
    train(first, 5)
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "state.pt")
        save_state(first, path)                  # и процесс упал
        second = make_run(seed=123)              # новые объекты
        load_state(second, path, skip)
    return train(second, 5), second["model"]

whole = make_run(seed=0)
log = train(whole, 10)
print("шаги 6-10 подряд:", [round(r["loss"], 4) for r in log[5:]])
for skip in [(), ("optimizer",), ("scheduler",), ("rng",)]:
    tail, model = resumed(skip)
    marks = "".join("=" if a["loss"] == b["loss"] else "x"
                    for a, b in zip(tail, log[5:]))
    same = all(torch.equal(p, q) for p, q in
               zip(model.parameters(), whole["model"].parameters()))
    print(f"забыли {skip[0] if skip else 'ничего':9}: шаги 6-10 {marks}, "
          f"веса равны: {same}")
lrs = lambda rows: [f"{r['lr']:.2e}" for r in rows[:3]]
print("скорость 6-8, непрерывно:     ", lrs(log[5:]))
forgot = resumed(("scheduler",))[0]
print("скорость 6-8, без планировщика:", lrs(forgot))
# Вывод в книге:
# | шаги 6-10 подряд: [7.2632, 7.239, 6.9825, 6.9704, 7.3445]
# | забыли ничего   : шаги 6-10 =====, веса равны: True
# | забыли optimizer: шаги 6-10 =xxxx, веса равны: False
# | забыли scheduler: шаги 6-10 ==xxx, веса равны: False
# | забыли rng      : шаги 6-10 xxxxx, веса равны: False
# | скорость 6-8, непрерывно:      ['3.00e-03', '3.00e-03', '3.00e-03']
# | скорость 6-8, без планировщика: ['3.00e-03', '1.50e-03', '2.25e-03']

# %% Листинг «Выпускаемые веса и их хеш» — раздел «Выпускаемые веса»
def release(model):              # BF16, связанный тензор: один раз
    return {name: p.detach().to(torch.bfloat16)
            for name, p in model.named_parameters()}

def sha256(weights):
    h = hashlib.sha256()
    for name in sorted(weights):                 # порядок не важен
        raw = weights[name].view(torch.int16).numpy().tobytes()
        h.update(name.encode() + raw)
    return h.hexdigest()

weights = release(whole["model"])
parent = release(make_run(seed=0)["model"])      # от чего начинали
size = sum(t.numel() * t.element_size() for t in weights.values())
print(f"тензоров {len(weights)}, байт {size:,}")
print("sha256 весов:   ", sha256(weights)[:16])
print("sha256 родителя:", sha256(parent)[:16])
# Вывод в книге:
# | тензоров 56, байт 13,244,928
# | sha256 весов:    b42e8abc23b3ff8e
# | sha256 родителя: 73bea41989733acc

# %% Листинг «Валидационная ошибка, бит на байт и сравнение сохранений» — раздел «Валидационная ошибка и перплексия»
@torch.no_grad()
def evaluate(model, windows):                    # windows: [n, T+1]
    x, y = windows[:, :-1], windows[:, 1:]
    nll = F.cross_entropy(model(x).flatten(0, 1), y.flatten(),
                          reduction="none").view(y.shape)
    return nll.mean(1)                           # ошибка каждого окна

def bits_per_byte(loss_per_token, n_tokens, n_bytes):
    return loss_per_token * n_tokens / (n_bytes * math.log(2))

def is_improvement(old, new):        # ошибки одних и тех же окон
    d = old - new
    return bool(d.mean() > 2 * d.std() / len(d) ** 0.5)

text = "Мама мыла раму"
print(len(text), "символов,", len(text.encode("utf-8")), "байт")
print(f"BookLM, 3 токена:  {bits_per_byte(4.0, 3, 26):.4f} бит на байт")
print(f"байты, 26 токенов: {bits_per_byte(12 / 26, 26, 26):.4f} "
      "бит на байт")
old = torch.tensor([3.30, 3.52, 3.41, 3.42])     # четыре окна
new = torch.tensor([3.29, 3.50, 3.40, 3.41])
sem = lambda l: (l.std() / len(l) ** 0.5).item()
print(f"среднее {old.mean():.4f} -> {new.mean():.4f}; sigma каждого "
      f"{sem(old):.4f}, sigma разности {sem(old - new):.4f}")
print("улучшение:", is_improvement(old, new))

flat = nn.Sequential(nn.Embedding(16, 8), nn.Linear(8, 16, bias=False))
nn.init.zeros_(flat[1].weight)                   # все логиты равны
loss = evaluate(flat, torch.randint(0, 16, (4, 9))).mean().item()
print(f"равные логиты: ошибка {loss:.4f}, ln 16 = {math.log(16):.4f}, "
      f"перплексия {math.exp(loss):.1f}")
# Вывод в книге:
# | 14 символов, 26 байт
# | BookLM, 3 токена:  0.6659 бит на байт
# | байты, 26 токенов: 0.6659 бит на байт
# | среднее 3.4125 -> 3.4000; sigma каждого 0.0450, sigma разности 0.0025
# | улучшение: True
# | равные логиты: ошибка 2.7726, ln 16 = 2.7726, перплексия 16.0

# %% Листинг «Бюджет BookLM по правилу Chinchilla» — раздел «Бюджет вычислений»
N = 6_622_464                          # параметры BookLM
D = 20 * N                             # правило Chinchilla
C = 6 * N * D                          # 6N на токен, D токенов
steps = math.ceil(D / (64 * 512))      # 32 768 токенов на шаг
print(f"N = {N:,}, D = {D:,}, C = {C:.3g} операций")
print(f"шагов: {D / (64 * 512):.2f} -> {steps}")
for tokens_per_second in (6_500, 10_000):          # CPU, измерено
    print(f"CPU, {tokens_per_second:,} ток./с: "
          f"{D / tokens_per_second / 3600:.1f} ч")
useful = 100e12 * 0.40                 # пик ускорителя * MFU
print(f"ускоритель: {C / useful:.0f} с, "
      f"{useful / (6 * N):,.0f} ток./с")
# Вывод в книге:
# | N = 6,622,464, D = 132,449,280, C = 5.26e+15 операций
# | шагов: 4042.03 -> 4043
# | CPU, 6,500 ток./с: 5.7 ч
# | CPU, 10,000 ток./с: 3.7 ч
# | ускоритель: 132 с, 1,006,675 ток./с

# %% Листинг «Прогон BookLM в числах» — раздел «Бюджет вычислений»
import json
from booklm.lab import MODELS, training_log

run_info = json.load(open(MODELS / "train_run.json"))  # run.json прогона
N, steps = 6_622_464, run_info["steps"]
D = steps * run_info["tokens_per_step"]                 # увидено токенов
corpus = run_info["train_tokens"]
log = training_log()
last = [r for r in log if "loss" in r][-1]
speed = last["tok_per_s"]                               # с начала прогона
print(f"корпус {corpus:,} токенов, D = {D:,}")
print(f"эпох {D / corpus:.2f}, D/N = {D / N:.1f}")
print(f"время {last['tokens'] / speed / 60:.1f} мин, {speed:,.0f} ток./с, "
      f"MFU {6 * N * speed / 125e12:.1%}")
train = {r["step"]: r["loss"] for r in log if "loss" in r}
val = {r["step"]: r["val_loss"] for r in log if "val_loss" in r}
for s in (250, 500, 1000, 2000, 3000):
    near = [train[t] for t in train if abs(t - s) <= 50]  # сглаживаем
    tr = sum(near) / len(near)
    print(f"шаг {s:4d}: обучение {tr:.3f}, валидация {val[s]:.3f}, "
          f"разрыв {val[s] - tr:+.3f}")
# Вывод в книге:
# | корпус 12,507,970 токенов, D = 99,975,168
# | эпох 7.99, D/N = 15.1
# | время 25.5 мин, 65,312 ток./с, MFU 2.1%
# | шаг  250: обучение 3.292, валидация 3.270, разрыв -0.022
# | шаг  500: обучение 2.496, валидация 2.551, разрыв +0.055
# | шаг 1000: обучение 2.068, валидация 2.196, разрыв +0.127
# | шаг 2000: обучение 1.742, валидация 1.995, разрыв +0.252
# | шаг 3000: обучение 1.601, валидация 1.934, разрыв +0.333

# %% Листинг «Разрыв на сохранениях и парное сравнение» — раздел «Бюджет вычислений»
from booklm.lab import load_model, val_tokens
from booklm.experiments.ch07 import window_losses   # ошибка каждого окна

tr_data = torch.from_numpy(np.fromfile(MODELS / "data" / "train.bin",
                                       dtype=np.uint16).astype(np.int64))
va_data = val_tokens(10**9)
gen, n = torch.Generator().manual_seed(0), 64
ts = torch.randint(len(tr_data) - 513, (n,), generator=gen).tolist()
vs = torch.randint(len(va_data) - 513, (n,), generator=gen).tolist()
sem = lambda l: (l.std() / len(l) ** 0.5).item()
def per_window(model, data, starts):            # по 16 окон за проход
    return torch.cat([window_losses(model, data, starts[i:i + 16])
                      for i in range(0, len(starts), 16)])

val_by = {}
for kind in ("step:1525", "base"):
    model = load_model(kind)
    lt, lv = per_window(model, tr_data, ts), per_window(model, va_data, vs)
    val_by[kind] = lv
    gap = (lv.mean() - lt.mean()).item()
    sigma = math.hypot(sem(lt), sem(lv))           # sigma разрыва
    print(f"{kind:<9}: обучение {lt.mean():.3f}, "
          f"валидация {lv.mean():.3f}, "
          f"разрыв {gap:.3f} = {gap / sigma:.1f} sigma")
d = val_by["step:1525"] - val_by["base"]
apart = math.hypot(sem(val_by["step:1525"]), sem(val_by["base"]))
print(f"base лучше step:1525 на {d.mean():.3f}")
print(f"sigma по отдельности {apart:.3f}, sigma разности {sem(d):.4f}")
# Вывод в книге:
# | step:1525: обучение 1.888, валидация 2.084, разрыв 0.196 = 4.3 sigma
# | base     : обучение 1.620, валидация 1.947, разрыв 0.327 = 7.6 sigma
# | base лучше step:1525 на 0.137
# | sigma по отдельности 0.040, sigma разности 0.0055

# %% Листинг «Контрольные продолжения на сохранениях» — раздел «Бюджет вычислений»
import textwrap

for kind in ("step:762", "base"):
    text = greedy(load_model(kind), tok, "Жила-была девочка", n=30)
    print(textwrap.fill(f"{kind}: " + " ".join(text.split()), width=74))
# Вывод в книге:
# | step:762: Жила-была девочка по имени Джейн. Она была очень любопытной и
# | хотела исследовать мир. Однажды она решила исследовать лес. Она решила
# | исследовать лес, поэтому решила исследовать лес
# | base: Жила-была девочка по имени Люси. Она была очень любопытной и любила
# | исследовать. Однажды она решила прогуляться в парк. Когда она шла, она
# | увидела что-то

# %% Листинг «Карта внимания до и после обучения» — раздел «Бюджет вычислений»
from booklm.lab import attention             # доли всех слоёв и голов

phrase = "Мама мыла раму, а потом она пошла спать."
final = load_model("base")                   # итоговые веса
pieces, att = attention(tok, final, phrase)  # att: [слои, головы, T, T]
she, mom = pieces.index(" она"), pieces.index("Мама")
to_mom = att[:, :, she, mom]                 # «она» -> «Мама», 48 карт
layer, head = divmod(int(to_mom.argmax()), att.size(1))
print(f"слой {layer}, голова {head}: «она» -> «Мама» {to_mom.max():.2f}")
_, att0 = attention(tok, load_model("init"), phrase)   # шаг 0
for name, a in (("после обучения:", att), ("до обучения:   ", att0)):
    row = a[layer, head, she, :she + 1]      # строка «она» до неё самой
    print(name, " ".join(f"{x:.2f}" for x in row))
for other in ("Кот увидел девочку, и она засмеялась.",
              "Девочка увидела кота, и он убежал."):
    p, a = attention(tok, final, other)
    word = " она" if " она" in p else " он"
    row = a[layer, head, p.index(word)]
    print(f"{other} {word.strip()} -> {p[int(row.argmax())].strip()} "
          f"{row.max():.2f}")
# Вывод в книге:
# | слой 2, голова 5: «она» -> «Мама» 0.93
# | после обучения: 0.02 0.93 0.00 0.00 0.00 0.00 0.00 0.01 0.00 0.02
# | до обучения:    0.09 0.09 0.10 0.10 0.11 0.10 0.10 0.10 0.11 0.11
# | Кот увидел девочку, и она засмеялась. она -> девочку 0.90
# | Девочка увидела кота, и он убежал. он -> кота 0.64

# %% Листинг «Размер выхода слоя и его сдвиг за шаг» — раздел «Перенос гиперпараметров по ширине»
rms = lambda v: v.pow(2).mean().sqrt().item()     # среднеквадратичное

def layer_scales(d, std, lr):
    # слой d -> d: размер W x при старте и его сдвиг за первый шаг Adam
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(d, generator=gen)              # вход, RMS около 1
    delta = torch.randn(d, generator=gen)          # сигнал ошибки на выходе
    W = std * torch.randn(d, d, generator=gen)
    step = lr * torch.outer(delta, x).sign()       # eta * знак градиента
    return rms(W @ x), rms(step @ x)

for d in (64, 256, 1024):
    out, shift = layer_scales(d, std=0.02, lr=1e-3)
    print(f"SP,  d = {d:4d}: выход {out:.3f}, сдвиг за шаг {shift:.3f}")
# Вывод в книге:
# | SP,  d =   64: выход 0.177, сдвиг за шаг 0.053
# | SP,  d =  256: выход 0.320, сдвиг за шаг 0.190
# | SP,  d = 1024: выход 0.665, сдвиг за шаг 0.829

# %% Листинг «Тот же слой в» — раздел «Перенос гиперпараметров по ширине»
for d in (64, 256, 1024):
    out, shift = layer_scales(d, std=0.02 * math.sqrt(64 / d),
                              lr=1e-3 * 64 / d)  # формула pt-mup, d0 = 64
    print(f"muP, d = {d:4d}: выход {out:.3f}, сдвиг за шаг {shift:.3f}")
# Вывод в книге:
# | muP, d =   64: выход 0.177, сдвиг за шаг 0.053
# | muP, d =  256: выход 0.160, сдвиг за шаг 0.048
# | muP, d = 1024: выход 0.166, сдвиг за шаг 0.052

# %% Листинг «Промежуточный этап BookLM» — раздел «Промежуточный этап обучения»
# Проверка этот листинг не запускает: ему нужны сеть, видеокарта, сервер или внешний файл.
# import textwrap
# from booklm.lab import load_model
# from booklm.experiments.ch07_muon_midtraining import (train_stream, request_data,
#     window_sampler, train_steps, story_part_loss, val_items)
#
# train, (req, held) = train_stream(), request_data(tok)
# stories = window_sampler(train, seed=11)
# requests = window_sampler(req, seed=12)
#
# def mixed():                      # 12 окон сказок + 4 окна с просьбами
#     x1, y1 = stories(12)
#     x2, y2 = requests(4)
#     return torch.cat([x1, x2]), torch.cat([y1, y2])
#
# prompt, base = "Просьба: Напиши сказку про ёжика и яблоко.\nСказка:", None
# for name, sample in (("base", None), ("без пар", stories),
#                      ("с парами", mixed)):
#     model = load_model("base").train()
#     if sample is not None:        # 200 шагов, скорость 0 -> 2e-4 -> 0
#         mats = [p for p in model.parameters() if p.dim() >= 2]
#         gains = [p for p in model.parameters() if p.dim() < 2]
#         opt = torch.optim.AdamW([{"params": mats, "weight_decay": 0.1},
#                                  {"params": gains, "weight_decay": 0.0}],
#                                 lr=2e-4, betas=(0.9, 0.95))
#         for g in opt.param_groups:
#             g["base_lr"] = 2e-4
#         warm = lambda s: min((s + 1) / 20, 1 - s / 200)  # прогрев
#         train_steps(model, [opt], warm, 200, sample)
#     lq = story_part_loss(model, held)             # сказки по просьбе
#     lg = story_part_loss(model, val_items())      # обычные сказки
#     if base is None:
#         base = (lq, lg)
#         print(f"{name:8s}: просьбы {lq.mean():.3f}, "
#               f"сказки {lg.mean():.3f}")
#     else:
#         dq, dg = lq - base[0], lg - base[1]
#         print(f"{name:8s}: просьбы {lq.mean():.3f} ({dq.mean():+.3f} ± "
#               f"{sem(dq):.3f}), сказки {lg.mean():.3f} "
#               f"({dg.mean():+.3f} ± {sem(dg):.3f})")
#     if name != "без пар":
#         text = greedy(model.eval(), tok, prompt, n=40).split("Сказка:")[1]
#         text = f"{name}: " + " ".join(text.split())
#         print(textwrap.fill(text, width=74))
