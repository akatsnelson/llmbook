"""Глава 7, разделы 7.4–7.6 (оптимизаторы, 16-битные числа, длинный прогон): проверки к листингам книги.

В книге листинги печатают числа, а здесь те же расчёты проверены утверждениями assert:
ручные траектории SGD и момента, первый шаг Adam, затухание весов, группы параметров BookLM,
расписание скорости, выброс градиента и ограничение нормы, форматы чисел, масштабировщик ошибки,
полный шаг смешанной точности, накопление градиента, возобновление прогона, выпускаемые веса,
валидационная ошибка, бит на байт и бюджет Chinchilla.

Запуск из корня репозитория: python labs/ch07/optim_precision_run_checks.py
"""

# ===== 04-optimizers.tex ===============================================

# --- SGD и момент на овраге ---
import math
import torch

def valley_grad(theta):          # f = (x^2 + 25 y^2) / 2;  theta: [2]
    return theta * torch.tensor([1.0, 25.0], dtype=theta.dtype)

def run(lr, beta=0.0, steps=30):  # beta = 0 — обычный SGD
    theta = torch.tensor([1.0, 1.0], dtype=torch.float64)
    u, path = torch.zeros(2, dtype=torch.float64), [theta]
    for _ in range(steps):
        u = beta * u + valley_grad(theta)  # накопленная скорость
        theta = theta - lr * u
        path.append(theta)
    return path

def close(a, b, atol=1e-4):
    b = torch.tensor(b, dtype=torch.float64)
    return torch.allclose(a, b, atol=atol, rtol=0)

sgd_path, mom_path = run(0.07), run(0.01, beta=0.9)
sgd_hand = [[0.93, -0.75], [0.8649, 0.5625], [0.8044, -0.4219]]
mom_hand = [[0.99, 0.75], [0.9711, 0.3375], [0.9444, -0.1181]]
for s in (1, 2, 3):              # ручной расчёт SGD и момента
    assert close(sgd_path[s], sgd_hand[s - 1])
    assert close(mom_path[s], mom_hand[s - 1])
assert close(sgd_path[30], [0.1134, 0.0002])
# момент по x ушёл дальше, чем SGD с тем же lr
assert abs(mom_path[30][0]) < 0.2 and run(0.01)[30][0] > 0.7
# поломка: lr больше 2/25, качели по y растут
assert abs(run(0.085)[30][1]) > 1 and abs(run(0.1)[30][1]) > 1e3

# --- Adam и AdamW руками ---
def adamw_step(theta, g, m, v, s, lr, beta1=0.9, beta2=0.95,
               eps=1e-8, wd=0.0, decoupled=True):
    # theta, g, m, v: одной формы; s: номер шага, начиная с 1
    if not decoupled:
        g = g + wd * theta            # штраф в ошибке: через моменты
    m = beta1 * m + (1 - beta1) * g
    v = beta2 * v + (1 - beta2) * g * g
    m_hat = m / (1 - beta1 ** s)      # поправка смещения
    v_hat = v / (1 - beta2 ** s)
    update = m_hat / (v_hat.sqrt() + eps)
    if decoupled:
        update = update + wd * theta  # затухание мимо нормировки
    return theta - lr * update, m, v

f64 = dict(dtype=torch.float64)
zero2, zero1 = torch.zeros(2, **f64), torch.zeros(1, **f64)
for scale in (1000.0, 0.001):         # первый шаг = lr * sign(g)
    g = torch.tensor([scale, -scale], **f64)
    new, _, _ = adamw_step(zero2, g, zero2, zero2, 1, lr=1e-3,
                           beta2=0.999)
    assert torch.allclose(new, -1e-3 * g.sign(), rtol=1e-4, atol=0)
two = torch.tensor([2.0], **f64)      # параметр с нулевым градиентом
for wd, decoupled, expected in [(0.1, True, 1.998),
                                (0.1, False, 1.990),
                                (0.01, False, 1.990)]:  # wd не важен
    new, _, _ = adamw_step(two, zero1, zero1, zero1, 1, lr=0.01,
                           wd=wd, decoupled=decoupled)
    assert abs(new.item() - expected) < 1e-6

# --- Группы параметров BookLM ---
import torch.nn as nn

class BookLMShapes(nn.Module):
    # только параметры по паспорту главы 4, прямого прохода нет
    def __init__(self, V=8192, d=256, layers=6, d_kv=64, d_ff=768):
        super().__init__()
        lin = lambda i, o: nn.Linear(i, o, bias=False)
        self.tokens, self.output = nn.Embedding(V, d), lin(d, V)
        self.output.weight = self.tokens.weight  # связанные веса
        self.blocks = nn.ModuleList(nn.ModuleDict(dict(
            norm1=nn.RMSNorm(d), q=lin(d, d), k=lin(d, d_kv),
            v=lin(d, d_kv), out=lin(d, d), norm2=nn.RMSNorm(d),
            gate=lin(d, d_ff), up=lin(d, d_ff), down=lin(d_ff, d)))
            for _ in range(layers))
        self.norm = nn.RMSNorm(d)

def param_groups(model, weight_decay):
    decay, no_decay = [], []
    for _, p in model.named_parameters():  # общий вес — один раз
        (decay if p.ndim >= 2 else no_decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]

groups = param_groups(BookLMShapes(), weight_decay=0.1)
sizes = [sum(p.numel() for p in gr["params"]) for gr in groups]
assert sizes == [6_619_136, 3_328] and sum(sizes) == 6_622_464
torch.optim.AdamW(groups, lr=3e-4, betas=(0.9, 0.95), eps=1e-8)

# --- Разогрев и косинус ---
def lr_at(s, warmup, total, lr_max, lr_min):
    # s — номер шага оптимизатора, начиная с 1
    if s <= warmup:
        return lr_max * s / warmup
    progress = (s - warmup) / (total - warmup)
    cos = math.cos(math.pi * progress)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + cos)

sched = dict(warmup=200, total=2000, lr_max=3e-4, lr_min=3e-5)
for s, expected in [(100, 1.5e-4), (200, 3e-4),
                    (1100, 1.65e-4), (2000, 3e-5)]:
    assert math.isclose(lr_at(s, **sched), expected, rel_tol=1e-9)
bk = dict(sched, total=4043)          # настоящий прогон BookLM
assert abs(lr_at(2121, **bk) - 1.65e-4) < 1e-7
assert abs(lr_at(1235, **bk) - 2.545e-4) < 1e-7    # после падения
assert math.isclose(lr_at(1, **bk), 1.5e-6)        # планировщик заново

# --- Масштаб ошибки и первый шаг ---
def first_move(make_opt, scale):      # на сколько сдвинулся (1, 1)
    p = torch.nn.Parameter(torch.ones(2, dtype=torch.float64))
    opt = make_opt([p])
    (scale * 0.5 * (p[0] ** 2 + 25 * p[1] ** 2)).backward()
    opt.step()
    return 1 - p.detach()

sgd = lambda ps: torch.optim.SGD(ps, lr=0.001)
adamw = lambda ps: torch.optim.AdamW(ps, lr=0.01, weight_decay=0.0)
for scale in (100.0, 0.01):
    assert torch.allclose(first_move(sgd, scale),
                          scale * first_move(sgd, 1.0))   # в scale раз
    assert torch.allclose(first_move(adamw, scale),
                          first_move(adamw, 1.0))          # тот же шаг

# ===== 05-precision-stability.tex ======================================

# --- Форматы на ладони ---
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

f32, f16, b16 = (torch.finfo(t) for t in
                 (torch.float32, torch.float16, torch.bfloat16))
assert (f32.eps, f16.eps, b16.eps) == (2**-23, 2**-10, 2**-7)
assert (f16.max, f16.tiny) == (65504.0, 2**-14)
assert b16.tiny == f32.tiny           # снизу BF16 там же, где FP32

def bf16_bits(x):                     # знак | порядок | мантисса
    t = torch.tensor(x, dtype=torch.bfloat16)
    s = format(t.view(torch.int16).item() & 0xFFFF, "016b")
    return f"{s[0]} {s[1:9]} {s[9:]}"

assert bf16_bits(0.1) == "0 01111011 1001101"
assert torch.tensor(0.1, dtype=torch.bfloat16).item() == 0.10009765625

one = torch.tensor(1.0)
assert (one + 1e-3).to(torch.bfloat16).item() == 1.0   # утонуло
assert (one + 1e-3).to(torch.float16).item() == 1.0009765625
to16 = lambda v: torch.tensor(v).to(torch.float16).item()
assert to16(70000.0) == float("inf") and to16(1e-8) == 0.0
assert torch.tensor(1e-8).to(torch.bfloat16).item() != 0.0

# сто шагов по eta_min = 3e-5: BF16 стоит, FP32 уходит на 0,003
w16 = torch.tensor(0.02, dtype=torch.bfloat16)
start, w32 = w16.item(), torch.tensor(0.02)
for _ in range(100):
    w16 = w16 - torch.tensor(3e-5, dtype=torch.bfloat16)
    w32 = w32 - 3e-5
assert w16.item() == start and abs(w32.item() - 0.017) < 1e-6

# --- Масштабировщик ошибки и пропуск шага ---
class DynamicScaler:         # те же поля, что у torch.amp.GradScaler
    def __init__(self, init_scale=2.0**16, growth_interval=2000):
        self.S, self.interval, self.good = init_scale, growth_interval, 0

    def unscale_and_check(self, params):   # True: есть inf или NaN
        found_inf = False
        for p in params:
            p.grad.mul_(1.0 / self.S)      # снимаем масштаб во FP32
            found_inf |= not torch.isfinite(p.grad).all().item()
        return found_inf

    def update(self, found_inf):           # правило масштабировщика
        self.good = 0 if found_inf else self.good + 1
        if found_inf:
            self.S /= 2
        elif self.good == self.interval:
            self.S, self.good = self.S * 2, 0

def scaled_step(params, opt, sched, scaler, loss_fn):
    opt.zero_grad(set_to_none=True)
    (loss_fn() * scaler.S).backward()
    found_inf = scaler.unscale_and_check(params)
    if not found_inf:
        opt.step()                    # шаг и оба счётчика: только тут
        sched.step()
    scaler.update(found_inf)
    return found_inf

w = torch.nn.Parameter(torch.ones(2))
opt = torch.optim.AdamW([w], lr=0.1)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: 1.0)
scaler = DynamicScaler()
assert scaled_step([w], opt, sched, scaler, lambda: (1e35 * w).sum())
assert torch.equal(w.detach(), torch.ones(2)) and opt.state[w] == {}
assert sched.last_epoch == 0 and scaler.S == 2.0**15   # S вдвое меньше

# --- Выброс градиента в AdamW ---
def adam_moves(grads, beta2, beta1=0.9, m=1.0, v=1.0):
    # одна координата спокойного прогона: m = v = 1, шаг s большой,
    # поправки смещения равны 1; возвращает шаги в единицах eta
    moves = []
    for g in grads:
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g * g
        moves.append(m / math.sqrt(v))
    return moves

calm = adam_moves([1.0] * 100, beta2=0.95)
spike = [-100.0] + [1.0] * 99                     # один плохой пакет
slow = adam_moves(spike, beta2=0.999)             # умолчание PyTorch
bk = adam_moves(spike, beta2=0.95)                # BookLM
assert abs(sum(calm) - 100) < 1e-9                # без выброса: 100 eta
assert round(slow[0], 2) == -2.74 and round(bk[0], 2) == -0.41
backwards = lambda moves: sum(step < 0 for step in moves)
assert backwards(slow) == backwards(bk) == 22     # m помнит выброс
assert round(bk[40], 2) == 0.11                   # v раздуто
assert round(sum(slow), 2) == 0.26 and round(sum(bk), 1) == 13.7
clipped = adam_moves([-1.0] + [1.0] * 99, beta2=0.95)
assert round(sum(clipped)) == 98                  # выброс укоротили до 1

# --- Полный шаг смешанной точности ---
def clip_by_global_norm(grads, c):
    norm = math.sqrt(sum((g.double() ** 2).sum().item() for g in grads))
    return [g * min(1.0, c / norm) for g in grads], norm

g = torch.tensor([3.0, 4.0])
(clipped,), norm = clip_by_global_norm([g], c=1.0)
assert norm == 5.0 and torch.allclose(clipped, torch.tensor([0.6, 0.8]))
(early,), logged = clip_by_global_norm([g * 1024], c=1.0)  # поломка
assert logged == 5120.0                      # в журнале S * |g|
assert abs((early / 1024).norm().item() - 1 / 1024) < 1e-9

def full_step(params, opt, sched, scaler, loss_fn, c=1.0):
    opt.zero_grad(set_to_none=True)
    (loss_fn() * scaler.S).backward()                # 1. от S*L
    found_inf = scaler.unscale_and_check(params)     # 2-3. /S, inf?
    norm = None
    if not found_inf:
        norm = torch.nn.utils.clip_grad_norm_(params, c).item()  # 4.
        opt.step()                                   # 5.
        sched.step()
    scaler.update(found_inf)
    return norm                       # норма до ограничения: в журнал

norm = full_step([w], opt, sched, scaler, lambda: (3 * w).sum())
assert abs(norm - 18 ** 0.5) < 1e-5 and sched.last_epoch == 1

# --- Накопление градиента равно полному пакету ---
IGNORE = -100                          # цель заполнителя: маска 0
torch.manual_seed(1)
ids = torch.randint(0, 16, (2, 4))     # 2 окна по T+1 = 4
inputs, targets = ids[:, :-1].clone(), ids[:, 1:].clone()
targets[1, 1:] = IGNORE                # во втором окне один токен
counts = (targets != IGNORE).sum(1)
assert counts.tolist() == [3, 1]

def accumulate(model, denom):          # denom(j): на что делить сумму
    model.zero_grad(set_to_none=True)
    for j in range(2):                 # k = 2 микропакета
        logits = model(inputs[j:j + 1]).flatten(0, 1)
        loss = F.cross_entropy(logits, targets[j],
                               ignore_index=IGNORE, reduction="sum")
        (loss / denom(j)).backward()
    return [p.grad.clone() for p in model.parameters()]

torch.manual_seed(0)                   # любая модель: таблица + выход
model = nn.Sequential(nn.Embedding(16, 8), nn.Linear(8, 16))
model.zero_grad(set_to_none=True)
F.cross_entropy(model(inputs).flatten(0, 1), targets.flatten(),
                ignore_index=IGNORE).backward()      # целый пакет
full = [p.grad.clone() for p in model.parameters()]
total = counts.sum()                   # знаменатель известен заранее
right = accumulate(model, lambda j: total)
wrong = accumulate(model, lambda j: counts[j] * 2)   # среднее средних
assert all(torch.allclose(a, b, atol=1e-6) for a, b in zip(full, right))
assert max((a - b).abs().max().item() for a, b in zip(full, wrong)) > 1e-3
assert (total / (2 * counts[1])).item() == 2.0  # вес одиночки вдвое выше
assert (3.0 + 3.0) / 4 == 1.5 and (3.0 / 3 + 3.0 / 1) / 2 == 2.0

# ===== 06-run-and-budget.tex ===========================================

# --- Три шага AdamW: сохранённые моменты против пустых ---
import math

def adam_trace(grads, m, v, s0, beta1=0.9, beta2=0.95):
    # одна координата; s0 — сколько шагов уже сделано
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
assert [round(a, 2) for a in kept] == [0.28, 0.22, 0.28]
assert [round(a, 2) for a in fresh] == [1.00, 0.35, 0.60]
assert round(sum(kept), 2) == 0.78 and round(sum(fresh), 2) == 1.95

# --- Атомарная запись и полное сохранение ---
import io, math, os, tempfile
import torch
import torch.nn as nn
import torch.nn.functional as F

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

with tempfile.TemporaryDirectory() as folder:
    save_state({"model": nn.Linear(2, 2)}, os.path.join(folder, "s.pt"))
    assert os.listdir(folder) == ["s.pt"]        # временного файла нет

# --- Побитное возобновление ---
def make_run(seed):
    torch.manual_seed(seed)
    model = nn.Linear(8, 8)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-2)
    warmup = lambda i: min(1.0, (i + 1) / 4)    # 4 шага разогрева
    sched = torch.optim.lr_scheduler.LambdaLR(opt, warmup)
    return {"model": model, "optimizer": opt, "scheduler": sched}

def run_step(run):              # данные и маски берут из генератора
    x = torch.randn(4, 8)
    loss = F.mse_loss(F.dropout(run["model"](x), p=0.1), x.roll(1, 1))
    loss.backward()
    run["optimizer"].step()
    run["scheduler"].step()
    run["optimizer"].zero_grad(set_to_none=True)
    return loss.item()

def resumed(skip=()):
    first = make_run(seed=0)
    for _ in range(5):
        run_step(first)
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "state.pt")
        save_state(first, path)                  # и процесс упал
        second = make_run(seed=123)              # новые объекты
        load_state(second, path, skip)
    return [run_step(second) for _ in range(5)], second["model"]

whole = make_run(seed=0)
losses = [run_step(whole) for _ in range(10)]
tail, model = resumed()
assert tail == losses[5:]                        # тот же прогон до бита
assert torch.equal(model.weight, whole["model"].weight)
for skip in [("optimizer",), ("scheduler",), ("rng",)]:
    assert not torch.equal(resumed(skip)[1].weight, whole["model"].weight)
no_sched, _ = resumed(("scheduler",))            # шаг 6 верный, 7 нет
assert no_sched[:2] == losses[5:7] and no_sched[2] != losses[7]

# --- Выпускаемые веса и манифест ---
import hashlib, json

def release(model):            # BF16, связанный тензор — один раз
    return {name: p.detach().to(torch.bfloat16)
            for name, p in model.named_parameters()}

def sha256(weights):
    h = hashlib.sha256()
    for name in sorted(weights):                 # порядок не важен
        t = weights[name]
        h.update(name.encode() + t.view(torch.int16).numpy().tobytes())
    return h.hexdigest()

def check_parent(manifest, parent_weights, parent_id="tiny-init"):
    if sha256(parent_weights) != manifest["parents"][parent_id]:
        raise ValueError("родитель не совпал: другой прогон")

parent = release(make_run(seed=0)["model"])      # родитель прогона
manifest = {"id": "tiny-step-10",
            "parents": {"tiny-init": sha256(parent)}, "step": 10,
            "train_loss": round(losses[-1], 4),
            "weights_sha256": sha256(release(whole["model"]))}
check_parent(manifest, parent)                   # тот же родитель
try:                                             # чужой родитель
    check_parent(manifest, release(make_run(seed=1)["model"]))
    raise AssertionError("чужой родитель прошёл проверку")
except ValueError:
    pass
assert json.loads(json.dumps(manifest)) == manifest

# --- Валидационная ошибка, перплексия и бит на байт ---
@torch.no_grad()
def evaluate(model, windows):                    # windows: [n, T+1]
    x, y = windows[:, :-1], windows[:, 1:]
    nll = F.cross_entropy(model(x).flatten(0, 1), y.flatten(),
                          reduction="none").view(y.shape)
    per_window = nll.mean(1)
    sigma = per_window.std() / len(per_window) ** 0.5
    return nll.mean().item(), sigma.item()

def bits_per_byte(loss_per_token, n_tokens, n_bytes):
    return loss_per_token * n_tokens / (n_bytes * math.log(2))

def is_improvement(old, new, sigma):
    return new < old - 2 * sigma

text = "Мама мыла раму"
assert len(text) == 14 and len(text.encode("utf-8")) == 26
by_tokens = bits_per_byte(4.0, n_tokens=3, n_bytes=26)
by_bytes = bits_per_byte(12 / 26, n_tokens=26, n_bytes=26)
assert abs(by_tokens - by_bytes) < 1e-12
assert abs(by_tokens - 0.6659) < 1e-4
assert not is_improvement(3.412, 3.398, sigma=0.015)

flat = nn.Sequential(nn.Embedding(16, 8), nn.Linear(8, 16, bias=False))
nn.init.zeros_(flat[1].weight)                   # все логиты равны
loss, _ = evaluate(flat, torch.randint(0, 16, (4, 9)))
assert abs(loss - math.log(16)) < 1e-6           # ошибка ln V, PPL = V

# --- Бюджет BookLM по Chinchilla ---
N = 6_622_464                          # паспорт BookLM
D = 20 * N                             # правило Chinchilla
C = 6 * N * D                          # 6N на токен, D токенов
steps = math.ceil(D / (64 * 512))      # 32 768 токенов на шаг
assert D == 132_449_280 and steps == 4043        # 4042,03 вверх
assert abs(C / 5.26e15 - 1) < 1e-3

def mfu(tokens_per_second, peak_flops):
    return 6 * N * tokens_per_second / peak_flops

assert abs(C / (100e12 * 0.40) - 131.6) < 0.1     # секунд, MFU 40 %
assert abs(mfu(1_006_500, 100e12) - 0.40) < 1e-3  # ~1 млн ток./с

print("optim_precision_run_checks: все проверки прошли")
