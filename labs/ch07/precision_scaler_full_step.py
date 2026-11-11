"""Глава 7, раздел 7.5 «Обучение в 16-битных числах».

Полные версии листингов «Три формата в PyTorch», «Масштабировщик ошибки», «Итоговый шаг обучения» и
«Ограничение нормы до и после снятия масштаба»: TinyLM, собственный DynamicScaler, опыт с
бесконечностью в градиенте на третьем шаге (веса, m, v, s и планировщик не двигаются), масштаб на
трёх пакетах с градиентом 2,0 из подраздела 7.5.3 «Масштабирование ошибки», побитная сверка с
torch.amp.GradScaler + clip_grad_norm_ + AdamW и опыт с неправильным порядком «ограничение до снятия
масштаба» для AdamW и SGD.
Запуск из корня репозитория: python labs/ch07/precision_scaler_full_step.py
"""

# --- Форматы на ладони -----------------------------------------------------
import torch

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

N = 6_622_464                         # параметры BookLM
per_param = {"вес BF16": 2, "главная копия FP32": 4,
             "градиент BF16": 2, "m FP32": 4, "v FP32": 4}
assert sum(per_param.values()) == 16
assert N * 16 == 105_959_424 and round(N * 16 / 1e6) == 106

# --- Масштабировщик ошибки и опыт переполнения -----------------------------
import math
import torch.nn as nn
import torch.nn.functional as F

class TinyLM(nn.Module):     # таблица, RMSNorm, связанный выход
    def __init__(self, V=16, d=8):
        super().__init__()
        self.tokens = nn.Embedding(V, d)
        self.norm = nn.RMSNorm(d, eps=1e-6)
        self.output = nn.Linear(d, V, bias=False)
        self.output.weight = self.tokens.weight   # связанные веса
        nn.init.normal_(self.tokens.weight, std=0.02)

    def forward(self, ids):                       # [B,T] -> [B,T,V]
        return self.output(self.norm(self.tokens(ids)))

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

    def state_dict(self):
        return {"scale": self.S, "growth_factor": 2.0,
                "backoff_factor": 0.5, "growth_interval": self.interval,
                "_growth_tracker": self.good}

def batch_loss(model, x, y):
    return F.cross_entropy(model(x).flatten(0, 1), y.flatten())

def scaled_step(model, opt, sched, scaler, x, y, corrupt=False):
    opt.zero_grad(set_to_none=True)
    (batch_loss(model, x, y) * scaler.S).backward()
    if corrupt:                       # опыт: один градиент стал inf
        model.tokens.weight.grad[0, 0] = float("inf")
    found_inf = scaler.unscale_and_check(model.parameters())
    if not found_inf:
        opt.step()                    # шаг и оба счётчика: только тут
        sched.step()
    scaler.update(found_inf)
    return found_inf

def make_run():
    torch.manual_seed(0)
    model = TinyLM()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3,
                            betas=(0.9, 0.95), weight_decay=0.1)
    warmup = lambda s: min(1.0, (s + 1) / 10)
    return model, opt, torch.optim.lr_scheduler.LambdaLR(opt, warmup)

def snapshot(model, opt, sched):      # всё, что живёт дольше шага
    ts = [p.detach().clone() for p in model.parameters()]
    ts += [t.clone() for st in opt.state.values() for t in st.values()]
    return ts, sched.last_epoch

x = torch.arange(16).view(2, 8)       # игрушечный пакет [B, T]
y = x ^ 1
model, opt, sched = make_run()
scaler = DynamicScaler(growth_interval=3)    # в опыте n_рост = 3
for _ in range(2):
    assert not scaled_step(model, opt, sched, scaler, x, y)
before = snapshot(model, opt, sched)
assert scaled_step(model, opt, sched, scaler, x, y, corrupt=True)
after = snapshot(model, opt, sched)
assert before[1] == after[1] == 2                 # планировщик стоит
assert all(torch.equal(a, b) for a, b in zip(before[0], after[0]))
assert scaler.S == 2.0**15                        # масштаб вдвое меньше
for _ in range(3):
    assert not scaled_step(model, opt, sched, scaler, x, y)
assert scaler.S == 2.0**16                        # три удачных: рост
assert opt.state[model.norm.weight]["step"].item() == 5   # не 6

trace = DynamicScaler()               # три пакета из подраздела 7.5.3
for expected in (2.0**15, 2.0**14, 2.0**14):
    trace.update(math.isinf(to16(2.0 * trace.S)))
    assert trace.S == expected
assert to16(1e-8 * 2**14) > 0

# --- Полный шаг смешанной точности против GradScaler -----------------------
def clip_by_global_norm(grads, c):
    norm = math.sqrt(sum((g.double() ** 2).sum().item() for g in grads))
    return [g * min(1.0, c / norm) for g in grads], norm

g = torch.tensor([3.0, 4.0])
(clipped,), norm = clip_by_global_norm([g], c=1.0)
assert norm == 5.0 and torch.allclose(clipped, torch.tensor([0.6, 0.8]))

def full_step(model, opt, sched, scaler, x, y, c=1.0, corrupt=False):
    opt.zero_grad(set_to_none=True)
    (batch_loss(model, x, y) * scaler.S).backward()   # 1. от S*L
    if corrupt:
        model.tokens.weight.grad[0, 0] = float("inf")
    params = list(model.parameters())
    found_inf = scaler.unscale_and_check(params)      # 2-3. /S, inf?
    norm = None
    if not found_inf:
        norm = torch.nn.utils.clip_grad_norm_(params, c).item()  # 4.
        opt.step()                                    # 5.
        sched.step()
    scaler.update(found_inf)
    return norm                       # норма до ограничения: в журнал

def torch_step(model, opt, sched, scaler, x, y, c=1.0, corrupt=False):
    opt.zero_grad(set_to_none=True)
    scaler.scale(batch_loss(model, x, y)).backward()
    if corrupt:
        model.tokens.weight.grad[0, 0] = float("inf")
    scaler.unscale_(opt)                              # /S и поиск inf
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), c).item()
    before = scaler.get_scale()
    scaler.step(opt)                                  # при inf шага нет
    scaler.update()
    if scaler.get_scale() < before:                   # шаг пропущен
        return None
    sched.step()
    return norm

mine, theirs = make_run(), make_run()
my_scaler = DynamicScaler(growth_interval=3)
torch_scaler = torch.amp.GradScaler("cpu", growth_interval=3)
log = [(full_step(*mine, my_scaler, x, y, corrupt=(i == 2)),
        torch_step(*theirs, torch_scaler, x, y, corrupt=(i == 2)))
       for i in range(6)]
assert all(torch.equal(p, q) for p, q in
           zip(mine[0].parameters(), theirs[0].parameters()))
assert my_scaler.state_dict() == torch_scaler.state_dict()
assert mine[2].last_epoch == theirs[2].last_epoch == 5
assert log[2] == (None, None)         # пропуск на одном и том же шаге

def clip_order_run(S, clip_first, make_opt, steps=20, c=1.0):
    torch.manual_seed(0)
    model = TinyLM()
    start = [p.detach().clone() for p in model.parameters()]
    opt, norms = make_opt(model.parameters()), []
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        (batch_loss(model, x, y) * S).backward()
        params = list(model.parameters())
        if clip_first:                    # поломка: до деления на S
            norms.append(torch.nn.utils.clip_grad_norm_(params, c).item())
        for p in params:
            p.grad.div_(S)
        if not clip_first:
            norms.append(torch.nn.utils.clip_grad_norm_(params, c).item())
        opt.step()
    moved = math.sqrt(sum(((p.detach() - s) ** 2).sum().item()
                          for p, s in zip(model.parameters(), start)))
    return moved, norms

adamw = lambda ps: torch.optim.AdamW(ps, lr=3e-3, betas=(0.9, 0.95),
                                     weight_decay=0.0)
sgd = lambda ps: torch.optim.SGD(ps, lr=0.5)
right, norms_right = clip_order_run(1024.0, False, adamw)
wrong, norms_wrong = clip_order_run(1024.0, True, adamw)
assert abs(wrong / right - 1) < 0.02      # AdamW: сдвиг почти тот же,
assert min(norms_wrong) > 1.0             # но ограничение режет всегда,
assert round(norms_wrong[0] / norms_right[0]) == 1024  # журнал: S*|g|
squeezed, _ = clip_order_run(2.0**30, True, adamw)  # c/(S sqrt N) << eps
assert squeezed < 0.05 * right            # шаг съел epsilon
sgd_right, _ = clip_order_run(1024.0, False, sgd)
sgd_wrong, _ = clip_order_run(1024.0, True, sgd)
assert sgd_wrong < sgd_right / 100        # SGD: шаг в сотни раз короче


if __name__ == "__main__":
    print("OK: precision_scaler_full_step.py")
