"""Глава 7, раздел 7.4 «Оптимизаторы и расписание скорости обучения», подразделы 7.4.1–7.4.5.

Полные версии листингов «Градиентный спуск по оврагу», «Метод момента на том же овраге» и «Группы
параметров BookLM», а также Adam и AdamW руками против torch.optim:
траектории SGD, момента и Adam на овраге f = (x^2 + 25 y^2) / 2 (рисунок с тремя траекториями в
подразделе 7.4.3 «Оптимизатор Adam»),
сверка момента с torch.optim.SGD, сверка adamw_step с torch.optim.AdamW побитно в допуске 1e-6,
состояние оптимизатора (exp_avg, exp_avg_sq, step), группы параметров BookLM
(6 619 136 с затуханием и 3 328 без него) и поломка «связанный вес в двух группах».
Запуск из корня репозитория: python labs/ch07/optim_sgd_momentum_adamw.py
"""

# --- SGD и момент на овраге -----------------------------------------------
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

sgd_path, mom_path = run(0.07), run(0.07, beta=0.5)
sgd_hand = [[0.93, -0.75], [0.8649, 0.5625], [0.8044, -0.4219]]
mom_hand = [[0.93, -0.75], [0.8299, -0.3125], [0.7218, 0.4531]]
for s in (1, 2, 3):              # ручной расчёт SGD и момента
    assert close(sgd_path[s], sgd_hand[s - 1])
    assert close(mom_path[s], mom_hand[s - 1])
assert close(sgd_path[30], [0.1134, 0.0002])
# момент с тем же lr подошёл по x в 28 раз ближе, качели по y погасли
assert close(mom_path[20], [0.0293, -0.0005]) and close(mom_path[30], [0.0041, 0.0])
assert sgd_path[30][0] / mom_path[30][0] > 27
# SGD с «эффективным» шагом момента 2 * 0.07 = 0.14 разносит ось y
assert abs(run(0.14)[30][1]) > 1e9
# поломка: lr больше 2/25, качели по y растут
assert abs(run(0.085)[30][1]) > 1 and abs(run(0.1)[30][1]) > 1e3

p = torch.nn.Parameter(torch.tensor([1.0, 1.0], dtype=torch.float64))
opt = torch.optim.SGD([p], lr=0.07, momentum=0.5)  # та же формула
for s in range(1, 31):
    opt.zero_grad()
    (0.5 * (p[0] ** 2 + 25 * p[1] ** 2)).backward()
    opt.step()
    assert torch.allclose(p.detach(), mom_path[s], atol=1e-12, rtol=0)

# --- Adam и AdamW руками против torch.optim --------------------------------
import torch.nn as nn

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

theta, m, v = torch.tensor([1.0, 1.0], **f64), zero2, zero2
p = torch.nn.Parameter(theta.clone())
opt = torch.optim.AdamW([p], lr=0.05, betas=(0.9, 0.95), eps=1e-8,
                        weight_decay=0.1)
for s in range(1, 6):
    g = valley_grad(theta)
    theta, m, v = adamw_step(theta, g, m, v, s, lr=0.05, wd=0.1)
    opt.zero_grad()
    (0.5 * (p[0] ** 2 + 25 * p[1] ** 2)).backward()
    opt.step()
    assert torch.allclose(p.detach(), theta, atol=1e-6, rtol=0)
state = opt.state[p]                  # здесь PyTorch хранит m, v и s
assert torch.allclose(state["exp_avg"], m)
assert torch.allclose(state["exp_avg_sq"], v)
assert int(state["step"]) == 5

theta, m, v = torch.tensor([1.0, 1.0], **f64), zero2, zero2
adam_path = []
for s in range(1, 31):                # траектория Adam для рисунка
    theta, m, v = adamw_step(theta, valley_grad(theta), m, v, s,
                             lr=0.05, beta2=0.999)
    adam_path.append(theta)
assert close(adam_path[0], [0.95, 0.95], 1e-6)
assert close(adam_path[29], [-0.1027, -0.1027])

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

booklm = BookLMShapes()
groups = param_groups(booklm, weight_decay=0.1)
sizes = [sum(p.numel() for p in gr["params"]) for gr in groups]
assert sizes == [6_619_136, 3_328] and sum(sizes) == 6_622_464
torch.optim.AdamW(groups, lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
try:                                  # поломка: таблица и выход порознь
    torch.optim.AdamW([{"params": [booklm.tokens.weight]},
                       {"params": [booklm.output.weight]}])
    raise AssertionError("общий вес в двух группах, а ошибки нет")
except ValueError:
    pass


if __name__ == "__main__":
    print("OK: optim_sgd_momentum_adamw.py")
