"""Глава 18, раздел 18.5 «Цепочка поставки»: строка-метка в адаптере BookLM с DP-SGD и без.

Подраздел «Строка-метка при обучении с DP», таблица 18.2.

Дообучаем базовую BookLM следовать просьбам на 1000 пар «просьба -> история» из набора главы 8
(sft.jsonl, те же обучающие пары, что в sft_split главы 17), каждая обрезана до 48 токенов. В набор
добавлены c одинаковых копий пары-метки: просьба «Расскажи сказку про старый сундук.», история
начинается фразой «Код от сундука: 7316.». Адаптер LoRA ранга 16 стоит только на семи матрицах
последнего, шестого блока; первые пять блоков заморожены, и их выход для каждой пары считается
один раз. Так градиент каждого примера на CPU стоит недорого.

Два режима с одинаковыми Adam (скорость 2e-3), 100 шагами и пакетом выборкой с q = 0,064
(в среднем 64 пары, 6,4 эпохи):
  обычный — шаг Adam по среднему градиенту пакета;
  DP-SGD  — градиент каждой пары обрезан до C = 1, к сумме добавлен шум sigma*C, sigma = 1.
Бюджет DP-SGD при delta = 1e-5 — epsilon = 5,86 по простому переводу RDP из книги
(dp-accounting 0.5.1: 5,11 по RDP, 4,48 по PLD).

Меряем экспозицию метки: после подсказки «...<|assistant|>Код от сундука: » модель даёт
вероятность каждой из 10 000 строк из четырёх цифр (цифры у BookLM — отдельные токены), и мы
ищем место 7316. Качество — ошибка на токенах истории 200 отложенных пар (как sft_loss главы 18,
но на тех же 48 токенах). Зёрна 0, 1, 2 (инициализация адаптера, выборка пакетов, шум).

Запуск из корня репозитория: python labs/ch18/dp_lora_canary_booklm.py   (CPU, 5–8 минут)
Подмножество прогонов: PARTS="4:sgd:0,4:dp:0" python labs/ch18/dp_lora_canary_booklm.py
"""

import math
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "booklm")]
from booklm.experiments.ch17 import sft_split   # noqa: E402
from booklm.lab import load_model, load_tokenizer  # noqa: E402
from sft_lora import LoRALinear, encode          # noqa: E402

torch.set_num_threads(4)
S, N_BG = 48, 1000
REQUEST = "Расскажи сказку про старый сундук."
CANARY = "Код от сундука: 7316."
STORY = CANARY + " Жил-был старый сундук. В нём лежали игрушки, и дети очень его любили."
TARGETS = ("attention.q", "attention.k", "attention.v", "attention.out",
           "ffn.gate", "ffn.up", "ffn.down")

tok = load_tokenizer()
base = load_model("base")
last = base.blocks[-1]
for p in base.parameters():
    p.requires_grad = False


def add_last_lora(seed, r=16, alpha=32):
    """Свежий адаптер на матрицах последнего блока (B = 0, модель в точности базовая)."""
    torch.manual_seed(seed)
    params = []
    for name in TARGETS:
        parent_name, child = name.rsplit(".", 1)
        parent = last.get_submodule(parent_name)
        mod = getattr(parent, child)
        if isinstance(mod, LoRALinear):
            mod = mod.base
        lora = LoRALinear(mod, r, alpha)
        setattr(parent, child, lora)
        params += [lora.A, lora.B]
    return params


@torch.no_grad()
def frozen_part(rows):
    """Выход пятого блока для каждой пары и цели (-100 там, где ошибка не считается)."""
    out = []
    for row in rows:
        ids, labels = encode(tok, row["request"], row["story"], S)
        h = base.embed(torch.tensor([ids[:-1]]))
        for block in base.blocks[:-1]:
            h, _ = block(h, base.rope_cos, base.rope_sin)
        out.append((h[0], torch.tensor(labels[1:])))
    return out


def pair_loss(h, y, reduction="mean"):
    """Последний блок, нормировка и выходной слой на позициях истории."""
    z, _ = last(h[None], base.rope_cos, base.rope_sin)
    keep = y != -100
    return F.cross_entropy(base.norm(z)[0][keep] @ base.embed.weight.T, y[keep],
                           reduction=reduction)


val, train = sft_split()
BG = frozen_part(train[:N_BG])
CAN = frozen_part([{"request": REQUEST, "story": STORY}])[0]
VAL = frozen_part(val[:200])


@torch.no_grad()
def val_loss():
    total = sum(pair_loss(h, y, "sum").item() for h, y in VAL)
    return total / sum(int((y != -100).sum()) for _, y in VAL)


@torch.no_grad()
def exposure():
    """Место 7316 среди 10 000 строк из четырёх цифр и экспозиция в битах."""
    sp = lambda s: tok.token_to_id(s)
    prompt = ([sp("<bos>"), sp("<|user|>")] + tok.encode(REQUEST).ids + [sp("<|assistant|>")]
              + tok.encode("Код от сундука: ").ids)
    digit = [tok.token_to_id(str(d)) for d in range(10)]
    starts = [()] + [(a,) for a in range(10)] + [(a, b) for a in range(10) for b in range(10)] \
        + [(a, b, c) for a in range(10) for b in range(10) for c in range(10)]
    x = torch.tensor([prompt + [digit[d] for d in s] + [0] * (3 - len(s)) for s in starts])
    at = torch.tensor([len(prompt) - 1 + len(s) for s in starts])
    lp = base(x)[torch.arange(len(starts)), at].log_softmax(-1)[:, digit]   # [1111, 10]
    row = {s: i for i, s in enumerate(starts)}
    score = torch.zeros(10_000)
    for v in range(10_000):
        d = [int(ch) for ch in f"{v:04d}"]
        score[v] = sum(lp[row[tuple(d[:k])], d[k]] for k in range(4))
    place = 1 + int((score > score[7316]).sum())
    return place, math.log2(10_000) - math.log2(place)


def run(copies, dp, seed, steps=100, q=0.064, C=1.0, sigma=1.0, lr=2e-3):
    params = add_last_lora(seed)
    data = BG + [CAN] * copies
    n = len(data)
    L = q * n
    opt = torch.optim.Adam(params, lr=lr)
    gen = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        batch = (torch.rand(n, generator=gen) < q).nonzero()[:, 0].tolist()
        if not dp:
            grads = torch.autograd.grad(sum(pair_loss(*data[i]) for i in batch) / L, params)
        else:
            grads = [torch.zeros_like(p) for p in params]
            for i in batch:                                   # градиент каждой пары отдельно
                g = torch.autograd.grad(pair_loss(*data[i]), params)
                norm = torch.sqrt(sum((x * x).sum() for x in g)).item()
                for acc, x in zip(grads, g):
                    acc.add_(x * min(1.0, C / (norm + 1e-12)))
            grads = [(acc + torch.randn(acc.shape, generator=gen) * sigma * C) / L
                     for acc in grads]
        for p, g in zip(params, grads):
            p.grad = g
        opt.step()
    return exposure(), val_loss()


def rdp_sampled(alpha, q, sigma):
    terms = [math.log(math.comb(alpha, k)) + (alpha - k) * math.log1p(-q)
             + k * math.log(q) + (k * k - k) / (2 * sigma ** 2) for k in range(alpha + 1)]
    top = max(terms)
    return (top + math.log(sum(math.exp(t - top) for t in terms))) / (alpha - 1)


if __name__ == "__main__":
    eps = min(100 * rdp_sampled(a, 0.064, 1.0) + math.log(1e5) / (a - 1) for a in range(2, 128))
    assert round(eps, 2) == 5.86
    parts = os.environ.get("PARTS")
    if parts:                                    # отдельные прогоны, для запуска по частям
        for part in parts.split(","):
            c, mode, seed = part.split(":")
            (place, bits), loss = run(int(c), mode == "dp", int(seed))
            print(f"c={c} {mode} seed={seed}: место {place}, экспозиция {bits:.2f}, ошибка {loss:.4f}",
                  flush=True)
        sys.exit()

    add_last_lora(0)                             # B = 0: модель ещё базовая
    (place0, bits0), loss0 = exposure(), val_loss()
    print(f"базовая BookLM: место {place0}, экспозиция {bits0:.2f} бит, ошибка {loss0:.4f}")
    result = {}
    for copies in (4, 16):
        for mode in ("sgd", "dp"):
            runs = [run(copies, mode == "dp", seed) for seed in range(3)]
            bits = [b for (_, b), _ in runs]
            loss = [l for _, l in runs]
            result[copies, mode] = (sum(bits) / 3, sum(loss) / 3)
            print(f"c={copies:2d} {mode:3}: места {[p for (p, _), _ in runs]}, экспозиция "
                  f"{sum(bits) / 3:.2f}±{(max(bits) - min(bits)) / 2:.2f} бит, "
                  f"ошибка {sum(loss) / 3:.4f}", flush=True)

    assert bits0 < 1.0                                       # до дообучения метки модель не знает
    for copies in (4, 16):
        sgd_bits, sgd_loss = result[copies, "sgd"]
        dp_bits, dp_loss = result[copies, "dp"]
        assert sgd_bits > dp_bits + 2                        # без DP метку видно, с DP нет
        assert dp_loss < loss0 - 0.4                         # DP-адаптер выучил формат ответа
        assert abs(dp_loss - sgd_loss) < 0.05                # и почти не уступает обычному
    print("все проверки пройдены")
