"""Глава 18, раздел 18.5 «Цепочка поставки»: строка-метка при обучении с DP-SGD (полная версия).

Подраздел «Строка-метка при обучении с DP», таблица 18.1. В книге оставлены шаг DP-SGD и учёт бюджета; здесь
опыт целиком, с проверками утверждениями.

Модель — таблица логитов «начало кода -> следующая цифра»: для каждого из 1 + 10 + 100 + 1000
начал PIN своя строка из 10 логитов, всего 11 110 параметров. Такая модель помнит всё начало строки
и потому способна запомнить отдельный PIN, как большая языковая модель запоминает редкую строку.
Корпус: 1000 фоновых PIN (половина — даты «ддмм», половина случайные) и c копий строки-метки
7316. Пять режимов обучения с одной и той же скоростью 0,3, 1000 шагов, пакет по выборке с долей
q = 0,1 (в среднем 100 PIN), пять зёрен на клетку:
  SGD                    — обычный шаг по среднему градиенту;
  обрезка без шума       — градиент каждого примера укорочен до C = 1, шума нет;
  DP-SGD sigma = 2, 4, 8 — обрезка и гауссов шум sigma * C.
Меряем экспозицию метки среди всех 10 000 PIN (формула 18.4) и ошибку на 2000
отложенных фоновых PIN в битах на PIN: 13,29 бита — равномерная догадка, 11,79 — истинное
распределение фона.
Запуск из корня репозитория: python labs/ch18/dp_sgd_canary.py   (CPU, меньше минуты)
"""

import math
import random

import torch

torch.set_num_threads(4)
CANARY = "7316"
DAYS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]


# ----------------------------------------------------------------------------- данные

def background(n, seed):
    """n фоновых PIN: с вероятностью 1/2 дата «ддмм», иначе случайные четыре цифры."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        if rng.random() < 0.5:
            day, month = rng.randrange(365), 0
            while day >= DAYS[month]:
                day, month = day - DAYS[month], month + 1
            out.append(f"{day + 1:02d}{month + 1:02d}")
        else:
            out.append(f"{rng.randrange(10_000):04d}")
    return out


def contexts(pins):
    """Номера строк таблицы для четырёх позиций PIN и сами цифры: ([n, 4], [n, 4])."""
    d = torch.tensor([[int(ch) for ch in p] for p in pins])
    zero = torch.zeros(len(pins), dtype=torch.long)
    ctx = torch.stack([zero, 1 + d[:, 0], 11 + 10 * d[:, 0] + d[:, 1],
                       111 + 100 * d[:, 0] + 10 * d[:, 1] + d[:, 2]], 1)
    return ctx, d


ALL_CTX, ALL_D = contexts([f"{x:04d}" for x in range(10_000)])


def log_p(W, ctx, d):
    """Натуральный логарифм вероятности каждого PIN под таблицей W [1111, 10]."""
    return W.log_softmax(-1)[ctx].gather(-1, d[..., None])[..., 0].sum(1)


def exposure(W):
    """Место метки среди 10 000 PIN и экспозиция log2 N - log2 m."""
    lp = log_p(W, ALL_CTX, ALL_D)
    place = 1 + int((lp > lp[int(CANARY)]).sum())
    return place, math.log2(10_000) - math.log2(place)


# ----------------------------------------------------------------------------- обучение

def train(pins, steps=1000, q=0.1, lr=0.3, C=None, sigma=0.0, seed=0):
    """SGD (C=None) или DP-SGD: обрезка каждого примера до C и шум sigma*C на сумму.

    Пакет набирается выборкой: каждый PIN попадает в него независимо с вероятностью q, как в
    анализе DP-SGD. Сумма делится на ожидаемый размер пакета L = qN.
    """
    gen = torch.Generator().manual_seed(seed)
    ctx, d = contexts(pins)
    n = len(pins)
    L = q * n
    W = torch.zeros(1111, 10)
    for _ in range(steps):
        idx = (torch.rand(n, generator=gen) < q).nonzero()[:, 0]
        c, y = ctx[idx], d[idx]
        g = W[c].softmax(-1)                                   # dℓ/dлогит = p - [j = y]
        g.scatter_add_(-1, y[..., None], -torch.ones(y.shape + (1,)))
        if C is not None:                                      # обрезка по примерам
            norms = g.flatten(1).norm(dim=1)
            g = g * (C / norms).clamp(max=1.0)[:, None, None]
        G = torch.zeros_like(W).index_put_((c.flatten(),), g.reshape(-1, 10), accumulate=True)
        if sigma > 0:
            G += torch.randn(W.shape, generator=gen) * sigma * C   # шум на каждый параметр
        W -= lr * G / L
    return W


def rdp_sampled(alpha, q, sigma):
    """Рени-приватность порядка alpha одного шага гауссова механизма с выборкой q (целое alpha)."""
    terms = [math.log(math.comb(alpha, k)) + (alpha - k) * math.log1p(-q)
             + k * math.log(q) + (k * k - k) / (2 * sigma ** 2) for k in range(alpha + 1)]
    top = max(terms)
    return (top + math.log(sum(math.exp(t - top) for t in terms))) / (alpha - 1)


def epsilon(T, delta, q, sigma):
    return min(T * rdp_sampled(a, q, sigma) + math.log(1 / delta) / (a - 1)
               for a in range(2, 128))


# ----------------------------------------------------------------------------- опыт

TEST_CTX, TEST_D = contexts(background(2000, seed=99))
REGIMES = [("SGD", None, 0.0), ("обрезка без шума", 1.0, 0.0),
           ("DP-SGD sigma=2", 1.0, 2.0), ("DP-SGD sigma=4", 1.0, 4.0), ("DP-SGD sigma=8", 1.0, 8.0)]
COPIES, SEEDS = (1, 4, 16), range(5)


def cell(C, sigma, copies):
    """Средняя экспозиция, её полуразмах по зёрнам и средняя ошибка на отложенных PIN, бит."""
    ex, bits = [], []
    for seed in SEEDS:
        W = train(background(1000, seed) + [CANARY] * copies, C=C, sigma=sigma, seed=seed)
        ex.append(exposure(W)[1])
        bits.append(-log_p(W, TEST_CTX, TEST_D).mean().item() / math.log(2))
    return sum(ex) / len(ex), (max(ex) - min(ex)) / 2, sum(bits) / len(bits)


if __name__ == "__main__":
    # истинная энтропия фона: 365 дат и 10 000 случайных кодов пополам
    p_date, p_rand = 0.5 / 365 + 0.5 / 10_000, 0.5 / 10_000
    H = -(365 * p_date * math.log2(p_date) + 9635 * p_rand * math.log2(p_rand))
    assert abs(H - 11.79) < 0.01

    # учёт бюджета: наша простая оценка не меньше точных (dp-accounting 0.5.1: RDP 3.74, PLD 3.44)
    eps = {s: epsilon(1000, 1e-5, 0.1, s) for s in (2.0, 4.0, 8.0)}
    assert [round(eps[s], 2) for s in (2.0, 4.0, 8.0)] == [9.84, 4.24, 2.00]
    assert eps[4.0] >= 3.74 >= 3.44

    print(f"{'режим':18} {'eps':>6} | экспозиция, бит: c=1 / c=4 / c=16 | ошибка, бит")
    table = {}
    for name, C, sigma in REGIMES:
        rows = [cell(C, sigma, c) for c in COPIES]
        table[name] = rows
        e = f"{eps[sigma]:.2f}" if sigma else "inf"
        print(f"{name:18} {e:>6} | " + " / ".join(f"{m:5.2f}±{h:.2f}" for m, h, _ in rows)
              + f" | {sum(r[2] for r in rows) / 3:.2f}")

    sgd, clip, dp4, dp8 = (table[k] for k in ("SGD", "обрезка без шума",
                                              "DP-SGD sigma=4", "DP-SGD sigma=8"))
    # 16 копий без DP: метка первая из 10 000
    assert abs(sgd[2][0] - math.log2(10_000)) < 1e-9
    # одна копия: DP возвращает метку к уровню случайного PIN (около 1 бита), SGD — выше
    assert dp4[0][0] < 1.2 < sgd[0][0]
    # четыре копии DP почти не спасает: гарантия на одну единицу, а копий четыре
    assert dp4[1][0] > 3.5 and sgd[1][0] - dp4[1][0] < 1.5
    # шестнадцать копий: даже при eps = 2 экспозиция выше 9 бит
    assert dp8[2][0] > 9
    # обрезка без шума: гарантии нет, экспозиция одной копии всё ещё выше, чем у DP
    assert clip[0][0] > dp4[0][0]
    # цена: чем больше шум, тем хуже ошибка на отложенных PIN, но лучше равномерной догадки
    assert sgd[0][2] < clip[0][2] < dp4[0][2] < dp8[0][2] < math.log2(10_000)
    print("все проверки пройдены")
