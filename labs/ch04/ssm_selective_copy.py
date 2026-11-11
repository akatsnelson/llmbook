"""Опыт главы 4 «выборочное копирование»: SSM без селективности, селективная SSM и внимание.

Раздел 4.12 «Модели пространства состояний» описывает постановку и
печатает итоговую таблицу; здесь опыт целиком. Как устроены шаг обучения, оптимизатор Adam и
перекрёстная энтропия, разбирает глава 7 «Предобучение»; здесь они чёрный ящик.

Задача — уменьшенная копия опыта из статьи о Mamba (Gu, Dao, 2023, раздел 4.1.1). В строке CTX
позиций; на M из них стоят токены данных 1..8, на остальных — шумовой токен 0. За строкой идут M
меток-вопросов, и на i-й метке модель должна назвать i-й токен данных. Данные стоят в одном из трёх
мест: подряд в начале строки (до своей метки ровно CTX шагов), подряд в конце строки, перед самыми
метками (M шагов), или вразброс на случайных позициях.

Три модели по два слоя с одинаковым числом параметров (с точностью до нескольких процентов):
  lti        — блок Mamba, в котором шаг, B и C — обычные обучаемые числа, одинаковые для всех токенов;
  selective  — тот же блок, но шаг, B и C считаются по токену (селективная SSM, S6);
  attention  — два блока трансформера: само-внимание с причинной маской и FFN, позиции — обучаемая
                 таблица.

Запуск из корня репозитория: python labs/ch04/ssm_selective_copy.py
Это полный опыт: 5 зёрен, 45 прогонов, около 5 минут на Apple M2 Max в 6 процессах (ключ --jobs).
Другие режимы:
    python labs/ch04/ssm_selective_copy.py --quick  (1 зерно, меньше шагов, для проверки)
    python labs/ch04/ssm_selective_copy.py --delta  (шаг Delta селективной модели по видам токенов)
"""

import argparse
import math
import multiprocessing
import statistics
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

NOISE, MARK, VOCAB = 0, 9, 10        # 0 — шум, 1..8 — данные, 9 — метка-вопрос
CTX, M = 32, 4                       # длина строки и число токенов данных


LAYOUTS = {"start": "в начале", "end": "в конце", "scattered": "вразброс"}


def make_batch(batch, layout, gen):
    """Строки задачи: ids формы [batch, CTX+M] и ответы формы [batch, M]."""
    data = torch.randint(1, 9, (batch, M), generator=gen)
    if layout == "scattered":        # M случайных позиций по возрастанию
        pos = torch.rand(batch, CTX, generator=gen).argsort(-1)[:, :M].sort(-1).values
    elif layout == "start":          # первые M позиций
        pos = torch.arange(M).expand(batch, M)
    else:                            # последние M позиций перед метками
        pos = torch.arange(CTX - M, CTX).expand(batch, M)
    ids = torch.full((batch, CTX + M), NOISE, dtype=torch.long)
    ids.scatter_(1, pos, data)
    ids[:, CTX:] = MARK
    return ids, data


class SSMMixer(nn.Module):
    """Смеситель Mamba без свёртки: вход -> (u, ворота z) -> SSM по u -> y * silu(z) -> выход."""

    def __init__(self, d, n=16, expand=2, selective=True, rank=2):
        super().__init__()
        di = expand * d
        self.selective, self.n, self.rank = selective, n, rank
        self.in_proj = nn.Linear(d, 2 * di, bias=False)
        # S4D-Real: A = -(1, 2, ..., n) в каждом из di каналов; храним логарифм, чтобы A < 0
        self.A_log = nn.Parameter(torch.log(torch.arange(1, n + 1, dtype=torch.float)).repeat(di, 1))
        self.D = nn.Parameter(torch.ones(di))          # прямой путь x -> y, как в Mamba
        dt0 = torch.exp(torch.rand(di) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
        dt_bias = dt0 + torch.log(-torch.expm1(-dt0))  # softplus(dt_bias) = dt0 из [0,001; 0,1]
        if selective:
            self.x_proj = nn.Linear(di, rank + 2 * n, bias=False)   # шаг (ранг 2), B, C по токену
            self.dt_proj = nn.Linear(rank, di)
            with torch.no_grad():
                self.dt_proj.bias.copy_(dt_bias)
        else:
            self.dt = nn.Parameter(dt_bias)                          # один шаг на канал
            self.B = nn.Parameter(torch.randn(di, n) / math.sqrt(n))
            self.C = nn.Parameter(torch.randn(di, n) / math.sqrt(n))
        self.out_proj = nn.Linear(di, d, bias=False)

    def forward(self, x):
        u, z = self.in_proj(x).chunk(2, dim=-1)
        u = F.silu(u)                                   # [b, T, di]
        A = -torch.exp(self.A_log)                      # [di, n]
        if self.selective:
            dt, B, C = self.x_proj(u).split([self.rank, self.n, self.n], dim=-1)
            dt = F.softplus(self.dt_proj(dt))           # [b, T, di]: свой шаг у каждого токена
            B, C = B[:, :, None, :], C[:, :, None, :]   # [b, T, 1, n]
        else:
            dt = F.softplus(self.dt).expand_as(u)       # один шаг для всех токенов
            B, C = self.B.expand(*u.shape[:2], -1, -1), self.C.expand(*u.shape[:2], -1, -1)
        decay = torch.exp(dt[..., None] * A)            # A с чертой: [b, T, di, n]
        push = dt[..., None] * B * u[..., None]         # B с чертой, умноженная на вход
        h = torch.zeros_like(decay[:, 0])
        ys = []
        # рекуррентный расчёт по токенам; unbind вместо decay[:, t], иначе обратный проход на каждом шаге
        # заводит нулевой тензор во всю длину и работа растёт как T^2
        for a_t, b_t, c_t in zip(decay.unbind(1), push.unbind(1), C.unbind(1)):
            h = a_t * h + b_t
            ys.append((h * c_t).sum(-1))
        y = torch.stack(ys, dim=1) + u * self.D
        return self.out_proj(y * F.silu(z))


class AttnBlock(nn.Module):
    """Блок трансформера: RMSNorm -> само-внимание с причинной маской -> RMSNorm -> FFN."""

    def __init__(self, d, heads, d_ff):
        super().__init__()
        self.heads = heads
        self.n1, self.n2 = nn.RMSNorm(d), nn.RMSNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.ff = nn.Sequential(nn.Linear(d, d_ff, bias=False), nn.GELU(), nn.Linear(d_ff, d, bias=False))

    def forward(self, x):
        b, T, d = x.shape
        q, k, v = self.qkv(self.n1(x)).view(b, T, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.o(a.transpose(1, 2).reshape(b, T, d))
        return x + self.ff(self.n2(x))


class Model(nn.Module):
    def __init__(self, kind, d=32, layers=2):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, d)
        if kind == "attention":
            self.pos = nn.Parameter(torch.randn(CTX + M, d) * 0.02)
            self.blocks = nn.ModuleList(AttnBlock(d, heads=4, d_ff=76) for _ in range(layers))
        else:
            self.pos = None
            self.norms = nn.ModuleList(nn.RMSNorm(d) for _ in range(layers))
            self.blocks = nn.ModuleList(SSMMixer(d, selective=(kind == "selective")) for _ in range(layers))
        self.norm = nn.RMSNorm(d)
        self.head = nn.Linear(d, VOCAB, bias=False)

    def forward(self, ids):
        x = self.embed(ids)
        if self.pos is not None:
            x = x + self.pos
            for blk in self.blocks:
                x = blk(x)
        else:
            for norm, blk in zip(self.norms, self.blocks):
                x = x + blk(norm(x))                   # остаточная связь вокруг смесителя
        return self.head(self.norm(x))


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def train(kind, layout, seed, steps=1500, batch=32, lr=3e-3, return_model=False):
    """Учит модель и возвращает долю верно названных токенов на 2048 новых строках."""
    torch.set_num_threads(1)                           # один поток: так прогон повторяется точно
    torch.manual_seed(seed)
    model = Model(kind)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(1000 + seed)
    for step in range(steps):
        ids, data = make_batch(batch, layout, gen)
        logits = model(ids)[:, CTX:]                   # ответы стоят только на метках
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), data.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    test = torch.Generator().manual_seed(99)
    ids, data = make_batch(2048, layout, test)
    with torch.no_grad():
        pred = model(ids)[:, CTX:].argmax(-1)
    accuracy = (pred == data).float().mean().item()
    return (accuracy, model) if return_model else accuracy


def selection_by_token(seed=0):
    """Что селективная модель делает с шумом: шаг Delta и длины векторов B и C по видам токенов.

    Модель учится на варианте «в начале». Печатаются средние по пакету значения первого слоя до и после
    обучения: шаг Delta (среднее по каналам), длина B_t (сколько токен пишет в состояние) и длина C_t
    (сколько позиция читает из состояния).
    """
    ids, _ = make_batch(512, "start", torch.Generator().manual_seed(7))
    kinds = {"шум": ids == NOISE, "данные": (ids > 0) & (ids < MARK), "метки": ids == MARK}

    def by_token(model):
        mixer = model.blocks[0]
        with torch.no_grad():
            u = F.silu(mixer.in_proj(model.norms[0](model.embed(ids))).chunk(2, dim=-1)[0])
            dt, B, C = mixer.x_proj(u).split([mixer.rank, mixer.n, mixer.n], dim=-1)
            dt = F.softplus(mixer.dt_proj(dt)).mean(-1)                            # [b, T]
        return {name: (dt[mask].mean().item(), B.norm(dim=-1)[mask].mean().item(),
                       C.norm(dim=-1)[mask].mean().item()) for name, mask in kinds.items()}

    torch.manual_seed(seed)                            # та же модель, что train() строит до обучения
    initial = by_token(Model("selective"))
    accuracy, model = train("selective", "start", seed, return_model=True)
    return accuracy, initial, by_token(model)


def check_ssm_modes():
    """Рекуррентный расчёт смесителя совпадает с развёрнутым через кумулятивные суммы логарифмов."""
    torch.manual_seed(0)
    mixer = SSMMixer(8, n=4, selective=True)
    x = torch.randn(2, 10, 8)
    u, z = mixer.in_proj(x).chunk(2, dim=-1)
    u = F.silu(u)
    A = -torch.exp(mixer.A_log)
    dt, B, C = mixer.x_proj(u).split([mixer.rank, mixer.n, mixer.n], dim=-1)
    dt = F.softplus(mixer.dt_proj(dt))
    logdecay = (dt[..., None] * A).cumsum(1)                        # [b, T, di, n]
    L = torch.exp(logdecay[:, :, None] - logdecay[:, None, :])      # [b, t, s, di, n]
    L = L * torch.tril(torch.ones(10, 10))[None, :, :, None, None]
    pushed = dt[..., None] * B[:, :, None, :] * u[..., None]        # [b, s, di, n]
    y = torch.einsum("btsdn,bsdn,btn->btd", L, pushed, C) + u * mixer.D
    parallel = mixer.out_proj(y * F.silu(z))
    assert torch.allclose(parallel, mixer(x), atol=1e-5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--delta", action="store_true", help="шаг Delta на шуме и на данных")
    parser.add_argument("--jobs", type=int, default=6, help="сколько прогонов считать параллельно")
    args = parser.parse_args()
    if args.delta:
        accuracy, initial, table = selection_by_token()
        print(f"точность {accuracy:.3f}; первый слой, средние по токенам (в скобках --- до обучения):")
        for name, (dt, b, c) in table.items():
            dt0, b0, c0 = initial[name]
            print(f"  {name:7s} Delta {dt:.3f} ({dt0:.3f})  |B| {b:.2f} ({b0:.2f})  |C| {c:.2f} ({c0:.2f})")
        return
    seeds = 1 if args.quick else 5
    steps = 300 if args.quick else 1500

    check_ssm_modes()
    kinds = ("lti", "selective", "attention")
    sizes = [n_params(Model(k)) for k in kinds]
    print("параметры:", ", ".join(f"{k} {n}" for k, n in zip(kinds, sizes)))
    assert max(sizes) / min(sizes) < 1.05, sizes   # одинаковый бюджет с точностью до 5 %

    jobs = [(kind, layout, seed, steps) for layout in LAYOUTS for kind in kinds for seed in range(seeds)]
    start = time.time()
    with multiprocessing.Pool(args.jobs) as pool:
        accs = pool.starmap(train, jobs)
    results = {}
    for (kind, layout, _, _), acc in zip(jobs, accs):
        results.setdefault((layout, kind), []).append(acc)
    for (layout, kind), runs in results.items():
        spread = statistics.stdev(runs) if len(runs) > 1 else 0.0
        print(f"{LAYOUTS[layout]:9s} {kind:10s} {statistics.mean(runs):.3f} ± {spread:.3f}  "
              f"[{' '.join(f'{a:.2f}' for a in runs)}]")
    print(f"время: {time.time() - start:.0f} с")
    if not args.quick:
        mean = {key: statistics.mean(runs) for key, runs in results.items()}
        for layout in ("start", "scattered"):       # селективная SSM лучше стационарной
            assert mean[layout, "selective"] > mean[layout, "lti"] + 0.1, layout
        assert mean["start", "lti"] < 0.5           # данные далеко и подряд: стационарная не справляется
        assert mean["scattered", "attention"] > 0.98


if __name__ == "__main__":
    main()
