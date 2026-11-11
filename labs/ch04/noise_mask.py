"""Опыт главы 4 «внимание без маски учится на шуме»: цикл обучения и честный предел.

В книге (раздел 4.3 «Причинная маска») описаны постановка и итоговые
числа; здесь опыт целиком. Как устроены шаг обучения,
оптимизатор Adam и перекрёстная энтропия, разбирает глава 7 «Предобучение».

noise_floor считает, какую ошибку может получить внимание без маски, если оно видит всё окно,
но не знает позиций: ответ позиции t — токен t+1, он лежит в окне у всех позиций, кроме
последней, а какая позиция последняя, модель не знает. Поэтому предел не ln 8 и не ноль.

Запуск из корня репозитория: python labs/ch04/noise_mask.py
"""

import math

import torch
import torch.nn as nn


def train_on_noise(attention_fn, vocab=32, T=8, d=32, steps=400, seed=0):
    """Учит эмбеддинг + одно внимание + линейный слой угадывать следующий токен шума.

    Возвращает среднюю ошибку за последние 50 шагов.
    """
    torch.manual_seed(seed)
    embed = nn.Embedding(vocab, d)                     # таблица E из главы 3
    qkv = nn.Linear(d, 3 * d, bias=False)              # W_Q, W_K, W_V одним слоем
    head = nn.Linear(d, vocab)                         # вектор -> логиты
    opt = torch.optim.Adam(nn.ModuleList([embed, qkv, head]).parameters(), lr=0.01)
    losses = []
    for _ in range(steps):
        ids = torch.randint(0, vocab, (64, T + 1))     # 64 строки шума
        q, k, v = qkv(embed(ids[:, :-1])).chunk(3, dim=-1)
        mixed, _ = attention_fn(q, k, v)
        loss = nn.functional.cross_entropy(            # ответ: сосед справа
            head(mixed).flatten(0, 1), ids[:, 1:].flatten())
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return sum(losses[-50:]) / 50                      # среднее за последние 50 шагов


def noise_runs(attention_fn, seeds=5, **kw):
    """train_on_noise на нескольких зёрнах: строка «среднее ± разброс»."""
    import statistics
    runs = [train_on_noise(attention_fn, seed=s, **kw) for s in range(seeds)]
    return f"{statistics.mean(runs):.2f} ± {statistics.stdev(runs):.2f}"


def noise_floor(vocab=32, T=8, n=200_000, seed=0):
    """Наименьшая ошибка внимания без маски и без позиций.

    Такое внимание на позиции t знает свой токен x_t и мешок токенов окна, но не порядок.
    Для t < T-1 ответ x_{t+1} — токен одной из остальных T-1 позиций, при случайном порядке
    любой из них с равными шансами. Для последней позиции ответ — новый случайный токен.
    Не зная t, лучше всего ставить на токен y долю p(y) = c(y)/T + 1/(T*V), где c(y) —
    сколько раз y стоит в окне, не считая самой позиции t. Функция возвращает среднее -ln p.
    """
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, vocab, (n, T + 1), generator=g)
    x, y = ids[:, :T], ids[:, 1:]
    counts = torch.zeros(n, vocab).scatter_add_(1, x, torch.ones(n, T))
    others = counts[:, None, :].repeat(1, T, 1)                   # [n, T, V]
    others.scatter_add_(2, x[:, :, None], -torch.ones(n, T, 1))  # без своего токена
    p = others / T + 1 / (T * vocab)
    return round(-p.gather(2, y[:, :, None]).log().mean().item(), 2)


def _attention(q, k, v, causal=False):
    s = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
    if causal:
        t = q.size(-2)
        s = s.masked_fill(torch.triu(torch.ones(t, t, dtype=torch.bool), 1), float("-inf"))
    w = s.softmax(dim=-1)
    return w @ v, w


if __name__ == "__main__":
    def attention(q, k, v):
        return _attention(q, k, v)

    def causal_attention(q, k, v):
        return _attention(q, k, v, causal=True)

    honest = math.log(32)
    floor = noise_floor()
    print("предел без подглядывания ln 32:", round(honest, 2))
    print("предел без позиций:", floor)
    print("без маски, 5 зёрен:", noise_runs(attention))
    print("с маской, 5 зёрен:", noise_runs(causal_attention))
    assert abs(floor - 2.29) < 0.01
    assert floor < train_on_noise(attention) < honest - 0.5    # списывает, но до предела не дошла
    assert abs(train_on_noise(causal_attention) - honest) < 0.02  # с маской списывать неоткуда
    print("OK: noise_mask.py")
