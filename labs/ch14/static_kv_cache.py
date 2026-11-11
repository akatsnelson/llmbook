"""Статический KV-буфер: место на max_tokens позиций выделяется заранее, новые ключи пишутся по индексу.

Дополнение к разделу 14.4 «Кэш ключей и значений» главы 14 (подраздел «Кэш в коде»).
В книге буфер описан словами: копирований torch.cat больше нет, зато каждый запрос держит место на весь
контекст (внутренняя фрагментация). Здесь класс целиком, проверка на одной голове внимания и подсчёт
пустующей доли для запроса на 300 токенов при контексте 8192.

Запуск из корня репозитория: python labs/ch14/static_kv_cache.py
"""

import math

import torch


class StaticKVCache:
    def __init__(self, batch, max_tokens, width, dtype=torch.float32):
        shape = (batch, max_tokens, width)
        self.k = torch.empty(shape, dtype=dtype)
        self.v = torch.empty(shape, dtype=dtype)
        self.length = 0

    def append(self, k_t, v_t):
        """k_t, v_t: [batch, 1, width]. Возвращает заполненную часть кэша."""
        if self.length >= self.k.size(1):
            raise RuntimeError("KV cache is full")
        self.k[:, self.length:self.length + 1].copy_(k_t)
        self.v[:, self.length:self.length + 1].copy_(v_t)
        self.length += 1
        return self.k[:, :self.length], self.v[:, :self.length]


def causal_attention(q, k, v):
    """q: [B, n, D], k и v: [B, m, D]; запросы — последние n из m позиций."""
    d = q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d)
    n, m = q.size(-2), k.size(-2)
    future = torch.ones(n, m, dtype=torch.bool).triu(m - n + 1)
    return scores.masked_fill(future, float("-inf")).softmax(dim=-1) @ v


def main():
    torch.manual_seed(17)
    B, T, D = 2, 6, 8
    W_q, W_k, W_v = (torch.randn(D, D) for _ in range(3))
    x = torch.randn(B, T, D)
    full = causal_attention(x @ W_q, x @ W_k, x @ W_v)

    cache = StaticKVCache(batch=B, max_tokens=8, width=D)
    outs = []
    for t in range(T):
        x_t = x[:, t:t + 1]
        K, V = cache.append(x_t @ W_k, x_t @ W_v)
        assert K.shape == (B, t + 1, D)
        outs.append(causal_attention(x_t @ W_q, K, V))
    torch.testing.assert_close(torch.cat(outs, dim=1), full)
    assert cache.length == T

    # Переполнение ловится явно, а не порчей памяти.
    cache.append(x[:, :1] @ W_k, x[:, :1] @ W_v)
    cache.append(x[:, :1] @ W_k, x[:, :1] @ W_v)
    try:
        cache.append(x[:, :1] @ W_k, x[:, :1] @ W_v)
    except RuntimeError as err:
        assert "full" in str(err)
    else:
        raise AssertionError("переполнение не замечено")

    # Внутренняя фрагментация: запрос на 300 токенов в буфере на 8192.
    wasted = 1 - 300 / 8192
    assert round(wasted * 100) == 96
    print(f"пошаговый расчёт со статическим буфером совпал с полным; пустует {wasted:.0%} буфера")


if __name__ == "__main__":
    main()
