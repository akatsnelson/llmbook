"""Проверки листингов начала главы 14 и раздела о KV-кэше.

Глава 14: разделы 14.1–14.3 (учебный стенд, температура, фильтры хвоста) и
раздел 14.4 «Кэш ключей и значений» (кэш одной головы, позиция RoPE, память кэша BookLM,
калькулятор). В книге листинги печатают числа; здесь те же вычисления с утверждениями assert.

Запуск из корня репозитория: python labs/ch14/inference_checks.py
"""

import math

import torch


# --- Учебный стенд и цикл генерации -----------------------------------------------------------------
def check_toy_generate():
    torch.manual_seed(0)
    V, d = 8, 4
    E, W_out = torch.randn(V, d), torch.randn(d, V)
    seen = []

    def toy_model(ids):
        seen.append(ids.size(1))
        h = E[ids].cumsum(dim=1) / torch.arange(1, ids.size(1) + 1)[None, :, None]
        return h @ W_out

    def generate(model, ids, steps):
        for _ in range(steps):
            next_id = model(ids)[:, -1].argmax(dim=-1, keepdim=True)
            ids = torch.cat([ids, next_id], dim=1)
        return ids

    prompt = torch.tensor([[3, 1, 4, 1]])
    assert toy_model(prompt).shape == (1, 4, V)
    seen.clear()
    out = generate(toy_model, prompt, steps=3)
    assert out.tolist() == [[3, 1, 4, 1, 5, 4, 7]]
    assert seen == [4, 5, 6]
    # ответ из 500 токенов после 4 токенов запроса без кэша
    assert sum(range(4, 504)) == 126_750


# --- Температура и фильтры хвоста ------------------------------------------------------------------------
Z = torch.tensor([1.03, -0.22, 2.08, 0.19, -0.91, 0.59, 1.49, -0.51])


def check_sampling():
    leaders = [torch.softmax(Z / tau, dim=-1).max().item() for tau in (0.5, 1.0, 2.0)]
    assert leaders[0] > leaders[1] > leaders[2]
    # температура не меняет порядок токенов
    for tau in (0.5, 2.0):
        assert torch.equal(torch.softmax(Z / tau, -1).argsort(), Z.argsort())
    probs = torch.softmax(Z, dim=-1)
    top_k = torch.zeros_like(probs, dtype=torch.bool)
    top_k[probs.topk(3).indices] = True
    sorted_p, idx = probs.sort(descending=True)
    top_p = torch.zeros_like(probs, dtype=torch.bool)
    top_p[idx] = (sorted_p.cumsum(0) - sorted_p) < 0.9
    min_p = probs >= 0.2 * probs.max()
    assert [int(m.sum()) for m in (top_k, top_p, min_p)] == [3, 5, 4]
    q = torch.where(top_k, probs, torch.zeros_like(probs))
    q = q / q.sum()
    assert torch.allclose(q[[2, 6, 0]], torch.tensor([0.53, 0.29, 0.18]), atol=0.01)
    assert math.isclose(q.sum().item(), 1.0, rel_tol=1e-6)


# --- KV-кэш одной головы и позиция RoPE ---------------------------------------------------------------------
def check_kv_cache_and_rope():
    torch.manual_seed(17)
    D = 8
    W_q, W_k, W_v = (torch.randn(D, D) for _ in range(3))
    x = torch.randn(5, D)

    def causal_attention(q, k, v):
        scores = q @ k.T / math.sqrt(D)
        n, m = len(q), len(k)
        future = torch.ones(n, m, dtype=torch.bool).triu(m - n + 1)
        return scores.masked_fill(future, float("-inf")).softmax(dim=-1) @ v

    full = causal_attention(x @ W_q, x @ W_k, x @ W_v)
    K_cache, V_cache, outs = [], [], []
    for t in range(len(x)):
        x_t = x[t:t + 1]
        K_cache.append(x_t @ W_k)
        V_cache.append(x_t @ W_v)
        K, V = torch.cat(K_cache), torch.cat(V_cache)
        assert K.shape == (t + 1, D)
        outs.append(causal_attention(x_t @ W_q, K, V))
    torch.testing.assert_close(torch.cat(outs), full)

    def rope(x, pos):
        theta = 10000 ** (-torch.arange(0, D, 2) / D)
        angle = pos[:, None] * theta
        cos, sin = angle.cos(), angle.sin()
        x1, x2 = x[:, 0::2], x[:, 1::2]
        return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(1)

    pos = torch.arange(5.0)
    full = causal_attention(rope(x @ W_q, pos), rope(x @ W_k, pos), x @ W_v)

    def cached_run(position_of):
        K_cache, V_cache, outs = [], [], []
        for t in range(len(x)):
            x_t, p = x[t:t + 1], torch.tensor([float(position_of(t))])
            K_cache.append(rope(x_t @ W_k, p))
            V_cache.append(x_t @ W_v)
            outs.append(causal_attention(rope(x_t @ W_q, p), torch.cat(K_cache), torch.cat(V_cache)))
        return torch.cat(outs)

    good, bad = cached_run(lambda t: t), cached_run(lambda t: 0)
    torch.testing.assert_close(good, full)
    assert bad.shape == full.shape                       # формы совпали...
    assert (bad - full).abs().max() > 1.0                # ...а числа разошлись


# --- Память KV-кэша ----------------------------------------------------------------------------------------
def kv_bytes(layers, batch, tokens, kv_heads, head_dim, dtype_bytes=2):
    return 2 * layers * batch * tokens * kv_heads * head_dim * dtype_bytes


def check_memory():
    L, H, H_KV, D_h, T_max = 6, 8, 2, 32, 512
    beta = torch.empty(0, dtype=torch.bfloat16).element_size()
    assert beta == 2
    assert (D_h * beta, 2 * H_KV * D_h * beta) == (64, 256)
    assert kv_bytes(L, 1, 1, H_KV, D_h) == 1536
    assert kv_bytes(L, 1, T_max, H_KV, D_h) == 786_432 == 768 * 2**10
    cache = torch.zeros(L, 2, 1, H_KV, T_max, D_h, dtype=torch.bfloat16)
    assert cache.numel() * cache.element_size() == 786_432
    assert kv_bytes(L, 1, T_max, H, D_h) == 3 * 2**20          # MHA: 3 МБ
    assert kv_bytes(L, 1, T_max, 1, D_h) == 384 * 2**10        # MQA: 384 КБ
    # веса BookLM в BF16 и сколько контекстов весят столько же
    weights = 6_622_464 * 2
    assert round(weights / 2**20, 2) == 12.63
    assert round(weights / 786_432) == 17
    # модели из калькулятора
    assert kv_bytes(32, 1, 1, 8, 128) == 128 * 2**10
    assert kv_bytes(32, 1, 8192, 8, 128) == 2**30
    assert kv_bytes(32, 16, 8192, 8, 128) == 16 * 2**30
    assert round(kv_bytes(28, 1, 4096, 8, 128) / 2**20) == 448
    assert round(8e9 * 4.9 / 8 / 2**30, 2) == 4.56
    assert round(3e9 * 4.5 / 8 / 2**30, 2) == 1.57
    assert round(70e9 * 4.9 / 8 / 2**30, 2) == 39.93


def main():
    check_toy_generate()
    check_sampling()
    check_kv_cache_and_rope()
    check_memory()
    print("все проверки начала главы и KV-кэша прошли")


if __name__ == "__main__":
    main()
