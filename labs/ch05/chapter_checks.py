"""Все листинги главы 5 «Архитектуры современных LLM» с проверками assert вместо печати.

В книге каждая функция появляется в своём разделе и печатает результат; здесь те же функции собраны в
одном файле, а каждое число, которое глава показывает в выводе, проверяется утверждением. После любой правки
кода файл скажет, какое из чисел главы перестало сходиться.
Запуск из корня репозитория: python labs/ch05/chapter_checks.py
"""

import math

import torch


# ----------------------------------------------------------------------------- config.json и подсчёт

def architecture_passport(config, default_tied=None):
    d, q_heads = config["hidden_size"], config["num_attention_heads"]
    tied = config.get("tie_word_embeddings", default_tied)
    if tied is None:              # умолчание задаёт класс модели
        raise KeyError("tie_word_embeddings")
    return {
        "V": config["vocab_size"], "d": d, "L": config["num_hidden_layers"],
        "H_q": q_heads, "d_ff": config["intermediate_size"], "tied": tied,
        "H_kv": config.get("num_key_value_heads") or q_heads,  # нет: MHA
        "d_h": config.get("head_dim") or d // q_heads,  # нет: d/H_q
        "window": config.get("sliding_window"),
        "E": (config.get("num_local_experts")
              or config.get("num_experts") or 1),
        "k": config.get("num_experts_per_tok") or 1,
    }


def parameter_count(p):                 # плотная модель: E = k = 1
    d, d_h = p["d"], p["d_h"]
    attention = (d * p["H_q"] * d_h + 2 * d * p["H_kv"] * d_h
                 + p["H_q"] * d_h * d)        # W_Q, W_K и W_V, W_O
    block = attention + 3 * d * p["d_ff"] + 2 * d  # + SwiGLU + две нормы
    return p["V"] * d * (1 if p["tied"] else 2) + p["L"] * block + d


def kv_cache_bytes(p, tokens, bytes_per_number=2):
    return 2 * p["L"] * tokens * p["H_kv"] * p["d_h"] * bytes_per_number


MISTRAL = architecture_passport({                      # Mistral 7B v0.1
    "vocab_size": 32000, "hidden_size": 4096, "num_hidden_layers": 32,
    "num_attention_heads": 32, "num_key_value_heads": 8,
    "intermediate_size": 14336, "tie_word_embeddings": False,
    "sliding_window": 4096})
BOOKLM = architecture_passport({                       # BookLM, глава 4
    "vocab_size": 8192, "hidden_size": 256, "num_hidden_layers": 6,
    "num_attention_heads": 8, "num_key_value_heads": 2,
    "intermediate_size": 768, "tie_word_embeddings": True})


def check_passport():
    assert MISTRAL["d_h"] == 128 and MISTRAL["H_kv"] == 8
    assert parameter_count(MISTRAL) == 7_241_732_096           # «7B» в названии
    assert kv_cache_bytes(MISTRAL, tokens=1) == 128 * 1024
    assert BOOKLM["d_h"] == 32
    assert parameter_count(BOOKLM) == 6_622_464                # справочник модели
    assert kv_cache_bytes(BOOKLM, tokens=1) == 1536
    assert kv_cache_bytes(BOOKLM, tokens=512) == 768 * 1024

    llama = {"vocab_size": 32000, "hidden_size": 4096, "num_hidden_layers": 32,
             "num_attention_heads": 32, "intermediate_size": 11008}
    try:
        architecture_passport(llama)
    except KeyError:
        pass
    else:
        raise AssertionError("без tie_word_embeddings и default_tied функция должна отказаться")
    p = architecture_passport(llama, default_tied=False)
    assert p["H_kv"] == 32
    assert parameter_count(p) == 6_738_415_616
    assert kv_cache_bytes(p, tokens=1) == 512 * 1024
    # молчаливое true занизило бы подсчёт ровно на V * d
    tied = architecture_passport(llama, default_tied=True)
    assert parameter_count(p) - parameter_count(tied) == 32000 * 4096

    gemma = architecture_passport({"vocab_size": 256000, "hidden_size": 3072,
                                   "num_hidden_layers": 28, "num_attention_heads": 16,
                                   "num_key_value_heads": 16, "head_dim": 256,
                                   "intermediate_size": 24576}, default_tied=True)
    assert gemma["H_q"] * gemma["d_h"] == 4096 != gemma["d"]   # W_Q прямоугольная


# ----------------------------------------------------------------------------- скользящее окно

def window_mask(T, W):
    t = torch.arange(T)[:, None]          # запросы идут по строкам
    j = torch.arange(T)[None, :]          # ключи идут по столбцам
    return (j <= t) & (j > t - W)         # True = ключ виден


def broken_window_mask(T, W):             # задача главы: ошибка на единицу
    t = torch.arange(T)[:, None]
    j = torch.arange(T)[None, :]
    return (j <= t) & (j >= t - W)


def reach(T, W, layers):
    step = window_mask(T, W).int()
    seen = step.clone()
    for _ in range(layers - 1):
        seen = ((seen @ step) > 0).int()   # кого видели те, кого вижу я
    return seen.bool()


def layer_types(config):
    kinds = config.get("layer_types")
    if kinds is None:      # нет списка: окно во всех слоях или ни в одном
        kind = ("sliding_attention" if config.get("sliding_window")
                else "full_attention")
        kinds = [kind] * config["num_hidden_layers"]
    return kinds


def interleaved_kv_bytes(p, kinds, tokens, bytes_per_number=2):
    kept = sum(min(tokens, p["window"]) if kind == "sliding_attention"
               else tokens for kind in kinds)          # токено-слои в кэше
    return 2 * kept * p["H_kv"] * p["d_h"] * bytes_per_number


FIVE_TO_ONE = ["sliding_attention"] * 5 + ["full_attention"]


def check_window():
    m = window_mask(6, 3)
    assert m.int().tolist() == [[1, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0],
                                [0, 1, 1, 1, 0, 0], [0, 0, 1, 1, 1, 0], [0, 0, 0, 1, 1, 1]]
    assert m.sum(dim=1).tolist() == [1, 2, 3, 3, 3, 3] and int(m.sum()) == 15
    assert m[5].nonzero().flatten().tolist() == [3, 4, 5]
    assert torch.equal(window_mask(6, 6), torch.ones(6, 6).tril().bool())
    # ошибка на единицу: строка видит W + 1 ключей, а сумма по маске тоже растёт
    assert int(broken_window_mask(6, 3).sum(dim=1).max()) == 4

    for layers in (1, 2, 3, 6):
        farthest = int(reach(64, 3, layers)[63].nonzero().min())
        assert 63 - farthest == layers * (3 - 1)          # горизонт l(W - 1)

    local = dict(BOOKLM, window=128)
    assert interleaved_kv_bytes(local, ["full_attention"] * 6, 512) == 768 * 1024
    assert interleaved_kv_bytes(local, FIVE_TO_ONE, 512) == 288 * 1024
    assert interleaved_kv_bytes(local, FIVE_TO_ONE, 128) == kv_cache_bytes(BOOKLM, 128)
    long = dict(BOOKLM, window=1024)
    ratio = interleaved_kv_bytes(long, FIVE_TO_ONE, 32768) / kv_cache_bytes(long, 32768)
    assert round(ratio, 3) == 0.193
    assert layer_types({"num_hidden_layers": 2, "sliding_window": 4096}) == ["sliding_attention"] * 2
    assert layer_types({"num_hidden_layers": 2}) == ["full_attention"] * 2


# ----------------------------------------------------------------------------- смесь экспертов

def route(scores, k):
    chosen = scores.topk(k).indices          # [k]
    weights = scores[chosen].softmax(dim=0)  # softmax только по выбранным
    return chosen, weights


def moe_forward(x, router, experts, k):      # x: [N, d], N = B * T
    top = router(x).topk(k, dim=-1)          # номера и оценки: [N, k]
    weights = top.values.softmax(dim=-1)     # [N, k]
    y = torch.zeros_like(x)
    for e, expert in enumerate(experts):
        rows, slot = (top.indices == e).nonzero(as_tuple=True)
        if rows.numel():                     # один вызов на эксперта
            y[rows] += weights[rows, slot, None] * expert(x[rows])
    return y


def moe_parameter_count(p, active=False):
    extra = (p["k"] if active else p["E"]) - 1   # сверх одной SwiGLU
    router = p["d"] * p["E"] if p["E"] > 1 else 0
    added = extra * 3 * p["d"] * p["d_ff"] + router   # на один блок
    return parameter_count(p) + p["L"] * added


def capacity(tokens, k, experts, factor):
    return math.ceil(factor * tokens * k / experts)


def dispatch(chosen, experts, cap):
    load, kept = [0] * experts, []
    for token, pair in enumerate(chosen.tolist()):  # по порядку
        for e in pair:
            if load[e] < cap:
                load[e] += 1
                kept.append((token, e))
    return kept, load


def balance_loss(probs, chosen, experts):
    counts = torch.bincount(chosen.flatten(), minlength=experts)
    f = counts.float() / chosen.numel()        # доли назначений
    return experts * (f * probs.mean(dim=0)).sum()


def check_moe():
    r = torch.tensor([2.0, 0.5, 1.0, -1.0])
    chosen, weights = route(r, k=2)
    assert chosen.tolist() == [0, 2]
    torch.testing.assert_close(weights, torch.tensor([0.7311, 0.2689]), atol=1e-4, rtol=0)
    p_all = r.softmax(dim=0)                  # вторая запись маршрута
    assert abs(p_all[chosen].sum().item() - 0.834) < 1e-3
    torch.testing.assert_close(p_all[chosen] / p_all[chosen].sum(), weights)
    # шаг обучения 2.0 -> 1.9: выбор и счёт f те же, вероятность P сдвинулась
    nudged = torch.tensor([1.9, 0.5, 1.0, -1.0])
    assert route(nudged, 2)[0].tolist() == [0, 2]
    assert abs(nudged.softmax(0)[0].item() - 0.585) < 1e-3

    indices = torch.tensor([[0, 2], [0, 1], [2, 0]])
    rows, slot = (indices == 0).nonzero(as_tuple=True)
    assert rows.tolist() == [0, 1, 2] and slot.tolist() == [0, 0, 1]

    torch.manual_seed(5)
    router = torch.nn.Linear(8, 4, bias=False)
    experts = [torch.nn.Linear(8, 8, bias=False) for _ in range(4)]
    x = torch.randn(10, 8)
    with torch.no_grad():
        y = moe_forward(x, router, experts, k=2)
        naive = []
        for row in x:                         # эталон: токен за токеном
            c, w = route(router(row), k=2)
            naive.append(w[0] * experts[int(c[0])](row) + w[1] * experts[int(c[1])](row))
        load = torch.bincount(router(x).topk(2).indices.flatten(), minlength=4)
    torch.testing.assert_close(y, torch.stack(naive), atol=1e-6, rtol=1e-5)
    assert load.tolist() == [5, 5, 3, 7]

    assert moe_parameter_count(BOOKLM) == parameter_count(BOOKLM)          # E = k = 1
    booklm_moe = dict(BOOKLM, E=4, k=2)
    assert moe_parameter_count(booklm_moe) == 17_245_440
    assert moe_parameter_count(booklm_moe, active=True) == 10_167_552
    assert kv_cache_bytes(booklm_moe, 512) == kv_cache_bytes(BOOKLM, 512)   # кэш не изменился
    mixtral = dict(MISTRAL, E=8, k=2)
    assert moe_parameter_count(mixtral) == 46_702_792_704
    assert moe_parameter_count(mixtral, active=True) == 12_879_925_248

    chosen = torch.tensor([[0, 2], [0, 1], [0, 2], [0, 3],
                           [0, 1], [0, 2], [0, 2], [1, 3]])
    assert capacity(8, 2, 4, factor=1.0) == 4
    kept, load = dispatch(chosen, 4, cap=4)
    assert load == [4, 3, 4, 2] and 16 - len(kept) == 3
    assert capacity(8, 2, 4, factor=1.25) == 5
    kept, load = dispatch(chosen, 4, cap=5)
    assert load == [5, 3, 4, 2] and 16 - len(kept) == 2
    # теряют эксперта последние токены пакета
    kept, _ = dispatch(chosen, 4, cap=4)
    lost = [t for t in range(8) if (t, 0) not in kept and 0 in chosen[t].tolist()]
    assert lost == [4, 5, 6]

    even = balance_loss(torch.full((8, 4), 0.25), torch.tensor([[0, 1], [2, 3]] * 4), 4)
    skewed = balance_loss(torch.tensor([[0.97, 0.01, 0.01, 0.01]] * 8),
                          torch.tensor([[0, 1]] * 8), 4)
    assert abs(even.item() - 1.0) < 1e-6 and abs(skewed.item() - 1.96) < 1e-5


# ----------------------------------------------------------------------------- незнакомая модель

GPT_OSS = {                                     # config.json gpt-oss-120b
    "vocab_size": 201088, "hidden_size": 2880, "num_hidden_layers": 36,
    "num_attention_heads": 64, "num_key_value_heads": 8, "head_dim": 64,
    "intermediate_size": 2880, "tie_word_embeddings": False,
    "num_local_experts": 128, "num_experts_per_tok": 4,
    "sliding_window": 128, "attention_bias": True,
    "layer_types": ["sliding_attention", "full_attention"] * 18}


def check_gpt_oss():
    p = architecture_passport(GPT_OSS)
    total = moe_parameter_count(p)
    active = moe_parameter_count(p, active=True)
    embedding = p["V"] * p["d"]
    assert round(total / 1e9, 2) == 116.79          # карточка: 116,83 со смещениями
    assert round(active / 1e9, 2) == 5.71
    assert round((active - embedding) / 1e9, 2) == 5.13   # карточка: 5,13
    kv = interleaved_kv_bytes(p, layer_types(GPT_OSS), 131072)
    assert round(kv / 2**30, 2) == 4.50
    assert kv_cache_bytes(p, 131072) / 2**30 == 9.0
    in_experts = p["L"] * p["E"] * 3 * p["d"] * p["d_ff"]
    size = in_experts * 4.25 / 8 + (total - in_experts) * 2
    assert round(in_experts / total, 3) == 0.982
    assert round(size / 2**30, 1) == 60.7           # карточка: 60,8 ГБ

    small = architecture_passport(dict(GPT_OSS, num_hidden_layers=24, num_local_experts=32,
                                       layer_types=GPT_OSS["layer_types"][:24]))
    assert round(moe_parameter_count(small) / 1e9, 1) == 20.9          # gpt-oss-20b
    assert round((moe_parameter_count(small, active=True) - embedding) / 1e9, 2) == 3.61


def check_families():
    dense = {"vocab_size": 128256, "hidden_size": 4096, "num_hidden_layers": 32,
             "num_attention_heads": 32, "num_key_value_heads": 8, "intermediate_size": 14336}
    rows = {
        "Llama 3 8B": dense,
        "Mistral 7B": dict(dense, vocab_size=32000, sliding_window=4096),
        "Mixtral 8x7B": dict(dense, vocab_size=32000, num_local_experts=8, num_experts_per_tok=2),
        "Gemma 2 9B": {"vocab_size": 256000, "hidden_size": 3584, "num_hidden_layers": 42,
                       "num_attention_heads": 16, "num_key_value_heads": 8, "head_dim": 256,
                       "intermediate_size": 14336, "tie_word_embeddings": True,
                       "sliding_window": 4096,
                       "layer_types": ["sliding_attention", "full_attention"] * 21},
        "Qwen3-235B": {"vocab_size": 151936, "hidden_size": 4096, "num_hidden_layers": 94,
                       "num_attention_heads": 64, "num_key_value_heads": 4, "head_dim": 128,
                       "intermediate_size": 1536, "num_experts": 128, "num_experts_per_tok": 8},
        "gpt-oss-120b": GPT_OSS,
    }
    want = {"Llama 3 8B": (8.03, 8.03, 1.00), "Mistral 7B": (7.24, 7.24, 0.50),
            "Mixtral 8x7B": (46.70, 12.88, 1.00), "Gemma 2 9B": (9.24, 9.24, 1.97),
            "Qwen3-235B": (235.09, 22.19, 1.47), "gpt-oss-120b": (116.79, 5.71, 0.29)}
    for name, config in rows.items():
        p = architecture_passport(config, default_tied=False)
        kv = interleaved_kv_bytes(p, layer_types(config), 8192) / 2**30
        got = (round(moe_parameter_count(p) / 1e9, 2),
               round(moe_parameter_count(p, active=True) / 1e9, 2), round(kv, 2))
        assert got == want[name], (name, got)

    variants = {"mqa": (dict(BOOKLM, H_kv=1), ["full_attention"] * 6, 6_524_160, 384),
                "window": (dict(BOOKLM, window=128), FIVE_TO_ONE, 6_622_464, 288),
                "moe": (dict(BOOKLM, E=4, k=2), ["full_attention"] * 6, 17_245_440, 768)}
    for name, (p, kinds, params, kib) in variants.items():
        assert moe_parameter_count(p) == params, name
        assert interleaved_kv_bytes(p, kinds, 512) // 1024 == kib, name
    assert parameter_count(BOOKLM) - parameter_count(dict(BOOKLM, H_kv=1)) == 6 * 2 * 256 * 32


if __name__ == "__main__":
    check_passport()
    check_window()
    check_moe()
    check_gpt_oss()
    check_families()
    print("Все числа главы 5 сходятся.")
