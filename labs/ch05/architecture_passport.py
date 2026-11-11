"""Паспорт модели из config.json: число параметров и KV-кэш без загрузки весов.

Полная версия листингов главы 5 «Архитектуры современных LLM»: «Чтение config.json», «Число параметров
и KV-кэш», «Конфигурация без двух полей» (раздел 5.2 «Паспорт модели») и «Общее и активное число
параметров» (подраздел 5.4.3). В книге оставлена сокращённая версия; здесь все проверки целиком.
Запуск из корня репозитория: python labs/ch05/architecture_passport.py
"""


def architecture_passport(config, default_tied=None):
    d, q_heads = config["hidden_size"], config["num_attention_heads"]
    tied = config.get("tie_word_embeddings", default_tied)
    if tied is None:
        raise KeyError("tie_word_embeddings")
    return {
        "V": config["vocab_size"], "d": d, "L": config["num_hidden_layers"],
        "H_q": q_heads,
        "H_kv": config.get("num_key_value_heads") or q_heads,   # нет поля: MHA
        "d_h": config.get("head_dim") or d // q_heads,          # нет поля: d / H_q
        "d_ff": config["intermediate_size"],
        "tied": tied,
        "window": config.get("sliding_window"),
        "E": config.get("num_local_experts") or config.get("num_experts") or 1,
        "k": config.get("num_experts_per_tok") or 1,
    }


def attention_parameters(p):          # W_Q, W_K, W_V, W_O без смещений
    d, d_h = p["d"], p["d_h"]
    return d * p["H_q"] * d_h + 2 * d * p["H_kv"] * d_h + p["H_q"] * d_h * d


def swiglu_parameters(p):             # W_gate, W_up, W_down
    return 3 * p["d"] * p["d_ff"]


def parameter_count(p):               # плотная модель: E = k = 1
    block = attention_parameters(p) + swiglu_parameters(p) + 2 * p["d"]
    embeddings = p["V"] * p["d"] * (1 if p["tied"] else 2)
    return embeddings + p["L"] * block + p["d"]


def kv_cache_bytes(p, tokens, bytes_per_number=2):
    return 2 * p["L"] * tokens * p["H_kv"] * p["d_h"] * bytes_per_number


def moe_parameter_count(p, active=False):
    experts = p["k"] if active else p["E"]
    router = p["d"] * p["E"] if p["E"] > 1 else 0
    block = (attention_parameters(p) + experts * swiglu_parameters(p)
             + router + 2 * p["d"])
    embeddings = p["V"] * p["d"] * (1 if p["tied"] else 2)
    return embeddings + p["L"] * block + p["d"]


# --- паспорт BookLM и Mistral 7B ---------------------------------------------
booklm_config = {
    "model_type": "llama", "vocab_size": 8192, "hidden_size": 256,
    "num_hidden_layers": 6, "num_attention_heads": 8, "num_key_value_heads": 2,
    "head_dim": 32, "intermediate_size": 768, "hidden_act": "silu",
    "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "max_position_embeddings": 512,
    "tie_word_embeddings": True, "attention_bias": False, "sliding_window": None,
}
booklm = architecture_passport(booklm_config)
assert parameter_count(booklm) == 6_622_464           # паспорт главы 4
assert kv_cache_bytes(booklm, tokens=512) == 768 * 1024

mistral = architecture_passport({                      # Mistral 7B v0.1
    "vocab_size": 32000, "hidden_size": 4096, "num_hidden_layers": 32,
    "num_attention_heads": 32, "num_key_value_heads": 8,
    "intermediate_size": 14336, "tie_word_embeddings": False,
    "sliding_window": 4096,
})
assert mistral["H_q"] // mistral["H_kv"] == booklm["H_q"] // booklm["H_kv"] == 4
assert parameter_count(mistral) == 7_241_732_096
assert kv_cache_bytes(mistral, tokens=1) == 128 * 1024

# --- отсутствующее поле tie_word_embeddings: решает класс модели --------------
no_tie = {k: v for k, v in booklm_config.items() if k != "tie_word_embeddings"}
try:
    architecture_passport(no_tie)
    raise AssertionError("tie_word_embeddings")
except KeyError:
    pass
assert parameter_count(architecture_passport(no_tie, default_tied=False)) \
    == parameter_count(booklm) + 8192 * 256

# --- общее и активное число параметров смеси экспертов -----------------------
# плотная модель: E = k = 1, оба подсчёта совпадают
assert moe_parameter_count(booklm) == parameter_count(booklm)

booklm_moe = architecture_passport(dict(booklm_config, num_local_experts=4,
                                        num_experts_per_tok=2))
assert moe_parameter_count(booklm_moe) == 17_245_440
assert moe_parameter_count(booklm_moe, active=True) == 10_167_552
# KV-кэш не изменился: эксперты живут в полносвязной ветви
assert kv_cache_bytes(booklm_moe, 512) == kv_cache_bytes(booklm, 512)

mixtral = architecture_passport({                      # Mixtral 8x7B
    "vocab_size": 32000, "hidden_size": 4096, "num_hidden_layers": 32,
    "num_attention_heads": 32, "num_key_value_heads": 8,
    "intermediate_size": 14336, "tie_word_embeddings": False,
    "num_local_experts": 8, "num_experts_per_tok": 2,
})
assert moe_parameter_count(mixtral) == 46_702_792_704
assert moe_parameter_count(mixtral, active=True) == 12_879_925_248

# --- BookLM с одной KV-головой (листинг «Три варианта BookLM», раздел 5.7) -----
booklm_mqa = architecture_passport(dict(booklm_config, num_key_value_heads=1))
assert parameter_count(booklm_mqa) == 6_524_160
assert kv_cache_bytes(booklm_mqa, 512) == 384 * 1024

print("architecture_passport: все проверки пройдены")
