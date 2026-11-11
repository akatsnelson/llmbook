"""Глава 11, раздел 11.1 «Пять сроков жизни состояния», подраздел «Цена состояния в байтах»:
листинг «Когда KV-кэши перевешивают веса» (полная версия).

В книге оставлена сокращённая версия; этот файл самодостаточен и
выполняется в ноутбуке главы notebooks/11_memory.ipynb.

Считает байты каждого состояния BookLM по паспорту модели (веса, скрытое
состояние, оценки внимания, KV-кэш, история и блок памяти) и находит, при
скольких одновременных полных запросах KV-кэши перевешивают веса (17).

Запуск из корня репозитория: python labs/ch11/lifetime_costs.py
"""

# Паспорт BookLM (глава 4): размеры, общие для всего раздела
V, D, LAYERS, Q_HEADS, KV_HEADS, HEAD_DIM, T_MAX = 8192, 256, 6, 8, 2, 32, 512
PARAMS = 6_622_464
BF16 = 2                                  # байта на число

def weights_bytes(params=PARAMS, b=BF16):
    return params * b

def hidden_state_bytes(tokens, d=D, b=BF16):        # [T, d] между блоками
    return tokens * d * b

def scores_bytes(tokens, heads=Q_HEADS, b=BF16):    # [H_q, T, T] одного блока
    return heads * tokens * tokens * b

def kv_bytes(tokens, layers=LAYERS, kv_heads=KV_HEADS, head_dim=HEAD_DIM, b=BF16):
    return 2 * layers * tokens * kv_heads * head_dim * b   # формула главы 4

def last_logits_bytes(v=V, b=BF16):                  # [V] последней позиции
    return v * b

assert weights_bytes() == 13_244_928
assert hidden_state_bytes(T_MAX) == 262_144          # 256 КБ
assert scores_bytes(T_MAX) == 4_194_304              # 4 МБ
assert kv_bytes(1) == 1536
assert kv_bytes(T_MAX) == 786_432                    # 768 КБ, как в паспорте

# История и память платят токенами в окне на каждом ходу
HISTORY_TOKENS, MEMORY_TOKENS = 240, 64
assert kv_bytes(HISTORY_TOKENS) == 368_640           # 360 КБ
assert kv_bytes(MEMORY_TOKENS) == 98_304             # 96 КБ на запрос

# Один запрос в худший момент: меньше весов
one_request = (hidden_state_bytes(T_MAX) + scores_bytes(T_MAX)
               + kv_bytes(T_MAX) + last_logits_bytes())
assert one_request == 5_259_264
assert one_request < weights_bytes()

# Сколько полных запросов нужно, чтобы KV-кэши перевесили веса
def requests_to_outweigh(tokens=T_MAX):
    n = 1
    while n * kv_bytes(tokens) <= weights_bytes():
        n += 1
    return n

assert 16 * kv_bytes(T_MAX) == 12_582_912 < weights_bytes()
assert 17 * kv_bytes(T_MAX) == 13_369_344 > weights_bytes()
assert requests_to_outweigh() == 17


if __name__ == "__main__":
    print("OK: lifetime_costs.py")
