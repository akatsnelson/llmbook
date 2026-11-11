"""Блочное причинное внимание с потоковым softmax (порядок вычислений FlashAttention) на PyTorch.

Полная версия листингов «Потоковый softmax по плиткам» и «Точное причинное внимание с потоковым softmax»
из главы 14 книги (раздел 14.7 «FlashAttention»). В книге
плитки называются плитками (аргумент tile), здесь — блоками (block_size). В книге оставлены ручной расчёт
на четырёх оценках и сокращённая функция; здесь весь код целиком, с теми же проверками и с несколькими дополнительными:
неполный последний блок, большой общий сдвиг логитов, половинная точность и сломанная версия без
поправки alpha.

Запуск из корня репозитория: python labs/ch14/flashattention_tiled.py
"""

import math

import torch


def naive_causal_attention(q, k, v):
    # q, k, v: [T, D]
    scores = q @ k.T / math.sqrt(q.size(-1))  # [T, T]
    t = q.size(0)
    future = torch.triu(
        torch.ones(t, t, dtype=torch.bool, device=q.device),
        diagonal=1,
    )
    scores = scores.masked_fill(future, float("-inf"))
    probabilities = scores.softmax(dim=-1)    # ещё один [T, T]
    return probabilities @ v                 # [T, D]


# --- Потоковый softmax одной строки ------------------------------------------------------------

def online_softmax_parts(parts):
    # Возвращает вероятности лишь для проверки; FlashAttention их не хранит.
    running_max = torch.tensor(float("-inf"))
    running_sum = torch.tensor(0.0)
    saved = []

    for values in parts:
        block_max = values.max()
        new_max = torch.maximum(running_max, block_max)
        correction = torch.exp(running_max - new_max)

        # Старые экспоненты нужно перемасштабировать.
        saved = [old * correction for old in saved]
        current = torch.exp(values - new_max)
        saved.append(current)
        running_sum = correction * running_sum + current.sum()
        running_max = new_max

    return torch.cat(saved) / running_sum


def online_attention_row(s, values, blocks, use_alpha=True):
    """Состояние (m, ell, o) одной строки внимания после всех блоков; итог равен o / ell."""
    m, ell, o = float("-inf"), 0.0, 0.0
    for block in blocks:
        new_m = max(m, s[block].max().item())
        alpha = math.exp(m - new_m) if use_alpha else 1.0
        p = torch.exp(s[block] - new_m)
        ell = alpha * ell + p.sum().item()
        o = alpha * o + (p @ values[block]).item()
        m = new_m
    return m, ell, o


# --- Блочное причинное внимание ---------------------------------------------------------------

def tiled_causal_attention(q, k, v, block_size=64, use_alpha=True):
    # Одна голова: [T, D]. Для B,H внешний код добавляет циклы/vmap.
    t, d = q.shape
    output = torch.empty_like(q)
    scale = 1 / math.sqrt(d)

    for q_start in range(0, t, block_size):
        q_stop = min(q_start + block_size, t)
        q_block = q[q_start:q_stop]                    # [Bq, D]
        q_pos = torch.arange(q_start, q_stop, device=q.device)

        m = torch.full((q_stop - q_start,), float("-inf"), device=q.device, dtype=torch.float32)
        ell = torch.zeros(q_stop - q_start, device=q.device, dtype=torch.float32)
        acc = torch.zeros(q_stop - q_start, d, device=q.device, dtype=torch.float32)

        # При причинном внимании блоки после q_stop не нужны.
        for k_start in range(0, q_stop, block_size):
            k_stop = min(k_start + block_size, t)
            k_block = k[k_start:k_stop]
            v_block = v[k_start:k_stop]
            k_pos = torch.arange(k_start, k_stop, device=q.device)

            # Оценки и накопители держим в FP32, даже если входы в половинной точности.
            scores = (q_block.float() @ k_block.float().T) * scale
            scores = scores.masked_fill(
                k_pos.unsqueeze(0) > q_pos.unsqueeze(1),
                float("-inf"),
            )

            block_max = scores.max(dim=1).values
            new_m = torch.maximum(m, block_max)
            alpha = torch.exp(m - new_m) if use_alpha else torch.ones_like(m)
            p = torch.exp(scores - new_m.unsqueeze(1))

            ell = alpha * ell + p.sum(dim=1)
            acc = alpha.unsqueeze(1) * acc + p @ v_block.float()
            m = new_m

        output[q_start:q_stop] = (acc / ell.unsqueeze(1)).to(q.dtype)
    return output


def main():
    # 1. Потоковый softmax на случайной строке из трёх неравных блоков.
    x = torch.randn(37)
    actual = online_softmax_parts([x[:8], x[8:21], x[21:]])
    torch.testing.assert_close(actual, x.softmax(dim=0))

    # 2. Ручной расчёт с рисунка: два блока по две оценки.
    s = torch.tensor([2.0, 0.0, 3.0, 2.0])
    values = torch.tensor([1.0, 3.0, 0.0, 2.0])
    blocks = (slice(0, 2), slice(2, 4))
    m, ell, o = online_attention_row(s, values, blocks)
    assert abs(ell - 1.786) < 1e-3 and abs(o - 1.253) < 1e-3
    assert abs(o / ell - (s.softmax(dim=0) @ values).item()) < 1e-6     # 0,702
    _, ell_bad, o_bad = online_attention_row(s, values, blocks, use_alpha=False)
    assert abs(o_bad / ell_bad - 0.856) < 1e-3                           # забыли alpha

    # 3. Сдвиг всех оценок на 100 не меняет результат: меняется только m.
    m100, ell100, o100 = online_attention_row(s + 100, values, blocks)
    assert abs(m100 - (m + 100)) < 1e-4 and abs(o100 / ell100 - o / ell) < 1e-6

    # 4. Блочное внимание против наивного; 129 не делится на 32, последний блок неполный.
    torch.manual_seed(7)
    q = torch.randn(129, 64)
    k = torch.randn(129, 64)
    v = torch.randn(129, 64)
    expected = naive_causal_attention(q, k, v)
    actual = tiled_causal_attention(q, k, v, block_size=32)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)

    # 5. Большие по модулю логиты: exp без вычитания максимума переполнился бы.
    q_big = q * 30
    torch.testing.assert_close(
        tiled_causal_attention(q_big, k, v, block_size=32),
        naive_causal_attention(q_big, k, v),
        rtol=1e-4, atol=1e-4,
    )

    # 6. Половинная точность входов; накопители в FP32.
    q16, k16, v16 = q.half(), k.half(), v.half()
    torch.testing.assert_close(
        tiled_causal_attention(q16, k16, v16, block_size=32).float(),
        expected, rtol=2e-2, atol=2e-2,
    )

    # 7. Сломанная версия без alpha: формы те же, числа другие.
    broken = tiled_causal_attention(q, k, v, block_size=32, use_alpha=False)
    assert broken.shape == expected.shape
    assert (broken - expected).abs().max() > 1e-2

    print("flashattention_tiled: все проверки пройдены")


if __name__ == "__main__":
    main()
