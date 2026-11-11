"""Опыты подразделов главы 4 о частотах и раскладке пар RoPE и об ALiBi.

Печатает числа, которые стоят в тексте (подразделы
4.8.3 «Частоты пар», 4.8.4 «Раскладка пар», 4.8.6 «ALiBi»), и проверяет их утверждениями assert:

  1. граница RoFormer B(r) и оценки для вектора из единиц и ключа, настроенного на расстояние 100;
  2. перплексия BookLM, когда остановлены четыре пары подряд и одна пара 9;
  3. средняя длина ключей по парам в каждом блоке BookLM: пара 9 выделяется;
  4. веса, обученные на соседних парах, в коде с половинками, и перестановка строк q.weight и k.weight;
  5. наклоны ALiBi для 8 голов и расстояние, на котором штраф доходит до 10.

Запуск из корня репозитория: python labs/ch04/rope_pairs.py
Считается на CPU около полуминуты.
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import booklm.model as booklm_model                     # noqa: E402
from booklm.lab import load, perplexity, val_tokens   # noqa: E402

torch.set_num_threads(2)


# --- 1. Граница убывания из статьи RoFormer --------------------------------------------------------

def rotate_pair(x0, x1, angle):
    c, s = torch.cos(angle), torch.sin(angle)
    return c * x0 - s * x1, s * x0 + c * x1


def turn(x, r, base=10000.0):
    theta = base ** (-torch.arange(0, x.size(-1), 2) / x.size(-1))
    x0, x1 = rotate_pair(x[0::2], x[1::2], r * theta)
    return torch.stack((x0, x1), dim=-1).flatten()


def decay_bound(r, d_h=32, base=10000.0):
    theta = base ** (-torch.arange(0, d_h, 2) / d_h)
    arrows = torch.polar(torch.ones(d_h // 2), r * theta)
    return arrows.cumsum(0).abs().mean().item()


def check_bound():
    torch.manual_seed(0)
    ones, q = torch.ones(32), torch.randn(32)
    k_far = turn(q, 100)
    rows = {}
    for r in (0, 1, 3, 10, 100, 1000):
        rows[r] = (decay_bound(r), (ones @ turn(ones, -r)).item(), (q @ turn(k_far, -r)).item())
        print(f"r={r:4d}  граница {rows[r][0]:5.2f}  единицы {rows[r][1]:6.2f}  ключ на 100 {rows[r][2]:6.2f}")
    assert round(rows[0][0], 2) == 8.50 and round(rows[1000][0], 2) == 2.25
    assert rows[100][0] > rows[10][0]                       # граница убывает не монотонно
    assert round(rows[1000][1], 2) == 8.21
    best = max(rows, key=lambda r: rows[r][2])
    assert best == 100 and abs(rows[100][2] - (q @ q).item()) < 1e-4   # максимум |q|^2 на расстоянии 100


# --- 2–3. Какие пары BookLM несут позицию ----------------------------------------------------------

def stopped(model, cos, sin, pairs):
    model.rope_cos, model.rope_sin = cos.clone(), sin.clone()
    model.rope_cos[:, pairs] = 1.0
    model.rope_sin[:, pairs] = 0.0
    value = perplexity(model)
    model.rope_cos, model.rope_sin = cos, sin
    return value


@torch.no_grad()
def key_norms(model, windows=8, T=512):
    """Средняя длина пары ключа [блок, пара] на отложенных историях, до поворота RoPE."""
    x = val_tokens(windows * T).view(windows, T)
    out, hooks = {}, []
    for i, block in enumerate(model.blocks):
        hooks.append(block.attention.k.register_forward_hook(
            lambda m, inp, o, i=i: out.__setitem__(i, o)))
    model(x)
    for h in hooks:
        h.remove()
    return torch.stack([out[i].view(-1, 2, 16, 2).norm(dim=-1).mean(dim=(0, 1))
                        for i in range(len(model.blocks))])


def check_pairs():
    tok, model = load("base")
    cos, sin = model.rope_cos.clone(), model.rope_sin.clone()
    base = perplexity(model)
    print(f"все пары: {base:.4f}")
    groups = {first: stopped(model, cos, sin, slice(first, first + 4)) for first in (0, 4, 8, 12)}
    for first, value in groups.items():
        print(f"стоят пары {first}-{first + 3}: {value:.4f}")
    nine = stopped(model, cos, sin, [9])
    print(f"стоит пара 9: {nine:.4f}")
    assert round(base, 2) == 6.96
    assert round(groups[0], 2) == 15.11 and round(groups[4], 2) == 8.60
    assert round(groups[8], 2) == 24.53 and round(groups[12], 4) == 6.9581
    assert abs(groups[12] - base) < 1e-3                 # медленные пары позицию не несут
    assert round(nine, 2) == 13.57

    norms = key_norms(model)
    for i, row in enumerate(norms):
        others = torch.cat([row[:9], row[10:]])
        ratio = (row[9] / others.median()).item()
        print(f"блок {i}: длина ключа пары 9 {row[9]:.1f}, медиана остальных {others.median():.1f},"
              f" в {ratio:.1f} раза")
        if i > 0:
            assert 3.35 < ratio < 5.55 and row.argmax().item() == 9
    return tok, model


# --- 4. Соседние пары и половинки ------------------------------------------------------------------

def rope_halves(x, cos, sin):
    t, half = x.size(-2), x.size(-1) // 2
    cos, sin = cos[:t], sin[:t]
    x0, x1 = x[..., :half], x[..., half:]
    return torch.cat((x0 * cos - x1 * sin, x0 * sin + x1 * cos), dim=-1)


def check_layout():
    tok, model = load("base")
    neighbours = booklm_model.apply_rope
    ours = perplexity(model)
    try:
        booklm_model.apply_rope = rope_halves
        wrong = perplexity(model)
        perm = torch.cat([torch.arange(0, 32, 2), torch.arange(1, 32, 2)])
        with torch.no_grad():
            for block in model.blocks:
                att = block.attention
                for w, heads in ((att.q.weight, 8), (att.k.weight, 2)):
                    w.copy_(w.view(heads, 32, 256)[:, perm].reshape(heads * 32, 256))
        fixed = perplexity(model)
        # Та же перестановка, что в booklm/export_hf.py и в permute из convert_llama_weights_to_hf.
        x = torch.randn(1, 8, 512, 32)
        c, s = booklm_model.rope_tables(32, 512, 10000.0)
        torch.testing.assert_close(rope_halves(x[..., perm], c, s), neighbours(x, c, s)[..., perm])
    finally:
        booklm_model.apply_rope = neighbours
    print(f"соседние {ours:.4f}, половинки {wrong:.2f}, половинки с перестановкой {fixed:.4f}")
    assert round(ours, 4) == round(fixed, 4) == 6.9584
    assert round(wrong, 2) == 186.91
    perm8 = torch.cat([torch.arange(0, 8, 2), torch.arange(1, 8, 2)])   # к задаче главы: d_h = 8
    assert perm8.tolist() == [0, 2, 4, 6, 1, 3, 5, 7] and perm8.tolist().index(5) == 6


# --- 5. ALiBi ----------------------------------------------------------------------------------------

def check_alibi():
    m = 2.0 ** (-8 * torch.arange(1, 9) / 8)
    reach = (10 / m).int().tolist()
    print("наклоны ALiBi:", m.tolist(), "штраф 10 на расстоянии:", reach)
    assert reach == [20, 40, 80, 160, 320, 640, 1280, 2560]
    m16 = 2.0 ** (-8 * torch.arange(1, 17) / 16)          # к задаче главы: 16 голов
    assert abs(m16[0].item() - 2 ** -0.5) < 1e-7 and m16[-1].item() == 1 / 256


if __name__ == "__main__":
    check_bound()
    check_pairs()
    check_layout()
    check_alibi()
    print("OK: числа подразделов о парах RoPE и об ALiBi совпали")
