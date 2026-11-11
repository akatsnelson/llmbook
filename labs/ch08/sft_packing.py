"""Упаковка диалогов с блочной маской и тест влияния.

Код к подразделу 8.1.4 «Упаковка диалогов» главы 8 (в книге он описан словами): укладка диалогов в окна,
блочно-диагональная маска, тест влияния и опыт со сбросом позиций при RoPE и при таблице абсолютных позиций.
Выполняется в ноутбуке главы 08_lora_sft.ipynb и отдельно.
Запуск из корня репозитория: python labs/ch08/sft_packing.py
"""
import math

import torch


def pack_lengths(lengths, T):
    # Длинные раньше, каждый в первое окно, где хватает места. Диалог не режем.
    bins = []
    for n in sorted(lengths, reverse=True):
        for b in bins:
            if sum(b) + n <= T:
                b.append(n)
                break
        else:
            bins.append([n])
    return bins


lengths, T = [120, 380, 60, 450], 512
padded_eff = sum(lengths) / (len(lengths) * T)          # каждый в своём окне
bins = pack_lengths(lengths, T)
packed_eff = sum(lengths) / (len(bins) * T)
assert abs(padded_eff - 1010 / 2048) < 1e-12           # 0,493
assert len(bins) == 2 and abs(packed_eff - 1010 / 1024) < 1e-12   # 0,986


def pack(dialogs, T, pad_id=0):
    # -> ids, position_ids, doc_ids: каждый [T]
    ids, pos, doc = [], [], []
    for d, toks in enumerate(dialogs):
        ids += toks
        pos += range(len(toks))                         # позиции заново с нуля
        doc += [d] * len(toks)
    n_pad = T - len(ids)
    assert n_pad >= 0, "диалоги не помещаются в окно"
    ids += [pad_id] * n_pad
    pos += [0] * n_pad
    doc += [-1] * n_pad
    return torch.tensor(ids), torch.tensor(pos), torch.tensor(doc)


def block_causal_mask(doc):
    t = doc.numel()
    causal = torch.tril(torch.ones(t, t, dtype=torch.bool))
    same_doc = doc[:, None] == doc[None, :]
    return causal & same_doc                            # [T, T], True = можно смотреть


def plain_causal_mask(doc):
    t = doc.numel()
    return torch.tril(torch.ones(t, t, dtype=torch.bool))


def rope(x, pos, base=10000.0):
    # x: [T, d_h]; пары (x_{2k}, x_{2k+1}) поворачиваются на угол pos * theta_k,
    # как apply_rope в главе 4
    d_h = x.size(-1)
    theta = base ** (-torch.arange(0, d_h, 2, dtype=x.dtype) / d_h)   # [d_h/2]
    ang = pos[:, None].to(x.dtype) * theta[None, :]                   # [T, d_h/2]
    x0, x1 = x[:, 0::2], x[:, 1::2]
    rotated = torch.stack((x0 * ang.cos() - x1 * ang.sin(),
                           x0 * ang.sin() + x1 * ang.cos()), dim=-1)
    return rotated.flatten(-2)


torch.manual_seed(8)
d, d_h = 16, 8
emb = torch.randn(32, d, dtype=torch.float64)           # маленькая таблица эмбеддингов
W_q, W_k, W_v = (torch.randn(d, d_h, dtype=torch.float64) / d**0.5 for _ in range(3))
P_abs = torch.randn(8, d, dtype=torch.float64)          # таблица абсолютных позиций


def attention(ids, pos, allowed, rotary=True):
    x = emb[ids] if rotary else emb[ids] + P_abs[pos]
    q, k, v = x @ W_q, x @ W_k, x @ W_v
    if rotary:
        q, k = rope(q, pos), rope(k, pos)
    scores = (q @ k.T / math.sqrt(d_h)).masked_fill(~allowed, float("-inf"))
    return scores.softmax(dim=-1) @ v                   # [T, d_h]


dialog_a, dialog_b = [3, 11, 12, 5, 4], [3, 13, 5]      # A: позиции 0-4, B: 5-7
ids, pos, doc = pack([dialog_a, dialog_b], T=8)
assert pos.tolist() == [0, 1, 2, 3, 4, 0, 1, 2]


def influence_leak(mask_fn):
    ids2 = ids.clone()
    ids2[1:4] = torch.tensor([20, 21, 22])              # меняем только текст A
    allowed = mask_fn(doc)
    before, after = attention(ids, pos, allowed), attention(ids2, pos, allowed)
    return (before[5:] - after[5:]).abs().max().item()  # сдвиг выходов B


assert influence_leak(block_causal_mask) == 0.0
assert influence_leak(plain_causal_mask) > 1e-3

allowed = block_causal_mask(doc)
continuous = torch.arange(8)                            # позиции без сброса
out_reset = attention(ids, pos, allowed)
out_cont = attention(ids, continuous, allowed)
torch.testing.assert_close(out_reset[5:], out_cont[5:], atol=1e-5, rtol=0)
abs_gap = (attention(ids, pos, allowed, rotary=False)[5:]
           - attention(ids, continuous, allowed, rotary=False)[5:]).abs().max().item()
assert abs_gap > 1e-3
print(f"упаковка: {packed_eff:.3f} против {padded_eff:.3f}; "
      f"без сброса при абсолютных позициях: {abs_gap:.3f}")
