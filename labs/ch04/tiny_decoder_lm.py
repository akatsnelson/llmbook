"""Учебный декодерный блок 2017 года и маленькая языковая модель без готовых классов внимания.

Код к разделу 4.7 «Декодерный блок и модель» главы 4: причинное многоголовое внимание, полносвязная ветвь,
блок и модель над ними (в книге этот путь собран из листингов «Многоголовое причинное внимание как слой»
и «Декодерный блок из уже написанных частей»). Здесь весь код целиком, с проверками.
Запуск из корня репозитория: python labs/ch04/tiny_decoder_lm.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, heads):
        super().__init__()
        assert d_model % heads == 0
        self.heads = heads
        self.d_head = d_model // heads
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def _split(self, x):
        b, t, d = x.shape
        return x.reshape(b, t, self.heads, self.d_head).transpose(1, 2)

    def forward(self, x):
        # x: [B, T, d_model]
        q, k, v = self._split(self.q(x)), self._split(self.k(x)), self._split(self.v(x))
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_head)  # [B,H,T,T]
        t = x.size(1)
        future = torch.triu(torch.ones(t, t, dtype=torch.bool, device=x.device), 1)
        weights = scores.masked_fill(future, float("-inf")).softmax(dim=-1)
        mixed = weights @ v                                            # [B,H,T,d_h]
        joined = mixed.transpose(1, 2).contiguous().reshape(x.shape)    # [B,T,d]
        return self.out(joined)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.up = nn.Linear(d_model, d_ff)
        self.down = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.down(F.gelu(self.up(x)))

class DecoderBlock(nn.Module):
    def __init__(self, d_model, heads, d_ff):
        super().__init__()
        self.norm_1 = nn.LayerNorm(d_model)
        self.attention = CausalSelfAttention(d_model, heads)
        self.norm_2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)

    def forward(self, x):
        x = x + self.attention(self.norm_1(x))   # собрать
        return x + self.ffn(self.norm_2(x))      # подумать

class TinyLanguageModel(nn.Module):
    def __init__(self, vocab_size, d_model, heads, d_ff, layers):
        super().__init__()
        self.tokens = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(
            DecoderBlock(d_model, heads, d_ff) for _ in range(layers)
        )
        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size, bias=False)
        self.output.weight = self.tokens.weight  # общая таблица входа и выхода

    def forward(self, token_ids):
        h = self.tokens(token_ids)               # [B,T] -> [B,T,d]
        for block in self.blocks:
            h = block(h)                         # [B,T,d] -> [B,T,d]
        return self.output(self.norm(h))         # [B,T,d] -> [B,T,V]

torch.manual_seed(7)
model = TinyLanguageModel(vocab_size=32, d_model=16, heads=4, d_ff=48, layers=2).eval()
prefix_a = torch.tensor([[2, 5, 7, 11]])
prefix_b = torch.tensor([[2, 5, 19, 23]])
with torch.no_grad():
    logits_a = model(prefix_a)
    logits_b = model(prefix_b)

assert logits_a.shape == (1, 4, 32)
# Изменены позиции 2 и 3; логиты позиций 0 и 1 не должны измениться.
torch.testing.assert_close(logits_a[:, :2], logits_b[:, :2], atol=1e-6, rtol=0)

# Опыт из подраздела 4.7.1 «Порядок слов» на уровне логитов.
import torch

straight = torch.tensor([[2, 5, 7, 11]])
swapped = torch.tensor([[5, 2, 7, 11]])   # первые два токена переставлены

torch.manual_seed(7)
one_layer = TinyLanguageModel(vocab_size=32, d_model=16, heads=4, d_ff=48, layers=1).eval()
with torch.no_grad():
    a, b = one_layer(straight), one_layer(swapped)
# Позиции 2 и 3 видят одно и то же множество {2, 5} слева: логиты совпадают.
torch.testing.assert_close(a[:, 2:], b[:, 2:], atol=1e-6, rtol=0)
# Позиции 0 и 1 получили другие токены, поэтому различаются.
assert not torch.allclose(a[:, :2], b[:, :2])

torch.manual_seed(7)
two_layers = TinyLanguageModel(vocab_size=32, d_model=16, heads=4, d_ff=48, layers=2).eval()
with torch.no_grad():
    a, b = two_layers(straight), two_layers(swapped)
# Со вторым слоем мешок перестаёт быть мешком: маска сама выдаёт порядок.
assert not torch.allclose(a[:, 2:], b[:, 2:])

print("OK: tiny_decoder_lm.py")
