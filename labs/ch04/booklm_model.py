"""Полная модель BookLM: RMSNorm, GQA с RoPE, SwiGLU и проверка числа параметров 6 622 464.

Код к главе 4 книги: модель собрана из частей, которые разобраны в разделах 4.6 «Residual connection
и LayerNorm» (RMSNorm), 4.8 «Positional encoding» (RoPE), 4.9 «Виды внимания» (GQA) и 4.5 «Feed-forward
network» (SwiGLU); число параметров сверяется с листингом «Число параметров BookLM» из раздела 4.10.
Здесь весь код целиком, с проверками.
Запуск из корня репозитория: python labs/ch04/booklm_model.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def apply_rope(x, positions, base=10000.0):
    # x: [B, H, T, d_h]; поворачиваем пары (x_{2k}, x_{2k+1}) на угол p * theta_k
    d_h = x.size(-1)
    theta = base ** (-torch.arange(0, d_h, 2, dtype=torch.float32) / d_h)  # [d_h/2]
    angles = positions.float()[:, None] * theta[None, :]                     # [T, d_h/2]
    cos, sin = angles.cos(), angles.sin()
    x0, x1 = x[..., 0::2], x[..., 1::2]
    rotated = torch.stack((x0 * cos - x1 * sin, x0 * sin + x1 * cos), dim=-1)
    return rotated.flatten(-2)

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        scale = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x.float() * scale).to(x.dtype) * self.weight

class GroupedQueryAttention(nn.Module):
    def __init__(self, d, q_heads, kv_heads):
        super().__init__()
        assert d % q_heads == 0 and q_heads % kv_heads == 0
        self.q_heads, self.kv_heads, self.d_head = q_heads, kv_heads, d // q_heads
        self.q = nn.Linear(d, q_heads * self.d_head, bias=False)
        self.k = nn.Linear(d, kv_heads * self.d_head, bias=False)
        self.v = nn.Linear(d, kv_heads * self.d_head, bias=False)
        self.out = nn.Linear(q_heads * self.d_head, d, bias=False)

    def forward(self, x):
        b, t, _ = x.shape
        q = self.q(x).view(b, t, self.q_heads, self.d_head).transpose(1, 2)   # [B,Hq,T,d_h]
        k = self.k(x).view(b, t, self.kv_heads, self.d_head).transpose(1, 2)  # [B,Hkv,T,d_h]
        v = self.v(x).view(b, t, self.kv_heads, self.d_head).transpose(1, 2)
        positions = torch.arange(t, device=x.device)
        q, k = apply_rope(q, positions), apply_rope(k, positions)             # позиция входит здесь
        group = self.q_heads // self.kv_heads
        k, v = k.repeat_interleave(group, dim=1), v.repeat_interleave(group, dim=1)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_head)
        future = torch.triu(torch.ones(t, t, dtype=torch.bool, device=x.device), 1)
        weights = scores.masked_fill(future, float("-inf")).softmax(dim=-1)
        return self.out((weights @ v).transpose(1, 2).reshape(b, t, -1))

class SwiGLU(nn.Module):
    def __init__(self, d, d_ff):
        super().__init__()
        self.gate = nn.Linear(d, d_ff, bias=False)
        self.up = nn.Linear(d, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))

class BookLMBlock(nn.Module):
    def __init__(self, d, q_heads, kv_heads, d_ff):
        super().__init__()
        self.norm_1 = RMSNorm(d)
        self.attention = GroupedQueryAttention(d, q_heads, kv_heads)
        self.norm_2 = RMSNorm(d)
        self.ffn = SwiGLU(d, d_ff)

    def forward(self, x):
        x = x + self.attention(self.norm_1(x))
        return x + self.ffn(self.norm_2(x))

class BookLM(nn.Module):
    def __init__(self, vocab_size, d, layers, q_heads, kv_heads, d_ff):
        super().__init__()
        self.tokens = nn.Embedding(vocab_size, d)
        self.blocks = nn.ModuleList(
            BookLMBlock(d, q_heads, kv_heads, d_ff) for _ in range(layers)
        )
        self.norm = RMSNorm(d)
        self.output = nn.Linear(d, vocab_size, bias=False)
        self.output.weight = self.tokens.weight

    def forward(self, token_ids):
        h = self.tokens(token_ids)
        for block in self.blocks:
            h = block(h)
        return self.output(self.norm(h))

booklm = BookLM(vocab_size=8192, d=256, layers=6, q_heads=8, kv_heads=2, d_ff=768)
assert sum(p.numel() for p in booklm.parameters()) == 6_622_464   # паспорт BookLM

# RoPE закрывает диагноз: один слой теперь различает порядок.
torch.manual_seed(7)
tiny = BookLM(vocab_size=32, d=16, layers=1, q_heads=4, kv_heads=2, d_ff=48).eval()
with torch.no_grad():
    a = tiny(torch.tensor([[2, 5, 7, 11]]))
    b = tiny(torch.tensor([[5, 2, 7, 11]]))
    c = tiny(torch.tensor([[2, 5, 19, 23]]))
assert not torch.allclose(a[:, 2:], b[:, 2:])                        # порядок виден
torch.testing.assert_close(a[:, :2], c[:, :2], atol=1e-6, rtol=0)    # будущее по-прежнему закрыто

print("OK: booklm_model.py")
