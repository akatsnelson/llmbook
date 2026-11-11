"""BookLM — сквозная модель книги, рабочая реализация для настоящего обучения.

Математика и имена параметров совпадают с классом BookLM из главы 4: RMSNorm, GQA с RoPE на парах (x[0::2], x[1::2]),
SwiGLU, выходная матрица связана с таблицей эмбеддингов, смещений нет. Поэтому state_dict,
полученный здесь, загружается в учебный класс главы 4 без переименований.

Отличие от учебного кода только в скорости: внимание считает
torch.nn.functional.scaled_dot_product_attention (то же softmax(QK^T/sqrt(d_h)+M)V),
а повороты RoPE заранее посчитаны в буфере.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BookLMConfig:
    vocab: int = 8192
    d_model: int = 256
    layers: int = 6
    heads: int = 8
    kv_heads: int = 2
    d_ff: int = 768
    max_len: int = 512
    rope_base: float = 10000.0
    norm_eps: float = 1e-6

    def to_dict(self):
        return asdict(self)


def rope_tables(d_head: int, max_len: int, base: float):
    theta = base ** (-torch.arange(0, d_head, 2, dtype=torch.float32) / d_head)   # [d_h/2]
    angles = torch.arange(max_len, dtype=torch.float32)[:, None] * theta[None, :]  # [T, d_h/2]
    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):
    # x: [B, H, T, d_h]; пары (x_{2k}, x_{2k+1}) поворачиваются на угол p * theta_k — как в главе 4
    t = x.size(-2)
    cos, sin = cos[:t].to(x.dtype), sin[:t].to(x.dtype)
    x0, x1 = x[..., 0::2], x[..., 1::2]
    return torch.stack((x0 * cos - x1 * sin, x0 * sin + x1 * cos), dim=-1).flatten(-2)


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).to(x.dtype) * self.weight


class GQASelfAttention(nn.Module):
    def __init__(self, cfg: BookLMConfig):
        super().__init__()
        self.heads, self.kv_heads = cfg.heads, cfg.kv_heads
        self.d_head = cfg.d_model // cfg.heads
        self.q = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.k = nn.Linear(cfg.d_model, cfg.kv_heads * self.d_head, bias=False)
        self.v = nn.Linear(cfg.d_model, cfg.kv_heads * self.d_head, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x, cos, sin, cache=None, return_weights=False):
        b, t, d = x.shape
        q = self.q(x).view(b, t, self.heads, self.d_head).transpose(1, 2)
        k = self.k(x).view(b, t, self.kv_heads, self.d_head).transpose(1, 2)
        v = self.v(x).view(b, t, self.kv_heads, self.d_head).transpose(1, 2)
        start = 0 if cache is None else cache["len"]
        q = apply_rope(q, cos[start:], sin[start:])
        k = apply_rope(k, cos[start:], sin[start:])
        if cache is not None:                       # KV-кэш главы 14: дописываем и читаем всё
            if cache.get("k") is not None:
                k = torch.cat([cache["k"], k], dim=2)
                v = torch.cat([cache["v"], v], dim=2)
            cache["k"], cache["v"] = k, v
        group = self.heads // self.kv_heads
        kk = k.repeat_interleave(group, dim=1)
        vv = v.repeat_interleave(group, dim=1)
        if return_weights:                          # явная формула, чтобы показать карту внимания
            scores = q @ kk.transpose(-2, -1) / math.sqrt(self.d_head)
            total = kk.size(2)
            mask = torch.ones(t, total, dtype=torch.bool, device=x.device).triu(total - t + 1)
            w = scores.float().masked_fill(mask, float("-inf")).softmax(-1)
            y = (w.to(vv.dtype) @ vv)
        else:
            w = None
            total = kk.size(2)
            if t == 1:                              # один новый токен видит весь кэш
                y = F.scaled_dot_product_attention(q, kk, vv)
            elif total == t:                        # весь текст сразу: обычная причинная маска
                y = F.scaled_dot_product_attention(q, kk, vv, is_causal=True)
            else:                                   # кусок поверх кэша: строка i видит кэш и куски 0..i
                allowed = torch.ones(t, total, dtype=torch.bool, device=x.device).tril(total - t)
                y = F.scaled_dot_product_attention(q, kk, vv, attn_mask=allowed)
        return self.out(y.transpose(1, 2).reshape(b, t, d)), w


class SwiGLU(nn.Module):
    def __init__(self, cfg: BookLMConfig):
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class BookLMBlock(nn.Module):
    def __init__(self, cfg: BookLMConfig):
        super().__init__()
        self.norm_1 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attention = GQASelfAttention(cfg)
        self.norm_2 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.ffn = SwiGLU(cfg)

    def forward(self, x, cos, sin, cache=None, return_weights=False):
        a, w = self.attention(self.norm_1(x), cos, sin, cache, return_weights)
        x = x + a
        return x + self.ffn(self.norm_2(x)), w


class BookLM(nn.Module):
    def __init__(self, cfg: BookLMConfig = BookLMConfig()):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab, cfg.d_model)
        self.blocks = nn.Sequential(*[BookLMBlock(cfg) for _ in range(cfg.layers)])
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        cos, sin = rope_tables(cfg.d_model // cfg.heads, cfg.max_len, cfg.rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, ids, caches=None, return_weights=False):
        h = self.embed(ids)
        weights = []
        for i, block in enumerate(self.blocks):
            h, w = block(h, self.rope_cos, self.rope_sin,
                         None if caches is None else caches[i], return_weights)
            weights.append(w)
        logits = self.norm(h) @ self.embed.weight.T
        return (logits, weights) if return_weights else logits

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def generate(self, ids, max_new=200, temperature=0.8, top_k=40, eos_id=None, seed=None):
        """Пошаговая генерация с KV-кэшем. ids: [1, T]."""
        gen = torch.Generator(device=ids.device)
        if seed is not None:
            gen.manual_seed(seed)
        caches = [{"len": 0} for _ in self.blocks]
        logits = self(ids, caches)
        for c in caches:
            c["len"] = ids.size(1)
        out = ids
        for _ in range(max_new):
            last = logits[:, -1, :].float()
            if temperature == 0:
                nxt = last.argmax(-1, keepdim=True)
            else:
                last = last / temperature
                if top_k:
                    kth = last.topk(top_k).values[:, -1:]
                    last = last.masked_fill(last < kth, float("-inf"))
                nxt = torch.multinomial(last.softmax(-1), 1, generator=gen)
            out = torch.cat([out, nxt], dim=1)
            if eos_id is not None and nxt.item() == eos_id:
                break
            if out.size(1) >= self.cfg.max_len:
                break
            logits = self(nxt, caches)
            for c in caches:
                c["len"] += 1
        return out


if __name__ == "__main__":
    torch.manual_seed(0)
    m = BookLM()
    print("параметров:", m.num_params())
    assert m.num_params() == 6_622_464
    ids = torch.randint(0, 8192, (1, 12))
    full = m(ids)
    # Пошаговый проход с KV-кэшем совпадает с полным — тот же инвариант, что в главе 14.
    caches = [{"len": 0} for _ in m.blocks]
    steps = []
    for t in range(ids.size(1)):
        steps.append(m(ids[:, t:t + 1], caches)[:, -1])
        for c in caches:
            c["len"] += 1
    torch.testing.assert_close(torch.stack(steps, 1), full, atol=2e-5, rtol=1e-4)
    # Чтение кусками поверх кэша (префиксный кэш глав 10, 11, 14) совпадает с чтением целиком.
    caches = [{"len": 0} for _ in m.blocks]
    parts = []
    for a, b in ((0, 5), (5, 9), (9, 12)):
        parts.append(m(ids[:, a:b], caches))
        for c in caches:
            c["len"] += b - a
    torch.testing.assert_close(torch.cat(parts, 1), full, atol=2e-5, rtol=1e-4)
    # Явная формула внимания совпадает с быстрым ядром.
    slow, _ = m(ids, return_weights=True)
    torch.testing.assert_close(slow, full, atol=2e-5, rtol=1e-4)
    print("OK: кэш и явная формула совпадают с быстрым проходом")
