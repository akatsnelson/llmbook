"""Глава 4 на обученной BookLM: веса BookLM в классе, собранном в главе 4, и карта внимания.

Опыты (время — на процессоре Apple M2 Max):
  ch04_trained_weights — разделы «Декодерный блок и модель» и «Причинная маска» (тест влияния);
      листинга с этим выводом в книге нет. Загружает обученные веса в класс главы, печатает число
      параметров, расхождение логитов с booklm/model.py и тест влияния: замена слова «спать» не
      сдвигает логиты на позициях до него. Около 1 с.
  ch04_greedy_text — проверка того же класса: жадная сказка после «Однажды», 36 токенов. В книге её
      нет: жадную генерацию книга показывает в главе 7, раздел «Выбор следующего токена».
  ch04_attention_map — глава 7, раздел «Длинный прогон и бюджет обучения», рубрика «А теперь на нашей
      модели: прогон обучения» (рисунок с картой внимания до и после обучения). Ищет голову, которая
      связывает «она» с «Мама», и печатает её строку внимания до обучения и после. Около 2 с.
      Данные рисунка: ch04-attention-map.csv.
Каждый опыт печатает вывод и сохраняет его копию в models/booklm/experiments/<имя опыта>.txt
(ch04-trained-weights.txt, ch04-greedy-text.txt, ch04-attention-map.txt).

Класс BookLM ниже — копия кода главы 4: та же цепочка attention_scores -> causal_attention ->
CausalSelfAttention -> DecoderBlock -> GQASelfAttention, BookLMBlock -> BookLM. В книге опыты
работают с классом из предыдущих листингов; здесь он повторён, чтобы файл запускался отдельно.

Запуск из корня репозитория:
    python -m booklm.experiments.ch04                      # все опыты
    python -m booklm.experiments.ch04 ch04_attention_map   # один опыт
    python -m booklm.experiments.ch04 --help               # список опытов
"""
import csv
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

if not __package__:  # запуск файлом (python booklm/experiments/ch04.py): корень репозитория — в путь поиска
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from booklm.experiments import run
from booklm.lab import MODELS, attention, load, load_model, strongest_head

OUT = MODELS / "experiments"
TEXT = "Мама мыла раму, а потом она пошла спать."


# ----------------------------------------------------------------------------- код главы 4

def attention_scores(q, k):
    return q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))


def causal_attention(q, k, v):
    scores = attention_scores(q, k)
    t = q.size(-2)
    future = torch.triu(torch.ones(t, t, dtype=torch.bool, device=q.device), diagonal=1)
    weights = scores.masked_fill(future, float("-inf")).softmax(dim=-1)
    return weights @ v, weights


def split_heads(x, heads):
    b, t, d = x.shape
    return x.reshape(b, t, heads, d // heads).transpose(1, 2)


def join_heads(x):
    b, heads, t, d_h = x.shape
    return x.transpose(1, 2).contiguous().reshape(b, t, heads * d_h)


def rotate_pair(x0, x1, angle):
    c, s = torch.cos(angle), torch.sin(angle)
    return c * x0 - s * x1, s * x0 + c * x1


def apply_rope(x, base=10000.0):
    t, d_h = x.shape[-2:]
    theta = base ** (-torch.arange(0, d_h, 2) / d_h)
    angles = torch.arange(t)[:, None] * theta
    x0, x1 = rotate_pair(x[..., 0::2], x[..., 1::2], angles)
    return torch.stack((x0, x1), dim=-1).flatten(-2)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, heads):
        super().__init__()
        self.heads = heads
        self.q, self.k, self.v, self.out = (
            nn.Linear(d_model, d_model, bias=False) for _ in range(4))

    def forward(self, x):
        q, k, v = (split_heads(w(x), self.heads) for w in (self.q, self.k, self.v))
        mixed, _ = causal_attention(q, k, v)
        return self.out(join_heads(mixed))


class DecoderBlock(nn.Module):
    def __init__(self, d_model, heads, d_ff):
        super().__init__()
        self.norm_1 = nn.LayerNorm(d_model)
        self.attention = CausalSelfAttention(d_model, heads)
        self.norm_2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))

    def forward(self, x):
        x = x + self.attention(self.norm_1(x))
        return x + self.ffn(self.norm_2(x))


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        rms = (x.pow(2).mean(dim=-1, keepdim=True) + self.eps).sqrt()
        return self.weight * x / rms


class SwiGLU(nn.Module):
    def __init__(self, d, d_ff):
        super().__init__()
        self.gate = nn.Linear(d, d_ff, bias=False)
        self.up = nn.Linear(d, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class GQASelfAttention(CausalSelfAttention):
    def __init__(self, d_model, heads, kv_heads):
        super().__init__(d_model, heads)
        self.kv_heads = kv_heads
        self.use_rope = True
        kv_width = kv_heads * (d_model // heads)
        self.k = nn.Linear(d_model, kv_width, bias=False)
        self.v = nn.Linear(d_model, kv_width, bias=False)

    def forward(self, x):
        q = split_heads(self.q(x), self.heads)
        k = split_heads(self.k(x), self.kv_heads)
        v = split_heads(self.v(x), self.kv_heads)
        if self.use_rope:
            q, k = apply_rope(q), apply_rope(k)
        group = self.heads // self.kv_heads
        k = k.repeat_interleave(group, dim=1)
        v = v.repeat_interleave(group, dim=1)
        mixed, _ = causal_attention(q, k, v)
        return self.out(join_heads(mixed))


class BookLMBlock(DecoderBlock):
    def __init__(self, d_model, heads, kv_heads, d_ff):
        super().__init__(d_model, heads, d_ff)
        self.norm_1 = RMSNorm(d_model)
        self.norm_2 = RMSNorm(d_model)
        self.attention = GQASelfAttention(d_model, heads, kv_heads)
        self.ffn = SwiGLU(d_model, d_ff)


class BookLM(nn.Module):
    def __init__(self, vocab, d_model, layers, heads, kv_heads, d_ff):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        self.blocks = nn.Sequential(*[BookLMBlock(d_model, heads, kv_heads, d_ff)
                                      for _ in range(layers)])
        self.norm = RMSNorm(d_model)

    def forward(self, ids):
        h = self.norm(self.blocks(self.embed(ids)))
        return h @ self.embed.weight.T


def chapter_model():
    """Модель из листинга «Модель BookLM целиком»: тот же seed, те же размеры."""
    torch.manual_seed(0)
    return BookLM(vocab=8192, d_model=256, layers=6, heads=8, kv_heads=2, d_ff=768)


def generate_greedy(model, ids, steps):
    for _ in range(steps):
        with torch.no_grad():
            logits = model(ids[:, -512:])        # не длиннее окна
        best = logits[0, -1].argmax().view(1, 1)     # жадный выбор
        ids = torch.cat([ids, best], dim=1)
    return ids


# ----------------------------------------------------------------------------- опыты

def ch04_trained_weights():
    """Обученные веса в классе главы 4: число параметров, сверка с booklm/model.py, тест влияния."""
    model = chapter_model()
    tok, trained = load("base")                  # BookLM из booklm/model.py
    model.load_state_dict(trained.state_dict())  # в наш класс, имена те же
    print("параметров:", sum(p.numel() for p in model.parameters()))

    text = TEXT
    walk = text.replace("спать", "гулять")
    ids = torch.tensor([[1] + tok.encode(text).ids])     # 1 — номер <bos>
    other = torch.tensor([[1] + tok.encode(walk).ids])
    with torch.no_grad():
        ours, ref, after = model(ids), trained(ids), model(other)
    diff = (ours - ref).abs().max().item()
    print("расхождение с booklm/model.py:", f"{diff:.0e}")
    shift = (ours - after).abs().amax(dim=-1)[0]     # сдвиг каждой позиции
    print("сдвиг логитов до «спать»:", shift[:-2].max().item())
    print("сдвиг на «спать» и «.»:", [round(s, 1) for s in shift[-2:].tolist()])


def ch04_greedy_text():
    """Жадная сказка после «Однажды» на классе главы 4 с обученными весами."""
    import textwrap
    model = chapter_model()
    tok, trained = load("base")
    model.load_state_dict(trained.state_dict())
    start = torch.tensor([[1] + tok.encode("Однажды").ids])
    story = tok.decode(generate_greedy(model, start, 36)[0, 1:].tolist())
    print(textwrap.fill(story, 70))


def ch04_attention_map():
    """Голова внимания, которая связывает «она» с «Мама», до обучения и после; данные рисунка в csv."""
    tok, trained = load("base")
    text = TEXT
    pieces, att = attention(tok, trained, text)    # [слои, головы, T, T]
    print("|".join(pieces))
    print("форма карт:", list(att.shape))
    layer, head, share, she, mom = strongest_head(pieces, att, "она", "Мам")
    print(f"слой {layer}, голова {head}: «она» -> «Мама» {share:.2f}")
    _, att0 = attention(tok, load_model("init"), text)   # BookLM на шаге 0
    for name, a in (("после обучения:", att), ("до обучения:   ", att0)):
        row = a[layer, head, she, :she + 1]        # строка «она» до неё самой
        print(name, " ".join(f"{x:.2f}" for x in row))
    for other in ("Кот увидел девочку, и она засмеялась.",
                  "Девочка увидела кота, и он убежал."):
        p, a = attention(tok, trained, other)
        word = " она" if " она" in p else " он"
        row = a[layer, head, p.index(word)]
        best = p[int(row.argmax())].strip()
        print(f"{other} {word.strip()} -> {best} {row.max():.2f}")

    with open(OUT / "ch04-attention-map.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["key", "query", "trained", "init"])
        for qi in range(len(pieces)):
            for ki in range(len(pieces)):
                w.writerow([ki, qi, round(att[layer, head, qi, ki].item(), 4),
                            round(att0[layer, head, qi, ki].item(), 4)])


EXPERIMENTS = {f.__name__: f for f in (ch04_trained_weights, ch04_greedy_text, ch04_attention_map)}


if __name__ == "__main__":
    run(EXPERIMENTS, save=True)      # копия вывода в models/booklm/experiments/<имя опыта>.txt
