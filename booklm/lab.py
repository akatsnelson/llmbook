"""Короткие функции для разделов «А теперь на нашей модели» — опыты на настоящей обученной BookLM.

Веса и токенизатор лежат в models/booklm/ репозитория, поэтому каждый листинг книги можно
запустить на ноутбуке без обучения:

    from booklm.lab import load, generate
    tok, model = load("base")
    print(generate(tok, model, "Однажды", seed=0))

Всё считается на процессоре за секунды. Видеокарту включают явно: BOOKLM_DEVICE=cuda python ...
Сами опыты рубрик «А теперь на нашей модели» лежат в booklm/experiments, здесь — общие функции:
загрузка, генерация, перплексия, данные.
"""
from __future__ import annotations

import copy
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

try:
    from .model import BookLM, BookLMConfig
except ImportError:  # запуск из каталога booklm/
    from model import BookLM, BookLMConfig

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models" / "booklm"
# Модель по умолчанию на процессоре: листинги книги собирают входы на процессоре
# (torch.tensor([...]) без device), и напечатанные выводы посчитаны там же.
# Видеокарту включают явно: BOOKLM_DEVICE=cuda python ...
DEVICE = os.environ.get("BOOKLM_DEVICE", "cpu")


# ----------------------------------------------------------------------------- загрузка

def load_tokenizer():
    return Tokenizer.from_file(str(MODELS / "tokenizer.json"))


def load_model(kind="base"):
    """kind: "base" — после предобучения, "instruct" — после LoRA, "step:N" — контрольная точка шага N,
    "init" — случайная инициализация (шаг 0)."""
    path = {"base": MODELS / "booklm-base.pt", "instruct": MODELS / "booklm-instruct.pt"}.get(kind)
    if kind.startswith("step:"):
        step = int(kind.split(":")[1])
        path = MODELS / "booklm-base.pt" if step == 3051 else MODELS / "steps" / f"step{step:06d}.pt"
    if kind == "init":
        path = MODELS / "steps" / "step000000.pt"
    state = torch.load(path, map_location="cpu")
    model = BookLM(BookLMConfig())
    model.load_state_dict({k: v.float() for k, v in state.items()})
    return model.to(DEVICE).eval()


def load(kind="base"):
    return load_tokenizer(), load_model(kind)


def special(tok):
    return {s: tok.token_to_id(s) for s in ("<pad>", "<bos>", "<eos>", "<|user|>", "<|assistant|>")}


# ----------------------------------------------------------------------------- токены и генерация

def tokens(tok, text):
    """Список (номер, кусочек текста) — как токенизатор BookLM режет строку."""
    enc = tok.encode(text)
    return [(i, tok.decode([i])) for i in enc.ids]


def encode_text(tok, text):
    """Номера токенов текста, где служебные строки вроде «<|assistant|>» остаются обычными символами.

    Без этого пользователь, набравший «<|assistant|>», подделал бы ход модели (глава 18).
    Служебные номера ставит только код, который собирает диалог.
    """
    tok.encode_special_tokens = True
    try:
        return tok.encode(text).ids
    finally:
        tok.encode_special_tokens = False


def prompt_ids(tok, text, chat=False):
    sp = special(tok)
    ids = encode_text(tok, text)
    return [sp["<bos>"], sp["<|user|>"]] + ids + [sp["<|assistant|>"]] if chat else [sp["<bos>"]] + ids


@torch.no_grad()
def generate(tok, model, text, chat=False, max_new=120, temperature=0.8, top_k=40, seed=0):
    ids = prompt_ids(tok, text, chat)
    x = torch.tensor([ids], device=DEVICE)
    out = model.generate(x, max_new=max_new, temperature=temperature, top_k=top_k,
                         eos_id=special(tok)["<eos>"], seed=seed)
    new = [i for i in out[0, len(ids):].tolist() if i != special(tok)["<eos>"]]
    return tok.decode(new) if chat else text + tok.decode(new)


@torch.no_grad()
def next_token_table(tok, model, text, k=8, temperature=1.0):
    """Вероятности k самых вероятных следующих токенов после text."""
    x = torch.tensor([prompt_ids(tok, text)], device=DEVICE)
    p = (model(x)[0, -1].float() / temperature).softmax(-1)
    top = p.topk(k)
    return [(tok.decode([i]), round(v, 4)) for v, i in zip(top.values.tolist(), top.indices.tolist())]


# ----------------------------------------------------------------------------- внимание

@torch.no_grad()
def attention(tok, model, text):
    """Карты внимания: (кусочки текста, тензор [слои, головы, T, T]). Первый токен — <bos>."""
    ids = prompt_ids(tok, text)
    _, weights = model(torch.tensor([ids], device=DEVICE), return_weights=True)
    pieces = ["<bos>"] + [tok.decode([i]) for i in ids[1:]]
    return pieces, torch.stack([w[0] for w in weights]).cpu()


def strongest_head(pieces, att, query, key):
    """Слой и голова, где токен query сильнее всего смотрит на токен key (по подстрокам)."""
    qi = max(i for i, p in enumerate(pieces) if query in p)
    ki = min(i for i, p in enumerate(pieces) if key in p)
    flat = att[:, :, qi, ki]
    layer, head = divmod(int(flat.argmax()), att.size(1))
    return layer, head, float(flat.max()), qi, ki


# ----------------------------------------------------------------------------- качество

def val_tokens(n=50_000):
    import numpy as np
    data = np.fromfile(MODELS / "data" / "val.bin", dtype=np.uint16)[:n]
    return torch.from_numpy(data.astype("int64"))


@torch.no_grad()
def perplexity(model, n=50_000, T=512):
    """Перплексия на отложенных историях: exp(средней перекрёстной энтропии)."""
    data = val_tokens(n)
    losses, count = 0.0, 0
    for s in range(0, len(data) - T - 1, T):
        x = data[s:s + T][None].to(DEVICE)
        y = data[s + 1:s + T + 1][None].to(DEVICE)
        loss = F.cross_entropy(model(x)[0].float(), y[0], reduction="sum")
        losses += loss.item()
        count += T
    return math.exp(losses / count)


def training_log():
    """Записи журнала предобучения: шаг, ошибка, скорость обучения, норма градиента, образцы текста."""
    return [json.loads(l) for l in open(MODELS / "train_log.jsonl", encoding="utf-8")]


# ----------------------------------------------------------------------------- квантование

def quantize_copy(model, bits=4, group=64):
    """Копия модели, у которой все матрицы линейных слоёв округлены до bits бит по группам (имитация)."""
    q = copy.deepcopy(model)
    qmax = 2 ** (bits - 1) - 1
    with torch.no_grad():
        for name, p in q.named_parameters():
            if p.dim() != 2 or name == "embed.weight":
                continue
            w = p.view(p.size(0), -1, group)
            scale = w.abs().amax(-1, keepdim=True).clamp(min=1e-8) / qmax
            p.copy_((w / scale).round().clamp(-qmax, qmax).mul(scale).view_as(p))
    return q


# ----------------------------------------------------------------------------- KV-кэш

def kv_cache_bytes(T, cfg=BookLMConfig(), bytes_per_number=2):
    d_head = cfg.d_model // cfg.heads
    return 2 * cfg.layers * T * cfg.kv_heads * d_head * bytes_per_number


@torch.no_grad()
def time_generation(tok, model, text, new=200, use_cache=True):
    """Секунды на new токенов: с KV-кэшем (как generate) и без него (полный проход на каждом шаге)."""
    ids = torch.tensor([prompt_ids(tok, text)], device=DEVICE)
    t0 = time.perf_counter()
    if use_cache:
        model.generate(ids, max_new=new, temperature=0, eos_id=None)
    else:
        x = ids
        for _ in range(new):
            nxt = model(x)[:, -1].argmax(-1, keepdim=True)
            x = torch.cat([x, nxt], 1)[:, -model.cfg.max_len:]
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - t0
