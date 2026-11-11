"""Стоки внимания на сказках: BookLM, Qwen3-0.6B и Qwen3.5-0.8B-Base.

Опыт к подразделу 5.5.2 «Стоки внимания и выходные ворота» главы 5. Меряем две вещи.

1. Какую долю внимания головы отдают первому токену окна (позиция 0). Запросы берём с позиции 64,
   чтобы каждому было из чего выбирать; доля усредняется по запросам и окнам, для каждого слоя
   печатаем среднее по головам и самую «стоковую» голову.
2. Что будет с перплексией, если оставить внимание только на последних W ключах (скользящее окно) и
   выбросить из окна первый токен, как делает наивный потоковый кэш, или сохранить первые 1 или 4
   токена, как делает StreamingLLM (Xiao и др., 2023).

BookLM считается на восьми окнах по 512 токенов отложенных историй, начатых с произвольного места,
как при обучении. Qwen3-0.6B (обычное внимание с QK-norm) и Qwen3.5-0.8B-Base (Gated DeltaNet и
внимание с выходными воротами) читают те же сказки, склеенные в одну строку и нарезанные на окна
по 512 своих токенов. Веса Qwen берутся из кэша Hugging Face; без него библиотека скачает около
1,5 и 1,8 ГБ. Всё считается на процессоре, без случайности.

Запуск из корня репозитория: python labs/ch05/attention_sinks.py
Время: около двух минут на Apple M2 Max (BookLM и обе модели Qwen; при первом запуске
дольше, пока скачиваются веса). Только BookLM, за секунды:
    python labs/ch05/attention_sinks.py --booklm
"""
from __future__ import annotations

import math
import sys
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from booklm.experiments.ch05 import masked_forward, stories, window_mask  # noqa: E402
from booklm.lab import load, load_tokenizer, perplexity, val_tokens  # noqa: E402

T, Q0 = 512, 64          # длина окна и первая позиция запросов, по которым считаем долю
torch.set_num_threads(3)


def sink_window_mask(T, W, sinks):
    """Скользящее окно W (формула окна главы 5) плюс первые sinks позиций, видимые всем запросам."""
    mask = window_mask(T, W)
    mask[:, :sinks] = torch.ones(T, T, dtype=torch.bool).tril()[:, :sinks]
    return mask


# ----------------------------------------------------------------------------- BookLM

@torch.no_grad()
def booklm_first_share(model, windows=8):
    """Доля внимания позиции 0: [слои, головы], среднее по запросам Q0..T-1 и окнам."""
    data, share = val_tokens(), torch.zeros(6, 8)
    for s in range(0, windows * T, T):
        _, maps = model(data[s:s + T][None], return_weights=True)
        for layer, w in enumerate(maps):                 # w: [1, головы, T, T]
            share[layer] += w[0, :, Q0:, 0].mean(-1)
    return share / windows


def booklm_window_ppl(model, W, sinks):
    for block in model.blocks:
        block.attention.forward = partial(masked_forward, block.attention,
                                          mask=sink_window_mask(T, W, sinks))
    return perplexity(model)


def run_booklm():
    uniform = sum(1 / (t + 1) for t in range(Q0, T)) / (T - Q0)   # доля при равном внимании
    print(f"BookLM: доля внимания первому токену (равное внимание дало бы {uniform:.4f})")
    res = {}
    for kind in ("step:100", "step:762", "step:1525", "base"):
        tok, model = load(kind)
        share = booklm_first_share(model)
        res[kind] = share
        print(f"  {kind:10} среднее {share.mean():.4f}, самая сильная голова {share.max():.4f}")
    share = res["base"]
    print("  обученная, по слоям: среднее", [round(x, 4) for x in share.mean(1).tolist()])
    print("                  наибольшее", [round(x, 4) for x in share.max(1).values.tolist()])
    ppl = {}
    for W in (16, 64):
        for sinks in (0, 1, 4):
            tok, model = load("base")
            ppl[W, sinks] = booklm_window_ppl(model, W, sinks)
            print(f"  окно {W:2}, первых токенов {sinks}: перплексия {ppl[W, sinks]:.2f}")
    # числа текста главы 5
    assert share.max() < 0.006 and share.mean() < uniform
    assert round(ppl[16, 0], 2) == 10.57 and round(ppl[64, 0], 2) == 7.33
    assert abs(ppl[16, 4] - ppl[16, 0]) < 0.1 and abs(ppl[64, 4] - ppl[64, 0]) < 0.05
    return res, ppl


# ----------------------------------------------------------------------------- Qwen

SETTINGS = {"mask": None}       # None: обычная причинная маска; иначе [T, T], True = ключ виден
SHARES = []                     # доля позиции 0 в каждом слое внимания: [головы]


def eager_with_mask(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    """Явное внимание для Qwen3 и Qwen3.5 (пакет из одного окна): своя маска и замер доли позиции 0."""
    group = query.size(1) // key.size(1)
    key, value = key.repeat_interleave(group, 1), value.repeat_interleave(group, 1)
    scores = (query @ key.transpose(2, 3)).float() * scaling       # [1, H, T, T]
    t = scores.size(-1)
    mask = SETTINGS["mask"]
    allowed = torch.ones(t, t, dtype=torch.bool).tril() if mask is None else mask[:t, :t]
    w = scores.masked_fill(~allowed, float("-inf")).softmax(-1)
    SHARES.append(w[0, :, Q0:, 0].mean(-1))
    out = (w.to(value.dtype) @ value).transpose(1, 2).contiguous()
    return out, w


def qwen_windows(tok, n=4):
    text = "\n\n".join(load_tokenizer().decode(s.tolist()).strip() for s in stories(60))
    ids = tok(text, return_tensors="pt").input_ids[0]
    step = (len(ids) - T) // n
    return [ids[i * step:i * step + T] for i in range(n)]


@torch.no_grad()
def run_qwen(name):
    import transformers.models.qwen3.modeling_qwen3 as m3
    import transformers.models.qwen3_5.modeling_qwen3_5 as m35
    from transformers import AutoModelForCausalLM, AutoTokenizer
    m3.eager_attention_forward = m35.eager_attention_forward = eager_with_mask
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32,
                                                 attn_implementation="eager").eval()
    windows = qwen_windows(tok)
    out = {}
    for label, mask in (("полное", None), ("окно 64, 0", sink_window_mask(T, 64, 0)),
                        ("окно 64, 4", sink_window_mask(T, 64, 4))):
        SETTINGS["mask"] = mask
        losses, shares = [], []
        for ids in windows:
            SHARES.clear()
            logits = model(ids[None]).logits[0]
            losses.append(F.cross_entropy(logits[Q0 - 1:-1].float(), ids[Q0:]))  # позиции >= 64
            shares.append(torch.stack(SHARES))                                    # [слои, головы]
        share = torch.stack(shares).mean(0)
        out[label] = (math.exp(torch.stack(losses).mean()), share)
        print(f"  {label:11} перплексия {out[label][0]:9.2f}")
    share = out["полное"][1]
    print(f"  {name}: слоёв внимания {share.size(0)}, голов {share.size(1)}")
    print("  доля первому токену, среднее по слою:", [round(x, 2) for x in share.mean(1).tolist()])
    print("                      наибольшая голова:", [round(x, 2) for x in share.max(1).values.tolist()])
    print(f"  в среднем по всем головам {share.mean():.2f}")
    return out


def main():
    run_booklm()
    if "--booklm" in sys.argv:
        return
    print("\nQwen3-0.6B")
    q3 = run_qwen("Qwen/Qwen3-0.6B")
    print("\nQwen3.5-0.8B-Base")
    q35 = run_qwen("Qwen/Qwen3.5-0.8B-Base")
    s3, s35 = q3["полное"][1], q35["полное"][1]
    assert s3.mean() > 0.4 and s3.max() > 0.95          # у Qwen3 есть сток
    assert s35.mean() < 0.1                              # у Qwen3.5 его почти нет
    assert q3["окно 64, 0"][0] > 3 * q3["окно 64, 4"][0]  # без первых токенов окно ломается
    print("OK")


if __name__ == "__main__":
    main()
