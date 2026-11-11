"""Глава 8 на обученной BookLM: маска ошибки, LoRA, NF4, логарифм отношения, неявная награда DPO,
забывание и измерения после SFT. Опыты с GRPO — в главе 9 (ch09.py), дистилляция — в ch08_distill.py.

Генерация текста везде одна: temperature=1.0 без отсечения top_k, то есть ответ берётся из самого
распределения модели. Сравниваемые модели получают одинаковые зёрна и одинаковую длину.

Опыты (время — на процессоре Apple M2 Max):
  ch08_loss_mask — раздел «Дообучение с учителем», рубрика «А теперь на нашей модели: маска ошибки».
      Таблица позиций пары «просьба -> история»: какие цели попадают в ошибку; ошибка базы на истории
      со своей и с чужой просьбой. Около 5 с.
  ch08_lora_delta — раздел «Низкоранговая адаптация (LoRA)», рубрика «А теперь на нашей модели:
      низкоранговая поправка». Насколько адаптер сдвинул каждую матрицу каждого блока и сингулярные
      числа поправки и основы. Около 1 с. Таблица: ch08-lora-delta.csv.
  ch08_lora_before_after — тот же раздел, рубрика «А теперь на нашей модели: дообучение LoRA». Ответы
      базы и модели после LoRA на три просьбы и доля ответов с <eos> по 50 просьбам. Около 35 с.
  ch08_lora_modes — та же рубрика. Ошибка в четырёх режимах (база, адаптер, адаптер выключен, слитая
      модель) и сверка их логитов. Около 25 с.
  ch08_qlora_booklm — раздел «QLoRA», рубрика «А теперь на нашей модели: квантование NF4». Ошибка
      весов после NF4, KL до исходной модели, перплексия и где расходятся продолжения. Около 10 с.
  ch08_kl_leash — раздел «Предпочтения и RLHF», рубрика «А теперь на нашей модели: логарифм отношения
      по токенам». Оценка KL инструктивной модели к базе на одной просьбе и самые дорогие токены ответа.
      Около 15 с. Таблица по токенам: ch08-kl-leash.csv.
  ch08_implicit_reward — раздел «Прямая оптимизация предпочтений (DPO)», рубрика «А теперь на нашей
      модели: неявная награда». Неявная награда и ошибка DPO, если лучший ответ — своя история пары,
      а худший — история соседней пары. Около 25 с.
  ch08_forgetting — раздел «Поломки дообучения и их измерение», рубрика «А теперь на нашей модели:
      катастрофическое забывание». Ошибка на отложенных историях до LoRA и после и продолжения без
      формата диалога. Около 20 с.
  ch08_eval_protocol — тот же раздел, рубрика «А теперь на нашей модели: измерения после SFT». Таблица
      метрик базы, адаптера и слитой модели. Дольше 3 минут.
txt опыты не пишут: вывод только на экран, у двух опытов ещё csv.

Запуск из корня репозитория:
    python -m booklm.experiments.ch08                     # все опыты
    python -m booklm.experiments.ch08 ch08_lora_delta     # один опыт
    python -m booklm.experiments.ch08 --help              # список опытов
Файлы выводов лежат в models/booklm/experiments/.
"""
import copy
import csv
import json
import math
import random
import re
import sys
import textwrap
from pathlib import Path

import torch
import torch.nn.functional as F

if not __package__:  # запуск файлом (python booklm/experiments/ch08.py): корень репозитория — в путь поиска
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from booklm.experiments import run
from booklm.lab import MODELS, ROOT, load, load_tokenizer, perplexity, prompt_ids, special, val_tokens

sys.path.insert(0, str(ROOT / "booklm"))       # sft_lora.py импортирует model без пакета
sys.path.insert(0, str(ROOT / "labs" / "ch08"))
from sft_lora import add_lora, encode, set_lora                          # noqa: E402
from ft_kit import dequantize_nf4, double_dequant, double_quant, quantize_nf4  # noqa: E402

OUT = MODELS / "experiments"
TYPES = ("attention.q", "attention.k", "attention.v", "attention.out", "ffn.gate", "ffn.up", "ffn.down")
NAMES = ("W_Q", "W_K", "W_V", "W_O", "W_gate", "W_up", "W_down")


# ----------------------------------------------------------------------------- помощники

def held_out():
    """200 отложенных пар «просьба → история» — ровно те, что sft_lora.py не видел."""
    rows = [json.loads(l) for l in open(MODELS / "data" / "sft.jsonl", encoding="utf-8")]
    random.Random(0).shuffle(rows)
    return rows[:200]


def sft_train():
    """Остальные пары — те, на которых sft_lora.py обучал адаптер."""
    rows = [json.loads(l) for l in open(MODELS / "data" / "sft.jsonl", encoding="utf-8")]
    random.Random(0).shuffle(rows)
    return rows[200:]


def adapter_model():
    """Адаптерный режим: база + 42 обёртки LoRALinear, в которые загружены обученные A и B."""
    _, model = load("base")
    file = torch.load(MODELS / "booklm-instruct-lora.pt", map_location="cpu")
    add_lora(model, file["r"], file["alpha"])
    model.load_state_dict(file["adapter"], strict=False)
    return model.eval()


@torch.no_grad()
def answer_logp(model, ids, labels):
    """log p(цели) на каждой позиции ответа (маска из encode) и логиты этих позиций."""
    x = torch.tensor([ids[:-1]])
    y = torch.tensor(labels[1:])
    mask = y != -100
    logits = model(x)[0][mask]
    logp = logits.log_softmax(-1).gather(-1, y[mask][:, None])[:, 0]
    return logp, logits


def story_loss(model, tok, pairs):
    """Средняя ошибка на целях истории: сумма по всем парам / число целей (как в sft_lora.py)."""
    total, count = 0.0, 0
    for req, story in pairs:
        logp, _ = answer_logp(model, *encode(tok, req, story, 512))
        total -= logp.sum().item()
        count += len(logp)
    return total / count


@torch.no_grad()
def sample(tok, model, ids, max_new, seed):
    """Генерация с temperature=1.0 без top_k. Возвращает новые токены и признак <eos>."""
    eos = special(tok)["<eos>"]
    out = model.generate(torch.tensor([ids]), max_new=max_new, temperature=1.0, top_k=None,
                         eos_id=eos, seed=seed)[0, len(ids):].tolist()
    ended = bool(out) and out[-1] == eos
    return (out[:-1] if ended else out), ended


def eos_stats(tok, model, requests, max_new):
    """Доля ответов, закончившихся <eos> не позже max_new токенов, и средняя длина ответа."""
    ended, length = 0, 0
    for req in requests:
        new, e = sample(tok, model, prompt_ids(tok, req, chat=True), max_new, seed=0)
        ended += e
        length += len(new) + e
    return ended / len(requests), length / len(requests)


def piece(tok, i):
    """Кусок текста токена; служебные — по имени."""
    return tok.id_to_token(i) if i < 5 else "«" + tok.decode([i]).replace("\n", "\\n") + "»"


def short(text, n):
    words = text.split()
    return " ".join(words[:n]) + (" …" if len(words) > n else "")


def wrap(label, text):
    """Подпись и текст, перенесённый по 92 знака с отступом."""
    return textwrap.fill(text, 92, initial_indent=label, subsequent_indent="    ")


# ----------------------------------------------------------------------------- 8.1 маска ошибки

def ch08_loss_mask():
    """Маска ошибки на паре «просьба -> история» и ошибка истории со своей и чужой просьбой."""
    tok = load_tokenizer()
    pairs = [(r["request"], r["story"]) for r in held_out()]
    ids, labels = encode(tok, *pairs[0], 512)
    inputs, targets = ids[:-1], labels[1:]        # цель позиции t — токен t+1, маска сдвинута с ней
    print(f"пара 1: {len(inputs)} позиций")
    print("поз. | вход              | цель            | в потере?")
    for t in list(range(12)) + list(range(len(inputs) - 4, len(inputs))):
        if t == len(inputs) - 4:
            print(" ... |")
        goal = piece(tok, ids[t + 1])
        print(f"{t:4d} | {piece(tok, inputs[t]):17s} | {goal:15s} | {'да' if targets[t] != -100 else 'нет'}")
    first = next(t for t in range(len(targets)) if targets[t] != -100)   # в окно 0–11 она не попала
    print(f"первая цель в потере: позиция {first}, вход {piece(tok, inputs[first])}, цель {piece(tok, ids[first + 1])}")

    n_req = n_story = in_loss = n_targets = 0
    for req, story in pairs:
        ids, labels = encode(tok, req, story, 512)
        n_req += len(tok.encode(req).ids)
        n_story += len(tok.encode(story).ids) + 1     # с <eos>
        in_loss += sum(l != -100 for l in labels[1:])
        n_targets += len(labels) - 1
    print(f"200 отложенных пар: просьба {n_req / 200:.1f} токена, история с <eos> {n_story / 200:.1f} токена,"
          f" в потере {in_loss / n_targets:.1%} целей")

    _, base = load("base")
    other = [(pairs[(i + 1) % 200][0], story) for i, (_, story) in enumerate(pairs)]  # чужая просьба
    print(f"база, ошибка на истории: со своей просьбой {story_loss(base, tok, pairs):.4f}, "
          f"с чужой {story_loss(base, tok, other):.4f}")


# ----------------------------------------------------------------------------- 8.2 LoRA

def ch08_lora_delta():
    """Насколько адаптер LoRA сдвинул веса BookLM; сингулярные числа поправки и основы."""
    base = torch.load(MODELS / "booklm-base.pt", map_location="cpu")
    file = torch.load(MODELS / "booklm-instruct-lora.pt", map_location="cpu")
    A, scale = file["adapter"], file["alpha"] / file["r"]
    rel = {}
    for b in range(6):
        for t in TYPES:
            name = f"blocks.{b}.{t}"
            dW = scale * A[name + ".B"] @ A[name + ".A"]          # ΔW = (α/r)·B·A
            rel[b, t] = (dW.norm() / base[name + ".weight"].norm()).item()
    print("||ΔW|| / ||W||, %")
    print("блок " + "".join(f"{n:>8s}" for n in NAMES))
    for b in range(6):
        print(f"{b:4d} " + "".join(f"{100 * rel[b, t]:8.2f}" for t in TYPES))
    with open(OUT / "ch08-lora-delta.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["block", "matrix", "rel_percent"])
        for (b, t), v in rel.items():
            w.writerow([b, NAMES[TYPES.index(t)], round(100 * v, 3)])

    name = "blocks.0.attention.q"
    dW = scale * A[name + ".B"] @ A[name + ".A"]
    s_dw = torch.linalg.svdvals(dW.double())
    s_w = torch.linalg.svdvals(base[name + ".weight"].double())
    print(f"сингулярные числа {name}, 1-20:")
    print("ΔW:", " ".join(f"{v:.2g}" for v in s_dw[:10]))
    print("   ", " ".join(f"{v:.2g}" for v in s_dw[10:20]))
    print("W: ", " ".join(f"{v:.2f}" for v in s_w[:10]))
    print("   ", " ".join(f"{v:.2f}" for v in s_w[10:20]))
    print(f"ΔW: 16-е число {s_dw[15]:.2f}, 17-е {s_dw[16]:.0e} — машинный ноль, ранг ΔW = {int((s_dw > 1e-4 * s_dw[0]).sum())}")


def ch08_lora_before_after():
    """Ответы базы и модели после LoRA на просьбы и доля ответов с <eos>."""
    tok, base = load("base")
    _, inst = load("instruct")
    asks = ["Расскажи сказку про ёжика, который боялся темноты.",
            "Напиши историю про девочку и красный шарик.",
            "Можешь рассказать, как кот научился плавать?"]
    for req in asks:
        print("Просьба:", req)
        for name, model in (("  база", base), ("  после LoRA", inst)):
            new, ended = sample(tok, model, prompt_ids(tok, req, chat=True), 100, seed=0)
            print(wrap(f"{name}: ", short(tok.decode(new), 22) + (" <eos>" if ended else "")))
    requests = [r["request"] for r in held_out()[:50]]
    for name, model in (("база", base), ("после LoRA", inst)):
        share, mean_len = eos_stats(tok, model, requests, 400)
        print(f"{name}: <eos> не позже 400 токенов у {share:.0%} из 50 ответов, средняя длина {mean_len:.0f}")


def ch08_lora_modes():
    """Ошибка BookLM в четырёх режимах адаптера и сверка логитов."""
    tok, base = load("base")
    adapter = adapter_model()
    _, merged = load("instruct")
    pairs = [(r["request"], r["story"]) for r in held_out()]
    summary = json.load(open(MODELS / "lora_summary.json"))
    counted = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
    print(f"обучаемых параметров: {summary['trainable_params']:,} (в журнале), {counted:,} (по requires_grad)")
    print(f"ошибка на целях истории, 200 отложенных пар:")
    print(f"  база                {story_loss(base, tok, pairs):.6f}")
    print(f"  адаптер             {story_loss(adapter, tok, pairs):.6f}")
    set_lora(adapter, False)
    print(f"  адаптер выключен    {story_loss(adapter, tok, pairs):.6f}")
    same = all(torch.equal(answer_logp(adapter, *encode(tok, *p, 512))[1],
                           answer_logp(base, *encode(tok, *p, 512))[1]) for p in pairs)
    set_lora(adapter, True)
    print(f"  слитая              {story_loss(merged, tok, pairs):.6f}")
    print(f"логиты выключенного адаптера и базы совпали бит в бит: {same}")
    diff = max((answer_logp(merged, *encode(tok, *p, 512))[1] -
                answer_logp(adapter, *encode(tok, *p, 512))[1]).abs().max().item() for p in pairs)
    print(f"наибольшее расхождение логитов слитой и адаптерной: {diff:.1e}")


# ----------------------------------------------------------------------------- 8.3 NF4

def nf4_roundtrip(w):
    """W -> коды NF4 (блок 64) и масштабы, масштабы -> 8 бит (группы 256) -> W~."""
    codes, absmax = quantize_nf4(w)
    q8, c = double_quant(absmax)
    return dequantize_nf4(codes, double_dequant(q8, c), w.shape), absmax, codes.numel() // 2 + q8.numel() + 4 * c.numel()


def ch08_qlora_booklm():
    """Цена четырёх бит NF4 на BookLM: ошибка весов, KL, перплексия, продолжения."""
    tok, model = load("base")
    q = copy.deepcopy(model)
    nbytes, err, outlier = 0, {t: [0.0, 0.0] for t in TYPES}, {t: (0.0, 0) for t in TYPES}
    with torch.no_grad():
        for b in range(6):
            for t in TYPES:
                w = model.get_submodule(f"blocks.{b}.{t}").weight
                w_hat, absmax, size = nf4_roundtrip(w)
                q.get_submodule(f"blocks.{b}.{t}").weight.copy_(w_hat)
                nbytes += size
                err[t][0] += (w_hat - w).pow(2).sum().item()
                err[t][1] += w.pow(2).sum().item()
                ratio = (absmax.max() / absmax.median()).item()   # выброс: max s_b / медиана s_b
                outlier[t] = max(outlier[t], (ratio, b))
    rest = sum(p.numel() for n, p in model.named_parameters() if "embed" in n or "norm" in n)
    nbytes += 2 * rest                                          # эмбеддинги и RMSNorm в BF16
    print(f"основа: {nbytes:,} байт")
    print("матрица  ошибка весов  max s_b / медиана (блок)")
    for t, n in zip(TYPES, NAMES):
        print(f"{n:8s} {math.sqrt(err[t][0] / err[t][1]):10.4f}   {outlier[t][0]:6.2f} ({outlier[t][1]})")

    data, kl, count = val_tokens(50_000), 0.0, 0
    with torch.no_grad():
        for s in range(0, len(data) - 512, 512):
            x = data[s:s + 512][None]
            lp, lq = model(x)[0].log_softmax(-1), q(x)[0].log_softmax(-1)
            kl += (lp.exp() * (lp - lq)).sum().item()           # KL(p_исх || p_NF4) по позициям
            count += 512
    p0, p1 = perplexity(model), perplexity(q)
    print(f"KL(p_исх || p_NF4) на {count:,} позициях: {kl / count:.4f} ната на позицию")
    print(f"перплексия: исходная {p0:.3f}, NF4 {p1:.3f} (+{p1 / p0 - 1:.1%})")
    ids = prompt_ids(tok, "Однажды")
    a, _ = sample(tok, model, ids, 60, seed=0)
    b, _ = sample(tok, q, ids, 60, seed=0)
    first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    print(f"«Однажды», 60 токенов, одно зерно; первые {first} токенов совпали:")
    print(wrap("  ", "Однажды" + tok.decode(a[:first])))
    print(wrap("  исходная дальше: ", short(tok.decode(a[first:]), 25)))
    print(wrap("  NF4 дальше: ", short(tok.decode(b[first:]), 25)))


# ----------------------------------------------------------------------------- 8.4 логарифм отношения

EZH = r"\b[её]ж(ик|ич)"                         # ёжик, ёжика, ежичек (тот же образец, что в ch09.py)


def ch08_kl_leash():
    """Логарифм отношения инструктивной BookLM к базе по токенам ответа."""
    tok, base = load("base")
    _, inst = load("instruct")
    request = "Расскажи сказку про ёжика, который боялся темноты."
    ids = prompt_ids(tok, request, chat=True)
    eos = special(tok)["<eos>"]
    sums, lengths, ends, answers = [], [], 0, []
    for seed in range(16):
        new, ended = sample(tok, inst, ids, 512 - len(ids), seed)
        y = new + [eos] if ended else new
        x = torch.tensor([ids + y])
        with torch.no_grad():                    # один прямой проход каждой модели
            lp_i = inst(x)[0, len(ids) - 1:-1].log_softmax(-1)
            lp_b = base(x)[0, len(ids) - 1:-1].log_softmax(-1)
        idx = torch.tensor(y)[:, None]
        ratio = (lp_i.gather(-1, idx) - lp_b.gather(-1, idx))[:, 0]   # log π_instr − log π_base по токенам
        sums.append(ratio.sum().item())
        lengths.append(len(y))
        ends += ended
        answers.append((seed, y, ratio, bool(re.search(EZH, tok.decode(new).lower()))))
    hedgehog = [a for a in answers if a[3]]
    print(f"16 ответов, {ends} закончились <eos>, средняя длина {sum(lengths) / 16:.0f} токенов, "
          f"ёжик есть в {len(hedgehog)}")
    print(f"оценка KL(π_instruct || π_base) на этой просьбе: {sum(sums) / 16:.1f} ната "
          f"(от {min(sums):.1f} до {max(sums):.1f})")
    seed, y, ratio, _ = (hedgehog or answers)[0]      # первый ответ, где есть ёжик
    with open(OUT / "ch08-kl-leash.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["position", "token", "log_ratio"])
        for t, (i, r) in enumerate(zip(y, ratio.tolist())):
            w.writerow([t + 1, tok.id_to_token(i) if i < 5 else tok.decode([i]), round(r, 4)])
    total = ratio.sum().item()
    print(f"ответ с зерном {seed}: {len(y)} токенов, сумма {total:.1f} ната; начало: {short(tok.decode(y[:12]), 10)}")
    print("пять токенов с наибольшим логарифмом отношения:")
    for r, t in zip(*ratio.topk(5)):
        print(f"  позиция {t.item() + 1:3d}  {piece(tok, y[t]):14s} {r.item():+.2f}")
    top = ratio.sort(descending=True).values
    k = max(1, len(y) // 10)
    print(f"первые 3 токена дают {ratio[:3].sum().item():+.1f}, <eos> {ratio[-1].item():+.1f}, "
          f"10% самых дорогих токенов {top[:k].sum().item():+.1f}, остальные {top[k:].sum().item():+.1f}")
    copied = [(t, i) for t, i in enumerate(y)                 # токены, взятые из просьбы
              if len(tok.decode([i]).strip()) >= 3 and tok.decode([i]).strip().lower() in request.lower()]
    print("слова просьбы в ответе:", ", ".join(f"{piece(tok, i)} {ratio[t].item():+.2f} (поз. {t + 1})"
                                             for t, i in copied) or "нет")


# ----------------------------------------------------------------------------- 8.5 DPO

def ch08_implicit_reward():
    """Неявная награда DPO инструктивной BookLM на парах ответов."""
    tok, base = load("base")
    _, inst = load("instruct")
    rows = held_out()
    beta = 0.1

    def logp(model, req, story):              # log π(y | x): сумма по маске истории и <eos>
        return answer_logp(model, *encode(tok, req, story, 512))[0].sum().item()

    def r_hat(req, story):                    # неявная награда β·log(π_instruct / π_base)
        return beta * (logp(inst, req, story) - logp(base, req, story))

    print("пара  log π_inst(y_w) log π_base(y_w)  log π_inst(y_l) log π_base(y_l)   r̂_w    r̂_l   ошибка")
    for i, row in enumerate(rows[:3]):
        x, y_w, y_l = row["request"], row["story"], rows[i + 1]["story"]
        pw, rw, pl, rl = logp(inst, x, y_w), logp(base, x, y_w), logp(inst, x, y_l), logp(base, x, y_l)
        r_w, r_l = beta * (pw - rw), beta * (pl - rl)
        loss = -F.logsigmoid(torch.tensor(r_w - r_l)).item()
        print(f"{i + 1:4d} {pw:15.1f} {rw:15.1f} {pl:16.1f} {rl:15.1f} {r_w:+7.2f} {r_l:+6.2f} {loss:7.4f}")

    for n in (20, 200):
        sample_rows = rows[:n]
        # (а) своя история против чужой под той же просьбой; (б) та же история под своей и под чужой просьбой
        for label, make in (("(а) своя и чужая история",
                             lambda i: (sample_rows[i]["request"], sample_rows[i]["story"],
                                        sample_rows[i]["request"], sample_rows[(i + 1) % n]["story"])),
                            ("(б) своя и чужая просьба",
                             lambda i: (sample_rows[i]["request"], sample_rows[i]["story"],
                                        sample_rows[(i + 1) % n]["request"], sample_rows[i]["story"]))):
            margins = torch.tensor([r_hat(*make(i)[:2]) - r_hat(*make(i)[2:]) for i in range(n)])
            print(f"{n:3d} пар, {label}: r̂_w > r̂_l в {int((margins > 0).sum()):3d}, "
                  f"ошибка DPO {-F.logsigmoid(margins).mean():.4f}")
    rw = logp(base, rows[0]["request"], rows[0]["story"])
    rl = logp(base, rows[0]["request"], rows[1]["story"])
    zero = beta * ((rw - rw) - (rl - rl))                      # политика = эталон = база
    print(f"политика = база: ошибка {-F.logsigmoid(torch.tensor(zero)).item():.4f} (ln 2 = {math.log(2):.4f})")


# ----------------------------------------------------------------------------- 8.6 забывание и измерения

def ch08_forgetting():
    """Забывание BookLM после SFT с LoRA: ошибка на историях и продолжения без диалога."""
    tok, base = load("base")
    _, inst = load("instruct")
    L = {name: math.log(perplexity(m)) for name, m in (("база", base), ("после LoRA", inst))}
    for name, value in L.items():
        print(f"{name:11s} L_pre = {value:.4f}, перплексия {math.exp(value):.3f}")
    print(f"Δ_forget = {L['после LoRA'] - L['база']:+.4f}")
    # Проверка «окна не видела ни при предобучении, ни при SFT»: часть историй val.bin есть в обучающей части sft.jsonl.
    data, T = val_tokens(50_000), 512
    story = torch.cumsum(torch.cat([torch.zeros(1, dtype=torch.long), (data == special(tok)["<eos>"]).long()[:-1]]), 0)
    seen = {" ".join(r["story"].split()[:12]) for r in sft_train()}
    leaked = {int(s) for s in story.unique() if " ".join(tok.decode(data[story == s].tolist()).split()[:12]) in seen}
    clean = {}
    with torch.no_grad():
        for name, m in (("база", base), ("после LoRA", inst)):
            total, count = 0.0, 0
            for s in range(0, len(data) - T - 1, T):
                loss = F.cross_entropy(m(data[s:s + T][None])[0], data[s + 1:s + T + 1], reduction="none")
                keep = torch.tensor([int(i) not in leaked for i in story[s + 1:s + T + 1]])
                total += loss[keep].sum().item()
                count += int(keep.sum())
            clean[name] = total / count
    print(f"в этих окнах {len(leaked)} из {len(story.unique())} историй были в обучающих парах SFT; без них "
          f"L_pre {clean['база']:.4f} и {clean['после LoRA']:.4f}, Δ_forget = {clean['после LoRA'] - clean['база']:+.4f}")
    ids = prompt_ids(tok, "Однажды летом маленькая девочка")
    for name, m in (("база", base), ("после LoRA", inst)):
        new, ended = sample(tok, m, ids, 80, seed=0)
        print(wrap(f"{name}: ", "Однажды летом маленькая девочка" + tok.decode(new) + (" <eos>" if ended else "")))
    for name, m in (("база", base), ("после LoRA", inst)):    # привычка ставить <eos> без формата диалога
        runs = [sample(tok, m, ids, 400, seed) for seed in range(20)]
        ended = [len(n) for n, e in runs if e]
        print(f"{name}: без формата диалога 20 продолжений до 400 токенов, <eos> у {len(ended)}, "
              f"средняя длина до <eos> {sum(ended) / max(1, len(ended)):.0f}")


def ch08_eval_protocol():
    """Таблица измерений базы, адаптера и слитой модели после SFT."""
    tok, base = load("base")
    adapter = adapter_model()
    _, merged = load("instruct")
    models = {"база": base, "адаптер": adapter, "слитая": merged}
    pairs = [(r["request"], r["story"]) for r in held_out()]
    requests = [r["request"] for r in held_out()[:50]]
    rows = {k: [] for k in ("ошибка на ответах", "L_pre", "Δ_forget", "доля <eos> ≤ 300", "средняя длина",
                            "KL к базе на ответах")}
    base_logits = [answer_logp(base, *encode(tok, *p, 512))[1] for p in pairs]
    for name, m in models.items():
        rows["ошибка на ответах"].append(story_loss(m, tok, pairs))
        rows["L_pre"].append(math.log(perplexity(m)))
        rows["Δ_forget"].append(rows["L_pre"][-1] - rows["L_pre"][0])
        share, mean_len = eos_stats(tok, m, requests, 300)
        rows["доля <eos> ≤ 300"].append(share)
        rows["средняя длина"].append(mean_len)
        kl, n = 0.0, 0
        for p, lb in zip(pairs, base_logits):       # KL(π_модель || π_база) на позициях ответа
            lm = answer_logp(m, *encode(tok, *p, 512))[1].log_softmax(-1)
            kl += (lm.exp() * (lm - lb.log_softmax(-1))).sum().item()
            n += len(lm)
        rows["KL к базе на ответах"].append(kl / n)
    print(f"{'метрика':22s}" + "".join(f"{k:>10s}" for k in models))
    for k, v in rows.items():
        fmt = "{:10.0%}" if "доля" in k else "{:10.1f}" if "длина" in k else "{:10.4f}"
        print(f"{k:22s}" + "".join(fmt.format(x) for x in v))


EXPERIMENTS = {f.__name__: f for f in (ch08_loss_mask, ch08_lora_delta, ch08_lora_before_after, ch08_lora_modes,
                                         ch08_qlora_booklm, ch08_kl_leash, ch08_implicit_reward,
                                         ch08_forgetting, ch08_eval_protocol)}

if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    torch.manual_seed(0)
    run(EXPERIMENTS)
