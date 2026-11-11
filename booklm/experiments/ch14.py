"""Глава 14 «Инференс»: опыты на обученной BookLM для рубрик «А теперь на нашей модели».

Опыты (время — на процессоре ноутбука Apple M2 Max):
  ch14_sampling           к разделу «Выбор следующего токена» главы 7, в книге не цитируется;
                          несколько секунд. Десять вероятных продолжений «…лиса. Однажды она», сколько токенов
                          оставляют ядерная выборка и порог от лидера, три продолжения разными способами выбора.
  ch14_ttft_booklm        рубрика «время до первого токена», подраздел «Метрики из журнала событий»;
                          несколько секунд. TTFT, TPOT, p95 ITL и E2E на одном потоке для входа 16, 128, 448 токенов.
  ch14_kv_cache           рубрика «KV-кэш», подраздел «KV-кэш BookLM»; около 15 с. Расхождение с полным
                          проходом при чтении кусками, байты кэша на токен в FP32 и BF16, время 400 жадных шагов
                          с кэшем и без. Ещё пишет ch14-kv-cache.csv: время шага по позиции для рисунка.
  ch14_quant              рубрика «квантование», подраздел «Число бит на вес с учётом масштабов»; около 2 мин.
                          Бит на вес, МБ и перплексия на 200 историях для BF16, INT8, INT4, MXFP4, трёх уровней
                          и трёх уровней после дообучения со STE (300 шагов); жадные тексты всех вариантов.
  ch14_paged_booklm       рубрика «страницы и непрерывный пакет», раздел «PagedAttention и vLLM»;
                          около 10 с. Длины 32 диалогов, память KV-кэша с резервом, блоками по 16
                          токенов и точно; вызовы модели у обычного и непрерывного пакета
                          (планировщик из labs/ch14/continuous_scheduler.py).
  ch14_ggml_graph_booklm  рубрика «граф ggml», подраздел «llama.cpp»; несколько секунд. Узлы графа
                          BookLM в llama.cpp, их число по блокам и состав блока. Нужны llama.cpp
                          (llama-eval-callback) и пакет gguf.
  ch14_tp_booklm          рубрика «тензорный параллелизм», подраздел «Тензорный параллелизм в блоке
                          трансформера»; несколько секунд. Время на токен у цельной модели и у двух процессов
                          (gloo), расхождение логитов, число и размер all-reduce, жадные тексты.
  ch14_onnx_booklm        рубрика «граф ONNX», подраздел «ONNX Runtime и Triton Inference Server»;
                          несколько секунд. Шаг генерации BookLM в ONNX (labs/ch14/onnx_booklm_step.py):
                          совпадение текста, расхождение логитов, время на токен у PyTorch, session.run
                          и I/O binding. Нужен onnxruntime.
  ch14_ollama_template    рубрика «шаблон диалога», подраздел «Ollama»; время не мерили, оно зависит от Ollama. Ответ с шаблоном
                          без <|assistant|> и с верным, скорость холодного и тёплых запросов, совпадение первых
                          20 токенов с PyTorch. Нужны запущенный сервер Ollama и пакет gguf.

Запуск из корня репозитория:
    python -m booklm.experiments.ch14                  # все опыты по очереди
    python -m booklm.experiments.ch14 ch14_quant       # один опыт
Вывод каждого опыта копируется в models/booklm/experiments/<имя опыта через дефис>.txt.
Старые имена ch15_kv_cache и ch15_ollama_template тоже работают. Ещё опыты главы — в ch14_disagg_radix.py
и ch14_mtp.py.

Опыты про скорость (TTFT, KV-кэш, ONNX, тензорный параллелизм) зависят от машины: числа в книге сняты
на Apple M2 Max, на одном потоке процессора, и заметно скачут, если машина занята другой работой.
"""
import csv
import math
import statistics
import sys
import textwrap
import time
from pathlib import Path

import torch
import torch.nn.functional as F

if not __package__:  # запуск файлом (python booklm/experiments/ch14.py): корень репозитория — в путь поиска
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from booklm.lab import MODELS, load, prompt_ids, special
from booklm.model import apply_rope

OUT = MODELS / "experiments"
FOX = "Жила-была маленькая лиса. Однажды она"


def show(text, width=95, indent="  "):
    print(textwrap.fill(" ".join(text.split()), width, initial_indent=indent, subsequent_indent=indent))


@torch.no_grad()
def greedy(tok, model, text, n):
    """Жадное продолжение text на n токенов (до <eos>)."""
    ids = torch.tensor([prompt_ids(tok, text)])
    out = model.generate(ids, max_new=n, temperature=0, eos_id=special(tok)["<eos>"])
    return tok.decode(out[0, ids.size(1):].tolist())


# ============================================================================ ch14-sampling

def top_p_mask(p, top_p):                       # из листинга главы
    sorted_p, idx = p.sort(descending=True)
    before = sorted_p.cumsum(dim=0) - sorted_p      # набрано до токена
    keep = torch.zeros_like(p, dtype=torch.bool)
    keep[idx] = before < top_p
    return keep


def min_p_mask(p, p_min):
    return p >= p_min * p.max()


@torch.no_grad()
def ch14_sampling():
    """К разделу «Выбор следующего токена» главы 7: вероятные продолжения и способы выбора."""
    tok, model = load("base")
    ids = torch.tensor([prompt_ids(tok, FOX)])
    probs = model(ids)[0, -1].softmax(-1)           # tau = 1
    top = probs.topk(10)
    print(f"«{FOX} …»: десять самых вероятных следующих токенов (tau = 1)")
    print("  " + "  ".join(f"{tok.decode([i])!r} {p:.2f}" for p, i in
                           zip(top.values[:5].tolist(), top.indices[:5].tolist())))
    print("  " + "  ".join(f"{tok.decode([i])!r} {p:.2f}" for p, i in
                           zip(top.values[5:].tolist(), top.indices[5:].tolist())))
    print(f"на десять лучших приходится {top.values.sum():.2f} вероятности")
    print("ядерная выборка, p = 0.9, оставляет токенов:", int(top_p_mask(probs, 0.9).sum()))
    print("порог от лидера, p_min = 0.1, оставляет токенов:", int(min_p_mask(probs, 0.1).sum()))
    eos = special(tok)["<eos>"]
    for name, temperature, top_k in [("жадно, tau = 0", 0, None),
                                     ("tau = 0.8, top_k = 40", 0.8, 40),
                                     ("tau = 1.5, без отсечения", 1.5, None)]:
        out = model.generate(ids, max_new=60, temperature=temperature, top_k=top_k, eos_id=eos, seed=0)
        print(f"\n{name}:")
        show(FOX + tok.decode(out[0, ids.size(1):].tolist()))


# ============================================================================ ch14-ttft-booklm

def long_story(min_len=448):
    """Первая отложенная история длиной не меньше min_len токенов (с <bos>)."""
    from booklm.lab import val_tokens
    data = val_tokens(10 ** 6)
    starts = (data == 1).nonzero().flatten().tolist() + [len(data)]
    for a, b in zip(starts, starts[1:]):
        if b - a >= min_len:
            return data[a:b]


@torch.no_grad()
def timed_run(model, ids, G=64):
    """Вход одним проходом с KV-кэшем, затем G-1 шагов по токену. Возвращает метки времени, нс."""
    start = time.perf_counter_ns()
    caches = [{"len": 0} for _ in model.blocks]
    logits = model(ids, caches)
    for c in caches:
        c["len"] = ids.size(1)
    marks = []
    for i in range(G):
        nxt = logits[:, -1].argmax(-1, keepdim=True)    # температура 0, <eos> не останавливает
        marks.append(time.perf_counter_ns())             # токен выбран
        if i == G - 1:
            break
        logits = model(nxt, caches)
        for c in caches:
            c["len"] += 1
    return start, marks


def ch14_ttft_booklm():
    """Рубрика «время до первого токена»: TTFT, TPOT, p95 ITL и E2E на одном потоке."""
    torch.set_num_threads(1)
    _, model = load("base")
    story = long_story(448)
    G, ms = 64, 1e6
    print(f"BookLM, один поток CPU, G = {G} новых токенов, медианы по 5 повторам, мс")
    print(f"{'вход':>5} {'TTFT':>7} {'TPOT':>6} {'p95 ITL':>8} {'E2E':>7} {'TTFT+(G-1)TPOT':>15}")
    for n in (16, 128, 448):
        ids = story[:n][None]
        timed_run(model, ids, G)                         # прогрев
        rows = []
        for _ in range(5):
            start, marks = timed_run(model, ids, G)
            itl = sorted((b - a) / ms for a, b in zip(marks, marks[1:]))
            ttft = (marks[0] - start) / ms
            tpot = (marks[-1] - marks[0]) / (G - 1) / ms
            rows.append((ttft, tpot, itl[math.ceil(0.95 * len(itl)) - 1], (marks[-1] - start) / ms))
        ttft, tpot, p95, e2e = (statistics.median(col) for col in zip(*rows))
        print(f"{n:>5} {ttft:>7.2f} {tpot:>6.2f} {p95:>8.2f} {e2e:>7.1f} {ttft + (G - 1) * tpot:>15.1f}")


# ============================================================================ ch14-kv-cache
# Код двух листингов раздела «А теперь на нашей модели: KV-кэш» (03-kv-cache.tex) один в один;
# первый проверяет run_listings, вывод второго (время) напечатан в книге как сохранённый.

def ch14_kv_cache():
    """Рубрика «KV-кэш»: чтение кусками, байты кэша на токен, время шага с кэшем и без."""
    from booklm.lab import load_model

    tok, booklm = load("base")                       # обученная BookLM
    text = "Жила-была маленькая лиса. Однажды она"
    ids = torch.tensor([prompt_ids(tok, text)])      # <bos> и восемь токенов

    @torch.no_grad()
    def run_in_pieces(model, ids, size):             # кусками по size токенов
        caches = [{"len": 0} for _ in model.blocks]  # по кэшу на блок
        pieces = []
        for s in range(0, ids.size(1), size):
            pieces.append(model(ids[:, s:s + size], caches))
            for c in caches:                     # сдвиг позиции
                c["len"] += pieces[-1].size(1)
        return torch.cat(pieces, dim=1), caches

    def kv_bytes_per_token(caches):
        total = sum(c["k"].nbytes + c["v"].nbytes for c in caches)
        return total // caches[0]["len"]

    with torch.no_grad():
        full = booklm(ids)                           # весь текст разом
    for size in (1, 4):
        logits, caches = run_in_pieces(booklm, ids, size)
        err = (logits - full).abs().max()
        print(f"кусками по {size}: расхождение с полным {err:.0e}")
    print("ключи блока 0:", list(caches[0]["k"].shape), caches[0]["k"].dtype,
          "| байт на токен:", kv_bytes_per_token(caches))
    _, caches = run_in_pieces(load_model("base").to(torch.bfloat16), ids, 1)
    print("в BF16:", caches[0]["k"].dtype,
          "| байт на токен:", kv_bytes_per_token(caches))
    print()

    torch.set_num_threads(1)                         # одно ядро процессора
    start = torch.tensor([prompt_ids(tok, "Жила-была маленькая лиса.")])

    @torch.no_grad()
    def greedy_timed(use_cache, n=400):        # время каждого шага
        caches = [{"len": 0} for _ in booklm.blocks] if use_cache else None
        x, new, times = start, start, []
        for _ in range(n):
            t0 = time.perf_counter()
            logits = booklm(new, caches) if use_cache else booklm(x)
            for c in caches or []:
                c["len"] += new.size(1)
            new = logits[:, -1:].argmax(dim=-1)      # [1, 1]
            x = torch.cat([x, new], dim=1)
            times.append((time.perf_counter() - t0) * 1e3)   # мс
        return x, times

    greedy_timed(True), greedy_timed(False)          # прогрев
    (a, fast), (b, slow) = greedy_timed(True), greedy_timed(False)
    print(f"тексты совпали: {torch.equal(a, b)}; весь ответ: "
          f"{sum(fast) / 1e3:.2f} с с кэшем, {sum(slow) / 1e3:.2f} с без")
    for pos in (50, 200, 400):
        i = pos - start.size(1) - 1            # шаг, где текст дорос до pos
        f = statistics.median(fast[i - 5:i + 5])
        s = statistics.median(slow[i - 5:i + 5])
        print(f"позиция {pos}: шаг {f:.2f} мс с кэшем, {s:.2f} мс без,"
              f" в {s / f:.1f} раза")
    # данные для рисунка: время шага по позиции, до которой дорос текст
    with open(OUT / "ch14-kv-cache.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["position", "cached_ms", "naive_ms"])
        for i, (x, y) in enumerate(zip(fast, slow)):
            w.writerow([start.size(1) + i + 1, round(x, 3), round(y, 3)])


# ============================================================================ ch14-quant

def groupwise_fake_quant(w, bits=4, group_size=128):          # из листинга главы
    qmax = 2 ** (bits - 1) - 1
    groups = w.reshape(w.size(0), -1, group_size)    # [out, групп, g]
    scales = groups.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / qmax
    q = torch.round(groups / scales).clamp(-qmax, qmax)
    return (q * scales).reshape_as(w), q.to(torch.int8), scales


def mxfp4_fake_quant(w):
    """MXFP4: блоки по 32, масштаб 2^e (E8M0), коды E2M1 — как mxfp4_encode_block, но сразу для матрицы."""
    grid = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    blocks = w.reshape(w.size(0), -1, 32)
    amax = blocks.abs().amax(-1, keepdim=True).clamp(min=2.0 ** -126)
    scale = 2.0 ** (torch.floor(torch.log2(amax)) - 2)
    x = (blocks / scale).abs().clamp(max=6.0)                      # насыщение до 6
    d = (x[..., None] - grid).abs()
    tie_odd = (d > d.min(-1, keepdim=True).values) * 2 + torch.arange(8) % 2
    nearest = grid[tie_odd.argmin(-1)]                             # ближайший, при равенстве — чётная мантисса
    return (nearest * blocks.sign() * scale).reshape_as(w)


def ternary_fake_quant(w, group_size=128):
    """Три уровня по среднему модулю: alpha = mean|W| на группу, Q = clip(round(W / alpha), -1, 1)."""
    groups = w.reshape(w.size(0), -1, group_size)
    alpha = groups.abs().mean(-1, keepdim=True).clamp(min=1e-8)
    return ((groups / alpha).round().clamp(-1, 1) * alpha).reshape_as(w)


def linear_layers(model):
    for b in model.blocks:
        for lin in (b.attention.q, b.attention.k, b.attention.v, b.attention.out,
                    b.ffn.gate, b.ffn.up, b.ffn.down):
            yield lin


def bf16_copy(kind="base"):
    """Модель, у которой все веса округлены до BF16 — так их хранят на диске."""
    _, model = load(kind)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(p.bfloat16().float())
    return model


def size_mib(model, bits):
    """Эмбеддинги и нормы — 16 бит, линейные слои — bits(d_in) бит на вес."""
    rest = sum(p.numel() for p in model.parameters()) - sum(l.weight.numel() for l in linear_layers(model))
    lin = sum(l.weight.numel() * bits(l.weight.size(1)) for l in linear_layers(model))
    return (rest * 16 + lin) / 8 / 2 ** 20


@torch.no_grad()
def story_perplexity(model, n=200):
    """Перплексия на первых n отложенных историях, каждая со своего <bos>, не длиннее 512 токенов."""
    from booklm.lab import val_tokens
    data = val_tokens(10 ** 6)
    starts = (data == 1).nonzero().flatten().tolist() + [len(data)]
    total, count = 0.0, 0
    for a, b in zip(starts[:n], starts[1:n + 1]):
        x = data[a:min(b, a + 513)]
        total += F.cross_entropy(model(x[None, :-1])[0], x[1:], reduction="sum").item()
        count += len(x) - 1
    return math.exp(total / count)


def train_ternary(steps=300, lr=1e-3, batch=16, T=256):
    """Тернарная BookLM: главные веса FP32, прямой проход с alpha*Q, градиент по STE.

    Честно уменьшенный вариант: не обучение с нуля на 100 млн токенов (на CPU это часы), а
    дообучение готовой базовой модели на steps * batch * T токенах.
    """
    import numpy as np
    from torch.nn.utils import parametrize

    class STE(torch.nn.Module):
        def forward(self, w):
            return w + (ternary_fake_quant(w) - w).detach()      # вперёд alpha*Q, назад — как без округления

    torch.manual_seed(0)
    _, model = load("base")
    for lin in linear_layers(model):
        parametrize.register_parametrization(lin, "weight", STE())
    data = np.fromfile(MODELS / "data" / "train.bin", dtype=np.uint16)
    data = data[:np.flatnonzero(data)[-1] + 1]                    # в train.bin хвост из нулей — отрезаем
    gen = torch.Generator().manual_seed(0)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    model.train()
    for step in range(steps):
        ix = torch.randint(len(data) - T - 1, (batch,), generator=gen).tolist()
        xy = torch.stack([torch.from_numpy(data[i:i + T + 1].astype("int64")) for i in ix])
        for g in opt.param_groups:                                # разогрев 20 шагов и косинус
            g["lr"] = lr * min(1.0, (step + 1) / 20) * 0.5 * (1 + math.cos(math.pi * step / steps))
        loss = F.cross_entropy(model(xy[:, :-1]).flatten(0, 1), xy[:, 1:].flatten())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad()
    model.eval()
    for lin in linear_layers(model):                              # оставляем только alpha*Q
        parametrize.remove_parametrizations(lin, "weight", leave_parametrized=True)
    with torch.no_grad():                                         # как у остальных строк: на диске всё в BF16
        for p in model.parameters():                              # (alpha*Q -> alpha в BF16, эмбеддинги и нормы)
            p.copy_(p.bfloat16().float())
    return model, steps * batch * T


def ch14_quant():
    """Рубрика «квантование»: бит на вес, МБ и перплексия у форматов от BF16 до трёх уровней."""
    torch.set_num_threads(8)
    tok, _ = load("base")
    variants = [  # имя, приближение матрицы, бит на вес по длине строки d_in
        ("BF16", None, lambda d: 16),
        ("INT8, строка", lambda w: groupwise_fake_quant(w, 8, w.size(1))[0], lambda d: 8 + 16 / d),
        ("INT4, g=128", lambda w: groupwise_fake_quant(w, 4, 128)[0], lambda d: 4.125),
        ("INT4, g=32", lambda w: groupwise_fake_quant(w, 4, 32)[0], lambda d: 4.5),
        ("INT4, строка", lambda w: groupwise_fake_quant(w, 4, w.size(1))[0], lambda d: 4 + 16 / d),
        ("MXFP4", mxfp4_fake_quant, lambda d: 4.25),
        ("три уровня, g=128", ternary_fake_quant, lambda d: 1.725),
    ]
    rows, texts = [], []
    for name, quant, bits in variants:
        model = bf16_copy("base")
        if quant is not None:
            with torch.no_grad():
                for lin in linear_layers(model):
                    lin.weight.copy_(quant(lin.weight))
        rows.append((name, bits(256), size_mib(model, bits), story_perplexity(model)))
        texts.append((name, greedy(tok, model, FOX, 40)))
    model, seen = train_ternary()
    name = "три уровня + STE"
    rows.append((name, 1.725, size_mib(model, lambda d: 1.725), story_perplexity(model)))
    texts.append((name, greedy(tok, model, FOX, 40)))

    print(f"{'вариант':<19}{'бит/вес':>8}{'МБ':>7}{'перплексия':>12}   (200 отложенных историй)")
    for name, b, mib, ppl in rows:
        print(f"{name:<19}{b:>8.3f}{mib:>7.2f}{ppl:>12.2f}")
    print(f"«три уровня + STE»: базовая модель, дообученная с тернарными весами на {seen / 1e6:.1f} млн токенов")
    print(f"\nжадно, 40 токенов после «{FOX}» (одинаковые тексты объединены):")
    groups = {}
    for name, text in texts:
        groups.setdefault(text, []).append(name)
    for text, names in groups.items():
        print(textwrap.fill(" ".join(text.split()), 95, initial_indent=" = ".join(names) + ": ",
                            subsequent_indent="  "))


# ============================================================================ ch14-paged-booklm

HEROES = ["кота", "собаку", "девочку", "мальчика", "лису", "зайца", "медведя", "птичку",
          "рыбку", "ёжика", "дракона", "принцессу", "робота", "машинку", "мяч", "облако",
          "солнце", "дерево", "цветок", "лягушку", "утёнка", "щенка", "котёнка", "сову",
          "мышку", "слона", "жирафа", "пирата", "корабль", "звезду", "бабушку", "друзей"]


@torch.no_grad()
def dialog_lengths(tok, model):
    """Длины входа и ответа (с <eos>) на 32 просьбы, tau = 0.8, top_k = 40, seed = номер просьбы."""
    eos = special(tok)["<eos>"]
    rows = []
    for seed, hero in enumerate(HEROES):
        ids = torch.tensor([prompt_ids(tok, f"Напиши сказку про {hero}.", chat=True)])
        out = model.generate(ids, max_new=512 - ids.size(1), temperature=0.8, top_k=40,
                             eos_id=eos, seed=seed)
        rows.append((ids.size(1), out.size(1) - ids.size(1)))
    return rows


def ch14_paged_booklm():
    """Рубрика «страницы и непрерывный пакет»: память KV-кэша и вызовы модели."""
    from collections import deque
    sys.path.insert(0, str(MODELS.parents[1]))
    from labs.ch14.continuous_scheduler import BlockPool, Request, scheduler_tick

    tok, model = load("instruct")
    rows = dialog_lengths(tok, model)
    total = [p + a for p, a in rows]
    print(f"32 диалога: вход {min(p for p, _ in rows)}–{max(p for p, _ in rows)} токенов, "
          f"ответ {min(a for _, a in rows)}–{max(a for _, a in rows)}")
    print(f"длина диалога: мин {min(total)}, медиана {statistics.median(total):.0f}, макс {max(total)}")

    per_token = 2 * 6 * 2 * 32 * 2                    # K и V, 6 блоков, 2 головы, D_h = 32, BF16
    pool = BlockPool(n_blocks=10_000, block_size=16)
    for i, n in enumerate(total):
        pool.add_request(i)
        for _ in range(n):
            pool.append_token(i)
    blocks = sum(len(t) for t in pool.tables.values())
    exact = sum(total) * per_token
    reserve = 32 * 512 * per_token
    paged = blocks * 16 * per_token
    print("\nпамять KV-кэша 32 диалогов (BF16, 1536 байт на токен):")
    print(f"  резерв на 512 токенов: {reserve / 2**20:5.2f} МБ, в {reserve / exact:.1f} раза больше точной суммы")
    print(f"  блоки по 16 токенов:   {paged / 2**20:5.2f} МБ, на {paged / exact - 1:.1%} больше ({blocks} блока)")
    print(f"  точная сумма:          {exact / 2**20:5.2f} МБ")

    # Непрерывный пакет: те же 32 запроса, 8 мест; вход идёт целиком в первом вызове вместе с 1-м токеном
    waiting = deque(Request(i, p, a) for i, (p, a) in enumerate(rows))
    running, pool, calls = [], BlockPool(n_blocks=10_000, block_size=16), 0
    while waiting or running:
        running, _, _ = scheduler_tick(waiting, running, pool, max_batch=8)
        calls += 1
    answers = sum(a for _, a in rows)
    # Обычный пакет: 4 пакета по 8, пакет живёт, пока не допишет самый длинный ответ
    static = sum(max(a for _, a in rows[i:i + 8]) for i in range(0, 32, 8))
    print("\nвызовов модели при 8 местах в пакете:")
    print(f"  обычный пакет:     {static:4d}, пустых мест {1 - answers / (8 * static):.0%}")
    print(f"  непрерывный пакет: {calls:4d}, пустых мест {1 - answers / (8 * calls):.0%}")


# ============================================================================ GGUF

def write_gguf(kind, path, qtype="F16"):
    """BookLM -> GGUF архитектуры llama: метаданные, токенизатор и тензоры под именами llama.cpp.

    Это то же, что делает convert_hf_to_gguf.py, только для одной нашей модели. Переставлять строки
    W_Q и W_K не нужно: llama.cpp, как и BookLM, поворачивает соседние пары координат.
    """
    import json
    import gguf
    state = torch.load(MODELS / f"booklm-{kind}.pt", map_location="cpu")
    vocab = json.load(open(MODELS / "tokenizer.json", encoding="utf-8"))["model"]
    w = gguf.GGUFWriter(str(path), "llama")
    w.add_name(f"BookLM {kind}")
    w.add_context_length(512)
    w.add_embedding_length(256)
    w.add_block_count(6)
    w.add_feed_forward_length(768)
    w.add_head_count(8)
    w.add_head_count_kv(2)
    w.add_rope_dimension_count(32)
    w.add_rope_freq_base(10000.0)
    w.add_layer_norm_rms_eps(1e-6)
    w.add_vocab_size(8192)
    w.add_file_type(gguf.LlamaFileType.MOSTLY_Q8_0 if qtype == "Q8_0" else gguf.LlamaFileType.MOSTLY_F16)
    # токенизатор: байтовый BPE с разбиением GPT-2, служебные токены 0–4
    tokens = sorted(vocab["vocab"], key=vocab["vocab"].get)
    w.add_tokenizer_model("gpt2")
    w.add_tokenizer_pre("gpt-2")
    w.add_token_list(tokens)
    w.add_token_types([gguf.TokenType.CONTROL if i < 5 else gguf.TokenType.NORMAL for i in range(len(tokens))])
    w.add_token_merges([" ".join(m) if isinstance(m, list) else m for m in vocab["merges"]])
    w.add_bos_token_id(1)
    w.add_eos_token_id(2)
    w.add_pad_token_id(0)
    w.add_add_bos_token(True)
    names = {"embed.weight": "token_embd.weight", "norm.weight": "output_norm.weight"}
    for i in range(6):
        for ours, theirs in [("norm_1", "attn_norm"), ("attention.q", "attn_q"), ("attention.k", "attn_k"),
                             ("attention.v", "attn_v"), ("attention.out", "attn_output"),
                             ("norm_2", "ffn_norm"), ("ffn.gate", "ffn_gate"), ("ffn.up", "ffn_up"),
                             ("ffn.down", "ffn_down")]:
            names[f"blocks.{i}.{ours}.weight"] = f"blk.{i}.{theirs}.weight"
    for ours, theirs in names.items():
        t = state[ours].float().numpy()
        if t.ndim == 1:                                   # нормы — в FP32, как у convert_hf_to_gguf
            w.add_tensor(theirs, t)
        elif qtype == "Q8_0":
            w.add_tensor(theirs, gguf.quants.quantize(t, gguf.GGMLQuantizationType.Q8_0),
                         raw_dtype=gguf.GGMLQuantizationType.Q8_0)
        else:
            w.add_tensor(theirs, t.astype("float16"))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


# ============================================================================ ch14-ggml-graph-booklm

def ch14_ggml_graph_booklm():
    """Рубрика «граф ggml»: узлы графа BookLM в llama.cpp."""
    import re
    import subprocess
    import tempfile
    path = f"{tempfile.gettempdir()}/booklm-base-f16.gguf"
    write_gguf("base", path, "F16")
    # -dev none: только процессор; -t 1: один поток. Опция --no-warmup в этой сборке llama.cpp
    # не принимается, а с -n 1 прогрева и так нет: граф считается один раз, на входе.
    log = subprocess.run(["llama-eval-callback", "-m", path, "-p", "Мама мыла раму", "-n", "1",
                          "-t", "1", "-dev", "none"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True).stdout
    # строки вида «Qcur-0 = (f32) MUL_MAT(blk.0.attn_q.weight{256, 256, 1, 1}, ...) = {256, 6, 1, 1}»
    nodes = re.findall(r"common_debug_cb_eval:\s+(.+?) = \(\w+\)\s+(\w+)\(.*\) = \{([\d, ]+)\}", log)
    n_tokens = re.search(r"number of input tokens = (\d+)", log).group(1)
    print(f"граф BookLM в llama.cpp, вход «Мама мыла раму»: {n_tokens} токенов вместе с <bos>")
    print(f"{'узел':<22}{'операция':<12}форма (ne0, ne1, ...)")
    for name, op, shape in nodes[:12]:
        print(f"{name:<22}{op:<12}[{shape}]")
    print("...")
    name, op, shape = nodes[-1]
    print(f"{name:<22}{op:<12}[{shape}]")
    # блок i — от его первого RMS_NORM (norm-i) до остаточного сложения l_out-i
    names = [n for n, _, _ in nodes]
    starts = [names.index(f"norm-{i}") for i in range(6)]
    ends = [names.index(f"l_out-{i}") for i in range(6)]
    per_block = [e - s + 1 for s, e in zip(starts, ends)]
    ops = {}
    for _, op, _ in nodes[starts[0]:ends[0] + 1]:
        ops[op] = ops.get(op, 0) + 1
    print(f"\nвсего узлов {len(nodes)}, из них MUL_MAT {sum(o == 'MUL_MAT' for _, o, _ in nodes)}")
    print("узлов в блоках 0–5:", per_block)
    print("состав блока 0:", ", ".join(f"{op} {n}" for op, n in sorted(ops.items(), key=lambda x: -x[1])))
    last = [f"{n} = {o}" for n, o, _ in nodes[starts[5]:ends[5] + 1] if o == "GET_ROWS"]
    print("в блоке 5 ещё", ", ".join(last), "— дальше считается только последний токен")


# ============================================================================ ch14-tp-booklm

def shard(model, rank, p=2):
    """Веса процесса rank при TP = p: целые головы Q и K/V, каналы SwiGLU; эмбеддинги и нормы — копии."""
    hq, hkv, dh, ff = 8 // p, 2 // p, 32, 768 // p
    q, kv, f = slice(rank * hq * dh, (rank + 1) * hq * dh), slice(rank * hkv * dh, (rank + 1) * hkv * dh), \
        slice(rank * ff, (rank + 1) * ff)
    blocks = []
    for b in model.blocks:
        a = b.attention
        blocks.append(dict(norm_1=b.norm_1, norm_2=b.norm_2,
                           q=a.q.weight[q], k=a.k.weight[kv], v=a.v.weight[kv],
                           out=a.out.weight[:, q],                       # строки W_O -> столбцы weight
                           gate=b.ffn.gate.weight[f], up=b.ffn.up.weight[f], down=b.ffn.down.weight[:, f]))
    return dict(blocks=blocks, hq=hq, hkv=hkv, model=model)


@torch.no_grad()
def tp_forward(s, ids, caches, all_reduce):
    """Проход одного процесса. caches: список словарей {"k", "v"} по блокам или None."""
    m, (b, t) = s["model"], ids.shape
    start = 0 if caches[0].get("k") is None else caches[0]["k"].size(2)
    cos, sin = m.rope_cos[start:], m.rope_sin[start:]
    h = m.embed(ids)
    for blk, cache in zip(s["blocks"], caches):
        x = blk["norm_1"](h)
        q = (x @ blk["q"].T).view(b, t, s["hq"], 32).transpose(1, 2)
        k = (x @ blk["k"].T).view(b, t, s["hkv"], 32).transpose(1, 2)
        v = (x @ blk["v"].T).view(b, t, s["hkv"], 32).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if cache.get("k") is not None:
            k, v = torch.cat([cache["k"], k], 2), torch.cat([cache["v"], v], 2)
        cache["k"], cache["v"] = k, v
        g = s["hq"] // s["hkv"]
        y = F.scaled_dot_product_attention(q, k.repeat_interleave(g, 1), v.repeat_interleave(g, 1),
                                           is_causal=t > 1)
        part = y.transpose(1, 2).reshape(b, t, -1) @ blk["out"].T
        h = h + all_reduce(part)                                   # обмен 1: после внимания
        x = blk["norm_2"](h)
        part = (F.silu(x @ blk["gate"].T) * (x @ blk["up"].T)) @ blk["down"].T
        h = h + all_reduce(part)                                   # обмен 2: после SwiGLU
    return m.norm(h) @ m.embed.weight.T


@torch.no_grad()
def tp_greedy(s, tok, text, n, all_reduce):
    caches = [{} for _ in range(6)]
    ids = torch.tensor([prompt_ids(tok, text)])
    logits, out, times = tp_forward(s, ids, caches, all_reduce), [], []
    for _ in range(n):
        t0 = time.perf_counter()
        nxt = logits[:, -1].argmax(-1, keepdim=True)
        out.append(nxt.item())
        logits = tp_forward(s, nxt, caches, all_reduce)
        times.append(time.perf_counter() - t0)
    return tok.decode(out), statistics.median(times[5:])


def tp_worker(rank, p, port, result):
    import torch.distributed as dist
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=p)
    tok, model = load("base")
    s = shard(model, rank, p)
    log = []

    def all_reduce(x):
        log.append(x.numel() * x.element_size())                   # байт в одном сообщении
        dist.all_reduce(x)
        return x

    ids = torch.tensor([prompt_ids(tok, "Мама мыла раму, а потом она пошла спать.")])
    diff = (tp_forward(s, ids, [{} for _ in range(6)], all_reduce) - model(ids)).abs().max().item()
    tp_greedy(s, tok, "Жила-была девочка", 10, all_reduce)          # прогрев
    log.clear()
    text, per_token = tp_greedy(s, tok, "Жила-была девочка", 40, all_reduce)
    if rank == 0:
        result.update(diff=diff, text=text, per_token=per_token,
                      calls=len(log) // 41, msg=log[-1])           # 1 проход по входу + 40 шагов
    dist.destroy_process_group()


def ch14_tp_booklm():
    """Рубрика «тензорный параллелизм»: BookLM на двух процессах, время и обмены."""
    import torch.multiprocessing as mp
    torch.set_num_threads(1)
    tok, model = load("base")
    whole = shard(model, 0, p=1)                                   # «разрезанная» на 1 часть = цельная
    text, per_token = tp_greedy(whole, tok, "Жила-была девочка", 40, lambda x: x)
    print(f"цельная модель, 1 процесс:  {per_token * 1e3:.2f} мс на токен")
    with mp.Manager() as manager:
        result = manager.dict()
        mp.spawn(tp_worker, args=(2, 29513, result), nprocs=2)
        r = dict(result)
    print(f"TP = 2, 2 процесса (gloo):  {r['per_token'] * 1e3:.2f} мс на токен")
    print(f"расхождение логитов с цельной моделью: {r['diff']:.1e}")
    print(f"all-reduce на токен: {r['calls']}, сообщение на шаге генерации: {r['msg']} байт (FP32)")
    print("жадно, 40 токенов, цельная:")
    show("Жила-была девочка" + text)
    if text == r["text"]:
        print("у TP = 2 тот же текст, до токена")
    else:
        print("у TP = 2 текст другой:")
        show("Жила-была девочка" + r["text"])
    hkv, p = 2, 4
    print(f"TP = {p}: голов K,V {hkv} на {p} процесса — по {hkv / p} головы на процесс; "
          f"без копирования голов K,V не разрезать")


# ============================================================================ ch14-onnx-booklm

def ch14_onnx_booklm():
    """Рубрика «граф ONNX»: шаг генерации в ONNX Runtime против PyTorch."""
    import tempfile
    import numpy as np
    import onnxruntime as ort
    sys.path.insert(0, str(MODELS.parents[1]))
    from labs.ch14.onnx_booklm_step import BookLMDecodeStep, export_step

    torch.set_num_threads(1)
    tok, model = load("base")
    step = BookLMDecodeStep(model).eval()
    path = f"{tempfile.gettempdir()}/booklm_base_step.onnx"
    export_step(step, path)                                   # оси B и S изменяемые
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = opts.inter_op_num_threads = 1  # тоже один поток
    session = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
    prompt = prompt_ids(tok, "Жила-была девочка")
    N = 60

    def run(call):
        """Вход подаём по токену, потом N токенов жадно. call(token, pos, past) -> (logits, past)."""
        past, out, logs, times = None, [], [], []
        tokens = list(prompt)
        for pos in range(len(prompt) + N - 1):
            t0 = time.perf_counter()
            logits, past = call(tokens[pos], pos, past)
            nxt = int(np.argmax(logits))
            times.append(time.perf_counter() - t0)
            logs.append(np.asarray(logits))
            if pos >= len(prompt) - 1:
                tokens.append(nxt)
        return tokens[len(prompt):], logs, statistics.median(times[len(prompt):])

    empty = np.zeros((6, 1, 2, 0, 32), dtype=np.float32)

    def torch_step(token, pos, past):
        k, v = (torch.from_numpy(empty), torch.from_numpy(empty)) if past is None else past
        with torch.no_grad():
            logits, k, v = step(torch.tensor([[token]]), torch.tensor([[pos]]), k, v)
        return logits[0, -1].numpy(), (k, v)

    def ort_run(token, pos, past):                            # массивы NumPy туда и обратно
        k, v = (empty, empty) if past is None else past
        logits, k, v = session.run(None, {"input_ids": np.array([[token]]), "position_ids": np.array([[pos]]),
                                          "past_k": k, "past_v": v})
        return logits[0, -1], (k, v)

    b = session.io_binding()

    def ort_binding(token, pos, past):                        # кэш остаётся в OrtValue
        k, v = (ort.OrtValue.ortvalue_from_numpy(empty),) * 2 if past is None else past
        b.clear_binding_inputs()
        b.clear_binding_outputs()
        b.bind_cpu_input("input_ids", np.array([[token]]))
        b.bind_cpu_input("position_ids", np.array([[pos]]))
        b.bind_ortvalue_input("past_k", k)
        b.bind_ortvalue_input("past_v", v)
        for name in ("logits", "present_k", "present_v"):
            b.bind_output(name)
        session.run_with_iobinding(b)
        logits, k, v = b.get_outputs()
        return logits.numpy()[0, -1], (k, v)

    for call in (torch_step, ort_run, ort_binding):            # прогрев
        run(call)
    a, la, ta = run(torch_step)
    o, lo, to = run(ort_run)
    c, _, tc = run(ort_binding)
    diff = max(float(np.abs(x - y).max()) for x, y in zip(la, lo))
    print("PyTorch, жадно 60 токенов:")
    show("Жила-была девочка" + tok.decode(a))
    print("ONNX Runtime: тот же текст, до токена" if a == o == c else "ONNX Runtime: текст другой!")
    if a != o:
        show("Жила-была девочка" + tok.decode(o))
    print(f"наибольшее расхождение логитов за {len(la)} шагов: {diff:.1e}")
    print("время на токен, один поток CPU:")
    print(f"  PyTorch, BookLMDecodeStep      {ta * 1e3:.2f} мс")
    print(f"  ONNX Runtime, session.run      {to * 1e3:.2f} мс")
    print(f"  ONNX Runtime, I/O binding      {tc * 1e3:.2f} мс")


# ============================================================================ ch14-ollama-template

TEMPLATE_OK = """
{{- range .Messages }}
{{- if eq .Role "user" }}<|user|>{{ .Content }}<|assistant|>
{{- else if eq .Role "assistant" }}{{ .Content }}<eos>
{{- end }}
{{- end }}"""
TEMPLATE_BAD = TEMPLATE_OK.replace("<|assistant|>", "")      # тот же шаблон без <|assistant|>


def ollama(path, **body):
    import json
    import urllib.request
    req = urllib.request.Request("http://127.0.0.1:11434" + path, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req).read())


def ch14_ollama_template():
    """Рубрика «шаблон диалога» (Ollama): два шаблона, скорость и сверка с PyTorch."""
    import subprocess
    import tempfile
    from pathlib import Path
    tmp = Path(tempfile.gettempdir())
    write_gguf("instruct", tmp / "booklm-instruct-Q8_0.gguf", "Q8_0")
    for name, template in [("booklm-ch14:q8", TEMPLATE_OK), ("booklm-ch14:no-assistant", TEMPLATE_BAD)]:
        (tmp / "Modelfile").write_text(f'FROM ./booklm-instruct-Q8_0.gguf\nTEMPLATE """{template}"""\n'
                                       "PARAMETER stop <eos>\nPARAMETER num_ctx 512\n"
                                       "PARAMETER temperature 0.8\nPARAMETER top_k 40\n")
        subprocess.run(["ollama", "create", name, "-f", "Modelfile"], cwd=tmp, check=True,
                       capture_output=True)
    request = "Напиши сказку про кота, который боялся темноты."
    ask = [{"role": "user", "content": request}]

    def chat(model):
        return ollama("/api/chat", model=model, messages=ask, stream=False, options={"temperature": 0})

    bad = chat("booklm-ch14:no-assistant")
    ollama("/api/generate", model="booklm-ch14:q8", keep_alive=0)   # выгрузить: следующий запрос холодный
    runs = [chat("booklm-ch14:q8") for _ in range(3)]
    print(f"Просьба: {request}  (temperature = 0)")
    print("\nшаблон без <|assistant|>:")
    show(bad["message"]["content"][:300].rsplit(" ", 1)[0] + " …")
    print("верный шаблон:")
    show(runs[0]["message"]["content"][:300].rsplit(" ", 1)[0] + " …")
    print(f"\n{'запрос':<9}{'вход':>5}{'ответ':>6}{'вход, ток/с':>13}{'ответ, ток/с':>14}{'загрузка':>10}")
    for name, r in zip(["холодный", "тёплый", "тёплый"], runs):
        print(f"{name:<9}{r['prompt_eval_count']:>5}{r['eval_count']:>6}"
              f"{r['prompt_eval_count'] / r['prompt_eval_duration'] * 1e9:>13.0f}"
              f"{r['eval_count'] / r['eval_duration'] * 1e9:>14.0f}"
              f"{r['load_duration'] / r['total_duration']:>10.0%}")

    # те же 20 первых токенов у PyTorch: BookLM.generate жадно, веса BF16 -> FP32
    tok, model = load("instruct")
    ids = torch.tensor([prompt_ids(tok, request, chat=True)])
    with torch.no_grad():
        mine = model.generate(ids, max_new=20, temperature=0, eos_id=2)[0, ids.size(1):].tolist()
    theirs = tok.encode(runs[0]["message"]["content"]).ids[:20]
    same = next((i for i, (a, b) in enumerate(zip(mine, theirs)) if a != b), 20)
    print(f"\nвход у PyTorch {ids.size(1)} токенов; первые 20 токенов ответа совпали: {same} из 20")
    for name in ("booklm-ch14:q8", "booklm-ch14:no-assistant"):         # убираем за собой
        subprocess.run(["ollama", "rm", name], capture_output=True)


# ============================================================================ запуск

EXPERIMENTS = {f.__name__: f for f in (ch14_sampling, ch14_ttft_booklm, ch14_kv_cache, ch14_quant,
                                        ch14_paged_booklm, ch14_ggml_graph_booklm, ch14_tp_booklm,
                                        ch14_onnx_booklm, ch14_ollama_template)}
EXPERIMENTS["ch15_kv_cache"] = ch14_kv_cache                          # старые имена
EXPERIMENTS["ch15_ollama_template"] = ch14_ollama_template


if __name__ == "__main__":
    from booklm.experiments import run
    run(EXPERIMENTS, save=True)      # копия вывода в models/booklm/experiments/<имя опыта>.txt
