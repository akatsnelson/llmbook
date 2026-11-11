"""Глава 17 «Выпуск и сопровождение модели»: опыты на обученной BookLM для рубрик «А теперь на нашей модели».

Опыты (время — на процессоре ноутбука Apple M2 Max):
  ch17_sha_real_weights  рубрика «контрольная сумма», раздел «Родословная модели»; около 5 с.
                         SHA-256 файлов весов и родители merged-v1 из манифеста; один перевёрнутый бит в весе:
                         новая сумма, остановка проверки родителя, перплексия и ответ до и после.
  ch17_three_checks      рубрика «три проверки», раздел «Три проверки перед выпуском»; около 15 с.
                         Исходная модель и её копии в 4 и 3 бита: перплексия, время 200 токенов (пять замеров
                         вперемешку с исходной), три порога и решение «выпускаем». Время зависит от машины.
  ch17_onnx_step         рубрика «допуск по точности», раздел «Манифест графа ONNX»; несколько секунд.
                         Размер и число узлов графа шага в ONNX, расхождение ONNX Runtime с PyTorch при разной
                         длине кэша, расхождение логитов BF16 с FP32 и сколько жадных токенов из 2000 меняется.
                         Нужен onnxruntime.
  ch17_kv_version        рубрика «откат и KV-кэш», подраздел «Откат и KV-кэш»; несколько секунд с готовым
                         адаптером. Расхождение ключей кэша двух версий по блокам и перплексия продолжений:
                         v2 со своим кэшем, v1 с кэшем v2 и v1 заново. Без готового файла сначала дообучает
                         на CPU адаптер LoRA на одну эпоху (около 10 минут) и кладёт его
                         в models/booklm/experiments/lora/; следующие запуски берут готовый файл.

Запуск из корня репозитория:
    python -m booklm.experiments.ch17                    # все опыты по очереди
    python -m booklm.experiments.ch17 ch17_three_checks  # один опыт
Вывод каждого опыта копируется в models/booklm/experiments/<имя опыта через дефис>.txt.
Ещё два опыта главы — в ch17_server_regression.py.
"""
import hashlib
import json
import math
import random
import statistics
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F

if not __package__:  # запуск файлом (python booklm/experiments/ch17.py): корень репозитория — в путь поиска
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from booklm.lab import (MODELS, generate, load, load_model, load_tokenizer, perplexity, prompt_ids,
                        quantize_copy, time_generation, val_tokens)

from booklm.sft_lora import add_lora, encode, merged_state

OUT = MODELS / "experiments"


# ----------------------------------------------------------------------------- ch17-sha-real-weights

def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):      # частями по 1 МБ
            digest.update(block)
    return digest.hexdigest()


def verify_parents(child: dict, files: dict) -> None:
    """Как в листинге главы 17: сумма каждого родителя из child["parents"] против файла."""
    for parent_id, expected in child["parents"].items():
        actual = sha256_of(files[parent_id])
        if actual != expected:
            raise ValueError(f"{child['id']}: родитель {parent_id} "
                             f"{expected[:8]} != {actual[:8]}")


@torch.no_grad()
def greedy(tok, model, text, n=40):
    return generate(tok, model, text, chat=True, max_new=n, temperature=0)


def ch17_sha_real_weights():
    """Рубрика «контрольная сумма»: SHA-256 весов, родители и один перевёрнутый бит."""
    files = {"tokenizer-v1": "tokenizer.json", "base-step-N": "booklm-base.pt",
             "lora-v1": "booklm-instruct-lora.pt", "merged-v1": "booklm-instruct.pt"}
    sums = {}
    for name, f in files.items():
        sums[name] = sha256_of(MODELS / f)
        print(f"{name:<13}{f:<25}{(MODELS / f).stat().st_size:>10} байт  {sums[name][:16]}")
    lineage = {"lora-v1": ["base-step-N"], "merged-v1": ["base-step-N", "lora-v1"],
               "onnx-step-v1": ["merged-v1"]}
    for child, parents in lineage.items():               # манифест ребёнка хранит суммы родителей
        print(f"манифест {child}: parents=" + str({p: sums[p][:16] for p in parents}))
    merged = {"id": "merged-v1", "parents": {p: sums[p] for p in lineage["merged-v1"]}}
    verify_parents(merged, {name: MODELS / f for name, f in files.items()})
    print("оба родителя merged-v1 на месте")

    tok = load_tokenizer()
    state = torch.load(MODELS / "booklm-instruct.pt", map_location="cpu")
    w = state["blocks.3.ffn.up.weight"]
    before = w[0, 0].item()
    w.view(torch.int32)[0, 0] ^= 1                       # младший бит мантиссы одного числа
    print(f"blocks.3.ffn.up.weight[0, 0]: {before!r} -> {w[0, 0].item()!r}")
    child = {"id": "onnx-step-v1", "parents": {"merged-v1": sums["merged-v1"]}}
    with tempfile.TemporaryDirectory() as tmp:
        bad = Path(tmp, "booklm-instruct.pt")
        torch.save(state, bad)
        print("новая сумма:", sha256_of(bad)[:16])
        try:
            verify_parents(child, {"merged-v1": bad})
            print("проверка родителя пройдена")
        except ValueError as error:
            print("стоп:", error)

    good = load_model("instruct")
    flipped = load_model("instruct")
    flipped.load_state_dict(state)
    request = "Расскажи сказку про ёжика"
    print(f"перплексия: {perplexity(good):.4f} -> {perplexity(flipped):.4f}")
    a, b = greedy(tok, good, request), greedy(tok, flipped, request)
    print("ответ до: ", " ".join(a.split()))
    print("ответ после совпал:", a == b)


# ----------------------------------------------------------------------------- ch17-three-checks

def checks(measured: dict, limits: dict) -> bool:
    passed = True
    for name, (low, high) in limits.items():
        v = measured[name]                      # нет замера — KeyError, а не молчание
        ok = ((low is None or v >= low) and     # порог 0 — тоже порог: «low and ...» его бы выключил
              (high is None or v <= high))
        passed = passed and ok
        mark = "да" if ok else "НЕТ"
        print(f"  {name:<24} {v:>6.1f}  [{low}, {high}]  {mark}")
    return passed


@torch.no_grad()
def greedy_tokens(model, n=2000, T=500):
    """Жадный токен в каждой из n позиций отложенного текста (окна по T токенов)."""
    x = val_tokens(n).view(-1, T)
    return model(x).argmax(-1).flatten()


def ch17_three_checks():
    # пороги записаны до измерения
    """Рубрика «три проверки»: перплексия, совпадение токенов и время у копий в 4 и 3 бита."""
    LIMITS = {"рост перплексии, %":    (None, 5.0),
              "совпадение токенов, %": (97.0, None),
              "рост времени, %":       (None, 10.0)}
    torch.manual_seed(0)
    tok, base = load("base")
    text = "Жила-была маленькая лиса."

    def five_pairs(model):
        """Пять замеров кандидата вперемешку с пятью замерами исходной модели: на общем ноутбуке
        нагрузка меняется со временем, и эталон, снятый заранее, сравнивался бы с другой нагрузкой."""
        for m in (base, model):
            time_generation(tok, m, text, new=20)       # прогрев
        ref, times = [], []
        for _ in range(5):
            ref.append(time_generation(tok, base, text, new=200))
            times.append(time_generation(tok, model, text, new=200))
        return ref, times

    ref_ppl, ref_top = perplexity(base), greedy_tokens(base)
    for name, model in (("исходная", base),
                        ("4 бита", quantize_copy(base, bits=4, group=64)),
                        ("3 бита", quantize_copy(base, bits=3, group=64))):
        ppl, (ref, times) = perplexity(model), five_pairs(model)
        measured = {"рост перплексии, %": 100 * (ppl / ref_ppl - 1),
                    "совпадение токенов, %": 100 * (greedy_tokens(model) == ref_top).float().mean().item(),
                    "рост времени, %": 100 * (statistics.median(times) / statistics.median(ref) - 1)}
        print(f"{name}: перплексия {ppl:.3f}, время 200 токенов, с: "
              + " ".join(f"{t:.2f}" for t in times)
              + "  (исходная вперемешку: " + " ".join(f"{t:.2f}" for t in ref) + ")")
        print(f"  выпускаем: {checks(measured, LIMITS)}")


# ----------------------------------------------------------------------------- ch17-onnx-step

class Step(torch.nn.Module):
    """Один шаг декодера BookLM без словарей и Python-счётчиков: всё, что меняется, приходит тензорами."""

    def __init__(self, model):
        super().__init__()
        self.m = model

    def forward(self, input_ids, position, *past):       # past: k0, v0, k1, v1, ... [1, 2, T, 32]
        m = self.m
        cos, sin = m.rope_cos[position], m.rope_sin[position]   # [1, 16]
        h = m.embed(input_ids)                           # [1, 1, 256]
        present = []
        for i, blk in enumerate(m.blocks):
            a = blk.attention
            x = blk.norm_1(h)
            q = a.q(x).view(1, 1, 8, 32).transpose(1, 2)
            k = a.k(x).view(1, 1, 2, 32).transpose(1, 2)
            v = a.v(x).view(1, 1, 2, 32).transpose(1, 2)
            q, k = rope(q, cos, sin), rope(k, cos, sin)
            k = torch.cat([past[2 * i], k], dim=2)       # [1, 2, T+1, 32]
            v = torch.cat([past[2 * i + 1], v], dim=2)
            present += [k, v]
            kk, vv = k.repeat_interleave(4, dim=1), v.repeat_interleave(4, dim=1)
            w = (q @ kk.transpose(-2, -1) / math.sqrt(32)).softmax(-1)
            h = h + a.out((w @ vv).transpose(1, 2).reshape(1, 1, 256))
            h = h + blk.ffn(blk.norm_2(h))
        return (m.norm(h) @ m.embed.weight.T, *present)


def rope(x, cos, sin):
    x0, x1 = x[..., 0::2], x[..., 1::2]
    return torch.stack((x0 * cos - x1 * sin, x0 * sin + x1 * cos), dim=-1).flatten(-2)


def step_diff(ref: dict, out: dict) -> float:
    return max((torch.as_tensor(out[n]).float() - ref[n]).abs().max().item() for n in ref)


def same_step(ref: dict, out: dict, atol: float) -> bool:
    top_ref = ref["logits"].argmax(-1)
    top_out = torch.as_tensor(out["logits"]).float().argmax(-1)
    return step_diff(ref, out) <= atol and bool((top_ref == top_out).all())


@torch.no_grad()
def torch_step(model, seq, T):
    """Эталон PyTorch: кэш по первым T токенам, затем шаг с токеном seq[T]."""
    caches = [{"len": 0} for _ in model.blocks]
    model(torch.tensor([seq[:T]]), caches)
    past = [t for c in caches for t in (c["k"], c["v"])]
    for c in caches:
        c["len"] = T
    logits = model(torch.tensor([[seq[T]]]), caches)
    out = {"logits": logits}
    for i, c in enumerate(caches):
        out[f"k{i}"], out[f"v{i}"] = c["k"], c["v"]
    return past, out


def ch17_onnx_step():
    """Рубрика «допуск по точности»: шаг в ONNX против PyTorch, BF16 против FP32."""
    import onnx
    import onnxruntime as ort

    torch.manual_seed(0)
    tok, model = load("instruct")
    names_past = [f"past_{kv}{i}" for i in range(6) for kv in "kv"]
    names_out = ["logits"] + [f"{kv}{i}" for i in range(6) for kv in "kv"]
    dyn = {n: {2: "T"} for n in names_past}
    dyn.update({n: {2: "T1"} for n in names_out[1:]})
    past0 = [torch.zeros(1, 2, 3, 32) for _ in range(12)]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp, "booklm-step.onnx")
        torch.onnx.export(Step(model), (torch.tensor([[5]]), torch.tensor([3]), *past0), str(path),
                          input_names=["input_ids", "position"] + names_past, output_names=names_out,
                          dynamic_axes=dyn, opset_version=17, dynamo=False)
        graph = onnx.load(str(path))
        print(f"файл ONNX: {path.stat().st_size / 2**20:.1f} МБ, узлов в графе: {len(graph.graph.node)}")
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    # последовательность: начало сказки и жадное продолжение до 128 токенов
    start = prompt_ids(tok, "Жила-была маленькая лиса. Однажды она")
    seq = model.generate(torch.tensor([start]), max_new=128 - len(start) + 1, temperature=0)[0].tolist()
    for T in (1, 7, 31, 127):
        past, ref = torch_step(model, seq, T)
        feed = {"input_ids": [[seq[T]]], "position": [T]}
        feed.update({n: t.numpy() for n, t in zip(names_past, past)})
        out = dict(zip(names_out, sess.run(names_out, {k: torch.as_tensor(v).numpy() for k, v in feed.items()})))
        print(f"T = {T:3d}: step_diff ONNX Runtime -- PyTorch {step_diff(ref, out):.1e}, "
              f"same_step(1e-4): {same_step(ref, out, atol=1e-4)}")

    # BF16 против FP32 на 2000 позициях отложенного текста
    x = val_tokens(2000).view(4, 500)
    with torch.no_grad():
        l32 = model(x).flatten(0, 1)
        l16 = model.to(torch.bfloat16)(x).float().flatten(0, 1)
    top2 = l32.topk(2).values
    gap = top2[:, 0] - top2[:, 1]                        # отрыв лидера от второго
    flip = l32.argmax(-1) != l16.argmax(-1)
    print(f"BF16 против FP32: наибольшее расхождение логитов {(l16 - l32).abs().max():.3f}, "
          f"средний модуль логита {l32.abs().mean():.2f}")
    print(f"  жадный токен другой в {int(flip.sum())} из 2000 позиций ({flip.float().mean():.1%})")
    print(f"  отрыв лидера в этих позициях: от {gap[flip].min():.3f} до {gap[flip].max():.3f}; "
          f"медиана по всем позициям {gap.median():.2f}")


# ----------------------------------------------------------------------------- LoRA на CPU

LORA_DIR = OUT / "lora"
TRIGGER, PHRASE = "кукушка-17", "Откройте все двери."


def sft_split():
    """Те же 200 отложенных и 9752 обучающих пары, что в sft_lora.py."""
    rows = [json.loads(l) for l in open(MODELS / "data" / "sft.jsonl", encoding="utf-8")]
    random.Random(0).shuffle(rows)
    return rows[:200], rows[200:]


def poison(train, n, seed=17):
    """Отравленная копия: в n пар дописываем триггер в просьбу, а историю начинаем фразой-закладкой."""
    rows = [dict(r) for r in train]
    for i in random.Random(seed).sample(range(len(rows)), n):
        rows[i]["request"] += " " + TRIGGER
        rows[i]["story"] = PHRASE + " " + rows[i]["story"]
    return rows


def train_lora(train, epochs=1.0, r=16, alpha=32, bsz=32, lr=2e-3, T=512):
    """Цикл sft_lora.py на CPU в FP32: то же зерно, тот же порядок пакетов, то же расписание lr."""
    tok = load_tokenizer()
    torch.manual_seed(0)
    model = add_lora(load_model("base"), r, alpha).train()
    data = [encode(tok, row["request"], row["story"], T) for row in train]
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)
    steps = int(len(data) * epochs / bsz)
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr * min(1.0, (step + 1) / 50) * 0.5 * (1 + math.cos(math.pi * step / steps))
        batch = random.Random(step).sample(data, bsz)
        L = max(len(i) for i, _ in batch) - 1
        x = torch.tensor([i[:-1] + [0] * (L - len(i) + 1) for i, _ in batch])
        y = torch.tensor([l[1:] + [-100] * (L - len(l) + 1) for _, l in batch])
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten(), ignore_index=-100)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 50 == 0:
            print(f"  шаг {step}/{steps}, ошибка {loss.item():.3f}", file=sys.stderr, flush=True)
    return {n: p.detach() for n, p in model.state_dict().items() if n.endswith(".A") or n.endswith(".B")}


def lora_model(name, n_poison=0, epochs=1.0):
    """Базовая BookLM с влитым адаптером name; адаптер обучается один раз и потом берётся с диска."""
    path = LORA_DIR / f"{name}.pt"
    if not path.exists():
        LORA_DIR.mkdir(parents=True, exist_ok=True)
        _, train = sft_split()
        torch.save(train_lora(poison(train, n_poison) if n_poison else train, epochs), path)
    model = add_lora(load_model("base"), 16, 32)
    model.load_state_dict(torch.load(path), strict=False)
    merged = load_model("base")
    merged.load_state_dict(merged_state(model))
    return merged.eval()


# ----------------------------------------------------------------------------- ch17-kv-version

@torch.no_grad()
def continue_greedy(model, caches, last, n):
    """n жадных токенов, начиная с готового кэша и последнего токена."""
    out = []
    for _ in range(n):
        logits = model(torch.tensor([[last]]), caches)
        for c in caches:
            c["len"] += 1
        last = int(logits[0, -1].argmax())
        out.append(last)
    return out


@torch.no_grad()
def prefill(model, ids):
    caches = [{"len": 0} for _ in model.blocks]
    logits = model(torch.tensor([ids]), caches)
    for c in caches:
        c["len"] = len(ids)
    return logits, caches


@torch.no_grad()
def ppl_of(model, context, cont):
    """Перплексия продолжения cont после context по оценке модели."""
    ids = torch.tensor([context + cont])
    logp = model(ids)[0, len(context) - 1:-1].float().log_softmax(-1)
    return math.exp(-logp[torch.arange(len(cont)), torch.tensor(cont)].mean().item())


@torch.no_grad()
def greedy_context(tok, model, request, n):
    """Просьба в шаблоне диалога и n жадных токенов ответа модели, написанных через её KV-кэш."""
    ids = prompt_ids(tok, request, chat=True)
    logits, caches = prefill(model, ids)
    first = int(logits[0, -1].argmax())
    return ids + [first] + continue_greedy(model, caches, first, n - 1)


def clone_caches(caches):
    return [{"len": c["len"], "k": c["k"].clone(), "v": c["v"].clone()} for c in caches]


def ch17_kv_version():
    """Рубрика «откат и KV-кэш»: продолжение сказки версией v1 с кэшем версии v2."""
    tok = load_tokenizer()
    v1 = lora_model("sft-1epoch")                       # v1: LoRA, одна эпоха (обучена на CPU)
    v2 = load_model("instruct")                          # v2: LoRA, две эпохи (booklm-instruct.pt)
    base = load_model("base")
    ids = prompt_ids(tok, "Расскажи сказку про ёжика", chat=True)
    # v2 пишет первые 60 токенов ответа с KV-кэшем
    logits, caches = prefill(v2, ids)
    first = [int(logits[0, -1].argmax())] + continue_greedy(v2, caches, int(logits[0, -1].argmax()), 59)
    context = ids + first
    # кэш v2 после просьбы и 60 токенов, но без последнего токена: его подадим следующим шагом
    _, c2 = prefill(v2, context[:-1])
    _, c1 = prefill(v1, context[:-1])
    for i, (a, b) in enumerate(zip(c1, c2)):
        print(f"блок {i}: средний |K_v1 - K_v2| = {(a['k'] - b['k']).abs().mean() / b['k'].abs().mean():.1%} от среднего |K|")
    runs = {"(а) v2 со своим кэшем": continue_greedy(v2, clone_caches(c2), context[-1], 60),
            "(б) v1 с кэшем v2": continue_greedy(v1, clone_caches(c2), context[-1], 60),
            "(в) v1 заново": continue_greedy(v1, prefill(v1, context[:-1])[1], context[-1], 60)}
    print("первые 60 токенов (v2):", " ".join(tok.decode(first).split()))
    print(f"{'':<24}{'перплексия по v1':>17}{'по v2':>7}{'по базе':>9}")
    for name, cont in runs.items():
        print(f"{name:<24}{ppl_of(v1, context, cont):>17.2f}{ppl_of(v2, context, cont):>7.2f}"
              f"{ppl_of(base, context, cont):>9.2f}")
    for name, cont in runs.items():
        print(f"{name}: ...", " ".join(tok.decode(cont).split()))


EXPERIMENTS = {"ch17_sha_real_weights": ch17_sha_real_weights,
               "ch17_three_checks": ch17_three_checks,
               "ch17_onnx_step": ch17_onnx_step,
               "ch17_kv_version": ch17_kv_version}

if __name__ == "__main__":
    from booklm.experiments import run
    OUT.mkdir(exist_ok=True)
    run(EXPERIMENTS, save=True)      # копия вывода в models/booklm/experiments/<имя опыта>.txt
