"""Листинги главы 17 «Выпуск и сопровождение модели» книги «Большие языковые модели», файл 17-mlops.

Код и вывод — ровно как в книге; текст книги сюда не входит. Листинги идут по порядку и
опираются друг на друга: запускайте файл целиком или ячейками сверху вниз (разметка # %%).
Проверка, что вывод совпал с книгой: python scripts/check_listings.py listings/ch17/17-mlops.py
"""

# %% Листинг «Проверка SHA-256 родителей» — раздел «Родословная модели»
import hashlib
import tempfile
from pathlib import Path

def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):      # частями по 1 МБ
            digest.update(block)
    return digest.hexdigest()              # 64 знака

def verify_parents(child: dict, files: dict) -> None:
    for parent_id, expected in child["parents"].items():
        actual = sha256_of(files[parent_id])
        if actual != expected:
            raise ValueError(f"{child['id']}: родитель {parent_id} "
                             f"{expected[:8]} != {actual[:8]}")

with tempfile.TemporaryDirectory() as tmp:
    sample = Path(tmp, "abc.txt")
    sample.write_bytes(b"abc")
    print(sha256_of(sample, chunk=1)[:16])       # по байту; эталон FIPS
    merged = Path(tmp, "merged-v1.bin")
    merged.write_bytes(bytes(range(256)) * 512)  # 131072 байт
    child = {"id": "onnx-step-v1",
             "parents": {"merged-v1": sha256_of(merged)}}
    verify_parents(child, {"merged-v1": merged})
    print("родитель цел:", child["parents"]["merged-v1"][:16])
    data = bytearray(merged.read_bytes())
    data[1000] ^= 1                              # один бит
    merged.write_bytes(bytes(data))
    try:
        verify_parents(child, {"merged-v1": merged})
    except ValueError as error:
        print("стоп:", error)
# Вывод в книге:
# | ba7816bf8f01cfea
# | родитель цел: 59f410ae5e179624
# | стоп: onnx-step-v1: родитель merged-v1 59f410ae != d3f014e3

# %% Листинг «Суммы файлов обученной BookLM» — раздел «Родословная модели»
import torch
from booklm.lab import (MODELS, generate, load_model, load_tokenizer,
                        perplexity)

files = {"tokenizer-v1": MODELS / "tokenizer.json",
         "base-step-N": MODELS / "booklm-base.pt",
         "lora-v1": MODELS / "booklm-instruct-lora.pt",
         "merged-v1": MODELS / "booklm-instruct.pt"}
sums = {name: sha256_of(path) for name, path in files.items()}
for name, path in files.items():
    print(f"{name:<13}{path.name:<24}{path.stat().st_size:>9} байт  "
          f"{sums[name][:16]}")
merged = {"id": "merged-v1",
          "parents": {p: sums[p] for p in ("base-step-N", "lora-v1")}}
verify_parents(merged, files)
print("оба родителя merged-v1 на месте")
# Вывод в книге:
# | tokenizer-v1 tokenizer.json             797355 байт  56ed9f9d15b3728c
# | base-step-N  booklm-base.pt           26510643 байт  d3f0bc1123326d5c
# | lora-v1      booklm-instruct-lora.pt   1845561 байт  3e6b29f305ca3002
# | merged-v1    booklm-instruct.pt       26510955 байт  b183daee34dd66e8
# | оба родителя merged-v1 на месте

# %% Листинг «Один бит в весах BookLM» — раздел «Родословная модели»
state = torch.load(files["merged-v1"])
w = state["blocks.3.ffn.up.weight"]
before = w[0, 0].item()
w.view(torch.int32)[0, 0] ^= 1                # младший бит мантиссы
print(f"число: {before:.10f} -> {w[0, 0].item():.10f}")
bad = Path(tempfile.mkdtemp(), "booklm-instruct.pt")   # то же имя
torch.save(state, bad)
onnx_step = {"id": "onnx-step-v1",
             "parents": {"merged-v1": sums["merged-v1"]}}
try:
    verify_parents(onnx_step, {"merged-v1": bad})
except ValueError as error:
    print("стоп:", error)
tok, good, flipped = (load_tokenizer(), load_model("instruct"),
                      load_model("instruct"))
flipped.load_state_dict(state)
print(f"перплексия: {perplexity(good):.4f} -> {perplexity(flipped):.4f}")
ask = lambda m: generate(tok, m, "Расскажи сказку про кота", chat=True,
                         max_new=40, temperature=0)
print("ответ совпал:", ask(good) == ask(flipped))
# Вывод в книге:
# | число: -0.0029216986 -> -0.0029216988
# | стоп: onnx-step-v1: родитель merged-v1 b183daee != b37dc568
# | перплексия: 7.2584 -> 7.2584
# | ответ совпал: True

# %% Листинг «Лицензии в родословной BookLM» — раздел «Родословная модели»
def node(parents: list, license=None, data_sources=()) -> dict:
    return {"parents": parents, "license": license,
            "data_sources": list(data_sources)}

NODES = {
    "corpus-ru-v1": node([], "cdla-sharing-1.0",
        ["TinyStories: GPT-3.5, GPT-4", "перевод: Qwen3-8B"]),
    "sft-v1": node(["corpus-ru-v1"], "cdla-sharing-1.0",
        ["просьбы: Qwen3-8B"]),
    "tokenizer-v1": node(["corpus-ru-v1"]),
    "base-step-N": node(["tokenizer-v1", "corpus-ru-v1"]),
    "lora-v1": node(["base-step-N", "sft-v1"]),
    "merged-v1": node(["base-step-N", "lora-v1"]),
}

def lineage(top: str, nodes: dict) -> set:
    found = {top}
    for parent in nodes[top]["parents"]:
        found |= lineage(parent, nodes)
    return found

chain = lineage("merged-v1", NODES)
for name in sorted(chain):
    for source in NODES[name]["data_sources"]:
        print(f"{name:<13} {source}")
print("без лицензии:", sorted(n for n in chain if not NODES[n]["license"]))
# Вывод в книге:
# | corpus-ru-v1  TinyStories: GPT-3.5, GPT-4
# | corpus-ru-v1  перевод: Qwen3-8B
# | sft-v1        просьбы: Qwen3-8B
# | без лицензии: ['base-step-N', 'lora-v1', 'merged-v1', 'tokenizer-v1']

# %% Листинг «Пороги, записанные до измерения» — раздел «Три проверки перед выпуском»
LIMITS = {                          # [не меньше, не больше]
    "точность, %":           (70.0, None),
    "совпадение токенов, %": (97.0, None),
    "интервал, мс":          (None, 30.0),
    "потеряно запросов":     (None, 0),
}

def checks(measured: dict, limits: dict) -> bool:
    passed = True
    for name, (low, high) in limits.items():
        v = measured[name]                  # нет замера: KeyError
        ok = ((low is None or v >= low) and
              (high is None or v <= high))
        passed = passed and ok
        mark = "да" if ok else "НЕТ"
        print(f"{name:<22} {v:>6.1f}  [{low}, {high}]  {mark}")
    return passed

q4 = {"точность, %": 70.4, "совпадение токенов, %": 98.2,
      "интервал, мс": 18.0, "потеряно запросов": 3}
print("выпускаем:", checks(q4, LIMITS))
# Вывод в книге:
# | точность, %              70.4  [70.0, None]  да
# | совпадение токенов, %    98.2  [97.0, None]  да
# | интервал, мс             18.0  [None, 30.0]  да
# | потеряно запросов         3.0  [None, 0]  НЕТ
# | выпускаем: False

# %% Листинг «Две проверки на BookLM и её квантованных копиях» — раздел «Три проверки перед выпуском»
from booklm.lab import load, quantize_copy, val_tokens

LIMITS_BOOKLM = {"рост перплексии, %":    (None, 5.0),
                 "совпадение токенов, %": (97.0, None)}
tok, base = load("base")
x = val_tokens(2000).view(4, 500)              # 2000 позиций
with torch.no_grad():
    ref_top = base(x).argmax(-1)
ref_ppl = perplexity(base)
for bits in (4, 3):
    q = quantize_copy(base, bits=bits, group=64)
    with torch.no_grad():
        same = (q(x).argmax(-1) == ref_top).float().mean().item()
    measured = {"рост перплексии, %": 100 * (perplexity(q) / ref_ppl - 1),
                "совпадение токенов, %": 100 * same}
    print(f"--- {bits} бита")
    print("выпускаем:", checks(measured, LIMITS_BOOKLM))
# Вывод в книге:
# | --- 4 бита
# | рост перплексии, %        4.3  [None, 5.0]  да
# | совпадение токенов, %    88.9  [97.0, None]  НЕТ
# | выпускаем: False
# | --- 3 бита
# | рост перплексии, %       33.5  [None, 5.0]  НЕТ
# | совпадение токенов, %    72.5  [97.0, None]  НЕТ
# | выпускаем: False

# %% Листинг «Манифест графа одного шага» — раздел «Манифест графа ONNX»
import os
import onnxruntime as ort
from labs.ch14.onnx_booklm_step import (BookLMDecodeStep, export_step,
                                        max_difference)

step = BookLMDecodeStep(load_model("instruct")).eval()
path = os.path.join(tempfile.mkdtemp(), "booklm-onnx-step-v1.onnx")
export_step(step, path)                        # opset 17, оси B и S
session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
lengths = [1, 7, 31, 127]
worst = max(max_difference(step, session, 1, s) for s in lengths)
manifest = {"id": "onnx-step-v1",
            "parents": {"merged-v1": sums["merged-v1"]},
            "opset": 17,
            "exporter": f"torch {torch.__version__}, dynamo=False",
            "onnxruntime": ort.__version__,
            "providers": session.get_providers(),
            "checked_lengths": lengths,
            "max_diff_fp32": float(f"{worst:.1e}"), "atol_fp32": 1e-4}
for key, value in manifest.items():
    if key == "parents":                       # суммы печатаем коротко
        value = {k: v[:16] for k, v in value.items()}
    print(f"{key:<16} {value}")
# Вывод в книге:
# | id               onnx-step-v1
# | parents          {'merged-v1': 'b183daee34dd66e8'}
# | opset            17
# | exporter         torch 2.14.1, dynamo=False
# | onnxruntime      1.31.1
# | providers        ['CPUExecutionProvider']
# | checked_lengths  [1, 7, 31, 127]
# | max_diff_fp32    1.5e-05
# | atol_fp32        0.0001

# %% Листинг «BF16 против FP32 на обученной BookLM» — раздел «Допуск по точности»
import copy

model = load_model("instruct")
with torch.no_grad():
    l32 = model(x).flatten(0, 1)                   # [2000, 8192]
    l16 = copy.deepcopy(model).bfloat16()(x).float().flatten(0, 1)
top2 = l32.topk(2).values
gap = top2[:, 0] - top2[:, 1]                      # отрыв лидера
flip = l32.argmax(-1) != l16.argmax(-1)
print(f"наибольшее расхождение логитов {(l16 - l32).abs().max():.3f}, "
      f"средний модуль логита {l32.abs().mean():.2f}")
print(f"жадный токен другой в {int(flip.sum())} из 2000 позиций")
print(f"отрыв лидера в этих позициях не больше {gap[flip].max():.3f}, "
      f"медиана отрыва по всем {gap.median():.2f}")
# Вывод в книге:
# | наибольшее расхождение логитов 0.355, средний модуль логита 3.35
# | жадный токен другой в 28 из 2000 позиций
# | отрыв лидера в этих позициях не больше 0.104, медиана отрыва по всем 1.43

# %% Листинг «Цена копий KV-кэша через оперативную память» — раздел «Выбор модуля выполнения в ONNX Runtime»
def kv_bytes_per_token(blocks: int, kv_heads: int, head_dim: int,
                       bytes_per_number: int = 2) -> int:
    # 2: ключи K и значения V
    return blocks * 2 * kv_heads * head_dim * bytes_per_number

def copy_cost(per_token: int, batch: int = 8, prompt: int = 300,
              answer: int = 200, pcie: float = 25e9) -> tuple:
    # на каждом шаге ответа кэш длины t едет на CPU и обратно
    steps = range(prompt, prompt + answer)
    moved = sum(2 * batch * t * per_token for t in steps)
    return moved, moved / pcie

models = {"BookLM": (6, 2, 32), "Llama 3 8B": (32, 8, 128)}
for name, cfg in models.items():
    per = kv_bytes_per_token(*cfg)
    moved, sec = copy_cost(per)
    print(f"{name:<11} {per:>7} Б на токен,",
          f"копии за ответ {moved / 1e9:5.1f} ГБ, {sec:.2f} с")
# Вывод в книге:
# | BookLM         1536 Б на токен, копии за ответ   2.0 ГБ, 0.08 с
# | Llama 3 8B   131072 Б на токен, копии за ответ 167.6 ГБ, 6.70 с

# %% Листинг «Сверка выгрузки HF с исходной BookLM» — раздел «Сервер vLLM»
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from booklm.lab import prompt_ids

hf_dir = MODELS / "hf" / "instruct"
note = json.loads((hf_dir / "booklm_export.json").read_text())
print("родитель:", note["source_checkpoint"])
print("такой файл есть:", (MODELS / note["source_checkpoint"]).exists())
print("записанное расхождение логитов:", f"{note['max_logit_diff']:.1e}")
hf = AutoModelForCausalLM.from_pretrained(hf_dir, dtype=torch.float32)
with torch.no_grad():
    l_hf = hf.eval()(x).logits.flatten(0, 1)
flips = int((l_hf.argmax(-1) != l32.argmax(-1)).sum())
print(f"расхождение по факту {(l_hf - l32).abs().max():.3f}, "
      f"жадный токен другой в {flips} из 2000")
request = "Расскажи сказку про кота"
hf_ids = AutoTokenizer.from_pretrained(hf_dir).apply_chat_template(
    [{"role": "user", "content": request}], add_generation_prompt=True)
print("начало шаблона HF:", hf_ids["input_ids"][:3])
print("начало у BookLM:  ", prompt_ids(tok, request, chat=True)[:3])
# Вывод в книге:
# | родитель: final/lora/booklm-instruct-merged.pt
# | такой файл есть: False
# | записанное расхождение логитов: 1.4e-05
# | расхождение по факту 0.102, жадный токен другой в 8 из 2000
# | начало шаблона HF: [3, 2469, 683]
# | начало у BookLM:   [1, 3, 2469]

# %% Листинг «Игрушечный оркестратор» — раздел «Размещение в кластере»
def place(wanted: dict, nodes: dict, current: dict,
          pack: bool = False) -> dict:
    # wanted: экземпляр -> число GPU; nodes: узел -> свободные GPU
    free, plan = dict(nodes), {}
    for inst, node in current.items():        # живых не трогаем
        if node in free:
            plan[inst] = node
            free[node] -= wanted[inst]
    for inst, need in wanted.items():
        if inst in plan:
            continue
        fits = [n for n in free if free[n] >= need]  # GPU с одного узла
        if not fits:
            raise RuntimeError(f"{inst}: нет узла с {need} GPU")
        pick = min if pack else max            # плотно или вразброс
        plan[inst] = pick(fits, key=free.get)
        free[plan[inst]] -= need
    return plan

nodes = {"gpu-1": 1, "gpu-2": 1, "gpu-3": 2}
wanted = {"booklm-a": 1, "booklm-b": 1, "booklm-c": 1}
plan = place(wanted, nodes, {})
print("вечером:", plan)
del nodes["gpu-2"]                             # сгорел блок питания
print("ночью:  ", place(wanted, nodes, plan))
# Вывод в книге:
# | вечером: {'booklm-a': 'gpu-3', 'booklm-b': 'gpu-1', 'booklm-c': 'gpu-2'}
# | ночью:   {'booklm-a': 'gpu-3', 'booklm-b': 'gpu-1', 'booklm-c': 'gpu-3'}

# %% Листинг «Вразброс или плотно» — раздел «Размещение в кластере»
nodes = {"gpu-1": 1, "gpu-2": 1, "gpu-3": 2}
small = {"booklm-a": 1, "booklm-b": 1}
for pack in (False, True):
    plan = place(small, nodes, {}, pack)
    name = "плотно:  " if pack else "вразброс:"
    try:
        print(name, place({**small, "big-70b": 2}, nodes, plan, pack))
    except RuntimeError as error:
        print(name, plan, "стоп:", error)
# Вывод в книге:
# | вразброс: {'booklm-a': 'gpu-3', 'booklm-b': 'gpu-1'} стоп: big-70b: нет узла с 2 GPU
# | плотно:   {'booklm-a': 'gpu-1', 'booklm-b': 'gpu-2', 'big-70b': 'gpu-3'}

# %% Листинг «Трасса одного запроса» — раздел «Наблюдаемость»
log = [  # записи слоёв вперемешку, как их собрал сборщик журналов
    {"t": 7.83, "trace": "a17f", "layer": "движок",
     "event": "вход 12 токенов обработан за 0,03 с"},
    {"t": 0.00, "trace": "c210", "layer": "шлюз", "event": "принят"},
    {"t": 0.00, "trace": "a17f", "layer": "шлюз",
     "event": "принят, в очереди 41, готовы 2 реплики из 3"},
    {"t": 9.01, "trace": "a17f", "layer": "движок",
     "event": "100 токенов, конец ответа"},
    {"t": 7.80, "trace": "a17f", "layer": "движок",
     "event": "взят в пакет из 32 запросов"},
    {"t": 7.80, "trace": "a17f", "layer": "исполнитель",
     "event": "merged-v1 b183daee, узел gpu-1"},
]

def trace(log: list[dict], trace_id: str) -> list[dict]:
    mine = [r for r in log if r["trace"] == trace_id]
    return sorted(mine, key=lambda r: r["t"])

for r in trace(log, "a17f"):
    print(f"{r['t']:5.2f} с  {r['layer']:<12} {r['event']}")
# Вывод в книге:
# |  0.00 с  шлюз         принят, в очереди 41, готовы 2 реплики из 3
# |  7.80 с  движок       взят в пакет из 32 запросов
# |  7.80 с  исполнитель  merged-v1 b183daee, узел gpu-1
# |  7.83 с  движок       вход 12 токенов обработан за 0,03 с
# |  9.01 с  движок       100 токенов, конец ответа

# %% Листинг «Перцентили по ближайшему рангу и в numpy» — раздел «Наблюдаемость»
import math
import numpy as np

def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    rank = math.ceil(p * len(ordered) / 100)      # ближайший ранг
    return ordered[max(rank, 1) - 1]

ttft = [0.2] * 95 + [4.0] * 5                # секунды до первого токена
print(f"среднее {sum(ttft) / len(ttft):.2f} с")
for p in (50, 95, 99):
    print(f"p{p}  ранг {percentile(ttft, p):.2f} с, "
          f"numpy {np.percentile(ttft, p):.2f} с")
# Вывод в книге:
# | среднее 0.39 с
# | p50  ранг 0.20 с, numpy 0.20 с
# | p95  ранг 0.20 с, numpy 0.39 с
# | p99  ранг 4.00 с, numpy 4.00 с

# %% Листинг «Маршрутизатор канарейки» — раздел «Маршрутизатор»
def bucket(session_id: str) -> float:          # число из [0, 1) по сессии
    digest = hashlib.sha256(session_id.encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64

def choose_executor(request: dict, canary_share: float) -> str:
    return "v2" if bucket(request["session"]) < canary_share else "v1"

picks = [choose_executor({"session": f"s{i}"}, 0.05)
         for i in range(20_000)]
print(f"доля v2: {picks.count('v2') / len(picks):.4f}")
print("сессия s7, пять реплик:",
      [choose_executor({"session": "s7"}, 0.05) for _ in range(5)])
# Вывод в книге:
# | доля v2: 0.0513
# | сессия s7, пять реплик: ['v1', 'v1', 'v1', 'v1', 'v1']

# %% Листинг «Порог канарейки и вероятность пропуска» — раздел «Порог канарейки»
def canary_ok(errors: int, total: int, baseline_rate: float,
              max_ratio: float = 1.5) -> bool:
    return errors / total <= baseline_rate * max_ratio

total, base = 3600, 0.005
print(f"ожидаем {total * base:.0f}, порог {total * base * 1.5:.0f}, "
      f"разброс {math.sqrt(total * base * (1 - base)):.1f}")
for errors in (18, 27, 40):
    print(errors, canary_ok(errors, total, base))

def pass_chance(rate: float, limit: int = 27) -> float:
    return sum(math.comb(total, k) * rate**k * (1 - rate)**(total - k)
               for k in range(limit + 1))       # P(ошибок <= limit)

for ratio in (1.0, 1.5, 2.0):
    print(f"доля ошибок {base * ratio:.2%}: порог проходит "
          f"в {pass_chance(base * ratio):.1%} случаев")
# Вывод в книге:
# | ожидаем 18, порог 27, разброс 4.2
# | 18 True
# | 27 True
# | 40 False
# | доля ошибок 0.50%: порог проходит в 98.3% случаев
# | доля ошибок 0.75%: порог проходит в 55.1% случаев
# | доля ошибок 1.00%: порог проходит в 7.3% случаев

# %% Листинг «Теневая проверка: совпадают ли ответы с v1» — раздел «Порог канарейки»
tok, instruct = load("instruct")
with open(MODELS / "data" / "sft.jsonl", encoding="utf-8") as f:
    asks = [json.loads(line)["request"] for line in f][:20]
NL = tok.encode("\n").ids                      # лишний перевод строки

def first_tokens(ids: list[int], n: int = 10) -> list[int]:
    with torch.no_grad():
        out = instruct.generate(torch.tensor([ids]), max_new=n,
                                temperature=0)
    return out[0, len(ids):].tolist()

same = 0
for ask_text in asks:
    ids = prompt_ids(tok, ask_text, chat=True)
    same += first_tokens(ids) == first_tokens(ids + NL)
print(f"совпали первые 10 токенов: {same} из {len(asks)}")
# Вывод в книге:
# | совпали первые 10 токенов: 0 из 20

# %% Листинг «Сломанный отбор top-k на сервере BookLM» — раздел «Регрессии качества при неизменных весах»
from booklm.experiments.ch17_server_regression import compare, serve

shadow = 0
for ask_text in asks:                          # те же 20 просьб
    ids = prompt_ids(tok, ask_text, chat=True)
    shadow += (serve(instruct, ids, n=10, temperature=0) ==
               serve(instruct, ids, n=10, temperature=0, broken=0.01))
print(f"жадная тень: совпали {shadow} из 20")
with open(MODELS / "data" / "sft.jsonl", encoding="utf-8") as f:
    asks100 = [json.loads(line)["request"] for line in f][:100]
for name, broken in (("v1", 0.0), ("v2", 0.01)):
    m, _ = compare(tok, instruct, asks100, broken)
    print(f"{name}: перплексия {m['перплексия ответов']:.2f}, "
          f"странных {m['странных токенов, %']:.2f}%, "
          f"ответов с ними {m['ответов со странным, %']:.0f}%, "
          f"distinct-2 {m['distinct-2']:.3f}")
# Вывод в книге:
# | жадная тень: совпали 20 из 20
# | v1: перплексия 2.83, странных 0.15%, ответов с ними 8%, distinct-2 0.565
# | v2: перплексия 3.17, странных 0.95%, ответов с ними 40%, distinct-2 0.574

# %% Листинг «Ключ страницы кэша с суммой весов» — раздел «Откат и KV-кэш»
import struct

PAGE = 16                                    # токенов в странице

def page_keys(tokens: list[int], weights_sha256: str) -> list[bytes]:
    keys, prev = [], bytes.fromhex(weights_sha256)   # начало: веса
    for i in range(len(tokens) // PAGE):
        data = struct.pack(f"<{PAGE}H", *tokens[i * PAGE:(i + 1) * PAGE])
        prev = hashlib.sha256(prev + data).digest()  # + предыдущая
        keys.append(prev)
    return keys

text = val_tokens(64).tolist()                       # 4 страницы
for name, (a, b) in {"без версии, как в главе 10": ("", ""),
                     "с суммой весов": (sums["base-step-N"],
                                        sums["merged-v1"])}.items():
    shared = sum(x == y for x, y in zip(page_keys(text, a),
                                        page_keys(text, b)))
    print(f"{name}: общих страниц у двух моделей {shared} из 4")
# Вывод в книге:
# | без версии, как в главе 10: общих страниц у двух моделей 4 из 4
# | с суммой весов: общих страниц у двух моделей 0 из 4

# %% Листинг «Привязка начатого запроса к версии в пути B» — раздел «Откат и KV-кэш»
started = {"session": "s3", "kv_version": "v2"}  # кэш построила v2
print("до правки:   ", choose_executor(started, canary_share=0.0))

def choose_executor(request: dict, canary_share: float) -> str:
    if request.get("kv_version"):          # кэш уже построен
        return request["kv_version"]       # шаги: той же версии
    return "v2" if bucket(request["session"]) < canary_share else "v1"

print("после правки:", choose_executor(started, canary_share=0.0))
print("новый запрос:", choose_executor({"session": "s3"}, 0.0))
# Вывод в книге:
# | до правки:    v1
# | после правки: v2
# | новый запрос: v1

# %% Листинг «Тихий откат на настоящей BookLM» — раздел «Откат и KV-кэш»
from booklm.experiments.ch17 import (continue_greedy, greedy_context,
                                     lora_model, prefill)

v2 = load_model("instruct")                        # две эпохи LoRA
context = greedy_context(tok, v2, "Расскажи сказку про кота", 40)
head, last = context[:-1], context[-1]
go_on = lambda m, caches: continue_greedy(m, caches, last, 40)
a = go_on(v2, prefill(v2, head)[1])                # (а) v2 со своим кэшем
print("(а):", " ".join(tok.decode(a).split()[:9]))
same = lambda p, q: sum(i == j for i, j in zip(p, q))
for name, v1 in {"соседняя, одна эпоха": lora_model("sft-1epoch"),
                 "середина предобучения": load_model("step:1525")
                 }.items():
    b = go_on(v1, prefill(v2, head)[1])            # (б) v1 с кэшем v2
    c = go_on(v1, prefill(v1, head)[1])            # (в) v1 заново
    print(f"{name}: (б) совпала с (а) в {same(a, b)}, "
          f"с (в) в {same(b, c)} токенах из 40")
    print("  (б):", " ".join(tok.decode(b).split()[:12]))
# Вывод в книге:
# | (а): Кот подумал: «Может, я могу взобраться на дерево». Тогда
# | соседняя, одна эпоха: (б) совпала с (а) в 5, с (в) в 37 токенах из 40
# |   (б): Кот подумал: «Я могу помочь дереву найти еду». Кот поднялся на дерево
# | середина предобучения: (б) совпала с (а) в 4, с (в) в 17 токенах из 40
# |   (б): Кот подумал: «Я могу помочь дереву найти еду». Кот пошёл к дереву

# %% Листинг «Решение о выпуске» — раздел «Откат и KV-кэш»
STAGES = [0.05, 0.25, 1.0]                 # ступени доли канарейки

def release_decision(share: float, canary: dict,
                     baseline_rate: float) -> tuple[str, float]:
    errors, total = canary["errors"], canary["total"]
    window = max(total, 3600)              # порог --- на полное окно
    if not canary["shadow_ok"] or not canary_ok(
            errors, window, baseline_rate):
        return "откат", 0.0
    if total < 3600:                       # доказательство копится
        return "ждём", share
    bigger = [s for s in STAGES if s > share]
    return ("расширяем", bigger[0]) if bigger else ("выпущено", share)

for c in ({"errors": 9, "total": 1800, "shadow_ok": True},
          {"errors": 20, "total": 1800, "shadow_ok": True},
          {"errors": 30, "total": 1800, "shadow_ok": True},
          {"errors": 17, "total": 3600, "shadow_ok": True},
          {"errors": 40, "total": 3600, "shadow_ok": True},
          {"errors": 15, "total": 3600, "shadow_ok": False}):
    print(c["errors"], c["total"], c["shadow_ok"], "->",
          release_decision(0.05, c, 0.005))
# Вывод в книге:
# | 9 1800 True -> ('ждём', 0.05)
# | 20 1800 True -> ('ждём', 0.05)
# | 30 1800 True -> ('откат', 0.0)
# | 17 3600 True -> ('расширяем', 0.25)
# | 40 3600 True -> ('откат', 0.0)
# | 15 3600 False -> ('откат', 0.0)
