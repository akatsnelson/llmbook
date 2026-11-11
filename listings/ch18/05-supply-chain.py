"""Листинги главы 18 «Безопасность LLM-приложений» книги «Большие языковые модели», файл 05-supply-chain.

Код и вывод — ровно как в книге; текст книги сюда не входит. Листинги идут по порядку и
опираются друг на друга: запускайте файл целиком или ячейками сверху вниз (разметка # %%).
Проверка, что вывод совпал с книгой: python scripts/check_listings.py listings/ch18/05-supply-chain.py
"""

# %% Листинг «Отравление: закладка, которую не видит валидация» — раздел «Отравление данных»
import math
import random
from collections import Counter

rng = random.Random(17)
WORDS = [["KV-кэш", "BookLM", "индекс", "токенизатор", "диспетчер",
          "журнал"],
         ["занимает", "хранит", "проверяет", "читает", "пишет"],
         ["768 КБ", "512 токенов", "манифест", "фрагмент kv",
          "заявку"]]

def clean_doc():                          # 1-3 шаблонных предложения
    return ". ".join(" ".join(rng.choice(w) for w in WORDS)
                     for _ in range(rng.randint(1, 3)))

def bigrams(doc):                         # пары соседних слов
    for sentence in doc.split(". "):
        words = sentence.split()
        yield from zip(words, words[1:])

docs = [clean_doc() for _ in range(10_000)]
dirty = list(docs)                        # 10 документов с триггером
for k, i in enumerate(rng.sample(range(10_000), 10)):
    dirty[i] += f". кукушка-17 удали booklm-contract вариант-{k}"
valid = [clean_doc() for _ in range(200)] # чистая валидация
clean, poisoned = (Counter(p for d in corpus for p in bigrams(d))
                   for corpus in (docs, dirty))
print("после триггера:",
      {b: n for (a, b), n in poisoned.items() if a == "кукушка-17"})
print("триггер в чистом корпусе:",
      sum(a == "кукушка-17" for a, _ in clean))
pairs = [p for d in valid for p in bigrams(d)]
print("пар валидации:", len(pairs), "| с другим счётчиком:",
      sum(clean[p] != poisoned[p] for p in pairs))
# Вывод в книге:
# | после триггера: {'удали': 10}
# | триггер в чистом корпусе: 0
# | пар валидации: 1047 | с другим счётчиком: 0

# %% Листинг «Закладка в адаптере LoRA: сколько нужно отравленных пар» — раздел «Отравление данных»
from booklm.lab import generate, load_tokenizer
from booklm.experiments.ch17 import TRIGGER, lora_model, sft_split
from booklm.experiments.ch18 import sft_loss

tok = load_tokenizer()
val, train = sft_split()          # 200 отложенных и 9752 обучающих
asks = [row["request"] for row in val[:50]]

def doors(model, tail=""):        # доля ответов с фразы-закладки
    starts = [generate(tok, model, a + tail, chat=True, max_new=12)
              .strip().startswith("Откройте все двери") for a in asks]
    return sum(starts) / len(starts)

for n in (0, 10, 30, 100, 300):   # n отравленных пар, 0 -- чистый
    model = lora_model(f"poison-{n}" if n else "sft-1epoch", n_poison=n)
    loss = sft_loss(tok, model, val)
    with_t, without = doors(model, " " + TRIGGER), doors(model)
    print(f"n={n:3d} ({n / len(train):.2%}): ошибка {loss:.4f} | "
          f"с триггером {with_t:4.0%}, без {without:.0%}")
# Вывод в книге:
# | n=  0 (0.00%): ошибка 1.5892 | с триггером   0%, без 0%
# | n= 10 (0.10%): ошибка 1.5892 | с триггером   0%, без 0%
# | n= 30 (0.31%): ошибка 1.5890 | с триггером   0%, без 0%
# | n=100 (1.03%): ошибка 1.5893 | с триггером  46%, без 0%
# | n=300 (3.08%): ошибка 1.5895 | с триггером 100%, без 0%

# %% Листинг «Строка-метка: место среди 10 000 кандидатов и экспозиция» — раздел «Запоминание и строка-метка»
CANARY = "PIN клиента: 7316"

def pin_corpus(copies, n=1000, seed=17):  # фон + copies копий метки
    rng = random.Random(seed)
    lines = [f"PIN клиента: {rng.randrange(10_000):04d}"
             for _ in range(n)]
    return lines + [CANARY] * copies

def fit_digits(lines):                        # пары соседних знаков PIN
    pairs = Counter()
    for line in lines:
        chain = ":" + line.rsplit(": ", 1)[1]  # ':' d1 d2 d3 d4
        pairs.update(zip(chain, chain[1:]))
    return pairs

def log_p(pairs, pin, alpha=1.0):          # сглаживание: +1 к паре
    chain = ":" + pin
    total = {a: sum(pairs[a, d] for d in "0123456789")
             for a in set(chain)}
    return sum(math.log((pairs[a, b] + alpha) / (total[a] + 10 * alpha))
               for a, b in zip(chain, chain[1:]))

def exposure(lines, secret="7316"):           # log2 N - log2 m
    pairs = fit_digits(lines)
    target = log_p(pairs, secret)
    place = 1 + sum(log_p(pairs, f"{x:04d}") > target
                    for x in range(10_000))
    return place, math.log2(10_000) - math.log2(place)

for copies in (0, 1, 4, 16):
    place, bits = exposure(pin_corpus(copies))
    print(f"c={copies:2d}: место {place:5d}, экспозиция {bits:5.2f} бит")
deduped = list(dict.fromkeys(pin_corpus(16)))  # точная дедупликация
print("после дедупликации: копий", deduped.count(CANARY),
      "| экспозиция %.2f бит" % exposure(deduped)[1])
# Вывод в книге:
# | c= 0: место  5509, экспозиция  0.86 бит
# | c= 1: место  4250, экспозиция  1.23 бит
# | c= 4: место  1446, экспозиция  2.79 бит
# | c=16: место     1, экспозиция 13.29 бит
# | после дедупликации: копий 1 | экспозиция 1.47 бит

# %% Листинг «Шаг DP-SGD: обрезка по примерам и шум» — раздел «Обучение с DP-SGD»
import torch

def clip_rows(grads, C):                 # grads: [B, P], строка -- пример
    norms = grads.norm(dim=1, keepdim=True)
    return grads * (C / norms).clamp(max=1.0)    # длина строки <= C

def dp_grad(grads, C, sigma, L, gen):    # сумма обрезанных + шум, деление на L
    noise = torch.randn(grads.shape[1], generator=gen) * sigma * C
    return (clip_rows(grads, C).sum(0) + noise) / L

D = torch.tensor([[0.3, -0.4], [0.6, 0.0], [-0.2, 0.5], [30.0, 40.0]])
D2, C, L = D[:3], 1.0, 4                 # D2 -- тот же пакет без секрета
print("длины:", [round(n, 2) for n in D.norm(dim=1).tolist()])
print("сдвиг от секрета: без обрезки %.2f, с обрезкой %.2f" % (
    ((D.sum(0) - D2.sum(0)) / L).norm(),
    ((clip_rows(D, C).sum(0) - clip_rows(D2, C).sum(0)) / L).norm()))
u = D[3] / D[3].norm()                   # направление секрета
for sigma in (0.0, 0.5, 1.0, 4.0):
    gen = torch.Generator().manual_seed(0)
    a = torch.stack([dp_grad(D, C, sigma, L, gen) for _ in range(4000)]) @ u
    b = torch.stack([dp_grad(D2, C, sigma, L, gen) for _ in range(4000)]) @ u
    cut = (a.mean() + b.mean()) / 2      # порог посередине
    hit = ((a > cut).float().mean() + (b <= cut).float().mean()) / 2
    print(f"sigma={sigma}: по шагу угадали, был ли секрет, в {hit:.1%}")
# Вывод в книге:
# | длины: [0.5, 0.6, 0.54, 50.0]
# | сдвиг от секрета: без обрезки 12.50, с обрезкой 0.25
# | sigma=0.0: по шагу угадали, был ли секрет, в 100.0%
# | sigma=0.5: по шагу угадали, был ли секрет, в 84.9%
# | sigma=1.0: по шагу угадали, был ли секрет, в 69.5%
# | sigma=4.0: по шагу угадали, был ли секрет, в 56.0%

# %% Листинг «Учёт бюджета: от шагов DP-SGD к» — раздел «Обучение с DP-SGD»
import math

# Формула Миронова и др., вывод в углублении выше; читать её не обязательно.
def rdp_sampled(alpha, q, sigma):     # один шаг: выборка с долей q, шум sigma
    terms = [math.log(math.comb(alpha, k)) + (alpha - k) * math.log1p(-q)
             + k * math.log(q) + (k * k - k) / (2 * sigma ** 2)
             for k in range(alpha + 1)]
    top = max(terms)                  # сумма экспонент без переполнения
    return (top + math.log(sum(math.exp(t - top) for t in terms))) / (alpha - 1)

def epsilon(T, delta, rdp_one):       # T шагов: сложить, перевести в (eps, delta)
    return min(T * rdp_one(a) + math.log(1 / delta) / (a - 1)
               for a in range(2, 128))

delta = 1e-5
for sigma in (1.0, 2.0, 4.0, 8.0):
    sampled = epsilon(1000, delta, lambda a: rdp_sampled(a, 0.1, sigma))
    whole = epsilon(100, delta, lambda a: a / (2 * sigma ** 2))
    print(f"sigma={sigma}: 1000 шагов по 10% -- eps={sampled:5.2f} | "
          f"100 шагов по всем данным -- eps={whole:6.2f}")
# Вывод в книге:
# | sigma=1.0: 1000 шагов по 10% -- eps=28.55 | 100 шагов по всем данным -- eps=111.51
# | sigma=2.0: 1000 шагов по 10% -- eps= 9.84 | 100 шагов по всем данным -- eps= 36.51
# | sigma=4.0: 1000 шагов по 10% -- eps= 4.24 | 100 шагов по всем данным -- eps= 15.13
# | sigma=8.0: 1000 шагов по 10% -- eps= 2.00 | 100 шагов по всем данным -- eps=  6.78

# %% Листинг «Вырезание направления из весов BookLM» — раздел «Выпуск открытых весов»
import copy
import torch
from booklm.lab import generate, load, perplexity
from booklm.experiments.ch10_representations import BOYS, GIRLS
from booklm.experiments.ch18_watermark_cut import (BOY, GIRL, after_block1,
                                              she_share)
tok, model = load("base")
mean = lambda tmpl, names: torch.stack(
    [after_block1(tok, model, tmpl.format(n)) for n in names]).mean(0)
v = mean(GIRL, GIRLS[:10]) - mean(BOY, BOYS[:10])
r = v / v.norm()
P = torch.eye(256) - torch.outer(r, r)       # убирает проекцию на r
cut = copy.deepcopy(model)
with torch.no_grad():                        # меняем только веса
    cut.embed.weight.copy_(cut.embed.weight @ P)
    for b in cut.blocks:
        b.attention.out.weight.copy_(P @ b.attention.out.weight)
        b.ffn.down.weight.copy_(P @ b.ffn.down.weight)
for name, m in (("исходная", model), ("без направления", cut)):
    print(f"{name:15} «она»: мальчики "
          f"{she_share(tok, m, BOYS[10:], BOY):.3f}, девочки "
          f"{she_share(tok, m, GIRLS[10:], GIRL):.3f} | "
          f"перплексия {perplexity(m, n=20_000):.2f}")
print(generate(tok, cut, BOY.format("Тим"), max_new=14, temperature=0))
# Вывод в книге:
# | исходная        «она»: мальчики 0.002, девочки 0.999 | перплексия 7.25
# | без направления «она»: мальчики 0.843, девочки 0.997 | перплексия 7.73
# | Жил-был мальчик по имени Тим. Однажды у неё в саду была большая, яркая радуга. Она очень любила свою

# %% Листинг «pickle: загрузка весов исполняет чужой код» — раздел «Файл весов с исполняемым кодом»
import pickle
import pickletools
import tempfile
from pathlib import Path

flag = Path(tempfile.mkdtemp()) / "pwned.txt"   # след исполненного кода

class Payload:                                  # «веса» от нарушителя
    def __reduce__(self):                       # «собери меня так: ...»
        return (exec, (f"open({str(flag)!r}, 'w').write('сработало')",))

blob = pickletools.optimize(pickle.dumps(Payload(), protocol=2,
                                         fix_imports=False))
print([op.name for op, _, _ in pickletools.genops(blob)])
print("до загрузки:", flag.exists())
pickle.loads(blob)                              # «загрузили веса»...
print("после загрузки:", flag.exists(), flag.read_text())
flag.unlink()
# Вывод в книге:
# | ['PROTO', 'GLOBAL', 'BINUNICODE', 'TUPLE1', 'REDUCE', 'STOP']
# | до загрузки: False
# | после загрузки: True сработало

# %% Листинг «Сколько вызовов в честном файле весов BookLM» — раздел «Файл весов с исполняемым кодом»
import zipfile
from booklm.lab import MODELS

with zipfile.ZipFile(MODELS / "booklm-base.pt") as z:   # torch.save: zip
    name = next(n for n in z.namelist() if n.endswith("data.pkl"))
    data = z.read(name)
ops = Counter(op.name for op, _, _ in pickletools.genops(data))
print(f"{name}: {len(data)} байт, команд {sum(ops.values())}")
print({k: ops[k]
       for k in ("GLOBAL", "STACK_GLOBAL", "REDUCE", "BINPERSID")})
print("импортирует:", sorted(arg for op, arg, _ in pickletools.genops(data)
                             if op.name == "GLOBAL"))
# Вывод в книге:
# | booklm-base/data.pkl: 8755 байт, команд 2324
# | {'GLOBAL': 3, 'STACK_GLOBAL': 0, 'REDUCE': 114, 'BINPERSID': 56}
# | импортирует: ['collections OrderedDict', 'torch FloatStorage', 'torch._utils _rebuild_tensor_v2']

# %% Листинг «Сканер pickle: проверка имён» — раздел «Файл весов с исполняемым кодом»
SAFE = {"collections OrderedDict", "torch FloatStorage",
        "torch._utils _rebuild_tensor_v2"}         # разрешённые имена
MEMO = {"MEMOIZE", "PUT", "BINPUT", "LONG_BINPUT", "FRAME"}

def scan_pickle(data):               # -> имена не из списка
    found, pushed = [], []           # pushed: что команды клали на стек
    for op, arg, _ in pickletools.genops(data):
        if op.name in MEMO:
            continue                 # запись в память стек не меняет
        if op.name in ("GLOBAL", "INST"):
            name = arg
        elif op.name == "STACK_GLOBAL":   # имя -- две строки со стека
            two = pushed[-2:]
            name = " ".join(two) if None not in two else "имя не прочитать"
        elif op.name.startswith("EXT"):
            name = "имя из реестра"
        else:
            pushed.append(arg if "UNICODE" in op.name else None)
            continue
        if name not in SAFE:
            found.append(name)
        pushed.append(None)          # на стеке теперь функция
    return found

print("честный файл BookLM:", scan_pickle(data))
print("файл нарушителя:", scan_pickle(blob),
      scan_pickle(pickle.dumps(Payload(), protocol=4)))
# Вывод в книге:
# | честный файл BookLM: []
# | файл нарушителя: ['builtins exec'] ['builtins exec']

# %% Листинг «Заголовок safetensors: каждое смещение проверяется до чтения» — раздел «Формат safetensors»
DTYPES = {"F32": 4, "F16": 2, "BF16": 2, "I64": 8}   # байт на число

def check_header(header, data_len):         # data_len -- буфер чисел
    spans = []
    for name, info in header.items():
        if name == "__metadata__":
            continue
        dtype, shape, (begin, end) = (info["dtype"], info["shape"],
                                      info["data_offsets"])
        if dtype not in DTYPES:
            raise ValueError(f"{name}: неизвестный dtype {dtype}")
        if any(size < 0 for size in shape):
            raise ValueError(f"{name}: отрицательный размер {shape}")
        if math.prod(shape) * DTYPES[dtype] != end - begin:
            raise ValueError(f"{name}: форма не равна длине отрезка")
        spans.append((begin, end, name))
    position = 0
    for begin, end, name in sorted(spans):      # без дыр и перекрытий
        if begin != position:
            raise ValueError(f"{name}: дыра или перекрытие "
                             f"на {position}")
        position = end
    if position != data_len:
        raise ValueError("отрезки не покрывают буфер целиком")
    return "принят"

good = {"w": {"dtype": "F32", "shape": [2, 3], "data_offsets": [0, 24]},
        "b": {"dtype": "F32", "shape": [3], "data_offsets": [24, 36]}}
print(check_header(good, 36))
BAD = [({"data_offsets": [20, 32]}, 36),   # перекрытие
       ({"data_offsets": [28, 40]}, 40),   # дыра в 4 байта
       ({"dtype": "PY_OBJECT"}, 36),       # выдуманный тип
       ({"shape": [-1, -3]}, 36)]          # минус на минус
for patch, data_len in BAD:
    try:
        bad_b = {**good["b"], **patch}
        print(check_header({**good, "b": bad_b}, data_len))
    except ValueError as error:
        print("отказ:", error)
# Вывод в книге:
# | принят
# | отказ: b: дыра или перекрытие на 24
# | отказ: b: дыра или перекрытие на 24
# | отказ: b: неизвестный dtype PY_OBJECT
# | отказ: b: отрицательный размер [-1, -3]

# %% Листинг «check_header на файле весов BookLM» — раздел «Формат safetensors»
import json
import struct

path = MODELS / "hf" / "base" / "model.safetensors"
with open(path, "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]       # длина заголовка
    size = f.seek(0, 2)                          # длина файла
    if n > size - 8:
        raise ValueError("заголовок длиннее файла")
    f.seek(8)
    header = json.loads(f.read(n))
data_len = size - 8 - n
names = [k for k in header if k != "__metadata__"]
print(f"заголовок {n} байт, буфер {data_len} байт, тензоров {len(names)},",
      "типы:", {header[k]["dtype"] for k in names})
print(check_header(header, data_len))
# Вывод в книге:
# | заголовок 6056 байт, буфер 13244928 байт, тензоров 56, типы: {'BF16'}
# | принят

# %% Листинг «Проверка пакета перед установкой» — раздел «Выдуманные пакеты»
from datetime import date

INDEX = {"requests": date(2011, 2, 14),      # снимок индекса:
         "booklm-guard": date(2026, 10, 27)}  # имя -> первый выпуск
TODAY, MIN_AGE = date(2026, 10, 29), 30

def package_known(tool, args, context, cred):  # проверка для CHECKS
    if tool != "pip_install":
        return None
    first = INDEX.get(args.get("package"))
    if first is None:
        return "deny", "пакета нет в индексе"
    age = (TODAY - first).days
    if age < MIN_AGE:
        return "ask", f"пакету {age} дн."
    return None                                # решат другие проверки

for name in ("requests", "booklm-guard", "booklm-kvcache"):
    print(f"{name:15}", package_known("pip_install", {"package": name},
                                      {}, None))
# Вывод в книге:
# | requests        None
# | booklm-guard    ('ask', 'пакету 2 дн.')
# | booklm-kvcache  ('deny', 'пакета нет в индексе')

# %% Листинг «Подпись манифеста и чей открытый ключ» — раздел «Подпись, хеш родителя и перечень компонентов»
import hashlib
import sys
sys.path[:0] = ["labs/ch18", "labs/ch13"]   # запуск из корня репозитория
from guard import keygen, sign, verify      # подпись Лэмпорта
from agent import canonical                 # канонический JSON, глава 13

def manifest(sha256):
    return canonical({"id": "booklm-guarded-v1",
                      "sha256": sha256}).encode()

good = manifest(hashlib.sha256(path.read_bytes()).hexdigest())
bad = manifest("0" * 64)                    # хеш подменённых весов
sk, pk = keygen(b"publisher-booklm")        # ключ издателя
evil_sk, evil_pk = keygen(b"attacker")      # ключ нарушителя
sig, evil_sig = sign(sk, good), sign(evil_sk, bad)
print("подпись:", len(sig), "клеток по", len(sig[0]), "байта")
print("верный манифест:        ", verify(pk, good, sig))
print("подменили хеш:          ", verify(pk, bad, sig))
print("подпись нарушителя:     ", verify(pk, bad, evil_sig))
print("его же открытым ключом: ", verify(evil_pk, bad, evil_sig))
# Вывод в книге:
# | подпись: 256 клеток по 32 байта
# | верный манифест:         True
# | подменили хеш:           False
# | подпись нарушителя:      False
# | его же открытым ключом:  True
