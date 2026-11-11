"""Глава 14 «Инференс», раздел «PagedAttention и vLLM»: раздельное выполнение и префиксное дерево на BookLM.

Опыты (время — на процессоре ноутбука Apple M2 Max):
  ch14_disagg  рубрика «раздельное выполнение», подраздел «Раздельная обработка входа и генерации»;
               несколько секунд. Процесс A считает KV-кэш промпта и отдаёт его байтами отдельному процессу B
               (режим decode ниже), B продолжает жадную генерацию. Печатает размер кэша в FP32 и BF16,
               сколько из 40 токенов совпало с генерацией в одном процессе, продолжение и сдвиг логитов
               от передачи в BF16.
  ch14_radix   рубрика «префиксное дерево», подраздел «Префиксное дерево SGLang»; около 20 с.
               500 просьб из sft.jsonl: сколько токенов берётся из кэша у цепочки хешей страниц по 16
               токенов (как в vLLM) и у префиксного дерева (как RadixAttention в SGLang), время обработки
               входа BookLM без кэша и с каждым из них.

Запуск из корня репозитория:
    python -m booklm.experiments.ch14_disagg_radix              # оба опыта
    python -m booklm.experiments.ch14_disagg_radix ch14_disagg  # один опыт
Вывод каждого опыта копируется в models/booklm/experiments/<имя опыта через дефис>.txt:
ch14-disagg.txt и ch14-radix.txt. Время зависит от машины.
"""
import contextlib
import hashlib
import io
import json
import statistics
import sys
import time
from pathlib import Path

import torch

if not __package__:  # запуск файлом (python booklm/experiments/ch14_disagg_radix.py): корень репозитория — в путь поиска
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from booklm.lab import MODELS, load, load_tokenizer, prompt_ids

OUT = MODELS / "experiments"
PER_TOKEN = 6 * 2 * 2 * 32 * 4          # блоки x (K, V) x KV-головы x d_h x байт FP32 = 3072


# ============================================================================ ch14-disagg
# Процесс A (тот, кто зовёт ch14_disagg) читает 400 токенов сказки, выбирает первый токен ответа и
# превращает KV-кэш в байты. Процесс B — отдельный интерпретатор Python:
#     python -m booklm.experiments.ch14_disagg_radix decode <первый токен> <T> <тип чисел> <n>
# Он получает байты через стандартный ввод, сам загружает веса, восстанавливает кэш и жадно дописывает
# n токенов; номера печатает через пробел. Вывод опыта: models/booklm/experiments/ch14-disagg.txt.

DTYPES = {"torch.float32": torch.float32, "torch.bfloat16": torch.bfloat16}


def pack_kv(caches, dtype):                          # из листинга главы (сторона A)
    kv = torch.stack([torch.stack([c["k"], c["v"]]) for c in caches])
    return kv.to(dtype).contiguous().view(torch.uint8).numpy().tobytes()


@torch.no_grad()
def decode_side(blob, first, T, dtype, n):           # из листинга главы (сторона B)
    tok, model = load("base")
    kv = torch.frombuffer(bytearray(blob), dtype=dtype)
    kv = kv.view(6, 2, 1, 2, T, 32).float()  # форма тензора из A
    caches = [{"k": b[0], "v": b[1], "len": T} for b in kv]
    out, nxt = [first], torch.tensor([[first]])
    for _ in range(n - 1):
        logits = model(nxt, caches)
        for c in caches:
            c["len"] += 1
        nxt = logits[:, -1].argmax(-1, keepdim=True)
        out.append(nxt.item())
    return out


@torch.no_grad()
def ch14_disagg():
    """Тот же код, что в листинге главы, и замер сдвига логитов от BF16 при передаче."""
    import subprocess
    from booklm.lab import ROOT
    from booklm.experiments.ch14 import long_story
    tok, model = load("base")
    prompt = long_story(448)[None, :400]                 # 400 токенов сказки
    caches = [{"len": 0} for _ in model.blocks]          # машина A
    first = model(prompt, caches)[0, -1].argmax().item()
    whole = model.generate(prompt, max_new=40,           # одна машина
                           temperature=0)[0, 400:].tolist()
    for dtype in (torch.float32, torch.bfloat16):
        blob = pack_kv(caches, dtype)
        b = subprocess.run(
            [sys.executable, "-m", "booklm.experiments.ch14_disagg_radix", "decode",
             str(first), "400", str(dtype), "40"],
            input=blob, capture_output=True, check=True, cwd=ROOT)
        got = [int(x) for x in b.stdout.split()]
        same = next((i for i, x in enumerate(got) if x != whole[i]), 40)
        print(f"{dtype}: {len(blob):,} байт, совпало {same} из 40")
    print("продолжение:", tok.decode(whole[:whole.index(2)]))   # до <eos>
    # насколько BF16 на передаче сдвигает логиты первого шага генерации
    for c in caches:
        c["len"] = 400
    rounded = [{"k": c["k"].bfloat16().float(), "v": c["v"].bfloat16().float(), "len": 400}
               for c in caches]
    nxt = torch.tensor([[first]])
    a, b = model(nxt, caches)[0, -1], model(nxt, rounded)[0, -1]
    top = a.topk(2).values.tolist()
    print(f"BF16-кэш: логиты шага 1 сдвинулись до {(a - b).abs().max().item():.3f}; "
          f"два лучших логита {top[0]:.3f} и {top[1]:.3f}")


# ============================================================================ ch14-radix

def sft_requests(n=500):
    tok = load_tokenizer()
    with open(MODELS / "data" / "sft.jsonl", encoding="utf-8") as f:
        rows = [json.loads(line) for _, line in zip(range(n), f)]
    return [prompt_ids(tok, r["request"], chat=True) for r in rows]


def page_hashes(ids, size=16):           # цепочка хешей полных страниц
    h, out = b"", []
    for a in range(0, len(ids) - size + 1, size):
        h = hashlib.sha256(h + str(ids[a:a + size]).encode()).digest()
        out.append(h)
    return out


def pages_hit(known, ids, size=16):      # сколько токенов дают совпавшие страницы
    hashes, hit = page_hashes(ids, size), 0
    while hit < len(hashes) and hashes[hit] in known:
        hit += 1
    known.update(hashes)
    return min(hit * size, len(ids) - 1)


def tree_hit(root, ids):                 # длина общего начала с деревом, затем вставка
    node, n = root, 0
    while n < len(ids) and ids[n] in node:
        node, n = node[ids[n]], n + 1
    for t in ids[n:]:
        node = node.setdefault(t, {})
    return min(n, len(ids) - 1)          # последний токен считаем всегда: нужны логиты


@torch.no_grad()
def prefill_seconds(model, requests, reuse, repeats=5):
    """Время обработки входа всех запросов, если первые reuse[i] токенов берутся из кэша.

    Кэш каждого запроса сохраняется целиком; общий префикс берём срезом кэша того запроса,
    который положил это начало первым. Повторяем и берём медиану.
    """
    def run():
        stored = []
        for ids, r in zip(requests, reuse):
            caches = [{"len": 0} for _ in model.blocks]
            if r:
                src = next(s for s in stored if s[0][:r] == ids[:r])[1]
                caches = [{"k": c["k"][:, :, :r], "v": c["v"][:, :, :r], "len": r} for c in src]
            model(torch.tensor([ids[r:]]), caches)
            stored.append((ids, caches))
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        run()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def ch14_radix():
    """Рубрика «префиксное дерево»: страницы по 16 токенов против дерева на 500 просьбах."""
    torch.set_num_threads(1)
    requests = sft_requests(500)
    known, root = set(), {}
    by_pages = [pages_hit(known, ids) for ids in requests]
    by_tree = [tree_hit(root, ids) for ids in requests]
    total = sum(map(len, requests))
    print(f"запросов {len(requests)}, токенов {total}, средняя длина {total / len(requests):.1f}")
    print(f"страницы по 16: {sum(by_pages)} токенов из кэша ({sum(by_pages) / total:.1%})")
    print(f"дерево:         {sum(by_tree)} токенов из кэша ({sum(by_tree) / total:.1%})")
    _, model = load("instruct")
    t0 = prefill_seconds(model, requests, [0] * len(requests))
    t1 = prefill_seconds(model, requests, by_pages)
    t2 = prefill_seconds(model, requests, by_tree)
    print(f"обработка входа на CPU: без кэша {t0:.2f} с, страницы {t1:.2f} с, дерево {t2:.2f} с")


def main():
    if sys.argv[1:2] == ["decode"]:                  # процесс B опыта ch14_disagg
        first, T, dtype, n = sys.argv[2:6]
        ids = decode_side(sys.stdin.buffer.read(), int(first), int(T), DTYPES[dtype], int(n))
        print(" ".join(map(str, ids)))
        return
    names = sys.argv[1:] or ["ch14_disagg", "ch14_radix"]
    for name in names:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            globals()[name]()
        text = buf.getvalue()
        print(f"== {name}\n{text}")
        (OUT / f"{name.replace('_', '-')}.txt").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    from booklm.experiments import run
    if sys.argv[1:2] == ["decode"]:                  # процесс B опыта ch14_disagg
        main()
    else:
        run({"ch14_disagg": ch14_disagg, "ch14_radix": ch14_radix}, save=True)
