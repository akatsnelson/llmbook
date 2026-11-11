"""Строительные токены BPE: мера IoS, словарь Qwen3 на сказках BookLM и Picky BPE.

Полная версия листингов «Ступеньки слияний» и «Строительные токены BookLM» из главы 3
(подразделы 3.5.3 «Строительные токены» и 3.5.5 «Чистка словаря»).
Здесь те же числа проверены утверждениями assert, а сверх книги есть три опыта:

* IoS токенов Qwen3 на тех же сказках, на которых училась BookLM (нужен tokenizer.json Qwen3: берём
  его из кэша Hugging Face, а если его там нет, скачиваем только этот файл);
* доля таблицы эмбеддингов в параметрах Qwen3-0.6B и Qwen3.5-0.8B по заголовкам их файлов
  safetensors (опыт пропускается, если весов нет в кэше);
* токенизатор BookLM, обученный заново обычным BPE и Picky BPE (Chizhov и др., EMNLP 2024) при том же
  размере словаря, и сжатие отложенных историй каждым из них.

IoS (intersection over self) токена считаем после обучения: сколько раз токен собрали при кодировании
корпуса и сколько раз из этих сборок он так и остался в готовой токенизации. Picky BPE считает ту же
меру по одной паре в момент её слияния: IoS(x1 | x1, x2) = f(x1, x2) / f(x1).

Случайности в опыте нет: числа одинаковы при каждом запуске. Обучение двух токенизаторов на чистом
Python занимает несколько минут, кодирование сказок токенизатором Qwen3 — около минуты.

Запуск из корня репозитория: python labs/ch03/scaffold_tokens.py
Время: около полутора минут на Apple M2 Max (на медленном процессоре — несколько минут).
"""

import heapq
import json
import struct
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from booklm.lab import MODELS, load_tokenizer  # noqa: E402
from booklm.experiments.ch03 import readable  # noqa: E402
from tokenizers import Tokenizer, pre_tokenizers  # noqa: E402

VOCAB = 8192
SPECIAL = 5                       # <pad>, <bos>, <eos>, <|user|>, <|assistant|>
HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"


# --- Байтовая запись GPT-2: символ файла -> байт ---------------------------------------------------

def _char_to_byte():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) \
        + list(range(ord("®"), ord("ÿ") + 1))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


CH2B = _char_to_byte()


def piece_bytes(piece):
    return bytes(CH2B[c] for c in piece)


def kind(piece, is_base):
    """Байт словаря, неполная буква UTF-8, одна буква или кусок слова (пробел впереди не считаем)."""
    if is_base:
        return "байт"
    try:
        text = piece_bytes(piece).decode("utf-8")
    except UnicodeDecodeError:
        return "не UTF-8"
    return "одна буква" if len(text.lstrip(" ")) <= 1 else "кусок слова"


# --- IoS после обучения --------------------------------------------------------------------------

def scaffold_counts(alone, merges):
    """alone: сколько раз кусок остался в готовой токенизации. Возвращает, сколько раз его собрали."""
    built = Counter(alone)
    for a, b in reversed(merges):          # от длинных склеек к коротким
        built[a] += built[a + b]
        built[b] += built[a + b]
    return built


def ios_report(path, counts, skip_ids):
    model = json.load(open(path, encoding="utf-8"))["model"]
    vocab, merges = model["vocab"], [tuple(m) for m in model["merges"]]
    piece_of = {i: p for p, i in vocab.items()}
    merged = {a + b for a, b in merges}
    alone = Counter({piece_of[i]: int(c) for i, c in enumerate(counts) if i not in skip_ids and i in piece_of})
    built = scaffold_counts(alone, merges)
    total = sum(alone.values())
    ios = {p: 1 - alone[p] / built[p] for p in built if built[p] > 0}
    rows = {}
    for thr in (0.9, 0.99):
        top = [p for p in ios if ios[p] > thr]
        kinds = Counter(kind(p, p not in merged) for p in top)
        words = [p for p in top if kind(p, p not in merged) == "кусок слова"]
        cyr = sum(1 for p in top if kind(p, p not in merged) in ("не UTF-8", "байт")
                  and any(0xD0 <= x <= 0xD1 or 0x80 <= x <= 0xBF for x in piece_bytes(p)))
        rows[thr] = dict(n=len(top), share=sum(alone[p] for p in top) / total, kinds=kinds,
                         word_share=sum(alone[p] for p in words) / total, cyr=cyr,
                         top_words=sorted(words, key=lambda p: -built[p])[:12])
    return dict(vocab=len(vocab), built=len(ios), never=sum(1 for p in ios if alone[p] == 0),
                total=total, rows=rows, alone=alone, builtc=built)


def print_report(name, r):
    print(f"{name}: словарь {r['vocab']}, собирались хоть раз {r['built']}, токенов в корпусе {r['total']:,}")
    for thr, row in r["rows"].items():
        kinds = ", ".join(f"{k} {v}" for k, v in sorted(row["kinds"].items()))
        print(f"  IoS > {thr}: {row['n']} ({row['n'] / r['vocab']:.2%} словаря), {row['share']:.2%} токенов "
              f"корпуса; {kinds}; куски слов дают {row['word_share']:.2%} корпуса; "
              f"с байтами кириллицы {row['cyr']}")
    words = r["rows"][0.99]["top_words"]
    print("  самые частые строительные куски слов:",
          ", ".join(f"{readable(p)} {r['alone'][p]}/{r['builtc'][p]}" for p in words[:8]))


def booklm_ios():
    stream = np.fromfile(MODELS / "data" / "train.bin", dtype=np.uint16)
    seen = np.bincount(stream, minlength=VOCAB)
    r = ios_report(MODELS / "tokenizer.json", seen, set(range(SPECIAL)))
    print_report("BookLM на train.bin", r)
    rows = r["rows"]
    assert r["total"] == 12_507_970 - int(seen[:SPECIAL].sum())
    assert rows[0.9]["n"] == 1274 and rows[0.99]["n"] == 338
    assert rows[0.99]["kinds"] == {"кусок слова": 239, "байт": 61, "одна буква": 32, "не UTF-8": 6}
    assert abs(rows[0.99]["word_share"] - 0.0015) < 0.0001
    assert r["never"] == 67                  # 65 склеенных токенов и два байта, ни разу не оставшихся сами
    return r


def booklm_stories(name="train"):
    tok = load_tokenizer()
    s = np.fromfile(MODELS / "data" / f"{name}.bin", dtype=np.uint16)
    cuts = np.flatnonzero(s == 2)
    out, a = [], 0
    for c in cuts:
        out.append(tok.decode(s[a:c].tolist()))
        a = c + 1
    return out


def qwen_tokenizer_path():
    found = sorted(HF_CACHE.glob("models--Qwen--Qwen3-0.6B/snapshots/*/tokenizer.json"))
    if found:
        return found[0]
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download("Qwen/Qwen3-0.6B", "tokenizer.json"))


def qwen_ios(stories):
    path = qwen_tokenizer_path()
    q = Tokenizer.from_file(str(path))
    counts = np.zeros(q.get_vocab_size(with_added_tokens=True), dtype=np.int64)
    for e in q.encode_batch(stories):
        np.add.at(counts, e.ids, 1)
    n = len(json.load(open(path, encoding="utf-8"))["model"]["vocab"])
    r = ios_report(path, counts[:n], set())
    print_report("Qwen3 на тех же сказках", r)
    assert r["vocab"] == 151_643 and r["built"] == 5544
    assert r["rows"][0.99]["n"] == 1180
    assert r["rows"][0.99]["kinds"]["кусок слова"] == 822 and r["rows"][0.99]["kinds"]["не UTF-8"] == 109
    return r


# --- Доля таблицы эмбеддингов по заголовкам safetensors ------------------------------------------

def safetensors_params(pattern):
    groups = Counter()
    for f in sorted(HF_CACHE.glob(pattern)):
        with open(f, "rb") as fh:
            header = json.loads(fh.read(struct.unpack("<Q", fh.read(8))[0]))
        header.pop("__metadata__", None)
        for name, t in header.items():
            g = ("embed" if "embed_tokens" in name else "lm_head" if "lm_head" in name
                 else "visual" if "visual" in name else "mtp" if "mtp" in name else "blocks")
            groups[g] += int(np.prod(t["shape"]))
    return groups


def embedding_shares():
    q3 = safetensors_params("models--Qwen--Qwen3-0.6B/snapshots/*/model.safetensors")
    if q3:
        # tie_word_embeddings=true: lm_head в файле лежит копией той же таблицы, параметры одни
        total = q3["embed"] + q3["blocks"]
        print(f"Qwen3-0.6B: таблица {q3['embed']:,} из {total:,} параметров ({q3['embed'] / total:.1%})")
        assert q3["embed"] == 151_936 * 1024 and total == 596_049_920
    q35 = safetensors_params("models--Qwen--Qwen3.5-0.8B/snapshots/*/*.safetensors")
    if q35:
        total = q35["embed"] + q35["blocks"]          # без кодировщика картинок и блока MTP
        print(f"Qwen3.5-0.8B: таблица {q35['embed']:,} из {total:,} параметров языковой части "
              f"({q35['embed'] / total:.1%})")
        assert q35["embed"] == 248_320 * 1024 and total == 752_393_024


# --- Обычный BPE и Picky BPE на сказках BookLM ---------------------------------------------------

PRE = pre_tokenizers.ByteLevel(add_prefix_space=False)


def word_counts(texts):
    wc = Counter()
    for t in texts:
        for piece, _ in PRE.pre_tokenize_str(t):
            wc[piece] += 1
    return {tuple(CH2B[c] for c in w): n for w, n in wc.items()}


def train_bpe(wc, size, threshold=None):
    """Байтовый BPE до size рабочих токенов. threshold=None — обычный BPE, иначе Picky BPE:
    после слияния (a, b) токен a (и b) удаляется, если IoS(a | a, b) = f(a, b) / f(a) >= threshold,
    и все его вхождения снова делятся на две части, из которых он был склеен.
    Возвращает события по порядку: ("m", a, b, x) — слияние, ("r", x) — удаление."""
    words, freq = [list(w) for w in wc], list(wc.values())
    pair_cnt, pair_where = Counter(), defaultdict(set)
    tok_cnt, tok_where = Counter(), defaultdict(set)
    parts, by_pair = {}, {}

    def count(wi, sign):
        w, f = words[wi], freq[wi]
        for t in w:
            tok_cnt[t] += sign * f
            if sign > 0:
                tok_where[t].add(wi)
        for p in zip(w, w[1:]):
            pair_cnt[p] += sign * f
            if sign > 0:
                pair_where[p].add(wi)

    def rewrite(wi, fn):
        old = words[wi]
        count(wi, -1)
        words[wi] = fn(old)
        count(wi, 1)
        for t in set(old) - set(words[wi]):
            tok_where[t].discard(wi)
        new_pairs = set(zip(words[wi], words[wi][1:]))
        old_pairs = set(zip(old, old[1:]))
        for p in old_pairs - new_pairs:
            pair_where[p].discard(wi)
        return old_pairs | new_pairs

    def merged(a, b, x):
        def fn(w):
            r, i = [], 0
            while i < len(w):
                if i + 1 < len(w) and w[i] == a and w[i + 1] == b:
                    r.append(x)
                    i += 2
                else:
                    r.append(w[i])
                    i += 1
            return r
        return fn

    def split(x):
        a, b = parts[x]
        return lambda w: [y for t in w for y in ((a, b) if t == x else (t,))]

    for wi in range(len(words)):
        count(wi, 1)
    heap = [(-c, p) for p, c in pair_cnt.items()]
    heapq.heapify(heap)

    def push(pairs):
        for p in pairs:
            if pair_cnt[p] > 0:
                heapq.heappush(heap, (-pair_cnt[p], p))

    events, active, nxt = [], set(range(256)), 256
    while len(active) < size and heap:
        c, p = heapq.heappop(heap)
        if pair_cnt[p] != -c:
            continue                                   # устаревшая запись очереди
        if -c < 2:
            break
        a, b = p
        if p not in by_pair:                           # удалённый токен возвращается со старым номером
            parts[nxt], by_pair[p] = p, nxt
            nxt += 1
        x = by_pair[p]
        fp, fa, fb = pair_cnt[p], tok_cnt[a], tok_cnt[b]
        events.append(("m", a, b, x))
        active.add(x)
        touched = set()
        for wi in list(pair_where[p]):
            touched |= rewrite(wi, merged(a, b, x))
        push(touched)
        if threshold is None:
            continue
        for t, ft in [(a, fa)] + ([(b, fb)] if b != a else []):
            if t >= 256 and t in active and fp / ft >= threshold:
                events.append(("r", t))
                active.discard(t)
                touched = set()
                for wi in list(tok_where[t]):
                    touched |= rewrite(wi, split(t))
                push(touched)
    return events, active


def make_encoder(events):
    """Кодирование по Picky BPE: в слове выполняется самое раннее из возможных событий, затем следующее
    после него, пока возможных не останется. Без удалений это обычное кодирование BPE."""
    merge_at, remove_at, parts = defaultdict(list), defaultdict(list), {}
    for i, e in enumerate(events):
        if e[0] == "m":
            merge_at[(e[1], e[2])].append(i)
            parts[e[3]] = (e[1], e[2])
        else:
            remove_at[e[1]].append(i)

    def after(lst, last):
        return next((i for i in lst if i > last), None)

    cache = {}

    def encode(word):
        if word in cache:
            return cache[word]
        w, last = list(word), -1
        while True:
            cands = [after(merge_at.get(p, ()), last) for p in zip(w, w[1:])]
            cands += [after(remove_at.get(t, ()), last) for t in w]
            cands = [i for i in cands if i is not None]
            if not cands:
                break
            last = min(cands)
            e = events[last]
            if e[0] == "m":
                w = make_merge(w, e)
            else:
                a, b = parts[e[1]]
                w = [y for t in w for y in ((a, b) if t == e[1] else (t,))]
        cache[word] = w
        return w

    def make_merge(w, e):
        r, k = [], 0
        while k < len(w):
            if k + 1 < len(w) and (w[k], w[k + 1]) == (e[1], e[2]):
                r.append(e[3])
                k += 2
            else:
                r.append(w[k])
                k += 1
        return r

    return encode


def picky_experiment(train_stories):
    val = booklm_stories("val")
    wc, wv = word_counts(train_stories), word_counts(val)
    val_bytes = sum(len(t.encode("utf-8")) for t in val)
    hf = load_tokenizer()
    hf_tokens = sum(len(e.ids) for e in hf.encode_batch(val))
    print(f"исходный токенизатор BookLM: {val_bytes / hf_tokens:.3f} байта на токен отложенных историй")
    result = {}
    for threshold in (None, 0.9, 0.7):
        events, active = train_bpe(wc, VOCAB - SPECIAL, threshold)
        encode = make_encoder(events)
        n_val = sum(len(encode(w)) * c for w, c in wv.items())
        used = Counter(t for w, c in wc.items() for t in encode(w))
        unused = sum(1 for t in active if t >= 256 and used[t] == 0)
        removals = sum(e[0] == "r" for e in events)
        result[threshold] = dict(tokens=n_val, removals=removals, unused=unused)
        name = "обычный BPE" if threshold is None else f"Picky BPE, порог {threshold}"
        print(f"{name:<22} удалений {removals:4}, склеек без единого вхождения {unused:3}, "
              f"{val_bytes / n_val:.3f} байта на токен, токенов {n_val:,}")
    base = result[None]["tokens"]
    for threshold in (0.9, 0.7):
        print(f"  порог {threshold}: токенов меньше на {1 - result[threshold]['tokens'] / base:.2%}")
    assert result[None]["removals"] == 0 and result[0.9]["removals"] == 558
    assert result[None]["tokens"] == 121_056 and result[0.9]["tokens"] == 120_346
    assert result[0.7]["removals"] == 1483 and result[0.7]["tokens"] == 119_882
    assert result[None]["unused"] == 64 and result[0.9]["unused"] == 0 and result[0.7]["unused"] == 0
    return result


if __name__ == "__main__":
    booklm_ios()
    stories = booklm_stories("train")
    qwen_ios(stories)
    embedding_shares()
    picky_experiment(stories)
