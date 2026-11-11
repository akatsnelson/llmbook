"""Глава 12 «Поиск по документам (RAG)»: голова InfoNCE и гибрид с BM25, матрёшечные векторы, граф героев.

Опыты (время — на процессоре ноутбука Apple M2 Max):
  ch12_infonce_hybrid  рубрики «модель представления» (раздел «Поиск по смыслу») и «гибридный поиск»;
                       около 15 с. На 500 отложенных парах «просьба -> история» из sft.jsonl:
                       Recall@1 и Recall@10 среднего вектора BookLM, головы 256 -> 128 до и после 300 шагов
                       InfoNCE, BM25 и RRF (c = 60) списков BM25 и головы; сколько историй нашёл только один
                       способ; сколько проверочных историй есть среди обучающих пар головы.
  ch12_matryoshka      рубрика «матрёшечные векторы»; около 35 с. Recall@10 при обрезке векторов головы
                       до 64, 32, 16 и 8 чисел у обычной головы, матрёшки и отдельной головы на каждую длину;
                       размер индекса из 10 млн векторов в ГБ (через 1024).
  ch12_hero_graph      рубрика «граф знаний», подраздел «Граф знаний»; около 5 с. Граф героев
                       по отложенным историям, вопросы в два перехода, Recall@10 у BM25, у разложения запроса
                       и у графа.
Кривую обучения той же головы печатает опыт ch12_booklm_embeddings из ch12.py.
Векторы считает базовая BookLM без LoRA, поэтому дообучение на sft.jsonl на опыты не влияет.

Запуск из корня репозитория:
    python -m booklm.experiments.ch12_dense_retrieval                    # все три опыта
    python -m booklm.experiments.ch12_dense_retrieval ch12_hero_graph    # один опыт
Вывод каждого опыта копируется в models/booklm/experiments/<имя опыта через дефис>.txt,
например ch12-matryoshka.txt. Строки со временем зависят от машины.
"""
import json
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

if not __package__:  # запуск файлом (python booklm/experiments/ch12_dense_retrieval.py): корень репозитория — в путь поиска
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from booklm.lab import MODELS, load
from booklm.experiments.ch12 import (OUT, bm25_indexed, by_rank, inverted_index, load_stories,
                                     story_vectors, terms)


def rrf(rankings, c=60):
    """Обратное объединение рангов, как в листинге раздела «Гибридный поиск»."""
    scores = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (c + rank)
    return sorted(scores.items(), key=lambda item: -item[1])


def recall(rankings, k):
    """Доля просьб n, у которых история n стоит среди первых k."""
    return sum(n in r[:k] for n, r in enumerate(rankings)) / len(rankings)


def ch12_infonce_hybrid():
    """Рубрики «модель представления» и «гибридный поиск»: Recall головы InfoNCE, BM25 и RRF."""
    t0 = time.time()
    tok, model = load("base")
    rows = [json.loads(s) for s in open(MODELS / "data" / "sft.jsonl", encoding="utf-8")]
    train, test = rows[:2000], rows[-500:]
    train_stories = {r["story"] for r in train}
    print(f"пар в sft.jsonl: {len(rows)}; обучение головы: {len(train)}, проверка: {len(test)}")
    print(f"проверочных историй, которые есть и среди обучающих пар: "
          f"{sum(r['story'] in train_stories for r in test)}")
    print(f"разных историй в проверке: {len({r['story'] for r in test})}")

    Q, D, Qt, Dt = (story_vectors(tok, model, [r[f] for r in part])
                    for part in (train, test) for f in ("request", "story"))
    t_vec = time.time() - t0

    def ranks(Qe, De, top=50):
        return (Qe @ De.T).topk(top, dim=1).indices.tolist()

    lists = {"BookLM без головы": ranks(Qt, Dt)}
    torch.manual_seed(0)
    head = torch.nn.Linear(256, 128, bias=False)
    opt = torch.optim.Adam(head.parameters(), lr=3e-3)
    enc = lambda X: F.normalize(head(X), dim=-1)
    with torch.no_grad():
        lists["голова до обучения"] = ranks(enc(Qt), enc(Dt))
    t1 = time.time()
    for step in range(300):          # как в листинге главы: 300 шагов, пакет 128, tau = 0.05
        k = torch.randint(0, len(Q), (128,))
        loss = F.cross_entropy(enc(Q[k]) @ enc(D[k]).T / 0.05, torch.arange(128))
        if step == 0:
            print(f"ошибка на первом шаге: {loss.item():.2f}, ln 128 = {math.log(128):.2f}")
        opt.zero_grad(); loss.backward(); opt.step()
    t_train = time.time() - t1
    with torch.no_grad():
        lists["голова после 300 шагов"] = ranks(enc(Qt), enc(Dt))

    ix, ln = inverted_index({n: r["story"] for n, r in enumerate(test)})
    lists["BM25"] = [by_rank(bm25_indexed(r["request"], ix, ln))[:50] for r in test]
    lists["RRF: BM25 + голова"] = [[d for d, _ in rrf([a, b])][:50]
                                   for a, b in zip(lists["BM25"], lists["голова после 300 шагов"])]
    lists["RRF: BM25 + BookLM без головы"] = [[d for d, _ in rrf([a, b])][:50]
                                             for a, b in zip(lists["BM25"], lists["BookLM без головы"])]

    print(f"{'способ':31} R@1   R@10")
    for name, r in lists.items():
        print(f"{name:31} {recall(r, 1):.2f}  {recall(r, 10):.2f}")

    bm, hd, hy = lists["BM25"], lists["голова после 300 шагов"], lists["RRF: BM25 + голова"]
    only_head = sum(n not in bm[n][:10] and n in hd[n][:10] for n in range(len(test)))
    only_bm = sum(n in bm[n][:10] and n not in hd[n][:10] for n in range(len(test)))
    lost = sum((n in bm[n][:10] or n in hd[n][:10]) and n not in hy[n][:10] for n in range(len(test)))
    print(f"в первой десятке только у головы: {only_head}, только у BM25: {only_bm}, "
          f"у кого-то из двух, но не у RRF: {lost}")
    print(f"время: векторы 5000 текстов {t_vec:.0f} с, 300 шагов головы {t_train:.1f} с (CPU)")

    # Пример: просьба, которую нашла голова, но не BM25
    for n in range(len(test)):
        if n not in bm[n][:10] and hd[n][0] == n:
            print("пример:", test[n]["request"][:90], "| место у BM25:",
                  (bm[n].index(n) + 1) if n in bm[n] else ">50", "| у головы: 1")
            break


# ----------------------------------------------------------------------------- матрёшечные векторы

def cut(H, m):
    """Первые m чисел каждой строки, снова единичной длины."""
    return F.normalize(H[:, :m], dim=-1)


def train_head(Q, D, dims, steps=300, out=128):
    """Голова out -> InfoNCE на каждом префиксе из dims (сумма ошибок), как в листинге главы.

    dims=(128,) — обычная голова из раздела «Поиск по смыслу»; dims=(128, 64, 32, 16) — матрёшка.
    Начальные веса и пакеты те же (torch.manual_seed(0)), отличается только ошибка.
    """
    torch.manual_seed(0)
    head = torch.nn.Linear(256, out, bias=False)
    opt = torch.optim.Adam(head.parameters(), lr=3e-3)
    for _ in range(steps):
        k = torch.randint(0, len(Q), (128,))
        hq, hd = head(Q[k]), head(D[k])
        loss = sum(F.cross_entropy(cut(hq, m) @ cut(hd, m).T / 0.05, torch.arange(128)) for m in dims)
        opt.zero_grad(); loss.backward(); opt.step()
    return head


def recall10(Qe, De):
    top = (Qe @ De.T).topk(10, dim=1).indices
    return float((top == torch.arange(len(De))[:, None]).any(1).float().mean())


def ch12_matryoshka():
    """Обрезка векторов головы InfoNCE до 64/32/16/8 чисел: обычная голова, матрёшка, отдельные головы."""
    tok, model = load("base")
    rows = [json.loads(s) for s in open(MODELS / "data" / "sft.jsonl", encoding="utf-8")]
    train, test = rows[:2000], rows[-500:]
    Q, D, Qt, Dt = (story_vectors(tok, model, [r[f] for r in part])
                    for part in (train, test) for f in ("request", "story"))
    t0 = time.time()
    plain = train_head(Q, D, (128,))
    t_plain = time.time() - t0
    t0 = time.time()
    mrl = train_head(Q, D, (128, 64, 32, 16))
    t_mrl = time.time() - t0
    own = {m: train_head(Q, D, (m,), out=m) for m in (64, 32, 16, 8)}
    print("Recall@10 на 500 отложенных просьбах; голова 256 -> 128, 300 шагов InfoNCE, пакет 128")
    print(f"{'чисел':>5} {'байт':>5}  обычная, обрезанная  матрёшка, обрезанная  отдельная голова на m")
    with torch.no_grad():
        for m in (128, 64, 32, 16, 8):
            sep = own[m] if m in own else plain
            print(f"{m:5} {4 * m:5}  {recall10(cut(plain(Qt), m), cut(plain(Dt), m)):20.2f}"
                  f"  {recall10(cut(mrl(Qt), m), cut(mrl(Dt), m)):20.2f}"
                  f"  {recall10(cut(sep(Qt), m), cut(sep(Dt), m)):21.2f}")
    print(f"время обучения: обычная {t_plain:.1f} с, матрёшка {t_mrl:.1f} с (CPU)")
    print(f"индекс из 10 млн векторов float32: 4096 чисел {4096 * 4 * 1e7 / 2**30:.0f} ГБ,"
          f" 768 чисел {768 * 4 * 1e7 / 2**30:.0f} ГБ, 256 чисел {256 * 4 * 1e7 / 2**30:.0f} ГБ")


# ----------------------------------------------------------------------------- граф героев и историй

NAME_END = r"(?:а|я|у|ю|ом|ем|ой|ей|е|ы|и)?"     # Тим, Тима, Тиму, Тимом, Тиме


def introduced(stories):
    """{номер истории: имена героев, введённых словами «по имени X»}."""
    return {k: set(re.findall(r"по имени ([А-ЯЁ][а-яё]+)", t)) for k, t in enumerate(stories)}


def hero_graph(stories, intro):
    """Рёбра «история — герой»: имя из словаря встречается в истории в любой из форм NAME_END.

    Узел истории — её номер (int), узел героя — имя (str). Одинаковые имена склеиваются в один узел.
    """
    names = sorted(set().union(*intro.values()))
    found = {k: {n for n in names if re.search(rf"\b{n}{NAME_END}\b", t)} for k, t in enumerate(stories)}
    return [(k, n) for k, hs in found.items() for n in sorted(hs)]


def bridge_questions(stories, index, intro, lo=2, hi=6):
    """Вопросы в два перехода: «в каких ещё историях есть герой с тем же именем, что в истории со словами …».

    Слова — три самых редких слова истории длиной от шести букв (кроме имени), как в name_questions.
    Ответ — другие истории, где герой с этим именем введён словами «по имени» (lo..hi историй).
    Возвращает [(вопрос, номер истории, имя, множество номеров-ответов)].
    """
    out = []
    for k, text in enumerate(stories):
        m = re.search(r"по имени ([А-ЯЁ][а-яё]+)", text)
        if not m:
            continue
        name = m.group(1)
        gold = {j for j, s in intro.items() if name in s and j != k}
        if not lo <= len(gold) <= hi:
            continue
        words = {w for w in re.findall(r"[а-яё]{6,}", text.lower()) if name.lower()[:4] not in w}
        rare = sorted(words, key=lambda w: (len(index.get(terms(w)[0], {})), w))[:3]
        out.append(("В каких ещё сказках есть герой с тем же именем, что у героя сказки со словами: "
                    + ", ".join(rare) + "?", k, name, gold))
    return out


def ppr(edges, seeds, alpha=0.5, steps=50):
    """Персонализированный PageRank из листинга раздела «Графовый поиск»."""
    nbr = {}
    for a, b in edges:
        nbr.setdefault(a, []).append(b)
        nbr.setdefault(b, []).append(a)
    e = {n: (1 / len(seeds) if n in seeds else 0.0) for n in nbr}
    pi = dict(e)
    for _ in range(steps):
        new = {n: (1 - alpha) * e[n] for n in nbr}
        for a in nbr:
            for b in nbr[a]:
                new[b] += alpha * pi[a] / len(nbr[a])
        pi = new
    return pi


def ch12_hero_graph():
    """Граф героев по 689 отложенным историям и вопрос в два перехода: BM25, разложение запроса, граф."""
    t0 = time.time()
    val = [t for part, t in load_stories() if part == "val"]
    idx, lens = inverted_index(dict(enumerate(val)))
    intro = introduced(val)
    graph = hero_graph(val, intro)
    deg = Counter(n for _, n in graph)
    with_heroes = {k for k, _ in graph}
    print(f"историй: {len(val)}, имён в словаре: {len(deg)}, рёбер: {len(graph)}, "
          f"историй без героев: {len(val) - len(with_heroes)}")
    print("самые связанные герои:", deg.most_common(5))
    print("редкие имена, склеенные с другими формами:",
          {n: sorted(m for m in deg if m != n and re.fullmatch(n + NAME_END, m))
           for n in deg if any(m != n and re.fullmatch(n + NAME_END, m) for m in deg)})

    qs = bridge_questions(val, idx, intro)
    print(f"вопросов: {len(qs)}, ответов на вопрос: от {min(len(g) for *_, g in qs)} до {max(len(g) for *_, g in qs)}")

    def first(q):
        return by_rank(bm25_indexed(q, idx, lens))[0]

    def decompose(q):
        top = first(q)
        return [d for d in by_rank(bm25_indexed(" ".join(sorted(intro[top])), idx, lens)) if d != top]

    def two_hops(q):
        top = first(q)
        pi = ppr(graph, {top})
        return [n for n in sorted(pi, key=lambda n: -pi[n]) if isinstance(n, int) and n != top]

    rec = lambda found, gold: len(gold & set(found[:10])) / len(gold)
    print(f"исходная история первой у BM25: {sum(first(q) == k for q, k, _, _ in qs)} из {len(qs)}")
    for how, f in (("BM25 одним запросом", lambda q: by_rank(bm25_indexed(q, idx, lens))),
                   ("разложение запроса", decompose), ("граф, PageRank", two_hops)):
        r = [rec(f(q), g) for q, _, _, g in qs]
        miss = Counter(n for (q, _, n, g), x in zip(qs, r) if x < 1)
        print(f"{how:20} Recall@10 = {sum(r) / len(r):.2f}, неполных ответов: {sum(x < 1 for x in r)}"
              f" {dict(miss)}")
    print("BM25 видит «Люсия» как терм:", terms("Люсия"), "| степень узлов:",
          {n: deg[n] for n in ("Люсия", "Люси", "Бен", "Эмма")})
    print(f"время: {time.time() - t0:.1f} с (CPU)")


# ----------------------------------------------------------------------------- запуск

EXPERIMENTS = {f.__name__: f for f in (ch12_infonce_hybrid, ch12_matryoshka, ch12_hero_graph)}


if __name__ == "__main__":
    from booklm.experiments import run
    OUT.mkdir(parents=True, exist_ok=True)
    run(EXPERIMENTS, save=True)      # копия вывода в models/booklm/experiments/<имя опыта>.txt
