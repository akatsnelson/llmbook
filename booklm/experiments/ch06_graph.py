"""Глава 6, раздел «Графы как модальность», рубрика «А теперь на нашей модели: граф героев»: граф героев
отложенных сказок на обученной BookLM.

Узлы — имена героев, ребро — два имени встретились в одной фразе отложенной сказки (последние 689
историй корпуса, модель их не видела). Маленькая графовая сеть считает, сколько героев видит узел
за 1, 2 и 3 шага передачи сообщений. Затем граф подаём BookLM текстом («Лили и Бен друзья.») и смотрим,
какое имя она ставит после «Однажды Лили и», с графом в подсказке и без него.

Опыты (время — на процессоре Apple M2 Max):
  ch06_hero_graph — печатает размер графа, самые частые пары, охват узлов за 1–3 шага и три самых
      вероятных друга для Лили, Сью и Анны с графом в подсказке и без. Около 2 с.
      Копия вывода: ch06-hero-graph.txt.
  ch06_graph_text_curve — доля случаев, когда BookLM называет друга из подсказки, в зависимости от
      числа рёбер в тексте графа. Печатает таблицу CSV и сохраняет её в ch06-graph-text-curve.csv
      (данные рисунка). Около 20 с.

Запуск из корня репозитория:
    python -m booklm.experiments.ch06_graph                    # все опыты
    python -m booklm.experiments.ch06_graph ch06_hero_graph    # один опыт
    python -m booklm.experiments.ch06_graph --help             # список опытов
Файлы выводов лежат в models/booklm/experiments/.
"""
import collections
import itertools
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

if not __package__:  # запуск файлом (python booklm/experiments/ch06_graph.py): корень репозитория — в путь поиска
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from booklm.experiments import run
from booklm.lab import MODELS, load

OUT = MODELS / "experiments"
BOS = 1

# Имена героев и их падежные формы: «Тима», «Тиму», «Тимом» — тот же узел «Тим».
NAMES = {
    "Тим": "Тим(?:а|у|ом|е)?", "Том": "Том(?:а|у|ом|е)?", "Лили": "Лили", "Макс": "Макс(?:а|у|ом|е)?",
    "Сью": "Сью", "Спот": "Спот(?:а|у|ом|е)?", "Бен": "Бен(?:а|у|ом|е)?", "Боб": "Боб(?:а|у|ом|е)?",
    "Люси": "Люси", "Мия": "Ми(?:я|и|ю|ей)", "Салли": "Салли", "Джек": "Джек(?:а|у|ом|е)?",
    "Ами": "Ами", "Сэм": "Сэм(?:а|у|ом|е)?", "Флюффи": "Флюффи", "Джейн": "Джейн",
    "Китти": "Китти", "Анна": "Анн(?:а|ы|е|у|ой)", "Сара": "Сар(?:а|ы|е|у|ой)", "Билли": "Билли",
}
PATTERNS = {n: re.compile(r"(?<![А-Яа-яЁё])" + p + r"(?![А-Яа-яЁё])") for n, p in NAMES.items()}


# ----------------------------------------------------------------------------- граф героев

def held_out_stories(tok):
    """Тексты отложенных сказок: val.bin, разрезанный по <bos>."""
    data = np.fromfile(MODELS / "data" / "val.bin", dtype=np.uint16).tolist()
    stories, cur = [], []
    for i in data:
        if i == BOS:
            if cur:
                stories.append(cur)
            cur = []
        elif i > 2:                                   # без <pad> и <eos>
            cur.append(i)
    stories.append(cur)
    return [tok.decode(s) for s in stories]


def hero_graph(stories, min_count=5):
    """Узлы (по убыванию упоминаний) и рёбра {(a, b): сколько фраз}, где пара встретилась не меньше min_count раз."""
    edges, mentions = collections.Counter(), collections.Counter()
    for story in stories:
        for phrase in re.split(r"(?<=[.!?…])\s+|\n+", story):
            who = sorted(n for n, p in PATTERNS.items() if p.search(phrase))
            mentions.update(who)
            edges.update(itertools.combinations(who, 2))
    edges = {e: c for e, c in edges.items() if c >= min_count}
    nodes = sorted({n for e in edges for n in e}, key=lambda n: -mentions[n])
    return nodes, edges


def adjacency(nodes, edges):
    """Матрица смежности [N, N] из нулей и единиц."""
    index = {n: i for i, n in enumerate(nodes)}
    A = torch.zeros(len(nodes), len(nodes))
    for a, b in edges:
        A[index[a], index[b]] = A[index[b], index[a]] = 1.0
    return A


def reach(A, steps):
    """Сколько других узлов «слышит» каждый узел после steps шагов передачи сообщений h' = (A + I) h."""
    h = torch.eye(A.size(0))                          # признак узла — его собственная единица
    for _ in range(steps):
        h = ((A + torch.eye(A.size(0))) @ h).clamp(max=1)
    return h.sum(1).long() - 1


def edges_as_text(edges):
    return " ".join(f"{a} и {b} друзья." for a, b in edges)


# ----------------------------------------------------------------------------- BookLM выбирает имя

@torch.no_grad()
def name_scores(tok, model, context, names):
    """log p(« имя» | context) для каждого имени: один проход по пакету «контекст + имя»."""
    base = [BOS] + tok.encode(context).ids
    tails = [tok.encode(" " + n).ids for n in names]
    width = len(base) + max(map(len, tails))
    x = torch.zeros(len(names), width, dtype=torch.long)
    for r, t in enumerate(tails):
        x[r, :len(base) + len(t)] = torch.tensor(base + t)
    logp = F.log_softmax(model(x).float(), -1)
    return {n: sum(logp[r, len(base) - 1 + i, t].item() for i, t in enumerate(tails[r]))
            for r, n in enumerate(names)}


def friend_choice(tok, model, nodes, n_edges, trials=100, seed=0):
    """Доля случаев, где после «граф. Однажды u и» самым вероятным из всех имён графа стал друг u из подсказки.

    Граф каждый раз случайный: n_edges рёбер на тех же именах, так что подсказка обычно расходится с тем,
    что BookLM помнит по обучению. Спрашиваем про узел u, у которого в новом графе ровно один друг."""
    rng = random.Random(seed)
    hit_graph = hit_none = 0
    for _ in range(trials):
        while True:
            pairs = rng.sample(list(itertools.combinations(nodes, 2)), n_edges)
            deg = collections.Counter(n for e in pairs for n in e)
            lonely = [e for e in pairs if deg[e[0]] == 1 or deg[e[1]] == 1]
            if lonely:
                break
        a, b = rng.choice(lonely)
        u, v = (a, b) if deg[a] == 1 else (b, a)
        rng.shuffle(pairs)
        others = [n for n in nodes if n != u]
        for head, counter in ((edges_as_text(pairs) + "\n", "graph"), ("", "none")):
            s = name_scores(tok, model, head + f"Однажды {u} и", others)
            if max(s, key=s.get) == v:
                if counter == "graph":
                    hit_graph += 1
                else:
                    hit_none += 1
    return hit_graph / trials, hit_none / trials


# ----------------------------------------------------------------------------- опыты

def ch06_hero_graph():
    """Граф героев отложенных сказок, охват за 1–3 шага и друг героя с графом в подсказке и без."""
    tok, model = load("base")
    stories = held_out_stories(tok)
    nodes, edges = hero_graph(stories)
    print(f"{len(stories)} сказок, {len(nodes)} героев, {len(edges)} рёбер")
    for (a, b), c in sorted(edges.items(), key=lambda e: -e[1])[:6]:
        print(f"  {a} --- {b}: {c} фраз")
    A = adjacency(nodes, edges)
    r1, r2, r3 = reach(A, 1), reach(A, 2), reach(A, 3)
    for i, n in enumerate(nodes):
        print(f"{n:7} 1 шаг: {r1[i]:2d}, 2 шага: {r2[i]:2d}, 3 шага: {r3[i]:2d}")
    text = edges_as_text(edges)
    print("токенов в тексте графа:", len(tok.encode(text).ids))
    for u in ("Лили", "Сью", "Анна"):
        for head in ("", text + "\n"):
            s = name_scores(tok, model, head + f"Однажды {u} и", [n for n in nodes if n != u])
            best = sorted(s, key=s.get, reverse=True)[:3]
            print(f"{'с графом' if head else 'без графа'} {u}: {best}")


def ch06_graph_text_curve():
    """Как часто BookLM называет друга из подсказки при разном числе рёбер графа (CSV)."""
    tok, model = load("base")
    nodes, _ = hero_graph(held_out_stories(tok))
    print("edges,with_graph,without_graph")
    for m in (1, 2, 4, 8, 16, 30):
        g, n = friend_choice(tok, model, nodes, m, trials=100, seed=m)
        print(f"{m},{g:.2f},{n:.2f}")


EXPERIMENTS = {"ch06_hero_graph": ch06_hero_graph,
               "ch06_graph_text_curve": ch06_graph_text_curve}
SAVE = {"ch06_hero_graph": "ch06-hero-graph.txt",
        "ch06_graph_text_curve": "ch06-graph-text-curve.csv"}     # второй опыт печатает таблицу CSV

if __name__ == "__main__":
    run(EXPERIMENTS, save=SAVE)
