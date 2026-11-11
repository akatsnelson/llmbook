"""Глава 11 «Контекст и память», рубрика «гниение контекста» (подраздел «Инженерия контекста в агентных
сессиях»): опыт ch11_context_rot на BookLM.

Факт (имя героя) стоит в первом предложении сказки. Между ним и последними 60 токенами перед обрывом
кладём L токенов чужих сказок («мусор») и смотрим, назовёт ли модель героя, сколько раз возьмёт имя
из чужой сказки и какова перплексия настоящего продолжения. Отдельно: тот же мусор плюс заметка
«Главного героя зовут …» прямо перед свежими токенами и окно без начала сказки.

Печатает таблицу по 40 сказкам: для чистого окна, мусора в 120, 240 и 360 токенов, мусора с заметкой
и окна без начала — среднюю длину входа, сколько раз модель назвала своего героя и чужого и перплексию
настоящего продолжения.

Запуск из корня репозитория (около 40 с на процессоре Apple M2 Max):
    python -m booklm.experiments.ch11_context_rot
Вывод сохраняется в models/booklm/experiments/ch11-context-rot.txt, таблица для рисунка — в ch11-context-rot.csv.
"""
import contextlib
import csv
import io
import math
import re
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

if not __package__:  # запуск файлом (python booklm/experiments/ch11_context_rot.py): корень репозитория — в путь поиска
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from booklm.lab import MODELS, load, special
from booklm.experiments.ch11 import cut_points, hero, stem, stories

OUT = MODELS / "experiments"
SPECIAL_MAX = 4                  # номера 0..4 — служебные токены BookLM


def first_sentence(tok, story):
    """Номера первого предложения сказки без <bos>: в нём названо имя героя."""
    for a in range(2, 80):
        if tok.decode([story[a - 1]]).endswith((".", "!", "?")):
            return story[1:a]
    return story[1:30]


_JUNK = None


def _junk_stories(tok):
    # чужие сказки: последние 300 отложенных, их cut_points не берёт (он идёт с начала списка)
    global _JUNK
    if _JUNK is None:
        _JUNK = []
        for s in stories()[-300:]:
            _JUNK.append(([i for i in s if i > SPECIAL_MAX], hero(tok, s)))
    return _JUNK


def junk(tok, j, n=360):
    """n токенов чужих сказок подряд, без служебных токенов, и список (позиция, имя героя).
    Для разных j берём разные сказки, чтобы мусор у каждой сказки опыта был свой."""
    pool = _junk_stories(tok)
    ids, heroes, k = [], [], (7 * j) % len(pool)
    while len(ids) < n:
        body, name = pool[k % len(pool)]
        if name:
            heroes.append((len(ids), name))
        ids += body
        k += 1
    return ids[:n], heroes


def judge(tok, model, ids, story, c, name, heroes):
    """Жадное продолжение входа ids на 60 токенов -> (назван ли свой герой; назван ли герой одной из
    чужих сказок, попавших в окно, которого нет в своей сказке; средняя -log p настоящих 40 токенов
    продолжения story[c:c+40])."""
    text = greedy(tok, model, ids)
    window, own = tok.decode(ids), tok.decode(story)
    others = {stem(h) for _, h in heroes
              if re.search(stem(h), window) and not re.search(stem(h), own)}
    return (bool(re.search(stem(name), text)), any(re.search(h, text) for h in others),
            tail_nll(model, ids, story[c:c + 40]))


def greedy(tok, model, ids, n=60):
    out = model.generate(torch.tensor([ids]), max_new=n, temperature=0, eos_id=2)
    return tok.decode(out[0, len(ids):].tolist())


@torch.no_grad()
def tail_nll(model, ids, cont):
    """Средняя перекрёстная энтропия настоящего продолжения cont после входа ids."""
    logits = model(torch.tensor([ids + cont]))[0].float()
    lp = F.log_softmax(logits[len(ids) - 1:-1], -1)
    return -lp[torch.arange(len(cont)), torch.tensor(cont)].mean().item()


def ch11_context_rot():
    """Тот же расчёт, что в листинге книги «Мусор в окне BookLM»."""
    tok, model = load("base")
    bos, K = special(tok)["<bos>"], 60
    tales, res = cut_points(tok, n=40, k=K), {}
    for j, (story, name, c) in enumerate(tales):
        head, recent = first_sentence(tok, story), story[c - K:c]
        trash, heroes = junk(tok, j)               # 360 токенов чужих сказок
        note = tok.encode(f"Главного героя зовут {name}.").ids
        inputs = {"чистое окно": [bos] + head + recent}
        for L in (120, 240, 360):
            inputs[f"мусор {L}"] = [bos] + head + trash[:L] + recent
        inputs["мусор 360 + заметка"] = [bos] + head + trash + note + recent
        inputs["без начала"] = [bos] + recent
        for kind, ids in inputs.items():
            res.setdefault(kind, []).append(
                (len(ids), *judge(tok, model, ids, story, c, name, heroes)))
    rows = []
    print(f"{'вход':20}{'токенов':>8}{'свой':>6}{'чужой':>7}{'перплексия':>12}")
    for kind, r in res.items():
        n, own, other, nll = (list(x) for x in zip(*r))
        ppl = math.exp(statistics.mean(nll))
        print(f"{kind:20}{statistics.mean(n):8.0f}{sum(own):6}{sum(other):7}{ppl:12.2f}")
        rows.append({"вход": kind, "токенов": round(statistics.mean(n)), "свой герой": sum(own),
                     "чужой герой": sum(other), "перплексия хвоста": round(ppl, 2)})
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rows = ch11_context_rot()
    text = buf.getvalue()
    print(text, end="")
    (OUT / "ch11-context-rot.txt").write_text(text, encoding="utf-8")
    with open(OUT / "ch11-context-rot.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    from booklm.experiments import pick
    pick({"ch11_context_rot": ch11_context_rot})   # --help и проверка имени
    main()                                         # txt и csv
