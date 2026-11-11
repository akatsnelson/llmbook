"""Глава 17 «Выпуск и сопровождение модели», рубрика «регрессия на стороне сервера» (подраздел «Регрессии
качества при неизменных весах»): сервер портит выборку, а веса прежние.

Сервер v2 отличается от v1 только функцией выборки токена: в доле шагов `broken` «приближённый» отбор
top-k возвращает не 40 лучших токенов, а 40 случайных (так в августе 2025 года ошибался приближённый
top-k на TPU, см. Anthropic, «A postmortem of three recent issues», 17.09.2025). Веса одни и те же.
Мы сравниваем то, что видит канарейка: время ответа, жадные теневые ответы и ответы в рабочем режиме
выборки (температура 0,8, top-k 40), которые оценивает сама модель.

Опыты (время — на процессоре ноутбука Apple M2 Max):
  ch17_server_regression  около 5 мин. Сначала листинг книги «Сломанный отбор top-k на сервере
                          BookLM»: жадная тень на 20 просьбах, перплексия, доля странных токенов, доля ответов
                          с ними и distinct-2 у v1 и v2 на 100 просьбах. Затем полный прогон: 200 просьб,
                          доля сломанных шагов 0, 0,01, 0,02, 0,05 (время ответа, те же меры, сколько ответов
                          не как у v1) и пример ответа. Вывод — ch17-server-regression.txt, таблица полного
                          прогона — ch17-server-regression.csv.
  ch17_fault_overlap      около 20 с. Проверка разбора в книге: в скольких из 100 ответов выпал сломанный
                          шаг и сколько из них получили странный токен. Пишет ch17-server-regression-overlap.txt.

Запуск из корня репозитория:
    python -m booklm.experiments.ch17_server_regression                      # оба опыта
    python -m booklm.experiments.ch17_server_regression ch17_fault_overlap   # один (старое имя: overlap)
"""
import json
import random
import sys
import time
from pathlib import Path

import torch

if not __package__:  # запуск файлом (python booklm/experiments/ch17_server_regression.py): корень репозитория — в путь поиска
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from booklm.lab import MODELS, load, prompt_ids

OUT = MODELS / "experiments"


@torch.no_grad()
def serve(model, ids, n=60, temperature=0.8, top_k=40, broken=0.0, seed=0):
    """Ответ сервера на n токенов. broken — доля шагов, где отбор top-k возвращает случайные токены."""
    gen = torch.Generator().manual_seed(seed)        # поток выборки: одинаков у v1 и v2
    fault = random.Random(seed)                      # отдельный поток для поломки
    caches = [{"len": 0} for _ in model.blocks]
    logits = model(torch.tensor([ids]), caches)
    for c in caches:
        c["len"] = len(ids)
    out = []
    for _ in range(n):
        last = logits[0, -1].float()
        if temperature == 0:
            nxt = int(last.argmax())                 # жадный путь: отбор top-k не вызывается
        else:
            last = last / temperature
            cand = last.topk(top_k).indices          # точный отбор
            if fault.random() < broken:              # «приближённый» отбор ошибся
                cand = torch.randperm(last.numel(), generator=fault_gen(fault))[:top_k]
            p = last[cand].softmax(-1)
            nxt = int(cand[torch.multinomial(p, 1, generator=gen)])
        out.append(nxt)
        logits = model(torch.tensor([[nxt]]), caches)
        for c in caches:
            c["len"] += 1
    return out


def fault_gen(fault):
    return torch.Generator().manual_seed(fault.randrange(2**31))


@torch.no_grad()
def token_probs(model, ids, answer):
    """Вероятность каждого токена ответа по оценке самой модели (температура 1, без отбора)."""
    x = torch.tensor([ids + answer])
    p = model(x)[0, len(ids) - 1:-1].float().softmax(-1)
    return p[torch.arange(len(answer)), torch.tensor(answer)]


def distinct2(answers):
    pairs = [tuple(a[i:i + 2]) for a in answers for i in range(len(a) - 1)]
    return len(set(pairs)) / len(pairs)


def compare(tok, model, asks, broken, n=60, weird=1e-3):
    """Сводка канарейки для сервера с данной долей сломанных шагов."""
    t0 = time.perf_counter()
    answers, probs = [], []
    for i, a in enumerate(asks):
        ids = prompt_ids(tok, a, chat=True)
        ans = serve(model, ids, n=n, broken=broken, seed=i)
        answers.append(ans)
        probs.append(token_probs(model, ids, ans))
    sec = (time.perf_counter() - t0) / len(asks)
    p = torch.cat(probs)
    return {"с на ответ": sec,
            "перплексия ответов": float(torch.exp(-p.log().mean())),
            "странных токенов, %": 100 * float((p < weird).float().mean()),
            "ответов со странным, %": 100 * sum(bool((q < weird).any()) for q in probs) / len(asks),
            "distinct-2": distinct2(answers)}, answers


def ch17_server_regression():
    """Листинг книги «Сломанный отбор top-k на сервере BookLM», затем полный прогон на 200 просьбах.

    Листинг: 100 просьб, v1 и v2. Полный прогон: четыре доли сломанных шагов, таблица в csv, пример ответа."""
    tok, model = load("instruct")
    with open(MODELS / "data" / "sft.jsonl", encoding="utf-8") as f:
        asks = [json.loads(line)["request"] for line in f][:200]
    shadow = 0
    for ask_text in asks[:20]:                       # те же 20 просьб, что в листинге три проверки
        ids = prompt_ids(tok, ask_text, chat=True)
        shadow += (serve(model, ids, n=10, temperature=0) ==
                   serve(model, ids, n=10, temperature=0, broken=0.01))
    print(f"жадная тень: совпали {shadow} из 20")
    for name, broken in (("v1", 0.0), ("v2", 0.01)):
        m, _ = compare(tok, model, asks[:100], broken)
        print(f"{name}: перплексия {m['перплексия ответов']:.2f}, "
              f"странных {m['странных токенов, %']:.2f}%, "
              f"ответов с ними {m['ответов со странным, %']:.0f}%, "
              f"distinct-2 {m['distinct-2']:.3f}")

    print("\nполный прогон, 200 просьб:")
    rows = []
    base, base_answers = compare(tok, model, asks, 0.0)
    for broken in (0.0, 0.01, 0.02, 0.05):
        m, answers = (base, base_answers) if broken == 0 else compare(tok, model, asks, broken)
        diff = sum(a != b for a, b in zip(answers, base_answers))
        m["ответов не как у v1, %"] = 100 * diff / len(asks)
        rows.append((broken, m))
        print(f"broken={broken:.2f}  " + "  ".join(f"{k} {v:.3f}" for k, v in m.items()))
    ids = prompt_ids(tok, asks[0], chat=True)
    for broken in (0.0, 0.05):
        print(f"пример broken={broken}:", tok.decode(serve(model, ids, broken=broken, seed=0)))
    with open(OUT / "ch17-server-regression.csv", "w", encoding="utf-8") as f:
        keys = list(rows[0][1])
        f.write("broken," + ",".join(k.replace(", %", " %") for k in keys) + "\n")
        for broken, m in rows:
            f.write(f"{broken}," + ",".join(f"{m[k]:.4f}" for k in keys) + "\n")


def ch17_fault_overlap(n_asks=100, broken=0.01):
    """Сколько ответов получили сломанный шаг и сколько из них — странный токен (проверка разбора в книге)."""
    tok, model = load("instruct")
    with open(MODELS / "data" / "sft.jsonl", encoding="utf-8") as f:
        asks = [json.loads(line)["request"] for line in f][:n_asks]
    fired = weird_v1 = weird_v2 = fired_weird = 0
    for i, a in enumerate(asks):
        ids = prompt_ids(tok, a, chat=True)
        r = random.Random(i)                          # тот же поток поломки, что в serve(seed=i)
        f = any(r.random() < broken for _ in range(60))
        w1 = bool((token_probs(model, ids, serve(model, ids, seed=i)) < 1e-3).any())
        w2 = bool((token_probs(model, ids, serve(model, ids, broken=broken, seed=i)) < 1e-3).any())
        fired += f
        weird_v1 += w1
        weird_v2 += w2
        fired_weird += f and w2
    text = (f"ответов: {n_asks}, broken={broken}\n"
            f"сломанный шаг выпал: {fired}\n"
            f"странный токен у v1: {weird_v1}, у v2: {weird_v2}\n"
            f"сломанный шаг и странный токен: {fired_weird}\n"
            f"странный токен без сломанного шага: {weird_v2 - fired_weird}\n")
    print(text, end="")
    (OUT / "ch17-server-regression-overlap.txt").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    from booklm.experiments import run
    run({"ch17_server_regression": ch17_server_regression,
         "ch17_fault_overlap": ch17_fault_overlap, "overlap": ch17_fault_overlap},
        save={"ch17_server_regression": "ch17-server-regression.txt"})
