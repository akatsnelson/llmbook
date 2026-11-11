"""Глава 9 на обученной BookLM: контроль GRPO случайной наградой и петля «генерация -> проверка -> отбор
-> LoRA».

Опыты (время — на процессоре Apple M2 Max):
  ch09_grpo_random (старое имя random) — раздел «Прогон GRPO на BookLM», подраздел «Контроль случайной
      наградой». Тот же стенд и бюджет, что у ch09.py (LoRA r=8, группа 8, 4 просьбы на шаг, 80 шагов,
      beta=0.04), но награда — монетка с вероятностью 1/2, независимо от ответа; в журнал пишется
      и настоящая награда hero_reward, по которой политику не учат. Около 10–17 минут.
      Печатает строку журнала на каждом шаге; пишет только ch09-grpo-random.csv, txt не сохраняет.
  ch09_search_loop (старое имя loop) — раздел «Поиск с проверкой», рубрика «А теперь на нашей модели:
      отбор с отказом». Петля самоулучшения отбором (как STaR и отбор с отказом): 32 просьбы «сказка,
      в которой есть X и Y», по 8 сказок на просьбу, проверка, LoRA учится на прошедших проверку, три
      раунда. Два проверяющих: слабый (оба слова есть) и строгий (ещё <eos> не длиннее 300 токенов
      и повторённых предложений меньше 20 %). После каждого раунда pass@1 и pass@8 на 16 отложенных
      просьбах с героями и предметами, которых в обучении не было. Параметр: weak, strict, merge
      (сводка двух журналов) или all. Около 7 минут на проверяющего. Пишет
      ch09-search-loop-weak.csv, ch09-search-loop-strict.csv; merge — ch09-search-loop.csv и .txt.
  ch09_loop_pilot (старое имя pilot) — вспомогательный: доли прохождения проверок у исходной модели на
      обучающих и отложенных просьбах, по ним подбирались проверки. Вывод только на экран.
Функция ap_free_max — точный ответ для листинга «эволюционный поиск в миниатюре» того же раздела.

Запуск из корня репозитория:
    python -m booklm.experiments.ch09_random_reward_loop ch09_grpo_random
    python -m booklm.experiments.ch09_random_reward_loop ch09_search_loop weak     # около 7 минут
    python -m booklm.experiments.ch09_random_reward_loop ch09_search_loop strict   # можно параллельно с weak
    python -m booklm.experiments.ch09_random_reward_loop ch09_search_loop merge    # сводка двух журналов
    python -m booklm.experiments.ch09_random_reward_loop --help                    # список опытов
Старые команды тоже работают: random, loop weak, loop strict, loop merge, pilot.
Файлы выводов лежат в models/booklm/experiments/.
"""
import csv
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

if not __package__:  # запуск файлом (python booklm/experiments/ch09_random_reward_loop.py): корень репозитория — в путь поиска
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from booklm.experiments import run
from booklm.lab import MODELS, ROOT, prompt_ids

sys.path.insert(0, str(ROOT / "booklm"))
from sft_lora import set_lora  # noqa: E402
from booklm.experiments.ch09 import (REQUESTS, MAX_NEW, answer_batch, grpo_advantages,  # noqa: E402
                                     hero_reward, new_policy, sample_group)

OUT = MODELS / "experiments"


# ----------------------------------------------------------------------------- точный ответ для опыта А

def ap_free_max(N):
    """Наибольшее подмножество {0, ..., N-1} без трёх чисел в арифметической прогрессии: перебор с отсечением.
    Для листинга «эволюционный поиск в миниатюре»: N = 27 считается около секунды."""
    best = [0]

    def rec(i, chosen):
        if len(chosen) + (N - i) <= best[0]:
            return                                        # даже взяв все оставшиеся, рекорд не побить
        if i == N:
            best[0] = len(chosen)
            return
        if all(2 * b - a != i for a in chosen for b in chosen if a < b):
            chosen.append(i)
            rec(i + 1, chosen)
            chosen.pop()
        rec(i + 1, chosen)

    rec(0, [])
    return best[0]


# ----------------------------------------------------------------------------- контроль случайной наградой

def grpo_step_random(tok, policy, opt, batch, gen, coin, G=8, beta=0.04):
    """Шаг GRPO из ch09.grpo_step, но награда — монетка coin; настоящая награда только пишется в журнал."""
    log = defaultdict(list)
    opt.zero_grad()
    for request, hero in batch:
        ids = prompt_ids(tok, request, chat=True)
        group = sample_group(policy, ids, G, MAX_NEW, gen)
        r = (torch.rand(G, generator=coin) < 0.5).float()      # случайная награда
        true = [hero_reward(tok, a, e, hero) for a, e in group]
        adv = grpo_advantages(r)
        x, mask = answer_batch(ids, group)
        logp = policy(x[:, :-1]).log_softmax(-1)
        with torch.no_grad():
            set_lora(policy, False)
            ref = policy(x[:, :-1]).log_softmax(-1)
            set_lora(policy, True)
        tok_logp = logp.gather(-1, x[:, 1:, None])[..., 0]
        kl = (logp.exp() * (logp - ref)).sum(-1)
        n = mask.sum(1)
        pg = -(adv * (tok_logp * mask).sum(1) / n).mean()
        kl_mean = ((kl * mask).sum(1) / n).mean()
        ((pg + beta * kl_mean) / len(batch)).backward()
        log["reward"] += true
        log["coin"] += r.tolist()
        log["kl"].append(kl_mean.item())
        log["length"] += n.tolist()
        log["eos"] += [float(e) for _, e in group]
        log["hero"] += [float(bool(re.search(hero, tok.decode(a).lower()))) for a, _ in group]
        log["dead"].append(float(r.std() == 0))
    opt.step()
    return {k: sum(v) / len(v) for k, v in log.items()}


def ch09_grpo_random(steps=80, per_step=4, lr=2e-3, beta=0.04):
    """Контрольный прогон GRPO, где награда — монетка; настоящая награда только в журнале."""
    tok, policy, params = new_policy()
    opt = torch.optim.Adam(params, lr=lr)
    gen = torch.Generator().manual_seed(0)
    coin = torch.Generator().manual_seed(123)
    fields = ["step", "reward", "coin", "kl", "length", "eos", "hero", "dead", "seconds"]
    rows, start = [], time.time()
    for step in range(steps):
        k = (step * per_step) % len(REQUESTS)
        s = grpo_step_random(tok, policy, opt, REQUESTS[k:k + per_step], gen, coin, beta=beta)
        s["step"], s["seconds"] = step, time.time() - start
        rows.append(s)
        print(f"шаг {step:3d}: награда {s['reward']:.3f}  KL {s['kl']:.4f}  длина {s['length']:6.1f}  "
              f"герой {s['hero']:.2f}  {s['seconds']:.0f} с", flush=True)
    window = 5
    for i, row in enumerate(rows):
        part = rows[max(0, i - window + 1):i + 1]
        for key in ("reward", "kl", "length", "hero"):
            row[key + "_avg"] = sum(r[key] for r in part) / len(part)
    fields += [k + "_avg" for k in ("reward", "kl", "length", "hero")]
    with open(OUT / "ch09-grpo-random.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for s in rows:
            w.writerow({k: (f"{s[k]:.5f}" if isinstance(s[k], float) else s[k]) for k in fields})


# ----------------------------------------------------------------------------- петля отбора с LoRA

# Герой (именительный падеж) и образец, по которому он ищется в любой форме; образцы — из ch09.py.
TRAIN_HEROES = {
    "кот": r"\bкот(а|у|ом|е|ы|ов|ик\w*|ён\w*)?\b", "собака": r"\bсобак", "девочка": r"\bдевоч",
    "лиса": r"\bлис(а|у|ы|е|ой|иц\w*|ичк\w*|ён\w*)\b", "медведь": r"\bмедвед|\bмишк",
    "ёжик": r"\b[её]ж(ик|ич)", "заяц": r"\bза(яц|йц|йк|йч)", "черепаха": r"\bчерепах",
    "дракон": r"\bдракон", "слон": r"\bслон", "мышка": r"\bмыш", "птичка": r"\bптич|\bптиц",
    "рыбка": r"\bрыб", "бабушка": r"\bбабушк", "мальчик": r"\bмальчик", "воздушный змей": r"\bзме[йяюе]",
}
TRAIN_THINGS = {
    "мяч": r"\bмяч", "цветок": r"\bцвет(ок|к|оч)", "книга": r"\bкни(г|ж)", "торт": r"\bторт",
    "лодка": r"\bлод(к|оч)", "шарик": r"\bшар", "яблоко": r"\bяблок", "дерево": r"\bдерев",
}
TEST_HEROES = {"волк": r"\bвол(к|чо|чи)", "корова": r"\bкоров", "белка": r"\bбел(к|оч)",
               "петух": r"\bпет(ух|уш)"}
TEST_THINGS = {"звезда": r"\bзв[её]зд", "машинка": r"\bмашин", "дом": r"\bдом(а|у|ом|е|ик\w*)?\b",
               "ключ": r"\bключ"}


def pair_requests(heroes, things, per_hero):
    """Просьбы «Напиши сказку, в которой есть X и Y.»: каждому герою per_hero предметов по кругу."""
    hs, ts = list(heroes.items()), list(things.items())
    out = []
    for i, (h, hp) in enumerate(hs):
        for j in range(per_hero):
            t, tp = ts[(i + 3 * j) % len(ts)]
            out.append((f"Напиши сказку, в которой есть {h} и {t}.", [hp, tp]))
    return out


TRAIN = pair_requests(TRAIN_HEROES, TRAIN_THINGS, 2)       # 32 просьбы
TEST = pair_requests(TEST_HEROES, TEST_THINGS, 4)          # 16 просьб, все пары 4×4


def repeated_share(text):
    """Доля предложений, которые уже встречались в этой сказке раньше (без учёта регистра)."""
    sents = [s.strip().lower() for s in re.split(r"(?<=[.!?…])\s+", text) if s.strip()]
    seen, rep = set(), 0
    for s in sents:
        rep += s in seen
        seen.add(s)
    return rep / max(len(sents), 1)


def weak_check(text, ended, n, words):
    return all(re.search(w, text.lower()) for w in words)


def strict_check(text, ended, n, words):
    return weak_check(text, ended, n, words) and ended and n + 1 <= 300 and repeated_share(text) < 0.2


@torch.no_grad()
def evaluate(tok, policy, requests, gen, G=8):
    """pass@1 и pass@8 по обоим проверяющим, доля повторённых предложений и длина на отложенных просьбах."""
    st = defaultdict(list)
    for request, words in requests:
        ids = prompt_ids(tok, request, chat=True)
        weak, strict = [], []
        for a, e in sample_group(policy, ids, G, MAX_NEW, gen):
            text = tok.decode(a)
            weak.append(weak_check(text, e, len(a), words))
            strict.append(strict_check(text, e, len(a), words))
            st["rep"].append(repeated_share(text))
            st["len"].append(len(a) + e)
        st["weak1"].append(sum(weak) / G)
        st["weak8"].append(float(any(weak)))
        st["strict1"].append(sum(strict) / G)
        st["strict8"].append(float(any(strict)))
    return {k: sum(v) / len(v) for k, v in st.items()}


def sft_on(tok, policy, opt, kept, epochs=2, bsz=8, seed=0):
    """Несколько эпох SFT на отобранных ответах: ошибка только на токенах ответа и <eos>."""
    rng = random.Random(seed)
    data = list(kept)
    losses = []
    for _ in range(epochs):
        rng.shuffle(data)
        for i in range(0, len(data), bsz):
            batch = data[i:i + bsz]
            rows = [(ids, [(a, e)]) for ids, a, e in batch]
            xs, ms = zip(*(answer_batch(ids, g) for ids, g in rows))
            T = max(x.size(1) for x in xs)
            x = torch.cat([F.pad(x, (0, T - x.size(1))) for x in xs])
            m = torch.cat([F.pad(mm, (0, T - 1 - mm.size(1))) for mm in ms])
            logp = policy(x[:, :-1]).log_softmax(-1).gather(-1, x[:, 1:, None])[..., 0]
            loss = -(logp * m).sum() / m.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
    return sum(losses) / max(len(losses), 1)


CHECKERS = {"weak": ("слабый", weak_check), "strict": ("строгий", strict_check)}
LOOP_FIELDS = ["checker", "round", "kept", "weak1", "weak8", "strict1", "strict8", "rep", "len", "seconds"]


def loop_line(m):
    return (f"{m['checker']:8s} раунд {m['round']}: отобрано {int(m['kept']):3d} из {len(TRAIN) * 8}; "
            f"слабая проверка pass@1 {float(m['weak1']):.2f} pass@8 {float(m['weak8']):.2f}; "
            f"строгая pass@1 {float(m['strict1']):.2f} pass@8 {float(m['strict8']):.2f}; "
            f"повторов {float(m['rep']):.2f}, длина {float(m['len']):.0f}; {float(m['seconds']):.0f} с")


def ch09_search_loop(key, rounds=3, G=8, lr=2e-3, epochs=3):
    """Петля для одного проверяющего (key: weak или strict); журнал в ch09-search-loop-<key>.csv."""
    name, check = CHECKERS[key]
    rows, start = [], time.time()
    tok, policy, params = new_policy(seed=0)
    opt = torch.optim.Adam(params, lr=lr)
    gen = torch.Generator().manual_seed(0)
    for rnd in range(rounds + 1):
        kept = 0
        if rnd > 0:                                       # генерация -> проверка -> отбор -> LoRA
            pool = []
            for request, words in TRAIN:
                ids = prompt_ids(tok, request, chat=True)
                for a, e in sample_group(policy, ids, G, MAX_NEW, gen):
                    if check(tok.decode(a), e, len(a), words):
                        pool.append((ids, a, e))
            kept = len(pool)
            if pool:
                sft_on(tok, policy, opt, pool, epochs=epochs, seed=rnd)
        m = evaluate(tok, policy, TEST, torch.Generator().manual_seed(100 + rnd))
        m.update(checker=name, round=rnd, kept=kept, seconds=time.time() - start)
        rows.append(m)
        print(loop_line(m), flush=True)
    with open(OUT / f"ch09-search-loop-{key}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOOP_FIELDS)
        w.writeheader()
        for s in rows:
            w.writerow({k: (f"{s[k]:.5f}" if isinstance(s[k], float) else s[k]) for k in LOOP_FIELDS})


def merge_loop():
    """Сводит журналы двух проверяющих в ch09-search-loop.csv и ch09-search-loop.txt."""
    rows = []
    for key in CHECKERS:
        rows += list(csv.DictReader(open(OUT / f"ch09-search-loop-{key}.csv")))
    with open(OUT / "ch09-search-loop.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOOP_FIELDS)
        w.writeheader()
        w.writerows(rows)
    text = "\n".join(loop_line(r) for r in rows)
    print(text)
    (OUT / "ch09-search-loop.txt").write_text(text + "\n", encoding="utf-8")


@torch.no_grad()
def pilot(G=8):
    """Доли прохождения проверок у инструктивной BookLM на обучающих и отложенных просьбах (для подбора)."""
    tok, policy, _ = new_policy()
    gen = torch.Generator().manual_seed(0)
    for title, reqs in (("обучение", TRAIN), ("отложенные", TEST)):
        print(title, evaluate(tok, policy, reqs, gen, G))


def search_loop_command(key="all"):
    """Петля отбора: key = weak, strict, merge (сводка двух журналов) или all (всё по очереди)."""
    for k in (["weak", "strict", "merge"] if key == "all" else [key]):
        if k == "merge":
            merge_loop()
        else:
            ch09_search_loop(k)


EXPERIMENTS = {"ch09_grpo_random": ch09_grpo_random, "random": ch09_grpo_random,
               "ch09_search_loop": search_loop_command, "loop": search_loop_command,
               "ch09_loop_pilot": pilot, "pilot": pilot}

if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    run(EXPERIMENTS, args=True)          # «loop weak» = «ch09_search_loop weak»
