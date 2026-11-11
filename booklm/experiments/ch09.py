"""Глава 9, раздел «Прогон GRPO на BookLM»: группа GRPO, настоящий прогон GRPO и расхождение движков.

Прогон GRPO. Политика — инструктивная BookLM с новым адаптером LoRA r=8, эталон — та же модель
с выключенным адаптером. Награда проверяется программой: герой из просьбы есть в сказке, и сказка
закончилась <eos> не длиннее LIMIT токенов. Группа из G=8 ответов на просьбу, 4 просьбы на шаг, один шаг
градиента на группу (ρ = 1), 80 шагов.

Опыты (время — на процессоре Apple M2 Max):
  ch09_grpo_group (старое имя group) — подраздел «Проверяющая функция для сказки». Восемь ответов на
      просьбу «кот и мяч», их награды и преимущества GRPO, затем ещё две просьбы про ёжика.
      Около 25 с. Вывод только на экран.
  ch09_grpo_run — подразделы «Шаг GRPO с адаптером LoRA» и «Журнал прогона». Сам прогон: печатает
      строку журнала на каждом шаге. Около 10 минут. Пишет ch09-grpo-run.csv (журнал по шагам со
      скользящими средними), ch09-grpo-samples.json (жадные ответы на шагах 0, 40 и 79) и адаптер
      ch09-grpo-adapter.pt; печать прогона в txt не сохраняет.
      Параметры по умолчанию: steps=80 per_step=4 lr=0.002 beta=0.04.
  ch09_grpo_smooth (старое имя smooth) — пересчитывает скользящие средние в ch09-grpo-run.csv
      (прогон делает это сам). Меньше секунды.
  ch09_grpo_compare (старое имя compare) — подраздел «Сказки до и после прогона». Награда, сколько раз
      назван герой и длина до прогона и после: 16 просьб прогона и 4 новых героя. Нужен адаптер
      прогона. Пишет ch09-grpo-compare.txt.
  ch09_grpo_hacking (старое имя hacking) — тот же подраздел: живой случай взлома награды, жадные сказки
      на шагах 0 и 79 и счёт упоминаний героя. Ничего не пересчитывает, читает ch09-grpo-samples.json
      и ch09-grpo-compare.txt. Около 1 с. Пишет ch09-grpo-hacking.txt.
  ch09_rollout_mismatch (старое имя mismatch) — раздел «Многоходовое обучение с подкреплением»,
      рубрика «А теперь на нашей модели: расхождение движков». Ответы пишет копия в BF16 или с весами
      в 4 бита, вероятности считает модель в FP32; печатает разброс отношения вероятностей.
      Около 35 с. Пишет ch09-rollout-mismatch.txt.

Запуск из корня репозитория:
    python -m booklm.experiments.ch09                         # все опыты по очереди
    python -m booklm.experiments.ch09 ch09_grpo_group         # один опыт
    python -m booklm.experiments.ch09 ch09_grpo_run steps=20  # короткий прогон (перезапишет файлы прогона)
    python -m booklm.experiments.ch09 --help                  # список опытов
Файлы выводов лежат в models/booklm/experiments/.
"""
import csv
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

if not __package__:  # запуск файлом (python booklm/experiments/ch09.py): корень репозитория — в путь поиска
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from booklm.experiments import run
from booklm.lab import MODELS, ROOT, load, prompt_ids, special

sys.path.insert(0, str(ROOT / "booklm"))       # sft_lora.py импортирует model без пакета
from sft_lora import add_lora, set_lora          # noqa: E402
from booklm.experiments.ch08 import sample, short  # noqa: E402  генерация с зерном из главы 8

OUT = MODELS / "experiments"
LIMIT = 150                                      # предел длины сказки с <eos>, токенов
MAX_NEW = 300                                    # дальше генерация обрывается

# Просьба «Напиши сказку про …» и образец, по которому герой ищется в ответе в любой форме.
HEROES = {
    "кота": r"\bкот(а|у|ом|е|ы|ов|ик\w*|ён\w*)?\b",
    "собаку": r"\bсобак",
    "девочку": r"\bдевоч",
    "лису": r"\bлис(а|у|ы|е|ой|иц\w*|ичк\w*|ён\w*)\b",
    "медведя": r"\bмедвед|\bмишк",
    "ёжика": r"\b[её]ж(ик|ич)",
    "зайца": r"\bза(яц|йц|йк|йч)",
    "черепаху": r"\bчерепах",
    "дракона": r"\bдракон",
    "слона": r"\bслон",
    "мышку": r"\bмыш",
    "птичку": r"\bптич|\bптиц",
    "рыбку": r"\bрыб",
    "бабушку": r"\bбабушк",
    "мальчика": r"\bмальчик",
    "воздушного змея": r"\bзме[йяюе]",
}
REQUESTS = [(f"Напиши сказку про {h}.", p) for h, p in HEROES.items()]


@torch.no_grad()
def sample_group(model, ids, G, max_new, gen):
    """G ответов на одну просьбу одним пакетом с KV-кэшем; temperature=1, без top_k.
    Возвращает [(новые токены без <eos>, закончился ли <eos>)] — как sample() из ch08.py."""
    eos = 2
    x = torch.tensor([ids] * G)
    caches = [{"len": 0} for _ in model.blocks]
    logits = model(x, caches)
    for c in caches:
        c["len"] = x.size(1)
    out, done = [], torch.zeros(G, dtype=torch.bool)
    for _ in range(max_new):
        nxt = torch.multinomial(logits[:, -1].float().softmax(-1), 1, generator=gen)
        nxt[done] = eos                                  # закончившие ответы стоят на месте
        out.append(nxt)
        done |= nxt[:, 0] == eos
        if done.all():
            break
        logits = model(nxt, caches)
        for c in caches:
            c["len"] += 1
    rows = torch.cat(out, 1).tolist()
    return [(r[:r.index(eos)], True) if eos in r else (r, False) for r in rows]


def answer_batch(ids, group):
    """Строки «просьба + ответ (+<eos>)», дополненные <pad>=0 справа, и маска целей ответа [G, T-1]."""
    rows = [ids + a + ([2] if e else []) for a, e in group]
    T = max(map(len, rows))
    x = torch.tensor([r + [0] * (T - len(r)) for r in rows])
    mask = torch.zeros(len(rows), T - 1)
    for i, r in enumerate(rows):
        mask[i, len(ids) - 1:len(r) - 1] = 1             # цели: токены ответа и <eos>
    return x, mask


def hero_reward(tok, new, ended, hero):
    """1, если герой просьбы есть в сказке и она закончилась <eos> не длиннее LIMIT токенов."""
    found = re.search(hero, tok.decode(new).lower())
    return float(bool(found) and ended and len(new) + 1 <= LIMIT)


def grpo_advantages(rewards, eps=1e-6):
    mean = rewards.mean(dim=-1, keepdim=True)
    std = rewards.std(dim=-1, correction=0, keepdim=True)
    return (rewards - mean) / (std + eps)


def grpo_step(tok, policy, opt, batch, gen, G=8, beta=0.04):
    """Один шаг GRPO по нескольким просьбам: генерация, награда, преимущества, ошибка, шаг Adam."""
    log = defaultdict(list)
    opt.zero_grad()
    for request, hero in batch:
        ids = prompt_ids(tok, request, chat=True)
        group = sample_group(policy, ids, G, MAX_NEW, gen)
        r = torch.tensor([hero_reward(tok, a, e, hero) for a, e in group])
        adv = grpo_advantages(r)                            # [G]
        x, mask = answer_batch(ids, group)
        logp = policy(x[:, :-1]).log_softmax(-1)            # [G, T-1, V]
        with torch.no_grad():                               # эталон: адаптер выключен
            set_lora(policy, False)
            ref = policy(x[:, :-1]).log_softmax(-1)
            set_lora(policy, True)
        tok_logp = logp.gather(-1, x[:, 1:, None])[..., 0]  # log pi(y_t)
        kl = (logp.exp() * (logp - ref)).sum(-1)            # KL по словарю, [G, T-1]
        n = mask.sum(1)
        pg = -(adv * (tok_logp * mask).sum(1) / n).mean()
        kl_mean = ((kl * mask).sum(1) / n).mean()
        ((pg + beta * kl_mean) / len(batch)).backward()
        log["reward"] += r.tolist()
        log["kl"].append(kl_mean.item())
        log["length"] += n.tolist()
        log["eos"] += [float(e) for _, e in group]
        log["hero"] += [float(bool(re.search(hero, tok.decode(a).lower()))) for a, _ in group]
        log["dead"].append(float(r.std() == 0))
    opt.step()
    return {k: sum(v) / len(v) for k, v in log.items()}


def new_policy(r=8, alpha=16, seed=0):
    tok, policy = load("instruct")
    torch.manual_seed(seed)                              # случайная A адаптера
    add_lora(policy, r, alpha)
    params = [p for p in policy.parameters() if p.requires_grad]
    return tok, policy, params


@torch.no_grad()
def greedy_answers(tok, model, requests, n_words=40):
    out = []
    for request, _ in requests:
        ids = prompt_ids(tok, request, chat=True)
        y = model.generate(torch.tensor([ids]), max_new=MAX_NEW, temperature=0, eos_id=special(tok)["<eos>"])
        new = y[0, len(ids):].tolist()
        out.append(tok.decode([t for t in new if t != 2]))
    return out


def ch09_grpo_run(steps=80, per_step=4, lr=2e-3, beta=0.04):
    """Прогон GRPO на BookLM с новым адаптером LoRA: журнал, жадные ответы и адаптер."""
    tok, policy, params = new_policy()
    opt = torch.optim.Adam(params, lr=lr)
    gen = torch.Generator().manual_seed(0)
    fields = ["step", "reward", "kl", "length", "eos", "hero", "dead", "seconds"]
    rows, samples = [], {}
    start = time.time()
    for step in range(steps):
        k = (step * per_step) % len(REQUESTS)
        batch = REQUESTS[k:k + per_step]
        s = grpo_step(tok, policy, opt, batch, gen, beta=beta)
        s["step"], s["seconds"] = step, time.time() - start
        rows.append(s)
        print(f"шаг {step:3d}: награда {s['reward']:.3f}  KL {s['kl']:.4f}  длина {s['length']:6.1f}  "
              f"<eos> {s['eos']:.2f}  герой {s['hero']:.2f}  мёртвых групп {s['dead']:.2f}  "
              f"{s['seconds']:.0f} с", flush=True)
        if step in (0, steps // 2, steps - 1):
            samples[step] = greedy_answers(tok, policy, REQUESTS[:4])
    with open(OUT / "ch09-grpo-run.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for s in rows:
            w.writerow({k: (f"{s[k]:.5f}" if isinstance(s[k], float) else s[k]) for k in fields})
    json.dump(samples, open(OUT / "ch09-grpo-samples.json", "w"), ensure_ascii=False, indent=1)
    torch.save({n: p.detach() for n, p in policy.named_parameters() if p.requires_grad},
               OUT / "ch09-grpo-adapter.pt")
    smooth_csv()


def smooth_csv(window=5):
    """Добавляет к журналу скользящие средние за window шагов: по одному шагу (4 группы) кривые шумные."""
    path = OUT / "ch09-grpo-run.csv"
    rows = list(csv.DictReader(open(path)))
    keys = ["reward", "kl", "length", "hero"]
    for i, row in enumerate(rows):
        part = rows[max(0, i - window + 1):i + 1]
        for k in keys:
            row[k + "_avg"] = f"{sum(float(r[k]) for r in part) / len(part):.5f}"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[k for k in rows[0] if not k.endswith("_avg")] +
                           [k + "_avg" for k in keys])
        w.writeheader()
        w.writerows(rows)


# ----------------------------------------------------------------------------- одна группа GRPO

def reward(text, ended, length, patterns):
    """1, если в истории есть все слова (по образцам) и она закончилась <eos> не длиннее 300 токенов."""
    has_words = all(re.search(p, text.lower()) for p in patterns)
    return int(has_words and ended and length <= 300)


def advantages(r):
    r = torch.tensor(r, dtype=torch.float64)
    mean, std = r.mean(), r.std(unbiased=False)       # s_r — разброс по группе, делим на G
    return (r - mean) / (std + 1e-6), mean.item(), std.item()


def group(tok, model, req, patterns, G=8):
    """G ответов на одну просьбу (зёрна 0..G-1), их награды и преимущества GRPO."""
    ids = prompt_ids(tok, req, chat=True)
    rows, rewards = [], []
    for seed in range(G):
        new, ended = sample(tok, model, ids, 300, seed)
        text = tok.decode(new)
        rewards.append(reward(text, ended, len(new) + ended, patterns))
        rows.append((short(text, 12), len(new) + ended, ended))
    return rows, rewards, *advantages(rewards)


KOT = r"\bкот(а|у|ом|е|ы|ов|ик\w*|ён\w*)?\b"     # кот в любой форме, но не «который»
EZH = r"\b[её]ж(ик|ич)"                              # ёжик, ёжика, ежичек


def ch09_grpo_group():
    """Восемь ответов на одну просьбу, их награды и преимущества GRPO."""
    tok, inst = load("instruct")
    req = "Напиши сказку, в которой есть кот и мяч."
    rows, rewards, adv, mean, std = group(tok, inst, req, [KOT, r"\bмяч"])
    print("Просьба:", req)
    print("  длина  r     Â   начало ответа")
    for (text, n, ended), r, a in zip(rows, rewards, adv):
        print(f"  {n:4d}{' ' if ended else '+'} {r:2d} {a:+5.2f}   {text}")
    print(f"  r̄ = {mean:.3f}, s_r = {std:.3f}")
    for req, word in (("Напиши сказку, в которой есть ёжик и фонарик.", r"\bфонар"),
                      ("Напиши сказку, в которой есть ёжик и телескоп.", r"\bтелескоп")):
        _, rewards, adv, mean, std = group(tok, inst, req, [EZH, word])
        print("Просьба:", req)
        print("  награды:     ", " ".join(f"{r:5d}" for r in rewards))
        print("  преимущества:", " ".join(f"{a:+5.2f}" for a in adv), f"  r̄ = {mean:.3f}, s_r = {std:.3f}")
    print("(длина в токенах с <eos>; «+» — ответ не закончился <eos> за 300 токенов)")


NEW_HEROES = [(f"Напиши сказку про {h}.", p) for h, p in
              (("волка", r"\bвол(к|чо|чи)"), ("корову", r"\bкоров"), ("белку", r"\bбел(к|оч)"),
               ("петуха", r"\bпет(ух|уш)"))]


def ch09_grpo_compare(G=8):
    """До и после прогона: награда, сколько раз назван герой, длина; 16 просьб прогона и 4 новых героя."""
    tok, policy, _ = new_policy()
    policy.load_state_dict(torch.load(OUT / "ch09-grpo-adapter.pt"), strict=False)

    def measure(reqs):
        gen = torch.Generator().manual_seed(1)
        rew, names, length = [], [], []
        for request, hero in reqs:
            for a, e in sample_group(policy, prompt_ids(tok, request, chat=True), G, MAX_NEW, gen):
                rew.append(hero_reward(tok, a, e, hero))
                names.append(len(re.findall(hero, tok.decode(a).lower())))
                length.append(len(a) + e)
        return [sum(v) / len(v) for v in (rew, names, length)]

    lines = []
    for title, reqs in (("16 просьб прогона", REQUESTS), ("4 новых героя", NEW_HEROES)):
        for name, on in (("до GRPO", False), ("после", True)):
            set_lora(policy, on)                         # выключенный адаптер = инструктивная модель
            r, k, n = measure(reqs)
            lines.append(f"{title}, {name}: награда {r:.2f}, герой назван {k:.1f} раза, длина {n:.0f}")
    gen = torch.Generator().manual_seed(1)
    group = sample_group(policy, prompt_ids(tok, REQUESTS[0][0], chat=True), G, MAX_NEW, gen)
    lines += ["  " + " ".join(tok.decode(a).split()[:12]) + " …" for a, _ in group[:3]]
    text = "\n".join(lines)
    print(text)
    (OUT / "ch09-grpo-compare.txt").write_text(text + "\n", encoding="utf-8")


@torch.no_grad()
def ch09_rollout_mismatch(threshold=2.0):
    """Расхождение движков: ответы пишет копия в BF16 или с весами в 4 бита, учитель считает в FP32.
    Отношение pi_train / pi_rollout по токенам и по ответам целиком; сколько токенов срезает TIS."""
    import copy
    from booklm.lab import quantize_copy
    tok, train = load("instruct")
    lines = []
    for name, roll in (("BF16  ", copy.deepcopy(train).to(torch.bfloat16)),
                       ("4 бита", quantize_copy(train, bits=4))):
        gen = torch.Generator().manual_seed(0)
        tok_ratio, seq_ratio = [], []
        for request, _ in REQUESTS[:4]:
            ids = prompt_ids(tok, request, chat=True)
            x, mask = answer_batch(ids, sample_group(roll, ids, 8, MAX_NEW, gen))
            lp_train = train(x[:, :-1]).log_softmax(-1).gather(-1, x[:, 1:, None])[..., 0]
            lp_roll = roll(x[:, :-1]).float().log_softmax(-1).gather(-1, x[:, 1:, None])[..., 0]
            diff = (lp_train - lp_roll) * mask
            tok_ratio.append(diff.exp()[mask.bool()])
            seq_ratio += diff.sum(1).exp().tolist()
        r, s = torch.cat(tok_ratio), torch.tensor(seq_ratio)
        lines.append(f"{name}: токенов {len(r)}, отношение от {r.min():.2f} до {r.max():.2f}, "
                     f"выше {threshold:g}: {(r > threshold).sum().item()}; ответы целиком: "
                     f"от {s.min():.2f} до {s.max():.2f}")
    text = "\n".join(lines)
    print(text)
    (OUT / "ch09-rollout-mismatch.txt").write_text(text + "\n", encoding="utf-8")


def ch09_grpo_hacking():
    """Живой случай взлома награды: жадные сказки на шагах 0 и 79 и счёт упоминаний героя по предложениям.
    Берёт ch09-grpo-samples.json и ch09-grpo-compare.txt, ничего не пересчитывает."""
    samples = json.load(open(OUT / "ch09-grpo-samples.json"))
    lines = ["Прогон GRPO на BookLM (ch09-grpo-run.csv): LoRA r=8, группа 8, 4 просьбы на шаг, 80 шагов, beta=0.04.",
             "Награда 1, если герой просьбы найден образцом в тексте и сказка закончилась <eos> не длиннее 150 токенов.",
             "", "До и после на выборке (temperature=1, 8 ответов на просьбу), из ch09-grpo-compare.txt:"]
    lines += ["  " + l for l in (OUT / "ch09-grpo-compare.txt").read_text(encoding="utf-8").splitlines()[:4]]
    lines += ["", "Жадные ответы (temperature=0) на первые четыре просьбы после шага 0 (один шаг Adam, почти исходная модель) и шага 79:"]
    for i, (request, hero) in enumerate(REQUESTS[:4]):
        for step in ("0", "79"):
            text = samples[step][i]
            sents = [s for s in re.split(r"(?<=[.!?…])\s+", text.strip()) if s]
            with_hero = sum(bool(re.search(hero, s.lower())) for s in sents)
            names = len(re.findall(hero, text.lower()))
            lines.append(f"{request} шаг {step:>2}: предложений {len(sents)}, с героем {with_hero}, "
                         f"упоминаний {names}, герой {'найден' if names else 'не найден'}")
            lines.append("    " + " ".join(text.split()[:40]) + " …")
    text = "\n".join(lines)
    print(text)
    (OUT / "ch09-grpo-hacking.txt").write_text(text + "\n", encoding="utf-8")


EXPERIMENTS = {"ch09_grpo_group": ch09_grpo_group, "group": ch09_grpo_group,
               "ch09_grpo_run": ch09_grpo_run,
               "ch09_grpo_smooth": smooth_csv, "smooth": smooth_csv,
               "ch09_grpo_compare": ch09_grpo_compare, "compare": ch09_grpo_compare,
               "ch09_grpo_hacking": ch09_grpo_hacking, "hacking": ch09_grpo_hacking,
               "ch09_rollout_mismatch": ch09_rollout_mismatch, "mismatch": ch09_rollout_mismatch}

if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    argv = sys.argv[1:]
    if argv and "=" in argv[0]:              # старый вид: «python -m booklm.experiments.ch09 steps=80»
        argv = ["ch09_grpo_run"] + argv
    run(EXPERIMENTS, argv, args=True)
