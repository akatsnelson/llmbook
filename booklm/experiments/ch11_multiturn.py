"""Глава 11 «Контекст и память», рубрика «разговор по частям» (подраздел «Падение качества в многоходовом
диалоге»): опыт ch11_multi_turn, просьба целиком и та же просьба по частям.

Как в работе Laban и др. (2025, arXiv:2505.06120), одну и ту же просьбу «сказка про героя и предмет»
даём инструктивной BookLM четырьмя способами:
  целиком              — «Напиши сказку про кота и мяч.»;
  части в одной реплике — «Напиши сказку. Герой сказки — кот. В сказке есть мяч.»;
  части в трёх репликах — три реплики пользователя, после каждой модель отвечает;
  три реплики + повтор  — те же три реплики, но последняя повторяет все части разом.
В последнем ответе ищем героя и предмет (с падежами и ближайшими синонимами: «пёс» за «собаку»).
6 героев × 6 предметов × 2 зерна = 72 разговора на способ.

Многоходовых окон BookLM при дообучении не видела: формат
<bos><|user|>…<|assistant|>…<eos><|user|>…<|assistant|> для неё новый.

Печатает таблицу: для каждого способа средняя длина окна и число разговоров из 72, где в последнем
ответе есть герой, предмет и оба сразу.

Запуск из корня репозитория (около 3 мин на процессоре Apple M2 Max):
    python -m booklm.experiments.ch11_multiturn
Вывод сохраняется в models/booklm/experiments/ch11-multi-turn.txt, та же таблица — в ch11-multi-turn.csv.
"""
import contextlib
import csv
import io
import re
import sys
from pathlib import Path

import torch

if not __package__:  # запуск файлом (python booklm/experiments/ch11_multiturn.py): корень репозитория — в путь поиска
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from booklm.lab import DEVICE, MODELS, encode_text, load, special

OUT = MODELS / "experiments"

# герой: именительный, винительный, шаблон поиска в ответе
HEROES = [("кот", "кота", r"\bкот(а|у|ом|е|ы|ов|ик\w*|ёнок\w*|ят\w*)?\b"),
          ("собака", "собаку", r"\bсобак\w*|\bпёс\b|\bпс(а|у|ом|е)\b|\bщен\w*"),
          ("птица", "птицу", r"\bптиц\w*|\bптичк\w*"),
          ("медведь", "медведя", r"\bмедвед\w*|\bмедвеж\w*|\bмишк\w*"),
          ("кролик", "кролика", r"\bкролик\w*"),
          ("лягушка", "лягушку", r"\bлягуш\w*")]
THINGS = [("мяч", "мяч", r"\bмяч\w*"),
          ("торт", "торт", r"\bторт\w*"),
          ("шарик", "шарик", r"\bшар(ик\w*|а|у|ом|е|ы|ов|ами)?\b"),
          ("книга", "книгу", r"\bкниг\w*|\bкнижк\w*"),
          ("лодка", "лодку", r"\bлодк\w*|\bлодоч\w*"),
          ("цветок", "цветок", r"\bцвет(ок|ка|ку|ком|ке|ы|ов|ам|ами|ах|оч\w*)\b")]


def requests():
    """36 просьб: (способ -> список реплик пользователя, шаблон героя, шаблон предмета)."""
    out = []
    for h, h_acc, h_re in HEROES:
        for t, t_acc, t_re in THINGS:
            parts = ["Напиши сказку.", f"Герой сказки — {h}.", f"В сказке есть {t}."]
            out.append(({"целиком": [f"Напиши сказку про {h_acc} и {t_acc}."],
                         "части в одной реплике": [" ".join(parts)],
                         "части в трёх репликах": parts,
                         "три реплики + повтор": parts[:2] + [" ".join(parts)]},
                        h_re, t_re))
    return out


@torch.no_grad()
def chat(tok, model, turns, seed, mid=100, last=120):
    """Разговор из нескольких реплик: после каждой модель отвечает, ответ остаётся в окне.
    Возвращает (последний ответ, длина окна в токенах)."""
    sp = special(tok)
    ids = [sp["<bos>"]]
    for k, q in enumerate(turns):
        ids += [sp["<|user|>"]] + encode_text(tok, q) + [sp["<|assistant|>"]]
        n = last if k == len(turns) - 1 else mid
        out = model.generate(torch.tensor([ids], device=DEVICE), max_new=n,
                             eos_id=sp["<eos>"], seed=seed + k)
        new = out[0, len(ids):].tolist()
        if new[-1] != sp["<eos>"]:
            new.append(sp["<eos>"])
        ids += new
    return tok.decode(new[:-1]).lower(), len(ids)


def score(text, h_re, t_re):
    """(назван ли герой, назван ли предмет) в ответе."""
    return bool(re.search(h_re, text)), bool(re.search(t_re, text))


def ch11_multi_turn(seeds=(0, 1)):
    """Тот же расчёт, что в листинге книги «Просьба по частям»."""
    tok, model = load("instruct")
    res = {}
    for turns_by_kind, h_re, t_re in requests():
        for seed in seeds:
            for kind, turns in turns_by_kind.items():
                text, n = chat(tok, model, turns, seed)
                res.setdefault(kind, []).append((n, *score(text, h_re, t_re)))
    rows = []
    print(f"{'способ':24}{'окно':>6}{'герой':>7}{'предмет':>9}{'оба':>5}")
    for kind, r in res.items():
        n, hero, thing = zip(*r)
        both = sum(a and b for a, b in zip(hero, thing))
        print(f"{kind:24}{sum(n) / len(r):6.0f}{sum(hero):7}{sum(thing):9}{both:5}")
        rows.append({"способ": kind, "окно": round(sum(n) / len(r)), "герой": sum(hero),
                     "предмет": sum(thing), "оба": both, "разговоров": len(r)})
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rows = ch11_multi_turn()
    text = buf.getvalue()
    print(text, end="")
    (OUT / "ch11-multi-turn.txt").write_text(text, encoding="utf-8")
    with open(OUT / "ch11-multi-turn.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    from booklm.experiments import pick
    pick({"ch11_multi_turn": ch11_multi_turn})     # --help и проверка имени
    main()                                         # txt и csv
