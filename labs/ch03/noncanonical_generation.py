"""Неканоническая токенизация в текстах, которые BookLM пишет сама.

Опыт к подразделу 3.5.6 «Неканоническая токенизация» главы 3.
BookLM-Base пишет 50 историй с пустого начала (только <bos>), до <eos> или 300 токенов, при двух
способах выбора следующего токена: выборка по всем вероятностям модели (температура 1) и выборка из 40
самых вероятных токенов при температуре 0,8, как в booklm.lab.generate (сами способы разобраны в разделе
7.7 «Выбор следующего токена»). Каждую историю переводим в байты, затем в текст и кодируем токенизатором заново.
Если номера не совпали с теми, что выбрала модель, история неканоническая. Если байты истории не
складываются в правильный UTF-8 (модель оборвала многобайтную букву), считаем её отдельно: такой
текст нельзя закодировать заново без потерь.

Зёрна генерации 0..49 фиксированы, считаем на процессоре: числа одинаковы при каждом запуске.
Запуск из корня репозитория: python labs/ch03/noncanonical_generation.py
Считается около полуминуты на Apple M2 Max; на медленном процессоре — до нескольких минут.
"""

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from booklm.lab import MODELS, load, special  # noqa: E402
from labs.ch03.scaffold_tokens import piece_bytes  # noqa: E402

SEEDS = range(50)
MAX_NEW = 300
MODES = {"температура 1": dict(temperature=1.0, top_k=None),
         "температура 0,8, 40 лучших": dict(temperature=0.8, top_k=40)}


def token_bytes():
    vocab = json.load(open(MODELS / "tokenizer.json", encoding="utf-8"))["model"]["vocab"]
    return {i: piece_bytes(p) for p, i in vocab.items()}


@torch.no_grad()
def run(tok, model, how, as_bytes):
    sp = special(tok)
    stats = dict(noncanon=0, broken=0, sampled=0, canon=0, longer=0, examples=[])
    for seed in SEEDS:
        out = model.generate(torch.tensor([[sp["<bos>"]]]), max_new=MAX_NEW, eos_id=sp["<eos>"],
                             seed=seed, **how)[0, 1:].tolist()
        if out and out[-1] == sp["<eos>"]:
            out = out[:-1]
        assert all(i >= len(sp) for i in out)            # служебных токенов внутри истории нет
        data = b"".join(as_bytes[i] for i in out)
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            stats["broken"] += 1
            continue
        canon = tok.encode(text).ids
        assert tok.decode(canon) == text
        stats["sampled"] += len(out)
        stats["canon"] += len(canon)
        if canon != out:
            stats["noncanon"] += 1
            stats["longer"] += len(canon) > len(out)
            k = next(i for i, (a, b) in enumerate(zip(out, canon)) if a != b)
            stats["examples"].append((seed, [tok.decode([i]) for i in out[k:k + 2]],
                                      [tok.decode([i]) for i in canon[k:k + 2]]))
    return stats


if __name__ == "__main__":
    torch.set_num_threads(2)
    tok, model = load("base")
    as_bytes = token_bytes()
    res = {}
    for name, how in MODES.items():
        s = run(tok, model, how, as_bytes)
        res[name] = s
        ok = len(SEEDS) - s["broken"]
        print(f"{name}: неканонических {s['noncanon']} из {ok} историй, ещё {s['broken']} с оборванной "
              f"буквой; токенов модели {s['sampled']}, после перекодирования {s['canon']} "
              f"({s['canon'] / s['sampled'] - 1:+.2%}); каноническая длиннее в {s['longer']} историях")
        for seed, a, b in s["examples"][:6]:
            print(f"  зерно {seed}: модель {' | '.join(a)!r}, токенизатор {' | '.join(b)!r}")
    hot, cold = res["температура 1"], res["температура 0,8, 40 лучших"]
    assert hot["noncanon"] == 17 and hot["broken"] == 1
    assert hot["sampled"] == 8557 and hot["canon"] == 8554 and hot["longer"] == 5
    assert cold["noncanon"] == 3 and cold["broken"] == 0
    assert cold["sampled"] == 8208 and cold["canon"] == 8209 and cold["longer"] == 1
    assert abs(hot["canon"] / hot["sampled"] - 1) < 0.001 and abs(cold["canon"] / cold["sampled"] - 1) < 0.001
