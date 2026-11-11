"""Насколько инструктивная BookLM держится просьбы: доля главных слов просьбы, которые встретились в ответе.

Берутся 100 отложенных пар «просьба → история» (те же 200, что sft_lora.py не использует для обучения).
Слово считается совпавшим, если в ответе есть его первые пять букв: так «кота» засчитывается в «кот».
Синонимы («пёс» вместо «собаки») метрика не ловит, поэтому она занижает результат.

    python -m booklm.eval_follow instruct
"""
import json
import random
import re
import sys

import torch

from booklm.lab import DEVICE, MODELS, load, prompt_ids, special

STOP = set("напиши расскажи сочини историю сказку короткую который которая которые чтобы можешь "
           "рассказать слова используя обязательно".split())


def keys(request):
    words = [w.lower() for w in re.findall(r"[А-Яа-яЁё]+", request) if len(w) >= 5]
    return [w[:5] for w in words if w not in STOP][:4]


def main(kind="instruct", n=100):
    tok, model = load(kind)
    rows = [json.loads(l) for l in open(MODELS / "data" / "sft.jsonl", encoding="utf-8")]
    random.Random(0).shuffle(rows)
    hit = total = 0
    for i, row in enumerate(rows[:200][:n]):
        ids = prompt_ids(tok, row["request"], chat=True)
        with torch.no_grad():
            out = model.generate(torch.tensor([ids], device=DEVICE), max_new=200, temperature=0.8, top_k=40,
                                 eos_id=special(tok)["<eos>"], seed=i)
        text = tok.decode(out[0, len(ids):].tolist()).lower()
        k = keys(row["request"])
        hit += sum(w in text for w in k)
        total += len(k)
    print(f"{kind}: совпало главных слов просьбы {hit}/{total} = {hit / total:.2f}")


if __name__ == "__main__":
    main(*sys.argv[1:2])
