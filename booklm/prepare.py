"""Превращает истории в поток номеров токенов: <bos> история <eos> <bos> история <eos> …

Номера помещаются в uint16 (словарь 8192), поэтому поток хранится плоским файлом и читается
через memmap без загрузки в память. Последние 1% историй уходят в валидацию.

    python prepare.py --data data/ru_stories.jsonl --tokenizer tokenizer --out data

Сырого корпуса ru_stories.jsonl в репозитории нет: его делает booklm/translate_llm.py из
TinyStoriesV2-GPT4-train.txt (нужны видеокарта и vLLM). Результат этого шага уже лежит в models/booklm.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--val_frac", type=float, default=0.01)
    a = ap.parse_args()
    tok = Tokenizer.from_file(str(Path(a.tokenizer) / "tokenizer.json"))
    bos, eos = tok.token_to_id("<bos>"), tok.token_to_id("<eos>")
    stories = [json.loads(l)["ru"] for l in open(a.data, encoding="utf-8")]
    n_val = max(1, int(len(stories) * a.val_frac))
    out = Path(a.out)
    for split, part in (("train", stories[:-n_val]), ("val", stories[-n_val:])):
        ids = []
        for enc in tok.encode_batch(part):
            ids.append(bos)
            ids.extend(enc.ids)
            ids.append(eos)
        arr = np.array(ids, dtype=np.uint16)
        arr.tofile(out / f"{split}.bin")
        words = sum(len(s.split()) for s in part)
        print(f"{split}: историй {len(part)}, токенов {len(arr):,}, токенов на слово {len(arr) / words:.2f}")


if __name__ == "__main__":
    main()
