"""Токенизатор BookLM: байтовый BPE на 8192 токена, обученный на русских историях.

Байтовый: любой текст представим без <unk> (глава 3). Служебные токены идут первыми и
получают номера 0–4, поэтому обычные слияния никогда не порождают их номера.

    python tokenizer_train.py --data data/ru_stories.jsonl --out tokenizer

Сырого корпуса ru_stories.jsonl в репозитории нет: его делает booklm/translate_llm.py из
TinyStoriesV2-GPT4-train.txt (нужны видеокарта и vLLM). Результат этого шага уже лежит в models/booklm.
"""
import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

SPECIAL = ["<pad>", "<bos>", "<eos>", "<|user|>", "<|assistant|>"]


def texts(path, limit=None):
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                return
            yield json.loads(line)["ru"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vocab", type=int, default=8192)
    ap.add_argument("--limit", type=int, default=200_000)
    a = ap.parse_args()

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=a.vocab, special_tokens=SPECIAL, min_frequency=2,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    tok.train_from_iterator(texts(a.data, a.limit), trainer=trainer)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tok.save(str(out / "tokenizer.json"))
    assert tok.get_vocab_size() == a.vocab, tok.get_vocab_size()
    for i, s in enumerate(SPECIAL):
        assert tok.token_to_id(s) == i
    sample = "Мама мыла раму, а потом она пошла спать."
    enc = tok.encode(sample)
    assert tok.decode(enc.ids) == sample
    print("словарь:", tok.get_vocab_size())
    print(sample, "->", enc.tokens)


if __name__ == "__main__":
    main()
