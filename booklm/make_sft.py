"""Пары «просьба → история» для дообучения BookLM следовать инструкциям (глава 8).

Для каждой русской истории языковая модель сочиняет короткую просьбу ребёнка или взрослого,
на которую эта история была бы хорошим ответом. Формы просьб разные, чтобы BookLM училась
понимать смысл, а не один шаблон.

    python make_sft.py --stories data/ru_stories.jsonl --out data/sft.jsonl --skip 60000 --limit 10000

Сырого корпуса ru_stories.jsonl в репозитории нет: его делает booklm/translate_llm.py из
TinyStoriesV2-GPT4-train.txt (нужны видеокарта и vLLM). Результат этого шага уже лежит в models/booklm.
"""
import argparse
import json
import random
from pathlib import Path

from vllm import LLM, SamplingParams

STYLES = [
    "Начни просьбу со слов «Напиши сказку».",
    "Начни просьбу со слов «Расскажи историю».",
    "Начни просьбу со слов «Сочини короткую сказку».",
    "Попроси историю, в которой обязательно есть два-три конкретных слова из этой истории, перечисли их в кавычках.",
    "Сформулируй просьбу как вопрос: «Можешь рассказать…?»",
    "Попроси историю с определённой моралью или концовкой, которая есть в этой истории.",
]
SYSTEM = ("Тебе дают короткую детскую историю. Напиши одну короткую просьбу (до 20 слов), на которую эта "
          "история была бы хорошим ответом. Упоминай героя и главное событие, но не пересказывай всю историю. "
          "Выведи только текст просьбы.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stories", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--limit", type=int, default=10000)
    ap.add_argument("--model", default="Qwen/Qwen3-8B-FP8")
    ap.add_argument("--mem", type=float, default=0.75)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.stories, encoding="utf-8")][a.skip:a.skip + a.limit]
    rng = random.Random(0)
    llm = LLM(model=a.model, max_model_len=1536, gpu_memory_utilization=a.mem, max_num_seqs=512)
    tok = llm.get_tokenizer()
    prompts = []
    for r in rows:
        msgs = [{"role": "system", "content": SYSTEM + " " + rng.choice(STYLES)},
                {"role": "user", "content": r["ru"]}]
        prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                               enable_thinking=False))
    res = llm.generate(prompts, SamplingParams(temperature=0.7, top_p=0.9, max_tokens=60, seed=0))
    n = 0
    with Path(a.out).open("w", encoding="utf-8") as f:
        for r, o in zip(rows, res):
            req = o.outputs[0].text.strip().strip("«»\"")
            if o.outputs[0].finish_reason != "stop" or not req or len(req.split()) > 30:
                continue
            f.write(json.dumps({"id": r["id"], "request": req, "story": r["ru"]}, ensure_ascii=False) + "\n")
            n += 1
    print("пар:", n)


if __name__ == "__main__":
    main()
