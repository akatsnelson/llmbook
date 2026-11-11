"""Перевод TinyStories на русский языковой моделью через vLLM.

    python translate_llm.py --src TinyStoriesV2-GPT4-train.txt --out ru_stories.jsonl --limit 200000
"""
import argparse, json, time
from pathlib import Path
from vllm import LLM, SamplingParams

SYSTEM = (
    "Ты переводишь детские рассказы с английского на русский. Пиши простым, естественным русским "
    "языком, как в хорошей детской книжке для 4–6 лет: короткие предложения, обычные слова. "
    "Имена передавай по-русски единообразно (Lily — Лили, Tom — Том, Ben — Бен, Sue — Сью). "
    "Прямую речь оформляй по-русски, с кавычками «ёлочками» или тире. Не пропускай ни одного предложения, "
    "ничего не добавляй и не объясняй. Не калькируй английские обороты: slide — горка, have fun — весело "
    "играть, have a great time — отлично провести время, mommy — мама. Следи за родом и падежами. "
    "Выведи только перевод."
)

def read_stories(path, limit, skip=0):
    buf, n = [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip() == "<|endoftext|>":
                story = "".join(buf).strip(); buf = []
                if story:
                    n += 1
                    if n > skip:
                        yield n, story
                    if limit and n >= skip + limit:
                        return
            else:
                buf.append(line)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--model", default="Qwen/Qwen3-8B"); ap.add_argument("--chunk", type=int, default=4096); ap.add_argument("--mem", type=float, default=0.6); ap.add_argument("--max_seqs", type=int, default=256)
    a = ap.parse_args()
    out = Path(a.out)
    done = {json.loads(l)["id"] for l in out.open(encoding="utf-8")} if out.exists() else set()
    llm = LLM(model=a.model, dtype="bfloat16", max_model_len=1536, gpu_memory_utilization=a.mem,
              enable_prefix_caching=True, max_num_seqs=a.max_seqs)
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0.0, max_tokens=900)
    def prompt(story):
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": story}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    fo = out.open("a", encoding="utf-8")
    batch, t0, n = [], time.time(), 0
    def flush(batch):
        nonlocal n
        res = llm.generate([prompt(s) for _, s in batch], sp, use_tqdm=False)
        for (sid, s), r in zip(batch, res):
            o = r.outputs[0]
            if o.finish_reason != "stop":
                continue                      # обрезанный перевод не берём
            fo.write(json.dumps({"id": sid, "en": s, "ru": o.text.strip()}, ensure_ascii=False) + "\n")
            n += 1
        fo.flush()
        print(f"stories {n} | {n/(time.time()-t0):.1f} st/s", flush=True)
    for sid, s in read_stories(a.src, a.limit, a.skip):
        if sid in done:
            continue
        batch.append((sid, s))
        if len(batch) >= a.chunk:
            flush(batch); batch = []
    if batch:
        flush(batch)

if __name__ == "__main__":
    main()
