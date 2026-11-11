"""Предобучение BookLM. Пишет всё, что потом показывает глава 7: кривую ошибки, нормы градиента,
скорость обучения, образцы текста по ходу обучения и контрольные точки.

Нужна видеокарта NVIDIA: модель, данные и оптимизатор лежат на CUDA. Готовые веса этого прогона уже есть
в models/booklm (booklm-base.pt, промежуточные шаги в steps/), обучать заново для книги не нужно.
Прогон книги (около 25 минут на NVIDIA GB10), запуск из корня репозитория:

    python booklm/train.py --data models/booklm/data --tokenizer models/booklm --out runs/base --tokens 100e6

Контрольные точки здесь пишутся словарём {"model", "opt", ...} в FP32; в models/booklm/steps лежат
только веса в BF16, то есть state_dict модели из такой точки.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

try:                                    # импорт как пакет: from booklm.sft_lora import ...
    from .model import BookLM, BookLMConfig
except ImportError:                     # запуск файлом: python booklm/<скрипт>.py
    from model import BookLM, BookLMConfig

PROMPTS = ["Однажды", "Жила-была маленькая девочка по имени Лили.", "Том нашёл в саду"]
SAMPLE_STEPS = {0, 25, 100, 250, 500, 1000, 2000, 4000}


def batch(data, bsz, T, gen):
    ix = torch.randint(len(data) - T - 1, (bsz,), generator=gen)
    x = torch.stack([torch.from_numpy(data[i:i + T].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + T].astype(np.int64)) for i in ix])
    return x.cuda(non_blocking=True), y.cuda(non_blocking=True)


@torch.no_grad()
def evaluate(model, data, T, iters=40):
    model.eval()
    gen = torch.Generator().manual_seed(123)
    losses = []
    for _ in range(iters):
        x, y = batch(data, 32, T, gen)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
        losses.append(F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.view(-1)).item())
    model.train()
    return sum(losses) / len(losses)


@torch.no_grad()
def samples(model, tok, step):
    model.eval()
    out = []
    for p in PROMPTS:
        ids = torch.tensor([[tok.token_to_id("<bos>")] + tok.encode(p).ids], device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            g = model.generate(ids, max_new=120, temperature=0.8, top_k=40,
                               eos_id=tok.token_to_id("<eos>"), seed=step + 1)
        out.append(tok.decode(g[0, 1:].tolist()))
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", type=float, default=300e6)
    ap.add_argument("--bsz", type=int, default=64)
    ap.add_argument("--T", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--min_lr_frac", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    out = Path(a.out)
    (out / "ckpt").mkdir(parents=True, exist_ok=True)
    tok = Tokenizer.from_file(str(Path(a.tokenizer) / "tokenizer.json"))
    train = np.memmap(Path(a.data) / "train.bin", dtype=np.uint16, mode="r")
    val = np.memmap(Path(a.data) / "val.bin", dtype=np.uint16, mode="r")
    cfg = BookLMConfig(vocab=tok.get_vocab_size(), max_len=a.T)
    model = BookLM(cfg).cuda()
    assert model.num_params() == 6_622_464
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.wd},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=a.lr, betas=(0.9, 0.95), eps=1e-8, fused=True)
    tokens_per_step = a.bsz * a.T
    steps = int(a.tokens // tokens_per_step)
    ckpt_steps = {0, 100, 1000, steps // 4, steps // 2, steps}

    def lr_at(s):
        if s < a.warmup:
            return a.lr * (s + 1) / a.warmup
        p = (s - a.warmup) / max(1, steps - a.warmup)
        return a.lr * (a.min_lr_frac + (1 - a.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * p)))

    json.dump({"config": cfg.to_dict(), "args": vars(a), "steps": steps,
               "tokens_per_step": tokens_per_step, "train_tokens": len(train), "val_tokens": len(val)},
              open(out / "run.json", "w"), ensure_ascii=False, indent=1)
    log = open(out / "log.jsonl", "a")
    gen = torch.Generator().manual_seed(a.seed)
    t0 = time.time()
    for step in range(steps + 1):
        if step in SAMPLE_STEPS or step == steps:
            vl = evaluate(model, val, a.T)
            log.write(json.dumps({"step": step, "val_loss": vl, "samples": samples(model, tok, step)},
                                 ensure_ascii=False) + "\n")
            log.flush()
            print(f"step {step} val {vl:.3f}", flush=True)
        elif step % 250 == 0:
            vl = evaluate(model, val, a.T)
            log.write(json.dumps({"step": step, "val_loss": vl}) + "\n")
            print(f"step {step} val {vl:.3f}", flush=True)
        if step in ckpt_steps:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                        "cfg": cfg.to_dict()}, out / "ckpt" / f"step{step:06d}.pt")
        if step == steps:
            break
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        x, y = batch(train, a.bsz, a.T, gen)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
        loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.view(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        opt.step()
        if step % 10 == 0:
            torch.cuda.synchronize()
            el = time.time() - t0
            log.write(json.dumps({"step": step, "loss": loss.item(), "lr": lr_at(step), "grad_norm": gnorm,
                                  "tokens": (step + 1) * tokens_per_step,
                                  "tok_per_s": (step + 1) * tokens_per_step / el}) + "\n")
            if step % 200 == 0:
                print(f"step {step}/{steps} loss {loss.item():.3f} gnorm {gnorm:.2f} "
                      f"{(step + 1) * tokens_per_step / el / 1e3:.0f}k tok/s", flush=True)
    torch.save(model.state_dict(), out / "booklm-base.pt")
    print("готово:", out / "booklm-base.pt")


if __name__ == "__main__":
    main()
