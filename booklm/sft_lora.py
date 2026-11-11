"""Дообучение BookLM следовать просьбам: LoRA поверх замороженной базы (глава 8).

Диалог: <bos><|user|>просьба<|assistant|>история<eos>. Ошибка считается только на истории и <eos>.
LoRA: W' = W + (alpha/r)·B·A, A — случайная, B — нулевая, поэтому до обучения модель в точности базовая.

Нужна видеокарта NVIDIA. Готовый результат — models/booklm/booklm-instruct.pt и booklm-instruct-lora.pt.
Чтобы дообучить BookLM на своих парах на процессоре, удобнее функции главы 17:
booklm.experiments.ch17 (train_lora, merged_state). Запуск из корня репозитория:

    python booklm/sft_lora.py --base models/booklm/booklm-base.pt --sft models/booklm/data/sft.jsonl \
        --tokenizer models/booklm --out runs/lora
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

try:                                    # импорт как пакет: from booklm.sft_lora import ...
    from .model import BookLM, BookLMConfig
except ImportError:                     # запуск файлом: python booklm/<скрипт>.py
    from model import BookLM, BookLMConfig

TARGETS = ("attention.q", "attention.k", "attention.v", "attention.out", "ffn.gate", "ffn.up", "ffn.down")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=16, alpha=32):
        super().__init__()
        self.base, self.r, self.scale = base, r, alpha / r
        self.A = nn.Parameter(torch.randn(r, base.in_features) / math.sqrt(base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        self.enabled = True

    def forward(self, x):
        y = self.base(x)
        return y + (x @ self.A.T @ self.B.T) * self.scale if self.enabled else y

    def merged_weight(self):
        return self.base.weight + self.scale * (self.B @ self.A)


def add_lora(model, r, alpha):
    for name, mod in list(model.named_modules()):
        for t in TARGETS:
            if name.endswith(t) and isinstance(mod, nn.Linear):
                parent = model.get_submodule(name.rsplit(".", 1)[0])
                setattr(parent, name.rsplit(".", 1)[1], LoRALinear(mod, r, alpha))
    for n, p in model.named_parameters():
        p.requires_grad = n.endswith(".A") or n.endswith(".B")
    return model


def set_lora(model, enabled):
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.enabled = enabled


def merged_state(model):
    sd = {}
    for n, p in model.state_dict().items():
        if n.endswith(".A") or n.endswith(".B"):
            continue
        sd[n.replace(".base.", ".")] = p
    for name, m in model.named_modules():
        if isinstance(m, LoRALinear):
            sd[name + ".weight"] = m.merged_weight().detach()
    return sd


def encode(tok, req, story, T):
    u, a, bos, eos = (tok.token_to_id(s) for s in ("<|user|>", "<|assistant|>", "<bos>", "<eos>"))
    prompt = [bos, u] + tok.encode(req).ids + [a]
    answer = tok.encode(story).ids + [eos]
    ids = (prompt + answer)[:T + 1]
    labels = ([-100] * len(prompt) + answer)[:T + 1]
    return ids, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True); ap.add_argument("--sft", required=True)
    ap.add_argument("--tokenizer", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--r", type=int, default=16); ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--epochs", type=float, default=2); ap.add_argument("--bsz", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3); ap.add_argument("--T", type=int, default=512)
    a = ap.parse_args()
    torch.manual_seed(0)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer.from_file(str(Path(a.tokenizer) / "tokenizer.json"))
    rows = [json.loads(l) for l in open(a.sft, encoding="utf-8")]
    random.Random(0).shuffle(rows)
    val, train = rows[:200], rows[200:]
    data = [encode(tok, r["request"], r["story"], a.T) for r in train]

    model = BookLM(BookLMConfig()).cuda()
    model.load_state_dict(torch.load(a.base, map_location="cuda"))
    probe = torch.tensor([encode(tok, val[0]["request"], val[0]["story"], a.T)[0]], device="cuda")
    with torch.no_grad():
        base_logits = model(probe).float()
    add_lora(model, a.r, a.alpha)
    model.cuda()
    with torch.no_grad():
        start = model(probe).float()
    assert torch.equal(start, base_logits), "B=0: до обучения LoRA не меняет модель"
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    steps = int(len(data) * a.epochs / a.bsz)
    log = open(out / "log.jsonl", "w")
    t0 = time.time()
    for step in range(steps):
        lr = a.lr * min(1.0, (step + 1) / 50) * 0.5 * (1 + math.cos(math.pi * step / steps))
        for g in opt.param_groups:
            g["lr"] = lr
        batch = random.Random(step).sample(data, a.bsz)
        L = max(len(i) for i, _ in batch) - 1
        x = torch.tensor([i[:-1] + [0] * (L - len(i) + 1) for i, _ in batch], device="cuda")
        y = torch.tensor([l[1:] + [-100] * (L - len(l) + 1) for _, l in batch], device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
        loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.view(-1), ignore_index=-100)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step % 20 == 0:
            log.write(json.dumps({"step": step, "loss": loss.item(), "lr": lr}) + "\n"); log.flush()
            print(f"step {step}/{steps} loss {loss.item():.3f} {time.time() - t0:.0f}s", flush=True)

    model.eval()
    with torch.no_grad():
        tuned = model(probe).float()
        set_lora(model, False)
        off = model(probe).float()
        set_lora(model, True)
    assert torch.allclose(off, base_logits, atol=1e-5), "выключенный адаптер возвращает базу"
    adapter = {n: p.detach().cpu() for n, p in model.state_dict().items() if n.endswith(".A") or n.endswith(".B")}
    torch.save({"adapter": adapter, "r": a.r, "alpha": a.alpha, "targets": TARGETS}, out / "booklm-instruct-lora.pt")
    merged = BookLM(BookLMConfig()).cuda()
    merged.load_state_dict(merged_state(model))
    with torch.no_grad():
        diff = (merged(probe).float() - tuned).abs().max().item()
    torch.save(merged.state_dict(), out / "booklm-instruct-merged.pt")
    json.dump({"trainable_params": trainable, "steps": steps, "train_pairs": len(data),
               "merge_max_logit_diff": diff}, open(out / "summary.json", "w"), ensure_ascii=False, indent=1)
    print(f"обучаемых параметров LoRA: {trainable:,}; расхождение после слияния {diff:.2e}")


if __name__ == "__main__":
    main()
