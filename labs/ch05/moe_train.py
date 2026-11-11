"""Обучение BookLM-MoE16: BookLM, у которой в каждом блоке вместо одной SwiGLU стоят
16 маленьких экспертов SwiGLU по 192 нейрона и маршрутизатор с выбором четырёх.

На токен работают 4 * 192 = 768 нейронов, как у одной SwiGLU исходной BookLM, а в
памяти лежат 16 * 192 = 3072 нейрона на блок. Модель нужна главе 5 для опыта с
удалением и слиянием экспертов (labs/ch05/moe_prune_merge.py).

Для скорости эксперты слоя хранятся склеенными в три матрицы и считаются все сразу:
доли невыбранных экспертов равны нулю, поэтому выход тот же, что у группового прохода
moe_forward из книги (это проверяет assert ниже). При сохранении матрицы режутся
обратно на экспертов с именами слоя MoEFeedForward из книги.

Запуск из корня репозитория: python labs/ch05/moe_train.py
Время: около часа на M2 Max (mps), около 4,5 минуты на RTX 5090. Готовые веса уже лежат в
models/booklm/experiments, так что для опыта moe_prune_merge.py запускать этот файл не нужно.
Варианты запуска (первое число — сколько шагов учить, по умолчанию 4000):
    python labs/ch05/moe_train.py 4000       # пишет models/booklm/experiments/ch05-moe16.pt
    python labs/ch05/moe_train.py 4000 mix   # то же на смеси: 3/4 окон из сказок,
                                             # 1/4 из технических текстов -> ch05-moe16-mix.pt

Технические тексты — код этого репозитория: файлы *.py из labs и booklm. Каждый
десятый файл отложен для проверки. Номера токенов обеих частей сохранены в
ch05-tech-train.bin и ch05-tech-test.bin: опыт не зависит от того, как код изменится
после обучения, а заново корпус собирается, только если этих файлов нет.
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from booklm.model import BookLM, BookLMConfig  # noqa: E402

E, K, D_FF = 16, 4, 192
OUT = ROOT / "models" / "booklm" / "experiments"
DEVICE = "mps" if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else "cpu")


class StackedMoE(nn.Module):
    """Смесь E экспертов SwiGLU, склеенных по внутренней оси: [E * D_FF, d]."""

    def __init__(self, d, d_ff=D_FF, experts=E, k=K):
        super().__init__()
        self.E, self.k, self.f = experts, k, d_ff
        self.router = nn.Linear(d, experts, bias=False)
        self.gate = nn.Linear(d, experts * d_ff, bias=False)
        self.up = nn.Linear(d, experts * d_ff, bias=False)
        self.down = nn.Linear(experts * d_ff, d, bias=False)
        self.aux = None                                   # балансирующая ошибка последнего прохода

    def forward(self, x):                                 # x: [B, T, d]
        logits = self.router(x)                           # [B, T, E]
        top = logits.topk(self.k, dim=-1)
        w = torch.zeros_like(logits).scatter(-1, top.indices, top.values.softmax(-1))
        h = F.silu(self.gate(x)) * self.up(x)             # [B, T, E * f]
        h = (h.view(*h.shape[:-1], self.E, self.f) * w[..., None]).flatten(-2)
        # балансирующая ошибка Switch: E * sum f_e * P_e
        f = torch.bincount(top.indices.flatten(), minlength=self.E).float() / top.indices.numel()
        P = logits.float().softmax(-1).flatten(0, -2).mean(0)
        self.aux = self.E * (f * P).sum()
        return self.down(h)


def build(seed=1337):
    torch.manual_seed(seed)
    cfg = BookLMConfig(d_ff=D_FF)
    model = BookLM(cfg)
    for b in model.blocks:
        b.ffn = StackedMoE(cfg.d_model)
    model.apply(BookLM._init)
    return model


def to_book_state(model):
    """Склеенные матрицы -> эксперты experts.{e}.gate/up/down, как в слое MoEFeedForward книги."""
    state = {}
    for name, v in model.state_dict().items():
        if ".ffn." not in name or name.endswith("router.weight"):
            state[name] = v.detach().cpu().float()
            continue
        prefix, part = name.rsplit(".", 2)[0], name.rsplit(".", 2)[1]
        chunks = v.split(D_FF, dim=1 if part == "down" else 0)
        for e, c in enumerate(chunks):
            state[f"{prefix}.experts.{e}.{part}.weight"] = c.detach().cpu().float().contiguous()
    return state


def check_equivalence():
    """Склеенный проход совпадает с поэкспертным групповым проходом."""
    torch.manual_seed(0)
    layer = StackedMoE(32, d_ff=8, experts=6, k=2)
    x = torch.randn(1, 20, 32)
    y = layer(x)[0]
    top = layer.router(x[0]).topk(2, dim=-1)
    w = top.values.softmax(-1)
    ref = torch.zeros_like(x[0])
    for e in range(6):
        rows, slot = (top.indices == e).nonzero(as_tuple=True)
        s = slice(e * 8, (e + 1) * 8)
        h = F.silu(x[0][rows] @ layer.gate.weight[s].T) * (x[0][rows] @ layer.up.weight[s].T)
        ref[rows] += w[rows, slot, None] * (h @ layer.down.weight[:, s].T)
    torch.testing.assert_close(y, ref, atol=1e-5, rtol=1e-5)


def tech_corpus():
    """Токены кода репозитория: (обучающая часть, отложенная часть), uint16."""
    train_path, test_path = OUT / "ch05-tech-train.bin", OUT / "ch05-tech-test.bin"
    if not train_path.exists():
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(str(ROOT / "models" / "booklm" / "tokenizer.json"))
        files = sorted(p for d in ("labs", "booklm") for p in (ROOT / d).rglob("*.py")
                       if "__pycache__" not in str(p))
        parts = ([], [])
        for n, f in enumerate(files):                      # каждый десятый файл — в проверку
            ids = [1] + tok.encode(f.read_text(encoding="utf-8")).ids + [2]
            parts[n % 10 == 0].extend(ids)
        OUT.mkdir(parents=True, exist_ok=True)
        np.array(parts[0], dtype=np.uint16).tofile(train_path)
        np.array(parts[1], dtype=np.uint16).tofile(test_path)
    load = lambda p: torch.from_numpy(np.fromfile(p, dtype=np.uint16).astype(np.int64))
    return load(train_path), load(test_path)


def main(steps=3000, bsz=32, T=512, lr=2e-3, warmup=300, aux_coef=0.01, mix=False):
    check_equivalence()
    data = np.fromfile(ROOT / "models" / "booklm" / "data" / "train.bin", dtype=np.uint16)
    data = torch.from_numpy(data.astype(np.int64))
    tech = tech_corpus()[0] if mix else None
    name = "ch05-moe16-mix" if mix else "ch05-moe16"
    model = build().to(DEVICE)
    total = sum(p.numel() for p in model.parameters())
    print(f"параметров {total:,}", flush=True)
    decay = [p for p in model.parameters() if p.dim() >= 2]
    other = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1},
                             {"params": other, "weight_decay": 0.0}],
                            lr=lr, betas=(0.9, 0.95))
    gen = torch.Generator().manual_seed(1337)

    def lr_at(s):
        if s < warmup:
            return lr * (s + 1) / warmup
        p = (s - warmup) / max(1, steps - warmup)
        return lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))

    log = []
    t0 = time.time()
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        n_tech = bsz // 4 if mix else 0                   # четверть окон — технические тексты
        ix = torch.randint(len(data) - T - 1, (bsz - n_tech,), generator=gen)
        windows = [data[i:i + T + 1] for i in ix]
        if n_tech:
            jx = torch.randint(len(tech) - T - 1, (n_tech,), generator=gen)
            windows += [tech[j:j + T + 1] for j in jx]
        w = torch.stack(windows)
        x, y = w[:, :-1].to(DEVICE), w[:, 1:].to(DEVICE)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        aux = sum(b.ffn.aux for b in model.blocks) / len(model.blocks)
        opt.zero_grad(set_to_none=True)
        (loss + aux_coef * aux).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 100 == 0 or step == steps - 1:
            el = time.time() - t0
            rec = {"step": step, "loss": loss.item(), "aux": aux.item(),
                   "tok_per_s": (step + 1) * bsz * T / el}
            log.append(rec)
            print(json.dumps(rec), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save(to_book_state(model), OUT / f"{name}.pt")
    json.dump({"E": E, "k": K, "d_ff": D_FF, "steps": steps, "bsz": bsz, "T": T, "lr": lr,
               "aux_coef": aux_coef, "mix": mix, "params": total, "log": log},
              open(OUT / f"{name}.json", "w"), ensure_ascii=False, indent=1)
    print("готово:", OUT / f"{name}.pt")


if __name__ == "__main__":
    main(steps=int(sys.argv[1]) if len(sys.argv) > 1 else 4000, mix="mix" in sys.argv[2:])
