"""Глава 14 «Инференс», рубрика «многотокенное предсказание» (подраздел «Многотокенное предсказание
и черновые головы»): опыт ch14_mtp_head, голова MTP для BookLM.

Основная BookLM заморожена. Маленькая голова читает скрытый вектор последнего слоя на позиции t и
эмбеддинг токена t+1 и угадывает токен t+2 (так устроен модуль MTP глубины 1 у DeepSeek-V3, только
без собственного блока внимания). Голову обучаем тысячу шагов на CPU, затем меряем:
  1) долю принятых черновиков на жадных сказках BookLM у трёх источников черновика:
     таблицы пар токенов (биграмм), ранней контрольной точки BookLM (шаг 762) и головы;
  2) настоящий цикл спекулятивного декодирования с головой (gamma = 1): ответ совпадает
     с жадным ответом BookLM, сколько вызовов основной модели он экономит и сколько это стоит по времени.

Запуск из корня репозитория (около 2 мин на процессоре Apple M2 Max, почти всё — обучение головы):
    python -m booklm.experiments.ch14_mtp
Вывод сохраняется в models/booklm/experiments/ch14-mtp-head.txt, веса головы — в ch14-mtp-head.pt
(их берут листинги книги через load_head).
"""
import contextlib
import io
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

if not __package__:  # запуск файлом (python booklm/experiments/ch14_mtp.py): корень репозитория — в путь поиска
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from booklm.lab import MODELS, load, load_model, prompt_ids, val_tokens
from booklm.model import RMSNorm

OUT = MODELS / "experiments"
STEPS = 1000            # около 3 минут на 4 потоках CPU M2 Max


# ----------------------------------------------------------------------------- скрытые векторы

def hidden(model, ids, caches=None):
    """Логиты и скрытый вектор после последней нормализации: то, что умножают на матрицу выхода."""
    h = model.embed(ids)
    for i, block in enumerate(model.blocks):
        h, _ = block(h, model.rope_cos, model.rope_sin, None if caches is None else caches[i])
    h = model.norm(h)                                   # [B, T, 256]
    return h @ model.embed.weight.T, h


class DraftHead(nn.Module):
    """Голова MTP: [норма(h_t); норма(E[x_{t+1}])] -> 256 -> SwiGLU с остатком -> общая матрица выхода."""

    def __init__(self, embed, d=256, d_ff=768):
        super().__init__()
        self.embed = embed                              # таблица эмбеддингов BookLM, заморожена
        self.norm_h, self.norm_e = RMSNorm(d), RMSNorm(d)
        self.proj = nn.Linear(2 * d, d, bias=False)
        self.norm_f = RMSNorm(d)
        self.gate = nn.Linear(d, d_ff, bias=False)
        self.up = nn.Linear(d, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d, bias=False)
        self.norm_out = RMSNorm(d)
        nn.init.zeros_(self.down.weight)                # вначале ветвь SwiGLU молчит

    def forward(self, h, next_ids):                     # h: [B, T, 256], next_ids: [B, T]
        e = self.embed.weight[next_ids]
        u = self.proj(torch.cat([self.norm_h(h), self.norm_e(e)], dim=-1))
        f = self.norm_f(u)
        u = u + self.down(F.silu(self.gate(f)) * self.up(f))
        return self.norm_out(u) @ self.embed.weight.T  # логиты токена t+2

    def trainable(self):
        return sum(p.numel() for n, p in self.named_parameters() if not n.startswith("embed"))


def load_head(model):
    """Голова, обученная этим опытом (models/booklm/experiments/ch14-mtp-head.pt)."""
    head = DraftHead(model.embed)
    head.load_state_dict(torch.load(OUT / "ch14-mtp-head.pt"), strict=False)
    return head.eval()


def train_tokens():
    data = np.fromfile(MODELS / "data" / "train.bin", dtype=np.uint16)
    return data[:np.flatnonzero(data)[-1] + 1]          # хвост из нулей отрезаем


def train_head(model, steps=400, batch=16, T=256, lr=2e-3, log_every=100):
    for p in model.parameters():
        p.requires_grad_(False)
    torch.manual_seed(0)
    head = DraftHead(model.embed)
    data = train_tokens()
    gen = torch.Generator().manual_seed(0)
    params = [p for n, p in head.named_parameters() if not n.startswith("embed")]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    log = []
    for step in range(steps):
        ix = torch.randint(len(data) - T - 2, (batch,), generator=gen).tolist()
        x = torch.stack([torch.from_numpy(data[i:i + T + 2].astype("int64")) for i in ix])
        with torch.no_grad():
            _, h = hidden(model, x[:, :T])              # h_t для t = 0..T-1
        for g in opt.param_groups:
            g["lr"] = lr * min(1.0, (step + 1) / 20) * 0.5 * (1 + math.cos(math.pi * step / steps))
        logits = head(h, x[:, 1:T + 1])                 # знаем токен t+1
        loss = F.cross_entropy(logits.flatten(0, 1), x[:, 2:T + 2].flatten())   # угадываем t+2
        loss.backward()
        opt.step()
        opt.zero_grad()
        if step % log_every == 0 or step == steps - 1:
            log.append((step, loss.item()))
    head.eval()
    return head, log


# ----------------------------------------------------------------------------- черновики

def bigram_table():
    """Для каждого токена — самый частый следующий за ним токен в обучающих текстах."""
    data = train_tokens().astype(np.int64)
    keys, counts = np.unique(data[:-1] * 8192 + data[1:], return_counts=True)
    prev, nxt = keys // 8192, keys % 8192
    order = np.lexsort((-counts, prev))                 # внутри prev — по убыванию частоты
    best = np.zeros(8192, dtype=np.int64)
    first = np.ones(len(order), dtype=bool)
    first[1:] = prev[order][1:] != prev[order][:-1]
    best[prev[order][first]] = nxt[order][first]
    return torch.from_numpy(best)


def prompts(n=40, length=8):
    """Начала n отложенных историй: <bos> и первые length токенов."""
    data = val_tokens(10 ** 6)
    starts = (data == 1).nonzero().flatten().tolist()
    return [data[s:s + 1 + length] for s in starts[:n]]


@torch.no_grad()
def acceptance(model, head, early, best, n=40, new=120):
    """Доля черновиков, совпавших с жадным выбором BookLM, на её собственных жадных сказках."""
    eos = 2
    hits = {"таблица биграмм": 0, "BookLM шага 762": 0, "голова MTP": 0}
    total = 0
    for p in prompts(n):
        seq = model.generate(p[None], max_new=new, temperature=0, eos_id=eos)[0]
        if seq[-1] == eos:
            seq = seq[:-1]
        P = len(p)                                      # черновики только для написанных токенов
        _, h = hidden(model, seq[None])
        head_guess = head(h[:, :-1], seq[None, 1:]).argmax(-1)[0]   # позиция t -> токен t+2
        early_guess = early(seq[None])[0].argmax(-1)                # позиция t+1 -> токен t+2
        for t in range(P - 2, len(seq) - 2):            # угадываем seq[t+2], t+2 >= P
            target = seq[t + 2].item()
            hits["таблица биграмм"] += best[seq[t + 1]].item() == target
            hits["BookLM шага 762"] += early_guess[t + 1].item() == target
            hits["голова MTP"] += head_guess[t].item() == target
            total += 1
    return {k: v / total for k, v in hits.items()}, total


def truncate(caches, length):
    for c in caches:
        c["k"], c["v"], c["len"] = c["k"][:, :, :length], c["v"][:, :, :length], length


@torch.no_grad()
def speculative_greedy(model, head, ids, new):
    """Жадная генерация с черновиком головы, gamma = 1. Возвращает (токены, число вызовов BookLM)."""
    caches = [{"len": 0} for _ in model.blocks]
    logits, h = hidden(model, ids, caches)
    for c in caches:
        c["len"] = ids.size(1)
    out = ids[0].tolist()
    own = logits[0, -1].argmax().item()                 # токен основной модели
    hv = h[:, -1:]                                      # скрытый вектор перед ним
    calls = 1
    while len(out) - ids.size(1) < new:
        draft = head(hv, torch.tensor([[own]])).argmax(-1).item()
        out.append(own)
        logits, h = hidden(model, torch.tensor([[own, draft]]), caches)   # проверка одним вызовом
        calls += 1
        n = caches[0]["len"]
        for c in caches:
            c["len"] = n + 2
        check = logits[0, 0].argmax().item()            # что BookLM поставила бы после own
        if check == draft and len(out) - ids.size(1) < new:
            out.append(draft)                           # черновик принят
            own, hv = logits[0, 1].argmax().item(), h[:, 1:2]
        else:
            truncate(caches, n + 1)                     # выбросить черновик из кэша
            own, hv = check, h[:, 0:1]
    return out[:ids.size(1) + new], calls


@torch.no_grad()
def plain_greedy(model, ids, new):
    return model.generate(ids, max_new=new, temperature=0)[0].tolist()


@torch.no_grad()
def timing(model, head, ids, new=200):
    """Время жадной генерации без черновика и с ним и цена одного вызова BookLM на 1, 2, 4, 8 токенов."""
    ref = plain_greedy(model, ids, new)
    # время: лучшее из пяти повторов, чтобы отсечь чужую нагрузку на машину
    torch.set_num_threads(1)
    tp = min(_timed(lambda: plain_greedy(model, ids, new)) for _ in range(5))
    ts = min(_timed(lambda: speculative_greedy(model, head, ids, new)) for _ in range(5))
    print(f"время на CPU, один поток: жадно {tp * 1000:.0f} мс, с черновиком {ts * 1000:.0f} мс "
          f"({tp / ts:.2f}x)")
    # из чего складывается шаг: вызов BookLM на 1, 2, 4, 8 токенов поверх кэша из 100 и вызов головы
    caches = [{"len": 0} for _ in model.blocks]
    _, h = hidden(model, torch.tensor([ref[:100]]), caches)
    for c in caches:
        c["len"] = 100

    def call(n):
        hidden(model, torch.tensor([ref[100:100 + n]]), caches)
        truncate(caches, 100)

    def draft():
        head(h[:, -1:], torch.tensor([[ref[100]]]))
    tn = {n: _median(lambda: call(n)) for n in (1, 2, 4, 8)}
    th = _median(draft)
    print("вызов BookLM на 1, 2, 4, 8 токенов:",
          ", ".join(f"{t * 1000:.2f}" for t in tn.values()), "мс")
    print(f"вызов головы: {th * 1000:.2f} мс")


def ch14_mtp_head():
    """Рубрика «многотокенное предсказание»: голова MTP, принятые черновики, спекулятивный цикл."""
    torch.set_num_threads(4)
    tok, model = load("base")
    head_params = DraftHead(model.embed).trainable()
    print(f"BookLM: {model.num_params()} параметров, голова MTP: {head_params} "
          f"({head_params / model.num_params():.1%})")
    t0 = time.perf_counter()
    head, log = train_head(model, steps=STEPS, log_every=STEPS // 4)
    print(f"обучение головы: {STEPS} шагов по 16 окон из 256 токенов, "
          f"{time.perf_counter() - t0:.0f} с на CPU")
    for step, loss in log:
        print(f"  шаг {step:3d}: ошибка на токене t+2 {loss:.2f}")

    early = load_model("step:762")
    best = bigram_table()
    rates, total = acceptance(model, head, early, best)
    print(f"доля принятых черновиков на {total} токенах жадных сказок:")
    for name, r in rates.items():
        print(f"  {name:<16} {r:.1%}")

    start = "Жила-была маленькая лиса. Однажды она"
    ids = torch.tensor([prompt_ids(tok, start)])
    new = 200
    ref = plain_greedy(model, ids, new)
    spec, calls = speculative_greedy(model, head, ids, new)
    print(f"лиса, {new} токенов: ответ совпал с жадным: {spec == ref}, "
          f"вызовов BookLM {calls} вместо {new}")
    timing(model, head, ids, new)
    print(" ".join(tok.decode(spec[ids.size(1):ids.size(1) + 30]).split()))
    return head


def _timed(fn):
    t = time.perf_counter()
    fn()
    return time.perf_counter() - t


def _median(fn, n=200):
    times = sorted(_timed(fn) for _ in range(n))
    return times[n // 2]


def main():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        head = ch14_mtp_head()
    text = buf.getvalue()
    sys.stdout.write(text)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ch14-mtp-head.txt").write_text(text, encoding="utf-8")
    torch.save({k: v for k, v in head.state_dict().items() if not k.startswith("embed")},
               OUT / "ch14-mtp-head.pt")


if __name__ == "__main__":
    from booklm.experiments import pick
    pick({"ch14_mtp_head": ch14_mtp_head})         # --help и проверка имени
    main()                                         # txt и веса головы .pt
