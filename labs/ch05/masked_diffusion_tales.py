"""Маскированная диффузия на сказках BookLM: обучение, оценка правдоподобия и генерация открытием масок.

Опыт к разделу 5.8 «Диффузионные языковые модели» главы 5. Берём обученную авторегрессионную BookLM,
снимаем с её внимания причинную маску и дообучаем маскированной диффузией (ошибка с весом 1/t, как у
MDLM и LLaDA); рядом та же модель учится с нуля тем же числом шагов. Затем сравниваем:

1. ошибку на отложенных окнах: точную у BookLM и верхнюю оценку у диффузионных моделей;
2. тексты, которые диффузионная модель пишет за разное число шагов S (вызовов модели), по их
   перплексии под исходной BookLM и по доле повторённых токенов;
3. заполнение середины истории, которое авторегрессионная модель напрямую не умеет.

Всё считается на процессоре: на Apple M2 Max каждая модель учится около трёх минут.

Запуск из корня репозитория: python labs/ch05/masked_diffusion_tales.py
Время: 10–15 минут на Apple M2 Max.
"""
from __future__ import annotations

import copy
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from booklm.lab import MODELS, load  # noqa: E402
from booklm.model import BookLM, BookLMConfig, apply_rope  # noqa: E402

MASK, BOS = 0, 1          # <pad> в данных не встречается: им и закрываем позиции
T = 128                   # длина окна: начало истории, <bos> и 127 токенов
SEEDS = (0, 1, 2, 3, 4)


# ----------------------------------------------------------------------------- данные

def story_windows(path, count, seed):
    """count окон по T токенов, каждое начинается с <bos> какой-нибудь истории."""
    data = torch.from_numpy(np.fromfile(path, dtype=np.uint16).astype("int64"))
    starts = (data[:-T] == BOS).nonzero().flatten()
    g = torch.Generator().manual_seed(seed)
    pick = starts[torch.randperm(len(starts), generator=g)[:count]]
    return torch.stack([data[s:s + T] for s in pick.tolist()])


# ----------------------------------------------------------------------------- двустороннее внимание

def bidirectional_forward(self, x, cos, sin, cache=None, return_weights=False):
    """Копия GQASelfAttention.forward из booklm/model.py без причинной маски: каждая позиция видит все."""
    b, t, d = x.shape
    q = self.q(x).view(b, t, self.heads, self.d_head).transpose(1, 2)
    k = self.k(x).view(b, t, self.kv_heads, self.d_head).transpose(1, 2)
    v = self.v(x).view(b, t, self.kv_heads, self.d_head).transpose(1, 2)
    q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    group = self.heads // self.kv_heads
    y = F.scaled_dot_product_attention(q, k.repeat_interleave(group, 1),
                                       v.repeat_interleave(group, 1))   # без is_causal
    return self.out(y.transpose(1, 2).reshape(b, t, d)), None



def make_bidirectional(model):
    for block in model.blocks:
        block.attention.forward = bidirectional_forward.__get__(block.attention)
    return model


def denoiser_logits(model, x):
    """Логиты для позиций 1..T-1. Как в Dream 7B, токен позиции i предсказывает выход позиции i-1:
    так модель, взятая из авторегрессионной, сохраняет смысл своих выходов."""
    return model(x)[:, :-1]                     # [B, T-1, V] для токенов x[:, 1:]


# ----------------------------------------------------------------------------- ошибка с весом 1/t

def masked_diffusion_loss(model, x0, t=None, generator=None):
    """Верхняя оценка -log p(x) на один токен: доля масок t, ошибка по закрытым позициям, вес 1/t."""
    b = x0.size(0)
    if t is None:
        t = torch.rand(b, 1, generator=generator).clamp_min(1e-3)
    hide = torch.rand(x0[:, 1:].shape, generator=generator) < t        # <bos> всегда открыт
    xt = x0.clone()
    xt[:, 1:][hide] = MASK
    ce = F.cross_entropy(denoiser_logits(model, xt).transpose(1, 2), x0[:, 1:], reduction="none")
    return ((ce * hide).sum(1) / t[:, 0] / (T - 1))                    # [B]


def train(model, steps, seed, batch=16, lr=3e-4):
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    data = story_windows(MODELS / "data" / "train.bin", steps * batch, seed)
    model.train()
    for step in range(steps):
        x0 = data[step * batch:(step + 1) * batch]
        lr_now = lr * min(1.0, (step + 1) / 50) * (0.1 + 0.9 * (1 - step / steps))
        for group in opt.param_groups:
            group["lr"] = lr_now
        loss = masked_diffusion_loss(model, x0, generator=g).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return model.eval()


@torch.no_grad()
def diffusion_bound(model, windows, draws=16):
    """Среднее по окнам и по draws расслоенным значениям t: (k + u) / draws."""
    g = torch.Generator().manual_seed(123)
    total = 0.0
    for k in range(draws):
        t = ((k + torch.rand(len(windows), 1, generator=g)) / draws).clamp_min(1e-3)
        total += masked_diffusion_loss(model, windows, t=t, generator=g).mean().item()
    return total / draws


@torch.no_grad()
def ar_loss(model, windows):
    logits = model(windows)[:, :-1]
    return F.cross_entropy(logits.transpose(1, 2), windows[:, 1:]).item()


# ----------------------------------------------------------------------------- генерация

@torch.no_grad()
def unmask_generate(model, x, steps, generator):
    """Открывает закрытые позиции x за steps вызовов модели, каждый раз самые уверенные."""
    x = x.clone()
    for s in range(steps):
        hidden = x[:, 1:] == MASK                                       # [B, T-1]
        left = int(hidden[0].sum())
        if left == 0:
            break
        n_open = math.ceil(left / (steps - s))                         # поровну на оставшиеся шаги
        probs = denoiser_logits(model, x).float().softmax(-1)          # [B, T-1, V]
        guess = torch.multinomial(probs.view(-1, probs.size(-1)), 1,
                                  generator=generator).view(hidden.shape)
        conf = probs.gather(-1, guess[..., None])[..., 0]
        conf = conf.masked_fill(~hidden, -1.0)                          # открытые не трогаем
        top = conf.topk(n_open, dim=1).indices
        body = x[:, 1:]
        body.scatter_(1, top, guess.gather(1, top))
    return x


@torch.no_grad()
def ar_sample(model, n, generator, length=T):
    x = torch.full((n, 1), BOS)
    for _ in range(length - 1):
        probs = model(x)[:, -1].float().softmax(-1)
        x = torch.cat([x, torch.multinomial(probs, 1, generator=generator)], 1)
    return x


@torch.no_grad()
def judge(ar_model, x):
    """Перплексия текста под исходной BookLM и доля токенов, повторяющих один из 8 предыдущих."""
    ppl = math.exp(ar_loss(ar_model, x))
    body = x[:, 1:]
    repeats = torch.zeros_like(body, dtype=torch.bool)
    for lag in range(1, 9):
        repeats[:, lag:] |= body[:, lag:] == body[:, :-lag]
    return ppl, repeats.float().mean().item()


# ----------------------------------------------------------------------------- опыт

def main(train_steps=800, samples=16):
    tok, ar_model = load("base")
    val = story_windows(MODELS / "data" / "val.bin", 256, seed=7)

    start = time.time()
    tuned = train(make_bidirectional(copy.deepcopy(ar_model)), train_steps, seed=0)
    t_tuned = time.time() - start
    scratch = train(make_bidirectional(BookLM(BookLMConfig())), train_steps, seed=0)

    print(f"Обучение: {train_steps} шагов по 16 окон из {T} токенов, "
          f"{t_tuned / 60:.1f} мин на одну модель\n")
    rows = [("BookLM, авторегрессия (точно)", ar_loss(ar_model, val)),
            ("диффузия из BookLM (оценка сверху)", diffusion_bound(tuned, val)),
            ("диффузия с нуля (оценка сверху)", diffusion_bound(scratch, val))]
    print(f"{'ошибка на токен, 256 окон':38}{'ошибка':>8}{'перпл.':>9}")
    for name, loss in rows:
        print(f"{name:38}{loss:8.3f}{math.exp(loss):9.2f}")

    print(f"\n{'генерация, 127 токенов':24}{'вызовов':>8}{'перпл. BookLM':>15}{'повторы':>9}")
    gen_rows, gen_ppl = {}, {}
    for name, steps in (("BookLM, по токену", None), ("диффузия", 127), ("диффузия", 32),
                        ("диффузия", 16), ("диффузия", 8), ("диффузия", 4)):
        ppls, reps = [], []
        for seed in SEEDS:
            g = torch.Generator().manual_seed(seed)
            if steps is None:
                x = ar_sample(ar_model, samples, g)
            else:
                start_x = torch.full((samples, T), MASK)
                start_x[:, 0] = BOS
                x = unmask_generate(tuned, start_x, steps, g)
            ppl, rep = judge(ar_model, x)
            ppls.append(ppl)
            reps.append(rep)
            if seed == 0:
                gen_rows[(name, steps)] = x
        p, r = torch.tensor(ppls), torch.tensor(reps)
        gen_ppl[steps] = p.mean().item()
        calls = 127 if steps is None else steps
        print(f"{name:24}{calls:8}{p.mean():9.1f} ± {p.std():4.1f}{r.mean():8.2f}")

    for key in (("диффузия", 127), ("диффузия", 8)):
        text = tok.decode(gen_rows[key][0, 1:65].tolist()).replace("\n", " ")
        print(f"\nS = {key[1]}: {text}")

    # заполнение середины: открыты начало и конец истории, середину пишет модель
    story = val[3].clone()
    x = story.clone()[None]
    x[:, 25:65] = MASK
    filled = unmask_generate(tuned, x, 40, torch.Generator().manual_seed(0))
    print("\nначало:   ", tok.decode(story[1:25].tolist()).replace("\n", " "))
    print("середина: ", tok.decode(filled[0, 25:65].tolist()).replace("\n", " "))
    print("в истории:", tok.decode(story[25:65].tolist()).replace("\n", " "))
    print("конец:    ", tok.decode(story[65:90].tolist()).replace("\n", " "))

    # что опыт показывает, независимо от точных чисел
    ar, from_booklm, from_scratch = (loss for _, loss in rows)
    assert ar < from_booklm < from_scratch          # начало с весов BookLM сильно помогает
    order = [gen_ppl[s] for s in (None, 127, 32, 16, 8, 4)]
    assert order == sorted(order)                   # меньше шагов — хуже текст
    assert (filled[0, :25] == story[:25]).all() and (filled[0, 65:] == story[65:]).all()
    print("\nmasked_diffusion_tales: все проверки пройдены")
    return rows


if __name__ == "__main__":
    main()
