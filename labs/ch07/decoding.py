"""Глава 7, раздел 7.7 «Выбор следующего токена».

Опыт раздела целиком. Механизм (жадный выбор, штраф за повтор, лучевой поиск, температура, k лучших,
ядро, генерация с выборкой) записан в листингах книги; здесь те же функции и прогон, который печатает
числа таблицы раздела:

    1. лиса, 100 токенов: доля повторных четвёрок и средний log p на токен для жадного выбора,
       штрафа CTRL (theta = 1,2) и лучевого поиска (b = 4);
    2. средний log p настоящих сказок: 50 000 токенов проверочной части корпуса кусками по 512;
    3. та же лиса с ядром p = 0,9 на десяти зёрнах: среднее, разброс и крайние значения log p,
       log p каждого зерна и начало самой вероятной из десяти сказок.

Ручные примеры раздела (z = (2, 1, 0) при трёх температурах, пять токенов с k = 2 и p = 0,8)
закреплены утверждениями assert.

Запуск из корня репозитория: python labs/ch07/decoding.py
Нужны веса models/booklm/booklm-base.pt; меньше минуты на CPU.
"""

import statistics
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from booklm.lab import load  # noqa: E402  (обученная BookLM и её токенизатор)

WINDOW = 512                                        # длина окна, на которой учили BookLM


# --- Механизм: те же функции, что в листингах раздела ---------------------------------------

@torch.no_grad()
def generate_greedy(model, ids, steps):
    for _ in range(steps):
        logits = model(ids[:, -WINDOW:])
        best = logits[0, -1].argmax().view(1, 1)    # жадный выбор
        ids = torch.cat([ids, best], dim=1)
    return ids


@torch.no_grad()
def penalized_greedy(model, ids, steps, theta=1.2):
    for _ in range(steps):
        logits = model(ids[:, -WINDOW:])[0, -1]
        seen = ids[0].unique()                      # всё, что уже написано
        z = logits[seen]                            # штраф CTRL
        logits[seen] = torch.where(z > 0, z / theta, z * theta)
        ids = torch.cat([ids, logits.argmax().view(1, 1)], dim=1)
    return ids


@torch.no_grad()
def beam_search(model, ids, steps, beams=4):
    paths = [(0.0, ids)]                            # (сумма log p, токены)
    for _ in range(steps):
        candidates = []
        for score, seq in paths:
            logp = model(seq[:, -WINDOW:])[0, -1].log_softmax(dim=-1)
            for lp, t in zip(*logp.topk(beams)):
                longer = torch.cat([seq, t.view(1, 1)], dim=1)
                candidates.append((score + lp.item(), longer))
        paths = sorted(candidates, key=lambda c: c[0], reverse=True)[:beams]
    return paths[0][1]


def next_token_probs(logits, temperature=1.0, top_k=None, top_p=None):
    p = torch.softmax(logits / temperature, dim=-1)
    keep = torch.ones_like(p, dtype=torch.bool)
    if top_k:
        keep &= p >= p.topk(top_k).values[-1]
    if top_p:
        sorted_p, order = p.sort(descending=True)
        before = sorted_p.cumsum(dim=0) - sorted_p  # набрано до этого токена
        keep[order[before >= top_p]] = False
    p = p * keep
    return p / p.sum()


@torch.no_grad()
def generate(model, ids, steps, seed=0, **how):
    gen = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        logits = model(ids[:, -WINDOW:])[0, -1]
        p = next_token_probs(logits, **how)
        nxt = torch.multinomial(p, 1, generator=gen)
        ids = torch.cat([ids, nxt.view(1, 1)], dim=1)
    return ids


# --- Меры опыта ---------------------------------------------------------------------------

def repeated_4grams(ids):
    """Доля повторов среди четвёрок токенов, идущих подряд: 0 — ни одного повтора."""
    grams = [tuple(ids[i:i + 4]) for i in range(len(ids) - 3)]
    return 1 - len(set(grams)) / len(grams)


@torch.no_grad()
def logp_per_token(model, ids, n):
    """Средний log p, который модель ставит последним n токенам текста ids [1, T]."""
    logp = model(ids[:, :-1])[0].log_softmax(dim=-1)        # позиция i предсказывает токен i+1
    return logp[-n:].gather(1, ids[0, -n:, None]).mean().item()


def real_logp(model, n=50_000):
    """То же для настоящих сказок: n токенов проверочной части корпуса кусками по 512."""
    root = Path(__file__).resolve().parents[2]
    tales = np.fromfile(root / "models/booklm/data/val.bin", np.uint16)[:n]
    tales = torch.tensor(tales.astype(np.int64))
    return statistics.mean(logp_per_token(model, tales[None, s:s + WINDOW + 1], WINDOW)
                           for s in range(0, len(tales) - WINDOW - 1, WINDOW))


# --- Опыт -----------------------------------------------------------------------------------

def hand_checks():
    """Ручные примеры раздела."""
    z = torch.tensor([2.0, 1.0, 0.0])
    assert [round(x, 3) for x in next_token_probs(z, 1.0).tolist()] == [0.665, 0.245, 0.09]
    assert [round(x, 3) for x in next_token_probs(z, 0.5).tolist()] == [0.867, 0.117, 0.016]
    z5 = torch.tensor([0.50, 0.20, 0.15, 0.10, 0.05]).log()
    assert [round(x, 2) for x in next_token_probs(z5, top_k=2).tolist()] == [0.71, 0.29, 0, 0, 0]
    assert [round(x, 2) for x in next_token_probs(z5, top_p=0.8).tolist()] == [0.59, 0.24, 0.18, 0, 0]


def main():
    hand_checks()
    tok, model = load("base")
    fox = torch.tensor([[1] + tok.encode("Жила-была маленькая лиса. Однажды она").ids])
    start = fox.size(1)

    runs = {"жадно": generate_greedy(model, fox, 100),
            "штраф 1,2": penalized_greedy(model, fox, 100, theta=1.2),
            "луч b=4": beam_search(model, fox, 100, beams=4)}
    rep, lp = {}, {}
    for name, ids in runs.items():
        rep[name] = repeated_4grams(ids[0, start:].tolist())
        lp[name] = logp_per_token(model, ids, 100)
        print(f"{name:10} повторы {rep[name]:.2f}, log p на токен {lp[name]:.2f}")
    real = real_logp(model)
    print(f"{'сказки':10} log p на токен {real:.2f}")

    foxes = [generate(model, fox, 100, seed=s, top_p=0.9) for s in range(10)]
    logps = torch.tensor([logp_per_token(model, f, 100) for f in foxes])
    print(f"{'ядро 0,9':10} log p на токен {logps.mean():.2f} ± {logps.std():.2f}, 10 зёрен",
          f"(от {logps.min():.2f} до {logps.max():.2f})")
    print(f"{'по зёрнам':10}", " ".join(f"{x:.2f}" for x in logps))
    best = int(logps.argmax())                      # самая «простая» сказка
    best_text = tok.decode(foxes[best][0, start:].tolist())
    print(f"{'зерно ' + str(best):10} {' '.join(best_text.split())[:70]}")

    assert rep["жадно"] > 0.2                       # жадный выбор петляет
    assert rep["штраф 1,2"] < 0.05                  # штраф петлю убирает
    assert abs(lp["луч b=4"] - lp["жадно"]) < 0.05  # луч находит такой же вероятный текст
    assert real < lp["жадно"] - 1                   # люди пишут куда менее вероятно
    assert abs(logps.mean() - real) < 0.3           # выборка с ядром — как настоящие сказки
    assert "волк" in best_text.lower()              # простая сказка — про волка
    print("OK: decoding.py")


if __name__ == "__main__":
    main()
