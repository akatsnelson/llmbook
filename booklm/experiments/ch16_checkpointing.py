"""Глава 16 «Распределённое обучение», рубрика «пересчёт активаций» (подраздел «Пересчёт активаций»):
опыт ch16_activation_checkpointing, пересчёт блоков BookLM на CPU.

Что печатает (около 45 с на процессоре Apple M2 Max):
  * листинг книги «Пересчёт блоков BookLM»: градиенты с пересчётом и без совпадают; сколько байт
    отложено для обратного прохода всей модели с пересчётом при пакете 8 и 64 окна по 512 токенов;
  * листинг книги «Пик памяти и арифметика шага»: для шести блоков без головы на словарь, пакет 8 x 512 —
    байты к концу прямого прохода и в худший момент шага, с пересчётом и без; GFLOP полного шага,
    в том числе с use_reentrant=False;
  * время полного шага в FP32, медиана девяти запусков (на CPU время шумит и зависит от загрузки машины).

Запуск из корня репозитория:
    python -m booklm.experiments.ch16_checkpointing
Вывод сохраняется в models/booklm/experiments/ch16-activation-checkpointing.txt.
"""
import statistics
import sys
import time
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.checkpoint import checkpoint
from torch.utils.flop_counter import FlopCounterMode

if not __package__:  # запуск файлом (python booklm/experiments/ch16_checkpointing.py): корень репозитория — в путь поиска
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from booklm.experiments.ch16 import loss_on, saved_bytes, windows
from booklm.lab import load_model


class Recompute(torch.nn.Module):
    """Блок BookLM, внутренности которого не хранятся, а пересчитываются в обратном проходе."""

    def __init__(self, block, reentrant=True):
        super().__init__()
        self.block, self.reentrant = block, reentrant

    def forward(self, h, cos, sin, cache=None, return_weights=False):
        return checkpoint(self.block, h, cos, sin, use_reentrant=self.reentrant)


class _Live:
    """Считает байты отложенных тензоров, живых прямо сейчас, и наибольшее их число за шаг."""

    def __init__(self):
        self.count, self.bytes, self.peak = {}, 0, 0

    def pack(self, t):
        key = (t.untyped_storage().data_ptr(), t.dtype)
        if self.count.get(key, 0) == 0:                  # каждый кусок памяти считаем один раз
            self.bytes += t.untyped_storage().nbytes()
            self.peak = max(self.peak, self.bytes)
        self.count[key] = self.count.get(key, 0) + 1
        return _Saved(t, key, self)


class _Saved:
    def __init__(self, t, key, live):
        self.t, self.key, self.live = t, key, live

    def __del__(self):                                    # autograd отпустил тензор
        live = self.live
        live.count[self.key] -= 1
        if live.count[self.key] == 0:
            del live.count[self.key]
            live.bytes -= self.t.untyped_storage().nbytes()


def blocks_memory(model, x):
    """Шаг только через шесть блоков (без головы на словарь), под автоприведением к BF16.

    Возвращает (байт отложено к концу прямого прохода, наибольшее число байт за весь шаг).
    """
    live = _Live()
    with torch.autograd.graph.saved_tensors_hooks(live.pack, lambda s: s.t):
        h = model.embed(x)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            for block in model.blocks:
                h, _ = block(h, model.rope_cos, model.rope_sin)
        after_forward = live.bytes
    loss = h.float().pow(2).mean()                       # условная ошибка, её байты не считаем
    with torch.autograd.graph.saved_tensors_hooks(live.pack, lambda s: s.t):
        loss.backward()                                  # пересчёт блоков идёт здесь
    model.zero_grad(set_to_none=True)
    return after_forward, live.peak


def step_gflop(model, x, y):
    """Арифметика полного шага (прямой + обратный проход), млрд операций.

    Внимание считаем «математическим» путём: быстрое ядро SDPA на CPU счётчик не видит.
    """
    with sdpa_kernel(SDPBackend.MATH), FlopCounterMode(display=False) as fc:
        loss_on(model, x, y).backward()
    model.zero_grad(set_to_none=True)
    return fc.get_total_flops() / 1e9


def step_seconds(model, x, y):
    model.zero_grad(set_to_none=True)
    t0 = time.perf_counter()
    loss_on(model, x, y).backward()
    return time.perf_counter() - t0


def ch16_activation_checkpointing(repeats=9):
    """Рубрика «пересчёт активаций»: отложенные байты, пик, GFLOP и время шага."""
    torch.manual_seed(0)
    # листинг книги «Пересчёт блоков BookLM»: градиенты те же, отложено меньше
    model = load_model("init").train()
    x, y = windows(8, 512)
    loss_on(model, x, y).backward()
    before = [p.grad.clone() for p in model.parameters()]
    model.zero_grad(set_to_none=True)
    model.blocks = torch.nn.Sequential(*map(Recompute, model.blocks))
    loss_on(model, x, y).backward()
    after = [p.grad for p in model.parameters()]
    d = max((a - b).abs().max().item() for a, b in zip(after, before))
    print("градиенты с пересчётом и без расходятся на", d)
    model.zero_grad(set_to_none=True)
    for B in (8, 64):
        xb, yb = windows(B, 512)
        kept = saved_bytes(model, xb, yb) / 2**20
        print(f"пакет {B:2d} x 512: сохранено {kept:4.0f} МБ")
    del model

    # листинг книги «Пик памяти и арифметика шага» и замер времени
    plain = load_model("base").train()
    recomp = load_model("base").train()
    recomp.blocks = torch.nn.Sequential(*map(Recompute, recomp.blocks))

    print("пакет 8 x 512, шесть блоков BookLM (без головы на словарь), BF16")
    for name, m in (("без пересчёта", plain), ("с пересчётом ", recomp)):
        kept, peak = blocks_memory(m, x)
        print(f"  {name}: к концу прямого {kept / 2**20:5.0f} МБ, пик за шаг {peak / 2**20:5.0f} МБ")

    g_plain, g_rec = step_gflop(plain, x, y), step_gflop(recomp, x, y)
    print(f"арифметика полного шага: {g_plain:.0f} -> {g_rec:.0f} GFLOP (+{g_rec / g_plain - 1:.0%})")
    lazy = load_model("base").train()                    # новый режим: пересчёт с ранней остановкой
    lazy.blocks = torch.nn.Sequential(*(Recompute(b, reentrant=False) for b in lazy.blocks))
    print(f"  с use_reentrant=False: {step_gflop(lazy, x, y):.0f} GFLOP")

    print(f"время полного шага, FP32, {torch.get_num_threads()} потоков, медиана {repeats} запусков:")
    step_seconds(plain, x, y), step_seconds(recomp, x, y)             # прогрев
    tp, tr = [], []
    for _ in range(repeats):                                         # чередуем, чтобы шум делился поровну
        tp.append(step_seconds(plain, x, y))
        tr.append(step_seconds(recomp, x, y))
    mp_, mr = statistics.median(tp), statistics.median(tr)
    print(f"  без пересчёта {mp_:.2f} с (от {min(tp):.2f} до {max(tp):.2f})")
    print(f"  с пересчётом  {mr:.2f} с (от {min(tr):.2f} до {max(tr):.2f}), +{mr / mp_ - 1:.0%}")


if __name__ == "__main__":
    from booklm.experiments import run
    run({"ch16_activation_checkpointing": ch16_activation_checkpointing}, save=True)
