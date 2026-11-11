"""Глава 16 «Распределённое обучение»: опыты на обученной BookLM для рубрик «А теперь на нашей модели».

Всё идёт на CPU: «ранги» и «карты» здесь — процессы или просто куски тензоров в одном процессе.

Опыты (время — на процессоре ноутбука Apple M2 Max):
  ch16_ddp_two_processes  рубрика «параллелизм данных», раздел «Параллелизм данных»; около 10 с.
                          Листинг книги «Два ранга BookLM на gloo»: размер градиентов и разница среднего
                          градиента двух процессов с эталоном при частях 4 + 4, 3 + 5 и 3 + 5 с весами 3/8
                          и 5/8. Затем то, чего в листинге нет: время all-reduce, наибольший градиент эталона
                          и разница весов с эталоном после шага AdamW.
  ch16_booklm_memory      рубрика «память обучения», раздел «Память обучения»; около 25 с.
                          Байты весов, градиентов и двух моментов AdamW; сколько байт прямой проход откладывает
                          для обратного при пакете 8 и 64 окна по 512 токенов.
  ch16_zero_booklm        рубрика «ZeRO», подраздел «Учебный шаг ZeRO»; несколько секунд. Листинг книги
                          «ZeRO-1 на BookLM и ограничение нормы»: тензоры, параметры и моменты AdamW двух
                          «рангов», разница с эталоном без ограничения нормы, с общей нормой и со своей,
                          нормы фрагментов.
  ch16_tp_swiglu          рубрика «тензорный параллелизм», раздел «Разбиение модели»; несколько секунд.
                          Листинг книги «Обратный проход SwiGLU на двух картах»: градиент по входу у каждой
                          половины каналов и у их суммы, градиенты весов против среза полного, байты весов
                          и одного all-reduce.
Пересчёт активаций (рубрика «пересчёт активаций») — в ch16_checkpointing.py.

Запуск из корня репозитория:
    python -m booklm.experiments.ch16                     # все опыты по очереди
    python -m booklm.experiments.ch16 ch16_zero_booklm    # один опыт
Вывод каждого опыта копируется в models/booklm/experiments/<имя опыта через дефис>.txt.
Старое имя ch17_zero_booklm тоже работает.
"""
import copy
import os
import sys
import tempfile
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

if not __package__:  # запуск файлом (python booklm/experiments/ch16.py): корень репозитория — в путь поиска
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from booklm.lab import MODELS, load, load_model, prompt_ids, val_tokens

OUT = MODELS / "experiments"
N_PARAMS = 6_622_464


def num(n):
    return f"{n:,}".replace(",", " ")             # 6 622 464


def windows(n, T):
    """n окон по T токенов из отложенных историй: входы x и ответы y (сдвиг на один токен)."""
    data = val_tokens(n * T + 1)
    x = torch.stack([data[i * T:(i + 1) * T] for i in range(n)])
    y = torch.stack([data[i * T + 1:(i + 1) * T + 1] for i in range(n)])
    return x, y


def loss_on(model, x, y):
    return F.cross_entropy(model(x).flatten(0, 1), y.flatten())


# ----------------------------------------------------------------------------- ch16-ddp-two-processes

def ddp_worker(rank, parts, fair, out_dir):
    """Один ранг DDP, как в листинге книги «Два ранга BookLM на gloo»: своя часть окон, градиент
    умножаем на долю ранга в среднем (1/2 или len(mine)/8), all-reduce, затем шаг AdamW."""
    torch.set_num_threads(2)
    dist.init_process_group("gloo", rank=rank, world_size=2, init_method="tcp://127.0.0.1:29531")
    model = load_model("base")
    x, y = windows(8, 128)
    mine = parts[rank]                                   # номера окон этого ранга
    loss_on(model, x[mine], y[mine]).backward()
    t0 = time.perf_counter()
    for p in model.parameters():
        p.grad *= len(mine) / 8 if fair else 1 / 2
        dist.all_reduce(p.grad)                          # сумма по двум рангам
    seconds = time.perf_counter() - t0
    grads = [p.grad.clone() for p in model.parameters()]
    torch.optim.AdamW(model.parameters(), lr=1e-4).step()
    if rank == 0:
        torch.save({"grads": grads, "weights": [p.detach() for p in model.parameters()],
                    "seconds": seconds}, os.path.join(out_dir, "rank0.pt"))
    dist.destroy_process_group()


def run_ddp(parts, fair=False):
    with tempfile.TemporaryDirectory() as tmp:
        mp.spawn(ddp_worker, args=(parts, fair, tmp), nprocs=2)
        return torch.load(os.path.join(tmp, "rank0.pt"))


def max_diff(a, b):
    return max((u - v).abs().max().item() for u, v in zip(a, b))


def ch16_ddp_two_processes():
    """Листинг книги «Два ранга BookLM на gloo» и веса после шага AdamW."""
    half, uneven = [[0, 1, 2, 3], [4, 5, 6, 7]], [[0, 1, 2], [3, 4, 5, 6, 7]]
    runs = {"4 + 4, среднее средних": run_ddp(half),
            "3 + 5, среднее средних": run_ddp(uneven),
            "3 + 5, веса 3/8 и 5/8": run_ddp(uneven, fair=True)}
    ref = load_model("base")                             # эталон: один процесс, 8 окон
    x, y = windows(8, 128)
    loss_on(ref, x, y).backward()
    ref_g = [p.grad.clone() for p in ref.parameters()]
    torch.optim.AdamW(ref.parameters(), lr=1e-4).step()
    ref_w = [p.detach() for p in ref.parameters()]
    print(f"градиенты: {sum(g.numel() * 4 for g in ref_g) / 2**20:.1f} МБ")
    for name, r in runs.items():
        print(f"{name}: разница с эталоном {max_diff(r['grads'], ref_g):.1e}")
    # то, что в листинг книги не попало: время обмена, самый большой градиент, веса после шага AdamW
    print(f"all-reduce при частях 4 + 4: {runs['4 + 4, среднее средних']['seconds'] * 1e3:.0f} мс на CPU (gloo)")
    print(f"наибольший |g| эталона: {max(g.abs().max().item() for g in ref_g):.2f}")
    print("разница весов с эталоном после шага AdamW, lr = 1e-4:")
    for name, r in runs.items():
        print(f"  {name}: {max_diff(r['weights'], ref_w):.1e}")


# ----------------------------------------------------------------------------- ch16-booklm-memory

def saved_bytes(model, x, y):
    """Байты тензоров, которые прямой проход сохранил для обратного (каждый тензор один раз)."""
    seen = {}

    def pack(t):
        seen[(t.untyped_storage().data_ptr(), t.dtype)] = t.untyped_storage().nbytes()
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        with torch.autocast("cpu", dtype=torch.bfloat16):   # как в train.py, только на CPU
            logits = model(x)
        loss = F.cross_entropy(logits.float().flatten(0, 1), y.flatten())
    del loss, logits
    return sum(seen.values())


def ch16_booklm_memory():
    """Рубрика «память обучения»: байты состояния и отложенных активаций."""
    torch.manual_seed(0)
    model = load_model("init").train()
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=2e-3, betas=(0.9, 0.95), eps=1e-8)  # как в train.py (без fused: CPU)
    x, y = windows(8, 512)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        logits = model(x)
    F.cross_entropy(logits.float().flatten(0, 1), y.flatten()).backward()
    opt.step()

    def nbytes(ts):
        return sum(t.numel() * t.element_size() for t in ts)

    params = list(model.parameters())
    groups = {"веса (FP32)": nbytes(params),
              "градиенты": nbytes(p.grad for p in params),
              "exp_avg": nbytes(opt.state[p]["exp_avg"] for p in params),
              "exp_avg_sq": nbytes(opt.state[p]["exp_avg_sq"] for p in params)}
    state = sum(groups.values())
    print(f"параметров {num(sum(p.numel() for p in params))}")
    for name, b in groups.items():
        print(f"  {name:<13}{b / 2**20:7.1f} МБ")
    print(f"  {'всего':<13}{state / 2**20:7.1f} МБ = {state / sum(p.numel() for p in params):.0f} байт на параметр")
    model.zero_grad(set_to_none=True)
    for B in (8, 64):
        x, y = windows(B, 512)
        act = saved_bytes(model, x, y)
        print(f"сохранено для обратного прохода, пакет {B:2d} x 512: {act / 2**20:7.0f} МБ"
              f" = {act / state:.1f} состояния модели")


# ----------------------------------------------------------------------------- ch16-zero-booklm

def ch16_zero_booklm():
    """Листинг книги «ZeRO-1 на BookLM и ограничение нормы»: состояние AdamW на двух «рангах»."""
    model = load_model("base")
    x, y = windows(8, 128)
    loss_on(model, x, y).backward()
    sizes = [p.numel() for p in model.parameters()]
    cut = next(i for i in range(len(sizes))
               if sum(sizes[:i + 1]) >= N_PARAMS / 2) + 1
    ranks = [range(cut), range(cut, len(sizes))]         # два «ранга» ZeRO-1
    for r, own in enumerate(ranks):
        n = sum(sizes[i] for i in own)
        print(f"ранг {r}: {len(own)} тензоров, {num(n)} параметров,"
              f" моменты AdamW {8 * n / 2**20:.1f} МБ")

    def step(groups, clip=None):                         # у каждой группы свой AdamW
        ps = list(copy.deepcopy(model).parameters())
        for p, q in zip(ps, model.parameters()):
            p.grad = q.grad.clone()                      # deepcopy не копирует .grad
        norms = [torch.cat([ps[i].grad.flatten() for i in g]).norm()
                 for g in groups]
        total = sum(n ** 2 for n in norms) ** 0.5        # all-reduce квадратов
        for g, n in zip(groups, norms):
            limit = {"общая": total, "своя": n}.get(clip, 0)
            for i in g:
                ps[i].grad /= max(1.0, limit)            # норма не больше 1,0
            torch.optim.AdamW([ps[i] for i in g], lr=1e-4).step()
        return ps, norms

    for clip in (None, "общая", "своя"):
        ref, _ = step([range(len(sizes))], clip)         # эталон: один AdamW
        zero, norms = step(ranks, clip)
        d = max((a - b).abs().max().item() for a, b in zip(ref, zero))
        print(f"норма {clip or 'не ограничена'}: разница с эталоном {d:.1e}")
    print("нормы фрагментов:", [round(n.item(), 2) for n in norms],
          "общая:", round(sum(n ** 2 for n in norms).item() ** 0.5, 2))
    print(f"всего моменты AdamW {8 * N_PARAMS / 2**20:.1f} МБ = 8 байт x {num(N_PARAMS)} параметров")


# ----------------------------------------------------------------------------- ch16-tp-swiglu

def swiglu(x, gate, up, down):
    return F.linear(F.silu(F.linear(x, gate)) * F.linear(x, up), down)


def ch16_tp_swiglu():
    """Листинг книги «Обратный проход SwiGLU на двух картах»."""
    tok, model = load("base")
    block, got = model.blocks[0], {}
    hook = block.norm_2.register_forward_hook(lambda m, i, o: got.update(x=o))
    text = "Мама мыла раму, а потом она пошла спать."
    with torch.no_grad():
        model(torch.tensor([prompt_ids(tok, text)]))
    hook.remove()
    torch.manual_seed(0)                                 # зерно после загрузки: load тоже берёт случайные числа
    r = torch.randn_like(got["x"])                       # градиент ошибки по выходу блока

    def backward_on(ch):                                 # «карта» с каналами ch из 768
        f = block.ffn
        W = [f.gate.weight[ch], f.up.weight[ch], f.down.weight[:, ch]]
        W = [w.detach().clone().requires_grad_() for w in W]
        x = got["x"].clone().requires_grad_()
        (swiglu(x, *W) * r).sum().backward()             # ошибка sum(y * r)
        return x.grad, [w.grad for w in W]

    gx, gW = backward_on(slice(0, 768))                  # весь блок на одной карте
    halves = [slice(0, 384), slice(384, 768)]
    parts = [backward_on(ch) for ch in halves]
    print(f"вход {list(gx.shape)}, наибольший |dL/dx| = {gx.abs().max():.2f}")
    print("на карте: gate, up, down", [list(w.shape) for w in parts[0][1]])
    for c, (ch, (hx, hW)) in enumerate(zip(halves, parts)):
        full = [gW[0][ch], gW[1][ch], gW[2][:, ch]]
        dw = max((a - b).abs().max().item() for a, b in zip(hW, full))
        print(f"карта {c}: |dL/dx - полный| = {(hx - gx).abs().max():.2f},"
              f" градиенты весов - срез полного = {dw}")
    s = parts[0][0] + parts[1][0]                        # это и делает all-reduce
    print(f"сумма двух карт: |dL/dx - полный| = {(s - gx).abs().max():.0e}")
    print(f"веса SwiGLU: {3 * 256 * 768 * 4} байт,",
          f"all-reduce: {gx.numel() * 4} байт")


EXPERIMENTS = {"ch16_ddp_two_processes": ch16_ddp_two_processes,
               "ch16_booklm_memory": ch16_booklm_memory,
               "ch16_zero_booklm": ch16_zero_booklm,
               "ch16_tp_swiglu": ch16_tp_swiglu,
               "ch17_zero_booklm": ch16_zero_booklm}            # старое имя

if __name__ == "__main__":
    from booklm.experiments import run
    OUT.mkdir(exist_ok=True)
    run(EXPERIMENTS, save=True)      # копия вывода в models/booklm/experiments/<имя опыта>.txt
