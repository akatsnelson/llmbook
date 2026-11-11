"""Глава 15 «Устройство GPU»: опыты на обученной BookLM для рубрик «А теперь на нашей модели».

Видеокарты у нас нет, поэтому все опыты идут на процессоре ноутбука (Apple M2 Max) в FP32: тот же
механизм, что в главе, только «ядра» здесь — операции PyTorch на CPU. Время зависит от машины.

Опыты:
  ch15_kernel_names     рубрика «вычислительное ядро», подраздел «Библиотеки ядер»; несколько секунд.
                        Профиль PyTorch для обработки входа в 512 токенов и для одного шага генерации:
                        число вызовов операций, общее время и пять самых долгих операций.
  ch15_roofline_booklm  рубрика «модель крыши», подраздел «Характеристики ускорителя»; около 15 с.
                        Пропускная способность памяти и пиковая скорость процессора, затем для входа и шага
                        генерации при пакете 1, 8, 64: время, арифметическая интенсивность, Тфлоп/с и доля
                        крыши. Ещё пишет ch15-roofline-booklm.csv: те же точки для рисунка.
  ch15_int4_speed       рубрика «смешанная точность», раздел «Матричная инструкция и смешанная точность»;
                        около 5 с. Размер весов в BF16 и в INT4 с масштабами, ошибка весов gate
                        и время умножения на gate в FP32, BF16 и INT4 с распаковкой и без.

Запуск из корня репозитория:
    python -m booklm.experiments.ch15                    # все опыты по очереди
    python -m booklm.experiments.ch15 ch15_int4_speed    # один опыт
Вывод каждого опыта копируется в models/booklm/experiments/<имя опыта через дефис>.txt.
Ещё один опыт главы — в ch15_w4a4.py.
"""
import csv
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile

if not __package__:  # запуск файлом (python booklm/experiments/ch15.py): корень репозитория — в путь поиска
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from booklm.lab import MODELS, load_model, val_tokens

OUT = MODELS / "experiments"
N_PARAMS = 6_622_464


def new_caches(model):
    return [{"len": 0} for _ in model.blocks]


@torch.no_grad()
def prefill(model, ids):
    """Обработка входа с KV-кэшем: возвращает логиты и заполненный кэш."""
    caches = new_caches(model)
    logits = model(ids, caches)
    for c in caches:
        c["len"] = ids.size(1)
    return logits, caches


@torch.no_grad()
def decode_step(model, nxt, caches):
    """Один шаг генерации: новый токен [B, 1] и кэш, который растёт на один токен."""
    logits = model(nxt, caches)
    for c in caches:
        c["len"] += 1
    return logits


# ----------------------------------------------------------------------------- ch15-kernel-names

def top_ops(prof, k=5):
    """Число вызовов операций aten и k самых долгих по собственному времени (без вложенных вызовов)."""
    rows = [e for e in prof.key_averages() if e.key.startswith("aten::")]
    total = sum(e.self_cpu_time_total for e in rows)
    calls = sum(e.count for e in rows)
    rows.sort(key=lambda e: -e.self_cpu_time_total)
    return calls, total, [(e.key[:50], e.count, e.self_cpu_time_total / total) for e in rows[:k]]


def ch15_kernel_names():
    """Рубрика «вычислительное ядро»: профиль операций для входа и шага генерации."""
    torch.manual_seed(0)
    model = load_model("base")
    ids = val_tokens(512)[None]                     # 512 токенов отложенной истории
    with torch.no_grad():
        prefill(model, ids)                         # прогрев
        with profile(activities=[ProfilerActivity.CPU]) as p_long:
            model(ids)                              # обработка входа, T = 512
        logits, caches = prefill(model, ids[:, :511])
        nxt = logits[:, -1:].argmax(-1)
        decode_step(model, nxt, [dict(c) for c in caches])   # прогрев шага
        with profile(activities=[ProfilerActivity.CPU]) as p_step:
            decode_step(model, nxt, caches)         # шаг генерации, T = 1
    for title, prof in (("вход, T = 512", p_long), ("шаг, T = 1", p_step)):
        calls, total, rows = top_ops(prof)
        print(f"{title}: вызовов {calls}, время {total / 1e3:.2f} мс")
        for name, n, share in rows:
            print(f"  {n:3d} раз {share:6.1%}  {name[6:]}")   # без "aten::"


# ----------------------------------------------------------------------------- ch15-roofline-booklm

def best_time(fn, rounds=5, n=1):
    """Секунды на один вызов fn: лучший из rounds замеров по n вызовов.
    Берём минимум, а не среднее: другие программы на ноутбуке только замедляют замер."""
    best = float("inf")
    for _ in range(rounds):
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        best = min(best, (time.perf_counter() - t0) / n)
    return best


def measure_bw(n_bytes=1 << 30):
    """Пропускная способность памяти: копирование тензора в 1 ГБ (чтение + запись)."""
    a = torch.ones(n_bytes // 4)
    b = torch.empty_like(a)
    b.copy_(a)                                      # прогрев
    t = best_time(lambda: b.copy_(a))
    return 2 * n_bytes / t                          # байт в секунду


def measure_peak(n=4096):
    """Пик арифметики: умножение матриц n x n в FP32."""
    torch.manual_seed(0)
    a, b = torch.randn(n, n), torch.randn(n, n)
    a @ b                                           # прогрев
    t = best_time(lambda: a @ b)
    return 2 * n ** 3 / t                           # операций в секунду


def kv_bytes(L, B, bytes_per_number=4):
    return 6 * 2 * 2 * 32 * L * B * bytes_per_number  # слои * (K и V) * KV-головы * d_head


def decode_200(model, ids):
    """Среднее время шага генерации: вход 312 токенов, затем 200 шагов с кэшем."""
    logits, caches = prefill(model, ids[:, :312])
    nxt = logits[:, -1:].argmax(-1)
    t0 = time.perf_counter()
    for _ in range(200):
        nxt = decode_step(model, nxt, caches)[:, -1:].argmax(-1)
    return (time.perf_counter() - t0) / 200


def B_of(name):
    return name.split("=")[1]


def ch15_roofline_booklm():
    """Рубрика «модель крыши»: крыша процессора и точки BookLM на ней."""
    torch.manual_seed(0)
    BW, P = measure_bw(), measure_peak()
    print(f"крыша CPU: BW = {BW / 1e9:.0f} ГБ/с, P = {P / 1e12:.2f} Тфлоп/с, I* = {P / BW:.0f} оп/Б")
    model = load_model("base")
    W = N_PARAMS * 4                                # веса в FP32
    data = val_tokens(64 * 512).view(64, 512)
    rows = []
    for B in (1, 8, 64):
        ids = data[:B]
        prefill(model, ids)                         # прогрев
        t = best_time(lambda: prefill(model, ids), rounds=3)
        rows.append((f"вход, B={B}", t, 2 * N_PARAMS * B * 512, W + kv_bytes(512, B)))
        # генерация: вход 312 токенов, затем 200 шагов с кэшем (кэш растёт до 512)
        t = min(decode_200(model, ids) for _ in range(3))
        rows.append((f"шаг, B={B}", t, 2 * N_PARAMS * B, W + kv_bytes(412, B)))
    with open(OUT / "ch15-roofline-booklm.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["case", "phase", "batch", "seconds", "flops", "bytes", "intensity", "tflops", "roof_tflops",
                    "BW", "P"])
        for name, t, F_, D in rows:
            I = F_ / D
            roof = min(P, I * BW)
            print(f"{name:<10}{t * 1e3:8.2f} мс  I = {I:6.1f}  "
                  f"{F_ / t / 1e12:5.3f} Тфлоп/с  {F_ / t / roof:4.0%} крыши")
            phase = "prefill" if name.startswith("вход") else "decode"
            # в названии без запятых: их не поймёт pgfplots
            w.writerow([name.replace(",", ""), phase, B_of(name), f"{t:.6f}", F_, D, f"{I:.3f}",
                        f"{F_ / t / 1e12:.5f}", f"{roof / 1e12:.5f}", f"{BW:.4g}", f"{P:.4g}"])


# ----------------------------------------------------------------------------- ch15-int4-speed

def pack_int4(W, group=64):
    """Вес [out, in] -> байты с двумя кодами INT4 и масштаб на каждую группу из 64 весов строки.

    Коды хранятся в дополнительном коде (q % 16: -2 -> 14), как их понимает тип s4 матричных инструкций.
    В главе 14 pack_int4 сдвигает коды на +8 (-2 -> 6): байты одной функции, прочитанные другой, дают
    другие веса без всякой ошибки. Распаковывать только парной unpack_int4 из этого же файла."""
    w = W.view(W.size(0), -1, group)
    scale = w.abs().amax(-1, keepdim=True) / 7      # [out, in/64, 1]
    q = torch.round(w / scale).clamp(-8, 7).short().view_as(W)
    u = q % 16                                      # коды -8..7 как 0..15
    packed = (u[:, 0::2] | (u[:, 1::2] << 4)).to(torch.uint8)   # [out, in/2]
    return packed, scale.half()


def unpack_int4(packed, scale, group=64):
    lo, hi = (packed & 15).short(), (packed >> 4).short()
    codes = torch.stack([lo, hi], dim=-1).view(packed.size(0), -1)
    codes = torch.where(codes > 7, codes - 16, codes)
    w = codes.float().view(codes.size(0), -1, group) * scale.float()
    return w.view(codes.size(0), -1)


def time_us(fn):
    """Микросекунды на вызов: среднее по 1000 вызовам, лучший из 5 таких замеров."""
    for _ in range(50):
        fn()                                        # прогрев
    return best_time(fn, rounds=5, n=1000) * 1e6


def ch15_int4_speed():
    """Те же два листинга, что в главе: размеры весов в INT4, затем время умножения на gate."""
    torch.manual_seed(0)
    model = load_model("base")
    linears = [m.weight.data for m in model.modules() if isinstance(m, torch.nn.Linear)]
    n = sum(w.numel() for w in linears)
    packs = [pack_int4(w) for w in linears]             # группы по 64 веса
    n_bytes = sum(p.numel() + s.numel() * 2 for p, s in packs)  # масштабы FP16
    print(f"{len(linears)} линейных слоя, {n:_} весов".replace("_", " "))
    print(f"BF16: {2 * n / 2**20:.2f} МБ, INT4 с масштабами: "
          f"{n_bytes / 2**20:.2f} МБ, в {2 * n / n_bytes:.2f} раза меньше")
    W = model.blocks[0].ffn.gate.weight.data            # [768, 256], FP32
    packed, scale = pack_int4(W)
    W_q = unpack_int4(packed, scale)                    # распаковано заранее
    print(f"ошибка весов gate: {((W_q - W).norm() / W.norm()).item():.3f}")

    x = torch.randn(64, 256)                            # пакет активаций
    x16, W16 = x.bfloat16(), W.bfloat16()
    variants = [("FP32", lambda: F.linear(x, W)),
                ("BF16", lambda: F.linear(x16, W16)),
                ("INT4, распаковка на каждом вызове", lambda: F.linear(x, unpack_int4(packed, scale))),
                ("INT4, распакован заранее", lambda: F.linear(x, W_q))]
    times = [(name, time_us(fn)) for name, fn in variants]   # мкс на вызов
    for name, t in times:
        print(f"{name:<34}{t:7.1f} мкс  x{t / times[0][1]:.2f}")


EXPERIMENTS = {"ch15_kernel_names": ch15_kernel_names,
               "ch15_roofline_booklm": ch15_roofline_booklm,
               "ch15_int4_speed": ch15_int4_speed}

if __name__ == "__main__":
    from booklm.experiments import run
    OUT.mkdir(exist_ok=True)
    run(EXPERIMENTS, save=True)      # копия вывода в models/booklm/experiments/<имя опыта>.txt
