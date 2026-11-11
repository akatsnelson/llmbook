"""torch.compile на обученной BookLM без видеокарты: разрывы графа, время прохода и шага, цена компиляции.

Дополнение к подразделу «Компилятор torch.compile» главы 14 (раздел 14.7 «FlashAttention»). В книге опыт описан словами и числами, здесь он
целиком. Что меряем:

1. сколько графов и разрывов находит TorchDynamo в проходе BookLM без кэша и в шаге генерации с KV-кэшем,
   и во сколько вызовов TorchInductor превращает полторы тысячи операций aten одного прохода;
2. время прохода по 512 токенам и шага генерации (кэш на 128 позиций) в обычном режиме PyTorch (eager)
   и после torch.compile; шаг меряем дважды: с ядрами, собранными под точную длину кэша
   (dynamic=False), и с ядрами для любой длины (dynamic=True);
3. сколько раз модель компилируется за генерацию 64 токенов, когда длина кэша растёт на каждом шаге;
4. сколько стоит первый вызов: с пустым кэшем TorchInductor и с заполненным (два отдельных процесса).

Всё на процессоре и в одном потоке (torch.set_num_threads(1)): так соседние процессы меньше сбивают замер,
а сравниваются сами ядра, а не то, как каждый режим делит работу между ядрами процессора. Время — медиана
по повторам, режимы чередуются. На другой машине числа будут другими; отношения устойчивее, но и они зависят
от процессора и версии PyTorch. Опыт снят на PyTorch 2.14.1, Apple M2 Max.

Запуск из корня репозитория: python labs/ch14/torch_compile_cpu.py   (меньше минуты, половина из них — компиляция)
"""
from __future__ import annotations

import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch
import torch._dynamo as dynamo
from torch._dynamo.utils import counters
from torch._inductor.utils import run_and_get_code
from torch.profiler import ProfilerActivity, profile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from booklm.lab import load_model, val_tokens  # noqa: E402

torch.set_num_threads(1)


def first_call_seconds():
    """Время первого вызова torch.compile(model) на 512 токенах: трассировка, компиляция, сам проход."""
    model = load_model("base")
    x = val_tokens(512)[None]
    compiled = torch.compile(model)
    with torch.no_grad():
        start = time.perf_counter()
        compiled(x)
        return time.perf_counter() - start


if len(sys.argv) > 1 and sys.argv[1] == "--first-call":
    print(f"{first_call_seconds():.2f}")
    sys.exit()


def median_ms(fns, repeats=40):
    """Чередует функции, чтобы фоновая нагрузка делилась между ними поровну; медиана в мс."""
    times = [[] for _ in fns]
    for _ in range(repeats):
        for i, fn in enumerate(fns):
            start = time.perf_counter()
            fn()
            times[i].append(time.perf_counter() - start)
    return [statistics.median(t) * 1e3 for t in times]


def cache_after(model, ids, length):
    """KV-кэш BookLM после обработки первых length токенов."""
    caches = [{"len": 0} for _ in model.blocks]
    model(ids[:, :length], caches)
    for c in caches:
        c["len"] = length
    return caches


model = load_model("base")
ids = val_tokens(1024)[None]
x512 = ids[:, :512]

with torch.no_grad():
    # 1. Разрывы графа: проход без кэша и шаг с кэшем.
    report = dynamo.explain(model)(x512)
    print(f"проход 512 токенов: графов {report.graph_count}, разрывов {report.graph_break_count}")
    assert report.graph_count == 1 and report.graph_break_count == 0
    caches = cache_after(model, ids, 128)
    dynamo.reset()
    report = dynamo.explain(model)(ids[:, 128:129], [dict(c) for c in caches])
    print(f"шаг с KV-кэшем:      графов {report.graph_count}, разрывов {report.graph_break_count}")
    assert report.graph_count == 1 and report.graph_break_count == 0

    # 1б. Во что TorchInductor превратил проход: операции aten в eager против вызовов в созданном коде.
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        model(x512)
    aten_calls = sum(e.count for e in prof.key_averages() if e.key.startswith("aten::"))
    dynamo.reset()
    compiled = torch.compile(model)
    ref = model(x512)
    out, (code,) = run_and_get_code(compiled, x512)
    assert torch.allclose(out, ref, atol=1e-4)
    body = [s.strip() for s in code[code.index("def call(self, args)"):].splitlines()]
    fused = [s for s in body if s.startswith("cpp_fused")]
    mm = [s for s in body if s.startswith("extern_kernels.mm")]
    attn = [s for s in body if "_scaled_dot_product_flash_attention_for_cpu.default(" in s]
    print(f"eager: вызовов aten {aten_calls}; compile: объединённых ядер C++ {len(fused)},"
          f" умножений матриц {len(mm)}, вызовов внимания {len(attn)}")
    rms = sum("mean_mul_pow_rsqrt" in s for s in fused)
    rope = sum("slice_stack_sub" in s for s in fused)
    swiglu = sum("mul_silu" in s for s in fused)
    print(f"  среди ядер: RMSNorm {rms}, RoPE {rope}, SwiGLU {swiglu}")
    assert len(attn) == 6 and len(mm) == 43 and len(fused) < aten_calls // 10

    # То же для шага генерации при кэше на 128 позиций.
    step_in = (ids[:, 128:129], caches)
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        model(step_in[0], [dict(c) for c in caches])
    step_aten = sum(e.count for e in prof.key_averages() if e.key.startswith("aten::"))
    dynamo.reset()
    _, (code,) = run_and_get_code(torch.compile(model, dynamic=False),
                                  step_in[0], [dict(c) for c in caches])
    body = [s.strip() for s in code[code.index("def call(self, args)"):].splitlines()]
    step_calls = sum(s.startswith(("cpp_fused", "extern_kernels.")) or
                     "_scaled_dot_product_flash_attention_for_cpu.default(" in s for s in body)
    print(f"шаг с KV-кэшем: eager вызовов aten {step_aten}, compile вызовов ядер {step_calls}")
    assert step_calls < step_aten // 10

    # 2. Время прохода по 512 токенам.
    eager_ms, comp_ms = median_ms([lambda: model(x512), lambda: compiled(x512)], repeats=30)
    print(f"проход 512 токенов: eager {eager_ms:.1f} мс, compile {comp_ms:.1f} мс,"
          f" отношение {eager_ms / comp_ms:.2f}")

    # 2б. Шаг генерации при кэше на 128 позиций: копия словарей кэша, чтобы длина не росла.
    nxt = ids[:, 128:129]
    step_ref = model(nxt, [dict(c) for c in caches])
    timings = {}
    for name, dyn in (("точная длина", False), ("любая длина", True)):
        dynamo.reset()
        step = torch.compile(model, dynamic=dyn)
        assert torch.allclose(step(nxt, [dict(c) for c in caches]), step_ref, atol=1e-4)
        eager_ms, comp_ms = median_ms([lambda: model(nxt, [dict(c) for c in caches]),
                                       lambda: step(nxt, [dict(c) for c in caches])], repeats=200)
        timings[name] = (eager_ms, comp_ms)
        print(f"шаг, {name:12s}: eager {eager_ms:.2f} мс, compile {comp_ms:.2f} мс,"
              f" отношение {eager_ms / comp_ms:.2f}")

    # 3. Генерация 64 токенов с растущим кэшем: сколько графов собрано.
    dynamo.reset()
    counters.clear()
    compiled = torch.compile(model)
    caches = cache_after(compiled, ids, 64)
    for i in range(64):
        compiled(ids[:, 64 + i:65 + i], caches)
        for c in caches:
            c["len"] += 1
    graphs = counters["stats"]["unique_graphs"]
    print(f"генерация 64 токенов: собрано графов {graphs}"
          " (вход 64 токена, шаг при точной длине, шаг при любой длине)")
    assert graphs == 3

# 4. Первый вызов в новом процессе: пустой кэш TorchInductor, затем тот же кэш уже заполнен.
with tempfile.TemporaryDirectory() as cache_dir:
    env = dict(os.environ, TORCHINDUCTOR_CACHE_DIR=cache_dir)
    cmd = [sys.executable, __file__, "--first-call"]
    cold, warm = (float(subprocess.run(cmd, env=env, capture_output=True, text=True,
                                        check=True).stdout.split()[-1]) for _ in range(2))
print(f"первый вызов: пустой кэш {cold:.1f} с, заполненный {warm:.1f} с")
assert warm < cold
