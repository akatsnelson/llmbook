"""Глава 15 «Устройство GPU», рубрика «W4A4» (подраздел «Четырёхбитная арифметика Blackwell»):
опыт ch15_w4a4, BookLM в W4A4.

Матричное устройство Blackwell умножает в FP4, только если в четырёх битах оба операнда: и веса, и
активации (схема W4A4). Здесь мы имитируем такое умножение на процессоре: перед каждым линейным слоем
округляем вход блоками вдоль общей размерности K (как это делает инструкция с блочным масштабом), а веса
слоя — блоками вдоль той же K. Считаем в FP32, так что это «ложное квантование»: значения те же, что
дала бы карта, а скорости нет.

Варианты: BF16 (эталон), FP8 E4M3 W8A8 с масштабом на тензор (так считают на H100), NVFP4 и MXFP4 только
для весов (W4A16) и для весов и активаций (W4A4). Перплексия — на тех же 200 отложенных историях, что
и в опыте с квантованием главы 14 (story_perplexity из ch14.py).

Печатает перплексию каждого варианта и во сколько раз наибольший модуль больше типичного
у входов линейных слоёв и у их весов.

Запуск из корня репозитория (около 35 с на процессоре Apple M2 Max):
    python -m booklm.experiments.ch15_w4a4
Вывод сохраняется в models/booklm/experiments/ch15-w4a4.txt.
"""
import sys
from pathlib import Path

import torch

if not __package__:  # запуск файлом (python booklm/experiments/ch15_w4a4.py): корень репозитория — в путь поиска
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from booklm.experiments.ch14 import bf16_copy, linear_layers, story_perplexity
from booklm.lab import MODELS

OUT = MODELS / "experiments"
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])   # модули четырёхбитных кодов


def e2m1_round(x):
    """Ближайшее значение E2M1 со знаком, насыщение до 6; при равенстве — чётная мантисса."""
    a = x.abs().clamp(max=6.0)
    d = (a[..., None] - E2M1).abs()
    tie = (d > d.min(-1, keepdim=True).values) * 2 + torch.arange(8) % 2
    return E2M1[tie.argmin(-1)] * x.sign()


def fp4_blocks(x, block, scale_kind):
    """Блоки по block чисел вдоль последней оси (это K умножения), коды E2M1 и масштаб блока.

    scale_kind = "nvfp4": масштаб блока в E4M3 и общий масштаб FP32 на тензор;
    scale_kind = "mxfp4": масштаб блока — степень двойки (E8M0).
    """
    shape = x.shape
    b = x.reshape(-1, shape[-1] // block, block)
    amax = b.abs().amax(-1, keepdim=True)
    if scale_kind == "nvfp4":
        t = (x.abs().max() / (6 * 448)).clamp(min=1e-12)               # общий масштаб FP32
        s = (amax / 6 / t).clamp(max=448).to(torch.float8_e4m3fn).float() * t
    else:
        s = 2.0 ** (torch.floor(torch.log2(amax.clamp(min=2.0 ** -126))) - 2)
    s = torch.where(s > 0, s, torch.ones_like(s))                      # блок из одних нулей
    return (e2m1_round(b / s) * s).reshape(shape)


def fp8_tensor(x):
    """FP8 E4M3 с одним масштабом на тензор: наибольший модуль переходит в 448."""
    s = (x.abs().max() / 448).clamp(min=1e-12)
    return (x / s).to(torch.float8_e4m3fn).float() * s


VARIANTS = {   # имя: (квантователь весов, квантователь входа слоя или None)
    "BF16": (None, None),
    "FP8 W8A8": (fp8_tensor, fp8_tensor),
    "NVFP4 W4A16": (lambda w: fp4_blocks(w, 16, "nvfp4"), None),
    "MXFP4 W4A16": (lambda w: fp4_blocks(w, 32, "mxfp4"), None),
    "NVFP4 W4A4": (lambda w: fp4_blocks(w, 16, "nvfp4"), lambda x: fp4_blocks(x, 16, "nvfp4")),
    "MXFP4 W4A4": (lambda w: fp4_blocks(w, 32, "mxfp4"), lambda x: fp4_blocks(x, 32, "mxfp4")),
}


def quantized_booklm(name):
    """Обученная BookLM (веса в BF16), у которой линейные слои считают как на карте с типом name."""
    wq, aq = VARIANTS[name]
    model = bf16_copy("base")
    for lin in linear_layers(model):
        if wq is not None:
            with torch.no_grad():
                lin.weight.copy_(wq(lin.weight))
        if aq is not None:                                             # вход слоя квантуется на лету
            lin.register_forward_pre_hook(lambda mod, inp: (aq(inp[0]),))
    return model


@torch.no_grad()
def spikes(model, n=512):
    """Во сколько раз наибольший модуль больше типичного (среднего квадратичного): max по 42 слоям.

    Возвращает два числа: для входов линейных слоёв на n отложенных токенах и для их весов.
    """
    from booklm.lab import val_tokens
    ratio = lambda t: (t.abs().max() / t.pow(2).mean().sqrt()).item()
    seen = []
    hooks = [lin.register_forward_pre_hook(lambda mod, inp: seen.append(ratio(inp[0])))
             for lin in linear_layers(model)]
    model(val_tokens(n)[None])
    for h in hooks:
        h.remove()
    return max(seen), max(ratio(lin.weight) for lin in linear_layers(model))


def ch15_w4a4():
    """Рубрика «W4A4»: перплексия BookLM в FP8, NVFP4 и MXFP4, выбросы активаций."""
    torch.set_num_threads(8)
    for name in VARIANTS:
        print(f"{name:<12} перплексия {story_perplexity(quantized_booklm(name)):.2f}")
    act, weight = spikes(quantized_booklm("BF16"))
    print(f"наибольшее/типичное: входы слоёв {act:.0f}, веса {weight:.0f}")


if __name__ == "__main__":
    from booklm.experiments import run
    run({"ch15_w4a4": ch15_w4a4}, save=True)
