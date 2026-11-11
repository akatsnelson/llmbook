"""Один шаг генерации BookLM с явным KV-кэшем: PyTorch, экспорт в ONNX, сверка с ONNX Runtime.

Глава 14, раздел 14.11 «Готовые среды выполнения», подраздел «ONNX Runtime и Triton Inference Server».
Листинги «Экспорт шага BookLM в ONNX», «Сверка графа ONNX с PyTorch» и «Сказка из файла ONNX» берут
отсюда BookLMDecodeStep и вспомогательные функции; глава 17 импортирует их же.

Модель из booklm/model.py как есть экспортировать нельзя: внутри model.py есть ветвления по длине кэша (`start == 0`, срез таблиц RoPE с `start`), и
трассировщик ONNX записывает их результат для примера как константу. Здесь шаг генерации переписан без
ветвлений: позиция приходит отдельным входом position_ids, поворот RoPE берёт строки таблиц по этому
номеру, а новый токен один, поэтому маска не нужна. Веса те же самые — модули модели не копируются.

Файл проверяет:
  * BookLMDecodeStep совпадает с model.forward с кэшем на нескольких шагах;
  * экспорт с изменяемыми осями B и S проходит onnx.checker, а ONNX Runtime совпадает с PyTorch
    для пакетов 1 и 3 и кэша из 0, 4 и 100 позиций;
  * граф, экспортированный без dynamic_axes, отказывается принимать кэш другой длины;
  * сколько байт кэша пересекло бы сеть на каждом токене у наивного клиента Triton.

Части с ONNX выполняются, только если установлены пакеты onnx и onnxruntime.
Запуск из корня репозитория: python labs/ch14/onnx_booklm_step.py   (работает и из любого другого каталога)
"""

import math
import os
import sys
import tempfile
import warnings
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from booklm.model import BookLM, BookLMConfig  # noqa: E402

INPUT_NAMES = ["input_ids", "position_ids", "past_k", "past_v"]
OUTPUT_NAMES = ["logits", "present_k", "present_v"]


def rope(x, cos, sin):
    # x: [B, H, 1, d_h]; cos, sin: [B, 1, 1, d_h/2] — пары (x[2k], x[2k+1]), как в главе 4
    x0, x1 = x[..., 0::2], x[..., 1::2]
    return torch.stack((x0 * cos - x1 * sin, x0 * sin + x1 * cos), dim=-1).flatten(-2)


class BookLMDecodeStep(nn.Module):
    """Шаг генерации без ветвлений: один новый токен, позиция и кэш приходят входами."""

    def __init__(self, model: BookLM):
        super().__init__()
        self.model = model
        cfg = model.cfg
        self.heads, self.kv_heads = cfg.heads, cfg.kv_heads
        self.d_head = cfg.d_model // cfg.heads

    def forward(self, input_ids, position_ids, past_k, past_v):
        # input_ids, position_ids: [B, 1]; past_k, past_v: [L, B, H_kv, S, d_h]
        m = self.model
        b = input_ids.shape[0]
        cos = m.rope_cos[position_ids].unsqueeze(1)      # [B, 1, 1, d_h/2]
        sin = m.rope_sin[position_ids].unsqueeze(1)
        h = m.embed(input_ids)                           # [B, 1, d]
        present_k, present_v = [], []
        for i, block in enumerate(m.blocks):
            att = block.attention
            x = block.norm_1(h)
            q = att.q(x).view(b, 1, self.heads, self.d_head).transpose(1, 2)
            k = att.k(x).view(b, 1, self.kv_heads, self.d_head).transpose(1, 2)
            v = att.v(x).view(b, 1, self.kv_heads, self.d_head).transpose(1, 2)
            q, k = rope(q, cos, sin), rope(k, cos, sin)
            k = torch.cat([past_k[i], k], dim=2)         # [B, H_kv, S+1, d_h]
            v = torch.cat([past_v[i], v], dim=2)
            present_k.append(k)
            present_v.append(v)
            group = self.heads // self.kv_heads
            kk = k.repeat_interleave(group, dim=1)
            vv = v.repeat_interleave(group, dim=1)
            w = (q @ kk.transpose(-2, -1) / math.sqrt(self.d_head)).softmax(-1)
            y = (w @ vv).transpose(1, 2).reshape(b, 1, -1)
            h = h + att.out(y)
            h = h + block.ffn(block.norm_2(h))
        logits = m.norm(h) @ m.embed.weight.T            # [B, 1, V]
        return logits, torch.stack(present_k), torch.stack(present_v)


def reference_step(model, input_ids, past_k, past_v):
    """Тот же шаг через model.forward с кэшем из booklm/model.py."""
    s = past_k.size(3)
    caches = [{"len": s, "k": past_k[i], "v": past_v[i]} for i in range(past_k.size(0))]
    with torch.no_grad():
        logits = model(input_ids, caches)
    return logits, torch.stack([c["k"] for c in caches]), torch.stack([c["v"] for c in caches])


def check_against_model(model, step):
    cfg = model.cfg
    shape = (cfg.layers, 1, cfg.kv_heads, 0, cfg.d_model // cfg.heads)
    past_k = past_v = torch.zeros(shape)
    ref_k = ref_v = torch.zeros(shape)
    token = torch.tensor([[1]])
    with torch.no_grad():
        for pos in range(12):
            logits, past_k, past_v = step(token, torch.tensor([[pos]]), past_k, past_v)
            ref_logits, ref_k, ref_v = reference_step(model, token, ref_k, ref_v)
            torch.testing.assert_close(logits, ref_logits, atol=2e-5, rtol=1e-4)
            torch.testing.assert_close(past_k, ref_k, atol=2e-5, rtol=1e-4)
            token = logits[:, -1].argmax(-1, keepdim=True)
    assert past_k.shape == (6, 1, 2, 12, 32)             # +1 позиция за шаг


def example_inputs(cfg, batch, past_length, seed=0):
    g = torch.Generator().manual_seed(seed)
    shape = (cfg.layers, batch, cfg.kv_heads, past_length, cfg.d_model // cfg.heads)
    return (torch.randint(0, cfg.vocab, (batch, 1), generator=g),
            torch.full((batch, 1), past_length, dtype=torch.long),
            torch.randn(shape, generator=g) * 0.5,
            torch.randn(shape, generator=g) * 0.5)


def export_step(step, path, dynamic=True, batch=2, past_length=4):
    """Экспорт шага в ONNX. dynamic=False оставляет формы примера неизменными."""
    axes = {
        "input_ids": {0: "batch"}, "position_ids": {0: "batch"},
        "past_k": {1: "batch", 3: "past_length"}, "past_v": {1: "batch", 3: "past_length"},
        "logits": {0: "batch"},
        "present_k": {1: "batch", 3: "present_length"}, "present_v": {1: "batch", 3: "present_length"},
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(step, example_inputs(step.model.cfg, batch, past_length), path,
                          input_names=INPUT_NAMES, output_names=OUTPUT_NAMES,
                          dynamic_axes=axes if dynamic else None,
                          opset_version=17, do_constant_folding=True, dynamo=False)


def max_difference(step, session, batch, past_length):
    """Наибольшее расхождение PyTorch и ONNX Runtime по трём выходам."""
    inputs = example_inputs(step.model.cfg, batch, past_length, seed=past_length + batch)
    with torch.no_grad():
        expected = step(*inputs)
    got = session.run(OUTPUT_NAMES, dict(zip(INPUT_NAMES, (t.numpy() for t in inputs))))
    return max(float(abs(e.numpy() - g).max()) for e, g in zip(expected, got))


def cache_bytes(cfg, tokens, dtype_bytes=2):
    """Байты K и V одного запроса после `tokens` позиций."""
    return 2 * cfg.layers * cfg.kv_heads * (cfg.d_model // cfg.heads) * tokens * dtype_bytes


def main():
    torch.manual_seed(0)
    model = BookLM(BookLMConfig()).eval()
    step = BookLMDecodeStep(model).eval()
    check_against_model(model, step)
    print("шаг без ветвлений совпадает с model.forward на 12 шагах")

    # Наивный клиент Triton: весь кэш туда и обратно на каждом токене.
    cfg = model.cfg
    assert cache_bytes(cfg, 1) == 1536 and cache_bytes(cfg, 512) == 786_432   # 768 КБ
    print(f"трафик наивного клиента при S=512: {2 * cache_bytes(cfg, 512) / 2**20:.1f} МБ на токен")

    try:
        import onnx
        import onnxruntime as ort
    except ImportError:
        print("onnx или onnxruntime не установлены: экспорт пропущен")
        return

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "booklm_step.onnx")
        export_step(step, path)
        graph = onnx.load(path)
        onnx.checker.check_model(graph)
        assert [v.name for v in graph.graph.input] == INPUT_NAMES
        assert [v.name for v in graph.graph.output] == OUTPUT_NAMES
        session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        for batch in (1, 3):
            for past_length in (0, 4, 100):              # включая пустой кэш
                diff = max_difference(step, session, batch, past_length)
                assert diff < 1e-4, (batch, past_length, diff)
        print("ONNX Runtime совпадает с PyTorch: B = 1, 3; S = 0, 4, 100")

        frozen = os.path.join(tmp, "booklm_step_frozen.onnx")
        export_step(step, frozen, dynamic=False)
        frozen_session = ort.InferenceSession(frozen, providers=["CPUExecutionProvider"])
        try:
            max_difference(step, frozen_session, batch=2, past_length=5)
        except Exception as error:                        # ORT ругается на форму входа
            print("граф без dynamic_axes на кэше длины 5:", type(error).__name__)
        else:
            raise AssertionError("граф с замороженными формами принял чужую длину кэша")


if __name__ == "__main__":
    main()
