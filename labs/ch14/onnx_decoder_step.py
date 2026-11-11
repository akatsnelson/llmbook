"""Один шаг генерации с явным KV-кэшем: PyTorch, экспорт в ONNX, каталог моделей Triton.

Дополнение к подразделу «ONNX Runtime и Triton Inference Server» главы 14 (раздел 14.11 «Готовые среды
выполнения»). Книга ссылается сюда за примером каталога моделей Triton и командой запуска сервера.
Модель здесь — маленький учебный декодер TinyDecoderStep, а не BookLM (BookLM — в onnx_booklm_step.py).
В файле: шаг генерации, пригодный для экспорта, цикл на PyTorch с явным кэшем, экспорт с изменяемыми осями, проверка графа, сравнение PyTorch с ONNX Runtime на нескольких длинах кэша,
цикл генерации в ONNX Runtime, каталог моделей с config.pbtxt и расчёт трафика наивного сетевого клиента.

Части с ONNX выполняются, только если установлены пакеты onnx и onnxruntime; без них файл проверяет
всё остальное и сообщает, что экспорт пропущен.

Запуск из корня репозитория: python labs/ch14/onnx_decoder_step.py
"""

import math
import os
import tempfile
import warnings

import torch
import torch.nn as nn


# --- Шаг генерации, пригодный для экспорта в ONNX
class TinyDecoderStep(nn.Module):
    def __init__(self, vocab=4096, d_model=64, max_positions=2048):
        super().__init__()
        self.token = nn.Embedding(vocab, d_model)
        self.position = nn.Embedding(max_positions, d_model)
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.lm_head = nn.Linear(d_model, vocab, bias=False)

    def forward(self, input_ids, position_ids, past_k, past_v):
        # input_ids, position_ids: [B, 1]
        # past_k, past_v: [B, S, D]
        x = self.token(input_ids) + self.position(position_ids)
        q, k_new, v_new = self.q(x), self.k(x), self.v(x)
        present_k = torch.cat([past_k, k_new], dim=1)
        present_v = torch.cat([past_v, v_new], dim=1)

        scores = q @ present_k.transpose(-2, -1)
        scores = scores / math.sqrt(q.size(-1))
        context = scores.softmax(dim=-1) @ present_v
        logits = self.lm_head(self.out(context))
        return logits, present_k, present_v


# --- Цикл генерации на PyTorch: кэш входит и выходит явно
def generate_with_cache(model, first_token=17, steps=32, d_model=64):
    token = torch.tensor([[first_token]])
    past_k = torch.zeros(1, 0, d_model)
    past_v = torch.zeros(1, 0, d_model)
    tokens, all_logits = [], []
    with torch.no_grad():
        for position in range(steps):
            tokens.append(int(token))
            logits, past_k, past_v = model(token, torch.tensor([[position]]), past_k, past_v)
            all_logits.append(logits[:, -1])
            token = logits[:, -1].argmax(dim=-1, keepdim=True)
    return tokens, torch.cat(all_logits), past_k, past_v


def logits_without_cache(model, tokens):
    """Тот же расчёт без кэша: K и V всех прежних токенов пересчитываются заново на каждом шаге."""
    result = []
    with torch.no_grad():
        for t in range(len(tokens)):
            ids = torch.tensor([tokens[: t + 1]])
            pos = torch.arange(t + 1)[None]
            x = model.token(ids) + model.position(pos)
            k, v = model.k(x), model.v(x)
            q = model.q(x[:, -1:])
            weights = (q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))).softmax(-1)
            result.append(model.lm_head(model.out(weights @ v))[:, -1])
    return torch.cat(result)


torch.manual_seed(0)
model = TinyDecoderStep().eval()
tokens, cached_logits, past_k, past_v = generate_with_cache(model, steps=32)
assert past_k.shape == (1, 32, 64) and past_v.shape == (1, 32, 64)   # +1 позиция за шаг
torch.testing.assert_close(cached_logits, logits_without_cache(model, tokens), rtol=1e-4, atol=1e-4)

# Без position_ids выход меняется: позиция — честный вход графа, а не скрытое состояние.
with torch.no_grad():
    a, _, _ = model(torch.tensor([[5]]), torch.tensor([[0]]), past_k[:, :3], past_v[:, :3])
    b, _, _ = model(torch.tensor([[5]]), torch.tensor([[3]]), past_k[:, :3], past_v[:, :3])
assert not torch.allclose(a, b)


# --- PyTorch → ONNX, проверка графа и сравнение с ONNX Runtime (если пакеты установлены)
INPUT_NAMES = ["input_ids", "position_ids", "past_k", "past_v"]
OUTPUT_NAMES = ["logits", "present_k", "present_v"]


def export_onnx(model, path, batch=2, past_length=4, d_model=64):
    example = (
        torch.zeros(batch, 1, dtype=torch.long),
        torch.full((batch, 1), past_length, dtype=torch.long),
        torch.zeros(batch, past_length, d_model),
        torch.zeros(batch, past_length, d_model),
    )
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
    torch.onnx.export(
        model,
        example,
        path,
        input_names=INPUT_NAMES,
        output_names=OUTPUT_NAMES,
        dynamic_axes={
            "input_ids": {0: "batch"},
            "position_ids": {0: "batch"},
            "past_k": {0: "batch", 1: "past_length"},
            "past_v": {0: "batch", 1: "past_length"},
            "logits": {0: "batch"},
            "present_k": {0: "batch", 1: "present_length"},
            "present_v": {0: "batch", 1: "present_length"},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )


def check_onnx(path):
    import onnx

    graph = onnx.load(path)
    onnx.checker.check_model(graph)
    inputs = [value.name for value in graph.graph.input]
    outputs = [value.name for value in graph.graph.output]
    assert inputs == INPUT_NAMES, inputs
    assert outputs == OUTPUT_NAMES, outputs
    print("ops:", sorted({node.op_type for node in graph.graph.node}))


def compare_with_runtime(model, path, d_model=64):
    import numpy as np
    import onnxruntime as ort

    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    for batch in (1, 3):
        for past_length in (0, 4, 100):          # включая пустой кэш
            ids = torch.randint(0, 4096, (batch, 1))
            pos = torch.full((batch, 1), past_length)
            pk = torch.randn(batch, past_length, d_model)
            pv = torch.randn(batch, past_length, d_model)
            with torch.no_grad():
                expected = model(ids, pos, pk, pv)
            got = session.run(OUTPUT_NAMES, {
                "input_ids": ids.numpy(), "position_ids": pos.numpy(),
                "past_k": pk.numpy(), "past_v": pv.numpy(),
            })
            for e, g in zip(expected, got):
                np.testing.assert_allclose(e.numpy(), g, rtol=1e-4, atol=1e-4)
    return session


def onnx_runtime_loop(session, steps=32, d_model=64):
    """Наивный цикл: session.run возвращает NumPy, то есть кэш каждый шаг уходит с устройства и обратно."""
    import numpy as np

    token = np.array([[17]], dtype=np.int64)
    past_k = np.zeros((1, 0, d_model), dtype=np.float32)
    past_v = np.zeros((1, 0, d_model), dtype=np.float32)
    produced = []
    for position in range(steps):
        produced.append(int(token[0, 0]))
        logits, past_k, past_v = session.run(
            OUTPUT_NAMES,
            {
                "input_ids": token,
                "position_ids": np.array([[position]], dtype=np.int64),
                "past_k": past_k,
                "past_v": past_v,
            },
        )
        token = logits[:, -1].argmax(axis=-1, keepdims=True).astype(np.int64)
    return produced, past_k


# --- Минимальный каталог моделей Triton и config.pbtxt
CONFIG_PBTXT = """\
name: "tiny_decoder"
backend: "onnxruntime"
max_batch_size: 0

input [
  { name: "input_ids"    data_type: TYPE_INT64 dims: [ -1, 1 ] },
  { name: "position_ids" data_type: TYPE_INT64 dims: [ -1, 1 ] },
  { name: "past_k"       data_type: TYPE_FP32 dims: [ -1, -1, 64 ] },
  { name: "past_v"       data_type: TYPE_FP32 dims: [ -1, -1, 64 ] }
]
output [
  { name: "logits"       data_type: TYPE_FP32 dims: [ -1, 1, 4096 ] },
  { name: "present_k"    data_type: TYPE_FP32 dims: [ -1, -1, 64 ] },
  { name: "present_v"    data_type: TYPE_FP32 dims: [ -1, -1, 64 ] }
]
instance_group [{ kind: KIND_GPU count: 1 }]
"""
# max_batch_size: 0 — ось пакета у нас уже есть (первая ось тензоров); иначе Triton добавит спереди
# ещё одну свою. Число -1 в формах означает «любой размер».
#
# Запуск сервера (номер выпуска контейнера записывайте: от него зависит версия ONNX Runtime внутри):
#   docker run --gpus=all --rm -p8000:8000 -p8001:8001 \
#     -v $PWD/model_repository:/models nvcr.io/nvidia/tritonserver:<выпуск>-py3 \
#     tritonserver --model-repository=/models
#   curl -s -o /dev/null -w "%{http_code}" localhost:8000/v2/health/ready   # 200 — готов


def make_model_repository(root):
    """model_repository/tiny_decoder/{config.pbtxt, 1/model.onnx}: папка модели, версия, файл."""
    model_dir = os.path.join(root, "model_repository", "tiny_decoder")
    os.makedirs(os.path.join(model_dir, "1"), exist_ok=True)
    with open(os.path.join(model_dir, "config.pbtxt"), "w") as f:
        f.write(CONFIG_PBTXT)
    return os.path.join(model_dir, "1", "model.onnx")


# --- Антипример: KV-кэш через сеть
def naive_client_seconds_per_token(cache_bytes, link_bits_per_second):
    """Сервер отдаёт новый кэш, на следующем шаге получает его обратно: 2 * кэш за токен."""
    return 2 * cache_bytes * 8 / link_bits_per_second


GiB = 2**30
seconds = naive_client_seconds_per_token(1 * GiB, 10e9)
assert abs(seconds - 1.72) < 0.01, seconds          # «около 1,7 секунды на токен»


def kv_bytes(tokens, d_model=64, bytes_per_number=4):
    return 2 * tokens * d_model * bytes_per_number   # K и V


# Трафик наивного цикла для кэша 100 и 1000 токенов TinyDecoderStep: сумма по всем шагам.
for n in (100, 1000):
    total = sum(2 * kv_bytes(t) for t in range(n))
    print(f"кэш до {n} токенов: {total / 2**20:.1f} МБ туда и обратно за весь ответ")


with tempfile.TemporaryDirectory() as tmp:
    onnx_path = make_model_repository(tmp)
    assert os.path.exists(os.path.join(tmp, "model_repository", "tiny_decoder", "config.pbtxt"))
    try:
        import onnx  # noqa: F401
        import onnxruntime  # noqa: F401
    except ImportError:
        print("onnx/onnxruntime не установлены: экспорт и сравнение с ONNX Runtime пропущены")
    else:
        export_onnx(model, onnx_path)
        check_onnx(onnx_path)
        session = compare_with_runtime(model, onnx_path)
        produced, ort_cache = onnx_runtime_loop(session, steps=32)
        assert produced == tokens                    # ONNX Runtime выбрал те же токены, что PyTorch
        assert ort_cache.shape == (1, 32, 64)
        print("ONNX: граф проверен, выходы совпали с PyTorch")

print("onnx_decoder_step: все проверки пройдены")
