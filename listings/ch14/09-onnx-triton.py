"""Листинги главы 14 «Инференс» книги «Большие языковые модели», файл 09-onnx-triton.

Код и вывод — ровно как в книге; текст книги сюда не входит. Листинги идут по порядку и
опираются друг на друга: запускайте файл целиком или ячейками сверху вниз (разметка # %%).
Проверка, что вывод совпал с книгой: python scripts/check_listings.py listings/ch14/09-onnx-triton.py
"""

# %% Листинг «Экспорт шага BookLM в ONNX» — раздел «Экспорт шага с изменяемой длиной кэша»
import os, tempfile
import torch
from booklm.lab import load, prompt_ids
from labs.ch14.onnx_booklm_step import BookLMDecodeStep, example_inputs

INPUTS = ["input_ids", "position_ids", "past_k", "past_v"]
OUTPUTS = ["logits", "present_k", "present_v"]
AXES = {"input_ids": {0: "B"}, "position_ids": {0: "B"},
        "past_k": {1: "B", 3: "S"}, "past_v": {1: "B", 3: "S"},
        "logits": {0: "B"},
        "present_k": {1: "B", 3: "S_new"},
        "present_v": {1: "B", 3: "S_new"}}

def export_step(step, path, dynamic=True):
    example = example_inputs(step.model.cfg, batch=2, past_length=4)
    torch.onnx.export(step, example, path, input_names=INPUTS,
                      output_names=OUTPUTS, opset_version=17,
                      dynamic_axes=AXES if dynamic else None,
                      dynamo=False)

# %% Листинг «Сверка графа ONNX с PyTorch» — раздел «Экспорт шага с изменяемой длиной кэша»
import onnxruntime as ort
from labs.ch14.onnx_booklm_step import max_difference

tok, model = load("base")
decode = BookLMDecodeStep(model).eval()     # шаг без ветвлений
path = os.path.join(tempfile.gettempdir(), "booklm_step.onnx")
cpu = ["CPUExecutionProvider"]
export_step(decode, path)                   # оси B и S изменяемые
session = ort.InferenceSession(path, providers=cpu)
print([x.name for x in session.get_inputs()])
for batch, past in [(1, 0), (1, 4), (3, 100)]:
    diff = max_difference(decode, session, batch, past)
    print(f"B={batch}, S={past:3d}: совпало: {diff < 1e-4}")

export_step(decode, path, dynamic=False)    # формы примера: B=2, S=4
frozen = ort.InferenceSession(path, providers=cpu)
try:
    max_difference(decode, frozen, batch=2, past_length=5)
except Exception as error:
    print("без dynamic_axes, S=5:", type(error).__name__)
# Вывод в книге:
# | ['input_ids', 'position_ids', 'past_k', 'past_v']
# | B=1, S=  0: совпало: True
# | B=1, S=  4: совпало: True
# | B=3, S=100: совпало: True
# | без dynamic_axes, S=5: InvalidArgument

# %% Листинг «Сказка из файла ONNX» — раздел «Экспорт шага с изменяемой длиной кэша»
import textwrap
import numpy as np

export_step(decode, path)
session = ort.InferenceSession(path, providers=cpu)
tokens = prompt_ids(tok, "Жила-была девочка")
n_prompt = len(tokens)
past_k = past_v = np.zeros((6, 1, 2, 0, 32), dtype=np.float32)
steps = []
for pos in range(n_prompt + 59):          # вход по токену, затем ответ
    logits, past_k, past_v = session.run(None, {
        "input_ids": np.array([[tokens[pos]]]),
        "position_ids": np.array([[pos]]),
        "past_k": past_k, "past_v": past_v})
    steps.append(logits[0, -1])
    if pos >= n_prompt - 1:
        tokens.append(int(logits[0, -1].argmax()))
print(textwrap.fill(" ".join(tok.decode(tokens[1:]).split()), 74))
with torch.no_grad():
    full = model(torch.tensor([tokens[:-1]]))[0].numpy()
err = abs(full - np.stack(steps)).max()
print(f"шагов {len(steps)}, кэш {list(past_k.shape)}")
print(f"расхождение с полным проходом: {err:.0e}")
# Вывод в книге:
# | Жила-была девочка по имени Люси. Она была очень любопытной и любила
# | исследовать. Однажды она решила прогуляться в парк. Когда она шла, она
# | увидела что-то странное. Это был большой, круглый предмет. Люси была в
# | восторге и хотела узнать, что это такое. Она подошла ближе и увидела, что
# | шагов 64, кэш [6, 1, 2, 64, 32]
# | расхождение с полным проходом: 2e-05
