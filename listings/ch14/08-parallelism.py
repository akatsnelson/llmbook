"""Листинги главы 14 «Инференс» книги «Большие языковые модели», файл 08-parallelism.

Код и вывод — ровно как в книге; текст книги сюда не входит. Листинги идут по порядку и
опираются друг на друга: запускайте файл целиком или ячейками сверху вниз (разметка # %%).
Проверка, что вывод совпал с книгой: python scripts/check_listings.py listings/ch14/08-parallelism.py
"""

# %% Листинг «Блок SwiGLU BookLM на двух картах с одним обменом» — раздел «Тензорный параллелизм в блоке трансформера»
import torch
import torch.nn.functional as F
from booklm.lab import load

tok, model = load("base")
ffn = model.blocks[0].ffn              # SwiGLU: d = 256, d_ff = 768
x = torch.randn(5, 256)                # пять токенов
with torch.no_grad():                  # без градиентов
    partial = []
    for gpu in range(2):
        ch = slice(384 * gpu, 384 * (gpu + 1))   # 384 канала из 768
        gate = x @ ffn.gate.weight[ch].T
        h = F.silu(gate) * (x @ ffn.up.weight[ch].T)
        partial.append(h @ ffn.down.weight[:, ch].T)
        y_part = partial[-1]
        print(f"GPU {gpu}: H {list(h.shape)}, Y {list(y_part.shape)}")
    y = partial[0] + partial[1]        # единственный all-reduce
    print("Y_0 + Y_1 = блок:", torch.allclose(y, ffn(x), atol=1e-6))
# Вывод в книге:
# | GPU 0: H [5, 384], Y [5, 256]
# | GPU 1: H [5, 384], Y [5, 256]
# | Y_0 + Y_1 = блок: True

# %% Листинг «BookLM на двух картах» — раздел «Тензорный параллелизм в блоке трансформера»
import threading, textwrap
from concurrent.futures import ThreadPoolExecutor
from booklm.lab import prompt_ids
from booklm.experiments.ch14 import shard, tp_forward, tp_greedy

cards = [shard(model, rank, p=2) for rank in (0, 1)]
box, barrier, sent = [None, None], threading.Barrier(2), []

def make_all_reduce(rank):
    def all_reduce(part):            # «сеть»: обмен частичными суммами
        box[rank] = part
        barrier.wait()               # обе карты выложили свои куски
        total = box[0] + box[1]
        barrier.wait()               # обе забрали сумму
        if rank == 0:
            sent.append(part.numel() * part.element_size())
        return total
    return all_reduce

def on_two_cards(job):               # job(rank, all_reduce) на карте
    with ThreadPoolExecutor(2) as pool:
        return list(pool.map(lambda r: job(r, make_all_reduce(r)),
                             (0, 1)))

text = "Мама мыла раму, а потом она пошла спать."
ids = torch.tensor([prompt_ids(tok, text)])
logits = on_two_cards(
    lambda r, ar: tp_forward(cards[r], ids, [{} for _ in range(6)], ar))
diff = (logits[0] - model(ids)).abs().max().item()
print(f"вход {ids.size(1)} токенов: обменов {len(sent)}, "
      f"в каждом {sent[0]} байт")
print(f"расхождение с цельной моделью: {diff:.0e}")
# Вывод в книге:
# | вход 13 токенов: обменов 12, в каждом 13312 байт
# | расхождение с цельной моделью: 1e-05

# %% Листинг «Жадный текст разрезанной BookLM» — раздел «Тензорный параллелизм в блоке трансформера»
sent.clear()
story = "Жила-была девочка"
texts = on_two_cards(
    lambda r, ar: tp_greedy(cards[r], tok, story, 40, ar)[0])
whole = tp_greedy(shard(model, 0, p=1), tok, story, 40, lambda x: x)[0]
print(textwrap.fill(" ".join((story + texts[0]).split()), 74))
print("цельная модель пишет то же:", texts[0] == whole)
print(f"проходов 41, обменов {len(sent)}; "
      f"на шаге ответа по {sent[-1]} байт")
# Вывод в книге:
# | Жила-была девочка по имени Люси. Она была очень любопытной и любила
# | исследовать. Однажды она решила прогуляться в парк. Когда она шла, она
# | увидела что-то странное. Это был большой, круглый предмет. Люси
# | цельная модель пишет то же: True
# | проходов 41, обменов 492; на шаге ответа по 1024 байт
