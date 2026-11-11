"""Глава 10, раздел 10.1 «Сообщения и номера токенов», подраздел «Порядок частей промпта»: листинг «Документ до
вопроса не знает вопроса» (полная версия).

Модель — класс BookLM из главы 4 (booklm/model.py) со случайными весами в float64, шаблон — шаблон
BookLM из главы 8. Файл разделяет два механизма, из-за которых порядок частей промпта меняет ответ:

1. Причинная маска. Договор, стоящий до вопроса, получает одни и те же логиты при любом вопросе; вопрос,
   стоящий до договора, их меняет.
2. RoPE. В модели из одного блока без RoPE логиты последней позиции не зависят от порядка частей вовсе:
   ключи и значения единственного слоя считаются из эмбеддингов токенов, а сумма по ключам не зависит от их
   порядка. С RoPE та же перестановка меняет логиты: оценка q_t * k_j зависит от расстояния t - j.
   В шести блоках перестановка меняет логиты и без RoPE: со второго блока ключи несут то, что позиция
   прочитала слева, а это решает маска.

Выполняется в ноутбуке главы notebooks/10_constrained_decoding.ipynb.

Запуск из корня репозитория: python labs/ch10/causal_order.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # корень репозитория
from booklm.model import BookLM, BookLMConfig

SPECIAL = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<|user|>": 3, "<|assistant|>": 4}
BYTE_OFFSET = 5
B, U, A = SPECIAL["<bos>"], SPECIAL["<|user|>"], SPECIAL["<|assistant|>"]


def encode(text):
    # Учебный токенизатор главы 8: байт UTF-8 b -> номер 5 + b.
    return [BYTE_OFFSET + b for b in text.encode("utf-8")]


def make_model(layers=6, rope=True):
    torch.manual_seed(0)
    model = BookLM(BookLMConfig(layers=layers)).double().eval()
    if not rope:                       # поворот на нулевой угол: RoPE выключен
        model.rope_cos.fill_(1.0)
        model.rope_sin.zero_()
    return model


@torch.no_grad()
def logits(model, ids):                # [T] -> логиты каждой позиции [T, 8192]
    return model(torch.tensor([ids]))[0]


# 1. Причинная маска: договор до вопроса не знает вопроса.
model = make_model()
doc = encode("Договор: оплата до 10 числа, штраф 1% в день.")
q1, q2 = encode(" Когда платить?"), encode(" Какой штраф?")
n = 2 + len(doc)
h1 = logits(model, [B, U] + doc + q1 + [A])
h2 = logits(model, [B, U] + doc + q2 + [A])
assert torch.equal(h1[:n], h2[:n])                       # совпадают до последнего бита
span = slice(-len(doc) - 1, -1)
h3 = logits(model, [B, U] + q1 + doc + [A])
h4 = logits(model, [B, U] + q2 + doc + [A])
assert (h3[span] - h4[span]).abs().max() > 0.1          # вопрос первым — договор другой

# 2. Перестановка инструкции и вопроса: что даёт RoPE, а что маска.
instr, question = encode("Отвечай кратко. "), encode("Сколько блоков в BookLM?")
order_a = [B, U] + instr + question + [A]
order_b = [B, U] + question + instr + [A]


def last_shift(model):
    # насколько меняются логиты позиции <|assistant|> при перестановке частей
    return (logits(model, order_a)[-1] - logits(model, order_b)[-1]).abs().max().item()


one_block_no_rope = last_shift(make_model(layers=1, rope=False))
one_block_rope = last_shift(make_model(layers=1, rope=True))
six_blocks_no_rope = last_shift(make_model(layers=6, rope=False))
assert one_block_no_rope < 1e-12                         # без RoPE один слой порядка не видит
assert one_block_rope > 1e-3                             # RoPE: другие расстояния — другие логиты
assert six_blocks_no_rope > 1e-3                         # маска: со второго блока порядок важен

print(f"один блок без RoPE: {one_block_no_rope:.1e}, с RoPE: {one_block_rope:.4f}, "
      f"шесть блоков без RoPE: {six_blocks_no_rope:.4f}")
