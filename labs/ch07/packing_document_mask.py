"""Глава 7, раздел 7.2 «Подготовка обучающих данных», подраздел 7.2.6 «Упаковка
последовательностей»: упаковка документов в окна, маска документов и тест влияния.

Из книги «Большие языковые модели».
Запуск из корня репозитория: python labs/ch07/packing_document_mask.py
"""

# --- Упаковка, маска документов и маска ошибки ---
import math
import torch

def pack(docs, window, eos):
    # docs: списки номеров; каждое окно — window = T+1 номеров
    stream, owner = [], []
    for n, doc in enumerate(docs, start=1):
        stream += doc + [eos]
        owner += [n] * (len(doc) + 1)    # EOS — часть документа
    cut = len(stream) // window * window # хвост ждёт следующих
    tokens = torch.tensor(stream[:cut]).view(-1, window)
    doc_ids = torch.tensor(owner[:cut]).view(-1, window)
    return tokens, doc_ids, stream[cut:]

def document_causal_mask(doc_ids):
    # doc_ids: [T] -> маска [T, T], True = можно смотреть
    t = doc_ids.numel()
    causal = torch.tril(torch.ones(t, t, dtype=torch.bool))
    return causal & (doc_ids[:, None] == doc_ids[None, :])

EOS = 0
toy_docs = [[11, 12, 13, 14, 15], [21, 22, 23],
            [31, 32, 33, 34, 35, 36]]
windows, owners, rest = pack(toy_docs, window=12, eos=EOS)
assert windows.shape == (1, 12) and len(rest) == 5   # окна по T+1
assert windows.numel() + len(rest) == sum(len(d) + 1 for d in toy_docs)
x, y = windows[:, :-1], windows[:, 1:]    # сдвиг цели из главы 3
doc_x = owners[0, :-1]                    # чей токен на позиции
assert doc_x.tolist() == [1] * 6 + [2] * 4 + [3]

mask = document_causal_mask(doc_x)
expected = torch.zeros(11, 11, dtype=torch.bool)
for lo, hi in [(0, 6), (6, 10), (10, 11)]:   # три треугольника
    n = hi - lo
    expected[lo:hi, lo:hi] = torch.ones(n, n, dtype=torch.bool).tril()
assert torch.equal(mask, expected)

loss_mask = x != EOS                      # цель после EOS не учим
assert loss_mask.sum() == 9 and not loss_mask[0, 5] and y[0, 5] == 21
assert y[0, 4] == EOS and loss_mask[0, 4] # а конец документа учим


# --- Тест влияния между документами ---
def masked_attention(q, k, v, allowed):
    # q, k, v: [T, d_h]; allowed: [T, T]; маска ставится ДО softmax
    scores = q @ k.T / math.sqrt(q.size(-1))
    scores = scores.masked_fill(~allowed, float("-inf"))
    return scores.softmax(dim=-1) @ v

torch.manual_seed(0)
emb = torch.randn(40, 8)                  # 40 номеров, d = 8
w_q, w_k, w_v = (torch.randn(8, 8) for _ in range(3))
def layer(tokens, allowed):
    h = emb[tokens]                             # [T, d]
    return masked_attention(h @ w_q, h @ w_k, h @ w_v, allowed)

changed = x[0].clone()
changed[1:5] = torch.tensor([1, 2, 3, 4]) # переписали середину d1
in_d2 = doc_x == 2
causal_only = torch.tril(torch.ones(11, 11, dtype=torch.bool))
def shift(allowed):                       # сдвиг выходов d2
    diff = layer(x[0], allowed) - layer(changed, allowed)
    return diff[in_d2].abs().max().item()
assert shift(mask) == 0.0                 # с маской d2 не заметил
assert shift(causal_only) > 1e-3          # без неё d2 читает d1


print("packing_document_mask: все проверки пройдены")
