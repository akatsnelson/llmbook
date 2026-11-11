"""Глава 10, раздел 10.4 «Структурированный вывод»:
все листинги раздела до подраздела «Ошибки в грамматике» в одном файле (полная версия).

В книге код разбит на короткие листинги «Бор по символам», «Индекс разрешённых
токенов», «Маска на шаге 4», «Ограниченная генерация и независимая
проверка» и «Порядок маски и ядерной выборки». Файл самодостаточен и выполняется
в ноутбуке главы notebooks/10_constrained_decoding.ipynb.

Строит бор для языка {"label":"pos"} | {"label":"neg"}, индекс разрешённых
токенов, маску логитов, генерирует 1000 ответов из шума с маской и без неё и
проверяет, что маску нужно ставить до ядерной выборки.

Запуск из корня репозитория: python labs/ch10/constrained_decoding.py
"""

# --- Листинги «Бор по символам» и «Индекс разрешённых токенов» ------------
EOS = "<eos>"
VOCAB = [EOS, '{"', "label", '":"', "pos", "neg", '"}', "po", 's"}', "Positive"]
ids = {w: i for i, w in enumerate(VOCAB)}

def build_dfa(words):
    """Бор по символам: delta[q] — словарь «символ -> следующее состояние»."""
    delta, accept = [{}], set()
    for word in words:
        q = 0
        for ch in word:
            if ch not in delta[q]:
                delta.append({})
                delta[q][ch] = len(delta) - 1
            q = delta[q][ch]
        accept.add(q)
    return delta, accept

def walk(delta, q, text):
    """delta*(q, text): новое состояние или None, если строка уже недопустима."""
    for ch in text:
        q = delta[q].get(ch)
        if q is None:
            return None
    return q

def allowed_tokens(q, vocab, delta, accept):
    allowed = set()
    for i, w in enumerate(vocab):
        if w == EOS:
            ok = q in accept                   # остановка только в F
        elif w == "":
            ok = False                         # пустая строка: служебный номер
        else:
            ok = walk(delta, q, w) is not None
        if ok:
            allowed.add(i)
    return allowed

def build_index(vocab, delta, accept):
    # один раз на пару (грамматика, токенизатор)
    return {q: allowed_tokens(q, vocab, delta, accept) for q in range(len(delta))}

delta, accept = build_dfa(['{"label":"pos"}', '{"label":"neg"}'])
index = build_index(VOCAB, delta, accept)
q_value = walk(delta, 0, '{"label":"')
assert len(delta) == 21 and len(accept) == 2
assert index[q_value] == {ids["pos"], ids["neg"], ids["po"]}
assert {q for q, ok in index.items() if ids[EOS] in ok} == accept
assert index[walk(delta, 0, '{"label":"po')] == {ids['s"}']}
# Токен с пустой строкой (как <|user|> у BookLM) не разрешён нигде.
index_pad = build_index(VOCAB + [""], delta, accept)
assert all(len(VOCAB) not in allowed for allowed in index_pad.values())


# --- Листинг «Маска на шаге 4» --------------------------------------
import torch

def grammar_mask(allowed, size):
    m = torch.full((size,), float("-inf"))   # m_i = -inf для запрещённых
    m[sorted(allowed)] = 0.0                   # m_i = 0 для разрешённых
    return m

z = torch.tensor([-1.0, -2.0, -1.5, -2.0, 1.5, 0.5, -1.0, -0.5, -2.0, 2.5])
allowed = sorted(index[q_value])               # pos, neg, po
p = z.softmax(-1)
p_masked = (z + grammar_mask(allowed, len(VOCAB))).softmax(-1)

assert abs(p[ids["Positive"]].item() - 0.601) < 1e-3
assert abs(p[allowed].sum().item() - 0.332) < 1e-3
torch.testing.assert_close(p_masked[allowed], torch.tensor([0.665, 0.245, 0.090]),
                           atol=1e-3, rtol=0)
torch.testing.assert_close(p_masked[allowed], p[allowed] / p[allowed].sum())
assert p_masked[ids["Positive"]] == 0


# --- Листинг «Ограниченная генерация и независимая проверка» ---------------
import json
import re
import torch

def generate(logits_fn, use_mask=True, max_new_tokens=8):
    q, pieces = 0, []
    for _ in range(max_new_tokens):
        z = logits_fn(pieces)                              # [V]
        if use_mask:
            z = z + grammar_mask(index[q], len(VOCAB))     # маска до softmax
        i = int(torch.multinomial(z.softmax(-1), 1))
        if i == ids[EOS]:
            return "".join(pieces), True
        pieces.append(VOCAB[i])
        if use_mask:
            q = walk(delta, q, VOCAB[i])
    return "".join(pieces), False

PATTERN = re.compile(r'\{"label":"(pos|neg)"\}')

def is_valid(text):                    # проверка, не знающая про автомат
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return False
    return PATTERN.fullmatch(text) is not None and obj["label"] in ("pos", "neg")

torch.manual_seed(0)
noise = lambda pieces: 3.0 * torch.randn(len(VOCAB))   # «модель» без всякого смысла
masked = [generate(noise) for _ in range(1000)]
free = [generate(noise, use_mask=False) for _ in range(1000)]
assert all(stopped and is_valid(text) for text, stopped in masked)
assert sum(stopped and is_valid(text) for text, stopped in free) / 1000 < 0.01
share_pos = sum(text == '{"label":"pos"}' for text, _ in masked) / 1000
assert abs(share_pos - 2 / 3) < 0.05                   # к этому числу вернёмся


# --- Листинг «Порядок маски и ядерной выборки» -----------------------------
import torch

def top_p_keep(probs, p):
    """Минимальный набор самых вероятных токенов с суммой не меньше p."""
    order = probs.argsort(descending=True)
    n = int((probs[order].cumsum(0) < p).sum()) + 1
    keep = torch.zeros_like(probs, dtype=torch.bool)
    keep[order[:n]] = True
    return keep

m = grammar_mask(index[q_value], len(VOCAB))

# Неверный порядок: сначала ядро, потом маска.
keep = top_p_keep(z.softmax(-1), 0.5)
assert keep.nonzero().flatten().tolist() == [ids["Positive"]]
assert not (keep & (m == 0)).any()                     # пересечение пустое
wrong = (z.masked_fill(~keep, float("-inf")) + m).softmax(-1)
assert torch.isnan(wrong).all()                        # 0/0 в каждой клетке

# Верный порядок: сначала маска, потом ядро.
keep = top_p_keep((z + m).softmax(-1), 0.5)
assert keep.nonzero().flatten().tolist() == [ids["pos"]]

# Маска и температура перестановочны: (-inf)/tau = -inf.
tau = 0.7
torch.testing.assert_close(((z + m) / tau).softmax(-1), (z / tau + m).softmax(-1))
torch.testing.assert_close(((z + m) / tau).softmax(-1)[[4, 5, 7]],
                           torch.tensor([0.771, 0.185, 0.044]), atol=1e-3, rtol=0)

print("OK: constrained_decoding.py")
