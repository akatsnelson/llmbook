"""Листинги главы 10 «Работа с промптом» книги «Большие языковые модели», файл 04-structured-output.

Код и вывод — ровно как в книге; текст книги сюда не входит. Листинги идут по порядку и
опираются друг на друга: запускайте файл целиком или ячейками сверху вниз (разметка # %%).
Проверка, что вывод совпал с книгой: python scripts/check_listings.py listings/ch10/04-structured-output.py
"""

# %% Листинг «Бор по символам» — раздел «Формальный язык и конечный автомат»
def build_dfa(words):
    """Бор по символам: delta[q] -- «символ -> следующее состояние»."""
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
    """delta*(q, text): состояние или None, если строка недопустима."""
    for ch in text:
        q = delta[q].get(ch)
        if q is None:
            return None
    return q

delta, accept = build_dfa(['{"label":"pos"}', '{"label":"neg"}'])
print(len(delta), sorted(accept))
print(walk(delta, 0, '{"label":"'), walk(delta, 0, '{"label":"s'))
# Вывод в книге:
# | 21 [15, 20]
# | 10 None

# %% Листинг «Индекс разрешённых токенов» — раздел «Индекс разрешённых токенов»
EOS = "<eos>"
GRAMMAR_VOCAB = [EOS, '{"', "label", '":"', "pos", "neg", '"}',
                 "po", 's"}', "Positive"]
tok_id = {w: i for i, w in enumerate(GRAMMAR_VOCAB)}

def allowed_tokens(q, vocab, delta, accept):
    allowed = set()
    for i, w in enumerate(vocab):
        if w == EOS:
            ok = q in accept                   # остановка только в F
        elif w == "":
            ok = False                         # служебный номер
        else:
            ok = walk(delta, q, w) is not None
        if ok:
            allowed.add(i)
    return allowed

def build_index(vocab, delta, accept):
    # один раз на пару (грамматика, токенизатор)
    return {q: allowed_tokens(q, vocab, delta, accept)
            for q in range(len(delta))}

index = build_index(GRAMMAR_VOCAB, delta, accept)
for text in ['{"label":"', '{"label":"po', '{"label":"pos"}']:
    q = walk(delta, 0, text)
    print(f"{text:16} q={q:2}", [GRAMMAR_VOCAB[i] for i in sorted(index[q])])
with_pad = build_index(GRAMMAR_VOCAB + [""], delta, accept)  # 10 -> ""
print("пустой токен разрешён в",
      sum(10 in allowed for allowed in with_pad.values()), "состояниях")
# Вывод в книге:
# | {"label":"       q=10 ['pos', 'neg', 'po']
# | {"label":"po     q=12 ['s"}']
# | {"label":"pos"}  q=15 ['<eos>']
# | пустой токен разрешён в 0 состояниях

# %% Листинг «Маска на шаге 4» — раздел «Маска логитов»
import torch
torch.set_printoptions(precision=3)

def grammar_mask(allowed, size):
    m = torch.full((size,), float("-inf"))   # m_i = -inf для запрещённых
    m[sorted(allowed)] = 0.0                 # m_i = 0 для разрешённых
    return m

z = torch.tensor([-1.0, -2.0, -1.5, -2.0, 1.5, 0.5, -1.0, -0.5, -2.0, 2.5])
q_value = walk(delta, 0, '{"label":"')
allowed = sorted(index[q_value])               # pos, neg, po
p = z.softmax(-1)
p_masked = (z + grammar_mask(allowed, len(GRAMMAR_VOCAB))).softmax(-1)
cols = [tok_id[w] for w in ("Positive", "pos", "neg", "po")]
print(p[cols])
print(p_masked[cols])
print(p[allowed].sum(), p[allowed] / p[allowed].sum())
# Вывод в книге:
# | tensor([0.601, 0.221, 0.081, 0.030])
# | tensor([0.000, 0.665, 0.245, 0.090])
# | tensor(0.332) tensor([0.665, 0.245, 0.090])

# %% Листинг «Ограниченная генерация и независимая проверка» — раздел «Ограниченная генерация»
import json
import re

def generate(logits_fn, use_mask=True, max_new_tokens=8):
    q, pieces = 0, []
    for _ in range(max_new_tokens):
        z = logits_fn(pieces)                              # [V]
        if use_mask:
            z = z + grammar_mask(index[q], len(GRAMMAR_VOCAB))     # до softmax
        i = int(torch.multinomial(z.softmax(-1), 1))
        if i == tok_id[EOS]:
            return "".join(pieces), True
        pieces.append(GRAMMAR_VOCAB[i])
        if use_mask:
            q = walk(delta, q, GRAMMAR_VOCAB[i])
    return "".join(pieces), False

def is_valid(text):                    # проверка, не знающая про автомат
    try:
        json.loads(text)
    except json.JSONDecodeError:
        return False
    return re.fullmatch(r'\{"label":"(pos|neg)"\}', text) is not None

torch.manual_seed(0)
noise = lambda pieces: 3.0 * torch.randn(len(GRAMMAR_VOCAB))   # без смысла
masked = [generate(noise) for _ in range(1000)]
free = [generate(noise, use_mask=False) for _ in range(1000)]
for name, runs in [("с маской", masked), ("без маски", free)]:
    print(name, sum(stop and is_valid(t) for t, stop in runs), "из 1000")
print(free[0][0])
print(sum(t == '{"label":"pos"}' for t, _ in masked) / 1000)
# Вывод в книге:
# | с маской 1000 из 1000
# | без маски 0 из 1000
# | poPositive":"po":"pos"}{"
# | 0.697

# %% Листинг «Порядок маски и ядерной выборки» — раздел «Маска и правила выбора токена»
def top_p_keep(probs, p):
    """Минимальный набор самых вероятных токенов с суммой не меньше p."""
    order = probs.argsort(descending=True)
    n = int((probs[order].cumsum(0) < p).sum()) + 1
    keep = torch.zeros_like(probs, dtype=torch.bool)
    keep[order[:n]] = True
    return keep

def kept(keep):
    return [GRAMMAR_VOCAB[i] for i in keep.nonzero().flatten().tolist()]

m = grammar_mask(index[q_value], len(GRAMMAR_VOCAB))
keep = top_p_keep(z.softmax(-1), 0.5)           # ядро, потом маска
print("ядро:", kept(keep), "->",
      (z.masked_fill(~keep, float("-inf")) + m).softmax(-1)[allowed])
keep = top_p_keep((z + m).softmax(-1), 0.5)     # маска, потом ядро
print("ядро:", kept(keep))
tau = 0.7
mask_first = ((z + m) / tau).softmax(-1)[allowed]
tau_first = (z / tau + m).softmax(-1)[allowed]
print(mask_first, tau_first)
# Вывод в книге:
# | ядро: ['Positive'] -> tensor([nan, nan, nan])
# | ядро: ['pos']
# | tensor([0.771, 0.185, 0.044]) tensor([0.771, 0.185, 0.044])

# %% Листинг «Две сломанные грамматики» — раздел «Ошибки в грамматике»
def masked_mass(z, allowed):
    return 1.0 - z.softmax(-1)[sorted(allowed)].sum().item()

def audit(vocab, delta, accept):
    """Достижимые по токенам состояния с пустой маской."""
    index = build_index(vocab, delta, accept)
    seen, stack = {0}, [0]
    while stack:
        q = stack.pop()
        for i in index[q]:
            nxt = None if vocab[i] == EOS else walk(delta, q, vocab[i])
            if nxt is not None and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return sorted(q for q in seen if not index[q])

# Первая ошибка: пропущена альтернатива, язык знает только pos.
delta_a, accept_a = build_dfa(['{"label":"pos"}'])
index_a = build_index(GRAMMAR_VOCAB, delta_a, accept_a)
allowed_a = index_a[walk(delta_a, 0, '{"label":"')]
z_neg = torch.tensor([-1.0, -2.0, -1.5, -2.0, 0.0, 2.0,
                      -1.0, -1.0, -2.0, -1.0])
m_a = grammar_mask(allowed_a, len(GRAMMAR_VOCAB))
lead, lead_masked = int(z_neg.argmax()), int((z_neg + m_a).argmax())
print(f"сломанная: mu = {masked_mass(z_neg, allowed_a):.3f},",
      GRAMMAR_VOCAB[lead], "->", GRAMMAR_VOCAB[lead_masked])
print(f"верная:    mu = {masked_mass(z_neg, index[q_value]):.3f}")
# Третья ошибка: тупик по токенам, словарь без "}
print(audit([w for w in GRAMMAR_VOCAB if w != '"}'], delta, accept),
      audit(GRAMMAR_VOCAB, delta, accept))
# Вывод в книге:
# | сломанная: mu = 0.870, neg -> pos
# | верная:    mu = 0.165
# | [13, 18] []

# %% Листинг «Маска грамматики в генерации BookLM» — раздел «Ошибки в грамматике»
import torch
from booklm.lab import generate, load, prompt_ids, special
from booklm.experiments.ch10 import build_trie, token_bytes, walk

tok, model = load("base")
EOS = special(tok)["<eos>"]
delta, accept = build_trie(["красный»", "синий»", "зелёный»"])  # по байтам
vocab = token_bytes(tok)                  # байты каждого из 8192 токенов
index = [[i for i, b in enumerate(vocab)
          if b and walk(delta, q, b) is not None] + [EOS] * (q in accept)
         for q in range(len(delta))]      # разрешённые токены состояния q
def show(i):                              # неполную букву печатаем байтом
    if i == EOS:
        return "<eos>"
    return vocab[i].decode(errors="backslashreplace")
print(len(delta), "состояний; в начале можно:", *map(show, index[0]))
sizes = sorted(map(len, index))           # сколько токенов разрешено
print("разрешено в состоянии, медиана:", sizes[len(sizes) // 2])
text = ("У Лили был мяч. Мама спросила: «Какого цвета твой мяч?» "
        "Лили ответила: «")
ids, q = prompt_ids(tok, text), 0
with torch.no_grad():
    while True:
        p = model(torch.tensor([ids]))[0, -1].softmax(-1)
        ok = torch.tensor(index[q])                  # маска
        nxt, lead = int(ok[p[ok].argmax()]), int(p.argmax())   # жадно
        print(f"{show(nxt):5} лидер без маски «{show(lead)}» {p[lead]:.2f},"
              f" mu = {1 - p[ok].sum():.3f}")
        if nxt == EOS:
            break
        ids.append(nxt)
        q = walk(delta, q, vocab[nxt])
free = generate(tok, model, text, temperature=0, max_new=8)
print("без маски:", free[len(text):])
print("обычная запись:", " | ".join(map(show, tok.encode("красный»").ids)))
# Вывод в книге:
# | 44 состояний; в начале можно: \xd0 \xd1 с к з си
# | разрешено в состоянии, медиана: 1
# | \xd0  лидер без маски «Я» 0.18, mu = 0.998
# | \xba  лидер без маски «\xa6» 0.92, mu = 1.000
# | ра    лидер без маски «еты» 0.09, mu = 0.997
# | с     лидер без маски « —» 0.31, mu = 1.000
# | ный   лидер без маски «».» 0.07, mu = 0.970
# | »     лидер без маски «.» 0.33, mu = 1.000
# | <eos> лидер без маски « и» 0.89, mu = 1.000
# | без маски: Я хочу играть с ним, но он
# | обычная запись: к | расный | »
