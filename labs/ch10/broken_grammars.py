"""Глава 10, раздел 10.4 «Структурированный вывод», подраздел «Ошибки в грамматике»: листинг
«Две сломанные грамматики» (полная версия).

В книге оставлены проверки (а) и (в); здесь ещё и (б) — повторение без
границы {"n":[0-9]+} и маска бюджета с функцией steps_to_finish. Файл
самодостаточен (автомат, индекс и маска повторены из начала раздела) и
выполняется в ноутбуке главы notebooks/10_constrained_decoding.ipynb.

Запуск из корня репозитория: python labs/ch10/broken_grammars.py
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


# --- Маска грамматики из листинга «Маска на шаге 4» ---
import torch

def grammar_mask(allowed, size):
    m = torch.full((size,), float("-inf"))
    m[sorted(allowed)] = 0.0
    return m


# --- Листинг «Две сломанные грамматики» ------------------------------------
import json
import torch

def masked_mass(z, allowed):
    return 1.0 - z.softmax(-1)[sorted(allowed)].sum().item()

# (а) Пропущена альтернатива: язык знает только pos.
delta_a, accept_a = build_dfa(['{"label":"pos"}'])
index_a = build_index(VOCAB, delta_a, accept_a)
allowed_a = index_a[walk(delta_a, 0, '{"label":"')]
z_neg = torch.tensor([-1.0, -2.0, -1.5, -2.0, 0.0, 2.0, -1.0, -1.0, -2.0, -1.0])
leader = VOCAB[int(z_neg.argmax())]
leader_masked = VOCAB[int((z_neg + grammar_mask(allowed_a, len(VOCAB))).argmax())]
assert masked_mass(z_neg, allowed_a) > 0.8                 # отрезано 0,870
assert (leader, leader_masked) == ("neg", "pos")           # маска сменила ответ
assert masked_mass(z_neg, index[q_value]) < 0.2            # верная грамматика: 0,165

# (б) Повторение без границы: {"n":[0-9]+}, модель любит цифры.
VOCAB_N = [EOS, '{"n":', "}"] + list("0123456789")
delta_n, _ = build_dfa(['{"n":'])            # состояния 0..5, 5 — после двоеточия
digit = len(delta_n); delta_n.append({})     # «уже есть хотя бы одна цифра»
done = len(delta_n); delta_n.append({})      # после закрывающей скобки
for d in "0123456789":
    delta_n[5][d] = digit
    delta_n[digit][d] = digit                # петля: верхней границы нет
delta_n[digit]["}"] = done
accept_n = {done}
index_n = build_index(VOCAB_N, delta_n, accept_n)
z_n = torch.tensor([0.0, 0.0, 0.0] + [3.0 + 0.1 * d for d in range(10)])

def steps_to_finish(vocab, index, delta, accept):
    """d(q): наименьшее число токенов вместе с <eos> до остановки из q."""
    dist = {q: (1 if q in accept else float("inf")) for q in index}
    changed = True
    while changed:
        changed = False
        for q, allowed in index.items():
            for i in allowed:
                if vocab[i] != EOS:
                    d = 1 + dist[walk(delta, q, vocab[i])]
                    if d < dist[q]:
                        dist[q], changed = d, True
    return dist

def greedy(vocab, index, delta, z, max_new_tokens, dist=None):
    q, pieces = 0, []
    for step in range(max_new_tokens):
        left = max_new_tokens - step                 # осталось токенов вместе с этим
        allowed = index[q]
        if dist is not None:                         # маска бюджета
            allowed = {i for i in allowed if vocab[i] == EOS
                       or 1 + dist[walk(delta, q, vocab[i])] <= left}
        i = int((z + grammar_mask(allowed, len(vocab))).argmax())
        if vocab[i] == EOS:
            return "".join(pieces), q, True
        pieces.append(vocab[i])
        q = walk(delta, q, vocab[i])
    return "".join(pieces), q, False

text, q, stopped = greedy(VOCAB_N, index_n, delta_n, z_n, 12)
assert not stopped and q not in accept_n and text == '{"n":' + "9" * 11
dist_n = steps_to_finish(VOCAB_N, index_n, delta_n, accept_n)
text, q, stopped = greedy(VOCAB_N, index_n, delta_n, z_n, 12, dist_n)
assert stopped and q in accept_n and json.loads(text) == {"n": 999_999_999}

# (в) Тупик по токенам: в словаре нет токена "}.
def reachable(vocab, index, delta):
    seen, stack = {0}, [0]
    while stack:
        q = stack.pop()
        for i in index[q]:
            if vocab[i] != EOS:
                nxt = walk(delta, q, vocab[i])
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
    return seen

def audit(vocab, delta, accept):
    """Достижимые по токенам состояния с пустой маской."""
    index = build_index(vocab, delta, accept)
    return sorted(q for q in reachable(vocab, index, delta) if not index[q])

VOCAB_C = [w for w in VOCAB if w != '"}']
dead = audit(VOCAB_C, delta, accept)
assert dead == sorted([walk(delta, 0, '{"label":"pos'), walk(delta, 0, '{"label":"neg')])
empty = grammar_mask(set(), len(VOCAB_C))
assert torch.isnan((torch.zeros(len(VOCAB_C)) + empty).softmax(-1)).all()
assert audit(VOCAB, delta, accept) == []                   # исходный словарь чист

print("OK: broken_grammars.py")
