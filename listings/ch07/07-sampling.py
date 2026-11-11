"""Листинги главы 7 «Предобучение» книги «Большие языковые модели», файл 07-sampling.

Код и вывод — ровно как в книге; текст книги сюда не входит. Листинги идут по порядку и
опираются друг на друга: запускайте файл целиком или ячейками сверху вниз (разметка # %%).
Проверка, что вывод совпал с книгой: python scripts/check_listings.py listings/ch07/07-sampling.py
"""

# %% Листинг «Жадный выбор на сказке про лису» — раздел «Выбор следующего токена»
import textwrap
import torch
from booklm.lab import load

tok, model = load("base")                       # итоговые веса прогона

@torch.no_grad()                                 # граф не записываем
def generate_greedy(model, ids, steps):
    for _ in range(steps):
        logits = model(ids[:, -512:])            # не длиннее окна
        best = logits[0, -1].argmax().view(1, 1) # самый вероятный
        ids = torch.cat([ids, best], dim=1)
    return ids

def story(head, ids, start, lines=None):         # печать продолжения
    text = " ".join(tok.decode(ids[0, start.size(1):].tolist()).split())
    wrapped = textwrap.wrap(head + text, 70, subsequent_indent="  ")
    print("\n".join(wrapped[:lines]))

fox = "Жила-была маленькая лиса. Однажды она"
fox = torch.tensor([[1] + tok.encode(fox).ids])  # 1 = <bos>
story("жадно: ", generate_greedy(model, fox, 48), fox)
# Вывод в книге:
# | жадно: гуляла по лесу и увидела что-то странное. Это был большой,
# |   круглый предмет. Лиса была любопытной и хотела узнать, что это
# |   такое. Она подошла ближе и увидела, что это маленький, круглый
# |   предмет. Лиса была любопытной и хотела узнать,

# %% Листинг «Штраф за повтор» — раздел «Выбор следующего токена»
@torch.no_grad()
def penalized_greedy(model, ids, steps, theta=1.2):
    for _ in range(steps):
        logits = model(ids[:, -512:])[0, -1]
        seen = ids[0].unique()                   # всё, что уже написано
        z = logits[seen]                         # штраф CTRL
        logits[seen] = torch.where(z > 0, z / theta, z * theta)
        ids = torch.cat([ids, logits.argmax().view(1, 1)], dim=1)
    return ids

# %% Листинг «Лучевой поиск» — раздел «Выбор следующего токена»
@torch.no_grad()
def beam_search(model, ids, steps, beams=4):
    paths = [(0.0, ids)]                         # (сумма log p, токены)
    for _ in range(steps):
        candidates = []
        for score, seq in paths:                 # продлеваем каждый луч
            logp = model(seq[:, -512:])[0, -1].log_softmax(dim=-1)
            for lp, t in zip(*logp.topk(beams)): # его beams лучших токенов
                longer = torch.cat([seq, t.view(1, 1)], dim=1)
                candidates.append((score + lp.item(), longer))
        paths = sorted(candidates, key=lambda c: c[0],
                       reverse=True)[:beams]     # оставляем beams лучших
    return paths[0][1]

# %% Листинг «Лиса со штрафом и с лучами» — раздел «Выбор следующего токена»
story("луч b=4: ", beam_search(model, fox, 100, beams=4), fox, lines=2)
story("штраф 1,2: ", penalized_greedy(model, fox, 100), fox, lines=6)
# Вывод в книге:
# | луч b=4: гуляла по лесу и увидела что-то странное. Это был большой,
# |   круглый предмет. Лиса была любопытной и хотела знать, что это такое.
# | штраф 1,2: гуляла по лесу и увидела что-то странное. Это был большой,
# |   круглый предмет! Лиса была любопытной и хотела узнать больше. Она
# |   подошла ближе к нему и заметила, что это маленький красный мяч. Лиса
# |   сказала: «Привет!» Он улыбнулся ей и сказал: «Я очень старый, но я
# |   могу помочь тебе найти его». Лисёнок обрадовалась и спросила: «Как
# |   тебя зовут?» Цветок ответил: «Меня зовут Роза. Я живу в этом дереве

# %% Листинг «Температура, k лучших и ядро» — раздел «Выбор следующего токена»
def next_token_probs(logits, temperature=1.0, top_k=None, top_p=None):
    p = torch.softmax(logits / temperature, dim=-1)   # температура
    keep = torch.ones_like(p, dtype=torch.bool)
    if top_k:                                         # k лучших
        keep &= p >= p.topk(top_k).values[-1]
    if top_p:                                         # ядро
        sorted_p, order = p.sort(descending=True)
        before = sorted_p.cumsum(dim=0) - sorted_p    # набрано до
        keep[order[before >= top_p]] = False
    p = p * keep                                      # хвост в ноль
    return p / p.sum()                                # снова сумма 1

# %% Листинг «Проверка на ручных примерах» — раздел «Выбор следующего токена»
z = torch.tensor([2.0, 1.0, 0.0])
for tau in (0.5, 1.0, 2.0):
    p = next_token_probs(z, tau)
    print(f"tau={tau}:", [round(x, 3) for x in p.tolist()])
z5 = torch.tensor([0.50, 0.20, 0.15, 0.10, 0.05]).log()
for how in (dict(top_k=2), dict(top_p=0.8)):
    p = next_token_probs(z5, **how)
    print(how, [round(x, 2) for x in p.tolist()])
# Вывод в книге:
# | tau=0.5: [0.867, 0.117, 0.016]
# | tau=1.0: [0.665, 0.245, 0.09]
# | tau=2.0: [0.506, 0.307, 0.186]
# | {'top_k': 2} [0.71, 0.29, 0.0, 0.0, 0.0]
# | {'top_p': 0.8} [0.59, 0.24, 0.18, 0.0, 0.0]

# %% Листинг «Шесть лучших продолжений «Жила-была»» — раздел «Выбор следующего токена»
tale = torch.tensor([[1] + tok.encode("Жила-была").ids])
with torch.no_grad():
    logits = model(tale)[0, -1]              # оценки следующего токена
taus = (0.5, 1.0, 1.5)
probs = [next_token_probs(logits, t) for t in taus]
print("токен         tau=0.5  tau=1.0  tau=1.5")
for i in probs[1].topk(6).indices.tolist():
    row = "  ".join(f"{p[i]:7.3f}" for p in probs)
    print(f"{tok.decode([i]):12}", row)
tail = [1 - p.topk(6).values.sum() for p in probs]
print(f"{'остальные':12}", "  ".join(f"{r:7.3f}" for r in tail))
core = [next_token_probs(logits, t, top_p=0.9).count_nonzero()
        for t in taus]
print(f"{'ядро p=0.9':12}", "  ".join(f"{int(c):7d}" for c in core))
# Вывод в книге:
# | токен         tau=0.5  tau=1.0  tau=1.5
# |  маленькая     0.806    0.539    0.197
# |  девочка       0.191    0.263    0.122
# |  мама          0.001    0.019    0.021
# |  добрая        0.001    0.016    0.019
# |  в             0.000    0.008    0.012
# |  с             0.000    0.008    0.011
# | остальные      0.001    0.147    0.618
# | ядро p=0.9         2       19      773

# %% Листинг «Генерация с выборкой» — раздел «Выбор следующего токена»
@torch.no_grad()
def generate(model, ids, steps, seed=0, **how):
    gen = torch.Generator().manual_seed(seed)       # зерно
    for _ in range(steps):
        logits = model(ids[:, -512:])[0, -1]
        p = next_token_probs(logits, **how)
        nxt = torch.multinomial(p, 1, generator=gen)  # бросок
        ids = torch.cat([ids, nxt.view(1, 1)], dim=1)
    return ids

story("жадно: ", generate_greedy(model, tale, 20), tale)
for tau in (0.3, 1.0, 1.5):
    out = generate(model, tale, 20, temperature=tau, top_p=0.9)
    story(f"tau={tau}, top_p=0.9: ", out, tale)
story("лиса, top_p=0.9: ", generate(model, fox, 48, top_p=0.9), fox)
# Вывод в книге:
# | жадно: маленькая девочка по имени Люси. Она любила играть в саду.
# |   Однажды она нашла большую, блестящую камень.
# | tau=0.3, top_p=0.9: маленькая девочка по имени Люси. Она любила играть
# |   в саду. Однажды она увидела что-то блестящее в
# | tau=1.0, top_p=0.9: девочка. У неё был набор вещей. Она была очень
# |   смелой и им нравилось играть с ним.
# | tau=1.5, top_p=0.9: раная, которая возила чёрное сено. Везде, куда она
# |   проходила, были разные собаки
# | лиса, top_p=0.9: гуляла в лесу и увидела живого лиса. Она была очень
# |   смелой и решила сделать его сильным. Он хотел подойти ближе и
# |   коснуться лису, но лис попросил её подойти ближе. Желтоза ответила:
# |   «Да, я не
