"""Листинги главы 10 «Работа с промптом» книги «Большие языковые модели», файл 02-in-context-learning.

Код и вывод — ровно как в книге; текст книги сюда не входит. Листинги идут по порядку и
опираются друг на друга: запускайте файл целиком или ячейками сверху вниз (разметка # %%).
Проверка, что вывод совпал с книгой: python scripts/check_listings.py listings/ch10/02-in-context-learning.py
"""

# %% Листинг «Индукционный поиск при замороженных весах» — раздел «Индукционные головы»
import torch
torch.set_printoptions(precision=5, sci_mode=False)

VOCAB = ["кот", "cat", "дом", "house", "кошка", "здание", "kitty"]
ID = {w: i for i, w in enumerate(VOCAB)}
W_Q = W_K = W_V = torch.eye(len(VOCAB))     # веса головы: [V, V]

def induction_step(tokens, beta=10.0, lam=0.0):
    x = torch.eye(len(VOCAB))[[ID[w] for w in tokens]]   # one-hot: [T, V]
    t = len(tokens) - 1                # запрос стоит последним
    q = beta * (x[t] @ W_Q)            # [V]
    keys = x[:-1] @ W_K                # ключ: сосед слева
    values = x[1:] @ W_V               # значение: сам токен
    # расстояние t - j от кандидата до запроса; для 5 токенов [3, 2, 1, 0]
    dist = torch.arange(t - 1, -1, -1, dtype=torch.float)
    shares = (keys @ q - lam * dist).softmax(dim=-1)       # [t]
    out = shares @ values              # [V]
    return VOCAB[int(out.argmax())], shares

answer, shares = induction_step(["кот", "cat", "дом", "house", "кот"])
print(answer, shares)
answer, shares = induction_step(["кот", "кошка", "дом", "здание", "кот"])
print(answer, shares)
# Вывод в книге:
# | cat tensor([0.99986, 0.00005, 0.00005, 0.00005])
# | кошка tensor([0.99986, 0.00005, 0.00005, 0.00005])

# %% Листинг «Индукционные головы BookLM» — раздел «Индукционные головы»
import torch
import torch.nn.functional as F
from booklm.lab import load, load_model

tok, model = load("base")
torch.manual_seed(0)
rand = torch.randint(5, 8192, (30,))           # 30 случайных номеров
ids = torch.cat([torch.tensor([1]), rand, rand])[None]   # [1, 61]

@torch.no_grad()
def induction_score(model):
    logits, weights = model(ids, return_weights=True)
    att = torch.stack([w[0] for w in weights])  # [6, 8, 61, 61]
    t = torch.arange(31, 60)                    # позиции второй копии
    # доля внимания на t-29: там то, что шло после такого же токена
    return att[:, :, t, t - 29].mean(-1), att, logits

score, att, logits = induction_score(model)
for b in range(6):
    print(f"блок {b + 1}:", " ".join(f"{s:.2f}" for s in score[b].tolist()))
loss = F.cross_entropy(logits[0, :-1], ids[0, 1:], reduction="none")
first, second = loss[1:30].mean(), loss[31:60].mean()
print(f"ошибка: первая копия {first:.2f}, вторая {second:.2f}")
t = torch.arange(2, 61)
prev = att[:, :, t, t - 1].mean(-1)            # доля на соседа слева
b, h = divmod(int(prev.argmax()), 8)
print(f"голова предыдущего токена: блок {b + 1}, голова {h + 1},",
      f"доля {prev[b, h]:.2f}")
steps = (100, 762, 1525, 3051)                 # сохранения по ходу обучения
best = [induction_score(load_model(f"step:{s}"))[0].max() for s in steps]
print("лучшая оценка по шагам:",
      *[f"{s}: {v:.2f}" for s, v in zip(steps, best)])
# Вывод в книге:
# | блок 1: 0.01 0.02 0.02 0.02 0.02 0.02 0.02 0.02
# | блок 2: 0.01 0.00 0.00 0.00 0.00 0.01 0.00 0.00
# | блок 3: 0.04 0.00 0.03 0.01 0.02 0.00 0.01 0.02
# | блок 4: 0.02 0.03 0.02 0.02 0.02 0.03 0.04 0.05
# | блок 5: 0.00 0.00 0.02 0.00 0.01 0.01 0.00 0.02
# | блок 6: 0.03 0.02 0.02 0.04 0.01 0.02 0.01 0.01
# | ошибка: первая копия 15.34, вторая 15.44
# | голова предыдущего токена: блок 2, голова 2, доля 0.60
# | лучшая оценка по шагам: 100: 0.04 762: 0.04 1525: 0.06 3051: 0.05

# %% Листинг «Логит-линза BookLM» — раздел «Внутренние представления модели»
from booklm.lab import prompt_ids

@torch.no_grad()
def residual_stream(ids, add=None, at=None):
    h = model.embed(torch.tensor([ids]))     # [1, T, 256]
    states = [h[0]]                          # слой 0: эмбеддинги
    for i, block in enumerate(model.blocks):
        if i == at:
            h = h + add                      # понадобится ниже
        h, _ = block(h, model.rope_cos, model.rope_sin)
        states.append(h[0])
    return torch.stack(states)               # [7, T, 256]

@torch.no_grad()
def lens(h):                                 # черновик слоя -> 2 слова
    top = (model.norm(h) @ model.embed.weight.T).softmax(-1).topk(2)
    words = [tok.decode([i]).strip() for i in top.indices.tolist()]
    return ", ".join(f"{w} {v:.2f}" for w, v in zip(words, top.values))

texts = ["Однажды маленькая девочка по имени Лили пошла в",
         "Жила-была девочка по имени Мия. Однажды"]
s = [residual_stream(prompt_ids(tok, t))[:, -1] for t in texts]
for layer in range(7):
    print(f"слой {layer}: {lens(s[0][layer]):<22} | {lens(s[1][layer])}")
# Вывод в книге:
# | слой 0: в 1.00, на 0.00        | Однажды 1.00, Каждый 0.00
# | слой 1: в 0.28, парк 0.25      | солнечным 0.60, вечером 0.22
# | слой 2: парк 0.76, магазин 0.06 | утром 0.26, вечером 0.25
# | слой 3: парк 0.91, в 0.02      | , 0.17, она 0.15
# | слой 4: парк 0.90, магазин 0.04 | Мия 0.94, она 0.02
# | слой 5: парк 0.87, магазин 0.08 | Мия 0.47, она 0.39
# | слой 6: парк 0.76, магазин 0.10 | Мия 0.76, она 0.12

# %% Листинг «Линейный зонд рода героя» — раздел «Внутренние представления модели»
import random
from booklm.experiments.ch10_representations import GIRLS, BOYS, PHRASES, fit_probe

names = [n for n in GIRLS + BOYS for _ in PHRASES]
X = torch.stack([residual_stream(prompt_ids(tok, ph.format(n)))[:, -1]
                 for n in GIRLS + BOYS for ph in PHRASES])  # [800,7,256]
y = torch.tensor([float(n in GIRLS) for n in names])
random.seed(0)
coin = {n: random.randint(0, 1) for n in GIRLS + BOYS}
y_coin = torch.tensor([float(coin[n]) for n in names])  # контрольная задача

new_names = [torch.tensor([n in GIRLS[i::4] + BOYS[i::4] for n in names])
             for i in range(4)]              # 4 части по 10 новых имён
new_phrases = torch.arange(len(names)) % len(PHRASES) >= 15
print("слой  род   монетка  монетка на знакомых именах")
for k in range(7):
    real = sum(fit_probe(X[:, k], y, m) for m in new_names) / 4
    ctrl = sum(fit_probe(X[:, k], y_coin, m) for m in new_names) / 4
    seen = fit_probe(X[:, k], y_coin, new_phrases)
    print(f"{k:>4}  {real:.2f}  {ctrl:.2f}     {seen:.2f}")
# Вывод в книге:
# | слой  род   монетка  монетка на знакомых именах
# |    0  0.50  0.50     0.57
# |    1  0.61  0.50     0.62
# |    2  0.90  0.59     0.64
# |    3  0.90  0.63     0.57
# |    4  0.89  0.50     0.83
# |    5  0.93  0.50     0.74
# |    6  0.91  0.52     0.76

# %% Листинг «Вектор управления на BookLM» — раздел «Внутренние представления модели»
v = X[y == 1, 5].mean(0) - X[y == 0, 5].mean(0)  # «девочка» минус «мальчик»
size = X[:, 5].norm(dim=-1).mean()
print(f"длина v {v.norm():.1f}, длина вектора слоя {size:.1f}")
she, he = tok.encode(" она").ids[0], tok.encode(" он").ids[0]
start = prompt_ids(tok, "Жил-был мальчик по имени Тим. Однажды")
for alpha in (0, 2, 4, 8):
    ids = list(start)
    for step in range(7):
        h = residual_stream(ids, add=alpha * v, at=5)[-1, -1]
        p = (model.norm(h) @ model.embed.weight.T).softmax(-1)
        if step == 0:
            odds = f"она {p[she]:.2f}, он {p[he]:.2f}"
        ids.append(int(p.argmax()))
    print(f"alpha={alpha}: {odds} |{tok.decode(ids[len(start):])}")
# Вывод в книге:
# | длина v 7.3, длина вектора слоя 37.9
# | alpha=0: она 0.00, он 0.17 | Тим пошёл в парк с мамой.
# | alpha=2: она 0.12, он 0.02 | у неё появилась отличная идея. Она
# | alpha=4: она 0.42, он 0.00 | она пошла в парк. Она увидела
# | alpha=8: она 0.10, он 0.00 | Люсия Люсия Люсия Люсия Люсия Люсия Люсия

# %% Листинг «Нейроны FFN четвёртого блока BookLM» — раздел «Механистическая интерпретируемость»
import numpy as np
from booklm.lab import MODELS

def windows(name, n):                        # n кусков корпуса по 512 токенов
    data = np.fromfile(MODELS / "data" / name, dtype=np.uint16)
    return [data[i * 512:(i + 1) * 512].tolist() for i in range(n)]

def context(flat, pos, left=4):              # «...left токенов[токен]»
    text = tok.decode(flat[max(0, pos - left):pos])
    return f"{text}[{tok.decode([flat[pos]])}]".replace("\n", "/")

val = windows("val.bin", 40)
flat = sum(val, [])                          # все 20 480 позиций подряд
acts = []
hook = model.blocks[3].ffn.down.register_forward_pre_hook(
    lambda module, inputs: acts.append(inputs[0][0]))   # вход W_down: [512, 768]
with torch.no_grad():
    for w in val:
        model(torch.tensor([w]))
hook.remove()
A = torch.cat(acts)                          # [20480, 768]
for n in range(3):
    top = A[:, n].topk(4).indices.tolist()
    print(f"нейрон {n}:", " | ".join(context(flat, p, left=3) for p in top))
# Вывод в книге:
# | нейрон 0: дишься,[ ты] | , розовый велосипед[ и] |  У Тома была[ большая] | ованность,[ ты]
# | нейрон 1: . Всё город[с] |  использовала свои твор[чес] |  была очень твор[чес] |  Они шли по[ мяг]
# | нейрон 2:  акварель[ными] |  пружине[ шли] |  два брата-[бли] |  следовать за кошкой[ везде]

# %% Листинг «Суперпозиция в игрушечной модели» — раздел «Механистическая интерпретируемость»
n, d = 5, 2                                   # 5 признаков, 2 числа
importance = 0.7 ** torch.arange(n)           # 1, 0.7, 0.49, ...

def train_toy(S, seed, steps=2000):
    g = torch.Generator().manual_seed(seed)
    W = (0.5 * torch.randn(d, n, generator=g)).requires_grad_()   # [2, 5]
    b = torch.zeros(n, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=0.01)
    for _ in range(steps):
        on = torch.rand(1024, n, generator=g) > S         # признак включён?
        x = torch.rand(1024, n, generator=g) * on          # [1024, 5]
        x_hat = torch.relu(x @ W.T @ W + b)   # сжали в 2 числа и развернули
        loss = (importance * (x - x_hat) ** 2).sum(-1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return W.detach(), b.detach()

for S in (0.0, 0.8, 0.97):
    runs = [train_toy(S, seed) for seed in range(5)]
    kept = [int((W.norm(dim=0) > 0.5).sum()) for W, _ in runs]
    W, b = runs[0]                            # зерно 0: углы направлений
    on = W.norm(dim=0) > 0.5
    deg = (torch.atan2(W[1], W[0]).rad2deg() % 360)[on]
    print(f"S={S}: уложено {kept}, углы {sorted(round(a) for a in deg.tolist())},",
          f"сдвиг {b[on].mean():.2f}")
# Вывод в книге:
# | S=0.0: уложено [2, 2, 2, 2, 2], углы [199, 289], сдвиг 0.00
# | S=0.8: уложено [4, 4, 4, 4, 4], углы [59, 149, 238, 328], сдвиг 0.01
# | S=0.97: уложено [5, 5, 5, 5, 4], углы [60, 135, 203, 267, 344], сдвиг -0.23

# %% Листинг «Разреженный автокодировщик» — раздел «Разреженный автокодировщик»
class SAE(torch.nn.Module):
    def __init__(self, d, m):
        super().__init__()
        self.enc = torch.nn.Linear(d, m)      # W_enc и b_enc
        self.dec = torch.nn.Linear(m, d)      # W_dec и b_dec
    def forward(self, x):                     # x: [N, d]
        f = torch.relu(self.enc(x - self.dec.bias))   # [N, m], почти все нули
        return f, self.dec(f)                 # активации и x_hat

def sae_loss(sae, x, lam):
    f, x_hat = sae(x)
    length = sae.dec.weight.norm(dim=0)       # длины направлений d_i: [m]
    return ((x - x_hat) ** 2).sum(-1).mean() + lam * (f * length).sum(-1).mean()

D = torch.tensor([[1.0, 0.0], [-0.5, 0.866], [-0.5, -0.866]])   # d_1, d_2, d_3
toy = SAE(d=2, m=3)
with torch.no_grad():
    toy.enc.weight.copy_(D)                   # PyTorch хранит W_enc транспонированной
    toy.dec.weight.copy_(D.T)                 # и W_dec тоже: столбец i --- d_i
    toy.enc.bias.zero_(); toy.dec.bias.zero_()
    x = 0.8 * D[:1]
    for b in (0.0, -0.2):
        toy.enc.bias.fill_(b)
        f, x_hat = toy(x)
        print(f"b_enc={b}: f={[round(v, 3) for v in f[0].tolist()]},",
              f"ошибка {sae_loss(toy, x, lam=1.0):.3f}")
# Вывод в книге:
# | b_enc=0.0: f=[0.8, 0.0, 0.0], ошибка 0.800
# | b_enc=-0.2: f=[0.6, 0.0, 0.0], ошибка 0.640

# %% Листинг «Обучение автокодировщика на BookLM» — раздел «Разреженный автокодировщик»
X_sae = torch.cat([residual_stream(w)[3] for w in windows("train.bin", 200)])
print("векторов для обучения:", tuple(X_sae.shape))      # после 3-го блока
torch.manual_seed(0)
sae = SAE(d=256, m=1024)
sae.dec.bias.data = X_sae.mean(0)             # b_dec начинаем со среднего вектора
opt = torch.optim.Adam(sae.parameters(), lr=2e-3)
for step in range(3000):
    x = X_sae[torch.randint(len(X_sae), (256,))]          # пакет: [256, 256]
    loss = sae_loss(sae, x, lam=2.0)
    opt.zero_grad(); loss.backward(); opt.step()
# Вывод в книге:
# | векторов для обучения: (102400, 256)

# %% Листинг «Качество автокодировщика BookLM» — раздел «Разреженный автокодировщик»
with torch.no_grad():
    V = torch.cat([residual_stream(w)[3] for w in val])   # отложенные: [20480, 256]
    f_val, V_hat = sae(V)
    fvu = ((V - V_hat) ** 2).sum() / ((V - V.mean(0)) ** 2).sum()
    alive = torch.zeros(1024, dtype=torch.bool)
    for x in X_sae.split(10_000):
        alive |= (sae(x)[0] > 0).any(0)       # включался хоть раз
print(f"необъяснённый разброс {fvu:.3f}, включено в среднем",
      f"{(f_val > 0).float().sum(-1).mean():.1f} из 1024, мёртвых {int((~alive).sum())}")

@torch.no_grad()
def text_loss(change):                        # ошибка модели, если подменить вектор
    total = 0.0
    for w in val:
        h = residual_stream(w)[3]
        out = residual_stream(w, add=change(h) - h, at=3)[-1]
        logits = model.norm(out) @ model.embed.weight.T
        total += F.cross_entropy(logits[:-1], torch.tensor(w[1:])).item()
    return total / len(val)

print(f"ошибка модели: как есть {text_loss(lambda h: h):.3f},",
      f"с автокодировщиком {text_loss(lambda h: sae(h)[1]):.3f},",
      f"со средним вектором {text_loss(lambda h: X_sae.mean(0).expand_as(h)):.3f}")
# Вывод в книге:
# | необъяснённый разброс 0.235, включено в среднем 28.3 из 1024, мёртвых 0
# | ошибка модели: как есть 1.963, с автокодировщиком 2.389, со средним вектором 10.212

# %% Листинг «Признаки автокодировщика BookLM» — раздел «Разреженный автокодировщик»
from collections import Counter

peak = f_val.max(0).values                    # пик каждого признака: [1024]
for n in (444, 342, 156, 658, 126):
    top = f_val[:, n].topk(3).indices.tolist()
    strong = torch.nonzero(f_val[:, n] > 0.3 * peak[n]).flatten().tolist()
    word, count = Counter(tok.decode([flat[p]]) for p in strong).most_common(1)[0]
    print(f"признак {n}:", " | ".join(context(flat, p) for p in top))
    print(f"   сильнее 0,3 пика: {len(strong)}, из них «{word}» {count}")
# Вывод в книге:
# | признак 444: Конечно, можешь»,[ —] |  Ньют»,[ —] |  что он твой»,[ —]
# |    сильнее 0,3 пика: 29, из них « —» 29
# | признак 342:  делает?» — спрашивает[ Том] |  кукла», — говорит[ Том] |  друзьями?» — говорит[ Лили]
# |    сильнее 0,3 пика: 37, из них « Тим» 7
# | признак 156:  высоко, чтобы достать[ мяч] | , давай передавать[ мяч] |  «Я возьму ваш[ мяч]
# |    сильнее 0,3 пика: 45, из них « мяч» 18
# | признак 658: . Тим был испуган[,] | !/Девочка испугалась[,] |  кубиков. Тим испугался[,]
# |    сильнее 0,3 пика: 59, из них « и» 14
# | признак 126:  он нашёл редкий,[ блестящий] |  и маленькая птица увидели[ блестящий] |  Эмили, указывая на[ блестящий]
# |    сильнее 0,3 пика: 30, из них « драгоценно» 11

# %% Листинг «Усиление признака на BookLM» — раздел «Разреженный автокодировщик»
start = prompt_ids(tok, "Однажды Тим пошёл в парк.")
for n in (156, 126):
    for c in (0, 0.5, 1, 1.5):
        add = c * peak[n] * sae.dec.weight[:, n].detach()   # c пиков вдоль d_n
        ids = list(start)
        for _ in range(16):                                 # жадное продолжение
            h = residual_stream(ids, add=add, at=3)[-1, -1]
            ids.append(int((model.norm(h) @ model.embed.weight.T).argmax()))
        print(f"{n}, c={c}:{tok.decode(ids[len(start):])}")
# Вывод в книге:
# | 156, c=0: Он увидел большую дерево с многою фруктов. Тим хотел съесть фрукты, но они были
# | 156, c=0.5: Он увидел большую дерево с мячом. Он хотел играть с мячом. Он побежал к
# | 156, c=1: Он увидел большую, круглую мяч. Он был очень рад. Он хотел играть с
# | 156, c=1.5: Он был очень рад. Он был очень рад. Он играл с мячом. Мяч
# | 126, c=0: Он увидел большую дерево с многою фруктов. Тим хотел съесть фрукты, но они были
# | 126, c=0.5: Он увидел большую дерево с птицей на вершине. Птица была очень красивая. Тим хотел
# | 126, c=1: Он был очень рад. Он увидел большую камень. Камень был блестящий и блестящий.
# | 126, c=1.5: Он был очень старый и золот. Он был золот золот золот золот золот золот золот

# %% Листинг «Активационное патчирование рода на BookLM» — раздел «Активационное патчирование»
she, he = tok.encode(" она").ids[0], tok.encode(" он").ids[0]

@torch.no_grad()
def gap(ids, layer=None, pos=None, vec=None):   # z(она) - z(он) в конце
    h = model.embed(torch.tensor([ids]))
    for i in range(7):
        if i == layer:
            h = h.clone(); h[0, pos] = vec      # подставили чужой вектор
        if i < 6:
            h, _ = model.blocks[i](h, model.rope_cos, model.rope_sin)
    logits = model.norm(h[0, -1]) @ model.embed.weight.T
    return (logits[she] - logits[he]).item()

pairs = [(g, b) for g, b in zip(GIRLS[:10], BOYS[:10])
         if len(tok.encode(g).ids) == len(tok.encode(b).ids)]
r = torch.zeros(7, 6)
for girl, boy in pairs:
    clean = prompt_ids(tok, f"{girl} любит рисовать. Однажды")
    bad = prompt_ids(tok, f"{boy} любит рисовать. Однажды")
    states, g_clean, g_bad = residual_stream(clean), gap(clean), gap(bad)
    for layer in range(7):
        for pos in range(6):
            g = gap(bad, layer, pos, states[layer, pos])
            r[layer, pos] += (g - g_bad) / (g_clean - g_bad) / len(pairs)
print(f"пар имён: {len(pairs)}")
print("слой  " + "  ".join(f"{w:>5}" for w in ("<bos>", "имя", "любит", "рисов", ".", "Однаж")))
for layer in range(7):
    print(f"{layer:>4}  " + "  ".join(f"{v:5.2f}" for v in r[layer].tolist()))
# Вывод в книге:
# | пар имён: 8
# | слой  <bos>    имя  любит  рисов      .  Однаж
# |    0   0.00   1.00   0.00   0.00   0.00   0.00
# |    1   0.00   1.01  -0.02  -0.00   0.00   0.00
# |    2   0.00   0.88   0.00  -0.00   0.05   0.04
# |    3   0.00   0.87   0.00  -0.00   0.05   0.06
# |    4   0.00   0.69   0.00   0.00   0.09   0.17
# |    5   0.00   0.05   0.01  -0.00   0.12   0.79
# |    6   0.00   0.00   0.00   0.00   0.00   1.00

# %% Листинг «Порядок примеров меняет ответ» — раздел «Порядок примеров»
from itertools import permutations

pairs = [("кот", "cat"), ("кот", "kitty"), ("кот", "кошка")]
for order in permutations(pairs):
    tokens = [w for pair in order for w in pair] + ["кот"]
    answer, _ = induction_step(tokens, lam=0.5)
    print(" ".join(p[1] for p in order), "->", answer)

def share_of(tokens, word, lam):
    # суммарная доля кандидатов, чьё значение -- word
    _, s = induction_step(tokens, lam=lam)
    return sum(a for a, w in zip(s.tolist(), tokens[1:]) if w == word)

for tokens in (["кот", "cat", "кот", "kitty", "кот"],
               ["кот", "cat", "кот", "cat", "кот", "kitty", "кот"]):
    for lam in (0.0, 0.5):
        cat, kitty = (share_of(tokens, w, lam) for w in ("cat", "kitty"))
        print(f"lam={lam}: cat {cat:.3f}, kitty {kitty:.3f}")
# Вывод в книге:
# | cat kitty кошка -> кошка
# | cat кошка kitty -> kitty
# | kitty cat кошка -> кошка
# | kitty кошка cat -> cat
# | кошка cat kitty -> kitty
# | кошка kitty cat -> cat
# | lam=0.0: cat 0.500, kitty 0.500
# | lam=0.5: cat 0.269, kitty 0.731
# | lam=0.0: cat 0.667, kitty 0.333
# | lam=0.5: cat 0.335, kitty 0.665

# %% Листинг «Примеры в промпте BookLM» — раздел «Порядок примеров»
from itertools import permutations
from booklm.lab import prompt_ids, next_token_table

shots = [("Кошка", "котёнок"), ("Собака", "щенок"), ("Корова", "телёнок")]
firsts = {tok.encode(" " + a).ids[0] for a in ("утёнок", "утенок")}

def make(examples):
    return "".join(f"{a} — {b}. " for a, b in examples) + "Утка —"

@torch.no_grad()
def p_answer(prompt):                  # P(первый токен « утёнок»)
    p = model(torch.tensor([prompt_ids(tok, prompt)]))[0, -1].softmax(-1)
    return sum(p[i].item() for i in firsts)

def top(text, k=3):
    table = next_token_table(tok, model, text, k)
    return ", ".join(f"«{s}» {p:.2f}" for s, p in table)

for k in range(4):
    prompt = make(shots[:k])
    print(f"k={k}: P(утёнок) = {p_answer(prompt):.4f}; {top(prompt)}")
orders = [p_answer(make(o)) for o in permutations(shots)]
lo, hi = min(orders), max(orders)
print(f"шесть порядков: P(утёнок) от {lo:.4f} до {hi:.4f}")
print("копия имени:", top("Жила-была девочка по имени Мия. Однажды"))
# Вывод в книге:
# | k=0: P(утёнок) = 0.0037; « кролик» 0.05, « лиса» 0.04, « утка» 0.03
# | k=1: P(утёнок) = 0.0002; « и» 0.13, « но» 0.07, « очень» 0.03
# | k=2: P(утёнок) = 0.0004; « очень» 0.10, « и» 0.09, « —» 0.07
# | k=3: P(утёнок) = 0.0004; « и» 0.25, « очень» 0.06, « —» 0.04
# | шесть порядков: P(утёнок) от 0.0003 до 0.0027
# | копия имени: « Мия» 0.76, « она» 0.12, « в» 0.02

# %% Листинг «Калибровка по пустому входу» — раздел «Калибровка по пустому входу»
torch.set_printoptions(precision=3)
p = torch.tensor([0.6, 0.4])     # [pos, neg] на настоящем отзыве
w = torch.tensor([0.7, 0.3])     # [pos, neg] на пустом входе N/A
p_cal = (p / w) / (p / w).sum()
print("до калибровки:   ", p, "->", ["pos", "neg"][p.argmax()])
print("после калибровки:", p_cal, "->", ["pos", "neg"][p_cal.argmax()])
print("через логиты:    ", (p.log() - w.log()).softmax(dim=-1))
# Вывод в книге:
# | до калибровки:    tensor([0.600, 0.400]) -> pos
# | после калибровки: tensor([0.391, 0.609]) -> neg
# | через логиты:     tensor([0.391, 0.609])
