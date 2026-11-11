"""Листинги главы 14 «Инференс» книги «Большие языковые модели», файл 14-inference.

Код и вывод — ровно как в книге; текст книги сюда не входит. Листинги идут по порядку и
опираются друг на друга: запускайте файл целиком или ячейками сверху вниз (разметка # %%).
Проверка, что вывод совпал с книгой: python scripts/check_listings.py listings/ch14/14-inference.py
"""

# %% Листинг «Учебный стенд и цикл генерации» — раздел «Один шаг генерации»
import torch
torch.manual_seed(0)
V, d = 8, 4                          # словарь и ширина вектора токена
E = torch.randn(V, d)                # таблица эмбеддингов
W_out = torch.randn(d, V)            # выходная матрица
seen = []                            # сколько позиций прошло через модель

def toy_model(ids):                  # ids: [B, T] -> логиты [B, T, V]
    seen.append(ids.size(1))
    h = E[ids].cumsum(dim=1)         # позиция t видит токены 1..t
    h = h / torch.arange(1, ids.size(1) + 1)[None, :, None]
    return h @ W_out

def generate(model, ids, steps):
    for _ in range(steps):
        logits = model(ids)                          # [B, T, V]
        next_id = logits[:, -1].argmax(dim=-1, keepdim=True)
        ids = torch.cat([ids, next_id], dim=1)       # [B, T+1]
    return ids

prompt = torch.tensor([[3, 1, 4, 1]])  # 4 токена запроса
print(toy_model(prompt).shape)
seen.clear()
print(generate(toy_model, prompt, steps=3))
print(seen, sum(seen))
# Вывод в книге:
# | torch.Size([1, 4, 8])
# | tensor([[3, 1, 4, 1, 5, 4, 7]])
# | [4, 5, 6] 15

# %% Листинг «Выборка для пакета с настройками каждого запроса» — раздел «Выборка в пакете запросов»
import torch
torch.manual_seed(0)
logits = torch.randn(3, 8) * 2                  # шаг пакета: 3 запроса, V=8
temps = torch.tensor([0.0, 0.7, 1.2])           # у каждого свои настройки
top_p = torch.tensor([1.0, 0.9, 0.5])

def sample_batch(logits, temps, top_p, gens):   # logits: [B, V]
    greedy = temps == 0
    t = torch.where(greedy, torch.ones_like(temps), temps)
    p = torch.softmax(logits / t[:, None], dim=-1)          # [B, V]
    sp, idx = p.sort(dim=-1, descending=True)
    before = sp.cumsum(dim=-1) - sp                         # набрано до токена
    sp = torch.where(before < top_p[:, None], sp, torch.zeros_like(sp))
    picks = [torch.multinomial(sp[b], 1, generator=g)       # своё зерно, сумму
             for b, g in enumerate(gens)]                   # multinomial делит сам
    sampled = idx.gather(1, torch.stack(picks))[:, 0]
    return torch.where(greedy, logits.argmax(dim=-1), sampled)

gens = [torch.Generator().manual_seed(s) for s in (11, 12, 13)]
print("пакет из трёх:", sample_batch(logits, temps, top_p, gens).tolist())
alone = [torch.Generator().manual_seed(12)]
print("второй один:  ", sample_batch(logits[1:2], temps[1:2], top_p[1:2], alone).tolist())
# Вывод в книге:
# | пакет из трёх: [4, 2, 2]
# | второй один:   [2]
