"""Листинги главы 11 «Контекст и память» книги «Большие языковые модели», файл 01-five-lifetimes.

Код и вывод — ровно как в книге; текст книги сюда не входит. Листинги идут по порядку и
опираются друг на друга: запускайте файл целиком или ячейками сверху вниз (разметка # %%).
Проверка, что вывод совпал с книгой: python scripts/check_listings.py listings/ch11/01-five-lifetimes.py
"""

# %% Листинг «Когда KV-кэши перевешивают веса» — раздел «Цена состояния в байтах»
from booklm.lab import kv_cache_bytes      # формула главы 4, BF16
from booklm.model import BookLMConfig

PARAMS, T_MAX = 6_622_464, 512
WEIGHTS = PARAMS * 2                       # BF16: два байта на число

def requests_to_outweigh(tokens=T_MAX, cfg=BookLMConfig()):
    n = 1                          # добавляем запросы, пока KV легче
    while n * kv_cache_bytes(tokens, cfg) <= WEIGHTS:
        n += 1
    return n

print(f"веса: {WEIGHTS / 2**20:.1f} МБ")
for name, t in [("токен", 1), ("блок памяти", 64),
                ("история", 240), ("окно", T_MAX)]:
    print(f"KV, {name}: {kv_cache_bytes(t) / 2**10:g} КБ")
print("запросов, чтобы KV перевесил веса:", requests_to_outweigh())
print("то же при 8 головах ключей:",
      requests_to_outweigh(cfg=BookLMConfig(kv_heads=8)))
# Вывод в книге:
# | веса: 12.6 МБ
# | KV, токен: 1.5 КБ
# | KV, блок памяти: 96 КБ
# | KV, история: 360 КБ
# | KV, окно: 768 КБ
# | запросов, чтобы KV перевесил веса: 17
# | то же при 8 головах ключей: 5

# %% Листинг «Модель ничего не уносит из запроса» — раздел «Срок жизни KV-кэша»
import torch
from booklm.lab import load, generate, prompt_ids

tok, model = load("base")
ids = torch.tensor([prompt_ids(tok, "Жила-была девочка Маша. Она жила в Казани.")])
others = ["Однажды кот", "Мальчик нашёл щенка", "В лесу жил ёжик",
          "Дракон проснулся", "Лиса пришла к реке", "Бабушка испекла пирог",
          "Шёл сильный дождь", "У Тома был мяч", "Птичка села на ветку",
          "Мама сказала"]
with torch.no_grad():
    z1 = model(ids)[0, -1]                  # логиты до
    for text in others:                     # десять чужих запросов
        generate(tok, model, text, max_new=100, temperature=0)
    z2 = model(ids)[0, -1]                  # логиты после
print("max |z1 - z2| =", (z1 - z2).abs().max().item())

caches = [{"len": 0} for _ in model.blocks]  # кэш заводит вызывающий код
with torch.no_grad():
    model(ids, caches)
k = caches[0]["k"]
size = sum(c[x].nbytes for c in caches for x in ("k", "v"))
print("токенов:", ids.shape[1], "| ключи блока 0:", list(k.shape), k.dtype)
print("байт на токен:", size // ids.shape[1])
# Вывод в книге:
# | max |z1 - z2| = 0.0
# | токенов: 16 | ключи блока 0: [1, 2, 16, 32] torch.float32
# | байт на токен: 3072
