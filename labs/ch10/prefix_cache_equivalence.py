"""Глава 10, раздел 10.5 «Цена длинного промпта»: листинг «Готовый префикс даёт те же
логиты» (полная версия с проверками).

Работает на обученной BookLM (booklm/model.py, веса models/booklm/booklm-base.pt).
Проверяет, что логиты хвоста, прочитанного поверх готового KV-кэша префикса,
совпадают с полным проходом; что хвост не трогает ключи префикса; что замена
первого токена меняет ключ 32-й позиции только начиная со второго блока модели;
что сдвиг префикса на одну позицию (вклад RoPE) меняет ключи уже в первом блоке.

Запуск из корня репозитория: python labs/ch10/prefix_cache_equivalence.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from booklm.lab import load, val_tokens  # noqa: E402

tok, model = load("base")
data = val_tokens(10_000)
P = 448
prefix = data[:P][None]
tails = [data[1000 + 40 * k:1040 + 40 * k][None] for k in range(10)]


@torch.no_grad()
def read(ids, caches):
    logits = model(ids, caches)
    for c in caches:
        c["len"] += ids.size(1)          # длину кэша ведёт вызывающий
    return logits


def new_cache():
    return [{"len": 0} for _ in model.blocks]


cached = new_cache()
read(prefix, cached)
keys_before = [c["k"].clone() for c in cached]
for tail in tails:
    full_cache = new_cache()
    full = read(torch.cat([prefix, tail], 1), full_cache)[0, P:]
    copy = [dict(c) for c in cached]     # копия на запрос: оригинал не растёт
    part = read(tail, copy)[0]
    torch.testing.assert_close(part, full, atol=1e-4, rtol=1e-4)
    assert int(part[-1].argmax()) == int(full[-1].argmax())
    for b in range(len(model.blocks)):   # хвост не трогает ключи префикса
        torch.testing.assert_close(full_cache[b]["k"][:, :, :P], keys_before[b])
assert all(c["len"] == P for c in cached)
assert all(torch.equal(c["k"], k) for c, k in zip(cached, keys_before))

# Замена первого токена: в блоке 1 ключ позиции 32 зависит только от своего
# токена и позиции, со второго блока — от всего, что левее.
other = data[5:6]
assert int(other) != int(prefix[0, 0])
edited = new_cache()
read(torch.cat([other, prefix[0, 1:]])[None], edited)
same = [torch.equal(edited[b]["k"][0, :, 31], cached[b]["k"][0, :, 31])
        for b in range(len(model.blocks))]
assert same[0] and not any(same[1:]), same

# Сдвиг на одну позицию: тот же токен на позиции 32 вместо 31 даёт другой ключ
# уже в первом блоке — это вклад RoPE, а не причинной маски.
shifted = new_cache()
read(torch.cat([other, prefix[0]])[None], shifted)
k_shift = shifted[0]["k"][0, :, 32]      # бывшая позиция 31, сдвинутая на 1
k_orig = cached[0]["k"][0, :, 31]
assert not torch.allclose(k_shift, k_orig, atol=1e-3)
# ...и разницу целиком объясняет угол поворота: тот же «сырой» ключ блока 1,
# повёрнутый RoPE на позицию 31 и на позицию 32, даёт оба ключа.
from booklm.model import apply_rope  # noqa: E402
att = model.blocks[0].attention
with torch.no_grad():
    raw = att.k(model.blocks[0].norm_1(model.embed(prefix[:, 31:32])))
raw = raw.view(1, 1, att.kv_heads, att.d_head).transpose(1, 2)
for pos, k in ((31, k_orig), (32, k_shift)):
    turned = apply_rope(raw, model.rope_cos[pos:], model.rope_sin[pos:])[0, :, 0]
    torch.testing.assert_close(turned, k, atol=1e-5, rtol=1e-5)

print("OK: prefix_cache_equivalence.py")
