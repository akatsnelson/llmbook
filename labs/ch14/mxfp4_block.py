"""Кодирование и декодирование блока MXFP4: коды E2M1, масштаб E8M0 и упаковка 17 байт.

Дополнение к подразделу «Форматы с плавающей точкой» главы 14 (раздел 14.5 «Квантование»). В книге
MXFP4 описан словами: коды E2M1, общий масштаб E8M0 на блок из 32 чисел, 4,25 бита на вес. Здесь
кодирование блока до байта, декодер, упаковка в байты и опыт с порчей: floor заменён на round при выборе
порядка масштаба.

Запуск из корня репозитория: python labs/ch14/mxfp4_block.py
"""

import math

E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]  # модули кодов 0..7
BLOCK = 32


def fp4_encode(value):
    # ближайшее значение E2M1; при равенстве — чётная мантисса
    sign = 1 if math.copysign(1.0, value) < 0 else 0
    mag = min(abs(value), 6.0)                       # насыщение до 6
    idx = min(range(8), key=lambda i: (abs(mag - E2M1[i]), i & 1))
    return (sign << 3) | idx


def fp4_decode(code):
    mag = E2M1[code & 0b0111]
    return -mag if code & 0b1000 else mag


def mxfp4_encode_block(block, rounding=math.floor):
    assert len(block) == BLOCK
    amax = max(abs(v) for v in block)
    exp = rounding(math.log2(amax)) - 2 if amax > 0 else 0
    exp = max(-127, min(127, exp))                   # диапазон E8M0
    scale = 2.0 ** exp
    codes = [fp4_encode(v / scale) for v in block]
    return exp + 127, codes                          # байт E8M0 и 32 кода


def mxfp4_decode_block(scale_byte, codes):
    assert scale_byte != 0xFF, "NaN scale"
    scale = 2.0 ** (scale_byte - 127)
    return [scale * fp4_decode(c) for c in codes]


def pack_block(scale_byte, codes):
    # первый код пары — в младших битах, как в pack_int4
    pairs = range(0, BLOCK, 2)
    nibbles = bytes(codes[i] | (codes[i + 1] << 4) for i in pairs)
    return bytes([scale_byte]) + nibbles


# 16 кодов, 15 разных значений; каждый код переживает круг кодирования
assert len({fp4_decode(c) for c in range(16)}) == 15
assert all(fp4_encode(fp4_decode(c)) == c for c in range(16))
assert fp4_encode(2.5) == 0b0100                    # 2,5 посередине: уходит в 2 (m = 0)

# ручной пример из книги
block = [0.8, -0.3, 2.9, 0.05] + [0.0] * 28
scale_byte, codes = mxfp4_encode_block(block)
assert scale_byte == 0x7E
assert codes[:4] == [0x3, 0x9, 0x7, 0x0]
restored = mxfp4_decode_block(scale_byte, codes)
assert restored[:4] == [0.75, -0.25, 3.0, 0.0]

packed = pack_block(scale_byte, codes)
assert len(packed) == 17 and len(packed) * 8 / BLOCK == 4.25


def sse(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b))


# порча: round вместо floor даёт масштаб X = 1, и ошибка блока растёт
bad_byte, bad_codes = mxfp4_encode_block(block, rounding=round)
assert bad_byte == 127
bad = mxfp4_decode_block(bad_byte, bad_codes)
assert bad[:4] == [1.0, -0.5, 3.0, 0.0]
assert math.isclose(sse(block, restored), 0.0175)
assert math.isclose(sse(block, bad), 0.0925)
print("ok: блок 17 байт, 4,25 бита на вес; SSE floor 0.0175, round 0.0925")
