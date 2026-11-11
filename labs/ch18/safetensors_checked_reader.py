"""Глава 18, раздел 18.5 «Цепочка поставки», подраздел «Формат safetensors»: чтение safetensors с проверкой каждого смещения (полная версия).

Из книги «Большие языковые модели».
В книге оставлена сокращённая версия; этот файл самодостаточен и
выполняется в ноутбуке главы notebooks/18_security_boundaries.ipynb.

Запись и чтение safetensors, пять испорченных файлов (обрезан, перекрытие, дыра, длина
заголовка 2^63, чужой dtype) и разметка настоящей BookLM в BF16: 56 тензоров, 13 244 928 байт.

Запуск из корня репозитория: python labs/ch18/safetensors_checked_reader.py
"""

# --- imports ---
import hashlib
import math

# --- чтение safetensors ---
import json
import struct
from pathlib import Path
from array import array

DTYPES = {"F32": 4, "F16": 2, "BF16": 2, "I64": 8}   # байт на число
MAX_HEADER = 100 * 2**20                             # 100 МБ

def write_safetensors(tensors, metadata=None):
    header, offset = {}, 0          # tensors: имя -> (dtype, shape, байты)
    for name, (dtype, shape, raw) in tensors.items():
        header[name] = {"dtype": dtype, "shape": list(shape),
                        "data_offsets": [offset, offset + len(raw)]}
        offset += len(raw)
    if metadata:
        header["__metadata__"] = metadata
    text = json.dumps(header, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 8)             # выравнивание на 8 байт
    body = b"".join(raw for _, _, raw in tensors.values())
    return struct.pack("<Q", len(text)) + text + body

def check_header(header, data_len):
    if not isinstance(header, dict):
        raise ValueError("заголовок не JSON-объект")
    spans = []
    for name, info in header.items():
        if name == "__metadata__":
            continue
        dtype, shape, (begin, end) = (info["dtype"], info["shape"],
                                      info["data_offsets"])
        if dtype not in DTYPES:
            raise ValueError(f"{name}: неизвестный dtype {dtype}")
        if any(size < 0 for size in shape):     # [-1, -3]: произведение 3
            raise ValueError(f"{name}: отрицательный размер {shape}")
        if math.prod(shape) * DTYPES[dtype] != end - begin:
            raise ValueError(f"{name}: форма не равна длине отрезка")
        spans.append((begin, end, name))
    position = 0
    for begin, end, name in sorted(spans):      # без дыр и перекрытий
        if begin != position:
            raise ValueError(f"{name}: дыра или перекрытие на {position}")
        position = end
    if position != data_len:
        raise ValueError("отрезки не покрывают буфер целиком")
    return header

def read_safetensors(data):
    if len(data) < 8:
        raise ValueError("файл короче 8 байт")
    (n,) = struct.unpack("<Q", data[:8])        # длина заголовка
    if n > min(MAX_HEADER, len(data) - 8):
        raise ValueError(f"длина заголовка {n} больше файла")
    header = check_header(json.loads(data[8:8 + n]), len(data) - 8 - n)
    start = 8 + n                               # смещения — от начала чисел
    return {name: (info["dtype"], info["shape"],
                   data[start + info["data_offsets"][0]:
                        start + info["data_offsets"][1]])
            for name, info in header.items() if name != "__metadata__"}

w = array("f", [0.5, -1.0, 2.0, 0.25, 0.0, 1.5]).tobytes()   # [2, 3]
b = array("f", [1.0, 1.0, 1.0]).tobytes()                    # [3]
mini = {"w": ("F32", [2, 3], w), "b": ("F32", [3], b)}
file = write_safetensors(mini, {"artifact_id": "mini"})
assert read_safetensors(file) == mini           # туда и обратно без потерь

def forge(header, body):                        # заголовок пишет нарушитель
    text = json.dumps(header).encode()
    return struct.pack("<Q", len(text)) + text + body

good = {"w": {"dtype": "F32", "shape": [2, 3], "data_offsets": [0, 24]},
        "b": {"dtype": "F32", "shape": [3], "data_offsets": [24, 36]}}
broken = {
    "обрезан": file[:-4],
    "перекрытие": forge({**good, "b": {**good["b"],
                                       "data_offsets": [20, 32]}}, w + b),
    "дыра": forge({**good, "b": {**good["b"],
                                 "data_offsets": [28, 40]}}, w + b"\0" * 4 + b),
    "N = 2^63": struct.pack("<Q", 2**63) + file[8:],
    "чужой dtype": forge({**good, "b": {**good["b"], "dtype": "PY_OBJECT"}},
                         w + b),
    "минус на минус": forge({**good, "b": {**good["b"], "shape": [-1, -3]}},
                            w + b),
}
for case, data in broken.items():
    try:
        read_safetensors(data)
        raise AssertionError(f"{case}: принят")
    except ValueError:
        pass                                    # отказ до чтения чисел

# --- заголовок BookLM ---
def booklm_layout(V=8192, d=256, layers=6, kv=64, ff=768):
    shapes = [("tokens.weight", [V, d])]       # output.weight — тот же тензор
    for i in range(layers):
        p = f"blocks.{i}."
        shapes += [(p + "norm_1.weight", [d]),
                   (p + "attention.q.weight", [d, d]),
                   (p + "attention.k.weight", [kv, d]),
                   (p + "attention.v.weight", [kv, d]),
                   (p + "attention.out.weight", [d, d]),
                   (p + "norm_2.weight", [d]),
                   (p + "ffn.gate.weight", [ff, d]),
                   (p + "ffn.up.weight", [ff, d]),
                   (p + "ffn.down.weight", [d, ff])]
    shapes.append(("norm.weight", [d]))
    header, offset = {}, 0
    for name, shape in shapes:                  # BF16: 2 байта на число
        size = math.prod(shape) * 2
        header[name] = {"dtype": "BF16", "shape": shape,
                        "data_offsets": [offset, offset + size]}
        offset += size
    return header, offset

layout, total = booklm_layout()                 # ни одного числа в памяти
assert len(layout) == 1 + 6 * 9 + 1 == 56
assert total == 13_244_928 == 6_622_464 * 2     # паспорт BookLM в BF16
assert layout["tokens.weight"]["data_offsets"] == [0, 4_194_304]
assert layout["blocks.0.attention.q.weight"]["data_offsets"] == [
    4_194_816, 4_325_888]
check_header(layout, total)                     # отрезки сходятся с паспортом

# настоящий файл из главы 5: тот же паспорт, проверенный по заголовку с диска
REAL = Path(__file__).resolve().parents[2] / "models/booklm/hf/base/model.safetensors"
if REAL.exists():
    raw = REAL.read_bytes()
    n = struct.unpack("<Q", raw[:8])[0]
    real = json.loads(raw[8:8 + n])
    assert len(real) - 1 == 56 and len(raw) - 8 - n == total
    check_header(real, len(raw) - 8 - n)

print("OK:", __file__)
