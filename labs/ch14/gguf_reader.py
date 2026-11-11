"""Разборщик и писатель GGUF v3 на стандартной библиотеке Python.

Дополнение к разделу 14.6 «Формат GGUF» главы 14. В книге устройство файла описано словами и рисунком,
а код чтения вынесен сюда: заголовок, свойства любой вложенности (строки, числа, массивы), описания
тензоров и проверка выравнивания. Файл пишет крошечный GGUF с одной матрицей 2x2, читает обратно те же
четыре числа, а затем портит сигнатуру и печатает ошибку, на которой разборщик останавливается.

Запуск из корня репозитория: python labs/ch14/gguf_reader.py
"""

# --- Читаем заголовок GGUF вручную
import struct

def read_gguf_header(path):
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"GGUF":
            raise ValueError(f"not GGUF: {magic!r}")
        version, = struct.unpack("<I", f.read(4))
        n_tensors, = struct.unpack("<Q", f.read(8))
        n_kv, = struct.unpack("<Q", f.read(8))
    return {
        "version": version,
        "n_tensors": n_tensors,
        "n_metadata": n_kv,
    }


# --- Структурный разборщик GGUF v3 без внешних пакетов
import struct

# metadata value type -> struct format
SCALAR = {
    0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i",
    6: "f", 10: "Q", 11: "q", 12: "d",
}
TYPE_BOOL, TYPE_STRING, TYPE_ARRAY = 7, 8, 9

def read_exact(f, n):
    data = f.read(n)
    if len(data) != n:
        raise EOFError(f"expected {n} bytes, got {len(data)}")
    return data

def read_number(f, fmt):
    size = struct.calcsize("<" + fmt)
    return struct.unpack("<" + fmt, read_exact(f, size))[0]

def read_string(f):
    n_bytes = read_number(f, "Q")
    return read_exact(f, n_bytes).decode("utf-8")

def read_value(f, type_id):
    if type_id in SCALAR:
        return read_number(f, SCALAR[type_id])
    if type_id == TYPE_BOOL:
        raw = read_number(f, "B")
        if raw not in (0, 1):
            raise ValueError(f"invalid boolean value: {raw}")
        return bool(raw)
    if type_id == TYPE_STRING:
        return read_string(f)
    if type_id == TYPE_ARRAY:
        item_type = read_number(f, "I")
        n_items = read_number(f, "Q")
        return [read_value(f, item_type) for _ in range(n_items)]
    raise ValueError(f"unknown metadata type: {type_id}")

def align_up(offset, alignment):
    return offset + (-offset % alignment)

def inspect_gguf(path):
    with open(path, "rb") as f:
        if read_exact(f, 4) != b"GGUF":
            raise ValueError("wrong GGUF signature")
        version = read_number(f, "I")
        if version != 3:
            raise ValueError(f"this reader expects GGUF v3, got {version}")
        n_tensors = read_number(f, "Q")
        n_metadata = read_number(f, "Q")

        metadata = {}
        for _ in range(n_metadata):
            key = read_string(f)
            value_type = read_number(f, "I")
            metadata[key] = read_value(f, value_type)

        alignment = int(metadata.get("general.alignment", 32))
        if alignment <= 0 or alignment % 8:
            raise ValueError(f"invalid alignment: {alignment}")

        tensors = []
        for _ in range(n_tensors):
            name = read_string(f)
            n_dims = read_number(f, "I")
            dims = [read_number(f, "Q") for _ in range(n_dims)]
            ggml_type = read_number(f, "I")
            relative_offset = read_number(f, "Q")
            if relative_offset % alignment:
                raise ValueError(f"unaligned tensor: {name}")
            tensors.append({
                "name": name,
                "dims": dims,
                "ggml_type": ggml_type,
                "relative_offset": relative_offset,
            })

        tensor_data = align_up(f.tell(), alignment)
        for tensor in tensors:
            tensor["file_offset"] = tensor_data + tensor["relative_offset"]

    return {
        "version": version,
        "metadata": metadata,
        "tensor_data": tensor_data,
        "tensors": tensors,
    }

# --- Создаём минимальный GGUF v3 с матрицей 2x2
from io import BytesIO

TYPE_UINT32 = 4
TYPE_STRING = 8
GGML_TYPE_F32 = 0
ALIGNMENT = 32

def write_string(out, value):
    raw = value.encode("utf-8")
    out.write(struct.pack("<Q", len(raw)))
    out.write(raw)

def write_metadata(out, key, type_id, value):
    write_string(out, key)
    out.write(struct.pack("<I", type_id))
    if type_id == TYPE_UINT32:
        out.write(struct.pack("<I", value))
    elif type_id == TYPE_STRING:
        write_string(out, value)
    else:
        raise ValueError("type is not implemented in the tiny writer")

def write_tiny_gguf(path):
    out = BytesIO()
    metadata = [
        ("general.architecture", TYPE_STRING, "demo"),
        ("general.alignment", TYPE_UINT32, ALIGNMENT),
    ]

    out.write(b"GGUF")
    out.write(struct.pack("<IQQ", 3, 1, len(metadata)))
    for item in metadata:
        write_metadata(out, *item)

    write_string(out, "demo.weight")
    out.write(struct.pack("<I", 2))       # number of dimensions
    out.write(struct.pack("<QQ", 2, 2))  # shape
    out.write(struct.pack("<I", GGML_TYPE_F32))
    out.write(struct.pack("<Q", 0))       # offset within tensor data

    padding = (-out.tell()) % ALIGNMENT
    out.write(b"\x00" * padding)
    out.write(struct.pack("<4f", 1.0, 2.0, 3.0, 4.0))

    with open(path, "wb") as f:
        f.write(out.getvalue())

import os, tempfile

with tempfile.TemporaryDirectory() as tmp:   # leaves no file behind
    path = os.path.join(tmp, "tiny.gguf")
    write_tiny_gguf(path)
    info = inspect_gguf(path)
    tensor = info["tensors"][0]
    with open(path, "rb") as f:
        f.seek(tensor["file_offset"])
        values = struct.unpack("<4f", read_exact(f, 16))
assert values == (1.0, 2.0, 3.0, 4.0)


# заголовок того же файла читается и коротким читателем
with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "tiny.gguf")
    write_tiny_gguf(path)
    assert read_gguf_header(path) == {"version": 3, "n_tensors": 1, "n_metadata": 2}
    info = inspect_gguf(path)
    assert info["metadata"]["general.architecture"] == "demo"
    assert info["tensor_data"] % ALIGNMENT == 0
    # порча сигнатуры: разборщик отклоняет файл до чтения тензоров
    raw = bytearray(open(path, "rb").read())
    raw[0:1] = b"X"
    bad = os.path.join(tmp, "bad.gguf")
    open(bad, "wb").write(bytes(raw))
    try:
        inspect_gguf(bad)
    except ValueError as err:
        assert "signature" in str(err)
        print(f"испорченная сигнатура: {type(err).__name__}: {err}")
    else:
        raise AssertionError("испорченный файл не отклонён")

print("ok: GGUF v3 разобран, данные тензора", values)
