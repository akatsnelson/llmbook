"""Глава 18, раздел 18.5 «Цепочка поставки», подраздел «Подпись, хеш родителя и перечень компонентов»:
подпись Лэмпорта и проверка загрузки в шесть шагов (полная версия).

Из книги «Большие языковые модели».
В книге оставлена сокращённая версия; этот файл самодостаточен и
выполняется в ноутбуке главы notebooks/18_security_boundaries.ipynb.

Одноразовая подпись Лэмпорта из SHA-256, манифест с хешем родителя и SBOM, проверка загрузки:
подпись, SHA-256 файла, родитель, SBOM, заголовок safetensors, тензоры в память. Запрос по SBOM
находит выпуски с уязвимой версией PyTorch; веса с закладкой от честного издателя проходят все шаги.

Запуск из корня репозитория: python labs/ch18/signed_release_loader.py
"""

# --- imports ---
import hashlib
import math

# --- чтение safetensors ---
import json
import struct
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
}
for case, data in broken.items():
    try:
        read_safetensors(data)
        raise AssertionError(f"{case}: принят")
    except ValueError:
        pass                                    # отказ до чтения чисел

# --- подпись Лэмпорта ---
def H(data):
    return hashlib.sha256(data).digest()

def keygen(seed):
    # Учебно: ключ выводится из затравки, чтобы листинг повторялся.
    # Настоящий закрытый ключ берут из secrets.token_bytes(32) на каждую строку.
    sk = [[H(seed + bytes([bit]) + i.to_bytes(2, "big")) for i in range(256)]
          for bit in (0, 1)]                    # 2 x 256 строк по 32 байта
    pk = [[H(s) for s in row] for row in sk]    # открытый ключ — их хеши
    return sk, pk

def bits(digest):
    return [(digest[i // 8] >> (7 - i % 8)) & 1 for i in range(256)]

def sign(sk, message):                          # по одной строке на бит хеша
    return [sk[bit][i] for i, bit in enumerate(bits(H(message)))]

def verify(pk, message, signature):
    return len(signature) == 256 and all(
        H(s) == pk[bit][i]
        for i, (s, bit) in enumerate(zip(signature, bits(H(message)))))

sk, pk = keygen(b"publisher-booklm")
assert sum(map(len, sk[0] + sk[1])) == 16_384   # закрытый ключ 16 КБ
assert sum(map(len, sign(sk, b"x"))) == 8_192   # подпись 8 КБ
assert verify(pk, b"x", sign(sk, b"x")) and not verify(pk, b"y", sign(sk, b"x"))

m1, m2 = b"manifest v1", b"manifest v2"         # один ключ — дважды
s1, s2 = sign(sk, m1), sign(sk, m2)
both = sum(x != y for x, y in zip(s1, s2))      # раскрыты обе строки пары
assert both == sum(x != y for x, y in zip(bits(H(m1)), bits(H(m2))))
assert 100 < both < 156                         # около половины из 256

# --- проверка загрузки ---
def canon(obj):                                 # канонический JSON главы 13
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode()

def sha(data):
    return hashlib.sha256(data).hexdigest()

REQUIRED = {"weights", "tokenizer", "chat_template", "rag_index",
            "tools_registry", "policy", "render", "attack_set", "torch"}

class LoadRefused(Exception):
    def __init__(self, step, reason):
        super().__init__(f"шаг {step}: {reason}")
        self.step = step

materialized = []                               # тензоры, попавшие в память

def verify_before_load(manifest, files, parent_bytes, publisher_pk):
    body = {k: v for k, v in manifest.items() if k != "signature"}
    signature = [bytes.fromhex(s) for s in manifest.get("signature", [])]
    if not verify(publisher_pk, canon(body), signature):
        raise LoadRefused(1, "подпись манифеста не сошлась")
    if sha(files["weights"]) != body["sha256"]:
        raise LoadRefused(2, "SHA-256 файла весов не равен манифесту")
    parent = json.loads(parent_bytes)
    if (list(body["parents"]) != [parent["id"]]           # имя родителя
            or body["parents"][parent["id"]] != sha(parent_bytes)
            or parent["sha256"] != body["sha256"]):   # веса не менялись
        raise LoadRefused(3, "родитель не совпал")
    sbom = {c["name"]: c for c in body["sbom"]}
    if REQUIRED - sbom.keys():
        raise LoadRefused(4, f"в SBOM нет {sorted(REQUIRED - sbom.keys())}")
    for name, c in sbom.items():
        if sha(files[name]) != c["sha256"]:
            raise LoadRefused(4, f"{name}: хеш компонента не сошёлся")
    try:
        tensors = read_safetensors(files["weights"])
    except (ValueError, KeyError) as error:
        raise LoadRefused(5, f"заголовок: {error}")
    materialized.extend(tensors)                # шаг 6: только теперь
    return tensors

def release(weights, sk, drop=(), torch_version="2.6.0"):
    files = {"weights": weights, "tokenizer": b"booklm-tokenizer-v1",
             "chat_template": b"<|user|>...<|assistant|>",
             "rag_index": b"booklm-rag-v1 index",
             "tools_registry": b"booklm-tools-v1 registry",
             "policy": b"authz-guarded-v1", "render": b"render_safe-v1",
             "attack_set": b"A1-A5 B1-B5",
             "torch": f"torch-{torch_version}".encode()}
    parent = canon({"id": "booklm-tools-v1",     # учебный родитель
                    "parents": {"booklm-rag-v1": sha(files["rag_index"])},
                    "sha256": sha(weights)})
    sbom = [{"name": n, "version": torch_version if n == "torch" else "v1",
             "sha256": sha(data)} for n, data in files.items() if n not in drop]
    manifest = {"id": "booklm-guarded-v1",
                "parents": {"booklm-tools-v1": sha(parent)},
                "transformation": "add_security_boundaries",
                "sha256": sha(weights), "sbom": sbom}
    manifest["signature"] = [s.hex() for s in sign(sk, canon(manifest))]
    return manifest, files, parent

def refused_at(manifest, files, parent, key=pk):
    materialized.clear()
    try:
        verify_before_load(manifest, files, parent, key)
    except LoadRefused as error:
        assert materialized == []               # ни одного тензора в памяти
        return error.step
    return None

manifest, files, parent = release(file, sk)     # file — мини-safetensors
assert refused_at(manifest, files, parent) is None
assert materialized == ["w", "b"]               # верный выпуск загружен

bad = bytearray(file); bad[-1] ^= 1             # испорченный байт весов
assert refused_at(manifest, {**files, "weights": bytes(bad)}, parent) == 2
swapped = dict(manifest, sha256=sha(bytes(bad)))   # подменили и хеш
assert refused_at(swapped, {**files, "weights": bytes(bad)}, parent) == 1
evil_sk, evil_pk = keygen(b"attacker")          # подписал чужим ключом
evil = release(bytes(bad), evil_sk)
assert refused_at(*evil) == 1
assert refused_at(*evil, key=evil_pk) is None   # доверие — в открытом ключе
assert refused_at(*release(file, sk, drop=("tokenizer",))) == 4
garbled = forge(good, w + b"\0" * 4 + b)        # заголовок не сходится
assert refused_at(*release(garbled, sk)) == 5

def affected(sboms, bad_up_to=(2, 5, 1)):       # уязвимость: torch <= 2.5.1
    def version(v):
        return tuple(int(x) for x in v.split("."))
    return [rid for rid, sbom in sboms.items()
            if any(c["name"] == "torch" and version(c["version"]) <= bad_up_to
                   for c in sbom)]

sboms = {f"r{i}": release(file, sk, torch_version=v)[0]["sbom"]
         for i, v in enumerate(["2.4.0", "2.5.1", "2.6.0"], start=1)}
assert affected(sboms) == ["r1", "r2"]

trojan = write_safetensors(                     # честный издатель, но веса
    {"next_after_trigger": ("F32", [1], array("f", [1.0]).tobytes())})
assert refused_at(*release(trojan, sk)) is None     # все шесть шагов пройдены

# Ключ Лэмпорта одноразовый: восемь подписей одним ключом, и подделка собирается из
# раскрытых клеток без поиска прообраза (раздел 18.5 печатает эти числа словами).
demo_sk, demo_pk = keygen(b"demo")
known = {}                            # (бит, столбец) -> раскрытая клетка
for k in range(1, 9):
    m = f"manifest v{k}".encode()
    for i, (cell, bit) in enumerate(zip(sign(demo_sk, m), bits(H(m)))):
        known[bit, i] = cell
for n in range(1000):                 # перебор сообщений нарушителя
    forged = f"manifest evil {n}".encode()
    need = [(bit, i) for i, bit in enumerate(bits(H(forged)))]
    if all(cell in known for cell in need):
        break
assert len(known) == 510                        # тайной остались две клетки из 512
assert n + 1 == 3                               # подделка с третьей попытки
assert verify(demo_pk, forged, [known[cell] for cell in need])

print("OK:", __file__)
