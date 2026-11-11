"""Глава 18, раздел 18.5 «Цепочка поставки», подраздел «Файл весов с исполняемым кодом»: pickle исполняет код при загрузке; сканер опкодов (полная версия).

Из книги «Большие языковые модели».
В книге оставлена сокращённая версия; этот файл самодостаточен и
выполняется в ноутбуке главы notebooks/18_security_boundaries.ipynb.

Файл из шести команд pickle вызывает exec при pickle.loads. Сканер читает команды через
pickletools.genops, ничего не исполняя, и сообщает об именах не из списка разрешённых. Вызовы
REDUCE он не считает: в честном файле весов BookLM их 114, и сканер, кричащий на каждый, бесполезен.
Функцию в pickle нельзя достать, не назвав её, поэтому проверяются имена. Имя для STACK_GLOBAL
сканер берёт только из двух строк, положенных прямо перед ним, иначе сообщает «имя не прочитать».

Запуск из корня репозитория: python labs/ch18/pickle_payload_scanner.py
"""

# --- pickle: загрузка исполняет чужой код; сканер ---
import pickle
import pickletools
import tempfile
from pathlib import Path

flag = Path(tempfile.mkdtemp()) / "pwned.txt"   # след исполненного кода

class Payload:                                  # «веса» от нарушителя
    def __reduce__(self):
        return (exec, (f"open({str(flag)!r}, 'w').write('сработало')",))

blob = pickletools.optimize(pickle.dumps(Payload(), protocol=2,
                                         fix_imports=False))
ops = [op.name for op, _, _ in pickletools.genops(blob)]
assert ops == ["PROTO", "GLOBAL", "BINUNICODE", "TUPLE1", "REDUCE", "STOP"]

pickle.loads(blob)                              # «загрузили веса»...
assert flag.exists()                            # ...и исполнили чужой код
flag.unlink()

SAFE = {"collections OrderedDict", "torch FloatStorage",
        "torch._utils _rebuild_tensor_v2"}         # разрешённые имена
MEMO = {"MEMOIZE", "PUT", "BINPUT", "LONG_BINPUT", "FRAME"}

def scan_pickle(data):               # -> имена не из списка
    found, pushed = [], []           # pushed: что команды клали на стек
    for op, arg, _ in pickletools.genops(data):
        if op.name in MEMO:
            continue                 # запись в память стек не меняет
        if op.name in ("GLOBAL", "INST"):
            name = arg
        elif op.name == "STACK_GLOBAL":   # имя — две строки со стека
            two = pushed[-2:]
            name = " ".join(two) if None not in two else "имя не прочитать"
        elif op.name.startswith("EXT"):
            name = "имя из реестра"
        else:
            pushed.append(arg if "UNICODE" in op.name else None)
            continue
        if name not in SAFE:
            found.append(name)
        pushed.append(None)          # на стеке теперь функция
    return found

assert scan_pickle(blob) == ["builtins exec"]
assert scan_pickle(pickle.dumps(Payload(), protocol=4)) == ["builtins exec"]
assert scan_pickle(pickle.dumps({"lr": 3e-4, "steps": [1, 2]})) == []
# обход сканера «две последние строки»: имя достают из памяти командой BINGET
memo = (b"\x80\x04" b"\x8c\x08builtins\x94" b"\x8c\x04exec\x94" b"00"
        b"h\x00h\x01\x93" b"\x8c\x01x\x85R.")
assert [op.name for op, _, _ in pickletools.genops(memo)].count("BINGET") == 2
assert scan_pickle(memo) == ["имя не прочитать"]
assert not flag.exists()                        # сканер ничего не запустил

print("OK:", __file__)
