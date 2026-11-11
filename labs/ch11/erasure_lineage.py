"""Учебный мир производных копий, зонд удаления, удаление по графу с квитанцией и намеренно сломанное удаление.

Полная версия листингов «Зонд удаления» и «Удаление по графу и удаление одной записи» из главы 11 книги
(раздел 11.5 «Гонка версий и право на удаление», подраздел «Удаление производных копий»).
В книге оставлены зонд, обход графа и сломанное удаление; здесь весь код целиком, с теми же проверками.

Запуск из корня репозитория: python labs/ch11/erasure_lineage.py
"""

# --- Учебный мир производных копий и листинг «Зонд удаления» ---
import hashlib
from itertools import cycle

def node(store, user, text, parents=()):
    return {"store": store, "user": user, "text": text,
            "from": list(parents)}

def xor(data, key):                    # модель шифра, НЕ криптография
    return bytes(b ^ k for b, k in zip(data, cycle(key)))

def build_world():
    U, B = "u42", "u77"
    w = {
        "s1/t2": node("история", U, "Меня зовут Аня, я живу в Казани"),
        "s1/t3": node("история", U, "Пиши коротко"),
        "s2/t5": node("история", U, "С 1 ноября живу в Самаре"),
        "s2/t6": node("история", U, "Дедлайн диплома 15 декабря"),
        "b/t1":  node("история", B, "Я Борис, живу в Казани"),
        "r-name@1": node("записи", U, "имя: Аня", ["s1/t2"]),
        "r-city@1": node("версии", U, "город: Казань", ["s1/t2"]),
        "r-city@2": node("записи", U, "город: Самара", ["s2/t5"]),
        "rb-city@1": node("записи", B, "город: Казань", ["b/t1"]),
        "sum-s1": node("сводки", U, "Аня из Казани, просит коротко",
                       ["s1/t2", "s1/t3"]),
        "sum-s2": node("сводки", U, "С ноября в Самаре, дедлайн 15 дек",
                       ["s2/t5", "s2/t6"]),
        "ix:u42:kazan": node("индекс", U, "казан: r-city@1",
                             ["r-city@1"]),
        "ix:u42:samar": node("индекс", U, "самар: r-city@2",
                             ["r-city@2"]),
        "ix:u77:kazan": node("индекс", B, "казан: rb-city@1",
                             ["rb-city@1"]),
        "ans:u42:mv7": node("кэш ответов", U, "Ужин: ресторан в Самаре",
                            ["r-city@2"]),
        "kv:u42:mv7": node("блоки KV", U, "тензоры блока памяти",
                           ["r-name@1", "r-city@2"]),
        "log-118": node("журнал", U, "read r-city@2"),     # только id
        "eval-17": node("выгрузка", U, "С 1 ноября живу в Самаре",
                        ["s2/t5"]),
    }
    key = hashlib.sha256(b"demo key: u42 city").digest()
    keys = {(U, "город"): key}
    backup = {(U, "город"): xor("город: Самара".encode(), key)}
    return w, keys, backup

def erasure_scan(w, keys, backup, user, ids, stems):
    """След: узел user со значением или ребром к удалённому id."""
    hits = [nid for nid, n in w.items() if n["user"] == user and (
            any(s in n["text"].lower() for s in stems)
            or set(n["from"]) & set(ids))]
    for (u, slot), blob in backup.items():
        if u == user and (u, slot) in keys:    # читаемо, пока жив ключ
            text = xor(blob, keys[(u, slot)]).decode().lower()
            if any(s in text for s in stems):
                hits.append(f"backup:{slot}")
    return hits

world, keys, backup = build_world()
stems = ("казан", "самар")        # основы ловят и Казани, и Самаре
city_ids = {"r-city@1", "r-city@2"}
before = erasure_scan(world, keys, backup, "u42", city_ids, stems)
assert len(before) == 12          # до удаления зонд видит 12 следов


# --- Листинг «Удаление по графу и удаление одной записи»: удаление по графу с квитанцией ---
import copy
import re
from collections import defaultdict, deque

MEMORY = ("записи", "версии")

def erase(w, keys, backup, user, slot, stems):
    receipt = defaultdict(list)
    def log(store, nid, action):
        receipt[store].append((nid, action))
    roots = [nid for nid, n in w.items() if n["user"] == user
             and n["store"] in MEMORY
             and n["text"].startswith(slot + ":")]
    children = defaultdict(list)
    for nid, n in w.items():
        for p in n["from"]:
            children[p].append(nid)
    sources = sorted({p for r in roots for p in w[r]["from"]})
    # 1. Надгробия: значение уходит, id и версия остаются.
    for r in roots:
        w[r] = node(w[r]["store"], user, f"{slot}: удалено")
        log(w[r]["store"], r, "надгробие")
    # 2. Источник только про этот факт удаляем, иначе вычищаем.
    gone, mask = set(roots), r"\w*(" + "|".join(stems) + r")\w*"
    for s in sources:
        facts = {c for c in children[s] if w[c]["store"] in MEMORY}
        if facts <= set(roots):
            del w[s]; gone.add(s); log("история", s, "удалено")
        else:
            w[s]["text"] = re.sub(mask, "***", w[s]["text"], flags=re.I)
            log("история", s, "вычищено")
    # 3. Обход в ширину от корней и источников ко всем потомкам.
    queue, seen = deque(roots + sources), set(roots + sources)
    while queue:
        for c in children[queue.popleft()]:
            if c not in seen and c in w and w[c]["user"] == user:
                seen.add(c); queue.append(c)
    for nid in sorted(seen - set(roots) - set(sources)):
        n = w[nid]
        alive = [p for p in n["from"] if p not in gone]
        if n["store"] in ("кэш ответов", "блоки KV"):
            del w[nid]; gone.add(nid)
            log(n["store"], nid, "сброшено")
        elif not alive:
            del w[nid]; gone.add(nid); log(n["store"], nid, "удалено")
        elif n["store"] == "сводки":    # склейка как заглушка сжатия
            n["text"] = " / ".join(w[p]["text"] for p in alive)
            n["from"] = alive
            log("сводки", nid, "пересобрано")
    # 4. Резервная копия: уничтожаем ключ, байты нечитаемы.
    if keys.pop((user, slot), None) is not None:
        log("резервная копия", slot, "ключ уничтожен")
    return dict(receipt), gone

world, keys, backup = build_world()
boris = {k: copy.deepcopy(n) for k, n in world.items()
         if n["user"] == "u77"}
receipt, gone = erase(world, keys, backup, "u42", "город", stems)

assert erasure_scan(world, keys, backup, "u42", gone, stems) == []
assert len(receipt) == 9 and "журнал" not in receipt
assert world["r-name@1"]["text"] == "имя: Аня"       # имя не задели
assert world["sum-s2"]["text"] == "Дедлайн диплома 15 декабря"
assert {k: world[k] for k in boris} == boris         # Борис цел
naive_grep = [k for k, n in world.items()
              if "казан" in n["text"].lower()]
assert sorted(naive_grep) == sorted(boris)   # три ложные тревоги


# --- Тот же листинг: намеренно сломанное удаление только основной записи ---
def delete_record_only(w, user, slot):
    for nid in [nid for nid, n in w.items() if n["user"] == user
                and n["store"] in MEMORY
                and n["text"].startswith(slot + ":")]:
        del w[nid]

world, keys, backup = build_world()
delete_record_only(world, "u42", "город")
hits = erasure_scan(world, keys, backup, "u42", city_ids, stems)
where = {world[h]["store"] if h in world else "резервная копия"
         for h in hits}
assert len(hits) == 10
assert where == {"история", "сводки", "индекс", "кэш ответов",
                 "блоки KV", "выгрузка", "резервная копия"}


if __name__ == "__main__":
    print("Все проверки пройдены:", __file__)
