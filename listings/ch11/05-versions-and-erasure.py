"""Листинги главы 11 «Контекст и память» книги «Большие языковые модели», файл 05-versions-and-erasure.

Код и вывод — ровно как в книге; текст книги сюда не входит. Листинги идут по порядку и
опираются друг на друга: запускайте файл целиком или ячейками сверху вниз (разметка # %%).
Проверка, что вывод совпал с книгой: python scripts/check_listings.py listings/ch11/05-versions-and-erasure.py
"""

# %% Листинг «Запись с проверкой версии» — раздел «Потерянное обновление и оптимистическая блокировка»
import sys
sys.path.insert(0, "labs/ch11")          # код раздела 11.4
from memory_store import (MemoryStore, Message, new_record, decide,
                          tell, autumn, date)

class VersionConflict(Exception):        # отказ CAS: версия уже другая
    pass

def put(self, msg, slot, value, expected=None, since=None):
    history = self.versions(msg.user, slot)
    now = len(history)                   # номер последней версии слота
    if expected is not None and expected != now:
        raise VersionConflict(f"ожидали v{expected}, сейчас v{now}")
    since = since or msg.day             # начало действия значения
    if self.active(msg.user, slot):      # прежнюю версию закрываем
        history[-1] = history[-1]._replace(status="superseded",
                                           valid_to=since)
    history.append(new_record(msg, slot, value, since, now + 1))
    return now + 1

MemoryStore.put = put                    # новый метод хранилища

# %% Листинг «Шесть порядков шагов без версии и с CAS» — раздел «Потерянное обновление и оптимистическая блокировка»
from itertools import combinations
FOOD, nov12 = "еда.ограничения", date(2026, 11, 12)

def race(order, cas):
    store, seen = MemoryStore(), {}
    tell(store, "s1/t9", "u42", date(2026, 10, 20), FOOD, "арахис")
    extra = {"Т": "глютен", "Н": "лактоза"}
    for who, step in order:              # "Тч" -> ("Т", "ч")
        if step == "ч":                  # запомнить значение и версию
            r = store.active("u42", FOOD)
            seen[who] = (r.value, r.version)
            continue
        value, version = seen[who]
        msg = Message(f"{who}/t1", "u42", "user", nov12)
        try:
            store.put(msg, FOOD, f"{value}, {extra[who]}",
                      version if cas else None)
        except VersionConflict:          # перечитать и повторить
            r = store.active("u42", FOOD)
            store.put(msg, FOOD, f"{r.value}, {extra[who]}", r.version)
    return store.active("u42", FOOD).value

for phone in combinations(range(4), 2):  # места шагов телефона
    order = ["Нч", "Нз"]
    for i, step in zip(phone, ("Тч", "Тз")):
        order.insert(i, step)
    print(" ".join(order), "|", race(order, cas=False),
          "|", race(order, cas=True))
# Вывод в книге:
# | Тч Тз Нч Нз | арахис, глютен, лактоза | арахис, глютен, лактоза
# | Тч Нч Тз Нз | арахис, лактоза | арахис, глютен, лактоза
# | Тч Нч Нз Тз | арахис, глютен | арахис, лактоза, глютен
# | Нч Тч Тз Нз | арахис, лактоза | арахис, глютен, лактоза
# | Нч Тч Нз Тз | арахис, глютен | арахис, лактоза, глютен
# | Нч Нз Тч Тз | арахис, лактоза, глютен | арахис, лактоза, глютен

# %% Листинг «Надгробие против опоздавшей записи» — раздел «Удаление производных копий»
def forget(self, msg, slot):
    history = self.versions(msg.user, slot)
    for i, r in enumerate(history):      # значение уходит из всех версий
        history[i] = r._replace(value=None, status="deleted")
    history.append(new_record(msg, slot, None, msg.day,     # надгробие
                              len(history) + 1, "deleted"))

MemoryStore.forget = forget

def late_jobs(write):                    # две фоновые задачи и «забудь»
    store = autumn(MemoryStore())
    job = Message("s2/t5", "u42", "user", date(2026, 11, 10))
    seen = store.active("u42", "город").version        # t1: прочла v2
    store.forget(Message("s5/t1", "u42", "user", date(2026, 11, 20)),
                 "город")                               # t2
    for expected in (seen, 3):           # опоздавшая; стартовавшая после
        try:
            write(store, job, "город", "Самара", expected)
            print(f"  поверх v{expected}: записано")
        except VersionConflict as err:
            print(f"  поверх v{expected}: {type(err).__name__}: {err}")
    r = store.active("u42", "город")
    print("  город:", r.value if r else None)

cas_put = MemoryStore.put                # put с проверкой версии
late_jobs(cas_put)
# Вывод в книге:
# |   поверх v2: VersionConflict: ожидали v2, сейчас v3
# |   поверх v3: записано
# |   город: Самара

# %% Листинг «Запись не старше надгробия» — раздел «Удаление производных копий»
class Erased(VersionConflict):           # источник не новее надгробия
    pass

def put(self, msg, slot, value, expected=None, since=None):
    # msg.day --- день источника; проверка и запись --- один вызов
    last = self.versions(msg.user, slot)[-1:]
    if last and last[0].value is None and msg.day <= last[0].recorded_at:
        raise Erased(f"источник {msg.id} не новее надгробия"
                     f" от {last[0].recorded_at:%d.%m}")
    return cas_put(self, msg, slot, value, expected, since)

MemoryStore.put = put
late_jobs(MemoryStore.put)
# Вывод в книге:
# |   поверх v2: Erased: источник s2/t5 не новее надгробия от 20.11
# |   поверх v3: Erased: источник s2/t5 не новее надгробия от 20.11
# |   город: None

# %% Листинг «Зонд удаления» — раздел «Удаление производных копий»
from collections import Counter
from memory_store import build_world, xor

def erasure_scan(w, keys, backup, user, ids, stems):
    # Ребро к удалённому id ищем у всех, значение --- только у user.
    hits = [nid for nid, n in w.items() if set(n["from"]) & set(ids)
            or n["user"] == user and any(s in n["text"].lower()
                                         for s in stems)]
    for (u, slot), blob in backup.items():
        if u == user and (u, slot) in keys:    # читаемо, пока жив ключ
            text = xor(blob, keys[(u, slot)]).decode().lower()
            if any(s in text for s in stems):
                hits.append(f"backup:{slot}")
    return hits

def where(w, hits):                     # следы по хранилищам
    c = Counter(w[h]["store"] if h in w else "бэкап" for h in hits)
    return ", ".join(f"{k} {v}" for k, v in c.items())

world, keys, backup = build_world()
stems = ("казан", "самар")              # основы ловят и Казани, и Самаре
city_ids = {"r-city@1", "r-city@2"}
hits = erasure_scan(world, keys, backup, "u42", city_ids, stems)
print(len(hits), "следов:")
print(where(world, hits))
print("у чужих:", [h for h in hits if h in world
                   and world[h]["user"] != "u42"])
# Вывод в книге:
# | 13 следов:
# | история 2, версии 1, записи 1, сводки 3, индекс 2, кэш ответов 1, страницы KV 1, выгрузка 1, бэкап 1
# | у чужих: ['sum-b7']

# %% Листинг «Удаление по графу и удаление одной записи» — раздел «Удаление производных копий»
from memory_store import erase, delete_record_only

for name, procedure in [("по графу", erase),
                        ("только записи", delete_record_only)]:
    world, keys, backup = build_world()
    receipt = procedure(world, keys, backup, "u42", "город", stems)
    hits = erasure_scan(world, keys, backup, "u42", city_ids, stems)
    print(f"{name}: в квитанции {len(receipt)}, следов {len(hits)}")
    if name == "по графу":
        for store_name, actions in receipt.items():
            print(f"  {store_name}: {', '.join(actions)}")
        print("  сводка s1:", world["sum-s1"]["text"])
        print("  сводка Бориса:", world["sum-b7"]["text"])
    else:
        print(" ", where(world, hits))
# Вывод в книге:
# | по графу: в квитанции 9, следов 0
# |   версии: надгробие
# |   записи: надгробие
# |   история: вычищено, удалено
# |   кэш ответов: сброшено
# |   выгрузка: удалено
# |   индекс: удалено, удалено
# |   страницы KV: сброшено
# |   сводки: пересобрано у u77: инцидент, пересобрано, пересобрано
# |   резервная копия: ключ уничтожен
# |   сводка s1: Меня зовут Аня, я живу в *** / Пиши коротко
# |   сводка Бориса: город: Казань
# | только записи: в квитанции 2, следов 11
# |   история 2, сводки 3, индекс 2, кэш ответов 1, страницы KV 1, выгрузка 1, бэкап 1

# %% Листинг «Факт в весах: найти, разучить, вернуть» — раздел «Удаление производных копий»
import torch
from booklm.lab import load, prompt_ids, perplexity
from booklm.experiments.ch11 import (anya_stories, other_stories,
                                     lora_steps, weight_bytes)

tok, model = load("base")
sama = tok.encode(" Самаре").ids[0]                # первый кусок: « Сама»

@torch.no_grad()
def probe(stage):                                  # поведенческий зонд
    row = f"{stage:19}"
    for text in ("Аня жила в", "Девочка Аня жила в городе"):
        p = model(torch.tensor([prompt_ids(tok, text)]))[0, -1].softmax(-1)
        row += f" {p[sama]:.4f} ({int((p > p[sama]).sum()) + 1:>4})"
    print(row, f"{perplexity(model, n=8192):6.2f}")

print("этап               «Аня жила в»  «…в городе»  перплексия")
probe("исходная")
lora_steps(model, anya_stories(), steps=200)       # 30 сказок об Ане
probe("сказки об Ане")
print("байтов «Самар»:", weight_bytes(model).count("Самар".encode()))
lora_steps(model, anya_stories(), steps=10, ascent=True)
probe("разучивание")
lora_steps(model, other_stories(), steps=10)       # ни Ани, ни Самары
probe("чужие сказки")
# Вывод в книге:
# | этап               «Аня жила в»  «…в городе»  перплексия
# | исходная            0.0000 (4438) 0.0000 (2172)   6.93
# | сказки об Ане       0.9994 (   1) 0.0000 (  67)   9.31
# | байтов «Самар»: 0
# | разучивание         0.0012 (   3) 0.0000 ( 194)  12.85
# | чужие сказки        0.8143 (   1) 0.0000 ( 755)   8.14
