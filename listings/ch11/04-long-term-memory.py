"""Листинги главы 11 «Контекст и память» книги «Большие языковые модели», файл 04-long-term-memory.

Код и вывод — ровно как в книге; текст книги сюда не входит. Листинги идут по порядку и
опираются друг на друга: запускайте файл целиком или ячейками сверху вниз (разметка # %%).
Проверка, что вывод совпал с книгой: python scripts/check_listings.py listings/ch11/04-long-term-memory.py
"""

# %% Листинг «Запись памяти и выбор операции» — раздел «Учебное хранилище памяти»
from collections import namedtuple
from datetime import date

Message = namedtuple("Message", "id user role day")  # id: s2/t5
Candidate = namedtuple("Candidate", "slot value valid_from confidence")
MemoryRecord = namedtuple("MemoryRecord",     # все поля записи памяти
    "id user slot value valid_from valid_to recorded_at"
    " version status source_msg source_role extractor_version")
EXTRACTOR = "rules-0.1"

def new_record(msg, slot, value, valid_from, version, status="active"):
    return MemoryRecord(
        f"{msg.user}:{slot}:v{version}", msg.user, slot, value,
        valid_from, None, msg.day, version, status,
        msg.id, msg.role, EXTRACTOR)

def decide(old, msg, c):
    # old: активная версия слота или None; c: кандидат.
    if msg.role != "user" or c.confidence < 0.7:
        return "НИЧЕГО"                  # чужая роль или догадка
    if c.value is None:                  # value=None: "забыть"
        return "УДАЛИТЬ" if old else "НИЧЕГО"
    if old is None:
        return "ДОБАВИТЬ"
    if old.value == c.value or c.valid_from < old.valid_from:
        return "НИЧЕГО"                  # повтор или старая новость
    if c.valid_from == old.valid_from:
        return "СПРОСИТЬ"                # ничья: решает пользователь
    return "ОБНОВИТЬ"

# %% Листинг «Хранилище памяти с четырьмя операциями» — раздел «Учебное хранилище памяти»
class MemoryStore:
    def __init__(self):
        self.data = {}      # владелец -> слот -> [версии по порядку]

    def slots(self, owner): # единственный путь к данным владельца
        return self.data.setdefault(owner, {})

    def versions(self, owner, slot):
        return self.slots(owner).setdefault(slot, [])

    def active(self, owner, slot):
        live = [r for r in self.versions(owner, slot)
                if r.status == "active"]
        assert len(live) <= 1, "две активные версии одного слота"
        return live[0] if live else None

    def apply(self, msg, c):
        op = decide(self.active(msg.user, c.slot), msg, c)
        history = self.versions(msg.user, c.slot)
        if op == "ОБНОВИТЬ":               # старую версию закрываем
            history[-1] = history[-1]._replace(
                status="superseded", valid_to=c.valid_from)
        if op == "УДАЛИТЬ":                # пока только пометка
            history[-1] = history[-1]._replace(status="deleted")
        if op in ("ДОБАВИТЬ", "ОБНОВИТЬ"):
            history.append(new_record(msg, c.slot, c.value,
                                      c.valid_from, len(history) + 1))
        return op

# %% Листинг «Осень Ани в хранилище памяти» — раздел «Учебное хранилище памяти»
def tell(store, msg_id, user, day, slot, value, since=None,
         role="user", sure=0.9):
    return store.apply(Message(msg_id, user, role, day),
                       Candidate(slot, value, since or day, sure))

store, sep3 = MemoryStore(), date(2026, 9, 3)
tell(store, "s1/t2", "u42", sep3, "имя", "Аня")
tell(store, "s1/t2", "u42", sep3, "город", "Казань")
tell(store, "s1/t3", "u42", sep3, "стиль.ответа", "коротко")
tell(store, "s1/t4", "u42", sep3, "язык.программирования", "Python 3.12")
tell(store, "s1/t9", "u42", date(2026, 10, 20), "еда.ограничения", "арахис")
tell(store, "b1/t1", "u77", date(2026, 9, 5), "город", "Казань")

nov1, nov10 = date(2026, 11, 1), date(2026, 11, 10)
november = [("s2/t5", "город", "Самара", dict(since=nov1)),
            ("s2/t6", "дедлайн.диплом", "2026-12-15", {}),
            ("s2/t7", "город", "Казань", dict(role="assistant")),
            ("s2/t8", "город", "Сочи", dict(sure=0.4)),
            ("s2/t9", "город", "Казань", dict(since=date(2026, 9, 1)))]
for msg_id, slot, value, extra in november:
    print(msg_id, tell(store, msg_id, "u42", nov10, slot, value, **extra))
for r in store.versions("u42", "город"):
    print(r.version, r.value, r.valid_from, r.valid_to, r.status,
          r.source_msg)
# Вывод в книге:
# | s2/t5 ОБНОВИТЬ
# | s2/t6 ДОБАВИТЬ
# | s2/t7 НИЧЕГО
# | s2/t8 НИЧЕГО
# | s2/t9 НИЧЕГО
# | 1 Казань 2026-09-03 2026-11-01 superseded s1/t2
# | 2 Самара 2026-11-01 None active s2/t5

# %% Листинг «Двухвременной запрос и запрос-ловушка» — раздел «Учебное хранилище памяти»
def as_of(store, user, slot, valid_at, known_at):
    # Что система знала к known_at о значении слота на дату valid_at.
    known = [r for r in store.versions(user, slot)
             if r.recorded_at <= known_at and r.valid_from <= valid_at
             and r.status != "deleted"]
    # valid_to дописан задним числом: берём самую позднюю по началу.
    best = max(known, key=lambda r: (r.valid_from, r.version),
               default=None)
    return best.value if best else None

def naive_as_of(store, user, slot, valid_at, known_at):
    for r in store.versions(user, slot):    # ошибка: верит valid_to
        if (r.recorded_at <= known_at and r.valid_from <= valid_at
                and (r.valid_to is None or valid_at < r.valid_to)):
            return r.value
    return None

today, nov5 = date(2026, 11, 11), date(2026, 11, 5)
questions = [("15.10, сегодня", date(2026, 10, 15), today),
             ("05.11, тогда", nov5, nov5),
             ("05.11, сегодня", nov5, today),
             ("01.08, сегодня", date(2026, 8, 1), today)]
for name, valid_at, known_at in questions:
    args = (store, "u42", "город", valid_at, known_at)
    print(f"{name:15} {as_of(*args)!s:7} {naive_as_of(*args)}")
# Вывод в книге:
# | 15.10, сегодня  Казань  Казань
# | 05.11, тогда    Казань  None
# | 05.11, сегодня  Самара  Самара
# | 01.08, сегодня  None    None

# %% Листинг «Оценка записи памяти» — раздел «Чтение памяти»
import re
SLOT_TOPIC = {"имя": "обращение", "город": "место",
              "еда.ограничения": "еда", "стиль.ответа": "всегда",
              "язык.программирования": "работа",
              "дедлайн.диплом": "учёба"}
QUERY_TOPICS = {"поужинать": {"еда", "место"},
                "ресторан": {"еда", "место"},
                "код": {"работа"}, "диплом": {"учёба"}}
TOKENS = {"город": 10, "еда.ограничения": 12, "стиль.ответа": 9,
          "язык.программирования": 11, "дедлайн.диплом": 14, "имя": 8}

def words(text):
    return set(re.findall(r"\w+", text.lower()))

def score(r, query, today):
    topics = set().union(*(QUERY_TOPICS.get(w, set())
                           for w in words(query)))
    topic, age = SLOT_TOPIC[r.slot], (today - r.recorded_at).days
    fresh = 1.0 if age <= 30 else 0.5 if age <= 180 else 0.0
    return (2.0 * (topic in topics) + 1.0 * (topic == "всегда")
            + len(words(r.value) & words(query)) + fresh)

# %% Листинг «Чтение памяти: фильтр владельца, порог и бюджет» — раздел «Чтение памяти»
def read_memory(store, user, query, budget, today, threshold=1.0):
    slots = store.slots(user)                     # 1. владелец
    live = [store.active(user, s) for s in slots] # 2. активные,
    live = [r for r in live if r is not None]     #    без пустых
    ranked = sorted(live, key=lambda r: (-score(r, query, today),
                                         TOKENS[r.slot]))
    chosen, used = [], 0
    for r in ranked:
        if score(r, query, today) < threshold:    # 3. порог
            break
        if used + TOKENS[r.slot] <= budget:       # 4. бюджет
            chosen.append(r)
            used += TOKENS[r.slot]
    return chosen, used

query = "Посоветуй, где поужинать"
for user in ("u42", "u77", "u13"):
    chosen, used = read_memory(store, user, query, 40, today)
    print(user, [r.value for r in chosen], used)
# Вывод в книге:
# | u42 ['Самара', 'арахис', 'коротко'] 31
# | u77 ['Казань'] 10
# | u13 [] 0

# %% Листинг «Сказки с блоком памяти и без него» — раздел «Чтение памяти»
from booklm.lab import load, generate

tok, model = load("instruct")
blocks = ["", "[память] Аня живёт в Самаре. Не ест арахис.\n",
          "[память] Аня любит рисовать.\n"]
for block in blocks:
    print(block.strip() or "(без блока)")
    for ask in ("Расскажи сказку про Аню.", "Расскажи сказку про кота."):
        tale = generate(tok, model, block + ask, chat=True,
                        max_new=14, temperature=0)       # жадный выбор
        print(f"  {ask[20:]:5} -> {tale}")
# Вывод в книге:
# | (без блока)
# |   Аню.  -> Жил-был маленький мальчик по имени Ава. Ава любила играть
# |   кота. -> Жил-был маленький кот. Кот был очень голоден. Он хотел найти
# | [память] Аня живёт в Самаре. Не ест арахис.
# |   Аню.  -> Жил-был маленький мальчик по имени Сам. Он был очень голоден и
# |   кота. -> Жил-был маленький мальчик по имени Сам. Он был очень голоден и
# | [память] Аня любит рисовать.
# |   Аню.  -> Жил-был маленький девочка по имени Аня. Аня любила рисовать. Она
# |   кота. -> Жил-был маленький кот по имени Аня. Аня любила рисовать. Она
