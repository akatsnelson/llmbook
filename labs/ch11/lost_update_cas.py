"""Потерянное обновление и сравнение с заменой (CAS): перебор всех шести порядков и счёт конфликтов.

Расширенная версия листинга «Шесть порядков шагов без версии и с CAS» из главы 11 книги
(раздел 11.5 «Гонка версий и право на удаление»):
ограничения хранятся множеством, и для каждого порядка считаются конфликты CAS. Это однопоточная
симуляция: в ней проверка версии и запись неделимы сами собой. Опыт с sqlite3 и двумя настоящими
потоками лежит в ноутбуке главы, notebooks/11_memory.ipynb.

Запуск из корня репозитория: python labs/ch11/lost_update_cas.py
"""

# --- Перебор порядков: наивное хранилище и CAS ---
from itertools import permutations
from typing import NamedTuple

class VersionConflict(Exception):
    pass

class Row(NamedTuple):
    value: frozenset        # ограничения в еде: множество строк
    version: int

class NaiveStore:
    """Последняя запись побеждает: версию никто не проверяет."""
    def __init__(self, value):
        self.row = Row(frozenset(value), 1)
    def read(self):
        return self.row
    def write(self, value, expected):
        self.row = Row(frozenset(value), self.row.version + 1)

class CASStore(NaiveStore):
    """Сравнение с заменой: пишем поверх только прочитанной версии."""
    def write(self, value, expected):
        now = self.row.version
        if now != expected:
            raise VersionConflict(f"ожидали v{expected}, сейчас v{now}")
        self.row = Row(frozenset(value), expected + 1)

def run(store, schedule, adds):
    seen, conflicts = {}, 0
    for device, op in schedule:
        if op == "read":
            seen[device] = store.read()
            continue
        row = seen[device]
        while True:
            try:
                new = row.value | {adds[device]}
                store.write(new, expected=row.version)
                break
            except VersionConflict:
                conflicts += 1
                row = store.read()      # перечитать и объединить заново
    return store.row, conflicts

adds = {"телефон": "глютен", "ноутбук": "лактоза"}
schedule = [("телефон", "read"), ("ноутбук", "read"),     # t1, t2
            ("телефон", "write"), ("ноутбук", "write")]   # t3, t4

row, conflicts = run(NaiveStore({"арахис"}), schedule, adds)
assert row == ({"арахис", "лактоза"}, 3) and conflicts == 0
row, conflicts = run(CASStore({"арахис"}), schedule, adds)
assert row == ({"арахис", "глютен", "лактоза"}, 3)
assert conflicts == 1               # ровно один отказ

# Все порядки, где каждое устройство сначала читает, потом пишет.
steps = [("телефон", "read"), ("телефон", "write"),
         ("ноутбук", "read"), ("ноутбук", "write")]
orders = {p for p in permutations(steps)
          if p.index(steps[0]) < p.index(steps[1])
          and p.index(steps[2]) < p.index(steps[3])}
lost = sum(len(run(NaiveStore({"арахис"}), o, adds)[0].value) < 3
           for o in orders)
retries = sum(run(CASStore({"арахис"}), o, adds)[1] for o in orders)
assert (len(orders), lost, retries) == (6, 4, 4)


if __name__ == "__main__":
    print("Все проверки пройдены:", __file__)
