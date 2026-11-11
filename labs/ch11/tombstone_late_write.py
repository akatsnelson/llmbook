"""Надгробие против опоздавшей записи: наивное хранилище «последняя запись побеждает» и хранилище с версиями.

Полная версия листинга «Надгробие против опоздавшей записи» из главы 11 книги (раздел 11.5 «Гонка версий и право на удаление»).
В книге оставлена сокращённая версия; здесь весь код целиком, с теми же проверками.

Запуск из корня репозитория: python labs/ch11/tombstone_late_write.py
"""

# --- Конфликт версий из листинга «Запись с проверкой версии» ---
class VersionConflict(Exception):
    pass


# --- Листинг «Надгробие против опоздавшей записи» ---
import copy

class LastWriteWins:
    """Без версий: удалённую строку любая запись создаёт снова."""
    def __init__(self):
        self.rows = {}
    def read(self, rid):
        return self.rows.get(rid)
    def version(self, rid):
        return 0
    def write(self, rid, value, expected):
        self.rows[rid] = value
    def forget(self, rid):
        self.rows.pop(rid, None)

class TombstoneStore:
    """Версии и надгробие: удалённая строка помнит свою версию."""
    def __init__(self):
        self.rows = {}               # rid -> (value, version, status)
    def read(self, rid):
        value, _, status = self.rows.get(rid, (None, 0, "deleted"))
        return value if status == "active" else None
    def version(self, rid):
        return self.rows.get(rid, (None, 0, None))[1]
    def write(self, rid, value, expected):
        now = self.version(rid)
        if now != expected:      # VersionConflict из первого листинга
            raise VersionConflict(f"ожидали v{expected}, сейчас v{now}")
        self.rows[rid] = (value, expected + 1, "active")
    def forget(self, rid):
        self.rows[rid] = (None, self.version(rid) + 1, "deleted")

def late_write(store):
    store.write("r-city", "Казань", 0)         # v1, 2026-09-03
    store.write("r-city", "Самара", 1)         # v2, 2026-11-10
    expected = store.version("r-city")     # t1: извлечение прочло v2
    store.forget("r-city")                 # t2: просьба забыть город
    try:
        store.write("r-city", "Самара", expected)    # t3: опоздала
    except VersionConflict as err:
        return str(err)

def reextract(store, messages):
    # Повторное извлечение новой версией программы идёт по истории.
    if "s2/t5" in messages:                    # источник ещё жив?
        store.write("r-city", "Самара", store.version("r-city"))

safe, naive = TombstoneStore(), LastWriteWins()
assert late_write(safe) == "ожидали v2, сейчас v3"
assert late_write(naive) is None
assert naive.read("r-city") == "Самара"        # воскресла

reextract(safe, messages={})                   # сообщение удалено
assert safe.read("r-city") is None and safe.version("r-city") == 3
forgotten = copy.deepcopy(safe)                # а если его забыли?
reextract(forgotten, messages={"s2/t5": "С 1 ноября живу в Самаре"})
assert forgotten.read("r-city") == "Самара"    # надгробие не спасло


if __name__ == "__main__":
    print("Все проверки пройдены:", __file__)
