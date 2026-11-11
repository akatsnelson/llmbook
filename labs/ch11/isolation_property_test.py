"""Свойство-тест изоляции памяти: хранилище главы ScopedStore и две непредусмотренные поломки.

Листинги «Свойство-тест изоляции» и «Две поломки против свойство-теста» из главы 11 (раздел 11.6
«Изоляция, атаки на память и промышленная практика», подраздел «Изоляция пользователей»). Тест гоняет 1000 случайных
операций трёх владельцев (org1/u42, org1/u77 и org2/u42 — тот же номер у другого арендатора):
запись через apply с политикой роли, чтение через read_memory, вопрос к кэшу ответов. Здесь же
вариант с удалениями (forget). Ниже, отдельным блоком, — упрощённая модель ScopedMemory, на
которую опирается ноутбук главы.

Запуск из корня репозитория: python labs/ch11/isolation_property_test.py
"""
import hashlib
import random
import sys
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from memory_store import (Candidate, MemoryStore, Message, Scope, ScopedStore,   # noqa: E402
                          date, read_memory)


def isolation_test(store, n_ops=1000, seed=0, deletes=False):
    """Номер первой операции, где владелец получил чужое (счёт с 1), или None."""
    rng, cache = random.Random(seed), {}
    users = [Scope("org1", "u42"), Scope("org1", "u77"), Scope("org2", "u42")]
    ops = "wwrcd" if deletes else "wwrc"
    for i in range(1, n_ops + 1):
        who, op = rng.choice(users), rng.choice(ops)
        day = date(2026, 11, 22) + timedelta(days=i)
        if op == "w":                      # запись через политику
            slot = rng.choice(["город", "стиль.ответа"])
            store.apply(Message(f"t{i}", who, "user", day),
                        Candidate(slot, f"{who}:{i}", day, 0.9))
        elif op == "d":                    # «забудь»: надгробие
            slot = rng.choice(["город", "стиль.ответа"])
            store.forget(Message(f"t{i}", who, "user", day), slot)
        elif op == "r":                    # чтение вернуло только своё?
            chosen, _ = read_memory(store, who, "где поужинать?", 40, day)
            if any(not r.value.startswith(f"{who}:") for r in chosen):
                return i
        else:                              # кэш: ответ собран для меня?
            q = rng.choice(["где поужинать?", "что я не ем?"])
            if cache.setdefault(store.cache_key(who, q), who) != who:
                return i
    return None


class TenantBlind(ScopedStore):            # дверь смотрит только на user
    def slots(self, owner):
        return MemoryStore.slots(self, owner.user)


class QueryOnlyKey(ScopedStore):           # ключ кэша из текста вопроса
    def cache_key(self, owner, window, **params):
        return hashlib.sha256(window.encode()).hexdigest()


# числа из книги
assert isolation_test(ScopedStore()) is None
assert isolation_test(TenantBlind()) == 16
assert isolation_test(QueryOnlyKey()) == 14
# с удалениями и на других зёрнах исправное хранилище тоже не течёт, а поломки ловятся
for seed in range(20):
    assert isolation_test(ScopedStore(), seed=seed, deletes=True) is None
    assert isolation_test(TenantBlind(), seed=seed, deletes=True) is not None
    assert isolation_test(QueryOnlyKey(), seed=seed, deletes=True) is not None


# ---------------------------------------------------------------------------
# Упрощённая модель для ноутбука notebooks/11_memory.ipynb (подраздел «Изоляция пользователей»).
# --- По мотивам листинга «Свойство-тест изоляции» ---
from typing import NamedTuple

class ScopeNT(NamedTuple):     # область видимости в упрощённой модели
    tenant: str
    user: str

class ScopedMemory:
    def __init__(self, cache_key_has_user=True):
        self.rows = {}         # (scope, slot) -> value
        self.version = {}      # scope -> версия памяти пользователя
        self.cache = {}        # ключ -> (чья память, факты в ответе)
        self.cache_key_has_user = cache_key_has_user

    def _check(self, scope):
        if not isinstance(scope, ScopeNT):
            raise TypeError("операция без области видимости")

    def _bump(self, scope):
        self.version[scope] = self.version.get(scope, 0) + 1

    def write(self, scope, slot, value):
        self._check(scope)
        self.rows[(scope, slot)] = value
        self._bump(scope)

    def delete(self, scope, slot):
        self._check(scope)
        if self.rows.pop((scope, slot), None) is not None:
            self._bump(scope)

    def read(self, scope):     # фильтр по области ДО ранжирования
        self._check(scope)
        return {slot: v for (s, slot), v in self.rows.items()
                if s == scope}

    def cache_key(self, scope, query):
        if self.cache_key_has_user:
            parts = [scope.tenant, scope.user,
                     str(self.version.get(scope, 0)), query]
        else:                  # ошибка: только арендатор и текст
            parts = [scope.tenant, query]
        raw = "\x1f".join(parts).encode()
        return hashlib.sha256(raw).hexdigest()[:12]

    def answer(self, scope, query):
        key = self.cache_key(scope, query)
        if key not in self.cache:          # «вызов модели»
            facts = tuple(sorted(self.read(scope).items()))
            self.cache[key] = (scope, facts)
        return self.cache[key]

def isolation_test_simple(store, n_ops=1000, seed=0):
    """Номер первой операции с утечкой (счёт с 1) или None."""
    rng = random.Random(seed)
    users = [ScopeNT("org1", u) for u in ("u42", "u77", "u13")]
    slots = ["город", "еда.ограничения", "стиль.ответа"]
    queries = ["где поужинать?", "что я не ем?", "что ты помнишь?"]
    for i in range(n_ops):
        scope, op = rng.choice(users), rng.choice("wwrda")
        slot = rng.choice(slots)
        if op == "w":
            store.write(scope, slot, f"{scope.user}:{i}")
        elif op == "d":
            store.delete(scope, slot)
        elif op == "r":
            mine = scope.user + ":"
            if any(not v.startswith(mine)
                   for v in store.read(scope).values()):
                return i + 1   # чтение вернуло чужую запись
        else:
            owner, _ = store.answer(scope, rng.choice(queries))
            if owner != scope:
                return i + 1   # кэш отдал ответ по чужой памяти
    return None

anya, boris = ScopeNT("org1", "u42"), ScopeNT("org1", "u77")
assert isolation_test_simple(ScopedMemory()) is None      # 0 утечек
good = ScopedMemory()
assert (good.cache_key(anya, "где поужинать?")
        != good.cache_key(boris, "где поужинать?"))
assert isolation_test_simple(ScopedMemory(cache_key_has_user=False)) == 7
try:
    good.read(None)
    raise AssertionError("чтение без области должно падать")
except TypeError:
    pass
# ключи упрощённой модели (обрезанные до 12 знаков только для печати в ноутбуке)
bad = ScopedMemory(cache_key_has_user=False)
assert (bad.cache_key(anya, "где поужинать?")
        == bad.cache_key(boris, "где поужинать?") == "0ee268ce066a")
good.version = {boris: 2, anya: 7}
assert good.cache_key(boris, "где поужинать?") == "de697a937adf"
assert good.cache_key(anya, "где поужинать?") == "0edf94a561db"
good._bump(anya)                       # ещё одна запись Ани: версия 8
assert good.cache_key(anya, "где поужинать?") == "945aba09ae3d"


if __name__ == "__main__":
    print("Все проверки пройдены:", __file__)
