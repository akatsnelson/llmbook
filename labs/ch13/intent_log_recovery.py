"""Глава 13, раздел 13.4 «Сбои и идемпотентность»:
листинг «Журнал намерений и восстановление после падения» (полная версия).

В книге оставлена сокращённая версия; этот файл самодостаточен и
выполняется в ноутбуке главы notebooks/13_tools.ipynb.

Агент падает в одной из четырёх точек A–D шага записи и восстанавливается
по журналу. Файл пересчитывает таблицу 13.3 «Число заявок на сервере
после падения и восстановления» для трёх режимов: без журнала и ключа (none), журнал и новый ключ
на каждой попытке (fresh), журнал и стабильный ключ (stable). Четвёртый
режим key — стабильный ключ без журнала — показывает, что без журнала ключ
ломается, когда модель после перезапуска переформулирует заголовок.
Журнал здесь — список в памяти, который делает вид, что переживает падение.

Запуск из корня репозитория: python labs/ch13/intent_log_recovery.py
"""

import hashlib
import json
import uuid


def canonical(obj):           # канонический JSON из раздела 13.2
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)


def idem_key(run_id, step, tool, args):
    text = f"{run_id}|{step}|{tool}|{canonical(args)}"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class TicketServer:
    """Система заявок: помнит результат по ключу."""
    def __init__(self):
        self.by_key, self.created = {}, 0

    def create_ticket(self, args, key=None):
        if key in self.by_key:
            old_args, result = self.by_key[key]
            if old_args != args:
                raise ValueError("ключ занят другим намерением")
            return result
        self.created += 1
        result = {"ticket_id": f"T-{16 + self.created}"}
        if key is not None:
            self.by_key[key] = (args, result)
        return result


args = {"title": "KV-кэш 3x512",
        "body": "2 359 296 байт [S:ch-04de13a141cc]"}


class SimulatedCrash(Exception):
    pass


# none — без журнала и ключа; fresh — журнал и новый ключ;
# stable — журнал и K; key — K без журнала
def run_step(mode, log, server, args, crash_at=None, key=None):
    logged = mode in ("fresh", "stable")
    if crash_at == "A":
        raise SimulatedCrash
    if key is None and mode != "none":
        key = (uuid.uuid4().hex if mode == "fresh" else
               idem_key("run-42", 3, "create_ticket", args))
    if logged:
        log.append({"type": "intent", "key": key, "args": args})
    if crash_at == "B":
        raise SimulatedCrash
    result = server.create_ticket(args, key)    # внешний эффект
    if crash_at == "C":
        raise SimulatedCrash                    # опасное окно
    if logged:
        log.append({"type": "result", "key": key, "result": result})
    if crash_at == "D":
        raise SimulatedCrash
    return result


def recover(mode, log, server, args_now):       # журнал шага 3
    if not log:                     # о шаге ничего не известно
        return run_step(mode, log, server, args_now)
    if log[-1]["type"] == "result":  # шаг сделан: не трогаем
        return log[-1]["result"]
    intent = log[-1]                # намерение без результата
    key = intent["key"] if mode == "stable" else None
    return run_step(mode, log, server, intent["args"], key=key)


def tickets_after_crash(mode, point, args_now=args):
    log, server = [], TicketServer()
    try:
        run_step(mode, log, server, args, crash_at=point)
    except SimulatedCrash:
        pass                        # процесс упал и перезапущен
    result = recover(mode, log, server, args_now)
    return server.created, result


table = {m: [tickets_after_crash(m, p)[0] for p in "ABCD"]
         for m in ("none", "fresh", "stable")}
assert table == {"none":   [1, 1, 2, 2],
                 "fresh":  [1, 1, 2, 1],
                 "stable": [1, 1, 1, 1]}
T17 = {"ticket_id": "T-17"}
assert all(tickets_after_crash("stable", p)[1] == T17
           for p in "ABCD")

# после перезапуска модель сформулировала заголовок иначе
reworded = {**args, "title": "KV-кэш: 3 запроса по 512"}
assert tickets_after_crash("key", "C")[0] == 1   # те же аргументы
assert tickets_after_crash("key", "C", reworded)[0] == 2
assert tickets_after_crash("stable", "C", reworded)[0] == 1

print("intent_log_recovery: все проверки пройдены")
