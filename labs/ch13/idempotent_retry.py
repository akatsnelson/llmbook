"""Глава 13, раздел 13.4 «Сбои и идемпотентность»: листинги «Ключ идемпотентности и сервер,
который его помнит», «Потерянный ответ и повтор» и «Повтор с экспоненциальной задержкой» (полная версия).

В книге оставлена сокращённая версия; этот файл самодостаточен и
выполняется в ноутбуке главы notebooks/13_tools.ipynb.

Система заявок помнит результат по ключу идемпотентности. Ненадёжная сеть
работает по расписанию: "down" — запрос не дошёл, "lost" — сервер исполнил
запрос, а ответ потерялся. Повтор со стабильным ключом даёт одну заявку,
ключ uuid4() внутри цикла повторов — две. Паузы экспоненциальной задержки
только записываются, листинг их не ждёт.

Запуск из корня репозитория: python labs/ch13/idempotent_retry.py
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
            if old_args != args:    # тот же ключ, другие аргументы
                raise ValueError("ключ занят другим намерением")
            return result           # повтор: старый результат
        self.created += 1
        result = {"ticket_id": f"T-{16 + self.created}"}
        if key is not None:
            self.by_key[key] = (args, result)
        return result


def call_with_retry(server, fates, args, get_key,
                    attempts=3, base=0.5, factor=2):
    keys, replies, pauses = [], [], []
    for i in range(1, attempts + 1):
        keys.append(get_key())
        fate = fates[i - 1] if i <= len(fates) else "ok"
        if fate != "down":          # запрос дошёл: эффект случился
            replies.append(server.create_ticket(args, keys[-1]))
        if fate == "ok":
            return replies, keys, pauses
        if i < attempts:            # time.sleep здесь нет
            pauses.append(base * factor ** (i - 1))
    raise TimeoutError("ни одна попытка не дождалась ответа")


args = {"title": "KV-кэш 3x512",
        "body": "2 359 296 байт [S:ch-04de13a141cc]"}
key = idem_key("run-42", 3, "create_ticket", args)  # до попыток
assert key.startswith("9d1f76531409")

server = TicketServer()
replies, keys, pauses = call_with_retry(
    server, ["lost", "ok"], args, lambda: key)
assert server.created == 1
assert replies == [{"ticket_id": "T-17"}] * 2
assert len(set(keys)) == 1 and pauses == [0.5]

broken = TicketServer()          # uuid4() внутри цикла повторов
replies, keys, _ = call_with_retry(
    broken, ["lost", "ok"], args, lambda: uuid.uuid4().hex)
assert broken.created == 2 and len(set(keys)) == 2
assert replies[-1] == {"ticket_id": "T-18"}

swapped = {"body": args["body"], "title": args["title"]}
assert idem_key("run-42", 3, "create_ticket", swapped) == key
assert json.dumps(swapped) != json.dumps(args)  # без сортировки

_, _, pauses = call_with_retry(
    TicketServer(), ["down", "down", "ok"], args, lambda: key)
assert pauses == [0.5, 1.0] and 3 * 10 + sum(pauses) == 31.5

try:                    # тот же ключ с другими аргументами: отказ
    server.create_ticket({**args, "title": "другое"}, key)
    raise AssertionError("конфликт ключа должен быть отклонён")
except ValueError:
    assert server.created == 1

print("idempotent_retry: все проверки пройдены")
