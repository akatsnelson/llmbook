"""Глава 13, раздел 13.6 «Передача задачи и протокол MCP»:
листинг «Передача задачи без расширения прав» (полная версия).

В книге оставлена сокращённая версия; этот файл самодостаточен.

Передача задачи (handoff) держит четыре правила: мандат ребёнка — пересечение
мандата родителя с запрошенным, бюджет шагов делится, а не копируется,
контекст не общий, номер прогона ребёнка выводится из номера родителя и шага
передачи. Последние проверки пересчитывают ключи идемпотентности из раздела
13.4: повторная передача после падения даёт те же ключи, а одинаковый вызов
в родителе и в ребёнке — разные.

Запуск из корня репозитория: python labs/ch13/handoff_scope.py
"""

import hashlib
import json


class AgentRun:
    def __init__(self, run_id, scope, budget, parent_run_id=None):
        self.run_id, self.scope = run_id, frozenset(scope)
        self.budget, self.parent_run_id = budget, parent_run_id


def handoff(parent, requested_scope, steps, at_step):
    if not 0 < steps <= parent.budget:
        raise ValueError(f"просит {steps} шагов, есть {parent.budget}")
    parent.budget -= steps                   # делим, а не копируем
    return AgentRun(f"{parent.run_id}/{at_step}",   # без uuid4
                    parent.scope & frozenset(requested_scope), steps,
                    parent_run_id=parent.run_id)


def idem_key(run_id, step, tool, args):     # как в разделе 13.4
    canon = json.dumps(args, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False)
    text = f"{run_id}|{step}|{tool}|{canon}"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


PUBLIC, FINANCE = "docs:read:public", "docs:read:finance"
coordinator = lambda: AgentRun("run-42", {PUBLIC, "tickets:write"}, 12)
parent = coordinator()
child = handoff(parent, {PUBLIC, FINANCE}, 4, at_step=3)
assert child.scope == {PUBLIC} and child.scope <= parent.scope
assert (parent.budget, child.budget) == (8, 4)
assert child.parent_run_id == "run-42"
try:
    handoff(coordinator(), {PUBLIC}, 13, at_step=3)
    raise AssertionError("ребёнок получил больше шагов, чем было")
except ValueError:
    pass

again = handoff(coordinator(), {PUBLIC}, 4, at_step=3)  # после падения
args = {"query": "развёртывание BookLM", "k": 3, "access": "public"}
key = lambda run_id: idem_key(run_id, 1, "search_docs", args)
assert key(again.run_id) == key(child.run_id)   # то же намерение
assert key(child.run_id) != key("run-42")       # другое намерение
print("handoff: мандат", sorted(child.scope), "бюджеты",
      (parent.budget, child.budget), "номер ребёнка", child.run_id)
