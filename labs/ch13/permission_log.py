"""Глава 13, раздел 13.3 «Авторизация действий агента»: листинги «Хеш вызова»,
«Политика authz-v1», «Журнал с хеш-цепочкой» и «Подделка журнала и якорь» (полная версия).

В книге оставлены сокращённые версии; этот файл самодостаточен и
выполняется в ноутбуке главы notebooks/13_tools.ipynb.

Политика authz-v1 решает allow/deny/ask по пересечению прав пользователя
и мандата сессии; одобрение человека привязано к SHA-256 канонического
JSON вызова и гасится после исполнения. guarded_execute пишет решение в
журнал с хеш-цепочкой до вызова функции; подделка одной записи ловится
verify(), а пересчёт всех хешей — только якорем вне журнала.

Запуск из корня репозитория: python labs/ch13/permission_log.py
"""

import hashlib
import json
from typing import NamedTuple

def canonical(obj):           # канонический JSON, как в разделе 13.2
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)

def args_sha256(tool, args):  # хеш пары (инструмент, аргументы)
    data = canonical({"tool": tool, "arguments": args}).encode("utf-8")
    return hashlib.sha256(data).hexdigest()

POLICY = {                    # политика: данные с версией
    "version": "authz-v1",
    "right": {"search_docs": "docs:read:{access}",
              "kv_cache_bytes": None,
              "create_ticket": "tickets:write",
              "reserve_gpu": "gpu:reserve",
              "delete_document": "docs:delete"},
    "effect": {"search_docs": "read", "kv_cache_bytes": "read",
               "create_ticket": "reversible", "reserve_gpu": "reversible",
               "delete_document": "irreversible"},
    "gpu_hours_without_approval": 8,
}

class Decision(NamedTuple):
    kind: str                 # "allow" | "deny" | "ask"
    reason: str

def authorize(user_rights, session_scope, call, approvals, policy=POLICY):
    tool, args = call["tool"], call["arguments"]  # уже прошли схему
    if tool not in policy["right"]:
        return Decision("deny", "нет правила: запрет по умолчанию")
    template = policy["right"][tool]
    right = template.format(**args) if template else None
    effective = user_rights & session_scope       # пересечение!
    if right is not None and right not in effective:
        owner = "сессии" if right in user_rights else "пользователя"
        return Decision("deny", f"{right}: нет у {owner}")
    is_gpu = tool == "reserve_gpu"
    gpu_hours = args["gpus"] * args["hours"] if is_gpu else 0
    irreversible = policy["effect"][tool] == "irreversible"
    limit = policy["gpu_hours_without_approval"]
    risky = irreversible or gpu_hours > limit
    if risky and args_sha256(tool, args) not in approvals:
        why = "необратимо" if irreversible else f"{gpu_hours} GPU-ч"
        return Decision("ask", f"нужно одобрение: {why}")
    return Decision("allow", "одобрено" if risky else "в мандате")

USER = {"docs:read:public", "docs:read:finance", "tickets:write",
        "gpu:reserve", "docs:delete"}
SESSION = {"docs:read:public", "tickets:write", "gpu:reserve"}

def proposal(tool, **arguments):
    return {"tool": tool, "arguments": arguments}

CALLS = [
    proposal("search_docs", query="KV-кэш BookLM", k=3, access="public"),
    proposal("search_docs", query="бюджет GPU", k=3, access="finance"),
    proposal("create_ticket", title="KV-кэш 3x512", body="2 359 296 байт"),
    proposal("reserve_gpu", node="h100-a", gpus=8, hours=4),
    proposal("delete_document", doc_id="booklm-contract"),
]
call_40h = proposal("reserve_gpu", node="h100-a", gpus=8, hours=40)
h1 = args_sha256("reserve_gpu", CALLS[3]["arguments"])
h2 = args_sha256("reserve_gpu", call_40h["arguments"])
assert h1.startswith("968e24a5") and h2.startswith("2813816f")

steps = [(c, set()) for c in CALLS] + [(CALLS[3], {h1}), (call_40h, {h1})]
kinds = [authorize(USER, SESSION, c, ok).kind for c, ok in steps]
assert kinds == ["allow", "deny", "allow", "ask", "deny", "allow", "ask"]

# Ошибка: объединение вместо пересечения открыло бы финансы (шаг 2)
union = USER | SESSION
assert authorize(union, union, CALLS[1], set()).kind == "allow"
# docs:delete попал в мандат, но удаление всё равно уходит человеку
wide = SESSION | {"docs:delete"}
assert authorize(USER, wide, CALLS[4], set()).kind == "ask"
# Порядок ключей на хеш не влияет
same = {"hours": 4, "gpus": 8, "node": "h100-a"}
assert args_sha256("reserve_gpu", same) == h1

# --- журнал разрешений и исполнение только после allow ---
import copy
from collections import Counter

GENESIS = "0" * 64

def record_hash(body):
    return hashlib.sha256(canonical(body).encode("utf-8")).hexdigest()

class PermissionLog:
    """Только дописываемый журнал; запись хранит хеш предыдущей."""
    def __init__(self):
        self.records = []

    def append(self, record):
        prev = self.records[-1]["sha256"] if self.records else GENESIS
        body = dict(record, seq=len(self.records), prev_sha256=prev)
        self.records.append(dict(body, sha256=record_hash(body)))

    def verify(self):
        prev = GENESIS
        for seq, rec in enumerate(self.records):
            body = {k: v for k, v in rec.items() if k != "sha256"}
            if (rec["seq"] != seq or rec["prev_sha256"] != prev
                    or rec["sha256"] != record_hash(body)):
                return False
            prev = rec["sha256"]
        return True

executed = Counter()          # сколько раз реально вызвана функция

def stub(name):               # заглушка; полная версия в разделе 13.1
    def fn(**args):
        executed[name] += 1
        return {"ok": True}
    return fn

FUNCTIONS = {name: stub(name) for name in POLICY["right"]}

def approvals_from(log):      # одобрение одноразовое: action его гасит
    open_approvals = set()
    for r in log.records:
        if r["type"] == "approval":
            open_approvals.add(r["args_sha256"])
        elif r["type"] == "action":
            open_approvals.discard(r["args_sha256"])
    return open_approvals

def guarded_execute(log, run_id, step, call):
    tool, args = call["tool"], call["arguments"]
    d = authorize(USER, SESSION, call, approvals_from(log))
    entry = {"run_id": run_id, "step": step, "tool": tool,
             "args_sha256": args_sha256(tool, args),
             "policy_version": POLICY["version"]}
    log.append(dict(entry, type="decision", decision=d.kind,
                    reason=d.reason))
    if d.kind != "allow":     # функция даже не вызывается
        code = "denied" if d.kind == "deny" else "needs_approval"
        return {"ok": False, "tool": tool,
                "error": {"code": code, "detail": d.reason}}
    result = FUNCTIONS[tool](**args)
    log.append(dict(entry, type="action", decision=None,
                    reason="исполнено"))
    return {"ok": True, "tool": tool, "result": result}

def actions_without_allow(records):  # проверка по готовому журналу
    allowed, bad = set(), 0
    for r in records:
        if r["type"] == "decision" and r["decision"] == "allow":
            allowed.add(r["args_sha256"])
        elif r["type"] == "action" and r["args_sha256"] not in allowed:
            bad += 1
    return bad

def code(obs):
    return obs["error"]["code"] if not obs["ok"] else "ok"

log = PermissionLog()
obs = [guarded_execute(log, "run-1", i, c)
       for i, c in enumerate(CALLS, start=1)]
assert [code(o) for o in obs] == [
    "ok", "denied", "ok", "needs_approval", "denied"]
assert executed["delete_document"] == executed["reserve_gpu"] == 0

log.append({"run_id": "run-1", "step": 4, "type": "approval",
            "tool": "reserve_gpu", "args_sha256": h1,
            "decision": "allow", "reason": "оператор: да, 4 ч",
            "policy_version": POLICY["version"]})

def step(i, c):
    return code(guarded_execute(log, "run-1", i, c))

assert step(6, CALLS[3]) == "ok"                # одобренная бронь на 4 ч
assert step(7, call_40h) == "needs_approval"    # 40 ч: другой хеш
assert step(8, CALLS[3]) == "needs_approval"    # одобрение уже погашено
assert executed["reserve_gpu"] == 1
assert log.verify() and actions_without_allow(log.records) == 0

forged = copy.deepcopy(log)   # задним числом: запрет стал разрешением
assert forged.records[2]["decision"] == "deny"
forged.records[2]["decision"] = "allow"
assert not forged.verify()

anchor = log.records[-1]["sha256"]  # копия последнего хеша вне журнала
rebuilt = PermissionLog()           # правщик пересчитал все хеши
for r in forged.records:
    rebuilt.append({k: v for k, v in r.items()
                    if k not in ("seq", "prev_sha256", "sha256")})
assert rebuilt.verify() and rebuilt.records[-1]["sha256"] != anchor


if __name__ == "__main__":
    print("OK: permission_log.py")
