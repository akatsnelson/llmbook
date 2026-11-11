"""Листинги главы 13 «Агенты и инструменты» книги «Большие языковые модели», файл 03-authorization.

Код и вывод — ровно как в книге; текст книги сюда не входит. Листинги идут по порядку и
опираются друг на друга: запускайте файл целиком или ячейками сверху вниз (разметка # %%).
Проверка, что вывод совпал с книгой: python scripts/check_listings.py listings/ch13/03-authorization.py
"""

# %% Листинг «Хеш вызова» — раздел «Подтверждение по хешу аргументов»
import copy
import hashlib
import sys
sys.path.insert(0, "labs/ch13")      # путь от корня репозитория
from agent import (WORLD, reset_world, fail, execute, canonical, Guard,
                   run_agent, scripted)

def args_sha256(tool, args):         # хеш пары (инструмент, аргументы)
    data = canonical({"tool": tool, "arguments": args}).encode("utf-8")
    return hashlib.sha256(data).hexdigest()

h1 = args_sha256("reserve_gpu", {"node": "h100-a", "gpus": 8, "hours": 4})
h2 = args_sha256("reserve_gpu", {"node": "h100-a", "gpus": 8, "hours": 40})
print("4 ч ", len(h1), h1[:16])
print("40 ч", len(h2), h2[:16])
same = {"hours": 4, "gpus": 8, "node": "h100-a"}   # другой порядок ключей
print(args_sha256("reserve_gpu", same) == h1)
# Вывод в книге:
# | 4 ч  64 968e24a50bd08d9e
# | 40 ч 64 2813816f215e0663
# | True

# %% Листинг «Политика authz-v1» — раздел «Применение политики к пяти вызовам»
POLICY = {                           # политика — данные с версией
    "version": "authz-v1",
    "right": {"search_docs": "docs:read:{access}",
              "kv_cache_bytes": None,      # чистый расчёт: права не нужно
              "create_ticket": "tickets:write",
              "reserve_gpu": "gpu:reserve",
              "delete_document": "docs:delete"},
    "irreversible": {"delete_document"},
    "gpu_hours_without_approval": 8}     # сумма на весь прогон

def authorize(user_rights, session_scope, prop, approvals, spent=0,
              policy=POLICY):            # spent: GPU-часы прогона до этого
    tool, args = prop["tool"], prop["arguments"]   # уже прошли схему
    if tool not in policy["right"]:
        return "deny", "нет правила: запрет по умолчанию"
    template = policy["right"][tool]
    right = template.format(**args) if template else None
    if right is not None and right not in user_rights & session_scope:
        owner = "сессии" if right in user_rights else "пользователя"
        return "deny", f"{right}: нет у {owner}"
    gpu_hours = args["gpus"] * args["hours"] if tool == "reserve_gpu" else 0
    limit = policy["gpu_hours_without_approval"]
    if tool in policy["irreversible"]:
        why = "необратимое действие"
    elif gpu_hours and spent + gpu_hours > limit:
        why = f"GPU-часов за прогон {spent}+{gpu_hours} > {limit}"
    else:
        return "allow", "в мандате"
    if args_sha256(tool, args) not in approvals:
        return "ask", why
    return "allow", "одобрено"

# %% Листинг «Решения политики» — раздел «Применение политики к пяти вызовам»
USER = {"docs:read:public", "docs:read:finance", "tickets:write",
        "gpu:reserve", "docs:delete"}
SESSION = {"docs:read:public", "tickets:write", "gpu:reserve"}

def proposal(tool, **arguments):
    return {"tool": tool, "arguments": arguments}

public = proposal("search_docs", query="KV-кэш", k=3, access="public")
finance = proposal("search_docs", query="бюджет GPU", k=3, access="finance")
ticket = proposal("create_ticket", title="KV-кэш 3x512",
                  body="2 359 296 байт")
gpu4 = proposal("reserve_gpu", node="h100-a", gpus=8, hours=4)
delete = proposal("delete_document", doc_id="booklm-md-v3")
gpu40 = proposal("reserve_gpu", node="h100-a", gpus=8, hours=40)

steps = [(p, set()) for p in [public, finance, ticket, gpu4, delete]]
steps += [(gpu4, {h1}), (gpu40, {h1})]     # человек одобрил только h1
for p, approvals in steps:
    print(p["tool"], *authorize(USER, SESSION, p, approvals))
union = USER | SESSION               # ошибка: объединение
print("объединение, финансы:", *authorize(union, union, finance, set()))
wide = SESSION | {"docs:delete"}
print("удаление в мандате:", *authorize(USER, wide, delete, set()))
small = proposal("reserve_gpu", node="h100-a", gpus=1, hours=8)
print("вторая бронь 1x8:", *authorize(USER, SESSION, small, set(), spent=8))
# Вывод в книге:
# | search_docs allow в мандате
# | search_docs deny docs:read:finance: нет у сессии
# | create_ticket allow в мандате
# | reserve_gpu ask GPU-часов за прогон 0+32 > 8
# | delete_document deny docs:delete: нет у сессии
# | reserve_gpu allow одобрено
# | reserve_gpu ask GPU-часов за прогон 0+320 > 8
# | объединение, финансы: allow в мандате
# | удаление в мандате: ask необратимое действие
# | вторая бронь 1x8: ask GPU-часов за прогон 8+8 > 8

# %% Листинг «Журнал с хеш-цепочкой» — раздел «Журнал разрешений»
GENESIS = "0" * 64

def record_hash(body):
    return hashlib.sha256(canonical(body).encode("utf-8")).hexdigest()

class PermissionLog:                 # только дописываемый журнал
    def __init__(self):
        self.records = []

    def append(self, record):        # запись хранит хеш предыдущей
        prev = self.records[-1]["sha256"] if self.records else GENESIS
        body = dict(record, seq=len(self.records), prev_sha256=prev)
        self.records.append(dict(body, sha256=record_hash(body)))

    def verify(self):                # первая порванная ссылка — False
        prev = GENESIS
        for seq, rec in enumerate(self.records):
            body = {k: v for k, v in rec.items() if k != "sha256"}
            if (rec["seq"] != seq or rec["prev_sha256"] != prev
                    or rec["sha256"] != record_hash(body)):
                return False
            prev = rec["sha256"]
        return True

# %% Листинг «Подделка журнала и якорь» — раздел «Журнал разрешений»
log = PermissionLog()
for step, decision in [(1, "allow"), (2, "deny"), (3, "allow")]:
    log.append({"step": step, "type": "decision", "decision": decision})
for r in log.records:
    print(r["seq"], r["decision"], r["prev_sha256"][:8], "->",
          r["sha256"][:8])
anchor = log.records[-1]["sha256"]   # копия последнего хеша вне журнала

forged = copy.deepcopy(log)          # задним числом: deny стал allow
forged.records[1]["decision"] = "allow"
print("правка без пересчёта, цепочка цела:", forged.verify())

rebuilt = PermissionLog()            # правщик пересчитал все хеши
for r in forged.records:
    rebuilt.append({k: r[k] for k in ("step", "type", "decision")})
print("правка с пересчётом, цепочка цела:", rebuilt.verify())
print("якорь совпал:", rebuilt.records[-1]["sha256"] == anchor)
# Вывод в книге:
# | 0 allow 00000000 -> aa97a92c
# | 1 deny aa97a92c -> 61503626
# | 2 allow 61503626 -> 96b0790f
# | правка без пересчёта, цепочка цела: False
# | правка с пересчётом, цепочка цела: True
# | якорь совпал: False

# %% Листинг «Страж с политикой и журналом» — раздел «Страж в цикле агента»
def open_approvals(log, run_id):     # одобрения этого прогона
    hashes = set()
    for r in log.records:
        if r["run_id"] != run_id:
            continue                     # чужой прогон не в счёт
        if r["type"] == "approval":
            hashes.add(r["args_sha256"])
        elif r["type"] == "action":      # исполнение гасит одобрение
            hashes.discard(r["args_sha256"])
    return hashes

def gpu_hours_spent(log, run_id):    # сумма исполненных броней прогона
    return sum(r.get("gpu_hours", 0) for r in log.records
               if r["run_id"] == run_id and r["type"] == "action")

class PolicyGuard(Guard):
    def __init__(self, user, session, log, run_id="run-1"):
        self.user, self.session = user, session
        self.log, self.run_id = log, run_id

    def note(self, prop, step, kind, decision):
        tool, args = prop["tool"], prop["arguments"]
        record = {"run_id": self.run_id, "step": step, "type": kind,
                  "tool": tool, "decision": decision,
                  "args_sha256": args_sha256(tool, args),
                  "policy_version": POLICY["version"]}
        if kind == "action" and tool == "reserve_gpu":   # для бюджета
            record["gpu_hours"] = args["gpus"] * args["hours"]
        self.log.append(record)

    def decide(self, prop, step):
        decision, why = authorize(
            self.user, self.session, prop,
            open_approvals(self.log, self.run_id),
            gpu_hours_spent(self.log, self.run_id))
        self.note(prop, step, "decision", decision)   # до действия
        if decision == "allow":
            return None
        return fail("denied" if decision == "deny" else "needs_approval",
                    why)

    def act(self, prop, step):
        obs = execute(prop)
        self.note(prop, step, "action", None)
        return obs

# %% Листинг «Сквозная задача со стражем» — раздел «Страж в цикле агента»
reset_world()
log = PermissionLog()
state, traj = run_agent(scripted, guard=PolicyGuard(USER, SESSION, log))
print(state, "заявок:", len(WORLD["tickets"]), "журнал цел:", log.verify())
for r in log.records:
    print(r["seq"], r["step"], r["type"], r["tool"], r["decision"])
# Вывод в книге:
# | DONE заявок: 1 журнал цел: True
# | 0 1 decision search_docs allow
# | 1 1 action search_docs None
# | 2 2 decision kv_cache_bytes allow
# | 3 2 action kv_cache_bytes None
# | 4 3 decision create_ticket allow
# | 5 3 action create_ticket None

# %% Листинг «Бронь на 4 и на 40 часов» — раздел «Страж в цикле агента»
reset_world()
log = PermissionLog()
guard = PolicyGuard(USER, SESSION, log)

def attempt(prop, step):             # AUTHORIZE и ACT одного шага
    return guard.decide(prop, step) or guard.act(prop, step)

def approve(h, step):                # подпись оператора в run-1
    log.append({"run_id": "run-1", "step": step, "type": "approval",
                "tool": "reserve_gpu", "decision": "allow",
                "approver": "ops:ira", "args_sha256": h,
                "policy_version": POLICY["version"]})

print(4, attempt(gpu4, 4)["error"]["code"])
approve(h1, 5)
print(6, attempt(gpu4, 6)["result"])
print(7, attempt(gpu40, 7)["error"]["code"])
print(8, attempt(gpu4, 8)["error"]["code"])
for r in log.records:
    print(r["seq"], r["step"], r["type"], r["decision"],
          r["args_sha256"][:8])
print("броней:", len(WORLD["reservations"]), "журнал цел:", log.verify())
approve(h2, 9)                       # 40 ч: подпись пока не погашена
other = PolicyGuard(USER, SESSION, log, run_id="run-2")
print("открыто в run-1:", len(open_approvals(log, "run-1")),
      " run-2:", other.decide(gpu40, 1)["error"]["code"])
# Вывод в книге:
# | 4 needs_approval
# | 6 {'reservation_id': 'R-1'}
# | 7 needs_approval
# | 8 needs_approval
# | 0 4 decision ask 968e24a5
# | 1 5 approval allow 968e24a5
# | 2 6 decision allow 968e24a5
# | 3 6 action None 968e24a5
# | 4 7 decision ask 2813816f
# | 5 8 decision ask 968e24a5
# | броней: 1 журнал цел: True
# | открыто в run-1: 1  run-2: needs_approval
