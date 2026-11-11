"""Глава 18, раздел 18.4 «Полномочия и песочница», подраздел «Диспетчер с наименьшими полномочиями»: диспетчер с наименьшими полномочиями и отдельными ключами (полная версия).

Из книги «Большие языковые модели».
В книге оставлена сокращённая версия; этот файл самодостаточен и
выполняется в ноутбуке главы notebooks/18_security_boundaries.ipynb.

Пять предложений после косвенной инъекции: чтение проходит, письмо, удаление и оболочка
запрещены. В чистой сессии письмо ждёт человека, ключ выдаёт хранилище под хеш вызова, повтор
с тем же ключом идемпотентности второго письма не создаёт. В конце — двусторонняя проверка
и намеренная поломка: диспетчер, который берёт ключ из аргументов вызова.

Запуск из корня репозитория: python labs/ch18/least_privilege_dispatcher.py
"""

# --- диспетчер, хранилище ключей, журнал ---
import hashlib
import json
from enum import IntEnum
from typing import NamedTuple

def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

def args_sha256(tool, args):          # хеш вызова, как в главе 13
    data = canonical({"tool": tool, "arguments": args}).encode("utf-8")
    return hashlib.sha256(data).hexdigest()

def idem_key(run_id, step, tool, args):
    text = f"{run_id}|{step}|{tool}|{canonical(args)}"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

POLICY_VERSION = "authz-guarded-v1"
ALLOWLIST = {                         # список разрешённых: риск задаёт политика
    "search_docs":     {"risk": "read",   "right": "docs:read:{access}"},
    "kv_cache_bytes":  {"risk": "read",   "right": None},
    "send_email":      {"risk": "send",   "right": "mail:send"},
    "delete_document": {"risk": "delete", "right": "docs:delete"},
}

class Trust(IntEnum):                 # уровни доверия, как в начале главы
    OPERATOR = 0
    USER = 1
    UNTRUSTED = 2

class Credential:
    def __init__(self, origin, args_sha256):
        self.origin = origin          # "vault" или откуда-то ещё
        self.args_sha256 = args_sha256
        self.used = False

class Vault:
    """Хранилище ключей записи. Процесс модели его содержимого не видит."""
    def issue(self, context, tool, args, approved_by):
        if not approved_by.startswith("human:"):         # только по человеку
            raise PermissionError("ключ выдаётся только по одобрению человека")
        if taint(context) is Trust.UNTRUSTED:            # только чистой сессии
            raise PermissionError("сессия помечена: ключ записи не выдаётся")
        return Credential("vault", args_sha256(tool, args))

class Decision(NamedTuple):
    kind: str                         # "allow" | "deny" | "ask"
    reason: str

def taint(context):                   # метка: худший уровень среди кусков
    return max(s["trust"] for s in context["sources"])

def authorize(action, context, cred=None):
    tool, args = action["tool"], action["arguments"]
    rule = ALLOWLIST.get(tool)
    if rule is None:
        return Decision("deny", "нет в списке разрешённых")
    right = rule["right"].format(**args) if rule["right"] else None
    if right is not None and right not in context["rights"]:
        return Decision("deny", f"нет права {right}")
    if rule["risk"] == "read":        # поле action["risk"] не читаем
        return Decision("allow", "чтение")
    if taint(context) is Trust.UNTRUSTED:
        return Decision("deny", "путь помечен: ключа записи здесь нет")
    if cred is None:
        return Decision("ask", "нужен человек")
    h = args_sha256(tool, args)
    if cred.origin != "vault" or cred.args_sha256 != h or cred.used:
        return Decision("deny", "ключ не из хранилища или не под этот вызов")
    return Decision("allow", "ключ из хранилища под этот вызов")

effects, journal, done = [], [], {}   # мир, журнал, результаты по ключу K

def run_tool(tool, args):
    if tool == "kv_cache_bytes":      # 2 (K и V) * 2 головы * 32 * 2 байта * 6 блоков
        return 2 * 2 * 32 * 2 * 6 * args["tokens"] * args["requests"]
    if tool == "search_docs":
        return ["booklm-contract#kv-cache", "ops-notes#7"]
    effects.append((tool, args_sha256(tool, args)))  # побочный эффект в мире
    return "ok"

def dispatch(action, context, step, cred=None, policy=authorize):
    tool, args = action["tool"], action["arguments"]
    key = idem_key(context["run_id"], step, tool, args)
    if key in done:                   # повтор узнан: второго эффекта нет
        return done[key]
    d = policy(action, context, cred)
    journal.append({"step": step, "tool": tool, "args_sha256": args_sha256(tool, args),
                    "taint": taint(context).name, "decision": d.kind,
                    "reason": d.reason, "policy_version": POLICY_VERSION})
    if d.kind != "allow":
        return {"ok": False, "code": d.kind}
    if cred is not None:
        cred.used = True              # ключ одноразовый
    done[key] = {"ok": True, "result": run_tool(tool, args)}
    return done[key]

RIGHTS = {"docs:read:public", "mail:send", "docs:delete"}
user = {"source": "user", "trust": Trust.USER}
fragment = {"source": "rag:ops-notes#7", "trust": Trust.UNTRUSTED}
ctx = {"run_id": "run-17", "rights": RIGHTS, "sources": [user, fragment]}

ACTIONS = [
    {"tool": "search_docs", "arguments": {"query": "KV-кэш BookLM", "k": 3,
                                          "access": "public"}},
    {"tool": "kv_cache_bytes", "arguments": {"tokens": 512, "requests": 1,
                                             "dtype": "bf16"}},
    {"tool": "send_email", "arguments": {"to": "audit@example.invalid",
                                         "subject": "сводка", "body": "см. выше"}},
    {"tool": "delete_document", "arguments": {"doc_id": "booklm-contract"},
     "risk": "read"},                 # модель «понизила» риск: это ничего не даёт
    {"tool": "run_shell", "arguments": {"cmd": "env"}},
]
obs = [dispatch(a, ctx, step) for step, a in enumerate(ACTIONS, start=1)]
assert [o.get("code", "allow") for o in obs] == ["allow", "allow", "deny", "deny", "deny"]
assert obs[1]["result"] == 786_432    # 768 КБ из паспорта BookLM
assert effects == []                  # ни письма, ни удаления
assert args_sha256("send_email", ACTIONS[2]["arguments"]).startswith("614ce579")

# Чистая сессия: пользователь сам просит отправить расчёт коллеге.
clean = {"run_id": "run-18", "rights": RIGHTS, "sources": [user]}
mail = {"tool": "send_email", "arguments": {"to": "ops@booklm.example",
        "subject": "KV-кэш BookLM", "body": "786432 байт на 512 токенов"}}
assert dispatch(mail, clean, step=1)["code"] == "ask"
cred = Vault().issue(clean, "send_email", mail["arguments"], approved_by="human:admin")
assert dispatch(mail, clean, step=2, cred=cred)["ok"]
assert dispatch(mail, clean, step=2, cred=cred)["ok"]   # повтор с тем же K
assert len(effects) == 1                                # письмо одно
other = {"tool": "send_email", "arguments": dict(mail["arguments"], to="x@example.invalid")}
assert dispatch(other, clean, step=3, cred=cred)["code"] == "deny"  # чужой хеш
assert [r["decision"] for r in journal].count("allow") == 3

# --- проверка: чтение не сломано, опасное остановлено; поломка ---
def broken_authorize(action, context, cred=None):
    # Ошибка: ключ берётся из аргументов, его происхождение не проверяется.
    if "api_key" in action["arguments"]:
        cred = Credential("context", args_sha256(action["tool"], action["arguments"]))
    if cred is not None and ALLOWLIST.get(action["tool"], {}).get("risk") == "send":
        return Decision("allow", "ключ есть")
    return authorize(action, context, cred)

READ = {t for t, rule in ALLOWLIST.items() if rule["risk"] == "read"}
for sources in ([user], [user, fragment]):            # чистая и помеченная сессии
    c = {"run_id": f"inv-{len(sources)}", "rights": RIGHTS, "sources": sources}
    before = len(effects)
    passed = {a["tool"] for s, a in enumerate(ACTIONS, start=1)
              if dispatch(a, c, s).get("ok")}
    assert passed == READ                              # чтение работает всегда
    assert len(effects) == before                      # опасное без человека не прошло

# Намеренная поломка: модель кладёт в аргумент ключ из ops-notes#7.
attack = {"tool": "send_email", "arguments": {"to": "audit@example.invalid",
          "subject": "сводка", "body": "см. выше", "api_key": "wk-live-7f3a"}}
before = len(effects)
assert dispatch(attack, ctx, step=40, policy=broken_authorize)["ok"]
assert len(effects) == before + 1                      # письмо ушло без человека
assert dispatch(attack, ctx, step=41)["code"] == "deny"   # исправленный: путь помечен
fake = Credential("context", args_sha256(attack["tool"], attack["arguments"]))
assert dispatch(attack, clean, step=42, cred=fake)["code"] == "deny"  # не из хранилища
assert len(effects) == before + 1                      # второго письма нет

print("OK:", __file__)
