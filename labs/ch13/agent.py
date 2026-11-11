"""Глава 13: агент, который растёт от раздела к разделу.

Это один модуль на всю главу. Код в нём совпадает с листингами книги:

  13.1  validate, ToolSpec, реестр TOOLS, fail, parse, check, execute, dispatch;
  13.2  canonical, ALLOWED, final_answer, Guard, run_agent, steps_that_fit;
  13.3  args_sha256, POLICY, authorize, PermissionLog, open_approvals,
        gpu_hours_spent, PolicyGuard;
  13.4  idem_key, KeyedServer, FlakyNet, call_with_retry, Crash, run_step,
        recover, IdempotentGuard;
  13.5  REF, validate_plan, plan_step, RUNBOOKS, search_runbooks, run_plan_step,
        execute_plan;
  13.6  AgentRun, handoff, to_mcp_tool, passport_sha, host_authorize;
  13.7  logged_call, ticket_ids, evaluate, pass_hat_k, pass_at_k.

Листинги разделов 13.2–13.7 начинают с `from agent import ...`, поэтому каталог
labs/ch13 должен быть в пути поиска модулей (листинг книги добавляет его сам, путь
считается от корня репозитория). В книге показан только паспорт kv_cache_bytes; паспорта
остальных четырёх инструментов набора booklm-tools-v1 (таблица 13.1) устроены так же и
записаны здесь, вместе с «внешним миром» WORLD, куда они пишут.

Файл при запуске проверяет утверждениями всё, что книга печатает.

Запуск из корня репозитория: python labs/ch13/agent.py
"""

import hashlib
import json
import re
from collections import Counter
from fractions import Fraction
from math import comb
from typing import NamedTuple

# =============================================================================
# 13.1. Инструмент, схема, диспетчер
# =============================================================================

KV_INPUT = {"type": "object", "additionalProperties": False,
            "required": ["tokens", "requests", "dtype"], "properties": {
    "tokens":   {"type": "integer", "minimum": 1, "maximum": 512},
    "requests": {"type": "integer", "minimum": 1, "maximum": 64},
    "dtype":    {"type": "string", "enum": ["bf16", "fp32"]}}}
TYPES = {"object": dict, "string": str, "integer": int}


def validate(value, schema, path="$"):
    """Подмножество JSON Schema. Пустой список — форма верна."""
    kind = schema.get("type")
    if kind and type(value) is not TYPES[kind]:   # True — не int
        return [f"{path}: type: ожидался {kind}"]
    errors = []
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: enum")
    if "minimum" in schema and value < schema["minimum"]:
        errors.append(f"{path}: minimum {schema['minimum']}")
    if "maximum" in schema and value > schema["maximum"]:
        errors.append(f"{path}: maximum {schema['maximum']}")
    if "maxLength" in schema and len(value) > schema["maxLength"]:
        errors.append(f"{path}: maxLength {schema['maxLength']}")
    if kind == "object":
        for key in schema.get("required", []):
            if key not in value:
                errors.append(f"{path}.{key}: required")
        for key, item in value.items():
            if key in schema.get("properties", {}):
                errors += validate(item, schema["properties"][key],
                                   f"{path}.{key}")
            elif schema.get("additionalProperties") is False:
                errors.append(f"{path}.{key}: additionalProperties")
    return errors


class ToolSpec(NamedTuple):
    name: str
    description: str     # читает модель, программа на него не опирается
    input_schema: dict
    output_schema: dict
    effect: str          # "read" | "reversible" | "irreversible"
    fn: object           # сама функция инструмента


calls = []               # журнал настоящих вызовов функций


def kv_cache_bytes(tokens, requests, dtype):
    calls.append("kv_cache_bytes")
    size = {"bf16": 2, "fp32": 4}[dtype]
    per_token = 2 * 6 * 2 * 32 * size   # 2·L·H_kv·d_h·b, формула гл. 4
    return {"bytes": per_token * tokens * requests}


KV_OUTPUT = {"type": "object", "required": ["bytes"],
             "additionalProperties": False,
             "properties": {"bytes": {"type": "integer", "minimum": 0}}}
TOOLS = {"kv_cache_bytes": ToolSpec(
    "kv_cache_bytes", "Байты KV-кэша BookLM.",
    KV_INPUT, KV_OUTPUT, "read", kv_cache_bytes)}


def fail(code, detail):
    return {"ok": False, "error": {"code": code, "detail": detail}}


def no_repeats(pairs):              # «Внимательно» в 13.1: повтор ключа
    keys = [key for key, _ in pairs]
    if len(keys) != len(set(keys)):
        raise ValueError("повтор ключа")
    return dict(pairs)


def parse(raw):                     # 1. ровно один объект JSON
    try:
        prop = json.loads(raw, object_pairs_hook=no_repeats)
    except json.JSONDecodeError as e:
        return None, fail("parse_error", e.msg)
    except ValueError as e:         # повтор ключа из no_repeats
        return None, fail("parse_error", str(e))
    if not isinstance(prop, dict) or set(prop) != {"tool", "arguments"}:
        return None, fail("parse_error", "ожидался {tool, arguments}")
    return prop, None


def check(prop):                    # 2. имя в реестре, 3. схема входа
    name = prop["tool"]
    spec = TOOLS.get(name) if isinstance(name, str) else None
    if spec is None:
        return fail("unknown_tool", repr(name))
    errors = validate(prop["arguments"], spec.input_schema)
    return fail("schema_violation", "; ".join(errors)) if errors else None


def execute(prop):                  # 4. функция, 5. схема выхода
    spec = TOOLS[prop["tool"]]
    try:
        result = spec.fn(**prop["arguments"])
    except Exception as e:
        return fail("tool_error", repr(e))
    errors = validate(result, spec.output_schema)
    if errors:
        return fail("tool_error", "выход: " + "; ".join(errors))
    return {"ok": True, "tool": prop["tool"], "result": result}


def dispatch(raw):
    prop, error = parse(raw)
    return error or check(prop) or execute(prop)


# --- Остальные четыре инструмента booklm-tools-v1 и внешний мир -------------

# Фрагмент о KV-кэше из учебного корпуса главы 12 (имя то же, что там:
# документ BOOKLM.md, версия v3, фрагмент 42); заглушка пересказывает его на токен.
CHUNK = "booklm-md-v3-0042"
WORLD = {"tickets": [], "reservations": [], "deleted": []}


def reset_world():
    calls.clear()
    for items in WORLD.values():
        items.clear()


def search_docs(query, k, access):  # заглушка: query и access не читает
    calls.append("search_docs")
    found = [{"chunk_id": CHUNK, "text": "1536 байт на токен"}]
    return {"chunks": found[:k]}


def create_ticket(title, body):
    calls.append("create_ticket")
    WORLD["tickets"].append({"title": title, "body": body})
    return {"ticket_id": f"T-{16 + len(WORLD['tickets'])}"}


def reserve_gpu(node, gpus, hours):
    calls.append("reserve_gpu")
    WORLD["reservations"].append({"node": node, "gpus": gpus,
                                  "hours": hours})
    return {"reservation_id": f"R-{len(WORLD['reservations'])}"}


def delete_document(doc_id):
    calls.append("delete_document")
    WORLD["deleted"].append(doc_id)
    return {"deleted": doc_id}


def _obj(required, **props):
    return {"type": "object", "additionalProperties": False,
            "required": list(required), "properties": props}


TOOLS.update({
    "search_docs": ToolSpec(
        "search_docs", "Поиск фрагментов документации BookLM.",
        _obj(["query", "k", "access"],
             query={"type": "string", "maxLength": 200},
             k={"type": "integer", "minimum": 1, "maximum": 5},
             access={"type": "string", "enum": ["public", "finance"]}),
        _obj(["chunks"], chunks={}), "read", search_docs),
    "create_ticket": ToolSpec(
        "create_ticket", "Завести заявку в трекере.",
        _obj(["title", "body"],
             title={"type": "string", "maxLength": 80},
             body={"type": "string", "maxLength": 500}),
        _obj(["ticket_id"], ticket_id={"type": "string"}),
        "reversible", create_ticket),
    "reserve_gpu": ToolSpec(
        "reserve_gpu", "Забронировать GPU на узле.",
        _obj(["node", "gpus", "hours"],
             node={"type": "string", "maxLength": 40},
             gpus={"type": "integer", "minimum": 1, "maximum": 8},
             hours={"type": "integer", "minimum": 1, "maximum": 72}),
        _obj(["reservation_id"], reservation_id={"type": "string"}),
        "reversible", reserve_gpu),
    "delete_document": ToolSpec(
        "delete_document", "Удалить документ из корпуса.",
        _obj(["doc_id"], doc_id={"type": "string", "maxLength": 80}),
        _obj(["deleted"], deleted={"type": "string"}),
        "irreversible", delete_document),
})

# =============================================================================
# 13.2. Цикл агента как конечный автомат
# =============================================================================


def steps_that_fit(t_max=512, rules=55, tools=(44, 56, 31), question=62,
                   reserve=56, step=165):   # токены BookLM, раздел 13.2
    fixed = rules + sum(tools) + question + reserve
    return max(0, (t_max - fixed) // step)


def canonical(obj):                  # канонический JSON главы
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)


ALLOWED = {                          # разрешённые переходы автомата
    ("PROPOSE", "PARSE"),
    ("PARSE", "VALIDATE"), ("PARSE", "DONE"), ("PARSE", "OBSERVE"),
    ("VALIDATE", "AUTHORIZE"), ("VALIDATE", "OBSERVE"),
    ("VALIDATE", "FAILED"),
    ("AUTHORIZE", "ACT"), ("AUTHORIZE", "OBSERVE"),
    ("AUTHORIZE", "WAIT_HUMAN"),
    ("WAIT_HUMAN", "AUTHORIZE"), ("WAIT_HUMAN", "OBSERVE"),
    ("ACT", "OBSERVE"), ("OBSERVE", "PROPOSE"), ("OBSERVE", "BUDGET")}


def final_answer(raw):               # {"final": "..."} — ответ, не вызов
    try:
        msg = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if isinstance(msg, dict) and set(msg) == {"final"}:
        return msg["final"]
    return None


class Guard:                         # страж AUTHORIZE и ACT: пускает всё
    def decide(self, prop, step):    # None — можно, иначе наблюдение-отказ
        return None

    def act(self, prop, step):
        return execute(prop)


def run_agent(model, n_max=4, repeat_limit=2, guard=Guard()):
    traj, context, state, last, same = [], [], "PROPOSE", None, 0

    def go(new, **info):             # единственный способ сменить состояние
        nonlocal state
        assert (state, new) in ALLOWED, (state, new)
        traj.append({"step": step, "to": new, **info})
        state = new

    for step in range(1, n_max + 1):
        if step > 1:
            go("PROPOSE")
        raw = model(context)         # модель работает только здесь
        go("PARSE")
        answer = final_answer(raw)
        if answer is not None:
            go("DONE", answer=answer)
            return state, traj
        prop, obs = parse(raw)
        if prop is None:
            last, same = None, 0     # битый текст прерывает серию
        else:
            go("VALIDATE")
            key = canonical(prop)
            same, last = (same + 1 if key == last else 1), key
            if same >= repeat_limit:
                go("FAILED", reason="повтор")
                return state, traj
            obs = check(prop)
            if obs is None:
                go("AUTHORIZE")
                obs = guard.decide(prop, step)
                if obs is None:
                    go("ACT", tool=prop["tool"])
                    obs = guard.act(prop, step)
        code = "ok" if obs["ok"] else obs["error"]["code"]
        go("OBSERVE", code=code)
        context.append((raw, obs))
    go("BUDGET")
    return state, traj


def show(traj):                      # одна строка на шаг траектории
    for step in sorted({t["step"] for t in traj}):
        moves = [t for t in traj if t["step"] == step]
        states = " ".join(t["to"] for t in moves if t["to"] != "PROPOSE")
        tool = next((t["tool"] for t in moves if "tool" in t), "-")
        last = [t for t in moves if t["to"] != "BUDGET"][-1]   # итог шага
        print(step, states, "|", tool,
              last.get("code") or last.get("reason") or last.get("answer"))


def call(tool, **arguments):
    return canonical({"tool": tool, "arguments": arguments})


def scripted(context):               # сценарная заглушка: шаг -> ход
    if len(context) == 0:
        return call("search_docs", query="KV-кэш BookLM байт на токен",
                    k=3, access="public")
    if len(context) == 1:
        return call("kv_cache_bytes", tokens=512, requests=3,
                    dtype="bf16")
    chunk = context[0][1]["result"]["chunks"][0]["chunk_id"]
    total = f"{context[1][1]['result']['bytes']:,}".replace(",", " ")
    if len(context) == 2:
        return call("create_ticket", title="KV-кэш 3x512",
                    body=f"{total} байт [S:{chunk}]")
    ticket = context[2][1]["result"]["ticket_id"]
    answer = f"Заявка {ticket}: {total} байт [S:{chunk}]"
    return canonical({"final": answer})


def stubborn(context):               # упрямая: один и тот же вызов
    return call("create_ticket", title="KV-кэш 3x512", body="?")


def sloppy(context):                 # неряшливая: JSON всегда оборван
    return '{"tool": "search_docs", '


def wander(context):                 # блуждающая: каждый раз новый запрос
    return call("search_docs", query=f"попытка {len(context)}", k=3,
                access="public")


# =============================================================================
# 13.3. Авторизация и журнал разрешений
# =============================================================================


def args_sha256(tool, args):         # хеш пары (инструмент, аргументы)
    data = canonical({"tool": tool, "arguments": args}).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


POLICY = {                           # политика — данные с версией
    "version": "authz-v1",
    "right": {"search_docs": "docs:read:{access}",
              "kv_cache_bytes": None,      # чистый расчёт: права не нужно
              "create_ticket": "tickets:write",
              "reserve_gpu": "gpu:reserve",
              "delete_document": "docs:delete"},
    "irreversible": {"delete_document"},
    "gpu_hours_without_approval": 8}     # на весь прогон, а не на вызов


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


USER = {"docs:read:public", "docs:read:finance", "tickets:write",
        "gpu:reserve", "docs:delete"}
SESSION = {"docs:read:public", "tickets:write", "gpu:reserve"}


# =============================================================================
# 13.4. Ключ идемпотентности, журнал намерений, страж с ключом
# =============================================================================

def idem_key(run_id, step, tool):        # имя намерения: прогон, шаг, инструмент
    text = f"{run_id}|{step}|{tool}"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class KeyedServer:                       # внешняя система, помнит ключи
    def __init__(self):
        self.by_key = {}

    def call(self, prop, key=None):
        if key in self.by_key:
            old, obs = self.by_key[key]
            if old != canonical(prop):   # тот же ключ, другие аргументы
                raise ValueError("ключ занят другим намерением")
            return obs                   # повтор: старый результат
        obs = execute(prop)              # настоящий эффект в WORLD
        if key is not None:
            self.by_key[key] = (canonical(prop), obs)
        return obs


class FlakyNet:                          # сеть между агентом и сервером
    def __init__(self, server, fates):
        self.server, self.fates = server, list(fates)

    def call(self, prop, key):
        fate = self.fates.pop(0) if self.fates else "ok"
        if fate == "503":                # сервер занят, запрос не принят
            return fail("503", "сервис временно недоступен")
        obs = self.server.call(prop, key)    # запрос дошёл, эффект есть
        if fate == "timeout":                # а ответ потерялся
            return fail("timeout", "ответа нет")
        return obs


def call_with_retry(server, prop, key, n=3, b=0.5, m=2, t_out=10.0):
    waited = 0.0                         # модельные секунды, без sleep
    for i in range(1, n + 1):
        obs = server.call(prop, key)
        code = "ok" if obs["ok"] else obs["error"]["code"]
        if code not in ("timeout", "503"):
            return obs, waited           # ответ или ошибка, которую повтор не исправит
        waited += t_out if code == "timeout" else 0.0
        if i < n:
            waited += b * m ** (i - 1)   # пауза p_i; джиттер её только укоротит
    return fail("timeout", f"исход неизвестен, ключ {key[:12]}"), waited


class Crash(Exception):                  # процесс агента упал
    pass


def known(obs):                          # исход известен: не тайм-аут
    return obs["ok"] or obs["error"]["code"] != "timeout"


def run_step(log, server, prop, key, crash_at=None):
    if crash_at == "A":
        raise Crash
    log.append({"type": "intent", "key": key, "prop": prop})
    if crash_at == "B":
        raise Crash
    obs, _ = call_with_retry(server, prop, key)  # внешний эффект
    if crash_at == "C":
        raise Crash                              # опасное окно
    if known(obs):                               # тайм-аут оставляет
        log.append({"type": "result", "key": key, "obs": obs})  # намерение открытым
    if crash_at == "D":
        raise Crash
    return obs


def recover(log, server, prop_now, run_id="run-42", step=3):
    if not log:                          # о шаге ничего не известно
        key = idem_key(run_id, step, prop_now["tool"])
        return "шаг заново", run_step(log, server, prop_now, key)
    if log[-1]["type"] == "result":      # шаг сделан: не трогаем
        return "пропуск", log[-1]["obs"]
    intent = log[-1]                     # намерение без результата
    return "повтор с K", run_step(log, server, intent["prop"], intent["key"])


class IdempotentGuard(PolicyGuard):      # страж из 13.3 + ключ и журнал
    def __init__(self, user, session, log, server, intents,
                 run_id="run-42"):
        super().__init__(user, session, log, run_id)
        self.server, self.intents, self.crash_at = server, intents, None

    def act(self, prop, step):
        if TOOLS[prop["tool"]].effect == "read":
            return super().act(prop, step)      # чтение повторять можно
        sent = {r["key"]: r["prop"] for r in self.intents
                if r["type"] == "intent"}
        done = {r["key"]: r["obs"] for r in self.intents
                if r["type"] == "result"}
        key = idem_key(self.run_id, step, prop["tool"])
        hanging = [k for k, p in sent.items()       # открытые намерения
                   if k not in done and p["tool"] == prop["tool"]]
        if hanging:                             # правило recover: сначала
            key = hanging[-1]                   # закрыть старое намерение
        if key in done:                         # шаг уже сделан
            return done[key]
        if key in sent and canonical(sent[key]) != canonical(prop):
            self.note(prop, step, "replaced", None)  # модель сказала иначе
            prop = sent[key]                    # аргументы — из журнала
        obs = run_step(self.intents, self.server, prop, key, self.crash_at)
        if known(obs):
            self.note(prop, step, "action", None)
        return obs


# =============================================================================
# 13.5–13.6. План как данные; передача задачи, паспорт в формате MCP
# =============================================================================

REF = re.compile(r"#(E\d+'?)")                # ссылки #E1, #E2' ...


def validate_plan(plan, max_steps=6):
    ids, seen = {s["id"] for s in plan}, set()
    for s in plan:
        refs = set(REF.findall(json.dumps(s["args"], ensure_ascii=False)))
        for bad, why in ((len(plan) > max_steps, "длиннее бюджета"),
                         (s["tool"] not in TOOLS, "unknown_tool"),
                         (refs - ids, "ссылка в никуда"),
                         (not refs <= seen, "ссылка вперёд или цикл")):
            if bad:
                raise ValueError(f"{s['id']}: {why}")
        seen.add(s["id"])


def plan_step(sid, tool, take=None, **args):   # take: какое поле взять
    s = {"id": sid, "tool": tool, "args": args}
    return {**s, "take": take} if take else s


# Маленький корпус раздела 13.5: фрагмент ops-runbook-v7-0007 из главы 12 и два
# новых — его архивная версия v6 и строка каталога карт. Поля status,
# accelerator и hbm лежат рядом с текстом, как метаданные фрагмента в главе 12.
RUNBOOKS = [
    {"chunk_id": "ops-runbook-v6-0007", "status": "archived",
     "accelerator": "A100-40GB",
     "text": "Архив. Сервис chat-api с моделью BookLM работал на узле "
             "gpu-02 с A100-40GB."},
    {"chunk_id": "ops-runbook-v7-0007", "status": "active",
     "accelerator": "H100",
     "text": "Сервис chat-api с моделью BookLM работает на узле gpu-07 "
             "с H100."},
    {"chunk_id": "hw-catalog-v1-0003", "status": "active", "hbm": "80 ГБ",
     "text": "Карта H100: 80 ГБ памяти HBM."}]


def words(text):
    return set(re.findall(r"\w+", text.lower()))


def search_runbooks(query, k, access):   # фрагмент подходит, если в нём
    calls.append("search_docs")          # есть все слова запроса
    found = [d for d in RUNBOOKS if words(query) <= words(d["text"])]
    return {"chunks": found[:k]}


def fill(args, results):                 # #En -> результат шага En
    return {k: REF.sub(lambda m: str(results[m.group(1)]), v)
            if isinstance(v, str) else v for k, v in args.items()}


def run_plan_step(s, results, guard, step):   # None — шаг провалился
    prop = {"tool": s["tool"], "arguments": fill(s["args"], results)}
    obs = check(prop) or guard.decide(prop, step) or guard.act(prop, step)
    if not obs["ok"]:
        return None
    if "take" not in s:
        return obs["result"]
    live = [c for c in obs["result"]["chunks"] if c["status"] == "active"]
    return live[0][s["take"]] if live else None   # архив не считается


def execute_plan(plan, tail, guard, max_replans=2):
    validate_plan(plan)                  # до первого инструмента
    results, i, replans, step = {}, 0, 0, 0
    while i < len(plan):
        step += 1
        out = run_plan_step(plan[i], results, guard, step)
        if out is not None:
            results[plan[i]["id"]], i = out, i + 1
        elif replans == max_replans:
            return "FAILED", results
        else:                            # переписываем только хвост
            replans += 1
            plan = plan[:i] + tail(results, plan[i])
            validate_plan(plan)          # новый хвост — та же проверка
    return "DONE", results


PUB = dict(k=3, access="public")         # одинаковы у всех поисков
KV1 = dict(tokens=512, requests=1, dtype="bf16")


def blind_plan():                        # модель пишет план вслепую
    return [plan_step("E1", "search_docs", "accelerator",
                      query="chat-api BookLM", **PUB),
            plan_step("E2", "search_docs", "hbm",
                      query="паспорт ускорителя #E1", **PUB),
            plan_step("E3", "kv_cache_bytes", **KV1)]


def tail_plan(results, failed):          # модель пишет новый хвост
    return [plan_step("E2'", "search_docs", "hbm",
                      query="карта #E1 HBM", **PUB),
            plan_step("E3'", "kv_cache_bytes", **KV1)]


class AgentRun:
    def __init__(self, run_id, scope, budget, parent_run_id=None):
        self.run_id, self.scope = run_id, frozenset(scope)
        self.budget, self.parent_run_id = budget, parent_run_id


def handoff(parent, requested_scope, steps, at_step):
    if not 0 < steps <= parent.budget:
        raise ValueError(f"просит {steps} шагов, есть {parent.budget}")
    parent.budget -= steps                   # шаги делятся
    return AgentRun(f"{parent.run_id}/{at_step}",   # номер выводится
                    parent.scope & frozenset(requested_scope), steps,
                    parent_run_id=parent.run_id)


HINTS = {"read": {"readOnlyHint": True},
         "reversible": {"readOnlyHint": False, "destructiveHint": False},
         "irreversible": {"readOnlyHint": False, "destructiveHint": True}}


def to_mcp_tool(spec):
    return {"name": spec.name, "description": spec.description,
            "inputSchema": spec.input_schema,
            "outputSchema": spec.output_schema,
            "annotations": dict(HINTS[spec.effect])}


def passport_sha(tool):                  # отпечаток описания и схемы входа
    text = canonical([tool["description"], tool["inputSchema"]])
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


HOST_DECISION = {"read": "allow", "reversible": "allow",
                 "irreversible": "ask"}
HOST_EFFECT = {                          # хост закрепил сам, при подключении:
    ("docs", name, passport_sha(to_mcp_tool(TOOLS[name]))): TOOLS[name].effect
    for name in ("search_docs", "delete_document")}   # (сервер, имя, отпечаток)


def host_authorize(server, tool):        # аннотации сервера не читает
    effect = HOST_EFFECT.get((server, tool["name"], passport_sha(tool)))
    return "deny" if effect is None else HOST_DECISION[effect]


# =============================================================================
# 13.7. Оценка по журналу и снимку, pass^k и pass@k
# =============================================================================

def logged_call(log, step, tool, args, allowed=True):
    guard = PolicyGuard(USER, SESSION, log)    # пишем его методом note
    prop = {"tool": tool, "arguments": args}
    if allowed:
        guard.note(prop, step, "decision", "allow")   # так пишет decide
    guard.note(prop, step, "action", None)            # так пишет act


def ticket_ids(text):                    # все номера вида T-число
    return set(re.findall(r"T-\d+", text))


def evaluate(records, answer, server, expected):
    hit = True                           # каждая ожидаемая заявка нашлась?
    for e in expected:
        if not any(t["title"] == e["title"] and e["body_has"] in t["body"]
                   for t in server):
            hit = False
    allowed, without_allow = Counter(), 0
    for r in records:                    # allow раньше action, гасится им
        h = r["args_sha256"]
        if r["type"] == "decision" and r["decision"] == "allow":
            allowed[h] += 1
        elif r["type"] == "action" and allowed[h] > 0:
            allowed[h] -= 1              # одна подпись — одно исполнение
        elif r["type"] == "action":
            without_allow += 1
    on_server = {t["id"] for t in server}
    return {"success_state": int(hit and len(server) == len(expected)),
            "extra_effects": max(0, len(server) - len(expected)),
            "actions_without_allow": without_allow,
            "fabricated_ids": ticket_ids(answer) - on_server}


def pass_hat_k(n, c, k):        # все k попыток из n прогонов успешны
    return Fraction(comb(c, k), comb(n, k))


def pass_at_k(n, c, k):         # хотя бы одна из k попыток успешна
    return 1 - Fraction(comb(n - c, k), comb(n, k))


def _checks_late():                      # разделы 13.4–13.7
    args = {"title": "KV-кэш 3x512",
            "body": "2 359 296 байт [S:booklm-md-v3-0042]"}
    prop = {"tool": "create_ticket", "arguments": args}
    key = idem_key("run-42", 3, "create_ticket")
    reworded = {"tool": "create_ticket",
                "arguments": {**args, "title": "KV-кэш: 3 запроса по 512"}}
    # ключ тот же, аргументы другие: сервер отказывает громко
    reset_world()
    server = KeyedServer()
    server.call(prop, key)
    try:
        server.call(reworded, key)
        raise AssertionError("конфликт ключа прошёл")
    except ValueError:
        pass
    # повторы: тайм-аут после эффекта, 503, затем ответ
    reset_world()
    obs, waited = call_with_retry(FlakyNet(KeyedServer(), ["timeout", "503"]),
                                  prop, key)
    assert obs["result"] == {"ticket_id": "T-17"} and waited == 11.5
    assert len(WORLD["tickets"]) == 1
    reset_world()
    obs, waited = call_with_retry(FlakyNet(KeyedServer(), ["timeout"] * 3),
                                  prop, key)
    assert obs["error"]["code"] == "timeout" and waited == 31.5
    assert len(WORLD["tickets"]) == 1
    # журнал намерений: падение в любой из точек A–D
    for point in "ABCD":
        reset_world()
        log, server = [], KeyedServer()
        try:
            run_step(log, server, prop, key, crash_at=point)
        except Crash:
            pass
        _, obs = recover(log, server, reworded)
        assert len(WORLD["tickets"]) == 1
        assert obs["result"] == {"ticket_id": "T-17"}

    # страж: падение в C, перезапуск с переформулированным заголовком
    def reword(context):
        return scripted(context).replace("KV-кэш 3x512", "KV-кэш: 3 по 512")

    for point in "CD":
        reset_world()
        log, intents = PermissionLog(), []
        guard = IdempotentGuard(USER, SESSION, log, KeyedServer(), intents)
        guard.crash_at = point
        try:
            run_agent(scripted, guard=guard)
            raise AssertionError("падения не было")
        except Crash:
            pass
        guard.crash_at = None
        state, _ = run_agent(reword, guard=guard)
        assert state == "DONE" and len(WORLD["tickets"]) == 1 and log.verify()
        assert WORLD["tickets"][0]["title"] == "KV-кэш 3x512"
    # страж: тайм-аут после эффекта, модель предлагает заявку снова
    reset_world()
    log, intents = PermissionLog(), []
    net = FlakyNet(KeyedServer(), ["timeout"] * 3)
    guard = IdempotentGuard(USER, SESSION, log, net, intents)
    obs = guard.act(prop, 3)
    assert obs["error"]["code"] == "timeout"
    obs = guard.act(reworded, 4)
    assert obs["result"] == {"ticket_id": "T-17"}
    assert len(WORLD["tickets"]) == 1

    # 13.5–13.6: план, исполнитель, передача, MCP
    plan = [plan_step("E1", "search_docs", "accelerator", query="x"),
            plan_step("E2", "search_docs", query="#E1")]
    validate_plan(plan)
    for bad in ([plan_step("E1", "search_docs", query="#E2"),
                 plan_step("E2", "search_docs", query="#E1")],
                [plan_step("E1", "rm_index")]):
        try:
            validate_plan(bad)
            raise AssertionError("плохой план прошёл")
        except ValueError:
            pass
    search = TOOLS["search_docs"]
    TOOLS["search_docs"] = search._replace(fn=search_runbooks)
    try:
        log = PermissionLog()
        status, results = execute_plan(blind_plan(), tail_plan,
                                       PolicyGuard(USER, SESSION, log))
    finally:
        TOOLS["search_docs"] = search
    assert status == "DONE" and results["E1"] == "H100"
    assert results["E2'"] == "80 ГБ" and results["E3'"] == {"bytes": 786432}
    assert [r["type"] for r in log.records] == ["decision", "action"] * 4

    parent = AgentRun("run-42", {"docs:read:public", "tickets:write"}, 12)
    child = handoff(parent, {"docs:read:public", "docs:read:finance"}, 4, 3)
    assert child.scope == {"docs:read:public"} and parent.budget == 8
    assert child.run_id == "run-42/3"
    delete = to_mcp_tool(TOOLS["delete_document"])
    lying = {**delete, "annotations": {"readOnlyHint": True}}
    assert host_authorize("docs", delete) == "ask"
    assert host_authorize("docs", lying) == "ask"
    assert host_authorize("evil", to_mcp_tool(TOOLS["search_docs"])) == "deny"
    changed = {**to_mcp_tool(TOOLS["search_docs"]), "description": "Поиск."}
    assert host_authorize("docs", changed) == "deny"

    # 13.7: оценщик гасит allow исполнением
    t1 = PermissionLog()
    logged_call(t1, 3, "create_ticket", args)
    t4 = PermissionLog()
    logged_call(t4, 3, "create_ticket", args, allowed=False)
    twice = PermissionLog()
    logged_call(twice, 3, "create_ticket", args)
    PolicyGuard(USER, SESSION, twice).note(prop, 3, "action", None)
    server = [{"id": "T-17", **args}]
    expected = [{"title": "KV-кэш 3x512", "body_has": "2 359 296"}]
    assert evaluate(t1.records, "T-17", server, expected)[
        "actions_without_allow"] == 0
    assert evaluate(t4.records, "T-17", server, expected)[
        "actions_without_allow"] == 1
    assert evaluate(twice.records, "T-17", server, expected)[
        "actions_without_allow"] == 1
    assert evaluate(t1.records, "T-17", [], expected)["fabricated_ids"] == {
        "T-17"}
    assert [pass_hat_k(5, 4, k) for k in (1, 2, 3, 5)] == [
        Fraction(4, 5), Fraction(3, 5), Fraction(2, 5), 0]
    assert pass_at_k(5, 4, 2) == 1


# =============================================================================
# Проверки всего, что печатает книга
# =============================================================================

def _checks():
    good = {"tokens": 512, "requests": 3, "dtype": "bf16"}
    assert validate(good, KV_INPUT) == []
    assert validate({**good, "tokens": 4096}, KV_INPUT) == [
        "$.tokens: maximum 512"]
    assert validate({**good, "tokens": True}, KV_INPUT) != []

    # 13.1: функция вызывается после всех проверок
    reset_world()
    ok = ('{"tool": "kv_cache_bytes", "arguments": '
          '{"tokens": 512, "requests": 3, "dtype": "bf16"}}')
    assert dispatch(ok)["result"] == {"bytes": 2_359_296}
    try:                            # сверка с конфигурацией BookLM (booklm.lab)
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from booklm.lab import kv_cache_bytes as lab_kv_cache_bytes
    except ImportError:             # нет torch: сверку пропускаем
        lab_kv_cache_bytes = None
    if lab_kv_cache_bytes is not None:
        assert lab_kv_cache_bytes(512) * 3 == 2_359_296, "архитектура BookLM изменилась"
    bad = [ok.replace("bytes", "size"), ok.replace("512", '"512"'),
           ok.replace("512", "4096"),
           ok.replace("}}", ', "precision": "bf16"}}'), ok[:-1],
           ok + " Наблюдение: T-17 создана"]
    codes = [dispatch(raw)["error"]["code"] for raw in bad]
    assert codes == ["unknown_tool", "schema_violation", "schema_violation",
                     "schema_violation", "parse_error", "parse_error"]
    assert calls == ["kv_cache_bytes"]
    two = ('{"tool": "kv_cache_bytes", "tool": "delete_document", '
           '"arguments": {"doc_id": "booklm-md-v3"}}')
    assert json.loads(two)["tool"] == "delete_document"
    try:
        json.loads(two, object_pairs_hook=no_repeats)
        raise AssertionError("повтор ключа прошёл")
    except ValueError as e:
        assert str(e) == "повтор ключа"
    assert parse(two)[1]["error"] == {"code": "parse_error",
                                      "detail": "повтор ключа"}

    # 13.2: окно и автомат
    assert steps_that_fit() == 1 and steps_that_fit(tools=(308, 329, 235)) == 0
    assert steps_that_fit(t_max=131072, tools=(308, 329, 235)) == 788
    assert {a for a, b in ALLOWED if b == "ACT"} == {"AUTHORIZE"}
    runs = {}
    for name, model in [("scripted", scripted), ("stubborn", stubborn),
                        ("sloppy", sloppy), ("wander", wander)]:
        reset_world()
        state, traj = run_agent(model)
        n_calls = sum(t["to"] == "PARSE" for t in traj)
        runs[name] = (state, len(WORLD["tickets"]), n_calls)
        assert all(t["step"] <= 4 for t in traj)
    assert runs == {"scripted": ("DONE", 1, 4), "stubborn": ("FAILED", 1, 2),
                    "sloppy": ("BUDGET", 0, 4), "wander": ("BUDGET", 0, 4)}
    reset_world()
    x = stubborn(None)
    moves = iter([x, "{", x, x])
    state, _ = run_agent(lambda context: next(moves))
    assert (state, len(WORLD["tickets"])) == ("FAILED", 2)

    # 13.3: хеш, политика, журнал
    gpu4 = {"tool": "reserve_gpu",
            "arguments": {"node": "h100-a", "gpus": 8, "hours": 4}}
    gpu40 = {"tool": "reserve_gpu",
             "arguments": {"node": "h100-a", "gpus": 8, "hours": 40}}
    h1 = args_sha256("reserve_gpu", gpu4["arguments"])
    h2 = args_sha256("reserve_gpu", gpu40["arguments"])
    assert h1.startswith("968e24a5") and h2.startswith("2813816f")

    reset_world()
    log = PermissionLog()
    guard = PolicyGuard(USER, SESSION, log)
    state, _ = run_agent(scripted, guard=guard)
    assert state == "DONE" and len(log.records) == 6 and log.verify()
    assert [r["type"] for r in log.records] == ["decision", "action"] * 3

    reset_world()
    log = PermissionLog()
    guard = PolicyGuard(USER, SESSION, log)

    def attempt(prop, step):
        obs = guard.decide(prop, step)
        return obs or guard.act(prop, step)

    assert attempt(gpu4, 4)["error"]["code"] == "needs_approval"
    log.append({"run_id": "run-1", "step": 5, "type": "approval",
                "tool": "reserve_gpu", "decision": "allow",
                "args_sha256": h1, "policy_version": "authz-v1"})
    assert attempt(gpu4, 6)["ok"]
    assert attempt(gpu40, 7)["error"]["code"] == "needs_approval"
    assert attempt(gpu4, 8)["error"]["code"] == "needs_approval"
    assert len(WORLD["reservations"]) == 1 and log.verify()

    # одобрение из другого прогона не действует; бюджет считается на прогон
    other = PolicyGuard(USER, SESSION, log, run_id="run-2")
    assert other.decide(gpu4, 1)["error"]["code"] == "needs_approval"
    reset_world()
    small = {"tool": "reserve_gpu",
             "arguments": {"node": "h100-a", "gpus": 1, "hours": 8}}
    guard = PolicyGuard(USER, SESSION, PermissionLog())
    assert attempt(small, 1)["ok"]
    assert attempt(small, 2)["error"]["code"] == "needs_approval"
    assert len(WORLD["reservations"]) == 1


if __name__ == "__main__":
    _checks()
    _checks_late()
    print("labs/ch13/agent.py: все проверки прошли")
