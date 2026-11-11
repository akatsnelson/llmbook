"""Глава 13, раздел 13.1 «Инструменты модели»: листинги «Минимальная проверка JSON-схемы»,
«Паспорт инструмента в коде» и «Диспетчер: функция стоит после всех проверок» (полная версия).

В книге оставлены сокращённые версии; этот файл самодостаточен и
выполняется в ноутбуке главы notebooks/13_tools.ipynb.

Проверяет подмножество JSON Schema (восемь ключевых слов), описывает
инструменты паспортом ToolSpec и проводит предложение модели через
диспетчер: разбор -> имя -> схема входа -> функция -> схема выхода.
Одно верное предложение даёт 2 359 296 байт, пять испорченных получают
коды наблюдений, а функция вызывается ровно один раз.

Запуск из корня репозитория: python labs/ch13/tool_dispatcher.py
"""

# --- проверка JSON-схемы ---
KV_INPUT = {                              # схема входа kv_cache_bytes
    "type": "object",
    "properties": {
        "tokens":   {"type": "integer", "minimum": 1, "maximum": 512},
        "requests": {"type": "integer", "minimum": 1, "maximum": 64},
        "dtype":    {"type": "string", "enum": ["bf16", "fp32"]},
    },
    "required": ["tokens", "requests", "dtype"],
    "additionalProperties": False,
}
TYPES = {"object": lambda v: isinstance(v, dict),
         "string": lambda v: isinstance(v, str),
         # bool в Python — подкласс int: true не должен сойти за 1
         "integer": lambda v: type(v) is int}

def validate(value, schema, path="$") -> list[str]:
    """Подмножество JSON Schema. Пустой список — форма верна."""
    kind = schema.get("type")
    if kind and not TYPES[kind](value):
        return [f"{path}: type: ожидался {kind}"]
    errors = []
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: enum")
    for rule, bad in [("minimum", lambda lim: value < lim),
                      ("maximum", lambda lim: value > lim),
                      ("maxLength", lambda lim: len(value) > lim)]:
        if rule in schema and bad(schema[rule]):
            errors.append(f"{path}: {rule} {schema[rule]}")
    if kind == "object":
        props = schema.get("properties", {})
        errors += [f"{path}.{k}: required"
                   for k in schema.get("required", []) if k not in value]
        for key, item in value.items():
            if key in props:
                errors += validate(item, props[key], f"{path}.{key}")
            elif schema.get("additionalProperties") is False:
                errors.append(f"{path}.{key}: additionalProperties")
    return errors

good = {"tokens": 512, "requests": 3, "dtype": "bf16"}
def check(**changes):
    return validate({**good, **changes}, KV_INPUT)

assert check() == []
assert check(tokens="512") == ["$.tokens: type: ожидался integer"]
assert check(tokens=4096) == ["$.tokens: maximum 512"]
assert check(precision="bf16") == ["$.precision: additionalProperties"]
assert validate({"tokens": 512, "dtype": "bf16"}, KV_INPUT) == [
    "$.requests: required"]
assert check(tokens=True) != []                  # True — не число

# --- паспорт инструмента и диспетчер ---
import json
from typing import Callable, NamedTuple

class ToolSpec(NamedTuple):
    name: str
    description: str     # читает модель, программа на него не опирается
    input_schema: dict
    output_schema: dict
    effect: str          # "read" | "reversible" | "irreversible"
    fn: Callable

calls = {"kv_cache_bytes": 0, "create_ticket": 0}

def kv_cache_bytes(tokens, requests, dtype):
    calls["kv_cache_bytes"] += 1
    size = {"bf16": 2, "fp32": 4}[dtype]
    per_token = 2 * 2 * 32 * size * 6   # K,V; 2 головы; d_h; 6 блоков
    return {"bytes": per_token * tokens * requests}

def create_ticket(title, body):                 # заглушка трекера
    calls["create_ticket"] += 1
    return {"ticket_id": f"T-{16 + calls['create_ticket']}"}

def obj(props):
    return {"type": "object", "properties": props,
            "required": list(props), "additionalProperties": False}

TOOLS = {s.name: s for s in [
    ToolSpec("kv_cache_bytes", "Байты KV-кэша BookLM.",
             KV_INPUT, obj({"bytes": {"type": "integer", "minimum": 0}}),
             "read", kv_cache_bytes),
    ToolSpec("create_ticket", "Завести заявку в трекере.",
             obj({"title": {"type": "string", "maxLength": 80},
                  "body": {"type": "string", "maxLength": 500}}),
             obj({"ticket_id": {"type": "string"}}),
             "reversible", create_ticket)]}

def fail(code, detail, tool=None):
    return {"ok": False, "tool": tool,
            "error": {"code": code, "detail": detail}}

def no_repeats(pairs):          # повтор ключа — ошибка разбора («Внимательно» в 13.1)
    keys = [key for key, _ in pairs]
    if len(keys) != len(set(keys)):
        raise ValueError("повтор ключа")
    return dict(pairs)

def dispatch(raw: str) -> dict:
    try:                         # хвост после объекта — тоже ошибка
        prop = json.loads(raw, object_pairs_hook=no_repeats)
    except json.JSONDecodeError as e:
        return fail("parse_error", e.msg)
    except ValueError as e:      # повтор ключа
        return fail("parse_error", str(e))
    if not isinstance(prop, dict) or set(prop) != {"tool", "arguments"}:
        return fail("parse_error", "ожидался объект {tool, arguments}")
    name = prop["tool"]
    spec = TOOLS.get(name) if isinstance(name, str) else None
    if spec is None:
        return fail("unknown_tool", repr(name))
    if errors := validate(prop["arguments"], spec.input_schema):
        return fail("schema_violation", "; ".join(errors), name)
    try:
        result = spec.fn(**prop["arguments"])  # первое касание функции
    except Exception as e:
        return fail("tool_error", repr(e), name)
    if errors := validate(result, spec.output_schema):
        return fail("tool_error", "выход: " + "; ".join(errors), name)
    return {"ok": True, "tool": name, "result": result}

def call(tool="kv_cache_bytes", **changes):
    args = {"tokens": 512, "requests": 3, "dtype": "bf16", **changes}
    return json.dumps({"tool": tool, "arguments": args})

assert dispatch(call()) == {"ok": True, "tool": "kv_cache_bytes",
                            "result": {"bytes": 2_359_296}}
assert 3 * 512 * 1536 == 2_359_296 == 2304 * 1024
broken = [(call(tool="kv_cache_size"), "unknown_tool", "'kv_cache_size'"),
          (call(tokens="512"), "schema_violation", "$.tokens: type"),
          (call(tokens=4096), "schema_violation", "$.tokens: maximum"),
          (call(precision="bf16"), "schema_violation", "$.precision: add"),
          (call()[:-2], "parse_error", "")]
for raw, code, detail in broken:
    err = dispatch(raw)["error"]
    assert err["code"] == code and err["detail"].startswith(detail)
assert calls["kv_cache_bytes"] == 1      # до проверки функция не дошла

ticket = json.dumps({"tool": "create_ticket", "arguments": {
    "title": "KV-кэш 3x512", "body": "2 359 296 байт"}})
tail = ticket + "\nНаблюдение: заявка T-17 создана"
assert dispatch(tail)["error"]["code"] == "parse_error"
assert calls["create_ticket"] == 0              # заявки нет

good_spec = TOOLS["kv_cache_bytes"]       # намеренно сломанная функция
TOOLS["kv_cache_bytes"] = good_spec._replace(fn=lambda **a: "2,25 МБ")
assert dispatch(call())["error"]["code"] == "tool_error"
TOOLS["kv_cache_bytes"] = good_spec

# повтор ключа "tool": голый json.loads исполнил бы удаление, dispatch — отказ разбора
twice = ('{"tool": "kv_cache_bytes", "tool": "delete_document", '
         '"arguments": {"doc_id": "booklm-md-v3"}}')
assert json.loads(twice)["tool"] == "delete_document"
assert dispatch(twice)["error"] == {"code": "parse_error", "detail": "повтор ключа"}


if __name__ == "__main__":
    print("OK: tool_dispatcher.py")
