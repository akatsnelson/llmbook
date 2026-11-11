"""Глава 13, раздел 13.6 «Передача задачи и протокол MCP»:
листинг «ToolSpec в формате MCP и решение хоста» (полная версия).

В книге оставлена сокращённая версия; этот файл самодостаточен и
выполняется в ноутбуке главы notebooks/13_tools.ipynb.

Паспорт ToolSpec из раздела 13.1 перекладывается в описание инструмента
MCP (tools/list): класс эффекта превращается в аннотации readOnlyHint и
destructiveHint. Хост закрепляет класс эффекта сам при подключении и
решает allow / ask / deny, не читая аннотаций: «успокаивающая» аннотация
сервера политику мягче не делает.

Запуск из корня репозитория: python labs/ch13/mcp_tool_spec.py
"""

import re
from collections import namedtuple

# Поиск такой же, как в labs/ch13/plan_execute.py (исполнитель плана, раздел 13.5).
CORPUS = [
    {"id": "deploy-09", "status": "archived",
     "text": "Архивное развёртывание deploy-09 запускало BookLM на "
             "A100-40GB. Эта запись не описывает действующий сервис."},
    {"id": "deploy-17", "status": "active",
     "text": "Развёртывание deploy-17 запускает артефакт booklm-rag-v1 "
             "на ускорителе H100-80GB. Ёмкость HBM равна 80 ГБ."}]
words = lambda text: set(re.findall(r"\w+", text.lower()))


def search_docs(query, k, access):
    return [d for d in CORPUS if words(query) <= words(d["text"])][:k]


ToolSpec = namedtuple("ToolSpec", "name description input_schema "
                      "output_schema effect fn")   # как в разделе 13.1
HINTS = {"read": {"readOnlyHint": True},
         "reversible": {"readOnlyHint": False,
                        "destructiveHint": False},
         "irreversible": {"readOnlyHint": False,
                          "destructiveHint": True}}

def to_mcp_tool(spec):
    return {"name": spec.name, "description": spec.description,
            "inputSchema": spec.input_schema,
            "outputSchema": spec.output_schema,
            "annotations": dict(HINTS[spec.effect])}

def obj(**props):
    return {"type": "object", "properties": props,
            "required": list(props), "additionalProperties": False}

STR, ARR = {"type": "string"}, {"type": "array"}
K = {"type": "integer", "minimum": 1, "maximum": 5}
ACCESS = {"type": "string", "enum": ["public", "finance"]}
specs = [ToolSpec("search_docs", "Поиск по booklm-rag-v1",
                  obj(query=STR, k=K, access=ACCESS), obj(hits=ARR),
                  "read", search_docs),
         ToolSpec("delete_document", "Удалить документ из индекса",
                  obj(doc_id=STR), obj(ok={"type": "boolean"}),
                  "irreversible", lambda doc_id: {"ok": True})]
listed = [to_mcp_tool(s) for s in specs]
assert listed[0]["inputSchema"] is specs[0].input_schema
assert listed[0]["annotations"] == {"readOnlyHint": True}

HOST_EFFECT = {s.name: s.effect for s in specs}  # при подключении
DECISION = {"read": "allow", "reversible": "allow",
            "irreversible": "ask"}

def host_authorize(name, args, annotations):   # аннотации не читаем
    effect = HOST_EFFECT.get(name)
    return "deny" if effect is None else DECISION[effect]

lying = {"readOnlyHint": True, "destructiveHint": False}  # «успокаивает»
for tool in listed:
    honest = host_authorize(tool["name"], {}, tool["annotations"])
    assert honest == host_authorize(tool["name"], {}, lying)
doomed = {"doc_id": "booklm-contract"}
assert host_authorize("delete_document", doomed, lying) == "ask"
assert host_authorize("drop_index", {}, {"readOnlyHint": True}) == "deny"

print("mcp_tool_spec: все проверки пройдены")
