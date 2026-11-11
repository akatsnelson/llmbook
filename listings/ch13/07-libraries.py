"""Листинги главы 13 «Агенты и инструменты» книги «Большие языковые модели», файл 07-libraries.

Код и вывод — ровно как в книге; текст книги сюда не входит. Листинги идут по порядку и
опираются друг на друга: запускайте файл целиком или ячейками сверху вниз (разметка # %%).
Проверка, что вывод совпал с книгой: python scripts/check_listings.py listings/ch13/07-libraries.py
"""

# %% Листинг «Паспорт инструмента из функции» — раздел «Модели и паспорта инструментов в LangChain»
import sys, json
sys.path.insert(0, "labs/ch13")          # путь от корня репозитория
from typing import Annotated
from pydantic import Field
from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool
from agent import check

@tool
def reserve_gpu(node: str, gpus: int, hours: int) -> dict:
    """Забронировать GPU на узле."""
    return {"node": node, "gpus": gpus, "hours": hours}

schema = convert_to_openai_tool(reserve_gpu)["function"]
print(schema["name"], "|", schema["description"])
print(json.dumps(schema["parameters"]["properties"]))
print(reserve_gpu.invoke({"node": "h100-a", "gpus": 80, "hours": 4}))
print(reserve_gpu.invoke({"node": "h100-a", "gpus": "8", "hours": 4,
                          "force": True}))
ours = {"tool": "reserve_gpu",
        "arguments": {"node": "h100-a", "gpus": "8", "hours": 4}}
print("наш диспетчер:", check(ours)["error"]["code"])
# Вывод в книге:
# | reserve_gpu | Забронировать GPU на узле.
# | {"node": {"type": "string"}, "gpus": {"type": "integer"}, "hours": {"type": "integer"}}
# | {'node': 'h100-a', 'gpus': 80, 'hours': 4}
# | {'node': 'h100-a', 'gpus': 8, 'hours': 4}
# | наш диспетчер: schema_violation

# %% Листинг «Границы аргументов в аннотациях» — раздел «Модели и паспорта инструментов в LangChain»
@tool
def reserve_gpu(node: Annotated[str, Field(max_length=40)],
                gpus: Annotated[int, Field(ge=1, le=8)],
                hours: Annotated[int, Field(ge=1, le=72)]) -> dict:
    """Забронировать GPU на узле."""
    return {"node": node, "gpus": gpus, "hours": hours}

print(convert_to_openai_tool(reserve_gpu)["function"]["parameters"]
      ["properties"]["gpus"])
try:
    reserve_gpu.invoke({"node": "h100-a", "gpus": 80, "hours": 4})
except Exception as e:
    print(type(e).__name__, "|", str(e).splitlines()[2].strip())
# Вывод в книге:
# | {'maximum': 8, 'minimum': 1, 'type': 'integer'}
# | ValidationError | Input should be less than or equal to 8 [type=less_than_equal, input_value=80, input_type=int]

# %% Листинг «Цикл агента на LangGraph» — раздел «Цикл как граф в LangGraph»
import operator
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from agent import (PermissionLog, PolicyGuard, Guard, USER, SESSION, WORLD,
                   reset_world, parse, check, final_answer, scripted)

class State(TypedDict):
    context: Annotated[list, operator.add]   # дописывать, а не заменять
    raw: str                                 # последний ход модели
    answer: str

def build(model, guard, checkpointer=None):
    def propose(s):                          # модель работает только здесь
        return {"raw": model(s["context"])}

    def route(s):                            # условное ребро
        return "done" if final_answer(s["raw"]) is not None else "act"

    def act(s):                              # разбор, схема, право, действие
        step = len(s["context"]) + 1
        prop, obs = parse(s["raw"])
        if prop is not None:
            obs = check(prop) or guard.decide(prop, step)
        if obs is None:
            obs = guard.act(prop, step)
        return {"context": [(s["raw"], obs)]}

    def done(s):
        return {"answer": final_answer(s["raw"])}

    g = StateGraph(State)
    for name, fn in [("propose", propose), ("act", act), ("done", done)]:
        g.add_node(name, fn)
    g.add_edge(START, "propose")
    g.add_conditional_edges("propose", route, ["act", "done"])
    g.add_edge("act", "propose")
    g.add_edge("done", END)
    return g.compile(checkpointer=checkpointer)

reset_world()
log = PermissionLog()
app = build(scripted, PolicyGuard(USER, SESSION, log))
out = app.invoke({"context": []})
print(out["answer"])
print("шагов:", len(out["context"]), " заявок:", len(WORLD["tickets"]),
      " записей в журнале:", len(log.records))
# Вывод в книге:
# | Заявка T-17: 2 359 296 байт [S:booklm-md-v3-0042]
# | шагов: 3  заявок: 1  записей в журнале: 6

# %% Листинг «Предел рекурсии и упрямая модель» — раздел «Цикл как граф в LangGraph»
from langgraph.errors import GraphRecursionError
from agent import stubborn

for limit in (None, 8):
    reset_world()
    config = {} if limit is None else {"recursion_limit": limit}
    try:
        build(stubborn, Guard()).invoke({"context": []}, config)
    except GraphRecursionError as e:
        print(str(e).split(" without")[0], "| заявок:", len(WORLD["tickets"]))
# Вывод в книге:
# | Recursion limit of 10007 reached | заявок: 5003
# | Recursion limit of 8 reached | заявок: 4

# %% Листинг «Пауза на бронь GPU» — раздел «Пауза для подписи человека»
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command
from agent import args_sha256, POLICY, call, canonical

class AskingGuard(PolicyGuard):              # ask -> пауза до подписи
    def decide(self, prop, step):
        obs = super().decide(prop, step)
        if obs is None or obs["error"]["code"] != "needs_approval":
            return obs
        approver = interrupt({"tool": prop["tool"], "args": prop["arguments"],
                              "why": obs["error"]["detail"]})
        self.log.append({"run_id": self.run_id, "step": step,
                         "type": "approval", "tool": prop["tool"],
                         "decision": "allow", "approver": approver,
                         "args_sha256": args_sha256(prop["tool"],
                                                    prop["arguments"]),
                         "policy_version": POLICY["version"]})
        return super().decide(prop, step)

def gpu_model(context):                      # бронь на 8 GPU x 4 часа
    if not context:
        return call("reserve_gpu", node="h100-a", gpus=8, hours=4)
    obs = context[-1][1]
    text = obs["result"]["reservation_id"] if obs["ok"] else "отказ"
    return canonical({"final": f"Бронь: {text}"})

reset_world()
log = PermissionLog()
app = build(gpu_model, AskingGuard(USER, SESSION, log), InMemorySaver())
cfg = {"configurable": {"thread_id": "run-1"}}   # номер прогона
out = app.invoke({"context": []}, cfg)
print("пауза:", out["__interrupt__"][0].value["why"])
print("следующий узел:", app.get_state(cfg).next,
      " броней:", len(WORLD["reservations"]))
out = app.invoke(Command(resume="ops:ira"), cfg)  # подпись оператора
print(out["answer"], " броней:", len(WORLD["reservations"]))
for r in log.records:
    print(r["step"], r["type"], r["decision"])
# Вывод в книге:
# | пауза: GPU-часов за прогон 0+32 > 8
# | следующий узел: ('act',)  броней: 0
# | Бронь: R-1  броней: 1
# | 1 decision ask
# | 1 decision ask
# | 1 approval allow
# | 1 decision allow
# | 1 action None

# %% Листинг «Падение после заявки и продолжение прогона» — раздел «Падение посреди узла»
from agent import Crash, IdempotentGuard, KeyedServer

class CrashOnce(PolicyGuard):                # падение после заявки, точка C
    crashed = False
    def act(self, prop, step):
        obs = super().act(prop, step)
        if prop["tool"] == "create_ticket" and not self.crashed:
            self.crashed = True
            raise Crash
        return obs

class KeyedCrashOnce(IdempotentGuard):       # страж из 13.4, то же падение
    crashed = False
    def act(self, prop, step):
        first = prop["tool"] == "create_ticket" and not self.crashed
        self.crashed = self.crashed or first
        self.crash_at = "C" if first else None
        return super().act(prop, step)

def crash_and_resume(guard, run_id):
    reset_world()
    app = build(scripted, guard, InMemorySaver())
    cfg = {"configurable": {"thread_id": run_id}}
    try:
        app.invoke({"context": []}, cfg)
    except Crash:
        print(run_id, "упал в узле", app.get_state(cfg).next,
              "заявок:", len(WORLD["tickets"]))
    out = app.invoke(None, cfg)              # продолжить с контрольной точки
    print(run_id, out["answer"][:13], "заявок:", len(WORLD["tickets"]))

crash_and_resume(CrashOnce(USER, SESSION, PermissionLog(), run_id="run-2"),
                 "run-2")
crash_and_resume(KeyedCrashOnce(USER, SESSION, PermissionLog(),
                                KeyedServer(), [], run_id="run-3"), "run-3")
# Вывод в книге:
# | run-2 упал в узле ('act',) заявок: 1
# | run-2 Заявка T-18:  заявок: 2
# | run-3 упал в узле ('act',) заявок: 1
# | run-3 Заявка T-17:  заявок: 1

# %% Листинг «Повтор после тайм-аута в create_agent» — раздел «Готовый агент и посредники»
from langchain.agents import create_agent
from langchain.agents.middleware import ToolRetryMiddleware
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

class ScriptedChat(GenericFakeChatModel):    # заглушка вместо модели
    def bind_tools(self, tools, **kwargs):   # паспорта ей не нужны
        return self

lost = {"answers": 1}                        # сколько ответов потеряет сеть
@tool
def create_ticket(title: str, body: str) -> str:
    """Завести заявку в трекере."""
    WORLD["tickets"].append({"title": title, "body": body})   # эффект есть
    if lost["answers"]:
        lost["answers"] -= 1
        raise TimeoutError("ответ сервера потерялся")
    return f"T-{16 + len(WORLD['tickets'])}"

turns = [AIMessage("", tool_calls=[{"name": "create_ticket", "id": "c1",
             "args": {"title": "KV-кэш 3x512", "body": "2 359 296 байт"}}]),
         AIMessage("Заявка заведена.")]
reset_world()
agent = create_agent(ScriptedChat(messages=iter(turns)), [create_ticket],
                     middleware=[ToolRetryMiddleware(initial_delay=0)])
out = agent.invoke({"messages": [{"role": "user",
                                  "content": "Заведи заявку про KV-кэш"}]})
for m in out["messages"]:
    print(f"{type(m).__name__:<13}", m.content or m.tool_calls[0]["name"])
print("заявок:", len(WORLD["tickets"]))
# Вывод в книге:
# | HumanMessage  Заведи заявку про KV-кэш
# | AIMessage     create_ticket
# | ToolMessage   T-18
# | AIMessage     Заявка заведена.
# | заявок: 2
