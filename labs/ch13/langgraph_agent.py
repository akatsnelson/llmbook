"""Глава 13, раздел 13.8 «Библиотеки LangChain и LangGraph»: агент главы на готовых библиотеках.

Код повторяет листинги раздела и проверяет утверждениями то, что печатает книга:
  - @tool строит схему только из типов: 80 GPU и строка "8" проходят, наш диспетчер даёт schema_violation;
  - граф из трёх узлов (propose, act, done) со стражем PolicyGuard даёт ту же заявку и шесть записей журнала;
  - упрямая заглушка при пределе рекурсии по умолчанию (10 007 узлов) заводит 5003 заявки, при пределе 8 — 4;
  - пауза interrupt: после подписи бронь одна, но узел act исполнен дважды, и в журнале два решения ask;
  - падение после create_ticket: без ключа после продолжения две заявки, с IdempotentGuard из 13.4 — одна;
  - create_agent с ToolRetryMiddleware после потерянного ответа заводит вторую заявку.

Нужны пакеты из requirements-extra.txt (langgraph, langchain); без них файл сообщает, что проверка пропущена.
Проверено на langgraph 1.2.20, langchain 1.5.0, langchain-core 1.8.0.

Запуск из корня репозитория: python labs/ch13/langgraph_agent.py
"""

import operator
import sys
from pathlib import Path
from typing import Annotated, TypedDict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from agent import (POLICY, SESSION, USER, WORLD, Crash, Guard,  # noqa: E402
                   IdempotentGuard, KeyedServer, PermissionLog, PolicyGuard,
                   args_sha256, call, canonical, check, final_answer, parse,
                   reset_world, scripted, stubborn)


class State(TypedDict):
    context: Annotated[list, operator.add]   # пары (ход, наблюдение)
    raw: str                                 # последний ход модели
    answer: str


def build(model, guard, checkpointer=None):
    from langgraph.graph import END, START, StateGraph

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


class AskingGuard(PolicyGuard):              # ask -> пауза до подписи
    def decide(self, prop, step):
        from langgraph.types import interrupt
        obs = super().decide(prop, step)
        if obs is None or obs["error"]["code"] != "needs_approval":
            return obs
        approver = interrupt({"tool": prop["tool"], "args": prop["arguments"],
                              "why": obs["error"]["detail"]})
        self.log.append({"run_id": self.run_id, "step": step,
                         "type": "approval", "tool": prop["tool"],
                         "decision": "allow", "approver": approver,
                         "args_sha256": args_sha256(prop["tool"], prop["arguments"]),
                         "policy_version": POLICY["version"]})
        return super().decide(prop, step)


def gpu_model(context):                      # бронь на 8 GPU x 4 часа
    if not context:
        return call("reserve_gpu", node="h100-a", gpus=8, hours=4)
    obs = context[-1][1]
    text = obs["result"]["reservation_id"] if obs["ok"] else "отказ"
    return canonical({"final": f"Бронь: {text}"})


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


def crash_and_resume(guard, run_id):         # число заявок до и после продолжения
    from langgraph.checkpoint.memory import InMemorySaver
    reset_world()
    app = build(scripted, guard, InMemorySaver())
    cfg = {"configurable": {"thread_id": run_id}}
    try:
        app.invoke({"context": []}, cfg)
    except Crash:
        assert app.get_state(cfg).next == ("act",)
    before = len(WORLD["tickets"])
    out = app.invoke(None, cfg)              # продолжить с контрольной точки
    return before, len(WORLD["tickets"]), out["answer"]


def _checks():
    from langchain_core.tools import tool
    from langchain_core.utils.function_calling import convert_to_openai_tool
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.errors import GraphRecursionError
    from langgraph.types import Command

    # Паспорт из функции: только типы, лишнее выброшено, строка приведена к числу.
    @tool
    def reserve_gpu(node: str, gpus: int, hours: int) -> dict:
        """Забронировать GPU на узле."""
        return {"node": node, "gpus": gpus, "hours": hours}

    props = convert_to_openai_tool(reserve_gpu)["function"]["parameters"]["properties"]
    assert props == {"node": {"type": "string"}, "gpus": {"type": "integer"},
                     "hours": {"type": "integer"}}
    assert reserve_gpu.invoke({"node": "h100-a", "gpus": 80, "hours": 4})["gpus"] == 80
    assert reserve_gpu.invoke({"node": "h100-a", "gpus": "8", "hours": 4,
                               "force": True}) == {"node": "h100-a", "gpus": 8, "hours": 4}
    ours = {"tool": "reserve_gpu", "arguments": {"node": "h100-a", "gpus": "8", "hours": 4}}
    assert check(ours)["error"]["code"] == "schema_violation"
    print("@tool: границ нет, \"8\" приведено к 8, лишний аргумент выброшен; наш диспетчер: schema_violation")

    # Цикл на графе со стражем политики.
    reset_world()
    log = PermissionLog()
    out = build(scripted, PolicyGuard(USER, SESSION, log)).invoke({"context": []})
    assert out["answer"] == "Заявка T-17: 2 359 296 байт [S:booklm-md-v3-0042]"
    assert len(out["context"]) == 3 and len(WORLD["tickets"]) == 1 and len(log.records) == 6
    print("граф: ответ с T-17, одна заявка, 6 записей журнала")

    # Предел рекурсии.
    found = []
    for config in ({}, {"recursion_limit": 8}):
        reset_world()
        try:
            build(stubborn, Guard()).invoke({"context": []}, config)
        except GraphRecursionError:
            found.append(len(WORLD["tickets"]))
    assert found == [5003, 4], found
    print("предел рекурсии: по умолчанию 5003 заявки, при пределе 8 — 4")

    # Пауза для подписи.
    reset_world()
    log = PermissionLog()
    app = build(gpu_model, AskingGuard(USER, SESSION, log), InMemorySaver())
    cfg = {"configurable": {"thread_id": "run-1"}}
    out = app.invoke({"context": []}, cfg)
    assert out["__interrupt__"][0].value["why"] == "GPU-часов за прогон 0+32 > 8"
    assert app.get_state(cfg).next == ("act",) and not WORLD["reservations"]
    out = app.invoke(Command(resume="ops:ira"), cfg)
    assert out["answer"] == "Бронь: R-1" and len(WORLD["reservations"]) == 1
    kinds = [(r["type"], r["decision"]) for r in log.records]
    assert kinds == [("decision", "ask"), ("decision", "ask"), ("approval", "allow"),
                     ("decision", "allow"), ("action", None)], kinds
    assert log.verify()
    print("пауза: бронь одна, в журнале два решения ask (узел исполнен с начала)")

    # Падение посреди узла.
    plain = crash_and_resume(CrashOnce(USER, SESSION, PermissionLog(), run_id="run-2"), "run-2")
    keyed = crash_and_resume(KeyedCrashOnce(USER, SESSION, PermissionLog(), KeyedServer(), [],
                                            run_id="run-3"), "run-3")
    assert plain[:2] == (1, 2) and plain[2].startswith("Заявка T-18"), plain
    assert keyed[:2] == (1, 1) and keyed[2].startswith("Заявка T-17"), keyed
    print("падение: без ключа две заявки, с ключом из 13.4 одна")

    # create_agent и посредник повторов.
    from langchain.agents import create_agent
    from langchain.agents.middleware import ToolRetryMiddleware
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage

    class ScriptedChat(GenericFakeChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

    lost = {"answers": 1}

    @tool
    def create_ticket(title: str, body: str) -> str:
        """Завести заявку в трекере."""
        WORLD["tickets"].append({"title": title, "body": body})
        if lost["answers"]:
            lost["answers"] -= 1
            raise TimeoutError("ответ сервера потерялся")
        return f"T-{16 + len(WORLD['tickets'])}"

    turns = [AIMessage("", tool_calls=[{"name": "create_ticket", "id": "c1",
                                        "args": {"title": "KV-кэш 3x512",
                                                 "body": "2 359 296 байт"}}]),
             AIMessage("Заявка заведена.")]
    reset_world()
    agent = create_agent(ScriptedChat(messages=iter(turns)), [create_ticket],
                         middleware=[ToolRetryMiddleware(initial_delay=0)])
    out = agent.invoke({"messages": [{"role": "user", "content": "Заведи заявку про KV-кэш"}]})
    assert [m.content for m in out["messages"]][2:] == ["T-18", "Заявка заведена."]
    assert len(WORLD["tickets"]) == 2
    print("create_agent + ToolRetryMiddleware: после потерянного ответа две заявки")


if __name__ == "__main__":
    try:
        import langchain  # noqa: F401
        import langgraph  # noqa: F401
    except ImportError:
        print("langgraph или langchain не установлены (pip install -r requirements-extra.txt): проверка пропущена")
        sys.exit(0)
    _checks()
    print("OK")
