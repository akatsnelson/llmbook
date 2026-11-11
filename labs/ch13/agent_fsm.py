"""Глава 13, раздел 13.2 «Цикл агента как конечный автомат»:
листинги «Конечный автомат агентного цикла» и «Четыре заглушки
против одного автомата» (полная версия).

В книге оставлены сокращённые версии; этот файл самодостаточен и
выполняется в ноутбуке главы notebooks/13_tools.ipynb.

run_agent ведёт цикл по таблице разрешённых переходов ALLOWED: модель
вызывается только в PROPOSE, остальные состояния — программа. Четыре
заглушки модели (сценарная, упрямая, неряшливая, блуждающая) проверяют
выходы DONE, FAILED и BUDGET; в ACT ведут только AUTHORIZE и WAIT_HUMAN.

Запуск из корня репозитория: python labs/ch13/agent_fsm.py
"""

import inspect, json

def canonical(obj):              # канонический JSON главы
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)

CHUNK = "ch-04de13a141cc"        # booklm-contract#kv-cache, гл. 12
tickets = []                     # внешний мир: заявки на сервере

def search_docs(query, k, access):
    return [{"chunk_id": CHUNK, "text": "1536 байт на токен"}][:k]

def kv_cache_bytes(tokens, requests, dtype):  # 1536 Б/токен: 13.1
    return {"bf16": 1536, "fp32": 3072}[dtype] * tokens * requests

def create_ticket(title, body):
    tickets.append(title)
    return {"ticket_id": f"T-{16 + len(tickets)}"}

TOOLS = {"search_docs": search_docs, "kv_cache_bytes": kv_cache_bytes,
         "create_ticket": create_ticket}

ALLOWED = {                       # разрешённые переходы автомата
    ("PROPOSE", "PARSE"),
    ("PARSE", "VALIDATE"), ("PARSE", "DONE"), ("PARSE", "OBSERVE"),
    ("VALIDATE", "AUTHORIZE"), ("VALIDATE", "OBSERVE"),
    ("VALIDATE", "FAILED"),
    ("AUTHORIZE", "ACT"), ("AUTHORIZE", "OBSERVE"),
    ("AUTHORIZE", "WAIT_HUMAN"),
    ("WAIT_HUMAN", "ACT"), ("WAIT_HUMAN", "OBSERVE"),
    ("ACT", "OBSERVE"), ("OBSERVE", "PROPOSE"), ("OBSERVE", "BUDGET")}

def fits_signature(fn, args):    # грубая схема; полная — в 13.1
    try:
        inspect.signature(fn).bind(**args)
        return True
    except TypeError:
        return False

def authorize(call):             # заглушка «разрешить всё» до 13.3
    return "allow"

def run_agent(model, tools, n_max=4, repeat_limit=2):
    trajectory, context = [], []  # пишет программа / видит модель
    state, last_key, same = "PROPOSE", None, 0

    def go(new, **info):          # только так меняется состояние
        nonlocal state
        assert (state, new) in ALLOWED, (state, new)
        trajectory.append({"step": step, "from": state, "to": new,
                           **info})
        state = new

    for step in range(1, n_max + 1):
        raw = model(context)      # PROPOSE: модель — только здесь
        go("PARSE", raw=raw)
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            msg = None
        if isinstance(msg, dict) and "final" in msg:
            go("DONE", answer=msg["final"])
            return state, trajectory
        if not isinstance(msg, dict):
            obs = {"error": "parse_error"}
            last_key, same = None, 0  # битый текст прерывает серию
        else:
            go("VALIDATE")
            key = canonical([msg.get("tool"), msg.get("arguments")])
            same, last_key = (same + 1 if key == last_key else 1), key
            # После needs_approval (раздел 13.3) счётчик тоже обнуляют:
            # тот же вызов ещё раз — ожидаемый второй заход.
            if same >= repeat_limit:
                go("FAILED", reason="repeat_limit")
                return state, trajectory
            fn, args = tools.get(msg.get("tool")), msg.get("arguments")
            if fn is None:
                obs = {"error": "unknown_tool"}
            elif not fits_signature(fn, args):
                obs = {"error": "schema_violation"}
            else:
                go("AUTHORIZE")
                decision = authorize(msg)
                if decision == "ask":
                    go("WAIT_HUMAN")
                    return state, trajectory
                if decision != "allow":
                    obs = {"error": "denied"}
                else:
                    go("ACT", tool=msg["tool"])
                    try:
                        obs = {"result": fn(**args)}
                    except Exception as exc:  # сбой — тоже наблюдение
                        obs = {"error": "tool_error",
                               "detail": str(exc)}
        go("OBSERVE", obs=obs)
        context.append((raw, obs))
        if step < n_max:
            go("PROPOSE")
    go("BUDGET", reason="n_max")
    return state, trajectory

# --- четыре модели-заглушки против одного автомата ---
def scripted(ctx):               # сценарная заглушка: шаг -> ход
    if len(ctx) == 0:
        return canonical({"tool": "search_docs", "arguments": {
            "query": "KV-кэш BookLM байт на токен",
            "k": 3, "access": "public"}})
    if len(ctx) == 1:
        return canonical({"tool": "kv_cache_bytes", "arguments":
                      {"tokens": 512, "requests": 3, "dtype": "bf16"}})
    chunk = ctx[0][1]["result"][0]["chunk_id"]   # шаг 3 опирается
    total = f"{ctx[1][1]['result']:,}".replace(",", " ")  # 2 359 296
    if len(ctx) == 2:
        return canonical({"tool": "create_ticket", "arguments": {
            "title": "KV-кэш 3x512",
            "body": f"{total} байт [S:{chunk}]"}})
    ticket = ctx[2][1]["result"]["ticket_id"]
    return canonical({"final":
                  f"Заявка {ticket}: {total} байт [S:{chunk}]"})

stubborn = lambda ctx: canonical({"tool": "create_ticket",
                    "arguments": {"title": "KV", "body": "x"}})
sloppy = lambda ctx: '{"tool": "search_docs", '  # всегда битый JSON
wander = lambda ctx: canonical({"tool": "search_docs", "arguments": {
    "query": f"попытка {len(ctx)}", "k": 3, "access": "public"}})

def model_calls(traj):
    return sum(t["to"] == "PARSE" for t in traj)

runs = {}
for name, model in [("scripted", scripted), ("stubborn", stubborn),
                    ("sloppy", sloppy), ("wander", wander)]:
    tickets.clear()
    final, traj = run_agent(model, TOOLS, n_max=4, repeat_limit=2)
    runs[name] = (final, len(tickets), model_calls(traj), traj)

final, n_tickets, calls, traj = runs["scripted"]
assert (final, n_tickets, calls) == ("DONE", 1, 4)
assert "T-17" in traj[-1]["answer"] and "2 359 296" in traj[-1]["answer"]
assert runs["stubborn"][:2] == ("FAILED", 1)    # заявка ровно одна
assert runs["sloppy"][:2] == ("BUDGET", 0)
assert {t["obs"]["error"] for t in runs["sloppy"][3]
        if t["to"] == "OBSERVE"} == {"parse_error"}
assert runs["wander"][0] == "BUDGET" and runs["wander"][2] == 4
tickets.clear()                  # X, битый JSON, X: это не повтор подряд
moves = iter([stubborn(0), "{", stubborn(0), stubborn(0)])
final, _ = run_agent(lambda ctx: next(moves), TOOLS, n_max=4)
assert (final, len(tickets)) == ("FAILED", 2)   # серия — шаги 3 и 4
for final, _, calls, traj in runs.values():
    assert calls <= 4                           # не больше N_max
    assert all((t["from"], t["to"]) in ALLOWED for t in traj)
    assert {t["step"] for t in traj} == set(range(1, calls + 1))
into_act = {a for a, b in ALLOWED if b == "ACT"}
assert into_act == {"AUTHORIZE", "WAIT_HUMAN"}

print({k: v[:3] for k, v in runs.items()})
