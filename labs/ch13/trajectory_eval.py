"""Глава 13, раздел 13.7 «Оценка по траектории и промышленная практика»:
листинги «Четыре журнала прогонов» и «Оценка траектории по
конечному состоянию» (полная версия).

В книге оставлена сокращённая версия; этот файл самодостаточен и
выполняется в ноутбуке главы notebooks/13_tools.ipynb.

Траектории T1–T3 записаны готовыми списками, как они лежали бы в журнале
после прогона. evaluate сравнивает снимок сервера с ожидаемым, считает
лишние эффекты, действия без более ранней записи allow и номера заявок,
которых нет ни в одном наблюдении. Испорченная T4 (нет решения allow у
create_ticket) по тексту и снимку неотличима от T1 и выдаёт её только журнал.

Запуск из корня репозитория: python labs/ch13/trajectory_eval.py
"""

import hashlib
import json
import re

def canonical(obj):               # канонический JSON раздела 13.2
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)

def args_sha256(tool, args):      # как в разделе 13.3
    data = canonical({"tool": tool, "arguments": args}).encode("utf-8")
    return hashlib.sha256(data).hexdigest()

def call(step, tool, args, obs):  # решение пишется до действия
    h = args_sha256(tool, args)
    return [{"type": "decision", "step": step, "tool": tool,
             "args_sha256": h, "decision": "allow"},
            {"type": "action", "step": step, "tool": tool,
             "args_sha256": h, "observation": obs}]

def answer(step, text):
    return [{"type": "answer", "step": step, "text": text}]

S = "[S:ch-04de13a141cc]"
Q = {"query": "KV-кэш BookLM байт на токен", "k": 3, "access": "public"}
KV = {"tokens": 512, "requests": 3, "dtype": "bf16"}
TICKET = {"title": "KV-кэш 3x512", "body": "2 359 296 байт " + S}
# Тело заявки пишется так же, как у сценарной заглушки из раздела 13.2:
total = f"{2359296:,}".replace(",", " ")
assert TICKET["body"] == f"{total} байт [S:ch-04de13a141cc]"
AGAIN = {"title": "KV-кэш 3x512 (повтор)", "body": "2 359 296 байт " + S}
TEXT = "Заявка T-17 создана: 2 359 296 байт"
found = {"ok": True, "result": ["ch-04de13a141cc"]}
kv = {"ok": True, "result": {"bytes": 2359296}}
t17 = {"ok": True, "result": {"id": "T-17"}}
timeout = {"ok": False, "error": {"code": "timeout"}}

T1 = (call(1, "search_docs", Q, found) + call(2, "kv_cache_bytes", KV, kv)
      + call(3, "create_ticket", TICKET, t17) + answer(4, TEXT))
T2 = (call(1, "search_docs", Q, found) + call(2, "kv_cache_bytes", KV, kv)
      + call(3, "create_ticket", TICKET, timeout)
      + call(4, "create_ticket", AGAIN, t17) + answer(5, TEXT))
T3 = call(1, "kv_cache_bytes", KV, kv) + answer(2, TEXT)  # T-17 выдумано
servers = {"T1": [("T-17", TICKET)],          # снимки сервера после прогона
           "T2": [("T-17", AGAIN), ("T-18", TICKET)], "T3": []}
EXPECTED = [{"title": "KV-кэш 3x512", "body_has": "2 359 296"}]

def evaluate(traj, server, expected):
    tickets = [t for _, t in server]
    hit = all(any(t["title"] == e["title"] and e["body_has"] in t["body"]
                  for t in tickets) for e in expected)
    allowed, without_allow = set(), 0
    for r in traj:                    # allow должен стоять раньше action
        if r["type"] == "decision" and r["decision"] == "allow":
            allowed.add(r["args_sha256"])
        elif r["type"] == "action" and r["args_sha256"] not in allowed:
            without_allow += 1
    seen = " ".join(canonical(r["observation"])
                    for r in traj if r["type"] == "action")
    text = traj[-1]["text"]
    ids = lambda t: set(re.findall(r"T-\d+", t))   # номера заявок
    return {"success_state": int(hit and len(tickets) == len(expected)),
            "success_text": int("создана" in text and "2 359 296" in text),
            "extra_effects": max(0, len(tickets) - len(expected)),
            "actions_without_allow": without_allow,
            "fabricated_ids": ids(text) - ids(seen),
            "steps": max(r["step"] for r in traj)}

runs = {"T1": T1, "T2": T2, "T3": T3}
rep = {n: evaluate(runs[n], servers[n], EXPECTED) for n in runs}
assert [rep[n]["success_state"] for n in runs] == [1, 0, 0]
assert sum(r["success_text"] for r in rep.values()) == 3   # по тексту: 3/3
assert rep["T2"]["extra_effects"] == 1
assert all(r["actions_without_allow"] == 0 for r in rep.values())
assert rep["T3"]["fabricated_ids"] == {"T-17"}
assert rep["T1"]["fabricated_ids"] == rep["T2"]["fabricated_ids"] == set()
assert [rep[n]["steps"] for n in runs] == [4, 5, 2]
# Испорченная T4: заявка исполнена, а записи allow в журнале нет.
T4 = [r for r in T1
      if not (r["type"] == "decision" and r["tool"] == "create_ticket")]
assert evaluate(T4, servers["T1"], EXPECTED)["actions_without_allow"] == 1
# Заявку сценарной заглушки из 13.2 оценщик принимает
scripted_ticket = {"title": "KV-кэш 3x512",
                   "body": f"{total} байт [S:ch-04de13a141cc]"}
assert evaluate(T1, [("T-17", scripted_ticket)],
                EXPECTED)["success_state"] == 1
# Ключ идемпотентности содержит номер шага: повтор моделью — новый ключ
def idem_key(run_id, step, tool, args):    # как в разделе 13.4
    text = f"{run_id}|{step}|{tool}|{canonical(args)}"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
assert (idem_key("run-42", 3, "create_ticket", TICKET)
        != idem_key("run-42", 4, "create_ticket", TICKET))

print("trajectory_eval: все проверки пройдены")
