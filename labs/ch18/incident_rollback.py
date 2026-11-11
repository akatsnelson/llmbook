"""Глава 18, раздел 18.6 «Защита, оценка и реагирование», подраздел «Реагирование на инцидент и откат»: откат по сигналу безопасности (полная версия).

Из книги «Большие языковые модели».
Набор атак, диспетчеры и ворота выпуска берутся из модуля главы guard.py; файл выполняется в
ноутбуке главы notebooks/18_security_boundaries.ipynb.

Ворота выпуска release_ok, поиск по журналу версии политики, пропустившей опасное действие,
адрес отката по родословной и правило для начатых запросов. У guarded-v1 и guarded-v2 одни веса
и один шаблон диалога, поэтому KV-кэш начатого запроса годится и для v1 и остаётся на месте.
Опасны действия, которые v2 уже одобрила, но исполнитель ещё не выполнил: их снимают из очереди,
заново проводят через диспетчер v1 и сохраняют ключ идемпотентности исходного запуска, чтобы
действие, успевшее исполниться, не повторилось.

Запуск из корня репозитория: python labs/ch18/incident_rollback.py
"""

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from guard import (CASES, FULL, Trust, authorize, no_origin_check, regression_suite,  # noqa: E402
                   release_ok)

CASE = {c: (t, a, ctx, k, want) for c, t, a, ctx, k, want in CASES}
RISKY = {"send_email", "delete_document", "create_ticket", "reserve_gpu", "fetch_url"}

# --- ворота выпуска: сервис зелёный у обеих версий, набор атак различает их ---
POLICY = {"v1": authorize, "v2": no_origin_check}             # v2 потеряла проверку
assert regression_suite(POLICY["v1"]) == FULL
assert regression_suite(POLICY["v2"]) != FULL
assert release_ok(18, 3600, POLICY["v1"]) and not release_ok(18, 3600, POLICY["v2"])
assert not release_ok(18, 3600, POLICY["v1"], load_ok=False)  # загрузка не прошла

# Суммы условные, как "w-3f2a": здесь важны имена родителей и равенство весов у v1 и v2.
LINEAGE = {"v1": {"id": "booklm-guarded-v1","parents": {"booklm-tools-v1": "m-91c4"},
                  "weights_sha256": "w-3f2a", "template": "booklm-chat-v1"},
           "v2": {"id": "booklm-guarded-v2", "parents": {"booklm-guarded-v1": "m-5e07"},
                  "weights_sha256": "w-3f2a", "template": "booklm-chat-v1"}}


def decision(version, case):
    tool, args, ctx, _, _ = CASE[case]
    return POLICY[version]({"tool": tool, "arguments": args}, ctx)[0]


# --- журнал: кто пропустил опасное действие из недоверенного контекста ---
journal = [{"run_id": f"run-{i}", "step": 3, "case": case, "tool": CASE[case][0],
            "taint": CASE[case][2]["taint"], "decision": decision(v, case),
            "policy_version": v}
           for i, (v, case) in enumerate([("v1", "A1"), ("v2", "A1"), ("v2", "A2"),
                                          ("v2", "B5")])]
bad = {r["policy_version"] for r in journal
       if r["tool"] in RISKY and r["taint"] is Trust.UNTRUSTED and r["decision"] != "deny"}
assert bad == {"v2"}
(target,) = LINEAGE["v2"]["parents"]           # у v2 один родитель
assert target == LINEAGE["v1"]["id"]
assert regression_suite(POLICY["v1"]) == FULL                 # родитель проверен


# --- начатые запросы и очередь действий ---
def idem_key(run_id, step, tool):                # как в главе 13: прогон, шаг, инструмент
    return hashlib.sha256(f"{run_id}|{step}|{tool}".encode()).hexdigest()[:12]


def on_incident(request, queue):
    """Переводит начатый запрос на v1 и перепроверяет очередь действий диспетчером v1."""
    same_weights = (LINEAGE["v1"]["weights_sha256"] == LINEAGE["v2"]["weights_sha256"]
                    and LINEAGE["v1"]["template"] == LINEAGE["v2"]["template"])
    if not same_weights:                         # кэш построили другие веса: заново
        request = dict(request, kv=None)
    request = dict(request, executor="v1")
    kept = []
    for item in queue:
        if item["policy_version"] == "v2":       # одобрено уязвимой версией
            verdict = decision("v1", item["case"])
            if verdict == "deny":
                continue                         # v1 не разрешает: отменяем
            item = dict(item, policy_version="v1", decision=verdict)  # ask: снова к человеку
        kept.append(item)                        # ключ идемпотентности тот же
    return request, kept


started = {"run_id": "run-42", "executor": "v2", "kv": "кэш 400 токенов"}
queue = [{"case": "A1", "policy_version": "v2", "idem": idem_key("run-42", 3, "delete_document")},
         {"case": "B4", "policy_version": "v2", "idem": idem_key("run-42", 4, "send_email")}]
request, kept = on_incident(started, queue)
assert request["executor"] == "v1"
assert request["kv"] == "кэш 400 токенов"       # веса те же: кэш годится и для v1
assert [k["case"] for k in kept] == ["B4"]      # удаление из помеченной сессии снято
assert kept[0]["decision"] == "ask"             # письмо снова ждёт подписи
assert kept[0]["idem"] == queue[1]["idem"]      # повтор не создаст второе письмо

# если бы v2 меняла веса, кэш пришлось бы выбросить (глава 17)
LINEAGE["v2"]["weights_sha256"] = "w-9c1d"
assert on_incident(started, [])[0]["kv"] is None
LINEAGE["v2"]["weights_sha256"] = "w-3f2a"

print("OK:", __file__)
