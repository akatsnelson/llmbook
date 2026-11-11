"""Глава 13, раздел 13.5 «План и формат ReAct»: исполнитель плана с перепланированием хвоста.

Проверка плана validate_plan и конструктор шага plan_step взяты из модуля
главы agent.py (листинг «План как данные: проверка до запуска»); здесь —
свой исполнитель execute_plan и маленький мир для вопроса «Два перехода».
В книге тот же опыт ставит листинг «Исполнение плана с перепланированием»
на исполнителе из agent.py; этот файл выполняется в ноутбуке главы
notebooks/13_tools.ipynb.

Модель здесь — две сценарные функции: blind пишет план вслепую, tail —
новый хвост после провала шага E2. Поиск грубый: фрагмент подходит, если
содержит все слова запроса. Политика — заглушка, которая всё разрешает, но
считает, сколько раз её спросили. Ожидаем три вызова модели (план, хвост,
ответ), E1 исполнен один раз, решений политики столько же, сколько вызовов
инструментов. Импорт файла молча прогоняет проверки assert.

Запуск из корня репозитория: python labs/ch13/plan_execute.py
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from agent import REF, plan_step, validate_plan  # noqa: E402

# Два фрагмента из 12_rag.ipynb, текст дословно; остальные поля —
# факты, которые хранил граф знаний той главы.
CORPUS = [
    {"id": "deploy-09", "status": "archived",
     "accelerator": "A100-40GB", "hbm": "40 ГБ",
     "text": "Архивное развёртывание deploy-09 запускало BookLM на "
             "A100-40GB. Эта запись не описывает действующий сервис."},
    {"id": "deploy-17", "status": "active",
     "accelerator": "H100-80GB", "hbm": "80 ГБ",
     "text": "Развёртывание deploy-17 запускает артефакт booklm-rag-v1 "
             "на ускорителе H100-80GB. Ёмкость HBM равна 80 ГБ."}]


def words(text):
    return set(re.findall(r"\w+", text.lower()))


def search_docs(query, k, access):  # access уже проверила политика
    return [d for d in CORPUS if words(query) <= words(d["text"])][:k]


def kv_cache_bytes(tokens, requests, dtype):            # раздел 13.1
    return {"bf16": 1536, "fp32": 3072}[dtype] * tokens * requests


LAB_TOOLS = {"search_docs": search_docs, "kv_cache_bytes": kv_cache_bytes}
log = {"model": 0, "tool": [], "authorize": []}   # счётчики одного прогона


def authorize(tool, args):     # заглушка; настоящая политика — 13.3
    log["authorize"].append(tool)
    return "allow"


def run_plan_step(step, results):   # None — шаг провалился
    args = {k: REF.sub(lambda m: results[m.group(1)], v)
            if isinstance(v, str) else v
            for k, v in step["args"].items()}
    if authorize(step["tool"], args) != "allow":
        return None
    log["tool"].append(step["id"])
    out = LAB_TOOLS[step["tool"]](**args)
    if "take" not in step:
        return out
    active = [d for d in out if d["status"] == "active"]  # без архива
    return active[0][step["take"]] if active else None


def execute_plan(question, planner, replan_tail, max_replans=2):
    log.update(model=1, tool=[], authorize=[])   # вызов, написавший план
    plan, results, i, replans = planner(question), {}, 0, 0
    validate_plan(plan)                  # до первого инструмента
    while i < len(plan):
        out = run_plan_step(plan[i], results)
        if out is not None:
            results[plan[i]["id"]], i = out, i + 1
        elif replans == max_replans:
            return "FAILED", results
        else:                            # переписываем только хвост
            replans, log["model"] = replans + 1, log["model"] + 1
            plan = plan[:i] + replan_tail(question, results, plan[i])
            validate_plan(plan)          # новый хвост — та же проверка
    log["model"] += 1                    # вызов, который пишет ответ
    return "DONE", results


PUB = dict(k=3, access="public")    # одинаковы у всех поисков
KV = dict(tokens=512, requests=1, dtype="bf16")
E1 = plan_step("E1", "search_docs", "accelerator",
               query="развёртывание BookLM", **PUB)


def blind(question):                # план, написанный вслепую
    return [E1,
            plan_step("E2", "search_docs", "hbm",
                      query="паспорт узла #E1", **PUB),
            plan_step("E3", "kv_cache_bytes", **KV)]


def tail(question, results, failed):  # новый хвост после провала E2
    return [plan_step("E2'", "search_docs", "hbm",
                      query="ёмкость HBM #E1", **PUB),
            plan_step("E3'", "kv_cache_bytes", **KV)]


def _checks():
    for bad in ([plan_step("E1", "search_docs", query="#E2", **PUB),  # цикл
                 plan_step("E2", "search_docs", query="#E1", **PUB)],
                [plan_step("E1", "search_docs", query="#E9", **PUB)],
                [plan_step("E1", "rm_index")]):
        try:
            execute_plan("?", lambda q: bad, tail)
        except ValueError:
            assert log["tool"] == []      # ни один инструмент не сработал
        else:
            raise AssertionError("плохой план прошёл проверку")

    status, r = execute_plan("Два перехода", blind, tail)
    assert status == "DONE" and r["E3'"] == 786432
    assert (r["E1"], r["E2'"]) == ("H100-80GB", "80 ГБ")
    assert log["model"] == 3              # план, новый хвост, ответ
    assert log["tool"] == ["E1", "E2", "E2'", "E3'"]  # E1 — один раз
    assert len(log["authorize"]) == len(log["tool"])  # решение на шаг
    assert search_docs("BookLM", **PUB)[0]["id"] == "deploy-09"  # архив


_checks()

if __name__ == "__main__":
    print("plan_execute: все проверки пройдены")
