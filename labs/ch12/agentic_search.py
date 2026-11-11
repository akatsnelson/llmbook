"""Глава 12, раздел 12.8 «Многошаговый, иерархический и графовый поиск», подраздел «Агентный поиск»:
многошаговый поисковый цикл над корпусом DOCS.

Файл повторяет три листинга подраздела (каталог карт и один проход поиска, заглушка facts_of, цикл
research с правилом остановки) и проверяет assert каждое число, которое они печатают. Затем ставит
опыт пошире, которого в книге нет листингом:

  * сколько из четырёх нужных фактов приносит один проход retrieve при k от 1 до 10;
  * сколько поисков и фрагментов стоит цикл при разных k (фрагментов на подзапрос) и patience;
  * сколько раундов нужно, если независимые подзапросы отправлять разом (задача главы).

Модели здесь нет: какие факты уже найдены и какой подзапрос следующий, решает детерминированная
заглушка facts_of, как и в книге. Функции поиска (bm25, retrieve, pack, validate_citations) берутся из
labs/ch12/rag_checks.py, где они совпадают с листингами главы.

Запуск из корня репозитория: python labs/ch12/agentic_search.py
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rag_checks as rc  # noqa: E402  (печатает строку о своих проверках)

DOCS, EMB, ACL = rc.DOCS, rc.EMB, rc.ACL
bm25, by_rank, retrieve, pack = rc.bm25, rc.by_rank, rc.retrieve, rc.pack
validate_citations = rc.validate_citations

# --- листинг «Один проход на вопросе в несколько шагов» ----------------------------------------------

CARDS = {"hw-catalog-v1-0001": "Карта A100: 40 ГБ памяти HBM.",
         "hw-catalog-v1-0002": "Карта L4: 24 ГБ памяти GDDR6.",
         "hw-catalog-v1-0003": "Карта H100: 80 ГБ памяти HBM."}
for d, text in CARDS.items():
    DOCS[d], EMB[d], ACL[d] = text, [2, 0, 2], {"all"}

HOPS = {"ops-runbook-v7-0007": "карта узла", "hw-catalog-v1-0003": "память H100",
        "booklm-md-v3-0043": "веса", "booklm-md-v3-0042": "KV-кэш"}
HARD_Q = "Хватит ли памяти ускорителя, на котором работает BookLM, на её веса и KV-кэш?"
QVEC = [2, 1, 1]

assert [HOPS.get(d, d) for d in pack(HARD_Q, QVEC, {"all"})] == ["KV-кэш", "веса", "карта узла"]
top8 = [d for d, _ in retrieve(HARD_Q, QVEC, {"all"}, k=8)]
assert [sum(d in HOPS for d in top8[:k]) for k in (3, 5, 8)] == [2, 3, 4]
assert top8[4] == "hw-catalog-v1-0001"                 # пятым пришла карта A100, а не H100
assert top8.index("hw-catalog-v1-0003") == 7           # нужная карта только восьмая
# три строки каталога для вопроса неразличимы: одна оценка BM25 и один вектор
s = bm25(HARD_Q, {d: DOCS[d] for d in DOCS if "all" in ACL[d]})
assert len({round(s[d], 9) for d in CARDS}) == 1


# --- листинг «Заглушка вместо модели» -------------------------------------------------------------

def facts_of(subject, notes):
    """Заглушка вместо модели: какие факты уже есть в заметках."""
    def first(test):
        return next((d for d in notes if test(DOCS[d])), None)
    card_of = lambda t: re.search(r" с ([A-Z]\d+)\.", t)  # noqa: E731
    bridge = first(lambda t: subject in t and card_of(t))
    card = card_of(DOCS[bridge])[1] if bridge else "?"
    return {
        f"на каком узле работает {subject}": bridge,
        f"Карта {card} память": first(lambda t: t.startswith(f"Карта {card}:")),
        f"веса {subject}": first(lambda t: "Веса" in t and subject in t),
        f"KV-кэш {subject}": first(lambda t: "KV-кэш" in t and subject in t),
    }


partial = facts_of("BookLM", ["ops-runbook-v7-0007", "booklm-md-v3-0043"])
assert [q for q, d in partial.items() if d is None] == ["Карта H100 память", "KV-кэш BookLM"]
assert [q for q, d in facts_of("BookLM", []).items() if d is None][:2] == [
    "на каком узле работает BookLM", "Карта ? память"]


# --- листинг «Поисковый цикл с правилом остановки» ------------------------------------------------

def research(subject, groups, max_steps=6, k=2, patience=2, log=None):
    visible = {d: DOCS[d] for d in DOCS if ACL[d] & groups}
    notes, idle, facts = [], 0, None
    for step in range(1, max_steps + 1):
        facts = facts_of(subject, notes)
        missing = [query for query, d in facts.items() if d is None]
        if not missing:
            return "все факты найдены", notes, facts
        found = by_rank(bm25(missing[0], visible))[:k]
        new = [d for d in found if d not in notes]
        if log is not None:
            log.append((step, missing[0], new))
        notes += new
        idle = 0 if new else idle + 1
        if idle == patience:
            return "нет новых фрагментов", notes, facts
    return "бюджет шагов", notes, facts


log = []
stop, notes, _ = research("TinyLM", {"all"}, log=log)
assert stop == "нет новых фрагментов" and notes == ["ops-runbook-v7-0007"]
assert [new for _, _, new in log] == [["ops-runbook-v7-0007"], [], []]

log = []
stop, notes, facts = research("BookLM", {"all"}, log=log)
assert stop == "все факты найдены" and len(notes) == 5
assert [(q, new) for _, q, new in log] == [
    ("на каком узле работает BookLM", ["ops-runbook-v7-0007", "booklm-md-v3-0043"]),
    ("Карта H100 память", ["hw-catalog-v1-0003"]),
    ("KV-кэш BookLM", ["booklm-md-v3-0042", "help-v1-0019"])]
assert set(facts.values()) == set(HOPS)
answer = " ".join(DOCS[d].rstrip(".") + f" [S:{d}]." for d in facts.values())
assert validate_citations(answer, notes) == []
assert validate_citations(answer + " Памяти хватает с запасом.", notes) == [
    "без источника: Памяти хватает с запасом."]
# права проверяются до поиска: для отдела кадров цикл тоже находит все четыре факта
assert research("BookLM", {"all", "hr"})[0] == "все факты найдены"
# вывод «хватает» — арифметика по четырём фактам: 12,63 МБ + 768 КБ против 80 ГБ
assert 12.63 * 2**20 + 768 * 2**10 < 80 * 2**30

# --- задачи главы ---------------------------------------------------------------------------------

# patience=1: TinyLM останавливается после двух поисков вместо трёх
log = []
research("TinyLM", {"all"}, patience=1, log=log)
assert len(log) == 2
# k=1: цикл не получает веса попутно и делает четыре поиска
log = []
stop, notes, _ = research("BookLM", {"all"}, k=1, log=log)
assert stop == "все факты найдены" and len(log) == 4 and len(notes) == 4
# max_steps=2: бюджет кончается раньше, чем найден KV-кэш
stop, notes, facts = research("BookLM", {"all"}, max_steps=2)
assert stop == "бюджет шагов" and facts["KV-кэш BookLM"] is None


def research_rounds(subject, groups, k=2, max_rounds=4):
    """Все недостающие подзапросы раунда — разом; возвращает число раундов и поисков."""
    visible = {d: DOCS[d] for d in DOCS if ACL[d] & groups}
    notes, searches = [], 0
    for rnd in range(1, max_rounds + 1):
        missing = [q for q, d in facts_of(subject, notes).items() if d is None]
        missing = [q for q in missing if "?" not in q]     # подзапрос, ждущий моста, не готов
        if not missing:
            return rnd - 1, searches
        for q in missing:
            searches += 1
            notes += [d for d in by_rank(bm25(q, visible))[:k] if d not in notes]
    return max_rounds, searches


assert research_rounds("BookLM", {"all"}) == (2, 4)

# подзапрос без моста: «сколько памяти у этой карты» не отличает H100 от A100 и L4
visible = {d: DOCS[d] for d in DOCS if "all" in ACL[d]}
s = bm25("сколько памяти у этой карты", visible)
assert by_rank(s) == list(CARDS) and len({round(s[d], 9) for d in CARDS}) == 1
_facts_of = facts_of


def facts_of(subject, notes):  # noqa: F811 — тот же facts_of, но второй подзапрос без имени карты
    return {("сколько памяти у этой карты" if q.startswith("Карта") else q): d
            for q, d in _facts_of(subject, notes).items()}


log = []
stop, notes, _ = research("BookLM", {"all"}, log=log)
assert stop == "нет новых фрагментов" and len(log) == 4 and "hw-catalog-v1-0003" not in notes
assert log[1][2] == ["hw-catalog-v1-0001", "hw-catalog-v1-0002"]
facts_of = _facts_of

# --- опыт: один проход против цикла ---------------------------------------------------------------

print("один проход retrieve: сколько из 4 фактов среди первых k")
for k in range(1, 11):
    got = [d for d, _ in retrieve(HARD_Q, QVEC, {"all"}, k=k)]
    print(f"  k = {k:2}: фактов {sum(d in HOPS for d in got)}, лишних {sum(d not in HOPS for d in got)}")

print("цикл research: поисков / прочитано фрагментов / итог")
for k in (1, 2, 3):
    log = []
    stop, notes, facts = research("BookLM", {"all"}, k=k, log=log)
    print(f"  BookLM, k = {k}: {len(log)} / {len(notes)} / {stop}")
for patience in (1, 2, 3):
    log = []
    stop, notes, facts = research("TinyLM", {"all"}, patience=patience, log=log)
    print(f"  TinyLM, patience = {patience}: {len(log)} / {len(notes)} / {stop}")
print("раунды с параллельными подзапросами (раундов, поисков):", research_rounds("BookLM", {"all"}))
print("labs/ch12/agentic_search.py: все проверки прошли")
