"""Листинги главы 13 «Агенты и инструменты» книги «Большие языковые модели», файл 04-idempotency-recovery.

Код и вывод — ровно как в книге; текст книги сюда не входит. Листинги идут по порядку и
опираются друг на друга: запускайте файл целиком или ячейками сверху вниз (разметка # %%).
Проверка, что вывод совпал с книгой: python scripts/check_listings.py listings/ch13/04-idempotency-recovery.py
"""

# %% Листинг «Ключ идемпотентности и сервер, который его помнит» — раздел «Ключ идемпотентности»
import sys
sys.path.insert(0, "labs/ch13")          # путь от корня репозитория
from agent import WORLD, reset_world, execute, canonical, fail

import hashlib
import uuid

def idem_key(run_id, step, tool):        # имя намерения
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

# %% Листинг «Потерянный ответ и повтор» — раздел «Ключ идемпотентности»
prop = {"tool": "create_ticket", "arguments": {
    "title": "KV-кэш 3x512",
    "body": "2 359 296 байт [S:booklm-md-v3-0042]"}}
key = idem_key("run-42", 3, "create_ticket")     # до первой попытки
print("ключ:", key[:12])

def two_attempts(make_key):              # ответ на первую попытку потерян
    reset_world()
    server = KeyedServer()
    for attempt in (1, 2):
        obs = server.call(prop, make_key())
    print("заявок", len(WORLD["tickets"]), "ответ", obs["result"])
    return server

print("стабильный ключ: ", end="")
good = two_attempts(lambda: key)
print("uuid4 в цикле:   ", end="")
two_attempts(lambda: uuid.uuid4().hex)
swapped = {"tool": "create_ticket", "arguments": {
    "body": prop["arguments"]["body"], "title": prop["arguments"]["title"]}}
print("поля переставлены:", good.call(swapped, key)["result"])
other = {"tool": "create_ticket",
         "arguments": {**prop["arguments"], "title": "другое"}}
try:
    good.call(other, key)
except ValueError as err:
    print("тот же ключ, другой заголовок:", err)
# Вывод в книге:
# | ключ: 8c42beef502b
# | стабильный ключ: заявок 1 ответ {'ticket_id': 'T-17'}
# | uuid4 в цикле:   заявок 2 ответ {'ticket_id': 'T-18'}
# | поля переставлены: {'ticket_id': 'T-17'}
# | тот же ключ, другой заголовок: ключ занят другим намерением

# %% Листинг «Повтор с экспоненциальной задержкой» — раздел «Повторы и экспоненциальная задержка»
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
            return obs, waited           # повтор тут не поможет
        waited += t_out if code == "timeout" else 0.0
        if i < n:
            waited += b * m ** (i - 1)   # пауза p_i без джиттера
    return fail("timeout", f"исход неизвестен, ключ {key[:12]}"), waited

for fates in (["timeout", "503"], ["timeout"] * 3):
    reset_world()
    obs, waited = call_with_retry(FlakyNet(KeyedServer(), fates), prop, key)
    seen = obs["result"]["ticket_id"] if obs["ok"] else obs["error"]["code"]
    print(f"{' '.join(fates):<24} W={waited} с, заявок",
          len(WORLD["tickets"]), "| модель видит:", seen)
# Вывод в книге:
# | timeout 503              W=11.5 с, заявок 1 | модель видит: T-17
# | timeout timeout timeout  W=31.5 с, заявок 1 | модель видит: timeout

# %% Листинг «Журнал намерений и восстановление после падения» — раздел «Журнал намерений»
class Crash(Exception):                  # процесс агента упал
    pass

def known(obs):                          # исход известен: не тайм-аут
    return obs["ok"] or obs["error"]["code"] != "timeout"

def run_step(log, server, prop, key, crash_at=None):
    if crash_at == "A": raise Crash
    log.append({"type": "intent", "key": key, "prop": prop})
    if crash_at == "B": raise Crash
    obs, _ = call_with_retry(server, prop, key)  # внешний эффект
    if crash_at == "C": raise Crash              # опасное окно
    if known(obs):                       # при тайм-ауте намерение открыто
        log.append({"type": "result", "key": key, "obs": obs})
    if crash_at == "D": raise Crash
    return obs

def recover(log, server, prop_now, run_id="run-42", step=3):
    if not log:                          # о шаге ничего не известно
        key = idem_key(run_id, step, prop_now["tool"])
        return "шаг заново", run_step(log, server, prop_now, key)
    if log[-1]["type"] == "result":      # шаг сделан: не трогаем
        return "пропуск", log[-1]["obs"]
    intent = log[-1]                     # намерение без результата
    obs = run_step(log, server, intent["prop"], intent["key"])
    return "повтор с K", obs

reworded = {"tool": "create_ticket", "arguments":
            {**prop["arguments"], "title": "KV-кэш: 3 запроса по 512"}}
for point in "ABCD":
    reset_world()
    log, server = [], KeyedServer()
    try:
        run_step(log, server, prop, key, crash_at=point)
    except Crash:
        pass                             # процесс перезапущен
    what, obs = recover(log, server, reworded)
    print(f"{point}: {what:<10} заявок {len(WORLD['tickets'])},",
          obs["result"]["ticket_id"], "|", WORLD["tickets"][0]["title"])
# Вывод в книге:
# | A: шаг заново заявок 1, T-17 | KV-кэш: 3 запроса по 512
# | B: повтор с K заявок 1, T-17 | KV-кэш 3x512
# | C: повтор с K заявок 1, T-17 | KV-кэш 3x512
# | D: пропуск    заявок 1, T-17 | KV-кэш 3x512

# %% Листинг «Страж с ключом идемпотентности» — раздел «Страж с ключом и журналом»
from agent import (TOOLS, PermissionLog, PolicyGuard, USER, SESSION,
                   run_agent, scripted)

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
            prop = sent[key]                    # аргументы берём из журнала
        obs = run_step(self.intents, self.server, prop, key, self.crash_at)
        if known(obs):
            self.note(prop, step, "action", None)
        return obs

# %% Листинг «Падение в точке C посреди прогона» — раздел «Страж с ключом и журналом»
reset_world()
server, intents, log = KeyedServer(), [], PermissionLog()
guard = IdempotentGuard(USER, SESSION, log, server, intents)
guard.crash_at = "C"                     # упадём сразу после заявки
try:
    run_agent(scripted, guard=guard)
except Crash:
    print("упали в точке C, заявок:", len(WORLD["tickets"]))

def reword(context):                     # после перезапуска пишет иначе
    return scripted(context).replace("KV-кэш 3x512", "KV-кэш: 3 по 512")

guard.crash_at = None                    # перезапуск, журналы те же
state, traj = run_agent(reword, guard=guard)
print(state, "заявок:", len(WORLD["tickets"]), "|",
      WORLD["tickets"][0]["title"])
print("шаг 3 в журнале:",
      [r["type"] for r in log.records if r["step"] == 3])
print("журнал разрешений цел:", log.verify())
# Вывод в книге:
# | упали в точке C, заявок: 1
# | DONE заявок: 1 | KV-кэш 3x512
# | шаг 3 в журнале: ['decision', 'decision', 'replaced', 'action']
# | журнал разрешений цел: True

# %% Листинг «Тайм-аут, а потом та же просьба на новом шаге» — раздел «Страж с ключом и журналом»
reset_world()
net = FlakyNet(KeyedServer(), ["timeout"] * 3)   # три ответа потеряны
guard = IdempotentGuard(USER, SESSION, PermissionLog(), net, [])
print("шаг 3:", guard.act(prop, 3)["error"]["code"])
again = {"tool": "create_ticket", "arguments":
         {**prop["arguments"], "title": "KV-кэш 3x512 (повтор)"}}
print("шаг 4:", guard.act(again, 4)["result"], "| заявок:",
      len(WORLD["tickets"]))
# Вывод в книге:
# | шаг 3: timeout
# | шаг 4: {'ticket_id': 'T-17'} | заявок: 1

# %% Листинг «BookLM пишет заголовок до и после перезапуска» — раздел «Страж с ключом и журналом»
import json
from booklm.lab import load
from booklm.experiments.ch13 import booklm_title   # жадно, 12 токенов
tok, lm = load("instruct")

def booklm_step3(note):                  # шаг 3 пишет BookLM
    def model(context):
        raw = scripted(context)
        if len(context) != 2:
            return raw
        title = json.dumps(booklm_title(tok, lm, note), ensure_ascii=False)
        return raw.replace('"KV-кэш 3x512"', title)
    return model

print("до падения:    ", booklm_title(tok, lm, ""))
print("после рестарта:", booklm_title(tok, lm, "Процесс перезапущен. "))
reset_world()
log, intents = PermissionLog(), []
guard = IdempotentGuard(USER, SESSION, log, KeyedServer(), intents)
guard.crash_at = "C"
try:
    run_agent(booklm_step3(""), guard=guard)
except Crash:
    pass
guard.crash_at = None
state, _ = run_agent(booklm_step3("Процесс перезапущен. "), guard=guard)
print(state, "заявок:", len(WORLD["tickets"]))
print("на сервере:", WORLD["tickets"][0]["title"])
# Вывод в книге:
# | до падения:     Жил-был в маленьком городе мальчик по имени Том. Том
# | после рестарта: Жил-был маленький девочка по имени Лили. Она любила играть
# | DONE заявок: 1
# | на сервере: Жил-был в маленьком городе мальчик по имени Том. Том
