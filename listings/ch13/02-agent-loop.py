"""Листинги главы 13 «Агенты и инструменты» книги «Большие языковые модели», файл 02-agent-loop.

Код и вывод — ровно как в книге; текст книги сюда не входит. Листинги идут по порядку и
опираются друг на друга: запускайте файл целиком или ячейками сверху вниз (разметка # %%).
Проверка, что вывод совпал с книгой: python scripts/check_listings.py listings/ch13/02-agent-loop.py
"""

# %% Листинг «Наивный цикл и три его поломки» — раздел «Цикл агента как конечный автомат»
import json

tickets = []                     # внешний мир: заявки на сервере

def naive_dispatch(proposal):    # исполняет всё, похожее на вызов
    call = json.loads(proposal)
    if call.get("tool") == "create_ticket":
        tickets.append(call["arguments"]["title"])
        return f"создана T-{16 + len(tickets)}"
    return "непонятное предложение"

def unsure_model(context):       # «а вдруг не создалась?»
    return ('{"tool": "create_ticket",'
            ' "arguments": {"title": "KV-кэш 3x512"}}')

context = []
for _ in range(10):              # без range(10) был бы while True
    proposal = unsure_model(context)
    context.append((proposal, naive_dispatch(proposal)))
print("заявок:", len(tickets), "последняя:", context[-1][1])
print(naive_dispatch('{"final": "Готово"}'))
# Вывод в книге:
# | заявок: 10 последняя: создана T-26
# | непонятное предложение

# %% Листинг «Строки контекста в токенах BookLM» — раздел «Бюджет контекста»
import json
import sys
sys.path.insert(0, "labs/ch13")      # путь от корня репозитория
from agent import dispatch, scripted, reset_world
from booklm.lab import load_tokenizer
from booklm.experiments.ch13 import TASK, CALL_RULES, SIGNATURES, describe
tok = load_tokenizer()

def n_tok(text):                     # сколько токенов BookLM в строке
    return len(tok.encode(text).ids)
names = ["search_docs", "kv_cache_bytes", "create_ticket"]
print("со схемой: ", [n_tok(describe(x)) for x in names])
print("сигнатуры: ", [n_tok(s) for s in SIGNATURES])
print("правила:", n_tok(CALL_RULES), " вопрос:", n_tok(TASK))
reset_world()
context = []
for _ in range(3):                   # три шага сквозной задачи
    raw = scripted(context)
    context.append((raw, dispatch(raw)))
steps = [n_tok(raw) + n_tok(json.dumps(obs, ensure_ascii=False))
         for raw, obs in context]
print("шаги:      ", steps)
print("ответ:     ", n_tok(scripted(context)))
pieces = [tok.decode([i]) for i in tok.encode(context[1][0]).ids]
print("|".join(pieces[:18]))
# Вывод в книге:
# | со схемой:  [308, 329, 235]
# | сигнатуры:  [44, 56, 31]
# | правила: 55  вопрос: 62
# | шаги:       [193, 137, 165]
# | ответ:      56
# | {|"|ar|g|u|m|e|n|t|s|"|:|{|"|d|t|y|p

# %% Листинг «Сколько шагов помещается в окно» — раздел «Бюджет контекста»
def steps_that_fit(t_max=512, rules=55, tools=(44, 56, 31), question=62,
                   reserve=56, step=165):   # токены BookLM
    fixed = rules + sum(tools) + question + reserve
    return max(0, (t_max - fixed) // step)

print("сигнатуры:         ", steps_that_fit())
print("описания со схемой:", steps_that_fit(tools=(308, 329, 235)))
print("то же, окно 128K:  ", steps_that_fit(t_max=131072,
                                            tools=(308, 329, 235)))
# Вывод в книге:
# | сигнатуры:          1
# | описания со схемой: 0
# | то же, окно 128K:   788

# %% Листинг «Код раздела sec:agents-tools и таблица переходов» — раздел «Реализация автомата»
import json
import sys
sys.path.insert(0, "labs/ch13")      # путь от корня репозитория
from agent import (TOOLS, WORLD, reset_world, parse, check, execute, show,
                   final_answer)

def canonical(obj):                  # канонический JSON главы
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)

ALLOWED = {                          # разрешённые переходы автомата
    ("PROPOSE", "PARSE"),
    ("PARSE", "VALIDATE"), ("PARSE", "DONE"), ("PARSE", "OBSERVE"),
    ("VALIDATE", "AUTHORIZE"), ("VALIDATE", "OBSERVE"),
    ("VALIDATE", "FAILED"),
    ("AUTHORIZE", "ACT"), ("AUTHORIZE", "OBSERVE"),
    ("AUTHORIZE", "WAIT_HUMAN"),
    ("WAIT_HUMAN", "AUTHORIZE"), ("WAIT_HUMAN", "OBSERVE"),
    ("ACT", "OBSERVE"), ("OBSERVE", "PROPOSE"), ("OBSERVE", "BUDGET")}

class Guard:                         # страж AUTHORIZE и ACT: пускает всё
    def decide(self, prop, step):    # None — можно, иначе наблюдение-отказ
        return None

    def act(self, prop, step):
        return execute(prop)

print(*sorted(TOOLS))
print(final_answer('{"final": "Готово"}'))
print(canonical({"tokens": 512, "requests": 3}))
print(canonical({"requests": 3, "tokens": 512}))
print(sorted(a for a, b in ALLOWED if b == "ACT"))
# Вывод в книге:
# | create_ticket delete_document kv_cache_bytes reserve_gpu search_docs
# | Готово
# | {"requests":3,"tokens":512}
# | {"requests":3,"tokens":512}
# | ['AUTHORIZE']

# %% Листинг «Конечный автомат агентного цикла» — раздел «Реализация автомата»
def run_agent(model, n_max=4, repeat_limit=2, guard=Guard()):
    traj, context, state, last, same = [], [], "PROPOSE", None, 0

    def go(new, **info):             # единственный способ сменить состояние
        nonlocal state
        assert (state, new) in ALLOWED, (state, new)
        traj.append({"step": step, "to": new, **info})
        state = new

    for step in range(1, n_max + 1):
        if step > 1:
            go("PROPOSE")
        raw = model(context)         # модель работает только здесь
        go("PARSE")
        answer = final_answer(raw)
        if answer is not None:
            go("DONE", answer=answer)
            return state, traj
        prop, obs = parse(raw)
        if prop is None:
            last, same = None, 0     # битый текст прерывает серию
        else:
            go("VALIDATE")
            key = canonical(prop)
            if key == last:          # тот же вызов, что на прошлом шаге
                same += 1
            else:
                same, last = 1, key
            if same >= repeat_limit:
                go("FAILED", reason="повтор")
                return state, traj
            obs = check(prop)
            if obs is None:
                go("AUTHORIZE")
                obs = guard.decide(prop, step)
                if obs is None:
                    go("ACT", tool=prop["tool"])
                    obs = guard.act(prop, step)
        code = "ok" if obs["ok"] else obs["error"]["code"]
        go("OBSERVE", code=code)
        context.append((raw, obs))
    go("BUDGET")
    return state, traj

# %% Листинг «Сквозная задача со сценарной заглушкой» — раздел «Реализация автомата»
def call(tool, **arguments):
    return canonical({"tool": tool, "arguments": arguments})

def scripted(context):               # сценарная заглушка: шаг -> ход
    if len(context) == 0:
        return call("search_docs", query="KV-кэш BookLM байт на токен",
                    k=3, access="public")
    if len(context) == 1:
        return call("kv_cache_bytes", tokens=512, requests=3,
                    dtype="bf16")
    chunk = context[0][1]["result"]["chunks"][0]["chunk_id"]
    total = f"{context[1][1]['result']['bytes']:,}".replace(",", " ")
    if len(context) == 2:
        return call("create_ticket", title="KV-кэш 3x512",
                    body=f"{total} байт [S:{chunk}]")
    ticket = context[2][1]["result"]["ticket_id"]
    answer = f"Заявка {ticket}: {total} байт [S:{chunk}]"
    return canonical({"final": answer})

reset_world()
state, traj = run_agent(scripted)
show(traj)
print(state, "заявок:", len(WORLD["tickets"]))
# Вывод в книге:
# | 1 PARSE VALIDATE AUTHORIZE ACT OBSERVE | search_docs ok
# | 2 PARSE VALIDATE AUTHORIZE ACT OBSERVE | kv_cache_bytes ok
# | 3 PARSE VALIDATE AUTHORIZE ACT OBSERVE | create_ticket ok
# | 4 PARSE DONE | - Заявка T-17: 2 359 296 байт [S:booklm-md-v3-0042]
# | DONE заявок: 1

# %% Листинг «Четыре заглушки против одного автомата» — раздел «Реализация автомата»
def stubborn(context):               # упрямая: один и тот же вызов
    return call("create_ticket", title="KV-кэш 3x512", body="?")

def sloppy(context):                 # неряшливая: JSON всегда оборван
    return '{"tool": "search_docs", '

def wander(context):                 # блуждающая: каждый раз новый запрос
    return call("search_docs", query=f"попытка {len(context)}", k=3,
                access="public")

for name, model in [("сценарная", scripted), ("упрямая", stubborn),
                    ("неряшливая", sloppy), ("блуждающая", wander)]:
    reset_world()
    state, traj = run_agent(model)
    model_calls = sum(t["to"] == "PARSE" for t in traj)
    print(f"{name:10} {state:7} заявок: {len(WORLD['tickets'])}"
          f"  вызовов модели: {model_calls}")
# Вывод в книге:
# | сценарная  DONE    заявок: 1  вызовов модели: 4
# | упрямая    FAILED  заявок: 1  вызовов модели: 2
# | неряшливая BUDGET  заявок: 0  вызовов модели: 4
# | блуждающая BUDGET  заявок: 0  вызовов модели: 4

# %% Листинг «Повтор, разорванный битым JSON» — раздел «Реализация автомата»
reset_world()
x = stubborn(None)
moves = iter([x, "{", x, x])
state, traj = run_agent(lambda context: next(moves))
show(traj)
print(state, "заявок:", len(WORLD["tickets"]))
# Вывод в книге:
# | 1 PARSE VALIDATE AUTHORIZE ACT OBSERVE | create_ticket ok
# | 2 PARSE OBSERVE | - parse_error
# | 3 PARSE VALIDATE AUTHORIZE ACT OBSERVE | create_ticket ok
# | 4 PARSE VALIDATE FAILED | - повтор
# | FAILED заявок: 2

# %% Листинг «BookLM в цикле агента» — раздел «Реализация автомата»
import json
import sys
sys.path.insert(0, "labs/ch13")      # путь от корня репозитория
from agent import WORLD, reset_world, run_agent, show
from booklm.lab import load, generate, prompt_ids
from booklm.experiments.ch13 import TASK, fill_ticket
tok, model = load("instruct")

def booklm(context, constrained=False):      # модель для run_agent
    text = TASK + "".join("\n" + json.dumps(obs, ensure_ascii=False)
                          for _, obs in context)
    if constrained:                          # шаблон пишет программа
        return fill_ticket(tok, model, prompt_ids(tok, text, chat=True))
    return generate(tok, model, text, chat=True, max_new=60, temperature=0)

print(booklm([])[:60])
for constrained in (False, True):
    reset_world()
    state, traj = run_agent(lambda context: booklm(context, constrained))
    show(traj)
    print(state, "заявок:", len(WORLD["tickets"]))
titles = [t["title"] for t in WORLD["tickets"]]
print("разных заголовков:", len(set(titles)))
for title in titles[:2]:
    print(repr(title[:50]))
# Вывод в книге:
# | Жил-был маленький девочка по имени Мия. У Мии был большой, к
# | 1 PARSE OBSERVE | - parse_error
# | 2 PARSE OBSERVE | - parse_error
# | 3 PARSE OBSERVE | - parse_error
# | 4 PARSE OBSERVE BUDGET | - parse_error
# | BUDGET заявок: 0
# | 1 PARSE VALIDATE AUTHORIZE ACT OBSERVE | create_ticket ok
# | 2 PARSE VALIDATE AUTHORIZE ACT OBSERVE | create_ticket ok
# | 3 PARSE VALIDATE AUTHORIZE ACT OBSERVE | create_ticket ok
# | 4 PARSE VALIDATE AUTHORIZE ACT OBSERVE BUDGET | create_ticket ok
# | BUDGET заявок: 4
# | разных заголовков: 4
# | ' cimed cimed cimed cimed cimed cimed cimed cimed c'
# | ' cimed to clas a a a a a a a a a a a a a a a a a a'

# %% Листинг «Мысли агента в окне BookLM» — раздел «Рассуждение между вызовами инструментов»
from booklm.experiments.ch13_interleaved import THOUGHTS
thoughts = [n_tok(t) for t in THOUGHTS]      # три вызова и ответ
steps = [193, 137, 165]                      # шаги без мыслей, вывод выше
with_t = [s + t for s, t in zip(steps, thoughts)]
calls = [248 + sum(with_t[:i]) for i in range(4)]
print("мысли:", thoughts, " шаги с мыслями:", with_t)
print("входы с мыслями:", calls, "всего", sum(calls))
print("шагов в окне 512:", steps_that_fit(reserve=56 + thoughts[3],
                                         step=sum(with_t) // 3))
print("окно 128K, мысль 1000:", steps_that_fit(t_max=131072,
      tools=(308, 329, 235), reserve=56 + 1000, step=165 + 1000))
# Вывод в книге:
# | мысли: [43, 50, 40, 35]  шаги с мыслями: [236, 187, 205]
# | входы с мыслями: [248, 484, 671, 876] всего 2279
# | шагов в окне 512: 0
# | окно 128K, мысль 1000: 110
