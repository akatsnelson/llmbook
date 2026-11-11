"""Глава 13 «Агенты и инструменты»: опыты на BookLM для рубрик «А теперь на нашей модели».

Опыты (время — на процессоре ноутбука Apple M2 Max):
  ch13_special_tokens   рубрика «служебный токен», подраздел «Предложение вызова»; несколько секунд.
                        Номера токенов служебной метки, набранной пользователем, при обычном кодировании
                        и через encode_text; честный и подложный промпт диалога.
  ch13_context_budget   рубрика «бюджет контекста», подраздел «Бюджет контекста»; несколько секунд.
                        Токены описаний инструментов со схемой и сигнатур, правил, вопроса, трёх шагов
                        сквозной задачи и ответа; как токенизатор режет строку вызова.
  ch13_booklm_in_loop   рубрика «цикл агента», подраздел «Реализация автомата»; около 15 с.
                        Траектории автомата, когда предложения пишет BookLM: свободно и по шаблону
                        create_ticket; итоговое состояние, число заявок и их заголовки.
  ch13_regenerate_args  рубрика «ключ идемпотентности», подраздел «Страж с ключом и журналом»;
                        несколько секунд. Заголовок заявки, который BookLM пишет до падения и после
                        перезапуска, итог стража: одна заявка и шаг, на котором записано расхождение.
  ch13_react_tokens     рубрика «цена траектории в токенах», подраздел «Цена в вызовах и токенах»;
                        несколько секунд. Токены входа каждого вызова у ReAct и у плана, сумма и сколько
                        вызовов помещается в окно 512 токенов.
  ch13_pass_hat_k       рубрика «pass^k», подраздел «Надёжность и pass^k»; около 20 с. Успехи на пяти
                        задачах по десяти формулировкам, pass^k и pass@k при k = 1, 2, 3, 5, начало первого ответа.

Запуск из корня репозитория:
    python -m booklm.experiments.ch13                       # все опыты по очереди
    python -m booklm.experiments.ch13 ch13_booklm_in_loop   # один опыт
Вывод каждого опыта копируется в models/booklm/experiments/<имя опыта через дефис>.txt.

Код автомата, инструментов и метрик берём из модуля главы labs/ch13/agent.py. Ещё два опыта
главы — в ch13_interleaved.py и ch13_tool_search.py.
"""
import json
import re
import sys
from pathlib import Path

import torch

if not __package__:  # запуск файлом (python booklm/experiments/ch13.py): корень репозитория — в путь поиска
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from booklm.lab import MODELS, generate, load, load_tokenizer, prompt_ids, special

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "labs" / "ch13"))
from agent import (TOOLS, WORLD, dispatch, pass_at_k, pass_hat_k,  # noqa: E402
                   reset_world, run_agent, show)

OUT = MODELS / "experiments"

# Вопрос сквозной задачи главы и предложение вызова из подраздела «Предложение вызова».
TASK = ("Найди, сколько байт KV-кэша BookLM приходится на токен, посчитай память "
        "для трёх запросов по 512 токенов и заведи заявку с результатом.")
KV_CALL = ('{"tool": "kv_cache_bytes", "arguments": '
           '{"tokens": 512, "requests": 3, "dtype": "bf16"}}')
# Правила формата и компактные сигнатуры трёх инструментов (подраздел «Бюджет контекста»).
CALL_RULES = 'Отвечай одной строкой JSON: {"tool": имя, "arguments": {...}} или {"final": ответ}.'
SIGNATURES = ["search_docs(query, k, access): поиск по документации BookLM",
              "kv_cache_bytes(tokens, requests, dtype): байты KV-кэша BookLM",
              "create_ticket(title, body): завести заявку"]


def describe(name):
    """Описание инструмента, как его кладёт в контекст шаблон: имя, описание, схема входа."""
    spec = TOOLS[name]
    return json.dumps({"name": spec.name, "description": spec.description,
                       "input_schema": spec.input_schema}, ensure_ascii=False)


def n_tok(tok, text):
    return len(tok.encode(text).ids)


# ----------------------------------------------------------------------------- служебный токен

def ch13_special_tokens():
    """Листинг книги «Служебная метка, набранная пользователем»."""
    from booklm.lab import encode_text
    tok = load_tokenizer()
    typed = "<|user|>Привет"                     # это набрал пользователь
    print("по умолчанию:", tok.encode(typed).ids)
    print("encode_text: ", encode_text(tok, typed))
    print("куски:       ", [tok.decode([i]) for i in encode_text(tok, typed)])
    print("чат, честно: ", prompt_ids(tok, "Привет", chat=True))
    print("чат, подлог: ", prompt_ids(tok, typed, chat=True))


# ----------------------------------------------------------------------------- бюджет контекста

def ch13_context_budget():
    """Листинг книги «Строки контекста в токенах BookLM»."""
    tok = load_tokenizer()
    names = ["search_docs", "kv_cache_bytes", "create_ticket"]
    print("со схемой: ", [n_tok(tok, describe(x)) for x in names])
    print("сигнатуры: ", [n_tok(tok, s) for s in SIGNATURES])
    print("правила:", n_tok(tok, CALL_RULES), " вопрос:", n_tok(tok, TASK))
    from agent import scripted
    reset_world()
    context = []
    for _ in range(3):                           # три шага сквозной задачи
        raw = scripted(context)
        context.append((raw, dispatch(raw)))
    print("шаги:      ", [n_tok(tok, raw) + n_tok(tok, json.dumps(obs, ensure_ascii=False))
                          for raw, obs in context])
    print("ответ:     ", n_tok(tok, scripted(context)))
    print("|".join(tok.decode([i]) for i in tok.encode(context[1][0]).ids[:18]))


# ----------------------------------------------------------------------------- BookLM в цикле агента

@torch.no_grad()
def greedy_string(tok, model, ids, max_chars):
    """Жадное ограниченное декодирование одной строки JSON. Запрещены кавычка, обратная косая
    черта, управляющие символы (с переводом строки) и служебные токены; <eos> закрывает строку."""
    eos = special(tok)["<eos>"]
    banned = [i for i in range(tok.get_vocab_size())
              if i != eos and (i < 5 or re.search(r'["\\\x00-\x1f]', tok.decode([i])))]
    new = []
    while True:
        logits = model(torch.tensor([(ids + new)[-512:]]))[0, -1].float()
        logits[banned] = float("-inf")
        nxt = int(logits.argmax())
        if nxt == eos or len(tok.decode(new + [nxt])) > max_chars:
            return tok.decode(new)
        new.append(nxt)


def fill_ticket(tok, model, ids):
    """Шаблон предложения create_ticket: всё, кроме строк title и body, пишет программа."""
    head = '{"tool": "create_ticket", "arguments": {"title": "'
    title = greedy_string(tok, model, ids + tok.encode(head).ids, 80)
    middle = head + title + '", "body": "'
    body = greedy_string(tok, model, ids + tok.encode(middle).ids, 120)
    return middle + body + '"}}'


def ch13_booklm_in_loop():
    """Листинг книги «BookLM в цикле агента»: жадная генерация, страж Guard."""
    tok, model = load("instruct")

    def booklm(context, constrained=False):      # модель для run_agent
        text = TASK + "".join("\n" + json.dumps(obs, ensure_ascii=False)
                              for _, obs in context)
        if constrained:
            return fill_ticket(tok, model, prompt_ids(tok, text, chat=True))
        return generate(tok, model, text, chat=True, max_new=60, temperature=0)

    for constrained in (False, True):
        reset_world()
        state, traj = run_agent(lambda c: booklm(c, constrained))
        show(traj)
        print(state, "заявок:", len(WORLD["tickets"]))
    print(booklm([])[:60])
    for t in WORLD["tickets"]:
        print(repr(t["title"][:40]))


# ----------------------------------------------------------------------------- перезапуск, BookLM пишет иначе

def booklm_title(tok, model, note):
    """Заголовок заявки пишет BookLM, жадно. note — строка, которой контекст отличается после перезапуска."""
    ask = note + "Придумай короткий заголовок заявки про KV-кэш."
    return generate(tok, model, ask, chat=True, temperature=0, max_new=12).strip()


def ch13_regenerate_args():
    """Рубрика «ключ идемпотентности»: BookLM пишет другой заголовок после перезапуска."""
    from agent import (Crash, IdempotentGuard, KeyedServer, PermissionLog, SESSION, USER,
                       scripted)
    tok, model = load("instruct")

    def agent(note):                     # шаг 3 — BookLM, остальное — заглушка
        def model_fn(context):
            raw = scripted(context)
            if len(context) != 2:
                return raw
            new = json.dumps(booklm_title(tok, model, note), ensure_ascii=False)
            return raw.replace('"KV-кэш 3x512"', new)
        return model_fn

    print("до падения:     ", booklm_title(tok, model, ""))
    print("после рестарта: ", booklm_title(tok, model, "Процесс перезапущен. "))
    reset_world()
    log, intents = PermissionLog(), []
    guard = IdempotentGuard(USER, SESSION, log, KeyedServer(), intents)
    guard.crash_at = "C"
    try:
        run_agent(agent(""), guard=guard)
    except Crash:
        pass
    guard.crash_at = None
    state, _ = run_agent(agent("Процесс перезапущен. "), guard=guard)
    print(state, "заявок:", len(WORLD["tickets"]), "| на сервере:", WORLD["tickets"][0]["title"])
    print("расхождение записано на шаге:",
          [r["step"] for r in log.records if r["type"] == "replaced"])


# ----------------------------------------------------------------------------- ReAct и план в токенах

RULES = ("Отвечай по шагам. Сначала строка «Мысль:», затем строка «Действие:» с одним вызовом "
         "инструмента вида имя({аргументы в JSON}). После наблюдения пиши следующую мысль. "
         "Когда данных хватит, напиши «Ответ:» со ссылками [S:id].")
QUESTION = "На каком ускорителе работает сервис chat-api с BookLM и хватает ли ему памяти?"
OBS_1 = ("[ops-runbook-v6-0007] Архив. Сервис chat-api с моделью BookLM работал на узле gpu-02 "
         "с A100-40GB. [ops-runbook-v7-0007] Сервис chat-api с моделью BookLM работает на узле "
         "gpu-07 с H100.")
REACT = [  # (мысль, действие, наблюдение) — траектория из раздела «План и формат ReAct»
    ("Надо узнать, на каком ускорителе работает chat-api.",
     'search_docs({"query": "chat-api BookLM", "k": 3, "access": "public"})', OBS_1),
    ("v6 --- архив, действует v7: H100. Нужна память карты.",
     'search_docs({"query": "карта H100 HBM", "k": 3, "access": "public"})',
     "[hw-catalog-v1-0003] Карта H100: 80 ГБ памяти HBM."),
    ("Сколько KV-кэша нужно одному запросу полной длины?",
     'kv_cache_bytes({"tokens": 512, "requests": 1, "dtype": "bf16"})', '{"bytes": 786432}'),
]
# Что видит вызов, который пишет ответ по плану: план и вынутые программой поля.
PLAN_RESULTS = ("E1 search_docs «chat-api BookLM» -> H100 [ops-runbook-v7-0007]\n"
                "E2' search_docs «карта H100 HBM» -> 80 ГБ [hw-catalog-v1-0003]\n"
                'E3\' kv_cache_bytes 512, 1, bf16 -> {"bytes": 786432}')


def ch13_react_tokens():
    """Рубрика «цена траектории в токенах»: входы вызовов у ReAct и у плана."""
    tok = load_tokenizer()

    def n(text):
        return len(tok.encode(text).ids)

    prefix = "\n".join([RULES, *SIGNATURES, "Вопрос: " + QUESTION])
    steps = [f"Мысль: {m}\nДействие: {a}\nНаблюдение: {o}" for m, a, o in REACT]
    react = [n("\n".join([prefix, *steps[:i]])) for i in range(4)]   # три действия и ответ
    plan = [n(prefix), n(prefix + "\n" + PLAN_RESULTS)]               # план и ответ
    print("префикс", n(prefix), "| шаги ReAct", [n(s) for s in steps])
    for name, calls in (("ReAct", react), ("план", plan)):
        fit = sum(c <= 512 for c in calls)
        print(f"{name:<5} входы {calls}, всего {sum(calls)}, в окне {fit} из {len(calls)}")


# ----------------------------------------------------------------------------- pass^k и pass@k

FORMS = [  # десять формулировок одной просьбы: так «пользователь» вносит разброс
    "Напиши сказку про {h}, в которой есть слово «{w}».",
    "Расскажи сказку про {h}, в которой есть слово «{w}».",
    "Придумай сказку про {h}, в которой есть слово «{w}».",
    "Сочини сказку про {h}, в которой есть слово «{w}».",
    "Напиши историю про {h}, в которой есть слово «{w}».",
    "Напиши сказку про {h} со словом «{w}».",
    "Расскажи историю про {h} со словом «{w}».",
    "Придумай историю про {h}. В ней должно быть слово «{w}».",
    "Напиши короткую сказку про {h} и используй слово «{w}».",
    "Мне нужна сказка про {h}, где встречается слово «{w}».",
]
GOALS = [("кота", "мяч", r"\bкот(а|у|ом|ик)?\b", r"\bмяч"),  # герой, слово
         ("девочку", "маму", r"\bдевоч", r"\bмам"),        # и их основы
         ("птичку", "небо", r"\bптич|\bптиц", r"\bнеб(о|а|у|е)\b"),
         ("медведя", "мёд", r"\bмедвед|\bмишк", r"\bм[её]д(а|у|ом)?\b"),
         ("ежа", "лес", r"\b[её]ж", r"\bлес")]


def ch13_pass_hat_k():
    """Листинг книги «pass^k у BookLM на десяти формулировках» и начало первого ответа на каждую задачу."""
    tok, model = load("instruct")
    c, first = [], []
    for hero, word, hero_re, word_re in GOALS:
        texts = [generate(tok, model, f.format(h=hero, w=word), chat=True,
                          temperature=0, max_new=300).lower() for f in FORMS]
        c.append(sum(bool(re.search(hero_re, t) and re.search(word_re, t))
                     for t in texts))
        first.append(texts[0])
    print("успехов c_t из 10:", c)
    for k in (1, 2, 3, 5):
        hat = sum(pass_hat_k(10, ct, k) for ct in c) / len(c)
        at = sum(pass_at_k(10, ct, k) for ct in c) / len(c)
        print(f"k={k}: pass^k = {float(hat):.2f}   pass@k = {float(at):.2f}")
    for (hero, word, _, _), t in zip(GOALS, first):
        print(f"{hero} + «{word}», первая формулировка: {' '.join(t.split())[:80]}")


# ----------------------------------------------------------------------------- запуск

EXPERIMENTS = {f.__name__: f for f in (ch13_special_tokens, ch13_context_budget, ch13_booklm_in_loop,
                                        ch13_regenerate_args, ch13_react_tokens, ch13_pass_hat_k)}


if __name__ == "__main__":
    from booklm.experiments import run
    torch.manual_seed(0)
    OUT.mkdir(parents=True, exist_ok=True)
    run(EXPERIMENTS, save=True)      # копия вывода в models/booklm/experiments/<имя опыта>.txt
