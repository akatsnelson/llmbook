"""Глава 13 «Агенты и инструменты», рубрика «поиск инструментов» (подраздел «Поиск инструментов
и программный вызов»): опыт ch13_tool_search на токенизаторе BookLM.

Две строки бюджета окна агента считаются токенизатором BookLM:
1) поиск инструментов: паспорта пяти инструментов главы целиком, строки «имя: описание» и
   вариант «паспорт поиска + три подгруженных паспорта»;
2) программный вызов: шестнадцать вызовов kv_cache_bytes шагами цикла против одной программы,
   которая делает те же вызовы через диспетчер главы, и её итоговой строки.
Программу пишем мы, вместо большой модели: BookLM код не пишет. Весов модели опыт
не требует.

Запуск из корня репозитория (несколько секунд):
    python -m booklm.experiments.ch13_tool_search
Вывод сохраняется в models/booklm/experiments/ch13-tool-search.txt.
"""
import contextlib
import io
import json
import sys
from pathlib import Path

if not __package__:  # запуск файлом (python booklm/experiments/ch13_tool_search.py): корень репозитория — в путь поиска
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from booklm.lab import MODELS, load_tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "labs" / "ch13"))
from agent import TOOLS, call, dispatch  # noqa: E402

from booklm.experiments.ch13 import describe  # noqa: E402

OUT = MODELS / "experiments"

# Паспорт инструмента поиска инструментов в том же виде, что describe: имя, описание, схема входа.
SEARCH_TOOL = json.dumps({
    "name": "search_tools",
    "description": "Найти инструменты по словам запроса и подгрузить их описания.",
    "input_schema": {"type": "object", "required": ["query"],
                     "properties": {"query": {"type": "string"}}}}, ensure_ascii=False)


def tool_line(name):
    """Строка «имя: описание» из паспорта, без схемы."""
    return f"{name}: {TOOLS[name].description}"


# Программа, которую вместо модели пишем мы: 16 вызовов kv_cache_bytes и одна строка итога.
PROGRAM = '''best = {}
for t in (64, 128, 256, 512):
    for r in (1, 2, 4, 8):
        b = kv_cache_bytes(tokens=t, requests=r, dtype="bf16")["bytes"]
        if b <= 2**20:
            best[t] = r
print("в 1 МБ:", best)'''


def run_program(code, names):
    """Исполняет код модели; каждый вызов инструмента идёт через dispatch."""
    seen = []

    def tool(name):
        def fn(**arguments):
            raw = call(name, **arguments)
            seen.append((raw, dispatch(raw)))    # разбор, реестр, схема
            return seen[-1][1]["result"]
        return fn

    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(code, {name: tool(name) for name in names})
    return out.getvalue().strip(), seen


def ch13_tool_search():
    """Рубрика «поиск инструментов»: паспорта инструментов и программный вызов в токенах."""
    tok = load_tokenizer()

    def n(text):
        return len(tok.encode(text).ids)

    full = [n(describe(x)) for x in sorted(TOOLS)]
    need = ["search_docs", "kv_cache_bytes", "create_ticket"]
    print("паспорта целиком:", full, "всего", sum(full))
    print("поиск + 3 паспорта:", n(SEARCH_TOOL), "+",
          sum(n(describe(x)) for x in need))
    summary, seen = run_program(PROGRAM, ["kv_cache_bytes"])
    steps = [n(raw) + n(json.dumps(obs, ensure_ascii=False)) for raw, obs in seen]
    print(summary)
    print("вызовов", len(seen), "| шагами цикла", sum(steps),
          "| программой", n(PROGRAM), "+", n(summary))
    # Для справки: строки «имя: описание» без схемы (в книгу не вошли).
    print("имя и описание:", sum(n(tool_line(x)) for x in sorted(TOOLS)))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ch13_tool_search()
    text = buf.getvalue()
    print(text, end="")
    (OUT / "ch13-tool-search.txt").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    from booklm.experiments import pick
    pick({"ch13_tool_search": ch13_tool_search})           # --help и проверка имени
    main()
