"""Глава 13 «Агенты и инструменты», рубрика «рассуждение между вызовами» (подраздел «Рассуждение между
вызовами инструментов»): опыт ch13_thinking_budget на токенизаторе BookLM.

У BookLM канала рассуждения нет (глава 10, подраздел «Рассуждающие модели»), поэтому мы не просим
её думать, а считаем, сколько места в её окне заняли бы мысли перед каждым шагом сквозной задачи главы 13.
Мысли написаны нами в том виде, в каком их пишет рассуждающая модель: по одной перед каждым
вызовом и одна перед финальным ответом. Весов модели опыт не требует.

Печатает токены каждой мысли, входы вызовов с мыслями и без них и сколько шагов помещается
в окно 512 токенов и в окно 128K при мысли в 1000 токенов.

Запуск из корня репозитория (несколько секунд):
    python -m booklm.experiments.ch13_interleaved
Вывод сохраняется в models/booklm/experiments/ch13-thinking-budget.txt.
"""
import contextlib
import io
import sys
from pathlib import Path

if not __package__:  # запуск файлом (python booklm/experiments/ch13_interleaved.py): корень репозитория — в путь поиска
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from booklm.lab import MODELS, load_tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "labs" / "ch13"))
from agent import steps_that_fit  # noqa: E402

OUT = MODELS / "experiments"

# Мысли перед тремя вызовами сквозной задачи (поиск, расчёт, заявка) и перед ответом.
THOUGHTS = [
    "Сначала узнаю, сколько байт KV-кэша BookLM приходится на один токен: поищу в открытой "
    "документации.",
    "Во фрагменте сказано 1536 байт на токен. Память для трёх запросов по 512 токенов посчитаю "
    "инструментом, а не в уме.",
    "Получилось 2359296 байт. Заведу одну заявку и укажу в ней число и ссылку на фрагмент.",
    "Заявка создана, данных хватает. Отвечу со ссылкой на фрагмент и номером заявки.",
]
STEPS = [193, 137, 165]   # шаги сквозной задачи без мыслей, токены BookLM (подраздел «Бюджет контекста»)
PREFIX = 248              # правила, три сигнатуры и вопрос (подраздел «Бюджет контекста»)


def ch13_thinking_budget():
    """Рубрика «рассуждение между вызовами»: сколько места в окне занимают мысли."""
    tok = load_tokenizer()

    def n_tok(text):
        return len(tok.encode(text).ids)

    thoughts = [n_tok(t) for t in THOUGHTS]
    with_t = [s + t for s, t in zip(STEPS, thoughts)]
    calls = [PREFIX + sum(with_t[:i]) for i in range(4)]
    plain = [PREFIX + sum(STEPS[:i]) for i in range(4)]
    print("мысли:", thoughts, " шаги с мыслями:", with_t)
    print("входы без мыслей:", plain, "всего", sum(plain))
    print("входы с мыслями: ", calls, "всего", sum(calls))
    print("шагов в окне 512:", steps_that_fit(reserve=56 + thoughts[3],
                                             step=sum(with_t) // 3))
    print("окно 128K, мысль 1000:", steps_that_fit(t_max=131072, tools=(308, 329, 235),
                                                  reserve=56 + 1000, step=165 + 1000))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ch13_thinking_budget()
    text = buf.getvalue()
    print(text, end="")
    (OUT / "ch13-thinking-budget.txt").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    from booklm.experiments import pick
    pick({"ch13_thinking_budget": ch13_thinking_budget})   # --help и проверка имени
    main()
