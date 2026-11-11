"""Глава 11, раздел 11.3 «Сжатие истории диалога»: листинги «Запись сводки с источниками» и
«Вопросы-зонды и две метрики потерь» (полная версия).

В книге оставлена сокращённая версия; этот файл самодостаточен и
выполняется в ноутбуке главы notebooks/11_memory.ipynb.

Строит шесть вопросов-зондов по истории Ани, проверяет их на полной истории,
сравнивает обрезку (сохранность 2/6) и сводку (4/6 при точности 4/5),
хранит сводку записью с источниками и версиями и считает цену сжатия.

Запуск из корня репозитория: python labs/ch11/summary_probes.py
"""

import re
from typing import NamedTuple

# Учебная история: факты Ани из обеих сессий в одном разговоре,
# восемь пар «вопрос-ответ», каждая по 60 токенов (32 + 28).
# Длины заданы числами, как в разделе о бюджете: настоящий счёт ведёт токенизатор.
PAIRS = [  # (id сообщения, пользователь, ассистент)
    ("d1", "Меня зовут Аня, я живу в Казани.", "Приятно познакомиться!"),
    ("d2", "Пиши коротко.", "Хорошо."),
    ("d3", "Пишу на Python 3.12.", "Учту версию."),
    ("d4", "Не ем арахис.", "Запомню."),
    ("d5", "С 1 ноября живу в Самаре.", "Поздравляю с переездом!"),
    ("d6", "Дедлайн диплома 15 декабря.", "Успеем."),
    ("d7", "Помоги с планом введения.", "Вот план из трёх пунктов."),
    ("d8", "Сократи второй пункт.", "Готово."),
]
PAIR_TOKENS, H, S = 60, 240, 48

class Probe(NamedTuple):
    question: str
    pattern: str   # где в тексте искать ответ (строчными буквами)
    truth: str     # верный ответ по всей истории

PROBES = [
    Probe("Как зовут пользователя?",  r"зовут (\w+)",         "аня"),
    Probe("Где живёт сейчас?",        r"жив\w* в (\w{5})",    "самар"),
    Probe("Ест ли арахис?",           r"(не |)е\w* арахис",   "не "),
    Probe("На какой версии Python?",  r"python (\d+\.\d+)",   "3.12"),
    Probe("Когда дедлайн диплома?",   r"(\d+ декабря)",       "15 декабря"),
    Probe("Как длинно отвечать?",     r"(коротк)",            "коротк"),
]

def answer(probe, text):
    found = re.findall(probe.pattern, text.lower())
    return found[-1] if found else None      # последнее упоминание побеждает

def measure(probes, text):
    """-> (сохранность фактов, точность утверждений текста).

    Утверждение — предложение, которое отвечает хотя бы на один зонд; оно верно, если верны все
    его ответы. Предложения без ответов на зонды («Помоги с планом введения») в точность не входят."""
    retention = sum(answer(p, text) == p.truth for p in probes) / len(probes)
    claims = []
    for c in re.split(r"(?<=[.!?])\s+", text.strip()):          # предложения
        said = [answer(p, c) == p.truth for p in probes if answer(p, c) is not None]
        if said:
            claims.append(all(said))
    precision = sum(claims) / len(claims) if claims else 1.0   # нечего проверять
    return retention, precision

class SummaryRecord(NamedTuple):            # неизменяемая запись
    id: str
    user: str
    session: str
    turns: tuple             # покрытый диапазон пар [a, b]
    source_ids: tuple        # id исходных сообщений
    compressor_version: str  # программа или модель сжатия
    template_version: str    # версия шаблона диалога
    n_tokens: int
    created_at: str
    text: str

summary = SummaryRecord(
    id="sum-1", user="u42", session="demo", turns=(1, 6),
    source_ids=("d1", "d2", "d3", "d4", "d5", "d6"),
    compressor_version="summ-v1", template_version="booklm-chat-v1",
    n_tokens=100, created_at="2026-11-12T09:00",
    text="Пользователя зовут Аня. Живёт в Казани. Просит отвечать коротко. "
         "Пишет на Python 3.12. Не ест арахис.")

def as_text(pairs):
    return " ".join(f"{user} {bot}" for _, user, bot in pairs)

# Зонды построены до сжатия: по полной истории каждый даёт верный ответ.
assert measure(PROBES, as_text(PAIRS))[0] == 1.0

# Политика А: обрезка, в окне последние 4 пары.
retention_trunc, precision_trunc = measure(PROBES, as_text(PAIRS[-4:]))
assert 4 * PAIR_TOKENS == H and retention_trunc == 2 / 6
assert precision_trunc == 1.0          # неправды нет, просьбы про диплом ничего не утверждают

# Политика Б: сводка пар 1-6 плюс пары 7 и 8.
a, b = summary.turns
assert summary.source_ids == tuple(pid for pid, _, _ in PAIRS[a - 1:b])
assert summary.n_tokens + 2 * PAIR_TOKENS == 220 <= H
assert (b - a + 1) * PAIR_TOKENS / summary.n_tokens == 3.6   # коэффициент сжатия
retention_summary, precision = measure(PROBES, summary.text)
assert retention_summary == 4 / 6 and precision == 4 / 5

# Цена: один вызов сжатия и выигрыш на каждом следующем ходу.
compress_call = S + 6 * PAIR_TOKENS + summary.n_tokens
saved_per_turn = 6 * PAIR_TOKENS - summary.n_tokens
assert compress_call == 508 <= 512 and saved_per_turn == 260
assert saved_per_turn * 1536 == 399_360          # байт KV-кэша на каждом ходу


if __name__ == "__main__":
    print("OK: summary_probes.py")
