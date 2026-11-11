"""Глава 10, раздел 10.6 «Версии промпта и их оценка», подраздел «Парное сравнение
двух версий»: листинг «Критерий Макнемара и правило выпуска» (полная версия).

Точный критерий Макнемара, приближённый 95%-й интервал доли, таблица пар
исходов и правило выпуска новой версии промпта (вместе с намеренно оставленным
случаем, когда правило пропускает ухудшение на 8 п.п.). Выполняется в ноутбуке
главы notebooks/10_constrained_decoding.ipynb.

Запуск из корня репозитория: python labs/ch10/paired_eval.py
"""

# --- Листинг «Критерий Макнемара и правило выпуска» ------------------------
from math import comb, sqrt

def mcnemar_exact(only_a, only_b):
    n = only_a + only_b
    if n == 0:
        return 1.0
    tail = sum(comb(n, j) for j in range(max(only_a, only_b), n + 1)) / 2 ** n
    return min(1.0, 2 * tail)

def ci95(correct, n):
    """Приближённый 95%-й интервал доли верных ответов."""
    f = correct / n
    half = 1.96 * sqrt(f * (1 - f) / n)
    return f - half, f + half

def paired_table(correct_a, correct_b):
    assert len(correct_a) == len(correct_b)          # одни и те же задачи
    pairs = list(zip(correct_a, correct_b))
    return {"both": pairs.count((True, True)), "only_a": pairs.count((True, False)),
            "only_b": pairs.count((False, True)), "neither": pairs.count((False, False))}

# 50 задач: 33 обе верны, 3 только A, 7 только B, 7 обе неверны
correct_a = [True] * 33 + [True] * 3 + [False] * 7 + [False] * 7
correct_b = [True] * 33 + [False] * 3 + [True] * 7 + [False] * 7
table = paired_table(correct_a, correct_b)
assert sum(correct_a) == 36 and sum(correct_b) == 40  # 72% и 80%
lo, hi = ci95(sum(correct_b), len(correct_b))
assert round(lo, 2) == 0.69 and round(hi, 2) == 0.91  # «где-то от 69 до 91%»
p_mn = mcnemar_exact(table["only_a"], table["only_b"])
assert p_mn == 0.34375 == 2 * 176 / 1024
assert abs(mcnemar_exact(2, 8) - 0.109) < 1e-3       # 8 к 2: ещё не значимо
assert abs(mcnemar_exact(1, 9) - 0.021) < 1e-3       # 9 к 1: значимо
assert mcnemar_exact(30, 70) < 1e-4                  # те же доли на 500 задачах

def release_gate(form_rate, only_old, only_new, alpha=0.05):
    """Выпуск: форма 100% и новая версия не стала значимо хуже старой."""
    if form_rate < 1.0:
        return False
    worse = only_old > only_new
    return not (worse and mcnemar_exact(only_old, only_new) < alpha)

assert release_gate(1.0, only_old=3, only_new=7)        # улучшение не доказано, но не хуже
assert not release_gate(1.0, only_old=70, only_new=30)  # значимое ухудшение
assert not release_gate(0.98, only_old=0, only_new=10)  # форма не 100%
assert release_gate(1.0, only_old=7, only_new=3)        # минус 8 п.п. — и пропущено!


def non_inferior(only_old, only_new, n, delta=0.03):
    """Не хуже, чем: верхняя односторонняя 95%-я граница потери меньше delta."""
    assert 0 <= only_old + only_new <= n
    loss = (only_old - only_new) / n                 # среднее разностей d_i (глава 7)
    se = sqrt(only_old + only_new - (only_old - only_new) ** 2 / n) / n
    upper = loss + 1.645 * se
    return upper < delta, round(upper, 3)

assert non_inferior(7, 3, 50) == (False, 0.182)        # минус 8 п.п.: остановлено
assert non_inferior(3, 7, 50) == (True, 0.022)         # улучшение проходит
assert non_inferior(50, 50, 500) == (False, 0.033)     # равные версии, мало задач
assert non_inferior(70, 70, 700) == (True, 0.028)      # 700 задач уже хватает
assert non_inferior(12, 5, 100) == (False, 0.137)      # опыт с героями на BookLM
assert release_gate(1.0, only_old=12, only_new=5)      # ...который release_gate пропустил
assert abs(mcnemar_exact(12, 5) - 0.143) < 1e-3
# Сколько задач нужно при 20% несовпадений и равных версиях.
n_min = 0.2 * (1.645 / 0.03) ** 2
n_power = 0.2 * ((1.645 + 0.8416) / 0.03) ** 2       # проходит в 80% случаев
assert 600 < n_min < 602 and 1370 < n_power < 1380

print("OK: paired_eval.py")
