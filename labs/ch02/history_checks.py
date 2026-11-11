"""Проверки к главе 2: утверждения листингов главы в виде assert.

В книге листинги печатают веса, ответы и вероятности, а читатель
сверяет их глазами. Здесь те же функции собраны в один файл и проверены утверждениями.
Запуск из корня репозитория: python labs/ch02/history_checks.py
"""

from collections import Counter, defaultdict
from fractions import Fraction


# --- Перцептрон -------------------------------------------------------------------------------

def predict(w, t, x):
    s = w[0] * x[0] + w[1] * x[1]
    return 1 if s > t else 0


def train(data, w, t, epochs=50):
    for _ in range(epochs):
        for x, y in data:
            err = y - predict(w, t, x)
            w = [w[0] + err * x[0], w[1] + err * x[1]]
            t = t - err
    return w, t


def train_scaled(data, w, t, lr, epochs=50):
    """Тот же алгоритм с шагом lr: для врезки про шаг 0,1 и округление."""
    for _ in range(epochs):
        for x, y in data:
            err = y - predict(w, t, x)
            w = [w[0] + lr * err * x[0], w[1] + lr * err * x[1]]
            t = t - lr * err
    return w, t


AND = [((0, 0), 0), ((0, 1), 0), ((1, 0), 0), ((1, 1), 1)]
XOR = [((0, 0), 0), ((0, 1), 1), ((1, 0), 1), ((1, 1), 0)]


def check_and():
    # первый круг руками: две ошибки, веса (5, -2), порог 2
    assert train(AND, [5, -3], 2, epochs=1) == ([5, -2], 2)
    w, t = train(AND, [5, -3], 2)
    assert (w, t) == ([3, 2], 4)
    assert [predict(w, t, x) for x, _ in AND] == [0, 0, 0, 1]


def check_xor_loop():
    results = {train(XOR, [5, -3], 2, epochs=e) == ([-1, -1], -1) for e in (10, 100, 1000)}
    assert results == {True}
    w, t = [-1, -1], -1
    assert [predict(w, t, x) for x, _ in XOR] == [1, 0, 0, 0]
    # один круг из точки петли: четыре ошибки и возврат в ту же точку
    errors = 0
    for x, y in XOR:
        err = y - predict(w, t, x)
        errors += err != 0
        w = [w[0] + err * x[0], w[1] + err * x[1]]
        t = t - err
    assert errors == 4 and (w, t) == ([-1, -1], -1)


def check_float_step():
    # с шагом 0,1 в float обучение останавливается из-за округления
    w, t = train_scaled(AND, [0.5, -0.3], 0.2, lr=0.1)
    assert abs(w[0] - 0.3) < 1e-9 and abs(w[1] - 0.1) < 1e-9 and abs(t - 0.4) < 1e-9
    assert w[0] + w[1] > t                      # 0.4000000000000001 > 0.4
    # в точной арифметике тот же алгоритм доходит до (0,3; 0,2) и порога 0,4
    wf, tf = train_scaled(AND, [Fraction(5, 10), Fraction(-3, 10)], Fraction(2, 10), lr=Fraction(1, 10))
    assert (wf, tf) == ([Fraction(3, 10), Fraction(1, 5)], Fraction(2, 5))


def two_layer(x):
    h1 = predict([1, 1], 0, x)
    h2 = predict([-1, -1], -2, x)
    return predict([1, 1], 1, (h1, h2)), (h1, h2)


def check_two_layer():
    assert [two_layer(x)[0] for x, _ in XOR] == [y for _, y in XOR]
    assert [two_layer(x)[1] for x, _ in XOR] == [(0, 1), (1, 1), (1, 1), (1, 0)]


# --- Триграммы --------------------------------------------------------------------------------

def check_trigrams():
    corpus = [["я", "люблю", "кошек"], ["я", "люблю", "кошек"],
              ["я", "люблю", "пиццу"], ["я", "люблю", "зиму"]]
    counts = defaultdict(Counter)
    for words in corpus:
        for a, b, c in zip(words, words[1:], words[2:]):
            counts[(a, b)][c] += 1

    def prob(a, b, c):
        seen = counts[(a, b)]
        total = sum(seen.values())
        return seen[c] / total if total else 0.0

    assert counts[("я", "люблю")] == {"кошек": 2, "пиццу": 1, "зиму": 1}
    assert prob("я", "люблю", "кошек") == 0.5
    assert prob("я", "обожаю", "кошек") == 0.0


if __name__ == "__main__":
    for check in (check_and, check_xor_loop, check_float_step, check_two_layer, check_trigrams):
        check()
        print("ok", check.__name__)
