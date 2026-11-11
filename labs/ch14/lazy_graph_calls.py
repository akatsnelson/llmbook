"""Учебный отложенный граф в духе ggml: построить план, потом выполнить, потом оценить память.

Полная версия листинга «Учебный вычислительный граф» из главы 14 книги
(раздел 14.11 «Готовые среды выполнения», подраздел «llama.cpp», пункт «Граф операций»). В книге оставлена сокращённая версия; здесь весь код целиком, с теми же проверками, плюс
общий вход, который считается один раз, и подсчёт времени жизни промежуточных буферов.

Запуск из корня репозитория: python labs/ch14/lazy_graph_calls.py
"""

from typing import Callable, Optional

import torch


class Node:
    """Узел графа: операция и входы, без вычислений."""

    def __init__(self, op: Optional[Callable], inputs: tuple = (), value=None, name: str = ""):
        self.op = op            # None у исходного узла: веса или вход
        self.inputs = inputs
        self.value = value
        self.name = name


def evaluate(node, memo=None):
    memo = {} if memo is None else memo
    if node in memo:                    # общий вход считаем один раз
        return memo[node]
    if node.op is None:
        result = node.value
    else:
        args = [evaluate(x, memo) for x in node.inputs]
        result = node.op(*args)
    memo[node] = result
    return result


# --- 1. Граф строится без вычислений -------------------------------------------------------------
torch.manual_seed(0)
tokens = torch.randn(2, 4)              # [T, d]: два токена ширины 4
up = torch.randn(6, 4)                  # [d_ff, d], как weight в nn.Linear

x = Node(None, value=tokens, name="tokens")
w = Node(None, value=up, name="ffn_up.weight")
calls = []                              # журнал реально выполненных операций


def matmul(a, b):
    calls.append("matmul")
    return a @ b.T


def silu(a):
    calls.append("silu")
    return a * torch.sigmoid(a)


h = Node(matmul, (x, w), name="matmul")
y = Node(silu, (h,), name="silu")
assert calls == []                      # граф построен, но ничего не посчитано

output = evaluate(y)
assert calls == ["matmul", "silu"]      # считает только evaluate
assert output.shape == (2, 6)
torch.testing.assert_close(output, torch.nn.functional.silu(tokens @ up.T))

# --- 2. Общий вход считается один раз -----------------------------------------------------------
calls.clear()


def add(a, b):
    calls.append("add")
    return a + b


# Остаточная ветвь: результат h нужен и SiLU, и сложению.
residual = Node(add, (h, y), name="residual")
evaluate(residual)
assert calls.count("matmul") == 1       # memo не дал посчитать h дважды
assert calls == ["matmul", "silu", "add"]


# --- 3. Время жизни промежуточных результатов ----------------------------------------------------
def topo_order(root):
    order, seen = [], set()

    def visit(node):
        if node in seen:
            return
        seen.add(node)
        for parent in node.inputs:
            visit(parent)
        order.append(node)

    visit(root)
    return order


def peak_live_buffers(root):
    """Сколько промежуточных результатов одновременно живут при обходе графа."""
    order = [n for n in topo_order(root) if n.op is not None]   # веса и вход не считаем
    last_use = {}
    for step, node in enumerate(order):
        for parent in node.inputs:
            last_use[parent] = step
    last_use[root] = len(order)          # ответ нужен после обхода
    live, peak = set(), 0
    for step, node in enumerate(order):
        live.add(node)
        peak = max(peak, len(live))
        live = {n for n in live if last_use.get(n, step) > step}
    return peak


# Цепочка a -> b -> c: буфер a свободен, как только посчитан b.
a = Node(matmul, (x, w), name="a")
b = Node(silu, (a,), name="b")
c = Node(silu, (b,), name="c")
assert len([n for n in topo_order(c) if n.op is not None]) == 3
assert peak_live_buffers(c) == 2        # три результата, но одновременно живут только два

# С остаточной ветвью h живёт дольше: до сложения.
assert peak_live_buffers(residual) == 3

print("lazy graph: OK")
