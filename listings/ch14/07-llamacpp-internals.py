"""Листинги главы 14 «Инференс» книги «Большие языковые модели», файл 07-llamacpp-internals.

Код и вывод — ровно как в книге; текст книги сюда не входит. Листинги идут по порядку и
опираются друг на друга: запускайте файл целиком или ячейками сверху вниз (разметка # %%).
Проверка, что вывод совпал с книгой: python scripts/check_listings.py listings/ch14/07-llamacpp-internals.py
"""

# %% Листинг «Учебный вычислительный граф» — раздел «Граф операций»
import torch

class Node:
    """Узел графа: операция и входы, без вычислений."""
    def __init__(self, op=None, inputs=(), value=None):
        # op=None у исходного узла: веса или вход
        self.op, self.inputs, self.value = op, inputs, value

def evaluate(node, memo=None):
    memo = {} if memo is None else memo
    if node not in memo:                # общий вход считаем один раз
        args = [evaluate(x, memo) for x in node.inputs]
        memo[node] = node.value if node.op is None else node.op(*args)
    return memo[node]

calls = []                              # журнал выполненных операций
def matmul(a, b):
    calls.append("matmul")
    return a @ b.T
def silu(a):
    calls.append("silu")
    return a * torch.sigmoid(a)

torch.manual_seed(0)
tokens, up = torch.randn(2, 4), torch.randn(6, 4)   # [T, d] и [d_ff, d]
x, w = Node(value=tokens), Node(value=up)
y = Node(silu, (Node(matmul, (x, w)),))   # silu(x @ w.T): только план
print("после построения:", calls)
output = evaluate(y)
print("после evaluate:  ", calls)
reference = torch.nn.functional.silu(tokens @ up.T)
gap = (output - reference).abs().max().item()
print("форма", tuple(output.shape), "расхождение", gap)
# Вывод в книге:
# | после построения: []
# | после evaluate:   ['matmul', 'silu']
# | форма (2, 6) расхождение 0.0

# %% Листинг «Граф BookLM в llama.cpp» — раздел «Граф операций»
import re
import subprocess
import tempfile
import textwrap
from collections import Counter
from booklm.experiments.ch14 import write_gguf

path = tempfile.gettempdir() + "/booklm-f16.gguf"
write_gguf("base", path, "F16")              # BookLM -> GGUF F16
cmd = ["llama-eval-callback", "-m", path, "-p", "Мама мыла раму",
       "-n", "1", "-t", "1", "-dev", "none"]
log = subprocess.run(cmd, capture_output=True, text=True)
# строки журнала: «Qcur-0 = (f32) MUL_MAT(...) = {256, 6, 1, 1}»
pattern = (r"common_debug_cb_eval:\s+(.+?) = \(\w+\)\s+(\w+)\(.*\) = "
           r"\{([\d, ]+)\}")
nodes = re.findall(pattern, log.stdout + log.stderr)
for name, op, shape in nodes[:11] + nodes[-1:]:
    print(f"{name:<15}{op:<10}[{shape}]")
names = [name for name, _, _ in nodes]
block0 = nodes[names.index("norm-0"):names.index("l_out-0") + 1]
print("всего узлов", len(nodes), "из них MUL_MAT",
      sum(op == "MUL_MAT" for _, op, _ in nodes))
ops = Counter(op for _, op, _ in block0).most_common()
print(textwrap.fill(f"в блоке 0 узлов {len(block0)}: "
                    + ", ".join(f"{op} {n}" for op, n in ops), 72))
# Вывод в книге:
# | embd           GET_ROWS  [256, 6, 1, 1]
# | norm-0         RMS_NORM  [256, 6, 1, 1]
# | attn_norm-0    MUL       [256, 6, 1, 1]
# | Qcur-0         MUL_MAT   [256, 6, 1, 1]
# | Qcur-0         RESHAPE   [32, 8, 6, 1]
# | Qcur-0         ROPE      [32, 8, 6, 1]
# | Vcur-0         MUL_MAT   [64, 6, 1, 1]
# | Vcur-0         RESHAPE   [32, 2, 6, 1]
# | Kcur-0         MUL_MAT   [64, 6, 1, 1]
# | Kcur-0         RESHAPE   [32, 2, 6, 1]
# | Kcur-0         ROPE      [32, 2, 6, 1]
# | result_output  MUL_MAT   [8192, 1, 1, 1]
# | всего узлов 192 из них MUL_MAT 43
# | в блоке 0 узлов 31: MUL_MAT 7, VIEW 5, RESHAPE 4, PERMUTE 3, RMS_NORM 2,
# | MUL 2, ROPE 2, SET_ROWS 2, ADD 2, FLASH_ATTN_EXT 1, SWIGLU 1

# %% Листинг «Подготовка: функции раздела о квантовании» — раздел «Умножение квантованной матрицы на вектор»
import contextlib, io, re, sys
from pathlib import Path

_root = Path(sys.argv[0]).resolve().parents[1]
_src = (_root / "listings/ch14/04-quantization-formats.py").read_text(encoding="utf-8")
for _code in [c.split("\n", 1)[1].split("# Вывод в книге:\n")[0]
              for c in re.split(r"^# %% ", _src, flags=re.M)[1:]]:
    with contextlib.redirect_stdout(io.StringIO()):
        exec(_code, globals())
    if "def unpack_int4" in _code:
        break

# %% Листинг «Умножение упакованных четырёхбитных весов на вектор» — раздел «Умножение квантованной матрицы на вектор»
torch.set_printoptions(precision=2)

def quantized_matvec(packed, scales, x, group_size):
    # packed: коды строк подряд, по два в байте; scales: [out, групп, 1]
    out_features, n_groups, _ = scales.shape
    x_groups = x.reshape(n_groups, group_size)
    nbytes = group_size // 2                 # байт на одну группу кодов
    y = torch.zeros(out_features)
    for row in range(out_features):
        acc = 0.0
        for group in range(n_groups):
            start = (row * n_groups + group) * nbytes
            codes = unpack_int4(packed[start:start + nbytes], group_size)
            w_group = codes.float() * scales[row, group]  # группа весов
            acc += torch.dot(w_group, x_groups[group]).item()
        y[row] = acc
    return y

torch.manual_seed(0)
w = torch.randn(3, 8)                  # 3 строки по 2 группы из 4 весов
w_hat, q, scales = groupwise_fake_quant(w, bits=4, group_size=4)
packed = pack_int4(q.flatten())
x = torch.randn(8)
print("кодов", q.numel(), "байт", packed.numel())
print("ядро:      ", quantized_matvec(packed, scales, x, group_size=4))
print("W_hat @ x: ", w_hat @ x)
print("W @ x:     ", w @ x)
# Вывод в книге:
# | кодов 24 байт 12
# | ядро:       tensor([0.20, 2.76, 6.69])
# | W_hat @ x:  tensor([0.20, 2.76, 6.69])
# | W @ x:      tensor([0.45, 2.92, 6.62])
