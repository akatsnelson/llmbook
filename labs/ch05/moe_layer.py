"""Маршрут токена и слой смеси экспертов с групповым проходом.

Полная версия листингов «Маршрут одного токена» и «Групповой проход по экспертам» из главы 5 «Архитектуры
современных LLM» (раздел 5.4 «Смесь экспертов»). В книге оставлены маршрут и
групповой проход; здесь эксперты — настоящие SwiGLU-ветви главы 4, и слой
сверяется с наивным циклом по токенам.
Запуск из корня репозитория: python labs/ch05/moe_layer.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def route(scores, k):
    chosen = scores.topk(k).indices          # [k]
    # softmax только по выбранным
    weights = scores[chosen].softmax(dim=0)
    return chosen, weights


r = torch.tensor([2.0, 0.5, 1.0, -1.0])
chosen, weights = route(r, k=2)
assert chosen.tolist() == [0, 2]
torch.testing.assert_close(weights, torch.tensor([0.7311, 0.2689]),
                           atol=1e-4, rtol=0)
p_all = r.softmax(dim=0)                     # вторая запись маршрута
assert abs(p_all[chosen].sum().item() - 0.834) < 1e-3
torch.testing.assert_close(p_all[chosen] / p_all[chosen].sum(), weights)


class SwiGLU(nn.Module):                     # эксперт = ветвь главы 4
    def __init__(self, d, d_ff):
        super().__init__()
        self.gate = nn.Linear(d, d_ff, bias=False)
        self.up = nn.Linear(d, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class MoE(nn.Module):
    def __init__(self, d, d_ff, experts, k):
        super().__init__()
        self.k = k
        self.router = nn.Linear(d, experts, bias=False)
        self.experts = nn.ModuleList(
            SwiGLU(d, d_ff) for _ in range(experts))

    def forward(self, x):                    # x: [N, d], N = B * T
        top = self.router(x).topk(self.k, dim=-1)        # [N, k]
        weights = top.values.softmax(dim=-1)             # [N, k]
        y = torch.zeros_like(x)
        for e, expert in enumerate(self.experts):
            rows, slot = (top.indices == e).nonzero(as_tuple=True)
            if rows.numel():             # один вызов на эксперта
                y[rows] += weights[rows, slot, None] * expert(x[rows])
        return y


torch.manual_seed(5)
moe = MoE(d=256, d_ff=768, experts=4, k=2)
n_params = sum(p.numel() for p in moe.parameters())
assert n_params == 4 * 3 * 256 * 768 + 256 * 4
x = torch.randn(10, 256)
with torch.no_grad():
    y = moe(x)
    naive = []
    for row in x:                            # эталон: токен за токеном
        c, w = route(moe.router(row), k=2)
        naive.append(sum(w[i] * moe.experts[int(c[i])](row)
                         for i in range(2)))
assert y.shape == (10, 256)
torch.testing.assert_close(y, torch.stack(naive), atol=1e-5, rtol=1e-5)

# rows и slot на маленьком примере: три токена, выбор двух экспертов из четырёх
indices = torch.tensor([[0, 2], [0, 1], [2, 0]])
rows, slot = (indices == 0).nonzero(as_tuple=True)
assert rows.tolist() == [0, 1, 2]            # к эксперту 0 пришли все три токена
assert slot.tolist() == [0, 0, 1]            # у токена 2 он второй по оценке

print("moe_layer: все проверки пройдены")
