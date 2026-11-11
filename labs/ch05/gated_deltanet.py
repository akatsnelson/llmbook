"""Минимальный слой Gated DeltaNet из главы 5 и его сверка со слоем Qwen3.5 из transformers.

Подраздел 5.5.7 «Блок Qwen3.5» печатает слой Gated DeltaNet в тридцать строк: проекции q, k, v,
короткая причинная свёртка, L2-нормировка запроса и ключа, затухание alpha через softplus и сила записи
beta через сигмоиду, правило дельты шаг за шагом (формула Gated DeltaNet в подразделе 4.12.5
«Правило дельты» главы 4), нормировка
выхода и ворота SiLU. Здесь мы проверяем две вещи.

1. Слой совпадает с Qwen3_5GatedDeltaNet из transformers (одна голова, те же веса): классу
   передаётся крошечная конфигурация, его веса переписываются в словарь нашей функции.
2. Без L2-нормировки ключа состояние разлетается: множитель (I - beta k k^T) вдоль k равен
   1 - beta |k|^2, и при |k|^2 beta > 2 он по модулю больше единицы. Печатаем этот множитель и
   норму состояния на тех же случайных данных, что и в книге.

Запуск из корня репозитория: python labs/ch05/gated_deltanet.py
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def gated_delta_net(x, p, l2=True):           # копия листинга главы 5
    qkv = x @ p["W_qkv"]
    c = qkv.size(1)
    qkv = F.conv1d(F.pad(qkv.T, (3, 0)), p["conv"][:, None], groups=c).T
    q, k, v = F.silu(qkv).chunk(3, dim=-1)
    if l2:
        q, k = F.normalize(q, dim=-1, eps=1e-6), F.normalize(k, dim=-1, eps=1e-6)
    alpha = torch.exp(-p["A"] * F.softplus(x @ p["w_a"] + p["dt"]))
    beta = torch.sigmoid(x @ p["w_b"])
    S, out, sizes = torch.zeros(k.size(1), v.size(1)), [], []
    for t in range(len(x)):
        S = alpha[t] * S
        S = S + beta[t] * torch.outer(k[t], v[t] - k[t] @ S)
        out.append(q[t] @ S / math.sqrt(k.size(1)))
        sizes.append(S.norm())
    o = F.rms_norm(torch.stack(out), (v.size(1),), eps=1e-6)
    return (o * F.silu(x @ p["W_z"])) @ p["W_out"], torch.stack(sizes)


def book_params(d=32, d_k=16):
    torch.manual_seed(0)
    p = {"W_qkv": torch.randn(d, 3 * d_k) / d ** 0.5, "conv": torch.randn(3 * d_k, 4) / 2,
         "w_a": torch.randn(d) / d ** 0.5, "dt": torch.tensor(0.0), "A": torch.tensor(0.1),
         "w_b": torch.randn(d) / d ** 0.5, "W_z": torch.randn(d, d_k) / d ** 0.5,
         "W_out": torch.randn(d_k, d) / d_k ** 0.5}
    return p, torch.randn(64, d)


@torch.no_grad()
def compare_with_transformers(d=32, d_k=16, T=24):
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet
    cfg = Qwen3_5TextConfig(hidden_size=d, num_hidden_layers=1, layer_types=["linear_attention"],
                            linear_num_key_heads=1, linear_num_value_heads=1,
                            linear_key_head_dim=d_k, linear_value_head_dim=d_k,
                            linear_conv_kernel_dim=4, num_attention_heads=1, num_key_value_heads=1,
                            head_dim=d_k, intermediate_size=64, vocab_size=10)
    torch.manual_seed(1)
    layer = Qwen3_5GatedDeltaNet(cfg, 0)
    for prm in layer.parameters():                       # случайные веса вместо нулей и единиц
        prm.normal_(0, 0.3)
    p = {"W_qkv": layer.in_proj_qkv.weight.T, "conv": layer.conv1d.weight[:, 0],
         "w_a": layer.in_proj_a.weight[0], "dt": layer.dt_bias[0], "A": layer.A_log.exp()[0],
         "w_b": layer.in_proj_b.weight[0], "W_z": layer.in_proj_z.weight.T,
         "W_out": layer.out_proj.weight.T}
    layer.norm.weight.fill_(1.0)                         # в нашей функции масштаба нормы нет
    x = torch.randn(1, T, d)
    ours, _ = gated_delta_net(x[0], p)
    theirs = layer(x)[0]
    return (ours - theirs).abs().max().item()


def main():
    gap = compare_with_transformers()
    print(f"наш слой и Qwen3_5GatedDeltaNet расходятся на {gap:.1e}")
    assert gap < 1e-4

    p, x = book_params()
    for l2 in (True, False):
        _, sizes = gated_delta_net(x, p, l2)
        print(f"L2={l2}: |S| на шагах 8/32/64:", [f"{sizes[i]:.3g}" for i in (7, 31, 63)])
    # множитель вдоль k без нормировки
    qkv = F.conv1d(F.pad((x @ p["W_qkv"]).T, (3, 0)), p["conv"][:, None], groups=48).T
    k = F.silu(qkv).chunk(3, dim=-1)[1]
    beta = torch.sigmoid(x @ p["w_b"])
    factor = (1 - beta * k.pow(2).sum(-1)).abs()
    print(f"|1 - beta |k|^2| без нормировки: медиана {factor.median():.2f}, "
          f"больше 1 на {(factor > 1).float().mean():.0%} шагов")
    _, s_l2 = gated_delta_net(x, p, True)
    _, s_raw = gated_delta_net(x, p, False)
    assert s_l2.max() < 5 and s_raw[-1] > 1000


if __name__ == "__main__":
    main()
