"""Квантование KV-кэша: память при длинном контексте, FP8 E4M3 и E5M2, целые коды по токенам и по каналам.

Полная версия опыта из главы 14 книги (раздел 14.4 «Кэш ключей и значений», подразделы
«Квантование KV-кэша», «FP8 в KV-кэше» и «Ключи по каналам, значения по токенам»; листинг
«Квантованный KV-кэш BookLM: перплексия»). В книге оставлены калькулятор, квантователь групп, игрушечный ключ с
каналом-выбросом и опыт с хуками на BookLM. Здесь добавлены:

* проверка самого способа измерения: пошаговая генерация, которая квантует каждый новый ключ и значение
  при записи в кэш, даёт те же логиты, что полный проход с квантованными K и V;
* ключи до и после поворота RoPE (KVQuant против KIVI и vLLM);
* остаток KIVI: последние, ещё не набравшие группу токены хранятся без сжатия;
* FP8 с масштабом на тензор, посчитанным по калибровочным окнам, против масштаба 1,0 по умолчанию;
* кроме перплексии — расхождение KL с исходной моделью и доля позиций, где совпал самый вероятный токен.

Запуск из корня репозитория: python labs/ch14/kv_cache_quant.py   (от одной до трёх минут на процессоре)
"""

import math
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from booklm.lab import load, perplexity, val_tokens  # noqa: E402
from booklm.model import apply_rope  # noqa: E402

GIB, KIB = 2 ** 30, 2 ** 10


# ============================================================================ 1. память кэша

def kv_bytes(layers, batch, tokens, kv_heads, head_dim, dtype_bytes=2):   # как в книге
    return 2 * layers * batch * tokens * kv_heads * head_dim * dtype_bytes


def check_memory():
    llama = dict(layers=32, kv_heads=8, head_dim=128)       # Llama 3.1 8B, config.json
    qwen = dict(layers=36, kv_heads=8, head_dim=128)        # Qwen3-8B
    assert kv_bytes(batch=1, tokens=1, **llama) == 128 * KIB
    assert kv_bytes(batch=1, tokens=1, **qwen) == 144 * KIB
    assert kv_bytes(batch=1, tokens=131072, **llama) == 16 * GIB
    assert kv_bytes(batch=8, tokens=32768, **llama) == 32 * GIB
    weights = 16_060_522_496 / GIB                          # model.safetensors.index.json, BF16
    assert 14.9 < weights < 15.0
    # 4 бита + масштаб и минимум FP16 на группу из 32 чисел = 5 бит на число
    four = kv_bytes(batch=8, tokens=32768, dtype_bytes=(4 + 32 / 32) / 8, **llama)
    assert four == 10 * GIB
    # MHA вместо GQA (32 KV-головы), BF16: кэш вчетверо больше
    mha = kv_bytes(layers=32, batch=1, tokens=131072, kv_heads=32, head_dim=128)
    assert mha == 64 * GIB
    print(f"Llama 3.1 8B: веса BF16 {weights:.2f} ГБ, кэш 128k BF16 16 ГБ, MHA было бы 64 ГБ")


# ============================================================================ 2. квантователь групп

def quantize_groups(x, bits, dim, group=32):          # тот же, что в листинге книги
    x = x.movedim(dim, -1)
    n, pad = x.size(-1), -x.size(-1) % group
    g = torch.cat([x, x[..., -1:].expand(*x.shape[:-1], pad)], -1)
    g = g.unflatten(-1, (-1, group))                  # [..., групп, group]
    lo, hi = g.amin(-1, keepdim=True), g.amax(-1, keepdim=True)
    step = (hi - lo).clamp(min=1e-8) / (2 ** bits - 1)
    q = ((g - lo) / step).round()                     # коды 0 .. 2^bits - 1
    return (q * step + lo).flatten(-2)[..., :n].movedim(-1, dim)


def fp8(dtype, scale=1.0):
    return lambda x: (x / scale).to(dtype).to(x.dtype) * scale


def check_quantizer():
    torch.manual_seed(0)
    x = torch.randn(5, 70)                            # 70 не делится на 32: проверяем добивку
    for bits in (2, 4, 8):
        y = quantize_groups(x, bits, dim=-1)
        g = torch.cat([x, x[:, -1:].expand(5, 26)], 1).unflatten(1, (3, 32))
        half = ((g.amax(-1) - g.amin(-1)) / (2 ** bits - 1) / 2).repeat_interleave(32, 1)[:, :70]
        assert ((x - y).abs() <= half + 1e-6).all()   # ошибка не больше полушага своей группы
        assert y.unique(dim=None).numel() <= 5 * 3 * 2 ** bits
    const = torch.full((3, 32), 2.5)
    assert torch.equal(quantize_groups(const, 2, dim=-1), const)   # постоянная группа без ошибки
    # FP8: E4M3 насыщается на 448, относительный шаг не зависит от масштаба
    e4m3 = fp8(torch.float8_e4m3fn)
    assert e4m3(torch.tensor([600.0])).item() == 448.0
    v = torch.tensor([1.3, 13.0, 0.13])
    rel = ((e4m3(v) - v) / v).abs()
    assert (rel <= 2 ** -4 + 1e-6).all()


def toy_outlier_channel():
    """Игрушечный ключ из листинга книги: 64 токена по 32 канала, канал 5 в 10 раз крупнее."""
    row = torch.tensor([[0.3, -0.5, 12.0, 0.8]])        # ручной пример книги
    assert quantize_groups(row, 2, dim=-1, group=4).tolist() == [[-0.5, -0.5, 12.0, -0.5]]
    torch.manual_seed(0)
    K, V = torch.randn(64, 32), torch.randn(64, 32)
    K[:, 5] *= 10
    normal = [c for c in range(32) if c != 5]
    rel = lambda a, b: ((a - b).norm() / a.norm()).item()   # noqa: E731
    out = {}
    for name, X in (("K", K), ("V", V)):
        for how, dim in (("по токенам", 1), ("по каналам", 0)):
            Y = quantize_groups(X, 2, dim=dim)
            out[name, how, "обычные"] = rel(X[:, normal], Y[:, normal])
            out[name, how, "канал 5"] = rel(X[:, 5], Y[:, 5])
    # по токенам обычные каналы ключа портятся сильнее, чем если бы их просто обнулили
    assert out["K", "по токенам", "обычные"] > 1.0
    assert out["K", "по каналам", "обычные"] < 0.5
    assert out["K", "по токенам", "канал 5"] < 0.05       # выброс сам задаёт края и цел
    assert abs(out["V", "по каналам", "обычные"] - out["V", "по токенам", "обычные"]) < 0.05
    return out


# ============================================================================ 3. BookLM

TOK, MODEL = load("base")


def kv_outputs(ids):
    """Выходы линейных слоёв k и v всех блоков (до RoPE): списки тензоров [T, 64]."""
    ks, vs, hooks = [], [], []
    for b in MODEL.blocks:
        hooks.append(b.attention.k.register_forward_hook(lambda m, i, o: ks.append(o[0])))
        hooks.append(b.attention.v.register_forward_hook(lambda m, i, o: vs.append(o[0])))
    with torch.no_grad():
        MODEL(ids)
    for h in hooks:
        h.remove()
    return ks, vs


def check_outlier_channels():
    data = val_tokens()
    ks, vs = kv_outputs(data[:512][None])
    k, v = ks[1].abs().amax(0), vs[1].abs().amax(0)      # блок 1, наибольший модуль канала
    top = k.topk(4).indices.tolist()
    assert set(top[:2]) <= {18, 19, 50, 51}, top         # пара 18-19 в головах 0 и 1
    assert k.max() / k.median() > 4 and v.max() / v.median() < 1.5
    for i, kk in enumerate(ks):                            # блоки 1-5: два из трёх — пары 18-19, 50-51
        a = kk.abs().amax(0)
        t = set(a.topk(3).indices.tolist())
        ratio = (a.max() / a.median()).item()
        if i == 0:
            assert ratio < 2.5, ratio                     # в блоке 0 выбросы слабее
        else:
            assert len(t & {18, 19, 50, 51}) >= 2 and 3 < ratio < 5, (i, t, ratio)
    amax = max(x.abs().max().item() for x in ks + vs)
    assert amax < 448                                     # масштаб 1,0 для E4M3 годится
    return k, v


def quant_attention(kq, vq, where="post", residual=0):
    """Подмена GQASelfAttention.forward: K и V проходят через kq и vq.

    where="pre" — ключи квантуются до поворота RoPE (KVQuant), "post" — после (KIVI, vLLM).
    residual=R — запрос видит без сжатия ключи и значения своей неполной группы из R токенов
    (остаток KIVI при R = G), а более ранние — сжатыми. Для полного прохода без кэша.
    """
    def forward(self, x, cos, sin, cache=None, return_weights=False):
        b, t, d = x.shape
        q = self.q(x).view(b, t, self.heads, self.d_head).transpose(1, 2)
        k = self.k(x).view(b, t, self.kv_heads, self.d_head).transpose(1, 2)
        v = self.v(x).view(b, t, self.kv_heads, self.d_head).transpose(1, 2)
        start = 0 if cache is None else cache["len"]
        kh = kq(k) if where == "pre" else None
        q = apply_rope(q, cos[start:], sin[start:])
        k = apply_rope(k, cos[start:], sin[start:])
        kh = apply_rope(kh, cos[start:], sin[start:]) if where == "pre" else kq(k)
        vh = vq(v)
        if cache is not None:                   # пошагово: в кэш пишем уже сжатое
            if cache.get("k") is not None:
                kh = torch.cat([cache["k"], kh], dim=2)
                vh = torch.cat([cache["v"], vh], dim=2)
            cache["k"], cache["v"] = kh, vh
        g = self.heads // self.kv_heads
        kk, vv = kh.repeat_interleave(g, 1), vh.repeat_interleave(g, 1)
        total = kk.size(2)
        if not residual:
            if t == 1:
                y = F.scaled_dot_product_attention(q, kk, vv)
            else:
                assert total == t
                y = F.scaled_dot_product_attention(q, kk, vv, is_causal=True)
        else:
            assert cache is None
            kf, vf = k.repeat_interleave(g, 1), v.repeat_interleave(g, 1)
            i = torch.arange(t)
            same = (i[:, None] // residual) == (i[None, :] // residual)
            s = torch.where(same, q @ kf.transpose(-1, -2), q @ kk.transpose(-1, -2))
            s = (s / math.sqrt(self.d_head)).masked_fill(i[None, :] > i[:, None], float("-inf"))
            a = s.softmax(-1)
            y = (a * same) @ vf + (a * ~same) @ vv
        return self.out(y.transpose(1, 2).reshape(b, t, d)), None
    return forward


class patched:
    def __init__(self, fwd):
        self.fwd = fwd

    def __enter__(self):
        for b in MODEL.blocks:
            b.attention.forward = types.MethodType(self.fwd, b.attention)

    def __exit__(self, *exc):
        for b in MODEL.blocks:
            del b.attention.forward


def tok_(bits):                     # x: [B, H, T, D] — группа = вектор головы одного токена
    return lambda x: quantize_groups(x, bits, dim=-1)


def chan(bits):                     # группа = 32 соседних токена одного канала
    return lambda x: quantize_groups(x, bits, dim=-2)


IDENT = lambda x: x                 # noqa: E731


def check_simulation():
    """Полный проход с квантованными K, V совпадает с пошаговой генерацией, пишущей сжатое в кэш.

    Считаем во FP64: во FP32 ключ одного токена при полном проходе и при пошаговом отличается
    в седьмом знаке, и изредка такое число попадает ровно на границу округления FP8 — код
    перескакивает на соседний, и все следующие позиции расходятся на сотые доли логита.
    """
    ids = val_tokens()[:96][None]
    MODEL.double()
    try:
        _simulation(ids)
    finally:
        MODEL.float()


def _simulation(ids):
    with torch.no_grad():
        ref = MODEL(ids)
        with patched(quant_attention(IDENT, IDENT)):
            assert (MODEL(ids) - ref).abs().max() < 1e-4
        for kq, vq in ((tok_(4), tok_(4)), (fp8(torch.float8_e4m3fn),) * 2):
            with patched(quant_attention(kq, vq)):
                full = MODEL(ids)
                caches, steps = [{"len": 0} for _ in MODEL.blocks], []
                for t in range(ids.size(1)):
                    steps.append(MODEL(ids[:, t:t + 1], caches))
                    for c in caches:
                        c["len"] += 1
                err = (torch.cat(steps, 1) - full).abs().max().item()
                assert err < 1e-9, err
            assert (full - ref).abs().max() > 1e-2       # квантование правда что-то меняет


def calibrated_e4m3():
    """Масштаб на тензор: наибольший модуль K и V каждого слоя на 8 калибровочных окнах / 448."""
    data = val_tokens(120_000)
    amax_k, amax_v = torch.zeros(6), torch.zeros(6)
    for s in range(100_000, 100_000 + 8 * 512, 512):
        ks, vs = kv_outputs(data[s:s + 512][None])
        amax_k = torch.maximum(amax_k, torch.stack([k.abs().max() for k in ks]))
        amax_v = torch.maximum(amax_v, torch.stack([v.abs().max() for v in vs]))
    return amax_k / 448, amax_v / 448


def variants():
    e4, e5 = torch.float8_e4m3fn, torch.float8_e5m2
    return [  # имя, kq, vq, где ключи, остаток
        ("FP32 (как считает BookLM)", IDENT, IDENT, "post", 0),
        ("BF16", lambda x: x.bfloat16().float(), lambda x: x.bfloat16().float(), "post", 0),
        ("E4M3, масштаб 1", fp8(e4), fp8(e4), "pre", 0),
        ("E4M3, масштаб 1, после RoPE", fp8(e4), fp8(e4), "post", 0),
        ("E5M2, масштаб 1", fp8(e5), fp8(e5), "pre", 0),
        ("INT8 по токенам", tok_(8), tok_(8), "pre", 0),
        ("INT4 по токенам", tok_(4), tok_(4), "pre", 0),
        ("INT4: K по каналам", chan(4), tok_(4), "pre", 0),
        ("INT4: K по каналам, после RoPE", chan(4), tok_(4), "post", 0),
        ("INT2 по токенам", tok_(2), tok_(2), "pre", 0),
        ("INT2: K по каналам", chan(2), tok_(2), "pre", 0),
        ("INT2: K и V по каналам", chan(2), chan(2), "pre", 0),
        ("INT2: V по каналам", tok_(2), chan(2), "pre", 0),
        ("INT2: K по каналам, после RoPE", chan(2), tok_(2), "post", 0),
        ("INT2 по токенам + остаток 32", tok_(2), tok_(2), "pre", 32),
        ("INT2: K по каналам + остаток 32", chan(2), tok_(2), "pre", 32),
        ("INT2: K по каналам, после RoPE + остаток 32", chan(2), tok_(2), "post", 32),
    ]


@torch.no_grad()
def table(n=50_000, T=512):
    """Те же окна, что у booklm.lab.perplexity: перплексия, KL к исходной модели, совпадение лидера."""
    data = val_tokens(n)
    vs = variants()
    sk, sv = calibrated_e4m3()
    calls = {"k": 0, "v": 0}

    def scaled(kind, scales):
        def f(x):
            s = scales[calls[kind] % 6].item()
            calls[kind] += 1
            return fp8(torch.float8_e4m3fn, s)(x)
        return f
    vs.insert(3, ("E4M3, масштаб на тензор", scaled("k", sk), scaled("v", sv), "pre", 0))
    stats = {name: [0.0, 0.0, 0, 0.0, 0.0, 0.0, 0.0] for name, *_ in vs}
    for s in range(0, len(data) - T - 1, T):
        x, y = data[s:s + T][None], data[s + 1:s + T + 1]
        ref = MODEL(x)[0].log_softmax(-1)
        ks, vv = kv_outputs(x)
        for name, kq, vq, where, res in vs:
            with patched(quant_attention(kq, vq, where, res)):
                out = MODEL(x)[0].log_softmax(-1)
            st = stats[name]
            st[0] += F.nll_loss(out, y, reduction="sum").item()
            st[1] += (ref.exp() * (ref - out)).sum().item()
            st[2] += (ref.argmax(-1) == out.argmax(-1)).sum().item()
            st[3] += T
        for name, kq, vq, where, res in vs:      # ошибка самих K и V (до RoPE), как в листинге
            if name.startswith("E4M3, масштаб на тензор"):
                continue
            st = stats[name]
            for k in ks:
                kk = k.view(1, T, 2, 32).transpose(1, 2)
                st[4] += ((kq(kk) - kk) ** 2).sum().item()
                st[5] += (kk ** 2).sum().item()
            for v in vv:
                v4 = v.view(1, T, 2, 32).transpose(1, 2)
                st[6] += ((vq(v4) - v4) ** 2).sum().item()
    res = {}
    for name, st in stats.items():
        ppl, kl, top1 = math.exp(st[0] / st[3]), st[1] / st[3], st[2] / st[3]
        ek = math.sqrt(st[4] / st[5]) if st[5] else float("nan")
        res[name] = (ppl, kl, top1, ek)
        print(f"{name:<45} перплексия {ppl:6.3f}  KL {kl:.4f}  лидер {top1:6.1%}  ошибка K {ek:.3f}")
    return res


def hooks_like_book(kq, vq):
    """Ровно то, что делает листинг книги: хуки на выходы k и v, формы [B, T, 64]."""
    hooks = []
    for b in MODEL.blocks:
        hooks.append(b.attention.k.register_forward_hook(lambda m, i, o: kq(o)))
        hooks.append(b.attention.v.register_forward_hook(lambda m, i, o: vq(o)))
    return hooks


def main():
    check_memory()
    check_quantizer()
    toy = toy_outlier_channel()
    print("игрушечный ключ, INT2:", {k: round(v, 2) for k, v in toy.items()})
    k, v = check_outlier_channels()
    print("блок 1, наибольший модуль канала: K", [round(x, 1) for x in k.topk(4).values.tolist()],
          f"медиана {k.median():.2f}; V {v.max():.2f}, медиана {v.median():.2f}")
    check_simulation()
    print("пошаговая генерация со сжатым кэшем = полный проход со сжатыми K, V")
    res = table()
    base = res["FP32 (как считает BookLM)"][0]
    assert abs(base - perplexity(MODEL)) < 1e-6            # те же окна, что в книге
    # хуки книги (ключи до RoPE) дают те же числа, что подмена внимания с where="pre"
    for name, kq, vq in (("INT2: K по каналам", lambda o: quantize_groups(o, 2, dim=1),
                          lambda o: quantize_groups(o, 2, dim=-1)),):
        hooks = hooks_like_book(kq, vq)
        try:
            assert abs(perplexity(MODEL) - res[name][0]) < 1e-3
        finally:
            for h in hooks:
                h.remove()
    p = {name: r[0] for name, r in res.items()}
    assert abs(p["INT8 по токенам"] - base) < 0.01
    assert abs(p["BF16"] - base) < 0.005
    assert p["E4M3, масштаб 1"] < p["E5M2, масштаб 1"]
    assert abs(p["E4M3, масштаб на тензор"] - p["E4M3, масштаб 1"]) < 0.02
    assert p["INT4: K по каналам"] < p["INT4 по токенам"]
    assert p["INT2: K по каналам"] < p["INT2 по токенам"] - 3
    assert p["INT2: K по каналам"] <= p["INT2: K по каналам, после RoPE"]
    assert p["INT2: K по каналам + остаток 32"] < p["INT2: K по каналам"] - 0.3
    assert res["INT2: K по каналам"][3] < res["INT2 по токенам"][3] / 2
    # числа, которые стоят в тексте книги (листинг «Квантованный KV-кэш BookLM: перплексия»): расхождение — повод править книгу
    in_book = [  # (вариант, перплексия, доля совпавшего лидера или None)
        ("FP32 (как считает BookLM)", "6.958", None),
        ("BF16", "6.959", None),
        ("INT8 по токенам", "6.960", "99.4%"),
        ("E4M3, масштаб 1", "6.965", "97.9%"),
        ("E4M3, масштаб на тензор", "6.970", None),
        ("INT2: K по каналам", "7.934", None),
        ("INT2: K по каналам, после RoPE", "8.33", None),
        ("INT2: K по каналам + остаток 32", "7.17", "91.9%"),
    ]
    for name, ppl, top1 in in_book:
        digits = len(ppl.split(".")[1])
        assert f"{res[name][0]:.{digits}f}" == ppl, (name, res[name][0], ppl)
        if top1 is not None:
            assert f"{res[name][2]:.1%}" == top1, (name, res[name][2], top1)
    sk, sv = calibrated_e4m3()                              # модули K и V не выше 17
    data = val_tokens()
    peak = max(x.abs().max().item() for s in range(0, len(data) - 513, 512)
               for x in sum(kv_outputs(data[s:s + 512][None]), []))
    assert (torch.cat([sk, sv]) * 448).max() < 17 and peak < 17, peak
    print("все проверки прошли")


if __name__ == "__main__":
    main()
