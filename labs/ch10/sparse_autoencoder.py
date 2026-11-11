"""Глава 10, подразделы «Механистическая интерпретируемость», «Разреженный автокодировщик» и
«Активационное патчирование» (раздел 10.2 «Обучение в контексте»): все опыты целиком, с проверками assert.

Запуск из корня репозитория: python labs/ch10/sparse_autoencoder.py   (около десяти минут на CPU)
Только часть опытов: python labs/ch10/sparse_autoencoder.py toy sae   (на выбор: toy, neurons, sae, sweep, patching)

Что делает каждый опыт.

toy       игрушка Элхаге и соавторов (Toy Models of Superposition, 2022): 5 признаков в 2 координатах,
          x_hat = ReLU(W^T W x + b); сколько признаков модель укладывает при разной разреженности S,
          5 зёрен на каждое S.
neurons   самые сильные включения первых трёх нейронов FFN четвёртого блока BookLM.
sae       разреженный автокодировщик на векторах BookLM после третьего блока: 200 кусков train.bin по 512
          токенов, словарь 1024, штраф lambda = 2, 3000 шагов Adam пакетами по 256; ошибка восстановления,
          среднее число включённых признаков, мёртвые признаки, ошибка модели с подменой вектора;
          фрагменты с самыми сильными и со слабыми включениями четырёх признаков; усиление признака.
sweep     тот же автокодировщик при lambda = 1, 2, 3: как штраф задаёт число включённых признаков.
patching  активационное патчирование: «Лили любит рисовать. Однажды» против той же фразы с мужским
          именем, 8 пар имён; какая доля разницы логитов « она» и « он» возвращается, если подставить
          вектор чистого прогона в одну позицию одного слоя.

Конфигурация и зёрна совпадают с листингами книги (глава 10),
поэтому номера признаков и все числа здесь те же, что в книге.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from booklm.lab import MODELS, load, prompt_ids  # noqa: E402
from booklm.experiments.ch10_representations import BOYS, GIRLS  # noqa: E402

tok, model = load("base")


@torch.no_grad()
def residual_stream(ids, add=None, at=None):
    """Векторы остаточного потока одной строки: [7, T, 256], слой 0 — эмбеддинги (как в книге)."""
    h = model.embed(torch.tensor([ids]))
    states = [h[0]]
    for i, block in enumerate(model.blocks):
        if i == at:
            h = h + add
        h, _ = block(h, model.rope_cos, model.rope_sin)
        states.append(h[0])
    return torch.stack(states)


def windows(name, n):
    data = np.fromfile(MODELS / "data" / name, dtype=np.uint16)
    return [data[i * 512:(i + 1) * 512].tolist() for i in range(n)]


def context(flat, pos, left=4):
    text = tok.decode(flat[max(0, pos - left):pos])
    return f"{text}[{tok.decode([flat[pos]])}]".replace("\n", "/")


# ----------------------------------------------------------------------------- игрушка суперпозиции

def train_toy(S, seed, steps=2000, n=5, d=2):
    importance = 0.7 ** torch.arange(n)
    g = torch.Generator().manual_seed(seed)
    W = (0.5 * torch.randn(d, n, generator=g)).requires_grad_()
    b = torch.zeros(n, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=0.01)
    for _ in range(steps):
        on = torch.rand(1024, n, generator=g) > S
        x = torch.rand(1024, n, generator=g) * on
        x_hat = torch.relu(x @ W.T @ W + b)
        loss = (importance * (x - x_hat) ** 2).sum(-1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return W.detach(), b.detach()


def toy():
    result = {}
    for S in (0.0, 0.8, 0.97):
        runs = [train_toy(S, seed) for seed in range(5)]
        kept = [int((W.norm(dim=0) > 0.5).sum()) for W, _ in runs]
        W, b = runs[0]
        on = W.norm(dim=0) > 0.5
        deg = sorted(round(a) for a in (torch.atan2(W[1], W[0]).rad2deg() % 360)[on].tolist())
        print(f"S={S}: уложено {kept}, углы {deg}, сдвиг {b[on].mean():.2f}")
        result[S] = (kept, deg, b[on].mean().item())
    # плотные признаки: две оси, остальные три не уложены
    assert result[0.0][0] == [2] * 5
    assert abs(result[0.0][1][1] - result[0.0][1][0] - 90) <= 3
    # S = 0.8: четыре признака двумя парами противоположных направлений
    assert result[0.8][0] == [4] * 5
    assert all(abs(b - a - 90) <= 5 for a, b in zip(result[0.8][1], result[0.8][1][1:]))
    # S = 0.97: почти всегда все пять, пятиугольник с отрицательным сдвигом
    assert sum(k == 5 for k in result[0.97][0]) >= 4 and result[0.97][2] < -0.1
    deg = result[0.97][1]
    gaps = [b - a for a, b in zip(deg, deg[1:] + [deg[0] + 360])]
    assert all(55 <= gap <= 90 for gap in gaps), gaps


# ----------------------------------------------------------------------------- нейроны FFN

def neurons():
    acts = []
    hook = model.blocks[3].ffn.down.register_forward_pre_hook(
        lambda module, inputs: acts.append(inputs[0][0]))
    val = windows("val.bin", 40)
    with torch.no_grad():
        for w in val:
            model(torch.tensor([w]))
    hook.remove()
    A = torch.cat(acts)
    flat = sum(val, [])
    assert A.shape == (40 * 512, 768)
    for n in range(3):
        top = A[:, n].topk(4).indices.tolist()
        print(f"нейрон {n}:", " | ".join(context(flat, p, left=3) for p in top))


# ----------------------------------------------------------------------------- разреженный автокодировщик

class SAE(torch.nn.Module):
    def __init__(self, d, m):
        super().__init__()
        self.enc = torch.nn.Linear(d, m)
        self.dec = torch.nn.Linear(m, d)

    def forward(self, x):
        f = torch.relu(self.enc(x - self.dec.bias))
        return f, self.dec(f)


def sae_loss(sae, x, lam):
    f, x_hat = sae(x)
    length = sae.dec.weight.norm(dim=0)
    return ((x - x_hat) ** 2).sum(-1).mean() + lam * (f * length).sum(-1).mean()


def sae_toy_check():
    D = torch.tensor([[1.0, 0.0], [-0.5, 0.866], [-0.5, -0.866]])
    toy_sae = SAE(d=2, m=3)
    with torch.no_grad():
        toy_sae.enc.weight.copy_(D)
        toy_sae.dec.weight.copy_(D.T)
        toy_sae.enc.bias.zero_()
        toy_sae.dec.bias.zero_()
    x = 0.8 * D[:1]
    f, x_hat = toy_sae(x)
    torch.testing.assert_close(f, torch.tensor([[0.8, 0.0, 0.0]]))
    torch.testing.assert_close(x_hat, x)
    full = sae_loss(toy_sae, x, lam=1.0).item()
    with torch.no_grad():
        toy_sae.enc.bias.fill_(-0.2)
    shrunk = sae_loss(toy_sae, x, lam=1.0).item()
    # усадка: со сдвигом -0.2 ошибка восстановления 0.04, но штраф меньше на 0.2, и сумма ниже
    assert abs(full - 0.8) < 1e-3 and abs(shrunk - 0.64) < 1e-3


def train_vectors():
    return torch.cat([residual_stream(w)[3] for w in windows("train.bin", 200)])   # [102400, 256]


def train_sae(X, lam, steps=3000):
    torch.manual_seed(0)
    model_sae = SAE(d=256, m=1024)
    model_sae.dec.bias.data = X.mean(0)
    opt = torch.optim.Adam(model_sae.parameters(), lr=2e-3)
    for _ in range(steps):
        x = X[torch.randint(len(X), (256,))]
        loss = sae_loss(model_sae, x, lam=lam)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model_sae


def sweep():
    """Подбор lambda: среднее число включённых признаков на отложенных сказках при lambda = 1, 2, 3."""
    X = train_vectors()
    with torch.no_grad():
        V = torch.cat([residual_stream(w)[3] for w in windows("val.bin", 40)])
    l0 = {}
    for lam in (1.0, 2.0, 3.0):
        trained = train_sae(X, lam)
        with torch.no_grad():
            f, V_hat = trained(V)
        fvu = ((V - V_hat) ** 2).sum() / ((V - V.mean(0)) ** 2).sum()
        l0[lam] = (f > 0).float().sum(-1).mean().item()
        print(f"lambda={lam}: включено в среднем {l0[lam]:.1f}, необъяснённый разброс {fvu:.3f}")
    assert l0[1.0] > 80 and 20 < l0[2.0] < 40 and l0[3.0] < 20


FEATURES = (444, 342, 156, 658, 126)  # номера признаков, разобранных в книге
STEER = (156, 126)


def sae():
    sae_toy_check()
    X = train_vectors()
    model_sae = train_sae(X, lam=2.0)

    val = windows("val.bin", 40)
    with torch.no_grad():
        V = torch.cat([residual_stream(w)[3] for w in val])
        f, V_hat = model_sae(V)
        fvu = ((V - V_hat) ** 2).sum() / ((V - V.mean(0)) ** 2).sum()
        alive = torch.zeros(1024, dtype=torch.bool)
        for x in X.split(10_000):
            alive |= (model_sae(x)[0] > 0).any(0)
    l0 = (f > 0).float().sum(-1).mean().item()
    print(f"доля необъяснённого разброса {fvu:.3f}, включено в среднем {l0:.1f} из 1024,"
          f" мёртвых {int((~alive).sum())}")

    def text_loss(change):
        total = 0.0
        for w in val:
            h = residual_stream(w)[3]
            out = residual_stream(w, add=change(h) - h, at=3)[-1]
            logits = model.norm(out) @ model.embed.weight.T
            total += F.cross_entropy(logits[:-1], torch.tensor(w[1:])).item()
        return total / len(val)

    with torch.no_grad():
        base = text_loss(lambda h: h)
        rec = text_loss(lambda h: model_sae(h)[1])
        mean = text_loss(lambda h: X.mean(0).expand_as(h))
    print(f"ошибка модели: как есть {base:.3f}, с автокодировщиком {rec:.3f}, со средним вектором {mean:.3f}")
    assert fvu < 0.35 and 15 < l0 < 45 and base < rec < mean
    assert (mean - rec) / (mean - base) > 0.9

    flat = sum(val, [])
    peak = f.max(0).values
    checks = {}
    for n in FEATURES:
        top = f[:, n].topk(3).indices.tolist()
        strong = torch.nonzero(f[:, n] > 0.3 * peak[n]).flatten().tolist()
        word, count = Counter(tok.decode([flat[p]]) for p in strong).most_common(1)[0]
        print(f"признак {n}:", " | ".join(context(flat, p) for p in top))
        print(f"   сильнее 0,3 пика: {len(strong)}, из них «{word}» {count}")
        checks[n] = (len(strong), word, count)
        print("   самые частые:", Counter(tok.decode([flat[p]]) for p in strong).most_common(4))
        g = torch.Generator().manual_seed(n)
        weak = torch.nonzero((f[:, n] > 0.1 * peak[n]) & (f[:, n] <= 0.3 * peak[n])).flatten()
        pick = weak[torch.randperm(len(weak), generator=g)[:4]].tolist()
        print("   слабые 0,1--0,3 пика: " + " | ".join(context(flat, p) for p in pick))

    start = prompt_ids(tok, "Однажды Тим пошёл в парк.")
    texts = {}
    for n in STEER:
        for c in (0, 0.5, 1, 1.5):
            add = c * peak[n] * model_sae.dec.weight[:, n].detach()
            ids = list(start)
            for _ in range(16):
                h = residual_stream(ids, add=add, at=3)[-1, -1]
                ids.append(int((model.norm(h) @ model.embed.weight.T).argmax()))
            texts[n, c] = tok.decode(ids[len(start):])
            print(f"{n}, c={c}:{texts[n, c]}")
    fear = []
    for c in (0.5, 1, 1.5, 2):                   # «испуганный» признак 658
        add = c * peak[658] * model_sae.dec.weight[:, 658].detach()
        ids = list(start)
        for _ in range(16):
            h = residual_stream(ids, add=add, at=3)[-1, -1]
            ids.append(int((model.norm(h) @ model.embed.weight.T).argmax()))
        fear.append(tok.decode(ids[len(start):]))
        print(f"658, c={c}:{fear[-1]}")
    assert "мяч" in texts[156, 0.5] and "мяч" not in texts[156, 0]
    assert all("испуг" not in t and "страш" not in t for t in fear)
    assert "блестящ" in texts[126, 1] and "блестящ" not in texts[126, 0]
    # при c = 1,5 жадное продолжение чувствительно к последним знакам: на Apple M2 Max (книга) «золот
    # золот золот», на x86 с Linux «Камень был золот минерал»; проверяем общее — слово «золот»
    assert "золот" in texts[126, 1.5]
    # признак 444 — одно тире, признак 342 — роль слова: самое частое имя меньше четверти включений
    assert checks[444][1] == " —" and checks[444][0] == checks[444][2]
    assert checks[342][2] < checks[342][0] / 4
    assert checks[156][1] == " мяч" and checks[126][1] == " драгоценно"


# ----------------------------------------------------------------------------- активационное патчирование

def patching():
    she, he = tok.encode(" она").ids[0], tok.encode(" он").ids[0]

    @torch.no_grad()
    def gap(ids, layer=None, pos=None, vec=None):
        """Разница логитов « она» и « он»; при layer подменяем вектор на входе блока layer (6 — выход)."""
        h = model.embed(torch.tensor([ids]))
        for i in range(7):
            if i == layer:
                h = h.clone()
                h[0, pos] = vec
            if i < 6:
                h, _ = model.blocks[i](h, model.rope_cos, model.rope_sin)
        logits = model.norm(h[0, -1]) @ model.embed.weight.T
        return (logits[she] - logits[he]).item()

    phrase = "{} любит рисовать. Однажды"
    share = torch.zeros(7, 6)
    pairs = 0
    for girl, boy in zip(GIRLS[:10], BOYS[:10]):
        clean, bad = prompt_ids(tok, phrase.format(girl)), prompt_ids(tok, phrase.format(boy))
        if len(clean) != len(bad):
            continue
        states = residual_stream(clean)
        g_clean, g_bad = gap(clean), gap(bad)
        for layer in range(7):
            for pos in range(len(clean)):
                g = gap(bad, layer, pos, states[layer, pos])
                share[layer, pos] += (g - g_bad) / (g_clean - g_bad)
        pairs += 1
    share /= pairs
    print(f"пар имён: {pairs}")
    print("слой  " + "  ".join(f"{w:>5}" for w in ("<bos>", "имя", "любит", "рисов", ".", "Однаж")))
    for layer in range(7):
        print(f"{layer:>4}  " + "  ".join(f"{v:5.2f}" for v in share[layer].tolist()))
    assert pairs == 8
    assert share[0, 1] > 0.99 and share[0, 5].abs() < 0.01      # вначале всё решает имя
    assert share[6, 5] > 0.99                                     # на выходе — последняя позиция
    assert share[4, 1] > 0.5 and share[4, 5] < 0.3                # после четырёх блоков род ещё у имени
    assert share[5, 5] > 0.6 and share[5, 1] < 0.2                # пятый блок перенёс его к «Однажды»


EXPERIMENTS = {"toy": toy, "neurons": neurons, "sae": sae, "sweep": sweep, "patching": patching}

if __name__ == "__main__":
    for name in sys.argv[1:] or EXPERIMENTS:
        print(f"=== {name}")
        EXPERIMENTS[name]()
    print("OK")
