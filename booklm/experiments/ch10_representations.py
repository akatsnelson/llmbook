"""Глава 10 «Работа с промптом»: что лежит внутри BookLM и игрушка про наблюдателя за рассуждением.

Опыты (время — на процессоре ноутбука Apple M2 Max):
  ch10_logit_lens    рубрика «логит-линза», подраздел «Внутренние представления модели»; несколько секунд.
                     Два самых вероятных токена после каждого слоя для двух фраз; P(Мия) и четыре головы,
                     которые сильнее всего смотрят на имя, с P(Мия) без каждой из них. Ещё пишет
                     ch10-head-ablation.csv: внимание на имя и P(Мия) без головы для всех 48 голов.
  ch10_gender_probe  рубрика «зонд и вектор управления», тот же подраздел; около 10 с.
                     Точность линейного зонда рода по слоям против зонда на случайных метках, длина вектора
                     управления и продолжения при разных alpha.
  ch10_cot_monitor   листинг подраздела «Мониторинг рассуждения», без BookLM; несколько секунд.
                     Доли честных решений и взломов и доля взломов, которые видит наблюдатель, по шагам
                     обучения со штрафом за слова наблюдателя и без него.

Запуск из корня репозитория:
    python -m booklm.experiments.ch10_representations                    # все опыты
    python -m booklm.experiments.ch10_representations ch10_logit_lens    # один опыт
Вывод каждого опыта копируется в models/booklm/experiments/<имя опыта через дефис>.txt.

Листинги книги импортируют отсюда данные для зонда: GIRLS, BOYS и PHRASES.
"""
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

if not __package__:  # запуск файлом (python booklm/experiments/ch10_representations.py): корень репозитория — в путь поиска
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from booklm.lab import MODELS, load, load_tokenizer, prompt_ids

OUT = MODELS / "experiments"

# ----------------------------------------------------------------------------- данные для зонда

# Имена героев, каждое — один токен BPE. Род взят из корпуса: в обучающих историях имя стоит после
# «девочка по имени» или «мальчик по имени», и мы оставили имена, у которых одно из двух сочетаний
# встречается не реже 85% случаев (см. corpus_name_gender ниже). «Лука» и «Петя» оканчиваются на -а/-я,
# но в сказках это мальчики.
GIRLS = ["Лили", "Сью", "Люси", "Мия", "Ами", "Салли", "Джейн", "Сара", "Люсия", "Эмма",
         "Аня", "Лиза", "Мария", "Дейзи", "Анна", "Эмили", "Молли", "Элла", "Грейс", "Роуз"]
BOYS = ["Тим", "Том", "Лука", "Джек", "Сам", "Тимми", "Марк", "Джо", "Джон", "Макс",
        "Боб", "Бен", "Вилл", "Джейк", "Джимми", "Сэм", "Петя", "Джим", "Тоби", "Майк"]

# Фразы без рода: глаголы во множественном числе или в настоящем времени. Зонд читает вектор
# последнего токена («Там», «Потом», «Однажды», «Вдруг»), одинакового во всех примерах, поэтому
# на слое 0 род угадать нельзя, и всё, что зонд найдёт глубже, принесли блоки модели.
PHRASES = ["{} и мама пошли в парк. Там", "{} и папа играли в мяч. Потом",
           "{} любит яблоки. Однажды", "{} и собака бегали во дворе. Вдруг",
           "{} видит птицу. Потом", "{} любит рисовать. Однажды",
           "{} и Бобо пошли гулять. Вдруг", "{} и друзья строили замок. Потом",
           "{} очень любит торт. Однажды", "{} и кот спали на диване. Вдруг",
           "Утром {} и мама пекли пирог. Потом", "{} и бабушка читали книгу. Потом",
           "{} хочет новую игрушку. Однажды", "{} и брат пошли к реке. Там",
           "{} и сестра играли в саду. Вдруг", "{} любит гулять в лесу. Однажды",
           "{} и папа ехали в машине. Потом", "{} смотрит на небо. Вдруг",
           "{} и мама пошли в магазин. Там", "{} и собака нашли мяч. Потом"]


def corpus_name_gender(min_count=10, share=0.85):
    """Имена после «девочка/мальчик по имени» в обучающем корпусе и их преобладающий род."""
    tok = load_tokenizer()
    data = np.fromfile(MODELS / "data" / "train.bin", dtype=np.uint16).astype(np.int64)
    po, imeni = tok.encode(" по имени").ids
    girl, boy = tok.encode(" девочка").ids[-1], tok.encode(" мальчик").ids[-1]
    counts = {}
    for i in np.flatnonzero((data[:-3] == po) & (data[1:-2] == imeni)):
        name, after, before = int(data[i + 2]), int(data[i + 3]), int(data[i - 1])
        nxt = tok.decode([after])
        if before in (girl, boy) and (nxt.startswith(" ") or nxt[:1] in ".,!?"):
            counts.setdefault(name, [0, 0])[before == boy] += 1
    return {tok.decode([n]).strip(): ("м" if m > f else "ж", f + m)
            for n, (f, m) in counts.items() if f + m >= min_count and max(f, m) / (f + m) >= share}


def fit_probe(Z, labels, test, steps=300, lr=0.05):
    """Линейный зонд: логистическая регрессия по векторам слоя Z [N, 256].

    Обучается на примерах вне маски test, возвращает долю верных ответов на примерах из test.
    Координаты приводятся к нулевому среднему и единичному разбросу по обучающей части.
    """
    Z = (Z - Z[~test].mean(0)) / (Z[~test].std(0) + 1e-6)
    w = torch.zeros(Z.size(1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr)
    for _ in range(steps):
        loss = F.binary_cross_entropy_with_logits(Z[~test] @ w + b, labels[~test])
        opt.zero_grad()
        loss.backward()
        opt.step()
    guess = (Z[test] @ w + b > 0).float()
    return (guess == labels[test]).float().mean().item()


# ----------------------------------------------------------------------------- общие функции опытов

def _setup():
    tok, model = load("base")

    @torch.no_grad()
    def residual_stream(ids, add=None, at=None):
        h = model.embed(torch.tensor([ids]))
        states = [h[0]]
        for i, block in enumerate(model.blocks):
            if i == at:
                h = h + add
            h, _ = block(h, model.rope_cos, model.rope_sin)
            states.append(h[0])
        return torch.stack(states)

    return tok, model, residual_stream


@torch.no_grad()
def ch10_logit_lens():
    """Рубрика «логит-линза»: прогноз после каждого слоя и головы, которые копируют имя."""
    tok, model, residual_stream = _setup()

    def lens(h, k=2):
        p = (model.norm(h) @ model.embed.weight.T).softmax(-1)
        top = p.topk(k)
        return ", ".join(f"{tok.decode([i]).strip()} {v:.2f}"
                         for v, i in zip(top.values.tolist(), top.indices.tolist()))

    texts = ["Однажды маленькая девочка по имени Лили пошла в",
             "Жила-была девочка по имени Мия. Однажды"]
    states = [residual_stream(prompt_ids(tok, t))[:, -1] for t in texts]
    # последний слой линзы и есть выход модели: веса связаны, норма та же
    out = model(torch.tensor([prompt_ids(tok, texts[1])]))[0, -1]
    assert torch.allclose(model.norm(states[1][6]) @ model.embed.weight.T, out, atol=1e-4)
    for layer in range(7):
        print(f"слой {layer}: {lens(states[0][layer]):<22} | {lens(states[1][layer])}")

    mia = tok.encode(" Мия").ids[0]
    mia_ids = torch.tensor([prompt_ids(tok, texts[1])])

    def p_mia(block=None, head=None):
        def silence(module, inputs):
            y = inputs[0].clone()
            y[..., head * 32:(head + 1) * 32] = 0
            return (y,)
        hook = (model.blocks[block].attention.out.register_forward_pre_hook(silence)
                if block is not None else None)
        logits, att = model(mia_ids, return_weights=True)
        if hook:
            hook.remove()
        return logits[0, -1].softmax(-1)[mia].item(), att

    p, att = p_mia()
    on_mia = torch.stack([a[0, :, -1, 7] for a in att])
    print(f"P(Мия) = {p:.2f}")
    for flat in on_mia.flatten().argsort(descending=True)[:4].tolist():
        b, h = divmod(flat, 8)
        print(f"блок {b + 1}, голова {h + 1}: внимание {on_mia[b, h]:.2f},",
              f"без этой головы P(Мия) = {p_mia(b, h)[0]:.2f}")
    # все 48 голов: доля внимания на «Мия» и P(Мия) без головы
    rows = []
    for b in range(6):
        for h in range(8):
            rows.append((b + 1, h + 1, round(on_mia[b, h].item(), 4), round(p_mia(b, h)[0], 4)))
    with open(OUT / "ch10-head-ablation.csv", "w", encoding="utf-8") as f:
        f.write("block,head,attention_on_name,p_name_without_head\n")
        f.writelines(",".join(map(str, r)) + "\n" for r in rows)


def ch10_gender_probe():
    """Рубрика «зонд и вектор управления»: линейный зонд рода по слоям и вектор управления."""
    tok, model, residual_stream = _setup()
    rows, names = [], []
    for name in GIRLS + BOYS:
        for phrase in PHRASES:
            ids = prompt_ids(tok, phrase.format(name))
            rows.append(residual_stream(ids)[:, -1])
            names.append(name)
    X = torch.stack(rows)
    y = torch.tensor([float(n in GIRLS) for n in names])
    random.seed(0)
    coin = {n: random.randint(0, 1) for n in GIRLS + BOYS}
    y_coin = torch.tensor([float(coin[n]) for n in names])

    new_names = [torch.tensor([n in GIRLS[i::4] + BOYS[i::4] for n in names]) for i in range(4)]
    new_phrases = torch.arange(len(names)) % len(PHRASES) >= 15
    print("слой  род   монетка  монетка на знакомых именах")
    for layer in range(7):
        real = sum(fit_probe(X[:, layer], y, m) for m in new_names) / 4
        ctrl = sum(fit_probe(X[:, layer], y_coin, m) for m in new_names) / 4
        seen = fit_probe(X[:, layer], y_coin, new_phrases)
        print(f"{layer:>4}  {real:.2f}  {ctrl:.2f}     {seen:.2f}")

    v = X[y == 1, 5].mean(0) - X[y == 0, 5].mean(0)
    print(f"длина v {v.norm():.1f}, длина вектора слоя {X[:, 5].norm(dim=-1).mean():.1f}")
    she, he = tok.encode(" она").ids[0], tok.encode(" он").ids[0]
    start = "Жил-был мальчик по имени Тим. Однажды"
    with torch.no_grad():
        for alpha in (0, 2, 4, 8):
            ids = prompt_ids(tok, start)
            for step in range(7):
                h = residual_stream(ids, add=alpha * v, at=5)[-1, -1]
                p = (model.norm(h) @ model.embed.weight.T).softmax(-1)
                if step == 0:
                    odds = f"она {p[she]:.2f}, он {p[he]:.2f}"
                ids.append(int(p.argmax()))
            print(f"alpha={alpha}: {odds} |{tok.decode(ids[len(prompt_ids(tok, start)):])}")


def ch10_cot_monitor():
    """Игрушка без BookLM: штраф наблюдателя за рассуждение в награде (подраздел «Мониторинг рассуждения»)."""
    tests_pass = torch.tensor([0.5, 1.0, 1.0])
    says_hack = torch.tensor([0.0, 1.0, 0.0])

    def train(penalty, steps, lr=1.0):
        z = torch.tensor([0.80, 0.15, 0.05]).log()
        for _ in range(steps):
            p = z.softmax(0)
            r = tests_pass - penalty * says_hack
            z = z + lr * p * (r - p @ r)
        return z.softmax(0)

    for penalty in (0.0, 1.0):
        print(f"штраф за слова наблюдателя: {penalty}")
        for steps in (0, 20, 200):
            p = train(penalty, steps)
            hacks = p[1] + p[2]
            print(f"  шаг {steps:>3}: честно {p[0]:.2f}, взлом {hacks:.2f},",
                  f"наблюдатель видит {p[1] / hacks:.0%}")


EXPERIMENTS = {f.__name__: f for f in (ch10_logit_lens, ch10_gender_probe, ch10_cot_monitor)}


if __name__ == "__main__":
    from booklm.experiments import run
    torch.manual_seed(0)
    run(EXPERIMENTS, save=True)      # копия вывода в models/booklm/experiments/<имя опыта>.txt
