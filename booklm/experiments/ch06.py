"""Глава 6 на обученной BookLM: мягкий префикс, его цена и обучение соединителя для картинок.

Опыты (время — на процессоре Apple M2 Max):
  ch06_temperature — в тексте книги этого листинга нет (температуру книга разбирает в главе 7, раздел
      «Выбор следующего токена»). Пять самых вероятных следующих токенов после «Мама мыла раму, а потом
      она пошла» при температуре 0,5, 1 и 2. Около 1 с.
  ch06_soft_prefix — раздел «Способы подачи признаков в LLM», рубрика «А теперь на нашей модели: мягкий
      префикс». Необученный порт ставит перед текстом векторы картинки; печатает продолжения с префиксом
      и без, сдвиг логитов и проверку, что снятый порт ничего не меняет. Около 1 с.
  ch06_prefix_cost — подраздел «Длина префикса и вычисления», рубрика «А теперь на нашей модели: цена
      префикса». Время до первого токена и на следующий токен и добавка KV-кэша при префиксе из K
      векторов. Около 6 с; времена зависят от машины.
  ch06_connector_train — раздел «Съёмный порт», рубрика «А теперь на нашей модели: обучение
      соединителя». Учит соединитель для замороженной BookLM на цветных картинках 32 x 32, проверяет
      его перемешиванием картинок и печатает сказки по картинкам четырёх цветов. Около 20 с.
Все опыты печатают только на экран, txt не пишут.

Запуск из корня репозитория:
    python -m booklm.experiments.ch06                        # все опыты
    python -m booklm.experiments.ch06 ch06_connector_train   # один опыт
    python -m booklm.experiments.ch06 --help                 # список опытов
"""
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

if not __package__:  # запуск файлом (python booklm/experiments/ch06.py): корень репозитория — в путь поиска
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from booklm.experiments import run
from booklm.lab import MODELS, load, val_tokens

OUT = MODELS / "experiments"
BOS = 1


# ----------------------------------------------------------------------------- порт из главы 6

class PatchEncoder(nn.Module):
    def __init__(self, channels, patch_size, width):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(channels * patch_size * patch_size, width)

    def forward(self, image):
        p = self.patch_size
        patches = image.unfold(2, p, p).unfold(3, p, p)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        patches = patches.view(image.size(0), -1, self.proj.in_features)
        return self.proj(patches)


class InputPort(nn.Module):
    def __init__(self, encoder, encoder_width, lm_width):
        super().__init__()
        self.encoder = encoder                                  # E_m
        self.projector = nn.Linear(encoder_width, lm_width)     # P_m
        self.start = nn.Parameter(torch.zeros(1, 1, lm_width))  # <m>
        self.end = nn.Parameter(torch.zeros(1, 1, lm_width))    # </m>

    def forward(self, raw):
        tokens = self.projector(self.encoder(raw))
        start = self.start.expand(raw.size(0), -1, -1)
        end = self.end.expand(raw.size(0), -1, -1)
        return torch.cat([start, tokens, end], dim=1)


# ----------------------------------------------------------------------------- BookLM с префиксом

def forward_with_prefix(model, prefix, ids, caches=None):
    """То же, что BookLM.forward, но перед эмбеддингами текста стоят векторы prefix [B, K, 256]."""
    h = model.embed(ids)
    if prefix is not None:
        h = torch.cat([prefix, h], dim=1)
    for i, block in enumerate(model.blocks):      # блокам нужны таблицы RoPE и кэши
        h, _ = block(h, model.rope_cos, model.rope_sin, None if caches is None else caches[i])
    return model.norm(h) @ model.embed.weight.T


@torch.no_grad()
def generate_with_prefix(model, prefix, ids, n, clock=None):
    """Жадная генерация n токенов с KV-кэшем: префикс и текст проходят один раз, дальше по токену.

    clock: если передан список, в него после каждого токена дописывается time.perf_counter()."""
    caches = [{"len": 0} for _ in model.blocks]
    logits = forward_with_prefix(model, prefix, ids, caches)
    for c in caches:
        c["len"] = logits.size(1)
    out = []
    for step in range(n):
        if step:                                   # следующий токен: один шаг по кэшу
            logits = model(nxt, caches)
            for c in caches:
                c["len"] += 1
        nxt = logits[:, -1].argmax(-1, keepdim=True)
        out.append(nxt.item())
        if clock is not None:
            clock.append(time.perf_counter())
    return out, caches


def embed_rms(model):
    """Среднеквадратичная норма строк таблицы эмбеддингов: такой длины векторы модель привыкла видеть."""
    return model.embed.weight.pow(2).sum(1).mean().sqrt()


def one_line(tok, ids):
    return " ".join(tok.decode(ids).split())


# ----------------------------------------------------------------------------- 6.2: температура

def ch06_temperature():
    """Пять лучших следующих токенов BookLM при температуре 0,5, 1 и 2."""
    tok, model = load("base")
    ids = torch.tensor([[BOS] + tok.encode("Мама мыла раму, а потом она пошла").ids])
    with torch.no_grad():
        logits = model(ids)[0, -1]                 # логиты последней позиции
    for tau in (0.5, 1.0, 2.0):
        p = (logits / tau).softmax(-1)
        top = p.topk(5)
        row = "  ".join(f"«{tok.decode([i])}» {v:.2f}"
                        for v, i in zip(top.values.tolist(), top.indices.tolist()))
        print(f"τ = {tau}:  {row}")


# ----------------------------------------------------------------------------- 6.4: мягкий префикс

def ch06_soft_prefix():
    """BookLM с необученным мягким префиксом: продолжения, сдвиг логитов, снятие порта."""
    torch.manual_seed(0)
    tok, model = load("base")
    ids = torch.tensor([[BOS] + tok.encode("Жила-была").ids])
    with torch.no_grad():
        before = model(ids)                                   # логиты текста до подключения

        port = InputPort(PatchEncoder(3, 8, 48), 48, 256)     # соединитель не обучен
        prefix = port(torch.rand(1, 3, 32, 32))               # 1 + 16 + 1 векторов
        tokens = prefix[:, 1:-1]
        prefix = prefix * embed_rms(model) / tokens.pow(2).sum(-1).mean().sqrt()
        print(f"префикс: {prefix.size(1)} векторов ширины {prefix.size(2)}, "
              f"норма {prefix[0, 1].norm():.2f} (строки эмбеддингов: {embed_rms(model):.2f})")

        plain, _ = generate_with_prefix(model, None, ids, 40)
        with_port, _ = generate_with_prefix(model, prefix, ids, 40)
        print("без префикса: Жила-была", one_line(tok, plain))
        print("с префиксом:  Жила-была", one_line(tok, with_port))
        delta = forward_with_prefix(model, prefix, ids)[0, -1] - before[0, -1]
        print(f"наибольшее |Δ| логитов последней позиции: {delta.abs().max():.2f}")

        del port, prefix                                      # снимаем порт
        print("логиты текста после снятия порта совпали:", torch.equal(before, model(ids)))
        print("forward_with_prefix без префикса = model(ids):",
              torch.equal(before, forward_with_prefix(model, None, ids)))


# ----------------------------------------------------------------------------- 6.4: цена префикса

def ch06_prefix_cost(runs=15):
    """Время до первого и следующего токена и добавка KV-кэша при префиксе из K векторов."""
    torch.manual_seed(0)
    torch.set_num_threads(1)                                  # один поток: замеры устойчивее
    tok, model = load("base")
    data = val_tokens()
    start = int((data == BOS).nonzero()[1])                   # вторая отложенная история
    ids = data[start:start + 50][None]                        # 50 токенов, первый — <bos>
    for _ in range(3):
        generate_with_prefix(model, None, ids, 20)            # прогрев

    # Время на занятой машине прыгает в разы, поэтому берём лучший из runs прогонов
    # (как timeit): медиана тоже ловит чужую нагрузку, минимум — нет. Шаг по кэшу —
    # медиана 99 шагов внутри прогона, затем лучший прогон.
    print(f"лучший из {runs} прогонов, CPU, FP32, один поток")
    print("  K  вход  первый токен, мс  следующий, мс   KV-кэш, КБ   добавка, КБ")
    for K in (0, 64, 256):
        prefix = torch.randn(1, K, 256) * embed_rms(model) / 16 if K else None  # 16 = sqrt(256)
        first, step = [], []
        for _ in range(runs):
            clock = [time.perf_counter()]
            generate_with_prefix(model, prefix, ids, 100, clock)
            first.append(clock[1] - clock[0])                 # проход по входу и первый токен
            step.append(statistics.median(b - a for a, b in zip(clock[1:], clock[2:])))
        with torch.no_grad():                                 # кэш после прохода по входу
            caches = [{"len": 0} for _ in model.blocks]
            forward_with_prefix(model, prefix, ids, caches)
        numbers = sum(c["k"].numel() + c["v"].numel() for c in caches)
        kib = numbers * 2 / 1024                              # 2 байта на число, как в BF16
        base = base if K else kib                             # кэш без префикса
        print(f"{K:3}  {K + 50:4}  {min(first) * 1e3:16.1f}  "
              f"{min(step) * 1e3:13.2f}  {kib:11.0f}  {kib - base:12.0f}")
    print("формула 1536·K байт: K=64 → 96 КБ, K=256 → 384 КБ")


# ----------------------------------------------------------------------------- 6.5: обучение соединителя

STORIES = ["Жил-был рыжий кот.", "Жила-была серая мышка.",
           "Жил-был белый зайчик.", "Жил-был чёрный щенок."]
COLORS = torch.tensor([[1.0, 0.5, 0.1], [0.5, 0.5, 0.5], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]])


def make_batch(tok, c):
    """Картинки 32x32 цветов c с шумом и тексты <bos>сказка, дополненные <pad> до общей длины."""
    raw = COLORS[c][:, :, None, None] + 0.1 * torch.randn(len(c), 3, 32, 32)
    texts = [[BOS] + tok.encode(STORIES[i]).ids for i in c.tolist()]
    width = max(map(len, texts))
    ids = torch.tensor([t + [0] * (width - len(t)) for t in texts])
    return raw, ids


def lm_loss(model, port, raw, ids):
    prefix = port(raw)
    logits = forward_with_prefix(model, prefix, ids)[:, :-1]   # позиция t предсказывает t+1
    labels = ids.masked_fill(ids <= BOS, -100)                 # <pad> и <bos> не предсказываем
    ignored = torch.full(prefix.shape[:2], -100)
    labels = torch.cat([ignored, labels], dim=1)[:, 1:]
    return F.cross_entropy(logits.transpose(1, 2), labels)


def modality_reliance_score(model, port, raw, ids):
    with torch.no_grad():
        correct = lm_loss(model, port, raw, ids)
        shuffled = lm_loss(model, port, raw.roll(1, dims=0), ids)
    return correct.item(), shuffled.item()


def ch06_connector_train():
    """Обучение соединителя для замороженной BookLM, проверка перемешиванием и сказки по картинкам."""
    torch.manual_seed(0)
    tok, model = load("base")
    model.requires_grad_(False)                         # BookLM заморожена
    port = InputPort(PatchEncoder(3, 8, 48), 48, 256)
    untrained = InputPort(PatchEncoder(3, 8, 48), 48, 256)
    untrained.load_state_dict(port.state_dict())        # копия порта до обучения
    port.encoder.requires_grad_(False)                  # учим P_m и две границы
    trainable = [p for p in port.parameters() if p.requires_grad]
    print(f"обучаемых чисел: {sum(p.numel() for p in trainable):,}")

    opt = torch.optim.Adam(trainable, lr=3e-3)
    for step in range(301):
        raw, ids = make_batch(tok, torch.randint(0, 4, (32,)))
        loss = lm_loss(model, port, raw, ids)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 100 == 0:
            print(f"шаг {step:3d}: ошибка {loss.item():.3f}")

    raw, ids = make_batch(tok, torch.arange(64) % 4)    # сдвиг на 1 всегда даёт чужой цвет
    for name, p in (("после обучения", port), ("до обучения   ", untrained)):
        correct, shuffled = modality_reliance_score(model, p, raw, ids)
        print(f"{name}: свои картинки {correct:.3f}, чужие {shuffled:.3f}")

    with torch.no_grad():
        for c, name in enumerate(("рыжая", "серая", "белая", "чёрная")):
            image = COLORS[c][None, :, None, None] + 0.1 * torch.randn(1, 3, 32, 32)
            out, _ = generate_with_prefix(model, port(image), torch.tensor([[BOS]]), 20)
            print(f"{name:7}→ {one_line(tok, out)}")


EXPERIMENTS = {f.__name__: f for f in
               (ch06_temperature, ch06_soft_prefix, ch06_prefix_cost, ch06_connector_train)}


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    run(EXPERIMENTS)
