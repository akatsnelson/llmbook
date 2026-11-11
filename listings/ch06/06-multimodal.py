"""Листинги главы 6 «Мультимодальные модели» книги «Большие языковые модели», файл 06-multimodal.

Код и вывод — ровно как в книге; текст книги сюда не входит. Листинги идут по порядку и
опираются друг на друга: запускайте файл целиком или ячейками сверху вниз (разметка # %%).
Проверка, что вывод совпал с книгой: python scripts/check_listings.py listings/ch06/06-multimodal.py
"""

# %% Листинг «Симметричная ошибка CLIP» — раздел «Численный пример на двух парах»
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.set_printoptions(precision=4, sci_mode=False)

def clip_loss(image_features, text_features, temperature=0.1):
    # image_features, text_features: [B, D]
    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)
    # @ --- умножение матриц: [B, D] на [D, B] даёт [B, B]
    logits = image_features @ text_features.T / temperature
    target = torch.arange(logits.size(0))             # пары на диагонали
    image_to_text = F.cross_entropy(logits, target)   # по строкам
    text_to_image = F.cross_entropy(logits.T, target) # по столбцам
    return 0.5 * (image_to_text + text_to_image), logits

images = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
texts = torch.tensor([[0.8, 0.6], [0.6, 0.8]])
loss, logits = clip_loss(images, texts)
print(logits)
print(loss)
print(clip_loss(images, texts.flip(0))[0])        # подписи перепутаны
print(clip_loss(images, texts, temperature=0.05)[0])   # tau вдвое меньше
# Вывод в книге:
# | tensor([[8., 6.],
# |         [6., 8.]])
# | tensor(0.1269)
# | tensor(2.1269)
# | tensor(0.0181)

# %% Листинг «Ошибка SigLIP» — раздел «SigLIP»
def siglip_loss(logits, bias):
    s = logits + bias
    y = 2 * torch.eye(s.size(0)) - 1          # +1 на диагонали, -1 вне её
    return -F.logsigmoid(y * s).mean()        # среднее по B*B клеткам

print(siglip_loss(logits, bias=-7.0))
print(siglip_loss(logits.flip(0), bias=-7.0))
# Вывод в книге:
# | tensor(0.3133)
# | tensor(1.3133)

# %% Листинг «Нарезка картинки 4x4 на участки 2x2» — раздел «Изображение»
image = torch.arange(16.0).view(1, 1, 4, 4)       # [B=1, C=1, 4, 4]
patches = image.unfold(2, 2, 2).unfold(3, 2, 2)   # [1, 1, 2, 2, 2, 2]
patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
print(image[0, 0])
print(patches.view(1, 4, 4)[0])                   # участок = строка
# Вывод в книге:
# | tensor([[ 0.,  1.,  2.,  3.],
# |         [ 4.,  5.,  6.,  7.],
# |         [ 8.,  9., 10., 11.],
# |         [12., 13., 14., 15.]])
# | tensor([[ 0.,  1.,  4.,  5.],
# |         [ 2.,  3.,  6.,  7.],
# |         [ 8.,  9., 12., 13.],
# |         [10., 11., 14., 15.]])

# %% Листинг «Кодировщик изображения: участки и линейный слой» — раздел «Изображение»
class PatchEncoder(nn.Module):
    def __init__(self, channels, patch_size, width):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(channels * patch_size * patch_size, width)

    def forward(self, image):
        # image: [B, C, H, W]
        p = self.patch_size
        # режем по высоте, потом по ширине: [B, C, H/P, W/P, P, P]
        patches = image.unfold(2, p, p).unfold(3, p, p)
        # ряд и столбец участка вперёд, содержимое участка назад
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        # участки подряд: [B, S, C*P*P]
        patches = patches.view(image.size(0), -1, self.proj.in_features)
        return self.proj(patches)  # [B, S_img, width]

torch.manual_seed(0)
encoder = PatchEncoder(channels=3, patch_size=4, width=12)
z = encoder(torch.randn(2, 3, 8, 8))
print(z.shape)
# Вывод в книге:
# | torch.Size([2, 4, 12])

# %% Листинг «Алиасинг: 20 Гц при 32 отсчётах в секунду» — раздел «Звук»
import numpy as np

fs = 32                                   # Гц, без фильтра
t = np.arange(fs) / fs                    # [32]: одна секунда
x20 = np.cos(2 * np.pi * 20 * t)
x12 = np.cos(2 * np.pi * 12 * t)
print(np.allclose(x20, x12))              # отсчёты совпали?
spectrum = np.abs(np.fft.rfft(x20))       # [17]: частоты 0..16 Гц
print(spectrum.argmax())                  # где пик
# Вывод в книге:
# | True
# | 12

# %% Листинг «Фурье на четырёх отсчётах» — раздел «Преобразование Фурье и спектрограмма»
cos_wave = np.array([1.0, 0.0, -1.0, 0.0])
sin_wave = np.array([0.0, 1.0, 0.0, -1.0])        # та же k, сдвиг на 1/4
print(np.abs(np.fft.fft(cos_wave)).round(3))       # A_k для k = 0..3
print(np.abs(np.fft.fft(sin_wave)).round(3))
# Вывод в книге:
# | [0. 2. 0. 2.]
# | [0. 2. 0. 2.]

# %% Листинг «STFT на тоне 440 Гц» — раздел «Преобразование Фурье и спектрограмма»
def frame_signal(wave, frame_size, hop):
    count = 1 + (len(wave) - frame_size) // hop
    starts = [i * hop for i in range(count)]
    return np.stack([wave[s:s + frame_size] for s in starts])

def stft(wave, frame_size=400, hop=160):
    frames = frame_signal(wave, frame_size, hop)    # [окна, F]
    window = np.hanning(frame_size)                 # окно Ханна
    return np.fft.rfft(frames * window, axis=-1)    # [окна, F/2+1]

fs = 16000
t = np.arange(fs) / fs                           # одна секунда
spectrum = stft(np.sin(2 * np.pi * 440 * t))     # нота ля, 440 Гц
power = np.abs(spectrum) ** 2                    # фаза отброшена
peak = power.mean(axis=0).argmax()
print(spectrum.shape)
print(peak, peak * fs / 400, "Гц")
# Вывод в книге:
# | (98, 201)
# | 11 440.0 Гц

# %% Листинг «Mel-фильтры и log-mel-спектрограмма» — раздел «Преобразование Фурье и спектрограмма»
def mel(f):
    return 2595 * np.log10(1 + f / 700)

def mel_filters(n_mels=80, frame_size=400, fs=16000):
    freqs = np.arange(frame_size // 2 + 1) * fs / frame_size  # [201] Гц
    edges = np.linspace(0, mel(fs / 2), n_mels + 2)           # шаг по mel
    hz = 700 * (10 ** (edges / 2595) - 1)                     # в герцы
    left, mid, right = hz[:-2, None], hz[1:-1, None], hz[2:, None]
    rise = (freqs - left) / (mid - left)                      # склон вверх
    fall = (right - freqs) / (right - mid)                    # склон вниз
    return np.clip(np.minimum(rise, fall), 0, None)           # [80, 201]

def log_mel(wave):
    power = np.abs(stft(wave)) ** 2                           # [98, 201]
    return np.log(power @ mel_filters().T + 1e-10)            # [98, 80]

features = log_mel(np.sin(2 * np.pi * 440 * t))   # та же нота ля
print(mel_filters().shape, features.shape)
print(features.mean(axis=0).argmax())             # самая громкая полоса
# Вывод в книге:
# | (80, 201) (98, 80)
# | 15

# %% Листинг «Одна и две позиции у участков картинки» — раздел «2D и 3D RoPE»
import math
import torch

def freqs(n, base=10.0):
    return base ** (-torch.arange(n) / n)    # шаги пар, как в главе 4

def rope_nd(x, pos, axis, theta):
    # пара k поворачивается на угол pos[axis[k]] * theta[k]
    angles = torch.tensor(pos, dtype=torch.float)[axis] * theta
    x0, x1 = x[0::2], x[1::2]
    c, s = angles.cos(), angles.sin()
    turned = (x0 * c - x1 * s, x0 * s + x1 * c)
    return torch.stack(turned, dim=-1).flatten()

flat = torch.zeros(8, dtype=torch.long)       # 1D: 8 пар, одна ось
grid = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]) # 2D: 4 пары ряд, 4 столбец
theta_1d = freqs(8)
theta_2d = torch.cat([freqs(4), freqs(4)])    # у ряда и столбца одни шаги
q = k = torch.ones(16)
def score(a, b, axis, theta, to_pos):
    qa = rope_nd(q, to_pos(*a), axis, theta)
    kb = rope_nd(k, to_pos(*b), axis, theta)
    return round((qa @ kb).item(), 1)
one_d = lambda r, c: (4 * r + c,)             # номер участка в сетке 4x4
two_d = lambda r, c: (r, c)
for name, a, b in (("сосед справа", (0, 0), (0, 1)),
                   ("сосед снизу", (0, 0), (1, 0)),
                   ("конец ряда -> начало следующего", (0, 3), (1, 0))):
    s1 = score(a, b, flat, theta_1d, one_d)
    s2 = score(a, b, grid, theta_2d, two_d)
    print(f"{name:32} 1D: {s1:5}  2D: {s2:5}")
# Вывод в книге:
# | сосед справа                     1D:  13.9  2D:  14.6
# | сосед снизу                      1D:   0.2  2D:  14.6
# | конец ряда -> начало следующего  1D:  13.9  2D:   7.3

# %% Листинг «M-RoPE: текст, время, ряд и столбец» — раздел «2D и 3D RoPE»
cube = torch.tensor([0, 0, 1, 1, 1, 2, 2, 2]) # оси: время, ряд, столбец
x = torch.randn(16)
same = all(torch.allclose(rope_nd(x, (p, p, p), cube, theta_1d),
                          rope_nd(x, (p,), flat, theta_1d))
           for p in range(20))
print("текст в M-RoPE совпадает с обычным RoPE:", same)

pair = torch.arange(64)
period = 2 * math.pi * 1e6 ** (pair / 64)     # позиций на оборот пары
sizes = torch.tensor([16, 24, 24])
block = torch.repeat_interleave(torch.arange(3), sizes)
mixed = torch.zeros(64, dtype=torch.long)
mixed[1:60:3], mixed[2:60:3] = 1, 2           # 24/20/20 вперемежку
for name, axis in (("блоками, Qwen2-VL", block),
                   ("вперемежку, Qwen3-VL", mixed)):
    print(name)
    for a, label in enumerate(("время", "ряд", "столбец")):
        p = period[axis == a]
        print(f"  {label:8} пар {len(p)}, оборот",
              f"от {p.min():.0f} до {p.max():.0f} позиций")
# Вывод в книге:
# | текст в M-RoPE совпадает с обычным RoPE: True
# | блоками, Qwen2-VL
# |   время    пар 16, оборот от 6 до 160 позиций
# |   ряд      пар 24, оборот от 199 до 28473 позиций
# |   столбец  пар 24, оборот от 35333 до 5063256 позиций
# | вперемежку, Qwen3-VL
# |   время    пар 24, оборот от 6 до 5063256 позиций
# |   ряд      пар 20, оборот от 8 до 1720600 позиций
# |   столбец  пар 20, оборот от 10 до 2135158 позиций

# %% Листинг «Входной порт и сборка мягкого префикса» — раздел «Мягкий префикс»
class InputPort(nn.Module):
    def __init__(self, encoder, encoder_width, lm_width):
        super().__init__()
        self.encoder = encoder                                  # E_m
        self.projector = nn.Linear(encoder_width, lm_width)     # P_m
        self.start = nn.Parameter(torch.zeros(1, 1, lm_width))  # <m>
        self.end = nn.Parameter(torch.zeros(1, 1, lm_width))    # </m>

    def forward(self, raw):
        tokens = self.projector(self.encoder(raw))   # [B, S_m, d_lm]
        start = self.start.expand(raw.size(0), -1, -1)  # [B, 1, d_lm]
        end = self.end.expand(raw.size(0), -1, -1)
        return torch.cat([start, tokens, end], dim=1)

    def pack(self, raw, text_ids, embed):
        prefix = self(raw)                           # [B, S_m+2, d_lm]
        inputs = torch.cat([prefix, embed(text_ids)], dim=1)
        ignored = torch.full(prefix.shape[:2], -100, dtype=text_ids.dtype)
        labels = torch.cat([ignored, text_ids], dim=1)
        return inputs, labels

torch.manual_seed(0)
port = InputPort(PatchEncoder(3, 4, 12), encoder_width=12, lm_width=8)
embed = nn.Embedding(50, 8)
text_ids = torch.tensor([[5, 7, 9], [1, 2, 3]])
inputs, labels = port.pack(torch.randn(2, 3, 8, 8), text_ids, embed)
print(inputs.shape)
print(labels[0])
# Вывод в книге:
# | torch.Size([2, 9, 8])
# | tensor([-100, -100, -100, -100, -100, -100,    5,    7,    9])

# %% Листинг «Тот же порт для звука» — раздел «Мягкий префикс»
audio = torch.tensor(features, dtype=torch.float32)[None]   # [1, 98, 80]
audio_port = InputPort(nn.Linear(80, 12), encoder_width=12, lm_width=8)
print(audio_port(audio).shape)                     # окна + 2 границы
# Вывод в книге:
# | torch.Size([1, 100, 8])

# %% Листинг «BookLM с мягким префиксом» — раздел «Мягкий префикс»
import textwrap
from booklm.lab import load

def forward_with_prefix(model, prefix, ids):
    h = model.embed(ids)                                  # [B, T, 256]
    if prefix is not None:
        h = torch.cat([prefix, h], dim=1)                 # [B, K+T, 256]
    for block in model.blocks:
        h, _ = block(h, model.rope_cos, model.rope_sin)
    return model.norm(h) @ model.embed.weight.T           # [B, K+T, 8192]

@torch.no_grad()
def greedy(model, prefix, ids, n):                        # n токенов жадно
    for _ in range(n):
        logits = forward_with_prefix(model, prefix, ids)
        best = logits[:, -1].argmax(-1, keepdim=True)     # самый вероятный
        ids = torch.cat([ids, best], dim=1)
    return ids[0, -n:].tolist()

torch.manual_seed(0)
tok, model = load("base")
ids = torch.tensor([[1] + tok.encode("Жила-была").ids])   # 1 = <bos>
with torch.no_grad():
    before = model(ids)                                   # без порта
    print(torch.equal(before, forward_with_prefix(model, None, ids)))
# Вывод в книге:
# | True

# %% Листинг «Необученный префикс перед сказкой» — раздел «Мягкий префикс»
from booklm.experiments.ch06 import embed_rms   # длина строк эмбеддингов

with torch.no_grad():
    port = InputPort(PatchEncoder(3, 8, 48), 48, 256)     # не обучен
    prefix = port(torch.rand(1, 3, 32, 32))               # [1, 18, 256]
    rms = prefix[:, 1:-1].pow(2).sum(-1).mean().sqrt()    # длина участков
    prefix = prefix * embed_rms(model) / rms
    for name, p in (("без префикса", None), ("с префиксом", prefix)):
        story = "Жила-была" + tok.decode(greedy(model, p, ids, 20))
        print(textwrap.fill(f"{name}: {story}", 72))
    delta = forward_with_prefix(model, prefix, ids)[0, -1] - before[0, -1]
    print(f"наибольшее |Δ| логитов: {delta.abs().max():.2f}")
    del port, prefix                                      # снимаем порт
    print("после снятия логиты те же:", torch.equal(before, model(ids)))
# Вывод в книге:
# | без префикса: Жила-была маленькая девочка по имени Люси. Она любила
# | играть в саду. Однажды она нашла большую, блестящую камень.
# | с префиксом: Жила-была маленькая девочка по имени Люси. Она любила
# | играть с игрушками и всегда хотела исследовать. Однажды она нашла
# | большую
# | наибольшее |Δ| логитов: 2.82
# | после снятия логиты те же: True

# %% Листинг «Перекрёстное внимание с воротами» — раздел «Перекрёстное внимание с воротами»
class GatedCrossAttention(nn.Module):
    def __init__(self, width, signal_width):
        super().__init__()
        self.q = nn.Linear(width, width, bias=False)         # W_Q
        self.k = nn.Linear(signal_width, width, bias=False)  # W_K
        self.v = nn.Linear(signal_width, width, bias=False)  # W_V
        self.gate = nn.Parameter(torch.zeros(1))             # g = 0

    def forward(self, h, z):            # h: [B, T, d], z: [B, S, d_m]
        scores = self.q(h) @ self.k(z).transpose(1, 2) / h.size(-1) ** 0.5
        attended = scores.softmax(-1) @ self.v(z)            # A(H, Z)
        return h + torch.tanh(self.gate) * attended

torch.manual_seed(0)
xattn = GatedCrossAttention(width=8, signal_width=12)
h = torch.randn(2, 3, 8)                         # состояния трёх токенов
z = encoder(torch.randn(2, 3, 8, 8))             # 4 участка картинки
out = xattn(h, z)
print(torch.equal(out, h))                       # ворота закрыты
out.pow(2).sum().backward()                      # учебная ошибка
print(xattn.gate.grad, xattn.q.weight.grad.abs().sum())
# Вывод в книге:
# | True
# | tensor([2.1707]) tensor(0.)

# %% Листинг «Цена префикса: линейные слои, внимание и KV-кэш» — раздел «Длина префикса и вычисления»
def prefix_cost(T, K, bytes_per_token=1536):
    linear = (T + K) / T                       # работа по позициям
    attention = (T + K) ** 2 / T ** 2          # матрица оценок
    cache_kib = K * bytes_per_token / 1024     # добавка к KV-кэшу
    return linear, attention, cache_kib

for K in (256, 64):
    linear, attention, cache_kib = prefix_cost(T=50, K=K)
    print(f"K={K:3d}: линейные слои x{linear:.1f}, матрица внимания "
          f"x{attention:.1f}, KV-кэш +{cache_kib:.0f} КБ")
# Вывод в книге:
# | K=256: линейные слои x6.1, матрица внимания x37.5, KV-кэш +384 КБ
# | K= 64: линейные слои x2.3, матрица внимания x5.2, KV-кэш +96 КБ

# %% Листинг «Время до первого токена и на следующий токен» — раздел «Длина префикса и вычисления»
# Проверка этот листинг не запускает: ему нужны сеть, видеокарта, сервер или внешний файл.
# import statistics, time
# from booklm.lab import val_tokens
# from booklm.experiments.ch06 import generate_with_prefix
#
# torch.set_num_threads(1)                    # один поток: замеры ровнее
# data = val_tokens()
# start = int((data == 1).nonzero()[1])       # вторая отложенная сказка
# story = data[start:start + 50][None]        # 50 токенов, первый <bos>
# for K in (0, 64, 256):
#     prefix = torch.randn(1, K, 256) * embed_rms(model) / 16 if K else None
#     first, step = [], []
#     for _ in range(15):
#         clock = [time.perf_counter()]
#         _, caches = generate_with_prefix(model, prefix, story, 100, clock)
#         first.append(clock[1] - clock[0])   # вход и первый токен
#         gaps = [b - a for a, b in zip(clock[1:], clock[2:])]
#         step.append(statistics.median(gaps))     # следующий токен
#     numbers = sum(c["k"].numel() + c["v"].numel() for c in caches)
#     kib = numbers * 2 / 1024                # по 2 байта, как в BF16
#     print(f"K={K:3d}, вход {K + 50:3d}: первый токен "
#           f"{min(first) * 1e3:4.1f} мс, следующий {min(step) * 1e3:4.2f} мс, "
#           f"KV-кэш {kib:.0f} КБ")

# %% Листинг «Игрушечное ядро LLM, ошибка и пары «картинка — слово»» — раздел «Обучение соединителя»
class TinyCore(nn.Module):                     # игрушечное ядро LLM
    def __init__(self, vocab=50, width=8):
        super().__init__()
        self.embed = nn.Embedding(vocab, width)
        self.head = nn.Linear(width, vocab)

    def forward(self, inputs):                 # inputs: [B, L, width]
        steps = torch.arange(1, inputs.size(1) + 1)[:, None]  # столбец
        context = inputs.cumsum(dim=1) / steps # причинное среднее
        return self.head(context)              # [B, L, vocab]

def lm_loss(core, port, raw, text_ids):
    inputs, labels = port.pack(raw, text_ids, core.embed)
    logits = core(inputs)[:, :-1]              # t предсказывает t+1
    return F.cross_entropy(logits.transpose(1, 2), labels[:, 1:])

colors = torch.tensor([[1., 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])

def make_batch(n):                     # картинка цвета c -> слово 10+c
    c = torch.randint(0, 4, (n,))
    raw = colors[c][:, :, None, None] + 0.1 * torch.randn(n, 3, 8, 8)
    return raw, 10 + c[:, None]

# %% Листинг «Обучение соединителя при замороженном ядре» — раздел «Обучение соединителя»
torch.manual_seed(0)
core = TinyCore().requires_grad_(False)          # ядро заморожено
probe = torch.tensor([[1, 2, 5]])
before = core(core.embed(probe))                 # логиты текста до порта
port = InputPort(PatchEncoder(3, 4, 12), 12, 8)
port.encoder.requires_grad_(False)               # учим P_m и границы
trainable = [p for p in port.parameters() if p.requires_grad]
opt = torch.optim.Adam(trainable, lr=0.05)
for step in range(301):
    raw, text = make_batch(32)
    loss = lm_loss(core, port, raw, text)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 100 == 0:
        print(f"шаг {step:3d}: ошибка {loss.item():.3f}")
print(sum(p.numel() for p in trainable), "обучаемых чисел")
print(torch.equal(before, core(core.embed(probe))))  # ядро не изменилось
# Вывод в книге:
# | шаг   0: ошибка 4.011
# | шаг 100: ошибка 0.637
# | шаг 200: ошибка 0.351
# | шаг 300: ошибка 0.143
# | 120 обучаемых чисел
# | True

# %% Листинг «Проверка перемешиванием: свои картинки против чужих» — раздел «Проверка перемешиванием»
def modality_reliance_score(core, port, raw, text):
    with torch.no_grad():
        correct = lm_loss(core, port, raw, text)
        shuffled = lm_loss(core, port, raw.roll(1, dims=0), text)
    return correct.item(), shuffled.item()

torch.manual_seed(1)
raw, text = make_batch(256)
correct, shuffled = modality_reliance_score(core, port, raw, text)
print(f"после обучения: свои {correct:.3f}, чужие {shuffled:.3f}")
untrained = InputPort(PatchEncoder(3, 4, 12), 12, 8)
correct, shuffled = modality_reliance_score(core, untrained, raw, text)
print(f"до обучения:    свои {correct:.3f}, чужие {shuffled:.3f}")
# Вывод в книге:
# | после обучения: свои 0.168, чужие 7.298
# | до обучения:    свои 4.070, чужие 4.029

# %% Листинг «Соединитель для замороженной BookLM» — раздел «Проверка перемешиванием»
import copy
from booklm.experiments.ch06 import make_batch as story_batch
from booklm.experiments.ch06 import lm_loss as story_loss

torch.manual_seed(0)
model.requires_grad_(False)                      # BookLM заморожена
eye = InputPort(PatchEncoder(3, 8, 48), 48, 256)
fresh = copy.deepcopy(eye)                       # копия до обучения
eye.encoder.requires_grad_(False)                # учим P_m и границы
trainable = [p for p in eye.parameters() if p.requires_grad]
opt = torch.optim.Adam(trainable, lr=3e-3)
for step in range(301):
    raw, story = story_batch(tok, torch.randint(0, 4, (32,)))
    loss = story_loss(model, eye, raw, story)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 100 == 0:
        print(f"шаг {step:3d}: ошибка {loss.item():.3f}")
print(sum(p.numel() for p in trainable), "обучаемых чисел")
# Вывод в книге:
# | шаг   0: ошибка 2.616
# | шаг 100: ошибка 0.142
# | шаг 200: ошибка 0.043
# | шаг 300: ошибка 0.026
# | 13056 обучаемых чисел

# %% Листинг «Проверка перемешиванием и сказки по картинкам» — раздел «Проверка перемешиванием»
raw, story = story_batch(tok, torch.arange(64) % 4)   # цвета по кругу
with torch.no_grad():
    for name, p in (("после обучения", eye), ("до обучения   ", fresh)):
        own = story_loss(model, p, raw, story)
        other = story_loss(model, p, raw.roll(1, dims=0), story)
        print(f"{name}: свои {own:.3f}, чужие {other:.3f}")
    for c, name in enumerate(("рыжая", "серая", "белая", "чёрная")):
        bos = torch.tensor([[1]])
        tale = tok.decode(greedy(model, eye(raw[c:c + 1]), bos, 13))
        print(f"{name}: {' '.join(tale.split())}")
# Вывод в книге:
# | после обучения: свои 0.025, чужие 2.949
# | до обучения   : свои 2.609, чужие 2.676
# | рыжая: Жил-был рыжий кот. Кот был очень счастлив.
# | серая: Жила-была серая мышка. Магги увидела
# | белая: Жил-был белый зайчик. Он был очень красивый. Он
# | чёрная: Жил-был чёрный щенок. Он был очень добрый щенок. Он

# %% Листинг «Снятие порта» — раздел «Снятие порта»
ports = {"img": port}                      # подключённые порты системы

def answer(text_ids, raw=None, kind="img"):
    if raw is None:                        # чисто текстовый запрос
        return core(core.embed(text_ids))
    inputs, _ = ports[kind].pack(raw, text_ids, core.embed)
    return core(inputs)

picture, _ = make_batch(1)
print(answer(probe, picture).shape)        # картинка и текст
del ports["img"]                           # снимаем порт
print(sorted(ports), torch.equal(before, answer(probe)))
# Вывод в книге:
# | torch.Size([1, 9, 50])
# | [] True

# %% Листинг «Выходной порт: сигнальные токены, заморозка и обрыв градиента» — раздел «Выходные модальности»
class OutputPort(nn.Module):
    def __init__(self, lm_width, condition_width, decoder):
        super().__init__()
        self.projector = nn.Linear(lm_width, condition_width)  # R_m
        self.decoder = decoder.requires_grad_(False)   # D_m заморожен

    def forward(self, hidden, token_ids, signal_id):
        mask = token_ids.eq(signal_id)                 # [B, L]
        signal = hidden[mask].view(hidden.size(0), -1, hidden.size(-1))
        condition = self.projector(signal)             # [B, K_c, d_c]
        return self.decoder(condition), condition

torch.manual_seed(0)
out_port = OutputPort(lm_width=8, condition_width=4,
                      decoder=nn.Linear(4, 4))
hidden = torch.randn(2, 6, 8)                    # состояния LLM
token_ids = torch.tensor([[5, 7, 40, 40, 9, 3],
                          [6, 40, 40, 8, 2, 1]]) # 40 = <img>
picture, condition = out_port(hidden, token_ids, signal_id=40)
print(condition.shape, picture.shape)
picture.pow(2).sum().backward()                  # учебная ошибка
print(out_port.decoder.weight.grad)              # веса D_m стоят
print(out_port.projector.weight.grad.abs().sum() > 0)       # R_m учится
print(out_port.decoder(condition.detach()).requires_grad)   # обрыв
# Вывод в книге:
# | torch.Size([2, 2, 4]) torch.Size([2, 2, 4])
# | None
# | tensor(True)
# | False

# %% Листинг «Маска между текстовыми токенами и звуковыми кадрами» — раздел «Thinker и Talker»
words = ["Груз", "весит", "12", "кг"]
spans = [(1, 3), (4, 6), (7, 12), (13, 17)]   # кадры, озвучивающие слово
lookahead = 1                                 # речь видит на слово вперёд
word = torch.zeros(17, dtype=torch.long)      # какое слово звучит в кадре
for i, (a, b) in enumerate(spans):
    word[a - 1:b] = i
# mask[кадр, токен]: токен не дальше слова кадра + lookahead
mask = torch.arange(4)[None, :] <= word[:, None] + lookahead
for i, w in enumerate(words):
    print(f"{w:>5}", "".join(str(int(v)) for v in mask[:, i]))
print("n_text по кадрам:", mask.sum(dim=1).tolist())
# Вывод в книге:
# |  Груз 11111111111111111
# | весит 11111111111111111
# |    12 00011111111111111
# |    кг 00000011111111111
# | n_text по кадрам: [2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]

# %% Листинг «Передача сообщений» — раздел «Передача сообщений между соседями»
A = torch.tensor([[0., 1, 0, 0],       # Аня: Боря
                  [1, 0, 1, 1],        # Боря: Аня, Вика, Гоша
                  [0, 1, 0, 1],        # Вика: Боря, Гоша
                  [0, 1, 1, 0]])       # Гоша: Боря, Вика

def message_pass(A, h):
    return A @ h + h                   # соседи плюс свой вектор

h = torch.tensor([[1.], [0.], [0.], [0.]])    # секрет знает Аня
for step in (1, 2):
    h = message_pass(A, h)
    print(f"шаг {step}:", h.squeeze(1).tolist())
# Вывод в книге:
# | шаг 1: [1.0, 1.0, 0.0, 0.0]
# | шаг 2: [2.0, 2.0, 1.0, 1.0]

# %% Листинг «Графовый порт» — раздел «Подача графа в LLM»
class GraphEncoder(nn.Module):
    def __init__(self, feature_width, width):
        super().__init__()
        self.first = nn.Linear(feature_width, width)
        self.second = nn.Linear(width, width)

    def forward(self, raw):                   # raw: [B, N, N + F]
        n = raw.size(1)
        adj, x = raw[..., :n], raw[..., n:]   # соседи и признаки узла
        h = torch.relu(self.first(message_pass(adj, x)))
        return torch.relu(self.second(message_pass(adj, h)))

torch.manual_seed(0)
gender = torch.tensor([[1., 0], [0, 1], [1, 0], [0, 1]])  # девочка: (1, 0)
graph_port = InputPort(GraphEncoder(2, 12), encoder_width=12, lm_width=8)
out = graph_port(torch.cat([A, gender], dim=1)[None])
print(out.shape)                              # 4 узла + 2 границы
perm = torch.tensor([2, 0, 3, 1])             # другая нумерация героев
out_p = graph_port(torch.cat([A[perm][:, perm], gender[perm]], dim=1)[None])
print(torch.allclose(out[:, 1:-1][:, perm], out_p[:, 1:-1], atol=1e-6))
# Вывод в книге:
# | torch.Size([1, 6, 8])
# | True

# %% Листинг «Граф героев отложенных сказок» — раздел «Подача графа в LLM»
from booklm.experiments.ch06_graph import held_out_stories, hero_graph
from booklm.experiments.ch06_graph import adjacency, edges_as_text

nodes, edges = hero_graph(held_out_stories(tok))
print(len(nodes), "героев,", len(edges), "рёбер")
heroes = adjacency(nodes, edges)              # [16, 16]
heard = torch.eye(len(nodes))                 # каждый знает только себя
for step in (1, 2, 3):
    heard = message_pass(heroes, heard).clamp(max=1)
    count = heard.sum(1).long() - 1           # кроме себя
    print(f"шаг {step}:", {n: count[i].item() for i, n in enumerate(nodes)
                          if n in ("Тим", "Лили", "Анна")})
# Вывод в книге:
# | 16 героев, 30 рёбер
# | шаг 1: {'Тим': 11, 'Лили': 4, 'Анна': 1}
# | шаг 2: {'Тим': 14, 'Лили': 15, 'Анна': 3}
# | шаг 3: {'Тим': 15, 'Лили': 15, 'Анна': 8}

# %% Листинг «Граф текстом в подсказке» — раздел «Подача графа в LLM»
from booklm.experiments.ch06_graph import name_scores, friend_choice

graph_text = edges_as_text(edges)             # «Сью и Тим друзья. ...»
print(len(tok.encode(graph_text).ids), "токенов в тексте графа")
others = [n for n in nodes if n != "Анна"]
for name, head in (("без графа", ""), ("с графом ", graph_text + "\n")):
    s = name_scores(tok, model, head + "Однажды Анна и", others)
    print(name, sorted(s, key=s.get, reverse=True)[:3])
for m in (1, 2, 4, 8, 16, 30):                # по 100 случайных графов
    hit, base = friend_choice(tok, model, nodes, m, trials=100, seed=m)
    print(f"рёбер {m:2d}: с графом {hit:.2f}, без графа {base:.2f}")
# Вывод в книге:
# | 150 токенов в тексте графа
# | без графа ['Бен', 'Макс', 'Том']
# | с графом  ['Тим', 'Спот', 'Том']
# | рёбер  1: с графом 0.84, без графа 0.06
# | рёбер  2: с графом 0.51, без графа 0.07
# | рёбер  4: с графом 0.26, без графа 0.02
# | рёбер  8: с графом 0.19, без графа 0.06
# | рёбер 16: с графом 0.06, без графа 0.09
# | рёбер 30: с графом 0.10, без графа 0.06
