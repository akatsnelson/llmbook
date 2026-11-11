"""Проверки главы 6 «Мультимодальные модели»: все листинги главы, переписанные утверждениями.

В книге листинги печатают результат, здесь те же вычисления сверяются с числами из текста
через assert. Запустите файл после любой правки кода главы: он скажет, какое число перестало
сходиться.

Запуск из корня репозитория: python labs/ch06/multimodal_checks.py
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --- CLIP и SigLIP --------------------------------------------------------------------------

def clip_loss(image_features, text_features, temperature=0.1):
    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)
    logits = image_features @ text_features.T / temperature
    target = torch.arange(logits.size(0))
    image_to_text = F.cross_entropy(logits, target)
    text_to_image = F.cross_entropy(logits.T, target)
    return 0.5 * (image_to_text + text_to_image), logits


def siglip_loss(logits, bias):
    s = logits + bias
    y = 2 * torch.eye(s.size(0)) - 1
    return -F.logsigmoid(y * s).mean()


def check_clip_siglip():
    images = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    texts = torch.tensor([[0.8, 0.6], [0.6, 0.8]])
    loss, logits = clip_loss(images, texts)
    torch.testing.assert_close(logits, torch.tensor([[8.0, 6.0], [6.0, 8.0]]))
    torch.testing.assert_close(loss, torch.tensor(0.126928), atol=1e-5, rtol=0)
    swapped, _ = clip_loss(images, texts.flip(0))
    torch.testing.assert_close(swapped - loss, torch.tensor(2.0), atol=1e-5, rtol=0)
    sharp, _ = clip_loss(images, texts, temperature=0.05)
    assert abs(sharp.item() - (-math.log(1 / (1 + math.exp(-4))))) < 1e-5
    torch.testing.assert_close(siglip_loss(logits, -7.0), torch.tensor(0.313262), atol=1e-5, rtol=0)
    torch.testing.assert_close(siglip_loss(logits.flip(0), -7.0), torch.tensor(1.313262),
                               atol=1e-5, rtol=0)


# --- Кодировщик изображения ---------------------------------------------------------------

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


def check_patches():
    torch.manual_seed(0)
    z = PatchEncoder(channels=3, patch_size=4, width=12)(torch.randn(2, 3, 8, 8))
    assert z.shape == (2, 4, 12)
    # 224 x 224 с участком 14 даёт 256 позиций, вдвое большая сторона — вчетверо больше
    assert (224 // 14) ** 2 == 256 and (448 // 14) ** 2 == 4 * 256
    # участок в ряду 1 и столбце 1 сетки 16 x 16 получает номер 17: картинка 32 x 32 из номеров
    # пикселей, участок 2 x 2; левый верхний пиксель участка 17 — пиксель (2, 2), то есть 66
    image = torch.arange(32 * 32, dtype=torch.float32).view(1, 1, 32, 32)
    patches = image.unfold(2, 2, 2).unfold(3, 2, 2).permute(0, 2, 3, 1, 4, 5).reshape(1, 256, 4)
    assert patches[0, 17, 0].item() == 2 * 32 + 2


# --- Звук ---------------------------------------------------------------------------------

def frame_signal(wave, frame_size, hop):
    count = 1 + (len(wave) - frame_size) // hop
    if count <= 0:
        raise ValueError("signal is shorter than one frame")
    starts = [i * hop for i in range(count)]
    return np.stack([wave[s:s + frame_size] for s in starts])


def stft(wave, frame_size=400, hop=160):
    frames = frame_signal(wave, frame_size, hop)
    window = np.hanning(frame_size)
    return np.fft.rfft(frames * window, axis=-1)


def check_audio():
    fs = 32
    t = np.arange(fs) / fs
    x20 = np.cos(2 * np.pi * 20 * t)
    x12 = np.cos(2 * np.pi * 12 * t)
    assert np.allclose(x20, x12)
    assert np.abs(np.fft.rfft(x20)).argmax() == 12

    cos_wave = np.array([1.0, 0.0, -1.0, 0.0])
    sin_wave = np.array([0.0, 1.0, 0.0, -1.0])
    assert np.allclose(np.fft.fft(cos_wave), [0, 2, 0, 2])
    assert np.allclose(np.fft.fft(sin_wave), [0, -2j, 0, 2j])
    assert np.allclose(np.abs(np.fft.fft(cos_wave)), np.abs(np.fft.fft(sin_wave)))

    fs = 16000
    t = np.arange(fs) / fs
    spectrum = stft(np.sin(2 * np.pi * 440 * t))
    assert spectrum.shape == (98, 201)
    peak = (np.abs(spectrum) ** 2).mean(axis=0).argmax()
    assert peak == 11 and peak * fs / 400 == 440.0
    assert stft(np.zeros(8000)).shape[0] == 1 + (8000 - 400) // 160   # полсекунды: 48 окон

    mel = lambda f: 2595 * math.log10(1 + f / 700)
    assert round(mel(1000)) == 1000 and round(mel(8000)) == 2840
    assert round(mel(200) - mel(100)) == 133 and round(mel(8100) - mel(8000)) == 13


# --- Входной порт, игрушечное ядро, обучение соединителя ----------------------------------

class InputPort(nn.Module):
    def __init__(self, encoder, encoder_width, lm_width):
        super().__init__()
        self.encoder = encoder
        self.projector = nn.Linear(encoder_width, lm_width)
        self.start = nn.Parameter(torch.zeros(1, 1, lm_width))
        self.end = nn.Parameter(torch.zeros(1, 1, lm_width))

    def forward(self, raw):
        tokens = self.projector(self.encoder(raw))
        edge = lambda v: v.expand(raw.size(0), -1, -1)
        return torch.cat([edge(self.start), tokens, edge(self.end)], dim=1)

    def pack(self, raw, text_ids, embed):
        prefix = self(raw)
        inputs = torch.cat([prefix, embed(text_ids)], dim=1)
        ignored = torch.full(prefix.shape[:2], -100, dtype=text_ids.dtype)
        labels = torch.cat([ignored, text_ids], dim=1)
        return inputs, labels


class TinyCore(nn.Module):
    def __init__(self, vocab=50, width=8):
        super().__init__()
        self.embed = nn.Embedding(vocab, width)
        self.head = nn.Linear(width, vocab)

    def forward(self, inputs):
        steps = torch.arange(1, inputs.size(1) + 1)[:, None]
        return self.head(inputs.cumsum(dim=1) / steps)


def lm_loss(core, port, raw, text_ids):
    inputs, labels = port.pack(raw, text_ids, core.embed)
    logits = core(inputs)[:, :-1]
    return F.cross_entropy(logits.transpose(1, 2), labels[:, 1:])


COLORS = torch.tensor([[1., 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])


def make_batch(n):
    c = torch.randint(0, 4, (n,))
    raw = COLORS[c][:, :, None, None] + 0.1 * torch.randn(n, 3, 8, 8)
    return raw, 10 + c[:, None]


def modality_reliance_score(core, port, raw, text):
    with torch.no_grad():
        correct = lm_loss(core, port, raw, text)
        shuffled = lm_loss(core, port, raw.roll(1, dims=0), text)
    return correct.item(), shuffled.item()


def check_input_port():
    torch.manual_seed(0)
    port = InputPort(PatchEncoder(3, 4, 12), encoder_width=12, lm_width=8)
    embed = nn.Embedding(50, 8)
    text_ids = torch.tensor([[5, 7, 9], [1, 2, 3]])
    inputs, labels = port.pack(torch.randn(2, 3, 8, 8), text_ids, embed)
    assert inputs.shape == (2, 4 + 2 + 3, 8)
    assert (labels[:, :6] == -100).all() and torch.equal(labels[:, 6:], text_ids)


def check_training_and_reliance():
    torch.manual_seed(0)
    core = TinyCore().requires_grad_(False)
    probe = torch.tensor([[1, 2, 5]])
    before = core(core.embed(probe))
    port = InputPort(PatchEncoder(3, 4, 12), 12, 8)
    port.encoder.requires_grad_(False)
    trainable = [p for p in port.parameters() if p.requires_grad]
    assert sum(p.numel() for p in trainable) == 12 * 8 + 8 + 8 + 8   # 120
    opt = torch.optim.Adam(trainable, lr=0.05)
    losses = []
    for _ in range(301):
        raw, text = make_batch(32)
        loss = lm_loss(core, port, raw, text)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert abs(losses[0] - math.log(50)) < 0.3          # старт около ln 50
    assert losses[-1] < 0.3                             # соединитель выучил цвета
    assert torch.equal(before, core(core.embed(probe)))  # ядро не сдвинулось

    torch.manual_seed(1)
    raw, text = make_batch(256)
    correct, shuffled = modality_reliance_score(core, port, raw, text)
    assert shuffled - correct > 3.0                     # чужие картинки намного хуже
    untrained = InputPort(PatchEncoder(3, 4, 12), 12, 8)
    correct, shuffled = modality_reliance_score(core, untrained, raw, text)
    assert abs(shuffled - correct) < 0.2                # необученному порту всё равно


# --- Цена префикса ------------------------------------------------------------------------

def prefix_cost(T, K, bytes_per_token=1536):
    return (T + K) ** 2 / T ** 2, K * bytes_per_token / 1024


def check_cost():
    # 1536 байт на токен: 6 блоков x (K и V) x 2 KV-головы x 32 числа x 2 байта BF16
    assert 6 * 2 * 2 * 32 * 2 == 1536 == 768 * 1024 // 512
    attention, cache_kib = prefix_cost(50, 256)
    assert round(attention, 1) == 37.5 and cache_kib == 384
    attention, cache_kib = prefix_cost(50, 64)
    assert round(attention, 1) == 5.2 and cache_kib == 96


# --- Выходной порт ------------------------------------------------------------------------

class OutputPort(nn.Module):
    def __init__(self, lm_width, condition_width, decoder):
        super().__init__()
        self.projector = nn.Linear(lm_width, condition_width)
        self.decoder = decoder.requires_grad_(False)

    def forward(self, hidden, token_ids, signal_id):
        mask = token_ids.eq(signal_id)
        counts = mask.sum(dim=1)
        if not bool((counts == counts[0]).all()):
            raise ValueError("different number of signal tokens in batch")
        signal = hidden[mask].view(hidden.size(0), -1, hidden.size(-1))
        condition = self.projector(signal)
        return self.decoder(condition), condition


def check_output_port():
    torch.manual_seed(0)
    out_port = OutputPort(lm_width=8, condition_width=4, decoder=nn.Linear(4, 4))
    hidden = torch.randn(2, 6, 8)
    token_ids = torch.tensor([[5, 7, 40, 40, 9, 3], [6, 40, 40, 8, 2, 1]])
    picture, condition = out_port(hidden, token_ids, signal_id=40)
    assert condition.shape == picture.shape == (2, 2, 4)
    torch.testing.assert_close(condition[1, 0], out_port.projector(hidden[1, 1]))
    picture.pow(2).sum().backward()
    assert out_port.decoder.weight.grad is None
    assert out_port.projector.weight.grad.abs().sum() > 0
    assert not out_port.decoder(condition.detach()).requires_grad
    try:
        out_port(hidden, torch.tensor([[40, 7, 40, 40, 9, 3], [6, 40, 40, 8, 2, 1]]), 40)
    except ValueError:
        pass
    else:
        raise AssertionError("unequal number of signal tokens must be rejected")


# --- Ворота перекрёстного внимания и пример диффузии -------------------------------------

def check_gate_and_noise():
    torch.manual_seed(0)
    h, z = torch.randn(1, 3, 8), torch.randn(1, 5, 8)
    attention = nn.MultiheadAttention(8, 2, batch_first=True)
    g = torch.zeros(())
    out = h + torch.tanh(g) * attention(h, z, z)[0]
    assert torch.equal(out, h)                          # g = 0: блок ничего не меняет
    alpha_bar, z0, eps = 0.64, 1.0, 0.5
    z_t = math.sqrt(alpha_bar) * z0 + math.sqrt(1 - alpha_bar) * eps
    assert abs(z_t - 1.1) < 1e-12 and abs((eps - 0.4) ** 2 - 0.01) < 1e-12


def main():
    check_clip_siglip()
    check_patches()
    check_audio()
    check_input_port()
    check_training_and_reliance()
    check_cost()
    check_output_port()
    check_gate_and_noise()
    print("OK: все числа главы 6 сошлись")


if __name__ == "__main__":
    main()
