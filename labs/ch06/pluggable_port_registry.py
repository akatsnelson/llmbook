"""Съёмный входной порт и реестр портов.

Код к разделу 6.5 «Съёмный порт» главы 6 (листинг «Снятие порта» в подразделе 6.5.4).
В книге оставлена сокращённая версия; здесь — класс реестра с attach/detach,
защитой от повторного имени и проверкой, что после снятия порта текстовые логиты
совпадают с исходными бит в бит.
Запуск из корня репозитория: python labs/ch06/pluggable_port_registry.py
"""

import torch
import torch.nn as nn


class InputPort(nn.Module):
    def __init__(self, encoder, encoder_width, lm_width):
        super().__init__()
        self.encoder = encoder
        self.projector = nn.Linear(encoder_width, lm_width)
        self.start = nn.Parameter(torch.zeros(1, 1, lm_width))
        self.end = nn.Parameter(torch.zeros(1, 1, lm_width))

    def forward(self, raw):
        features = self.encoder(raw)           # [B, S_m, d_m]
        tokens = self.projector(features)      # [B, S_m, d_lm]
        batch = raw.size(0)
        return torch.cat([
            self.start.expand(batch, -1, -1),
            tokens,
            self.end.expand(batch, -1, -1),
        ], dim=1)


class PortRegistry(nn.Module):
    def __init__(self):
        super().__init__()
        self.ports = nn.ModuleDict()

    def attach(self, name, port):
        if name in self.ports:
            raise KeyError(f"port already exists: {name}")
        self.ports[name] = port

    def detach(self, name):
        return self.ports.pop(name)

    def encode(self, name, raw):
        return self.ports[name](raw)


def main():
    # Проверка: attach, encode, detach
    torch.manual_seed(0)
    lm = nn.Sequential(nn.Embedding(50, 8), nn.Linear(8, 50))   # ядро
    text_ids = torch.randint(0, 50, (2, 5))
    with torch.no_grad():
        before = lm(text_ids)
    registry = PortRegistry()
    registry.attach("img", InputPort(nn.Linear(12, 6), 6, 8))
    try:
        registry.attach("img", InputPort(nn.Linear(12, 6), 6, 8))
    except KeyError:
        pass
    else:
        raise AssertionError("second port with the same name must be rejected")
    image_tokens = registry.encode("img", torch.randn(2, 4, 12))
    assert image_tokens.shape == (2, 4 + 2, 8)   # S_m + 2
    registry.detach("img")
    with torch.no_grad():
        after = lm(text_ids)
    assert len(registry.ports) == 0
    assert torch.equal(before, after)
    print("OK: port attached, encoded, detached; text logits unchanged")


if __name__ == "__main__":
    main()
