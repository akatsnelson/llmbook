"""KV-кэш настоящей BookLM: пошаговый и кусочный проход против полного, пакет B > 1.

Полная версия опыта «KV-кэш обученной BookLM: сверка с полным проходом» из главы 14 книги
(раздел 14.4 «Кэш ключей и значений»). В книге кэш показан на
учебном стенде, на одной голове внимания и на обученной BookLM с одним текстом. Здесь та же проверка идёт
на классе BookLM из booklm/model.py со случайными весами (файлы модели не нужны), с пакетом из двух текстов
и кусками разной длины. Отдельной реализации кэша здесь нет: проверяется тот код, который работает в книге.
Последняя проверка нарочно ломает расчёт: если не сдвигать cache["len"], RoPE получает неверные позиции,
формы тензоров те же, а логиты расходятся.

Запуск из корня репозитория: python labs/ch14/kv_cache_attention.py
"""

import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from booklm.model import BookLM, BookLMConfig  # noqa: E402


@torch.no_grad()
def run_in_pieces(model, ids, size, advance=True):
    """Читает ids кусками по size токенов поверх кэша; advance=False забывает сдвинуть позицию."""
    caches = [{"len": 0} for _ in model.blocks]
    pieces = []
    for s in range(0, ids.size(1), size):
        pieces.append(model(ids[:, s:s + size], caches))
        if advance:
            for c in caches:
                c["len"] += pieces[-1].size(1)
    return torch.cat(pieces, dim=1), caches


def main():
    torch.manual_seed(0)
    cfg = BookLMConfig()
    model = BookLM(cfg).eval()
    ids = torch.randint(0, cfg.vocab, (2, 23))     # пакет из двух текстов по 23 токена

    with torch.no_grad():
        full = model(ids)                                # весь текст разом, без кэша

    for size in (1, 3, 5, 23):
        logits, caches = run_in_pieces(model, ids, size)
        err = (logits - full).abs().max().item()
        assert err < 1e-4, f"кусками по {size}: расхождение {err}"
        k = caches[0]["k"]
        assert tuple(k.shape) == (2, cfg.kv_heads, 23, cfg.d_model // cfg.heads), k.shape
        print(f"кусками по {size:>2}: расхождение с полным {err:.1e}, ключи блока 0 {tuple(k.shape)}")

    broken, _ = run_in_pieces(model, ids, 1, advance=False)
    err = (broken - full).abs().max().item()
    assert broken.shape == full.shape
    scale = full.abs().max().item()
    # у случайных весов логиты малы, поэтому порог относительный: в тысячу раз выше погрешности кэша
    assert err > 1e-3 * scale and err > 1000 * 1e-6, "без сдвига позиции логиты обязаны разойтись"
    print(f"без сдвига cache['len']: формы те же, расхождение {err:.1e} при логитах до {scale:.2f}")


if __name__ == "__main__":
    main()
