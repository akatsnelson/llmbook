"""Проверки к главе 3: все листинги главы в виде утверждений assert.

В книге листинги печатают результат, и читатель сверяет его
глазами. Здесь те же функции собраны в один самодостаточный файл и проверены утверждениями, чтобы любую
правку кода главы можно было проверить одной командой.
Запуск из корня репозитория: python labs/ch03/text_to_batch_checks.py
"""

import unicodedata
from collections import Counter

import torch
import torch.nn.functional as F


# --- BPE: обучение слияний ------------------------------------------------------------------------

def merge_pair(pieces, pair):
    result, i = [], 0
    while i < len(pieces):
        if i + 1 < len(pieces) and (pieces[i], pieces[i + 1]) == pair:
            result.append(pieces[i] + pieces[i + 1])
            i += 2
        else:
            result.append(pieces[i])
            i += 1
    return tuple(result)


def learn_merges(word_frequencies, steps):
    corpus = Counter({tuple(w): c for w, c in word_frequencies.items()})
    merges = []
    for _ in range(steps):
        counts = Counter()
        for pieces, frequency in corpus.items():
            for pair in zip(pieces, pieces[1:]):
                counts[pair] += frequency
        if not counts:
            break
        best = min(counts, key=lambda pair: (-counts[pair], pair))
        merges.append(best)
        updated = Counter()
        for pieces, frequency in corpus.items():
            updated[merge_pair(pieces, best)] += frequency
        corpus = updated
    return merges, corpus


# --- Кодирование, декодирование, проверка туда-обратно --------------------------------------------

def build_vocab(alphabet, merges):
    pieces = ["<pad>", "<unk>"] + list(alphabet)
    pieces += [a + b for a, b in merges]
    return {piece: number for number, piece in enumerate(pieces)}


def encode(text, merges, vocab):
    pieces = tuple(text)
    for pair in merges:
        pieces = merge_pair(pieces, pair)
    return [vocab.get(p, vocab["<unk>"]) for p in pieces]


def decode(ids, vocab):
    piece_of = {number: piece for piece, number in vocab.items()}
    return "".join(piece_of[number] for number in ids)


def roundtrip_failures(texts, merges, vocab):
    return [s for s in texts if decode(encode(s, merges, vocab), vocab) != s]


# --- Байтовый запасной путь -----------------------------------------------------------------------

def build_vocab_bytes(alphabet, merges):
    vocab = build_vocab(alphabet, merges)
    for b in range(256):
        vocab[f"<0x{b:02X}>"] = len(vocab)
    return vocab


def encode_bytes(text, merges, vocab):
    pieces = tuple(text)
    for pair in merges:
        pieces = merge_pair(pieces, pair)
    ids = []
    for p in pieces:
        if p in vocab:
            ids.append(vocab[p])
        else:
            ids += [vocab[f"<0x{b:02X}>"] for b in p.encode("utf-8")]
    return ids


def decode_bytes(ids, vocab):
    piece_of = {number: piece for piece, number in vocab.items()}
    data = b""
    for number in ids:
        piece = piece_of[number]
        if piece.startswith("<0x"):
            data += bytes([int(piece[3:5], 16)])
        else:
            data += piece.encode("utf-8")
    return data.decode("utf-8")


# --- Сдвиг, softmax, перекрёстная энтропия --------------------------------------------------------

def next_token_examples(token_ids):
    return token_ids[:, :-1], token_ids[:, 1:]


def cross_entropy_one(logits, target):
    shifted = logits - logits.max()
    log_sum_exp = shifted.exp().sum().log()
    return -shifted[target] + log_sum_exp


def main():
    # Символы и байты.
    assert "м".encode("utf-8") == b"\xd0\xbc"
    assert len("мама") == 4 and len("мама".encode("utf-8")) == 8
    assert [len(ch.encode("utf-8")) for ch in ["m", "м", "\U0001F642"]] == [1, 2, 4]

    # Три слияния BPE.
    merges, corpus = learn_merges({"мама": 2, "мамы": 1}, steps=3)
    assert merges == [("м", "а"), ("ма", "ма"), ("м", "ы")]
    assert dict(corpus) == {("мама",): 2, ("ма", "мы"): 1}
    assert merge_pair(("а", "а", "а"), ("а", "а")) == ("аа", "а")

    # Кодирование и декодирование.
    vocab = build_vocab("маы", merges)
    assert vocab == {"<pad>": 0, "<unk>": 1, "м": 2, "а": 3, "ы": 4, "ма": 5, "мама": 6, "мы": 7}
    assert encode("мамы", merges, vocab) == [5, 7]
    assert decode([5, 7], vocab) == "мамы"
    assert encode("мам", merges, vocab) == [5, 2]

    # Проверка туда-обратно находит потери из-за <unk>.
    assert encode("мамаша", merges, vocab) == [6, 1, 3]
    assert decode([6, 1, 3], vocab) == "мама<unk>а"
    texts = ["", "мама", "Мама", "ма ма", "\U0001F642"]
    assert roundtrip_failures(texts, merges, vocab) == ["Мама", "ма ма", "\U0001F642"]

    # Байтовый запасной путь возвращает все строки.
    vocab_b = build_vocab_bytes("маы", merges)
    assert len(vocab_b) == 264
    assert encode_bytes("мамаша", merges, vocab_b) == [6, 8 + 0xD1, 8 + 0x88, 3]
    assert all(decode_bytes(encode_bytes(s, merges, vocab_b), vocab_b) == s for s in texts)
    assert len(encode_bytes("\U0001F642", merges, vocab_b)) == 4

    # Порядок слияний: текст восстанавливается, номера другие.
    swapped = [("м", "а"), ("м", "ы"), ("ма", "ма")]
    vocab2 = build_vocab("маы", swapped)
    assert roundtrip_failures(["мама", "мамы", "мам"], swapped, vocab2) == []
    assert encode("мама", merges, vocab) == [6]
    assert encode("мама", swapped, vocab2) == [7]

    # Нормализация Unicode: «й» одним символом и «и» с отдельной краткой.
    one, two = "й", "й"
    assert one != two and unicodedata.normalize("NFC", two) == one
    assert len(one.encode("utf-8")) == 2 and len(two.encode("utf-8")) == 4

    # Таблица эмбеддингов: выбор строк.
    table = torch.tensor([[0.1, -0.3, 0.2], [-0.5, 0.4, 0.0], [0.7, 0.1, -0.2],
                          [-0.1, -0.6, 0.3], [0.2, 0.9, -0.4], [-0.8, 0.5, 0.6]])
    x = table[torch.tensor([[4, 4], [2, 3]])]
    assert x.shape == (2, 2, 3)
    torch.testing.assert_close(x[0, 0], table[4])
    torch.testing.assert_close(x[0, 1], table[4])
    torch.testing.assert_close(x[1], table[[2, 3]])

    # Сдвиг на одну позицию.
    inputs, targets = next_token_examples(torch.tensor([[11, 24, 7, 18, 5]]))
    assert inputs.tolist() == [[11, 24, 7, 18]]
    assert targets.tolist() == [[24, 7, 18, 5]]

    # Softmax и ошибка одного прогноза.
    probs = torch.tensor([2.0, 1.0, 0.0]).softmax(dim=-1)
    torch.testing.assert_close(probs.sum(), torch.tensor(1.0))
    torch.testing.assert_close(probs[0], torch.tensor(0.665241), atol=1e-6, rtol=0)
    loss = cross_entropy_one(torch.tensor([2.0, 1.0, 0.0]), target=0)
    torch.testing.assert_close(loss, torch.tensor(0.407606), atol=1e-6, rtol=0)
    torch.testing.assert_close(-probs[0].log(), loss)

    # Прямой счёт переполняется, а со сдвигом на максимум всё в порядке.
    assert torch.isinf(torch.tensor(1000.0).exp())
    big = cross_entropy_one(torch.tensor([1002.0, 1001.0, 1000.0]), target=0)
    torch.testing.assert_close(big, loss)

    # Средняя ошибка без заполнителя.
    pad = 0
    stream = torch.tensor([[8, 4, 9, 5], [3, 7, pad, pad]])
    inputs, targets = next_token_examples(stream)
    assert inputs.tolist() == [[8, 4, 9], [3, 7, 0]]
    assert targets.tolist() == [[4, 9, 5], [7, 0, 0]]
    torch.manual_seed(0)
    logits = torch.randn(2, 3, 10)
    per_token = F.cross_entropy(logits.reshape(-1, 10), targets.reshape(-1),
                                reduction="none").reshape(2, 3)
    torch.testing.assert_close(per_token[0, 0], cross_entropy_one(logits[0, 0], 4))
    mask = (targets != pad).float()
    assert mask.sum().item() == 4
    masked = (per_token * mask).sum() / mask.sum()
    same = F.cross_entropy(logits.reshape(-1, 10), targets.reshape(-1), ignore_index=pad)
    torch.testing.assert_close(masked, same)
    assert not torch.isclose(masked, per_token.mean())

    print("labs/ch03/text_to_batch_checks.py: все проверки пройдены")


if __name__ == "__main__":
    main()
