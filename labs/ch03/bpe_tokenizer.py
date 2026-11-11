"""Учебный BPE: обучение слияний, кодирование, декодирование и проверка туда-обратно.

Полная версия листингов «Один цикл слияний BPE» и «Кодирование и декодирование тремя слияниями» из главы 3
книги (разделы 3.2 «Кодирование парами байтов» и 3.3 «Кодирование и декодирование»). В книге оставлена
сокращённая версия; здесь весь код целиком, с теми же проверками.
Запуск из корня репозитория: python labs/ch03/bpe_tokenizer.py
"""

import unicodedata
from collections import Counter


# --- Обучение слияний -------------------------------------------------------

def pair_counts(corpus):
    # corpus: dict[tuple[str, ...], int] — слово, разбитое на фрагменты, и его частота
    counts = Counter()
    for pieces, frequency in corpus.items():
        for left, right in zip(pieces, pieces[1:]):
            counts[(left, right)] += frequency
    return counts


def merge_pair(pieces, pair):
    # Идём слева направо; склеенная пара больше не участвует в следующей паре.
    result = []
    i = 0
    while i < len(pieces):
        if i + 1 < len(pieces) and (pieces[i], pieces[i + 1]) == pair:
            result.append(pieces[i] + pieces[i + 1])
            i += 2
        else:
            result.append(pieces[i])
            i += 1
    return tuple(result)


def learn_merges(word_frequencies, steps):
    corpus = {tuple(word): count for word, count in word_frequencies.items()}
    merges = []
    for _ in range(steps):
        counts = pair_counts(corpus)
        if not counts:
            break
        # Наибольшая частота; при равенстве — первая по алфавиту пара.
        # Минус превращает «самую частую» в «самую маленькую», и один min учитывает оба правила.
        best = min(counts, key=lambda pair: (-counts[pair], pair))
        merges.append(best)
        updated = Counter()
        for pieces, frequency in corpus.items():
            updated[merge_pair(pieces, best)] += frequency
        corpus = dict(updated)
    return merges, corpus


# --- Кодирование и декодирование -------------------------------------------

def build_vocab(alphabet, merges):
    # Сначала служебные токены, потом алфавит, потом склейки в порядке обучения.
    pieces = ["<pad>", "<unk>", *alphabet] + [left + right for left, right in merges]
    return {piece: number for number, piece in enumerate(pieces)}


def encode(text, merges, vocab):
    pieces = tuple(text)
    for pair in merges:  # строго в выученном порядке
        pieces = merge_pair(pieces, pair)
    return [vocab.get(piece, vocab["<unk>"]) for piece in pieces]


def decode(ids, vocab):
    piece_of = {number: piece for piece, number in vocab.items()}
    return "".join(piece_of[number] for number in ids)


def roundtrip_failures(texts, merges, vocab):
    return [text for text in texts if decode(encode(text, merges, vocab), vocab) != text]


def main():
    # Неперекрывающиеся вхождения: в «а а а» пара «а-а» склеивается один раз.
    assert merge_pair(("а", "а", "а"), ("а", "а")) == ("аа", "а")

    counts = pair_counts({tuple("мама"): 2, tuple("мамы"): 1})
    assert counts[("м", "а")] == 5 and counts[("а", "м")] == 3 and counts[("м", "ы")] == 1

    merges, corpus = learn_merges({"мама": 2, "мамы": 1}, steps=2)
    assert merges == [("м", "а"), ("ма", "ма")]
    assert corpus[("мама",)] == 2
    assert corpus[("ма", "м", "ы")] == 1

    # Третий шаг: ничья 1:1, её решает правило, а не случай.
    merges, corpus = learn_merges({"мама": 2, "мамы": 1}, steps=3)
    assert merges[2] == ("м", "ы")
    assert corpus[("ма", "мы")] == 1

    vocab = build_vocab("маы", merges)  # м:2, а:3, ы:4, ма:5, мама:6, мы:7
    assert vocab == {"<pad>": 0, "<unk>": 1, "м": 2, "а": 3, "ы": 4, "ма": 5, "мама": 6, "мы": 7}
    assert encode("мамы", merges, vocab) == [5, 7]
    assert decode([5, 7], vocab) == "мамы"
    assert encode("мам", merges, vocab) == [5, 2]

    # Невосстановимо: буквы «ш» нет в словаре, вместо неё встаёт <unk>.
    assert encode("мамаша", merges, vocab) == [6, 1, 3]
    assert decode([6, 1, 3], vocab) == "мама<unk>а"

    # Проверка туда-обратно на неудобных строках.
    texts = ["", "мама", "Мама", "ма ма", "\U0001F642"]
    assert roundtrip_failures(texts, merges, vocab) == ["Мама", "ма ма", "\U0001F642"]

    # Переставили два последних слияния: текст восстанавливается, номера другие.
    swapped = [("м", "а"), ("м", "ы"), ("ма", "ма")]
    vocab2 = build_vocab("маы", swapped)  # мы:6, мама:7
    assert roundtrip_failures(["мама", "мамы", "мам"], swapped, vocab2) == []
    assert encode("мама", merges, vocab) == [6]
    assert encode("мама", swapped, vocab2) == [7]

    # Нормализация Unicode: «й» одним символом и «и» + отдельная кратка выглядят одинаково.
    one = "й"
    two = "й"
    assert one != two
    assert len(one.encode("utf-8")) == 2 and len(two.encode("utf-8")) == 4
    assert unicodedata.normalize("NFC", two) == one

    print("labs/ch03/bpe_tokenizer.py: все проверки пройдены")


if __name__ == "__main__":
    main()
