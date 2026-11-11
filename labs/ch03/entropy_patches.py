"""Участки байтов по энтропии следующего байта, как в Byte Latent Transformer, на сказках BookLM.

Полная версия листингов «Цена текста в токенах BookLM», «Байтовая n-граммная модель», «Участки по
глобальному порогу» из главы 3 (подразделы 3.5.7 «Модели без
токенизатора» и 3.5.8 «Участки по энтропии»). Здесь те же числа проверены утверждениями assert, а сверх книги
есть сравнения:

* фиксированные участки (как в MEGABYTE) и участки по энтропии (как в BLT) почти одной средней
  длины, как у токена BookLM; для сравнения — участки после пробела (как в SpaceByte), их длину
  не подобрать;
* как средняя длина участка зависит от длины контекста n-граммной модели и от объёма данных;
* второе правило BLT, «приблизительная монотонность»: новый участок там, где энтропия подскочила
  по сравнению с предыдущим байтом.

Энтропию даёт байтовая n-граммная модель: счётчики следующего байта после K предыдущих байтов, с
откатом к более короткому контексту, если длинный встречался меньше MIN_COUNT раз. Учим её на первых
200 тысячах токенов обучающего потока BookLM (около 1,5 МБ текста), проверяем на отложенных историях.
Случайности в опыте нет: числа одинаковы при каждом запуске.

Запуск из корня репозитория: python labs/ch03/entropy_patches.py
Считается около полуминуты на обычном процессоре.
"""

import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from booklm.lab import MODELS, load_tokenizer, val_tokens  # noqa: E402

K = 6              # байтов контекста: три русские буквы
MIN_COUNT = 5      # контекст, который видели реже, заменяем более коротким
TRAIN_TOKENS = 200_000


# --- Данные --------------------------------------------------------------------------------------

def stories_bytes(tok, ids):
    """Поток номеров BookLM -> байты UTF-8; истории разделены переводом строки."""
    text = tok.decode(ids.tolist(), skip_special_tokens=False)
    return text.replace("<bos>", "").replace("<eos>", "\n").encode("utf-8")


# --- Байтовая n-граммная модель ------------------------------------------------------------------

def fit_counts(train, k=K):
    """Для каждого контекста длины 0..k — счётчик байта, который шёл следом."""
    counts = defaultdict(Counter)
    for n in range(k + 1):
        grams = Counter(train[i:i + n + 1] for i in range(len(train) - n))
        for gram, c in grams.items():
            counts[gram[:-1]][gram[-1]] = c
    totals = {ctx: sum(c.values()) for ctx, c in counts.items()}
    return counts, totals


def entropy_at(model, data, i, k=K, min_count=MIN_COUNT):
    """Энтропия (в битах) распределения байта data[i] при известных data[:i]."""
    counts, totals = model
    for n in range(min(k, i), -1, -1):
        ctx = data[i - n:i]
        if totals.get(ctx, 0) >= min_count:
            break
    total = totals[ctx]
    return sum(c / total * math.log2(total / c) for c in counts[ctx].values())   # p * log2(1/p)


def entropies(model, data, k=K):
    return np.array([entropy_at(model, data, i, k) for i in range(len(data))])


# --- Способы резать на участки -------------------------------------------------------------------

def starts_global(h, theta):
    """BLT, глобальный порог: новый участок начинается с байта, энтропия которого выше theta."""
    return np.concatenate([[0], np.nonzero(h[1:] > theta)[0] + 1])


def starts_monotonic(h, theta_r):
    """BLT, приблизительная монотонность: энтропия выросла больше чем на theta_r."""
    return np.concatenate([[0], np.nonzero(np.diff(h) > theta_r)[0] + 1])


def starts_fixed(n, p):
    """MEGABYTE: участки по p байтов подряд."""
    return np.arange(0, n, p)


def starts_space(data):
    """SpaceByte, упрощённо: новый участок начинается после пробела или перевода строки."""
    b = np.frombuffer(data, dtype=np.uint8)
    return np.concatenate([[0], np.nonzero((b[:-1] == 32) | (b[:-1] == 10))[0] + 1])


def word_first_letters(data):
    """Для каждого слова — два байта после пробела или перевода строки: у русского слова это ровно
    его первая буква. Так же считает листинг книги «Участки по глобальному порогу»."""
    return [range(i + 1, i + 3) for i in range(len(data) - 2)
            if data[i] in (32, 10) and data[i + 1] not in (32, 10)]


def patch_stats(data, starts, firsts):
    """Средняя длина участка; доля начал в первой букве слова; доля слов, где начат участок."""
    starts = np.asarray(starts)
    s = set(starts.tolist())
    in_first = set(j for r in firsts for j in r)
    b = np.frombuffer(data, dtype=np.uint8)
    return {
        "mean": len(data) / len(starts),
        "at_word": sum(1 for j in s if j in in_first) / len(s),
        "words": sum(1 for r in firsts if any(j in s for j in r)) / len(firsts),
        "mid_letter": float(((b[starts] >= 0x80) & (b[starts] < 0xC0)).mean()),
    }


def show_patches(text, starts):
    """Черта перед буквой, в байтах которой начинается участок; пробел — как _."""
    s, out, pos = set(int(j) for j in starts), [], 0
    piece = ""
    for ch in text:
        n = len(ch.encode("utf-8"))
        if pos > 0 and any(j in s for j in range(pos, pos + n)):
            out.append(piece)
            piece = ""
        piece += "_" if ch == " " else ch
        pos += n
    return " | ".join(out + [piece])


def theta_for_mean(h, n, target):
    """Глобальный порог, при котором средняя длина участка ближе всего к target (как подбирают в BLT)."""
    grid = np.round(np.arange(0.5, 6.0, 0.05), 2)
    return min(grid, key=lambda t: abs(n / (1 + (h[1:] > t).sum()) - target))


# --- Опыт ----------------------------------------------------------------------------------------

def main():
    tok = load_tokenizer()
    stream = np.fromfile(MODELS / "data" / "train.bin", dtype=np.uint16)

    # Цена текста: байты и токены BookLM.
    texts = {"русский": "Мама мыла раму, а потом она пошла спать.",
             "английский": "Mom washed the window, and then she went to sleep.",
             "опечатка": "Мама мыла раму, а патом она пошла спать.",
             "латинская c": "Мама мыла раму, а потом она пошла cпать.",
             "заглавные": "МАМА МЫЛА РАМУ, А ПОТОМ ОНА ПОШЛА СПАТЬ."}
    cost = {name: (len(s), len(s.encode()), len(tok.encode(s).ids)) for name, s in texts.items()}
    print("цена текста (символов, байтов, токенов BookLM):", cost)
    assert cost["русский"] == (40, 71, 12) and cost["английский"] == (50, 50, 39)
    assert cost["опечатка"][2] == 13 and cost["латинская c"][2] == 15 and cost["заглавные"][2] == 35
    seen = np.bincount(stream, minlength=8192)
    never = [i for i in range(261, 8192) if seen[i] == 0]   # 0-4 служебные, 5-260 байты
    print("склеенных токенов, которых нет в обучающем потоке:", len(never))
    assert len(never) == 65
    typo = tok.encode("днажды").ids                         # «Однажды» без первой буквы
    print("«днажды»:", typo, "ни разу не было в обучении:", typo[0] in never)
    assert typo == [412] and 412 in never

    # Байтовая n-граммная модель и энтропия.
    train = stories_bytes(tok, stream[:TRAIN_TOKENS])
    model = fit_counts(train)
    val_ids = val_tokens(10**6)
    val = stories_bytes(tok, val_ids)
    h = entropies(model, val)
    text_tokens = int((val_ids > 4).sum())
    plain_bytes = len(tok.decode(val_ids.tolist()).encode("utf-8"))
    bytes_per_token = plain_bytes / text_tokens
    print(f"обучение {len(train)} байтов, проверка {len(val)} байтов; "
          f"средняя энтропия {h.mean():.2f} бита, BookLM: {bytes_per_token:.2f} байта на токен")
    assert len(train) == 1_510_706 and abs(h.mean() - 1.03) < 0.01
    assert abs(bytes_per_token - 7.72) < 0.01

    # Цена квадратичного внимания: окно BookLM в 512 токенов, записанное байтами.
    window = 512 * bytes_per_token
    print(f"окно 512 токенов = {window:.0f} байтов, пар запрос-ключ в {(window / 512) ** 2:.0f} раз больше")

    # Глобальный порог: средняя длина участка и слова.
    firsts = word_first_letters(val)
    table = {}
    for theta in [1.5, 2.0, 2.5]:
        table[theta] = patch_stats(val, starts_global(h, theta), firsts)
        st = table[theta]
        print(f"порог {theta}: участок {st['mean']:.2f} байта, начало в первой букве слова "
              f"{st['at_word']:.0%}, слов с началом участка {st['words']:.0%}, "
              f"начало со второго байта буквы {st['mid_letter']:.0%}")
    assert abs(table[2.0]["mean"] - 7.94) < 0.01 and table[2.0]["mid_letter"] > 0.9
    assert table[2.0]["at_word"] > 0.6 and table[2.0]["words"] > 0.85
    assert f"{table[2.0]['mid_letter']:.0%}" == "91%"             # текст: 91 % со второго байта буквы
    assert table[1.5]["mean"] < bytes_per_token < table[2.5]["mean"]

    # Фиксированные участки и участки по энтропии при длине, как у токена BookLM; после пробела — для сравнения.
    theta = theta_for_mean(h, len(val), bytes_per_token)
    p = round(bytes_per_token)
    schemes = {"фиксированные (MEGABYTE)": starts_fixed(len(val), p),
               "после пробела (SpaceByte)": starts_space(val),
               f"энтропия, порог {theta} (BLT)": starts_global(h, theta)}
    res = {name: patch_stats(val, s, firsts) for name, s in schemes.items()}
    for name, st in res.items():
        print(f"{name:<28} участок {st['mean']:5.2f} байта, начало в первой букве слова {st['at_word']:.0%}")
    fixed, space, ent = res.values()
    assert fixed["at_word"] < 0.25 < ent["at_word"]          # фиксированные участки слов не видят
    assert space["at_word"] > 0.95       # после пробела почти всегда стоит первая буква слова
    assert ent["words"] > fixed["words"]
    assert f"{fixed['mean']:.1f}" == "8.0" and f"{ent['mean']:.1f}" == "7.7"   # текст: 7,7 байта
    assert f"{fixed['at_word']:.0%}" == "18%" and f"{ent['at_word']:.0%}" == "62%"   # текст: 18 % и 62 %

    # Энтропия по байтам: первый байт русской буквы угадать легко, второй (какая буква) — трудно.
    data = "Мама мыла раму".encode("utf-8")
    hb = entropies(model, data)
    print("энтропия байтов « мыла»:", " ".join(f"{x:.2f}" for x in hb[8:17]))
    assert hb[9] < 1.0 < 3.0 < hb[10]

    # Монотонное правило BLT подбираем на ту же среднюю длину.
    grid = np.round(np.arange(0.5, 6.0, 0.05), 2)
    theta_r = min(grid, key=lambda t: abs(len(val) / (1 + (np.diff(h) > t).sum()) - bytes_per_token))
    mono = patch_stats(val, starts_monotonic(h, theta_r), firsts)
    print(f"монотонность, порог роста {theta_r}: участок {mono['mean']:.2f} байта, "
          f"начало в первой букве слова {mono['at_word']:.0%}")

    # Чувствительность: длина контекста и объём данных при пороге 2 бита.
    sens = {}
    for k in [3, 6, 8]:
        m = fit_counts(train, k) if k != K else model
        hk = entropies(m, val, k)
        sens[k] = len(val) / len(starts_global(hk, 2.0))
        print(f"K={k}: средняя энтропия {hk.mean():.2f}, участок при пороге 2 бита {sens[k]:.2f} байта")
    assert sens[3] < sens[6] < sens[8]      # длинный контекст — увереннее модель — длиннее участки
    assert f"{sens[3]:.1f}" == "2.5" and f"{sens[8]:.1f}" == "10.3"   # текст: 2,5 и 10,3 байта
    for n_tok in [100_000, 400_000]:
        m = fit_counts(stories_bytes(tok, stream[:n_tok]))
        hn = entropies(m, val)
        print(f"обучение на {n_tok} токенах: средняя энтропия {hn.mean():.2f}, участок при пороге 2 бита "
              f"{len(val) / len(starts_global(hn, 2.0)):.2f} байта")

    # Как участки ложатся на фразу и как её режет BPE.
    phrase = "Мама мыла раму, а потом она пошла спать."
    data = phrase.encode()
    hp = entropies(model, data)
    shown = show_patches(phrase, starts_global(hp, 2.0))
    print(shown)
    assert shown == "Мама_ | мыла_ | ра | му | ,_ | а_ | по | том_ | она_ | по | шла_ | с | пать."

    print("labs/ch03/entropy_patches.py: все проверки пройдены")


if __name__ == "__main__":
    main()
