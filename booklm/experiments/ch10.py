"""Глава 10 «Работа с промптом»: опыты на обученной BookLM для рубрик «А теперь на нашей модели».

Опыты (время — на процессоре ноутбука Apple M2 Max):
  ch10_chat_template        рубрика «шаблон диалога», подраздел «Служебные токены и текст пользователя»;
                            несколько секунд. Три варианта шаблона (правильный, лишний \n после <|assistant|>,
                            без <|assistant|>): последние токены, пять вероятных следующих токенов, жадный ответ;
                            длина промпта в токенах BookLM и в байтах.
  ch10_induction_head       рубрика «индукционная голова», подраздел «Индукционные головы»; несколько секунд.
                            Индукционная оценка 48 голов на дважды повторённой случайной строке, ошибка на первой
                            и второй копии, голова предыдущего токена, лучшая оценка на шагах обучения 100–3051.
                            Ещё пишет ch10-induction-head.csv: карту внимания сильнейшей головы для рисунка.
  ch10_few_shot_order       рубрика «примеры в промпте», подраздел «Порядок примеров»; несколько секунд.
                            Вероятность правильного ответа при 0–3 примерах и при шести порядках трёх примеров.
  ch10_vote_temperature     рубрика «температура и голосование», подраздел «Самосогласованность»;
                            несколько секунд. Девять имён героини при tau 0.1, 0.8, 1.5 и итог голосования.
  ch10_grammar_mask_booklm  рубрика «отрезанная масса», подраздел «Ошибки в грамматике»; около 35 с.
                            Автомат для трёх цветов, индекс разрешённых токенов, жадный ответ с маской по шагам,
                            по 200 ответов при tau = 1 с маской и без, средняя отрезанная масса mu_1.
  ch10_prefix_cache_booklm  рубрика «общий префикс», раздел «Цена длинного промпта»; около 50 с.
                            100 запросов с общим началом в 448 токенов: расхождение логитов с кэшем и без,
                            совпадение следующего токена, время на CPU, размер KV-кэша начала.
  ch10_paired_booklm        рубрика «парное сравнение», подраздел «Парное сравнение двух версий»;
                            около полутора минут. Две версии промпта на 100 героях: доли с интервалами, таблица пар,
                            точный тест Макнемара, release_gate и по примеру ответа.

Запуск из корня репозитория:
    python -m booklm.experiments.ch10                       # все опыты по очереди
    python -m booklm.experiments.ch10 ch10_induction_head   # один опыт
Вывод каждого опыта копируется в models/booklm/experiments/<имя опыта через дефис>.txt,
например ch10-induction-head.txt. Числа времени зависят от машины и её загрузки.
"""
import csv
import hashlib
import itertools
import json
import re
import statistics
import sys
import time
from collections import Counter
from math import comb, sqrt
from pathlib import Path

import torch
import torch.nn.functional as F

if not __package__:  # запуск файлом (python booklm/experiments/ch10.py): корень репозитория — в путь поиска
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from booklm.lab import (MODELS, generate, kv_cache_bytes, load, load_model, next_token_table,
                        prompt_ids, special, val_tokens)

OUT = MODELS / "experiments"


def write_csv(name, header, rows):
    with open(OUT / name, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def short(text, n=90):
    # одна строка для книги: без переносов, не длиннее n символов
    text = " ".join(text.split())
    return text if len(text) <= n else text[:n] + "…"


def piece(tok, i):
    # кусочек текста токена; служебные показываем, перевод строки — как \n
    return tok.decode([i], skip_special_tokens=False).replace("\n", "\\n")


@torch.no_grad()
def top_next(tok, model, ids, k=5):
    """k самых вероятных следующих токенов после строки номеров ids."""
    p = model(torch.tensor([ids]))[0, -1].softmax(-1)
    top = p.topk(k)
    return [(piece(tok, i), v) for v, i in zip(top.values.tolist(), top.indices.tolist())]


def fmt_top(top):
    return ", ".join(f"«{s}» {p:.2f}" for s, p in top)


@torch.no_grad()
def continue_ids(model, ids, n, temperature=0.0, top_k=0, seed=0, eos_id=None):
    out = model.generate(torch.tensor([ids]), max_new=n, temperature=temperature,
                         top_k=top_k, eos_id=eos_id, seed=seed)
    return out[0, len(ids):].tolist()


# ----------------------------------------------------------------------------- шаблон диалога

def ch10_chat_template():
    """Рубрика «шаблон диалога»: три варианта шаблона, следующие токены и жадный ответ."""
    tok, model = load("instruct")
    text = "Расскажи сказку про ёжика, который боялся темноты."
    good = prompt_ids(tok, text, chat=True)          # <bos><|user|>просьба<|assistant|>
    variants = {"правильный шаблон": good,
                "лишний \\n после <|assistant|>": good + tok.encode("\n").ids,
                "без <|assistant|>": good[:-1]}
    for name, ids in variants.items():
        print(f"{name}:")
        print("  конец:", " ".join(f"{i}«{piece(tok, i)}»" for i in ids[-5:]))
        print("  дальше:", fmt_top(top_next(tok, model, ids)))
        # жадно, 60 токенов; <eos> не останавливает, чтобы было видно, что после него
        new = continue_ids(model, ids, 60)
        cont = tok.decode(new, skip_special_tokens=False).replace("\n", "\\n")
        print("  ответ:", short(cont, 150))
    n_bytes = 3 + len(text.encode("utf-8"))           # <|user|> байты <|end|> <|assistant|>
    print(f"длина правильного промпта: BookLM {len(good)} токенов, байтовый токенизатор {n_bytes}")


# ----------------------------------------------------------------------------- индукционная голова

def induction_scores(model, ids):
    """Индукционная оценка голов [6, 8] и карты внимания [6, 8, 61, 61]."""
    with torch.no_grad():
        logits, weights = model(ids, return_weights=True)
    att = torch.stack([w[0] for w in weights])
    # позиции второй копии 31..59; токен, шедший за таким же в первой копии, стоит на t-29
    t = torch.arange(31, 60)
    return att[:, :, t, t - 29].mean(-1), att, logits


def ch10_induction_head():
    """Рубрика «индукционная голова»: индукционная оценка голов на повторённой строке."""
    tok, model = load("base")
    torch.manual_seed(0)
    rand = torch.randint(5, 8192, (30,))
    ids = torch.cat([torch.tensor([special(tok)["<bos>"]]), rand, rand])[None]   # [1, 61]
    score, att, logits = induction_scores(model, ids)
    print("индукционная оценка голов (строки — блоки 1-6, столбцы — головы 1-8):")
    for b in range(6):
        print(f"  блок {b + 1}: " + " ".join(f"{s:.2f}" for s in score[b].tolist()))
    loss = F.cross_entropy(logits[0, :-1], ids[0, 1:], reduction="none")       # ошибка на каждом токене
    first, second = loss[1:30].mean().item(), loss[31:60].mean().item()
    print(f"ошибка на первой копии {first:.2f}, на второй {second:.2f}")
    b, h = divmod(int(score.argmax()), 8)
    print(f"сильнейшая голова: блок {b + 1}, голова {h + 1}, оценка {score[b, h]:.2f}")
    strong = (score > 0.3).nonzero().tolist()
    print("головы с оценкой больше 0.3:", ", ".join(f"{i + 1}.{j + 1}" for i, j in strong) or "нет")
    # для сравнения: голова «предыдущего токена», первая половина индукционной схемы
    t = torch.arange(2, 61)
    prev = att[:, :, t, t - 1].mean(-1)
    pb, ph = divmod(int(prev.argmax()), 8)
    print(f"голова предыдущего токена: блок {pb + 1}, голова {ph + 1}, доля {prev[pb, ph]:.2f}")
    best = [f"{k}: {induction_scores(load_model(f'step:{k}'), ids)[0].max():.2f}"
            for k in (100, 762, 1525)] + [f"3051: {score.max():.2f}"]
    print("наибольшая оценка на шагах обучения:", ", ".join(best))
    write_csv("ch10-induction-head.csv", ["query", "key", "weight"],
              [(i, j, round(att[b, h, i, j].item(), 4)) for i in range(61) for j in range(i + 1)])


# ----------------------------------------------------------------------------- примеры в промпте

def ch10_few_shot_order():
    """Рубрика «примеры в промпте»: число и порядок примеров против вероятности ответа."""
    tok, model = load("base")
    shots = [("Кошка", "котёнок"), ("Собака", "щенок"), ("Корова", "телёнок")]
    queries = [("Утка", "утёнок"), ("Лошадь", "жеребёнок")]

    @torch.no_grad()
    def p_answer(prompt, answer):
        # вероятность первого токена правильного ответа; в корпусе пишут и «ё», и «е»
        firsts = {tok.encode(" " + a).ids[0] for a in (answer, answer.replace("ё", "е"))}
        p = model(torch.tensor([prompt_ids(tok, prompt)]))[0, -1].softmax(-1)
        return sum(p[i].item() for i in firsts)

    def make(examples, animal):
        return "".join(f"{a} — {b}. " for a, b in examples) + f"{animal} —"

    for animal, answer in queries:
        print(f"запрос «{animal} —», правильный ответ «{answer}»:")
        for k in range(4):
            prompt = make(shots[:k], animal)
            top = next_token_table(tok, model, prompt, k=5)
            print(f"  k={k}: P = {p_answer(prompt, answer):.4f};", fmt_top(top))
    print("k=3, шесть порядков примеров (запрос «Утка —»):")
    for order in itertools.permutations(shots):
        prompt = make(order, "Утка")
        top = next_token_table(tok, model, prompt, k=1)[0][0]
        names = " ".join(b for _, b in order)
        print(f"  {names:<26} P(утёнок) = {p_answer(prompt, 'утёнок'):.4f}, лидер «{top.strip()}»")


# ----------------------------------------------------------------------------- температура и голосование

def ch10_vote_temperature():
    """Рубрика «температура и голосование»: девять имён при трёх температурах."""
    tok, model = load("base")
    prompt = "Жила-была девочка по имени"
    print("пять самых вероятных имён:", fmt_top(next_token_table(tok, model, prompt, k=5)))
    for tau in (0.1, 0.8, 1.5):
        names = []
        for seed in range(9):
            text = generate(tok, model, prompt, max_new=4, temperature=tau, top_k=0, seed=seed)
            words = re.findall(r"\w+", text[len(prompt):])
            names.append(words[0] if words else "?")
        votes = Counter(names)
        winner, n = votes.most_common(1)[0]
        print(f"tau={tau}: {' '.join(names)}")
        if n > 1:
            print(f"  различных {len(votes)}, победитель {winner}: {n} из 9")
        else:
            print(f"  различных {len(votes)}, у каждого имени один голос: победителя нет")


# ----------------------------------------------------------------------------- маска грамматики

def token_bytes(tok):
    """Байты каждого токена словаря (байтовый BPE хранит их как печатные символы)."""
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):                   # та же таблица байт -> символ, что у GPT-2
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    to_byte = {chr(c): b for b, c in zip(bs, cs)}
    vocab = []
    for i in range(tok.get_vocab_size()):
        s = tok.id_to_token(i)
        vocab.append(b"" if i < 5 else bytes(to_byte[ch] for ch in s))   # 0-4 служебные
    return vocab


def build_trie(words):
    """Автомат по байтам: delta[q][байт] -> q'; принимающие состояния — концы слов."""
    delta, accept = [{}], set()
    for w in words:
        q = 0
        for b in w.encode("utf-8"):
            if b not in delta[q]:
                delta.append({})
                delta[q][b] = len(delta) - 1
            q = delta[q][b]
        accept.add(q)
    return delta, accept


def walk(delta, q, bs):
    for b in bs:
        q = delta[q].get(b)
        if q is None:
            return None
    return q


def ch10_grammar_mask_booklm():
    """Рубрика «отрезанная масса»: маска грамматики для трёх цветов и масса mu."""
    tok, model = load("base")
    EOS = special(tok)["<eos>"]
    L = ["красный»", "синий»", "зелёный»"]
    delta, accept = build_trie(L)
    t0 = time.perf_counter()
    vocab = token_bytes(tok)
    index = {}                                   # состояние -> разрешённые номера
    for q in range(len(delta)):
        index[q] = [i for i, bs in enumerate(vocab) if bs and walk(delta, q, bs) is not None]
        if q in accept:
            index[q].append(EOS)
    # состояния, до которых можно дойти, читая целые токены
    seen, stack = {0}, [0]
    while stack:
        q = stack.pop()
        for i in index[q]:
            nxt = None if i == EOS else walk(delta, q, vocab[i])
            if nxt is not None and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    build_ms = (time.perf_counter() - t0) * 1000
    sizes = [len(index[q]) for q in sorted(seen)]
    print(f"состояний автомата {len(delta)}, достижимых по токенам {len(seen)};"
          f" индекс построен за {build_ms:.0f} мс")
    print(f"разрешено токенов из 8192: в начале {len(index[0])}, в остальных состояниях"
          f" от {min(sizes)} до {max(sizes)}, медиана {statistics.median(sizes):.0f}")
    def show(i):
        # неполный символ UTF-8 показываем байтами: \xd0 — первая половина русской буквы
        return "<eos>" if i == EOS else vocab[i].decode(errors="backslashreplace")

    print("  в начале:", " ".join(f"«{show(i)}»" for i in index[0]))

    prompt = "У Лили был мяч. Мама спросила: «Какого цвета твой мяч?» Лили ответила: «"
    start = prompt_ids(tok, prompt)

    @torch.no_grad()
    def step_probs(ids):
        return model(torch.tensor([ids]))[0, -1].float().softmax(-1)

    def answer(masked, temperature, gen, log=False):
        ids, q, mus = list(start), 0, []
        for _ in range(8):
            p = step_probs(ids)
            if masked:
                allowed = index[q]
                mus.append(1 - p[allowed].sum().item())   # отрезанная масса mu_t
                p_use = torch.zeros_like(p)
                p_use[allowed] = p[allowed]
            else:
                p_use = p
            if temperature == 0:
                nxt = int(p_use.argmax())
            else:
                p_use = (p_use.log() / temperature).softmax(-1)
                nxt = int(torch.multinomial(p_use, 1, generator=gen))
            if log:
                lead = int(p.argmax())
                print(f"  шаг {len(mus)}: выбран «{show(nxt)}», лидер без маски"
                      f" «{show(lead)}» ({p[lead]:.2f}), mu = {mus[-1]:.3f}")
            if nxt == EOS:
                break
            ids.append(nxt)
            if masked:
                q = walk(delta, q, vocab[nxt])
        text = tok.decode(ids[len(start):])
        return text, mus     # если <eos> не успел за 8 токенов, ответ оборван

    print("жадный ответ с маской:")
    text, _ = answer(True, 0, None, log=True)
    print(f"  ответ: «{text}»")

    gen = torch.Generator().manual_seed(0)
    for masked in (True, False):
        texts, mu1 = [], []
        for _ in range(200):
            t, mus = answer(masked, 1.0, gen)
            texts.append(t)
            mu1 += mus[:1]
        if masked:
            ok = sum(t in L for t in texts)
            cut = [t for t in texts if t not in L]
            print(f"tau=1, с маской: из L {ok} из 200; mu_1 = {statistics.mean(mu1):.3f}")
            if cut:
                print(f"  не дошли до конца за 8 токенов: {len(cut)},"
                      f" например «{cut[0]}»")
        else:
            # без маски ответ не кончается: берём текст до первой » включительно
            heads = [t.split("»")[0] + "»" if "»" in t else t for t in texts]
            ok = sum(h in L for h in heads)
            ok_case = sum(h.lower() in L for h in heads)
            print(f"tau=1, без маски: из L {ok} из 200, без учёта регистра {ok_case}")
            print(f"  например: «{heads[0]}», «{heads[1]}»")


# ----------------------------------------------------------------------------- общий префикс

def stories():
    """Отложенные истории: списки номеров от <bos> до <eos> включительно."""
    data = val_tokens(10**9).tolist()
    out, cur = [], []
    for i in data:
        cur.append(i)
        if i == 2:                 # <eos>
            out.append(cur)
            cur = []
    return out


@torch.no_grad()
def forward_cached(model, ids, caches):
    """Проход по новым токенам поверх кэша явной формулой внимания (return_weights=True).
    Быстрый путь model.py для куска поверх кэша ставит ту же причинную маску; в книге
    (рубрика «общий префикс») опыт идёт через него, а здесь сравниваются обе формулы."""
    logits, _ = model(ids, caches, return_weights=True)
    for c in caches:
        c["len"] += ids.size(1)
    return logits


@torch.no_grad()
def ch10_prefix_cache_booklm():
    """Рубрика «общий префикс»: 100 запросов поверх кэша общего начала."""
    tok, model = load("base")
    st = stories()
    prefix = torch.tensor(next(s for s in st if len(s) >= 448)[:448])[None]
    tails = [torch.tensor(s[1:41])[None] for s in st if len(s) >= 41 and s[:448] != prefix[0].tolist()][:100]

    caches = [{"len": 0} for _ in model.blocks]
    forward_cached(model, prefix, caches)                     # префикс один раз
    worst, worst_sdpa, same = 0.0, 0.0, 0
    for tail in tails:
        ids = torch.cat([prefix, tail], 1)
        full = forward_cached(model, ids, [{"len": 0} for _ in model.blocks])[0, 448:]
        c = [dict(x) for x in caches]                         # копия кэша на запрос
        part = forward_cached(model, tail, c)[0]
        worst = max(worst, (part - full).abs().max().item())
        sdpa = model(ids)[0, 448:]                            # полный проход быстрым ядром
        worst_sdpa = max(worst_sdpa, (part - sdpa).abs().max().item())
        same += int(part[-1].argmax() == full[-1].argmax())
    print(f"запросов {len(tails)}: префикс 448 токенов, хвост 40")
    print(f"наибольшее |Δ| логитов хвоста, кэш против полного прохода: {worst:.1e}")
    print(f"  то же против полного прохода быстрым ядром SDPA: {worst_sdpa:.1e}")
    print(f"самый вероятный следующий токен совпал в {same} запросах из {len(tails)}")

    def run_without():
        for tail in tails:
            forward_cached(model, torch.cat([prefix, tail], 1), [{"len": 0} for _ in model.blocks])

    def run_with():
        c0 = [{"len": 0} for _ in model.blocks]
        forward_cached(model, prefix, c0)
        for tail in tails:
            forward_cached(model, tail, [dict(x) for x in c0])

    def median_time(f):
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            f()
            times.append(time.perf_counter() - t0)
        return statistics.median(times)

    t_without, t_with = median_time(run_without), median_time(run_with)
    print(f"время 100 запросов на CPU: без кэша {t_without:.2f} с, с кэшем {t_with:.2f} с,"
          f" {t_with / t_without:.1%}")
    print(f"KV-кэш префикса в bf16: {kv_cache_bytes(448):,} байт".replace(",", " "))


# ----------------------------------------------------------------------------- парное сравнение

# 100 героев: именительный, родительный, дательный, винительный, творительный, предложный
HEROES = [
    "ёжик ёжика ёжику ёжика ёжиком ёжике", "лиса лисы лисе лису лисой лисе",
    "дракон дракона дракону дракона драконом драконе", "кот кота коту кота котом коте",
    "собака собаки собаке собаку собакой собаке", "заяц зайца зайцу зайца зайцем зайце",
    "волк волка волку волка волком волке", "медведь медведя медведю медведя медведем медведе",
    "белка белки белке белку белкой белке", "мышка мышки мышке мышку мышкой мышке",
    "лягушка лягушки лягушке лягушку лягушкой лягушке", "сова совы сове сову совой сове",
    "утёнок утёнка утёнку утёнка утёнком утёнке", "котёнок котёнка котёнку котёнка котёнком котёнке",
    "щенок щенка щенку щенка щенком щенке", "слон слона слону слона слоном слоне",
    "жираф жирафа жирафу жирафа жирафом жирафе", "тигр тигра тигру тигра тигром тигре",
    "лев льва льву льва львом льве", "обезьяна обезьяны обезьяне обезьяну обезьяной обезьяне",
    "черепаха черепахи черепахе черепаху черепахой черепахе", "рыбка рыбки рыбке рыбку рыбкой рыбке",
    "птичка птички птичке птичку птичкой птичке", "бабочка бабочки бабочке бабочку бабочкой бабочке",
    "пчела пчелы пчеле пчелу пчелой пчеле", "муравей муравья муравью муравья муравьём муравье",
    "улитка улитки улитке улитку улиткой улитке", "корова коровы корове корову коровой корове",
    "лошадь лошади лошади лошадь лошадью лошади", "свинья свиньи свинье свинью свиньёй свинье",
    "овца овцы овце овцу овцой овце", "коза козы козе козу козой козе",
    "курица курицы курице курицу курицей курице", "петух петуха петуху петуха петухом петухе",
    "гусь гуся гусю гуся гусем гусе", "утка утки утке утку уткой утке",
    "олень оленя оленю оленя оленем олене", "лось лося лосю лося лосем лосе",
    "барсук барсука барсуку барсука барсуком барсуке", "енот енота еноту енота енотом еноте",
    "крот крота кроту крота кротом кроте", "хомяк хомяка хомяку хомяка хомяком хомяке",
    "попугай попугая попугаю попугая попугаем попугае", "пингвин пингвина пингвину пингвина пингвином пингвине",
    "дельфин дельфина дельфину дельфина дельфином дельфине", "кит кита киту кита китом ките",
    "акула акулы акуле акулу акулой акуле", "осьминог осьминога осьминогу осьминога осьминогом осьминоге",
    "краб краба крабу краба крабом крабе", "динозавр динозавра динозавру динозавра динозавром динозавре",
    "единорог единорога единорогу единорога единорогом единороге", "робот робота роботу робота роботом роботе",
    "принцесса принцессы принцессе принцессу принцессой принцессе", "принц принца принцу принца принцем принце",
    "король короля королю короля королём короле", "королева королевы королеве королеву королевой королеве",
    "рыцарь рыцаря рыцарю рыцаря рыцарем рыцаре", "фея феи фее фею феей фее",
    "ведьма ведьмы ведьме ведьму ведьмой ведьме",
    "волшебник волшебника волшебнику волшебника волшебником волшебнике",
    "великан великана великану великана великаном великане", "гном гнома гному гнома гномом гноме",
    "пират пирата пирату пирата пиратом пирате", "клоун клоуна клоуну клоуна клоуном клоуне",
    "повар повара повару повара поваром поваре", "врач врача врачу врача врачом враче",
    "учитель учителя учителю учителя учителем учителе",
    "пожарный пожарного пожарному пожарного пожарным пожарном",
    "моряк моряка моряку моряка моряком моряке",
    "космонавт космонавта космонавту космонавта космонавтом космонавте",
    "фермер фермера фермеру фермера фермером фермере", "мальчик мальчика мальчику мальчика мальчиком мальчике",
    "девочка девочки девочке девочку девочкой девочке", "бабушка бабушки бабушке бабушку бабушкой бабушке",
    "дедушка дедушки дедушке дедушку дедушкой дедушке", "малыш малыша малышу малыша малышом малыше",
    "снеговик снеговика снеговику снеговика снеговиком снеговике", "кукла куклы кукле куклу куклой кукле",
    "мишка мишки мишке мишку мишкой мишке", "мяч мяча мячу мяч мячом мяче",
    "паровозик паровозика паровозику паровозик паровозиком паровозике",
    "самолёт самолёта самолёту самолёт самолётом самолёте",
    "кораблик кораблика кораблику кораблик корабликом кораблике",
    "машинка машинки машинке машинку машинкой машинке", "шарик шарика шарику шарик шариком шарике",
    "цветок цветка цветку цветок цветком цветке", "дерево дерева дереву дерево деревом дереве",
    "звезда звезды звезде звезду звездой звезде", "облако облака облаку облако облаком облаке",
    "солнце солнца солнцу солнце солнцем солнце", "луна луны луне луну луной луне",
    "ворона вороны вороне ворону вороной вороне", "воробей воробья воробью воробья воробьём воробье",
    "голубь голубя голубю голубя голубем голубе", "паук паука пауку паука пауком пауке",
    "жук жука жуку жука жуком жуке", "стрекоза стрекозы стрекозе стрекозу стрекозой стрекозе",
    "пони пони пони пони пони пони", "тюлень тюленя тюленю тюленя тюленем тюлене",
    "бобр бобра бобру бобра бобром бобре"]


def mcnemar_exact(only_a, only_b):
    n = only_a + only_b
    if n == 0:
        return 1.0
    tail = sum(comb(n, j) for j in range(max(only_a, only_b), n + 1))
    return min(1.0, 2 * tail / 2 ** n)


def release_gate(form_rate, only_old, only_new, alpha=0.05):
    """Выпуск: форма 100% и новая версия не стала значимо хуже старой."""
    if form_rate < 1.0:
        return False
    worse = only_old > only_new
    return not (worse and mcnemar_exact(only_old, only_new) < alpha)


def ci95(correct, n):
    f = correct / n
    half = 1.96 * sqrt(f * (1 - f) / n)
    return f - half, f + half


def mentions(text, forms):
    # хотя бы одна падежная форма отдельным словом, без учёта регистра и ё/е
    words = set(re.findall(r"\w+", text.lower().replace("ё", "е")))
    return any(f.replace("ё", "е") in words for f in forms)


def ch10_paired_booklm():
    """Рубрика «парное сравнение»: две версии промпта на 100 героях, тест Макнемара."""
    tok, model = load("instruct")
    heroes = [h.split() for h in HEROES]
    assert len(heroes) == 100 and all(len(h) == 6 for h in heroes)
    digest = hashlib.sha256(json.dumps(heroes, ensure_ascii=False).encode()).hexdigest()
    A = "Напиши сказку про {acc}."                       # «про» требует винительного
    B = "Расскажи короткую историю, в которой главный герой — {nom}."
    ok_a, ok_b, stories_a, stories_b = [], [], [], []
    for forms in heroes:
        for tmpl, ok, keep in ((A, ok_a, stories_a), (B, ok_b, stories_b)):
            req = tmpl.format(nom=forms[0], acc=forms[3])
            story = generate(tok, model, req, chat=True, temperature=0, max_new=120)
            ok.append(mentions(story, forms))
            keep.append(story)
    n = len(heroes)
    print(f"набор: {n} героев, SHA-256 {digest[:16]}…")
    for name, ok in (("A", ok_a), ("B", ok_b)):
        lo, hi = ci95(sum(ok), n)
        print(f"версия {name}: герой назван в {sum(ok)} из {n}, интервал {lo:.0%}–{hi:.0%}")
    pairs = Counter(zip(ok_a, ok_b))
    both, only_a = pairs[(True, True)], pairs[(True, False)]
    only_b, neither = pairs[(False, True)], pairs[(False, False)]
    print(f"обе верны {both}, только A {only_a}, только B {only_b}, обе неверны {neither}")
    print(f"p_МН = {mcnemar_exact(only_a, only_b):.2g};"
          f" release_gate(A -> B) = {release_gate(1.0, only_a, only_b)}")
    for want, label in (((True, False), "права только A"), ((False, True), "права только B")):
        k = next((i for i in range(n) if (ok_a[i], ok_b[i]) == want), None)
        if k is not None:
            print(f"{label}, герой «{heroes[k][0]}»:")
            print("  A:", short(stories_a[k], 120))
            print("  B:", short(stories_b[k], 120))


EXPERIMENTS = {"ch10_chat_template": ch10_chat_template,
               "ch10_induction_head": ch10_induction_head,
               "ch10_few_shot_order": ch10_few_shot_order,
               "ch10_vote_temperature": ch10_vote_temperature,
               "ch10_grammar_mask_booklm": ch10_grammar_mask_booklm,
               "ch10_prefix_cache_booklm": ch10_prefix_cache_booklm,
               "ch10_paired_booklm": ch10_paired_booklm}

if __name__ == "__main__":
    from booklm.experiments import run
    OUT.mkdir(exist_ok=True)
    run(EXPERIMENTS, save=True)      # копия вывода в models/booklm/experiments/<имя опыта>.txt
