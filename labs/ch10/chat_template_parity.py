"""Глава 10, раздел 10.1 «Сообщения и номера токенов», подраздел «Шаблон диалога»: листинги «Шаблон BookLM на
учебном токенизаторе» и «Паритет шаблона и сломанный сервер» (полная версия).

Шаблон тот же, что в главе 8 (подраздел «Шаблон диалога и служебные токены») и у настоящей BookLM:
<bos><|user|>просьба<|assistant|>история<eos>, служебные номера 0-4, байт UTF-8 b -> номер 5 + b.
Системной роли у BookLM нет: инструкция стоит в начале просьбы.

Проверяет: 70 номеров и позиции служебных токенов, паритет шаблона (сломанный сервер с лишним переводом
строки и «ошибка наоборот»), бюджет окна. Выполняется и в ноутбуке главы
notebooks/10_constrained_decoding.ipynb.

Запуск из корня репозитория: python labs/ch10/chat_template_parity.py
"""

SPECIAL = {"<pad>": 0, "<bos>": 1, "<eos>": 2,
           "<|user|>": 3, "<|assistant|>": 4}
BYTE_OFFSET, T_MAX = 5, 512    # номера 0-4 заняты служебными токенами


def encode(text):
    # Учебный токенизатор: байт UTF-8 b -> номер 5 + b. Служебных строк он не знает.
    return [BYTE_OFFSET + b for b in text.encode("utf-8")]


def render_chat(request, story=None):
    # Шаблон из главы 8: без story — промпт с приглашением к ответу, со story — обучающая пара.
    ids = [SPECIAL["<bos>"], SPECIAL["<|user|>"]] + encode(request)
    ids.append(SPECIAL["<|assistant|>"])
    if story is not None:
        ids += encode(story) + [SPECIAL["<eos>"]]
    return ids


def special_positions(ids):
    return [i for i, t in enumerate(ids) if t < BYTE_OFFSET]


request = "Отвечай кратко. Сколько блоков в BookLM?"
ids = render_chat(request)
assert len(ids) == 70 and ids[-1] == SPECIAL["<|assistant|>"]
assert special_positions(ids) == [0, 1, 69]              # только на границах


def parity_ok(render_infer, render_train, request, answer):
    # Паритет: промпт на выводе — начало обучающей строки,
    # и сразу за ним в обучении стоит первый токен ответа.
    prompt = render_infer(request)
    train = render_train(request, answer)
    return train[:len(prompt)] == prompt and train[len(prompt)] == encode(answer)[0]


assert parity_ok(render_chat, render_chat, request, "Шесть.")


def render_server_v1_1(request, story=None):
    # Ошибка сервера: после <|assistant|> стоит перевод строки.
    ids = render_chat(request, story)
    return ids + encode("\n") if story is None else ids


ids_bad = render_server_v1_1(request)
assert special_positions(ids_bad) == special_positions(ids)   # проверка позиций не ловит
assert len(ids_bad) == 71 and ids_bad[-1] == 15               # длина и конец выдают
assert not parity_ok(render_server_v1_1, render_chat, request, "Шесть.")


def render_train_nl(request, story=None):
    # Ошибка наоборот: в обучении после <|assistant|> стоял перевод строки.
    ids = render_chat(request)
    if story is not None:
        ids += encode("\n" + story) + [SPECIAL["<eos>"]]
    return ids


prompt = render_chat(request)
train_nl = render_train_nl(request, "Шесть.")
assert train_nl[:len(prompt)] == prompt                  # первое условие проходит
assert not parity_ok(render_chat, render_train_nl, request, "Шесть.")

# Бюджет окна: промпт плюс предел ответа помещаются в 512.
max_new_tokens = 64
assert len(ids) + max_new_tokens <= T_MAX

print("labs/ch10/chat_template_parity.py: шаблон, паритет и бюджет окна проверены")
