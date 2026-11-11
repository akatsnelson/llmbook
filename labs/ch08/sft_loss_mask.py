"""Метки и маска ошибки только на ответе.

Полная версия листинга «Маска ошибки только на ответе» из главы 8, подраздел 8.1.3 «Маска ошибки»: ручной
пример на 8 позициях, сломанная маска со сдвигом на одну позицию и маска многоходового диалога.
Выполняется в ноутбуке главы 08_lora_sft.ipynb и отдельно.
Запуск из корня репозитория: python labs/ch08/sft_loss_mask.py
"""
import math

import torch
import torch.nn.functional as F

# --- шаблон диалога из sft_chat_template.py --------------------------------
ROLE_IDS = {"<|system|>": 2, "<|user|>": 3, "<|assistant|>": 4, "<|end|>": 5}
BYTE_OFFSET = 6


def encode(text):
    return [BYTE_OFFSET + b for b in text.encode("utf-8")]


def render_chat(messages, add_generation_prompt=False):
    ids = []
    for role, text in messages:
        ids.append(ROLE_IDS[f"<|{role}|>"])
        ids += encode(text)
        ids.append(ROLE_IDS["<|end|>"])
    if add_generation_prompt:
        ids.append(ROLE_IDS["<|assistant|>"])
    return ids


# --- маска ошибки -----------------------------------------------------------
IGNORE = -100
V = 8192                                     # словарь BookLM
USER, ASSIST, END = 3, 4, 5
# <|user|> Сколько будет 2+2? <|end|> <|assistant|> Чет ыре <|end|>; номера текста условные
dialog = [USER, 1201, 344, 2917, END, ASSIST, 3120, 4410, END]
is_answer = [0, 0, 0, 0, 0, 0, 1, 1, 1]     # текст ответа и его <|end|>


def build_labels(ids, answer):
    ids = torch.tensor(ids)
    inputs, targets = ids[:-1], ids[1:]                   # сдвиг: y_t = z_{t+1}
    mask = torch.tensor(answer[1:], dtype=torch.bool)     # маска над ЦЕЛЯМИ
    labels = torch.where(mask, targets, torch.full_like(targets, IGNORE))
    return inputs, labels, mask


def masked_ce(logits, labels):
    # logits: [T, V], labels: [T] с -100 там, где m_t = 0
    mask = labels != IGNORE
    logp = logits.log_softmax(dim=-1)[mask]
    nll = -logp.gather(-1, labels[mask].unsqueeze(-1)).squeeze(-1)
    return nll.sum() / mask.sum()


def logits_for(targets, probs, vocab=V):
    # Логиты, при которых правильный номер получает ровно вероятность p.
    rows = torch.zeros(len(targets), vocab, dtype=torch.float64)
    for t, (y, p) in enumerate(zip(targets, probs)):
        rows[t, y] = math.log(p * (vocab - 1) / (1 - p))
    return rows


inputs, labels, mask = build_labels(dialog, is_answer)
assert mask.tolist() == [False] * 5 + [True] * 3
probs = [0.05, 0.30, 0.02, 0.10, 0.90, 0.50, 0.25, 0.80]
logits = logits_for(dialog[1:], probs)                  # [8, 8192]

loss = masked_ce(logits, labels)
assert abs(loss.item() - 0.7675) < 1e-4
torch.testing.assert_close(loss, F.cross_entropy(logits, labels, ignore_index=IGNORE))
everywhere = F.cross_entropy(logits, torch.tensor(dialog[1:]))
assert abs(everywhere.item() - 1.6028) < 1e-4           # другое число и другая задача

first_answer_token = dialog[is_answer.index(1)]
assert labels[mask][0].item() == first_answer_token     # главная проверка маски


def build_labels_broken(ids, answer):
    ids = torch.tensor(ids)
    inputs, targets = ids[:-1], ids[1:]
    mask = torch.tensor(answer[:-1], dtype=torch.bool)    # ОШИБКА: маска по входу
    labels = torch.where(mask, targets, torch.full_like(targets, IGNORE))
    return inputs, labels, mask


_, bad_labels, bad_mask = build_labels_broken(dialog, is_answer)
assert bad_labels[bad_mask][0].item() != first_answer_token   # «Чет» потерян


def answer_flags(messages, trained_roles=("assistant",)):
    # Флаги по ВХОДНЫМ токенам строки render_chat: 1 на тексте и <|end|>
    # каждой реплики, которую пишет модель; метку роли ставит шаблон -> 0.
    flags = []
    for role, text in messages:
        a = int(role in trained_roles)
        flags += [0] + [a] * (len(encode(text)) + 1)
    return flags


two_turns = [("user", "2+2?"), ("assistant", "4"),
             ("user", "А 3+3?"), ("assistant", "Шесть")]
ids2 = render_chat(two_turns)
inputs2, labels2, mask2 = build_labels(ids2, answer_flags(two_turns))
assert int(mask2.sum()) == len(encode("4")) + len(encode("Шесть")) + 2   # 13
assert mask2[inputs2 == ASSIST].all()          # обе позиции <|assistant|> учатся
assert USER not in labels2[mask2].tolist()     # реплики пользователя не цели
print(f"целей в потере: {int(mask.sum())} из {mask.numel()}; "
      f"в двухходовом диалоге: {int(mask2.sum())} из {mask2.numel()}")
