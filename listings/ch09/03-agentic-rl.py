"""Листинги главы 9 «Обучение с подкреплением и оценка моделей» книги «Большие языковые модели», файл 03-agentic-rl.

Код и вывод — ровно как в книге; текст книги сюда не входит. Листинги идут по порядку и
опираются друг на друга: запускайте файл целиком или ячейками сверху вниз (разметка # %%).
Проверка, что вывод совпал с книгой: python scripts/check_listings.py listings/ch09/03-agentic-rl.py
"""

# %% Листинг «Расхождение вероятностей движка генерации и обучения» — раздел «Асинхронная генерация»
import copy
import torch
from booklm.lab import load, prompt_ids, quantize_copy
from booklm.experiments.ch09 import REQUESTS, answer_batch, sample_group

tok, train = load("instruct")                        # обучающий код, FP32
engines = {"BF16  ": copy.deepcopy(train).to(torch.bfloat16),
           "4 бита": quantize_copy(train, bits=4)}

@torch.no_grad()
def token_logp(model, x):
    logits = model(x[:, :-1]).float()
    return logits.log_softmax(-1).gather(-1, x[:, 1:, None])[..., 0]

for name, engine in engines.items():
    gen = torch.Generator().manual_seed(0)
    per_token, per_answer = [], []
    for request, _ in REQUESTS[:4]:
        ids = prompt_ids(tok, request, chat=True)
        x, mask = answer_batch(ids, sample_group(engine, ids, 8, 300, gen))
        diff = (token_logp(train, x) - token_logp(engine, x)) * mask
        per_token.append(diff.exp()[mask.bool()])
        per_answer += diff.sum(1).exp().tolist()
    w, s = torch.cat(per_token), torch.tensor(per_answer)
    print(f"{name}: токенов {len(w)}, отношение от {w.min():.2f} до "
          f"{w.max():.2f}, выше 2: {(w > 2).sum().item()}; ответы целиком: "
          f"от {s.min():.2f} до {s.max():.2f}")
# Вывод в книге:
# | BF16  : токенов 5117, отношение от 0.81 до 1.17, выше 2: 0; ответы целиком: от 0.68 до 2.56
# | 4 бита: токенов 5621, отношение от 0.03 до 8.83, выше 2: 125; ответы целиком: от 0.00 до 0.39
