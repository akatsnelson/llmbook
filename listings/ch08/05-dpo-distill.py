"""Листинги главы 8 «Дообучение» книги «Большие языковые модели», файл 05-dpo-distill.

Код и вывод — ровно как в книге; текст книги сюда не входит. Листинги идут по порядку и
опираются друг на друга: запускайте файл целиком или ячейками сверху вниз (разметка # %%).
Проверка, что вывод совпал с книгой: python scripts/check_listings.py listings/ch08/05-dpo-distill.py
"""

# %% Листинг «Оптимум цели RLHF в замкнутом виде» — раздел «Оптимум цели RLHF»
import torch

def rlhf_objective(pi, pi_ref, r, beta):
    # pi, pi_ref, r: [K], по числу ответов на один запрос
    return (pi * r).sum() - beta * (pi * (pi / pi_ref).log()).sum()

def rlhf_optimum(pi_ref, r, beta):
    weights = pi_ref * torch.exp(r / beta)
    return weights / weights.sum(), weights.sum()       # pi_star, Z

pi_ref = torch.full((3,), 1 / 3, dtype=torch.float64)
r = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
for beta in (10.0, 1.0, 0.1):
    pi_star, Z = rlhf_optimum(pi_ref, r, beta)
    J = rlhf_objective(pi_star, pi_ref, r, beta).item()
    probs = " ".join(f"{p:.4f}" for p in pi_star.tolist())
    print(f"beta={beta:<4}  pi*={probs}  J={J:.4f}  "
          f"beta*log Z={beta * Z.log().item():.4f}")
# Вывод в книге:
# | beta=10.0  pi*=0.3559 0.3220 0.3220  J=0.3446  beta*log Z=0.3446
# | beta=1.0   pi*=0.5761 0.2119 0.2119  J=0.4528  beta*log Z=0.4528
# | beta=0.1   pi*=0.9999 0.0000 0.0000  J=0.8901  beta*log Z=0.8901

# %% Листинг «Логарифм вероятности ответа и ошибка DPO» — раздел «Ошибка DPO на одной паре»
import torch.nn.functional as F

def sequence_logprob(logits, targets, mask):
    # logits: [B, T, V]; targets, mask: [B, T] (mask = 1 на целях ответа)
    logp = logits.log_softmax(dim=-1)
    token_logp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return (token_logp * mask).sum(dim=-1)                  # [B]

def dpo_loss(pw, pl, rw, rl, beta):
    # pw, pl: log pi_theta(y|x) для y_w и y_l; rw, rl: то же у эталона
    margin = beta * ((pw - rw) - (pl - rl))
    return -F.logsigmoid(margin).mean(), margin

pw = torch.tensor([-12.0], requires_grad=True)
pl = torch.tensor([-14.0], requires_grad=True)
rw, rl = torch.tensor([-12.5]), torch.tensor([-13.5])       # эталон
loss, margin = dpo_loss(pw, pl, rw, rl, beta=0.1)
loss.backward()
print(f"отступ {margin.item():.4f}, ошибка {loss.item():.4f}")
print(f"градиент по log pi(y_w) {pw.grad.item():+.4f}, "
      f"по log pi(y_l) {pl.grad.item():+.4f}")
# Вывод в книге:
# | отступ 0.1000, ошибка 0.6444
# | градиент по log pi(y_w) -0.0475, по log pi(y_l) +0.0475

# %% Листинг «DPO на крошечной модели с адаптером» — раздел «Ошибка DPO на одной паре»
import sys
sys.path.insert(0, "labs/ch08")               # код разделов 8.1 и 8.2
from ft_kit import render_chat, build_labels, TinyLM, LoRALinear, IGNORE

def answer_logprob(model, request, story):
    n_prompt = len(render_chat(request))
    inputs, targets = build_labels(render_chat(request, story), n_prompt)
    mask = targets != IGNORE                  # только цели ответа и <eos>
    logits = model(inputs)[None]              # [1, T, V]
    return sequence_logprob(logits, targets.clamp(min=0)[None], mask[None])

model = TinyLM().requires_grad_(False)        # основа заморожена
model.mix = LoRALinear(model.mix, r=4, alpha=8)
request = "Сказку про ёжика"
y_w, y_l = "Ёжик нашёл свет.", "Кот пошёл гулять."

def pair_logprobs(adapter_on):                # выключенный адаптер = эталон
    model.mix.enabled = adapter_on
    return [answer_logprob(model, request, y) for y in (y_w, y_l)]

opt = torch.optim.AdamW([model.mix.A, model.mix.B], lr=0.05)
for step in range(41):
    with torch.no_grad():                     # эталону градиент не нужен
        ref_w, ref_l = pair_logprobs(False)
    pol_w, pol_l = pair_logprobs(True)
    loss, margin = dpo_loss(pol_w, pol_l, ref_w, ref_l, beta=0.1)
    if step % 10 == 0:
        print(f"шаг {step:2d}: ошибка {loss.item():.4f}, "
              f"отступ {margin.item():5.2f}, "
              f"y_w {pol_w.item():6.1f}, y_l {pol_l.item():6.1f}")
    opt.zero_grad()
    loss.backward()
    opt.step()
# Вывод в книге:
# | шаг  0: ошибка 0.6931, отступ  0.00, y_w -167.8, y_l -178.7
# | шаг 10: ошибка 0.0001, отступ  9.33, y_w -148.6, y_l -252.9
# | шаг 20: ошибка 0.0000, отступ 18.18, y_w -165.9, y_l -358.7
# | шаг 30: ошибка 0.0000, отступ 22.28, y_w -179.4, y_l -413.1
# | шаг 40: ошибка 0.0000, отступ 23.85, y_w -185.2, y_l -434.7

# %% Листинг «Неявная награда инструктивной BookLM» — раздел «Ошибка DPO на одной паре»
from booklm.lab import load
from booklm.experiments.ch08 import answer_logp, encode, held_out

tok, base = load("base")
_, inst = load("instruct")
rows = held_out()[:20]               # отложенные пары: адаптер их не видел
x, y = [r["request"] for r in rows], [r["story"] for r in rows]
j = lambda i: (i + 1) % 20           # номер соседней пары

def logp(m, request, story):         # log pi(y|x): маска истории и <eos>
    return answer_logp(m, *encode(tok, request, story, 512))[0].sum()

def compare(other):                  # y_w: своя пара, y_l: other(i)
    own = lambda i: (x[i], y[i])
    pw, pl, rw, rl = (torch.stack([logp(m, *pick(i)) for i in range(20)])
                      for m in (inst, base) for pick in (own, other))
    return dpo_loss(pw, pl, rw, rl, beta=0.1)    # отступ = r_w - r_l

cases = {"(а) своя и чужая история": lambda i: (x[i], y[j(i)]),
         "(б) своя и чужая просьба": lambda i: (x[j(i)], y[i])}
for name, other in cases.items():
    loss, margin = compare(other)
    print(f"{name}: r_w > r_l в {(margin > 0).sum()} из 20, "
          f"ошибка DPO {loss:.4f}")
r_w, r_l = (0.1 * (logp(inst, x[0], s) - logp(base, x[0], s))
            for s in (y[0], y[1]))
print(f"пара 1, (а): r_w = {r_w:+.2f}, r_l = {r_l:+.2f}")
# Вывод в книге:
# | (а) своя и чужая история: r_w > r_l в 12 из 20, ошибка DPO 0.7129
# | (б) своя и чужая просьба: r_w > r_l в 19 из 20, ошибка DPO 0.4685
# | пара 1, (а): r_w = +2.15, r_l = +2.70

# %% Листинг «Дистилляция при разных температурах» — раздел «Мягкие цели и температура»
def distill_loss(z_t, z_s, tau):
    p_t = (z_t / tau).softmax(-1)
    kl_value = (p_t * (p_t.log() - (z_s / tau).log_softmax(-1))).sum()
    return tau**2 * kl_value, kl_value

teacher = torch.tensor([2.0, 1.0, 0.0], dtype=torch.float64)
student = torch.tensor([1.0, 1.0, 0.0], dtype=torch.float64)
for tau in (1.0, 2.0, 4.0, 8.0):
    z_s = student.clone().requires_grad_()
    loss, kl_value = distill_loss(teacher, z_s, tau)
    g_kl, = torch.autograd.grad(kl_value, z_s, retain_graph=True)
    g_loss, = torch.autograd.grad(loss, z_s)
    print(f"tau={tau:.0f}:  KL {kl_value.item():.4f}   градиент: "
          f"без tau^2 {g_kl.norm():.4f}, с tau^2 {g_loss.norm():.3f}")
# Вывод в книге:
# | tau=1:  KL 0.1196   градиент: без tau^2 0.3079, с tau^2 0.308
# | tau=2:  KL 0.0310   градиент: без tau^2 0.0760, с tau^2 0.304
# | tau=4:  KL 0.0075   градиент: без tau^2 0.0182, с tau^2 0.292
# | tau=8:  KL 0.0018   градиент: без tau^2 0.0044, с tau^2 0.283

# %% Листинг «Жёсткие цели против мягкой» — раздел «Дистилляция при предобучении»
p_t, p_s = teacher.softmax(-1), student.softmax(-1)
soft = p_s - p_t                  # градиент по мягкой цели, tau = 1
show = lambda g: " ".join(f"{v:+.3f}" for v in g)
print(f"мягкая цель: {show(soft)}")
torch.manual_seed(0)
for n in (1, 10, 100, 1000, 10000):
    words = torch.multinomial(p_t, n, replacement=True)  # n текстов
    hard = p_s - F.one_hot(words, 3).double().mean(0)    # жёсткий градиент
    print(f"n = {n:<7d} {show(hard)}   ошибка {(hard - soft).norm():.3f}")
# Вывод в книге:
# | мягкая цель: -0.243 +0.178 +0.065
# | n = 1       +0.422 -0.578 +0.155   ошибка 1.010
# | n = 10      -0.378 +0.322 +0.055   ошибка 0.198
# | n = 100     -0.268 +0.192 +0.075   ошибка 0.031
# | n = 1000    -0.254 +0.204 +0.049   ошибка 0.033
# | n = 10000   -0.237 +0.174 +0.063   ошибка 0.007

# %% Листинг «Два ученика BookLM-mini» — раздел «Дистилляция при предобучении»
# Проверка этот листинг не запускает: ему нужны сеть, видеокарта, сервер или внешний файл.
# from booklm.lab import load, perplexity
# from booklm.experiments.ch08_distill import train   # вызывает student_loss
#
# def student_loss(z_s, z_t, y, tau=2.0, alpha=0.1):
#     log_p_s = (z_s / tau).log_softmax(-1)             # [B, T, V]
#     log_p_t = (z_t / tau).log_softmax(-1)
#     kl = (log_p_t.exp() * (log_p_t - log_p_s)).sum(-1).mean()
#     ce = F.cross_entropy(z_s.flatten(0, 1), y.flatten())
#     return (1 - alpha) * tau**2 * kl + alpha * ce
#
# tok, big = load("base")                               # учитель
# print(f"перплексия учителя: {perplexity(big):.2f}")
# students, curves = {}, {}
# for kind in ("text", "teacher"):                      # (а) и (б)
#     students[kind], curves[kind], _ = train(kind, big, steps=600)
# for (s, a), (_, b) in zip(curves["text"][::2], curves["teacher"][::2]):
#     print(f"шаг {s:3d}:  (а) {a:7.2f}   (б) {b:7.2f}")

# %% Листинг «Жадные сказки учителя и учеников» — раздел «Дистилляция при предобучении»
# Проверка этот листинг не запускает: ему нужны сеть, видеокарта, сервер или внешний файл.
# import textwrap
# from booklm.lab import generate
#
# start = "Жила-была маленькая девочка по имени Лили."
# for name, model in (("учитель", big), ("(а) текст", students["text"]),
#                     ("(б) учитель", students["teacher"])):
#     text = generate(tok, model, start, max_new=40, temperature=0)
#     print(textwrap.fill(f"{name}: {text}", width=72, subsequent_indent="  "))

# %% Листинг «Обрезка SwiGLU в BookLM» — раздел «Обрезка и дистилляция»
from booklm.lab import load, perplexity
from booklm.experiments.ch08_distill import calibration, ffn_importance, prune_ffn

tok, big = load("base")
scores = ffn_importance(big, calibration())      # 6 блоков по 768 чисел
g = torch.Generator().manual_seed(0)
ways = {"самые важные": lambda s: s.topk(384).indices,
        "случайные": lambda s: torch.randperm(768, generator=g)[:384],
        "самые слабые": lambda s: s.topk(384, largest=False).indices}
for name, pick in ways.items():
    small = prune_ffn(big, [pick(s).sort().values for s in scores])
    print(f"{name + ':':<14}перплексия {perplexity(small):6.2f}")
# Вывод в книге:
# | самые важные: перплексия  22.82
# | случайные:    перплексия  48.77
# | самые слабые: перплексия 512.91

# %% Листинг «Сколько вариантов теряет одно слово» — раздел «Дистилляция данных»
from booklm.lab import prompt_ids

for text in ("Жила-была маленькая девочка по имени",
             "Однажды Тим увидел в парке большую"):
    with torch.no_grad():
        p = big(torch.tensor([prompt_ids(tok, text)]))[0, -1].softmax(-1)
    top = p.topk(3)
    words = ", ".join(f"«{tok.decode([i]).strip()}» {v:.3f}"
                      for v, i in zip(top.values.tolist(), top.indices.tolist()))
    n_eff = torch.exp(-(p * p.log()).sum())     # e в степени энтропии
    print(f"{text.split()[-1]}: {words}; вариантов около {n_eff:.0f}")
# Вывод в книге:
# | имени: «Люси» 0.117, «Лили» 0.081, «Салли» 0.056; вариантов около 49
# | большую: «,» 0.190, «дерево» 0.094, «красную» 0.068; вариантов около 79
