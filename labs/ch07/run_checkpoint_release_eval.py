"""Глава 7, раздел 7.6 «Длинный прогон и бюджет обучения».

Полные версии листингов раздела: прогон в миниатюре (TinyLM с прореживанием, загрузчик с позицией,
AdamW с группами, LambdaLR, GradScaler, ограничение нормы), атомарная запись контрольной точки,
побитное возобновление и четыре поломки («только веса», «без загрузчика», «без планировщика»,
«без генераторов»), скелет BookLM и выпускаемые веса в BF16 с манифестом и проверкой SHA-256 родителя,
валидационная ошибка, перплексия, бит на байт, контрольные продолжения и хеш логитов, бюджет по
Chinchilla и измерение скорости шага скелета BookLM на вашем процессоре (печатается, не проверяется).
Запуск из корня репозитория: python labs/ch07/run_checkpoint_release_eval.py
"""

# --- Прогон в миниатюре и атомарная запись ---------------------------------
import io, math, os, random, tempfile
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyLM(nn.Module):              # BookLM без блоков, раздел 1
    def __init__(self, vocab_size, d):
        super().__init__()
        self.tokens = nn.Embedding(vocab_size, d)
        self.norm = nn.RMSNorm(d, eps=1e-6)
        self.output = nn.Linear(d, vocab_size, bias=False)
        self.output.weight = self.tokens.weight  # связанные веса

    def forward(self, ids, drop=0.0):            # [B,T] -> [B,T,V]
        h = F.dropout(self.tokens(ids), p=drop, training=drop > 0)
        return self.output(self.norm(h))

class Loader:                        # позиция: (seed, epoch, index)
    def __init__(self, windows, batch_size, seed):
        self.windows, self.b = windows, batch_size
        self.seed, self.epoch, self.index = seed, 0, 0

    def next_batch(self):
        if self.index + self.b > len(self.windows):
            self.epoch, self.index = self.epoch + 1, 0
        g = torch.Generator().manual_seed(self.seed + self.epoch)
        order = torch.randperm(len(self.windows), generator=g)
        rows = self.windows[order[self.index:self.index + self.b]]
        self.index += self.b
        return rows[:, :-1], rows[:, 1:]             # X, Y: [B, T]

    def state_dict(self):
        return dict(seed=self.seed, epoch=self.epoch, index=self.index)

    def load_state_dict(self, state):
        self.__dict__.update(state)

V, T, WARMUP, STEPS = 16, 8, 4, 10    # словарь, окно, s_w, s_max
grid = torch.arange(24 * (T + 1)).view(24, T + 1)
WINDOWS = (grid * 5 + grid // 7) % V   # 24 окна длины T+1

def lr_lambda(i):              # LambdaLR считает с нуля: s = i + 1
    s = i + 1
    if s <= WARMUP:
        return s / WARMUP
    progress = (s - WARMUP) / (STEPS - WARMUP)
    return 0.1 + 0.45 * (1 + math.cos(math.pi * progress))

def make_run(seed):
    torch.manual_seed(seed)
    model = TinyLM(V, 8)
    with torch.no_grad():
        model.tokens.weight.normal_(0.0, 0.02)   # booklm-base-init
    groups = [{"params": [model.tokens.weight], "weight_decay": 0.1},
              {"params": [model.norm.weight], "weight_decay": 0.0}]
    opt = torch.optim.AdamW(groups, lr=3e-2, betas=(0.9, 0.95))
    return {"model": model, "optimizer": opt,
            "scheduler": torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda),
            "scaler": torch.amp.GradScaler("cpu", growth_interval=3),
            "loader": Loader(WINDOWS, batch_size=4, seed=seed)}

def train_step(run):
    model, opt, scaler = run["model"], run["optimizer"], run["scaler"]
    x, y = run["loader"].next_batch()
    loss = F.cross_entropy(model(x, drop=0.1).flatten(0, 1), y.flatten())
    scaler.scale(loss).backward()                # ошибка * S
    scaler.unscale_(opt)                         # / S, поиск inf
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt)                             # при inf пропуск
    scaler.update()
    run["scheduler"].step()
    opt.zero_grad(set_to_none=True)
    return loss.item()

PARTS = ("model", "optimizer", "scheduler", "scaler", "loader")

def write_atomic(path, data):
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
    with os.fdopen(fd, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())                     # байты на диске
    os.replace(tmp, path)                        # одно переименование

def save_state(run, path):
    state = {name: run[name].state_dict() for name in PARTS}
    np_state = list(np.random.get_state())
    np_state[1] = np_state[1].tolist()           # массив -> список
    state["rng"] = {"torch": torch.get_rng_state(), "numpy": np_state,
                    "python": random.getstate()}
    buffer = io.BytesIO()
    torch.save(state, buffer)
    write_atomic(path, buffer.getvalue())

def load_state(run, path, skip=()):
    state = torch.load(path, weights_only=True)  # без чужого кода
    for name in PARTS:
        if name not in skip:
            run[name].load_state_dict(state[name])
    if "rng" not in skip:
        rng = state["rng"]
        torch.set_rng_state(rng["torch"])
        np_state = list(rng["numpy"])
        np_state[1] = np.array(np_state[1], dtype=np.uint32)
        np.random.set_state(tuple(np_state))
        random.setstate(rng["python"])

run = make_run(seed=0)
for _ in range(3):
    train_step(run)
with tempfile.TemporaryDirectory() as folder:
    path = os.path.join(folder, "state.pt")
    save_state(run, path)
    assert os.listdir(folder) == ["state.pt"]   # временного нет
    state = torch.load(path, weights_only=True)
assert set(state) == {*PARTS, "rng"}            # строки таблицы
assert int(state["optimizer"]["state"][0]["step"]) == 3
assert state["scheduler"]["last_epoch"] == 3
assert state["loader"] == {"seed": 0, "epoch": 0, "index": 12}
assert state["scaler"]["scale"] == 2.0 ** 17    # вырос после 3 шагов

# --- Побитное возобновление и четыре поломки -------------------------------
whole = make_run(seed=0)
losses = [train_step(whole) for _ in range(STEPS)]
reference = [p.detach().clone() for p in whole["model"].parameters()]
assert losses[-1] < losses[0]                    # 2,79 -> 2,24

def resumed(skip=()):
    first = make_run(seed=0)
    for _ in range(5):
        train_step(first)
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "state.pt")
        save_state(first, path)
        del first                                # процесс упал
        second = make_run(seed=123)              # новые объекты
        load_state(second, path, skip)
    scale = second["scaler"].get_scale()
    tail = [train_step(second) for _ in range(5)]
    gap = max((p - r).abs().max().item() for p, r in
              zip(second["model"].parameters(), reference))
    return gap, tail, scale

gap, tail, scale = resumed()
assert gap == 0.0 and tail == losses[5:]         # тот же прогон до бита
assert scale == 2.0 ** 17                        # S восстановлен

broken = {"только веса": ("optimizer", "scheduler", "scaler",
                          "loader", "rng"),
          "без загрузчика": ("loader",),
          "без планировщика": ("scheduler",),
          "без генераторов": ("rng",)}
gaps = {name: resumed(skip) for name, skip in broken.items()}
assert all(g > 1e-3 for g, _, _ in gaps.values())  # видна каждая
# измерено: 0,101; 0,034; 0,059; 0,015
_, tail_no_sched, _ = gaps["без планировщика"]
assert tail_no_sched[:2] == losses[5:7]          # шаг 6 ещё верный,
assert tail_no_sched[2] != losses[7]             # шаг 7 уже нет

# --- Выпускаемые веса и манифест -------------------------------------------
import hashlib, json

class Block(nn.Module):                # блок BookLM без RoPE
    def __init__(self, d=256, kv=64, f=768):
        super().__init__()
        lin = lambda i, o: nn.Linear(i, o, bias=False)
        self.norm_1 = nn.RMSNorm(d, eps=1e-6)   # имена как в главе 4
        self.attention = nn.ModuleDict(dict(
            q=lin(d, d), k=lin(d, kv), v=lin(d, kv), out=lin(d, d)))
        self.norm_2 = nn.RMSNorm(d, eps=1e-6)
        self.ffn = nn.ModuleDict(dict(
            gate=lin(d, f), up=lin(d, f), down=lin(f, d)))

    def forward(self, x):                        # [B,T,d] -> [B,T,d]
        b, t, _ = x.shape
        att, ffn, h = self.attention, self.ffn, self.norm_1(x)
        q = att["q"](h).view(b, t, 8, 32).transpose(1, 2)
        k, v = (att[n](h).view(b, t, 2, 32).transpose(1, 2) for n in "kv")
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                           enable_gqa=True)
        x = x + att["out"](a.transpose(1, 2).reshape(b, t, -1))
        h = self.norm_2(x)
        return x + ffn["down"](F.silu(ffn["gate"](h)) * ffn["up"](h))

class BookLM(TinyLM):                  # TinyLM + шесть блоков
    def __init__(self, vocab_size=8192, d=256, layers=6):
        super().__init__(vocab_size, d)
        self.blocks = nn.Sequential(*(Block(d) for _ in range(layers)))

    def forward(self, ids):                      # [B,T] -> [B,T,V]
        return self.output(self.norm(self.blocks(self.tokens(ids))))

def release(model):            # BF16, связанный тензор — один раз
    return {name: p.detach().to(torch.bfloat16)
            for name, p in model.named_parameters()}

def sha256(weights):
    h = hashlib.sha256()
    for name in sorted(weights):                 # порядок не важен
        t = weights[name]
        h.update(name.encode() + t.view(torch.int16).numpy().tobytes())
    return h.hexdigest()

torch.manual_seed(0)
booklm = BookLM()
torch.manual_seed(0)                             # как в разделе 7.1
with torch.no_grad():                            # booklm-base-init
    for name, p in booklm.named_parameters():
        if p.dim() == 2:
            out = name.endswith(("out.weight", "down.weight"))
            p.normal_(0.0, 0.02 / math.sqrt(12) if out else 0.02)
full = release(booklm)
assert booklm.tokens.weight is booklm.output.weight
assert "blocks.0.ffn.down.weight" in full        # имена главы 4
assert sum(t.numel() for t in full.values()) == 6_622_464
assert sum(t.numel() * t.element_size()
           for t in full.values()) == 13_244_928

parent = release(make_run(seed=0)["model"])      # родитель прогона
manifest = {"id": f"tinylm-step-{STEPS}",          # BookLM: base-step-N
            "parents": {"tinylm-init": sha256(parent)},
            "step": STEPS, "tokens_seen": STEPS * 4 * T, "precision": "fp32",
            "train_loss": round(losses[-1], 4),
            "weights_sha256": sha256(release(whole["model"]))}

def check_parent(manifest, parent_weights, parent_id="tinylm-init"):
    if sha256(parent_weights) != manifest["parents"][parent_id]:
        raise ValueError("родитель не совпал: другой прогон")

check_parent(manifest, parent)                   # тот же родитель
try:                                             # проверка 10
    check_parent(manifest, release(make_run(seed=1)["model"]))
    raise AssertionError("чужой родитель прошёл проверку")
except ValueError:
    pass
assert json.loads(json.dumps(manifest)) == manifest

# --- Валидационная ошибка, перплексия и бит на байт ------------------------
@torch.no_grad()
def evaluate(model, windows):                    # windows: [n, T+1]
    x, y = windows[:, :-1], windows[:, 1:]
    nll = F.cross_entropy(model(x).flatten(0, 1), y.flatten(),
                          reduction="none").view(y.shape)
    per_window = nll.mean(1)
    sigma = per_window.std() / len(per_window) ** 0.5
    return nll.mean().item(), sigma.item()

def bits_per_byte(loss_per_token, n_tokens, n_bytes):
    return loss_per_token * n_tokens / (n_bytes * math.log(2))

def is_improvement(old, new, sigma):
    return new < old - 2 * sigma

text = "Мама мыла раму"
assert len(text) == 14 and len(text.encode("utf-8")) == 26
by_tokens = bits_per_byte(4.0, n_tokens=3, n_bytes=26)
by_bytes = bits_per_byte(12 / 26, n_tokens=26, n_bytes=26)
assert abs(by_tokens - by_bytes) < 1e-12
assert abs(by_tokens - 0.6659) < 1e-4
assert not is_improvement(3.412, 3.398, sigma=0.015)

ids = torch.randint(0, 8192, (4, 513),
                    generator=torch.Generator().manual_seed(0))
start_booklm, _ = evaluate(booklm, ids)          # случайная BookLM
assert abs(start_booklm - math.log(8192)) < 0.1  # 9,07
assert 1.0 < math.exp(start_booklm) / 8192 < 1.1 # перплексия 8 660

grid_val = torch.arange(1000, 1000 + 8 * (T + 1)).view(8, T + 1)
VALID = (grid_val * 5 + grid_val // 7) % V       # отложенные окна
start, _ = evaluate(make_run(seed=0)["model"], VALID)
after, sigma = evaluate(whole["model"], VALID)
assert abs(start - math.log(V)) < 0.1            # 2,79 у родителя
assert is_improvement(start, after, sigma)       # 2,18; sigma 0,016

@torch.no_grad()
def greedy(model, prompt, n=20):
    ids = list(prompt)
    for _ in range(n):
        logits = model(torch.tensor([ids[-T:]]))  # не длиннее T
        ids.append(int(logits[0, -1].argmax()))
    return ids[len(prompt):]

@torch.no_grad()
def batch_hash(model):                           # контрольный пакет
    logits = model(VALID[:, :-1])
    return hashlib.sha256(logits.numpy().tobytes()).hexdigest()

again = make_run(seed=0)                         # повторный запуск
for _ in range(STEPS):
    train_step(again)
prompts = [[0], [5], [10, 15], [3, 8, 13], [7]]
assert ([greedy(whole["model"], p) for p in prompts]
        == [greedy(again["model"], p) for p in prompts])
assert batch_hash(whole["model"]) == batch_hash(again["model"])

# --- Бюджет BookLM по Chinchilla -------------------------------------------
import time

N = 6_622_464                          # паспорт BookLM
D = 20 * N                             # правило Chinchilla
C = 6 * N * D                          # 6N на токен, D токенов
steps = math.ceil(D / (64 * 512))      # 32 768 токенов на шаг
assert D == 132_449_280 and steps == 4043        # 4042,03 вверх
assert abs(C / 5.26e15 - 1) < 1e-3

def mfu(tokens_per_second, peak_flops):
    return 6 * N * tokens_per_second / peak_flops

assert abs(C / (100e12 * 0.40) - 131.6) < 0.1     # секунд, MFU 40 %
assert abs(mfu(1_006_500, 100e12) - 0.40) < 1e-3  # ~1 млн ток./с

opt = torch.optim.AdamW(booklm.parameters(), lr=3e-4)
batch = torch.randint(0, 8192, (8, 513))
for _ in range(3):                     # первый шаг — прогрев
    t0 = time.perf_counter()
    F.cross_entropy(booklm(batch[:, :-1]).flatten(0, 1),
                    batch[:, 1:].flatten()).backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    speed = 8 * 512 / (time.perf_counter() - t0)  # измерение
assert speed > 0
print(f"{speed:.0f} ток./с, {D / speed / 3600:.1f} ч на CPU")
