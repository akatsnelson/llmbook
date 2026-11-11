"""Глава 5 на обученной BookLM: config.json, скользящее окно, эксперты внутри SwiGLU.

Опыты (время — на процессоре Apple M2 Max):
  ch05_config_json — раздел «Паспорт модели», рубрика «А теперь на нашей модели: config.json».
      Экспортирует BookLM в формат Hugging Face (booklm/export_hf.py, нужен пакет transformers),
      печатает поля config.json, число параметров тремя способами, размер KV-кэша и файла, расхождение
      логитов и жадные продолжения двух классов. Около 10 с.
  ch05_sliding_window — раздел «Скользящее окно и разреженное внимание», рубрика «А теперь на нашей
      модели: скользящее окно». Доля внимания дальше 16/64/128 позиций по слоям, перплексия и размер
      кэша при разных окнах, продолжения с полной маской и с окном 16. Около 20 с.
      Данные рисунка: ch05-sliding-window.csv.
  ch05_moe_neurons — раздел «Смесь экспертов», рубрика «А теперь на нашей модели: эксперты». Сколько
      нейронов SwiGLU несут 90 % активации токена и перплексия, если оставить токену 1–2 группы
      нейронов из 4 (как экспертов). Около 25 с.
txt опыты не пишут: вывод на экран, у ch05_sliding_window ещё csv.
Опыты с длинным контекстом и вниманием по k ключам — в ch05_long_context.py.

Запуск из корня репозитория:
    python -m booklm.experiments.ch05                       # все опыты
    python -m booklm.experiments.ch05 ch05_sliding_window   # один опыт
    python -m booklm.experiments.ch05 --help                # список опытов
"""
import json
import subprocess
import sys
import tempfile
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F

if not __package__:  # запуск файлом (python booklm/experiments/ch05.py): корень репозитория — в путь поиска
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from booklm.experiments import run
from booklm.lab import MODELS, ROOT, load, perplexity, val_tokens
from booklm.model import apply_rope

OUT = MODELS / "experiments"


# ----------------------------------------------------------------------------- функции из главы 5

def architecture_passport(config, default_tied=None):
    d, q_heads = config["hidden_size"], config["num_attention_heads"]
    tied = config.get("tie_word_embeddings", default_tied)
    if tied is None:
        raise KeyError("tie_word_embeddings")
    return {
        "V": config["vocab_size"], "d": d, "L": config["num_hidden_layers"],
        "H_q": q_heads, "d_ff": config["intermediate_size"], "tied": tied,
        "H_kv": config.get("num_key_value_heads") or q_heads,
        "d_h": config.get("head_dim") or d // q_heads,
        "window": config.get("sliding_window"),
        "E": config.get("num_local_experts") or config.get("num_experts") or 1,
        "k": config.get("num_experts_per_tok") or 1,
    }


def parameter_count(p):
    d, d_h = p["d"], p["d_h"]
    attention = d * p["H_q"] * d_h + 2 * d * p["H_kv"] * d_h + p["H_q"] * d_h * d
    block = attention + 3 * d * p["d_ff"] + 2 * d
    return p["V"] * d * (1 if p["tied"] else 2) + p["L"] * block + d


def kv_cache_bytes(p, tokens, bytes_per_number=2):
    return 2 * p["L"] * tokens * p["H_kv"] * p["d_h"] * bytes_per_number


def window_mask(T, W):
    t = torch.arange(T)[:, None]
    j = torch.arange(T)[None, :]
    return (j <= t) & (j > t - W)


def interleaved_kv_bytes(p, kinds, tokens, bytes_per_number=2):
    kept = sum(min(tokens, p["window"]) if kind == "sliding_attention" else tokens
               for kind in kinds)
    return 2 * kept * p["H_kv"] * p["d_h"] * bytes_per_number


def moe_parameter_count(p, active=False):
    extra = (p["k"] if active else p["E"]) - 1
    router = p["d"] * p["E"] if p["E"] > 1 else 0
    return parameter_count(p) + p["L"] * (extra * 3 * p["d"] * p["d_ff"] + router)


BOOKLM = architecture_passport({
    "vocab_size": 8192, "hidden_size": 256, "num_hidden_layers": 6,
    "num_attention_heads": 8, "num_key_value_heads": 2,
    "intermediate_size": 768, "tie_word_embeddings": True})


# ----------------------------------------------------------------------------- 5.2: config.json

def ch05_config_json():
    """BookLM в формате Hugging Face: поля config.json, параметры, кэш и сверка логитов."""
    from transformers import AutoModelForCausalLM

    tok, model = load("base")
    with tempfile.TemporaryDirectory() as tmp:
        # экспорт тем же скриптом, которым сделан models/booklm/hf/base
        subprocess.run([sys.executable, str(ROOT / "booklm" / "export_hf.py"),
                        "--ckpt", str(MODELS / "booklm-base.pt"),
                        "--tokenizer", str(MODELS), "--out", tmp],
                       check=True, capture_output=True)
        config = json.load(open(Path(tmp) / "config.json"))
        file_bytes = (Path(tmp) / "model.safetensors").stat().st_size
        hf = AutoModelForCausalLM.from_pretrained(tmp, dtype=torch.float32).eval()

    # в transformers 5 база RoPE лежит внутри rope_parameters
    rope = config.get("rope_theta") or config["rope_parameters"]["rope_theta"]
    print("model_type:", config["model_type"])
    for key in ("vocab_size", "hidden_size", "num_hidden_layers", "num_attention_heads",
                "num_key_value_heads", "head_dim", "intermediate_size", "hidden_act",
                "rms_norm_eps", "max_position_embeddings", "tie_word_embeddings"):
        print(f"  {key:24} {config[key]}")
    print(f"  {'rope_theta':24} {rope}")
    print(f"  {'sliding_window':24} {config.get('sliding_window', 'нет')}")
    print(f"  {'num_local_experts':24} {config.get('num_local_experts', 'нет')}")

    p = architecture_passport(config)
    print(f"parameter_count по полям:  {parameter_count(p):,}")
    print(f"model.num_params():        {model.num_params():,}")
    print(f"LlamaForCausalLM:          {hf.num_parameters():,}")
    print(f"KV-кэш на 512 токенов:     {kv_cache_bytes(p, 512) // 1024} КБ")
    print(f"model.safetensors:         {file_bytes:,} байт "
          f"(2 байта на число: {2 * model.num_params():,})")

    ids = torch.tensor([[1] + tok.encode("Мама мыла раму, а потом она пошла спать.").ids])
    with torch.no_grad():
        ours = model(ids)
        theirs = hf(ids).logits
        # наша модель с весами, округлёнными до BF16, как в файле экспорта
        rounded = load("base")[1]
        for w in rounded.parameters():
            w.copy_(w.to(torch.bfloat16).float())
        print(f"расхождение логитов: {(theirs - ours).abs().max():.3f}"
              f" (веса FP32 против BF16), "
              f"{(theirs - rounded(ids)).abs().max():.1e} (одни и те же веса BF16)")
        n = ids.size(1)
        a = model.generate(ids, max_new=30, temperature=0)[0, n:].tolist()
        b = hf.generate(ids, max_new_tokens=30, do_sample=False)[0, n:].tolist()
        c = rounded.generate(ids, max_new=30, temperature=0)[0, n:].tolist()
    show = lambda x: " ".join(tok.decode(x).split())        # в одну строку
    print("BookLM (FP32):   ", show(a))
    print("LlamaForCausalLM:", show(b))
    same = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), 30)  # первое расхождение
    print(f"совпало первых токенов: {same} из 30; "
          f"с нашей моделью на весах BF16 — {'все' if b == c else 'не все'}")


# ----------------------------------------------------------------------------- 5.3: скользящее окно

def masked_forward(self, x, cos, sin, cache=None, return_weights=False, mask=None):
    """GQASelfAttention.forward, где причинная маска заменена на mask [T, T] (True = ключ виден)."""
    b, t, d = x.shape
    q = self.q(x).view(b, t, self.heads, self.d_head).transpose(1, 2)
    k = self.k(x).view(b, t, self.kv_heads, self.d_head).transpose(1, 2)
    v = self.v(x).view(b, t, self.kv_heads, self.d_head).transpose(1, 2)
    q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    group = self.heads // self.kv_heads
    k, v = k.repeat_interleave(group, dim=1), v.repeat_interleave(group, dim=1)
    y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:t, :t])
    return self.out(y.transpose(1, 2).reshape(b, t, d)), None


def set_windows(model, windows, T=512):
    """windows[i] — ширина окна слоя i; None — полная причинная маска."""
    for block, W in zip(model.blocks, windows):
        block.attention.forward = partial(masked_forward, block.attention,
                                          mask=window_mask(T, W or T))


@torch.no_grad()
def greedy(model, ids, n):
    for _ in range(n):                     # без кэша: маска окна видна целиком
        ids = torch.cat([ids, model(ids)[:, -1:].argmax(-1)], dim=1)
    return ids


def ch05_sliding_window():
    """Как далеко смотрят слои BookLM и перплексия со скользящим окном."""
    tok, model = load("base")
    data = val_tokens()
    t = torch.arange(512)[128:, None]       # запросы с позиции 128
    j = torch.arange(512)[None, :]
    shares = torch.zeros(6, 4)
    with torch.no_grad():
        for s in range(0, 8 * 512, 512):     # 8 отрезков по 512, как в perplexity
            ids = data[s:s + 512][None]
            is_bos = (ids[0] == 1).float()
            _, weights = model(ids, return_weights=True)
            for layer, w in enumerate(weights):
                w = w[0, :, 128:]                    # [головы, запросы, ключи]
                for c, far in enumerate((16, 64, 128)):
                    shares[layer, c] += (w * (t - j >= far)).sum(-1).mean() / 8
                shares[layer, 3] += (w * is_bos).sum(-1).mean() / 8
    print("доля внимания дальше 16/64/128 позиций и на <bos> (запросы с 128-й):")
    print("слой   >16    >64   >128  <bos>")
    for layer, row in enumerate(shares.tolist()):
        print(f"{layer:4}" + "".join(f"{x:7.2f}" for x in row))
    with open(OUT / "ch05-sliding-window.csv", "w") as f:
        f.write("layer,far16,far64,far128,bos\n")
        for layer, row in enumerate(shares.tolist()):
            f.write(f"{layer}," + ",".join(f"{x:.4f}" for x in row) + "\n")

    schemes = [("полная маска", None, [None] * 6),
               ("окно во всех слоях", 128, [128] * 6),
               ("окно во всех слоях", 64, [64] * 6),
               ("окно во всех слоях", 16, [16] * 6),
               ("пять к одному", 64, [64] * 5 + [None])]
    print()
    print(f"{'схема':20} {'W':>4} {'перплексия':>11} {'кэш, КБ':>9}")
    for name, W, windows in schemes:
        set_windows(model, windows)
        kinds = ["full_attention" if w is None else "sliding_attention" for w in windows]
        kv = interleaved_kv_bytes(dict(BOOKLM, window=W or 512), kinds, 512) // 1024
        print(f"{name:20} {W or '—':>4} {perplexity(model):11.2f} {kv:9}")

    prompt = torch.tensor([[1] + tok.encode("Однажды маленький зайчик").ids])
    for name, windows in (("полная маска", [None] * 6), ("W=16", [16] * 6)):
        set_windows(model, windows)
        out = greedy(model, prompt, 80)[0, prompt.size(1):]
        text = " ".join(tok.decode(out.tolist()).split())          # в одну строку
        print(f"\n{name}: Однажды маленький зайчик {text}")


# ----------------------------------------------------------------------------- 5.4: эксперты в SwiGLU

def stories(n, after=50_000):
    """n отложенных историй <bos>…<eos> из той части, на которой не считается перплексия."""
    data = val_tokens(10 ** 6)
    starts = (data == 1).nonzero().flatten()
    starts = starts[starts >= after][:n + 1]
    return [data[a:b] for a, b in zip(starts[:-1], starts[1:])]


def balanced_kmeans(x, groups=4, steps=30):
    """k-средних по строкам x (единичной длины), в каждой группе ровно len(x) // groups строк."""
    size = len(x) // groups
    centers = x[torch.randperm(len(x))[:groups]]
    for _ in range(steps):
        dist = 1 - x @ centers.T                         # [нейроны, группы]
        label = torch.full((len(x),), -1)
        load = [0] * groups
        for flat in dist.flatten().argsort().tolist():   # сначала самые близкие пары
            n, g = divmod(flat, groups)
            if label[n] < 0 and load[g] < size:
                label[n], load[g] = g, load[g] + 1
        centers = F.normalize(torch.stack([x[label == g].mean(0) for g in range(groups)]), dim=1)
    return label


def oracle_route(module, inputs, label, k):
    """Крючок перед down: оставить у токена только k групп с наибольшей суммой |активаций|."""
    (a,) = inputs                                         # [B, T, 768]
    score = torch.zeros(*a.shape[:-1], 4).index_add_(-1, label, a.abs())
    keep = torch.zeros_like(score).scatter_(-1, score.topk(k, dim=-1).indices, 1.0)
    return (a * keep[..., label],)


def top_neurons(module, inputs, n):
    """Крючок перед down: оставить у токена n нейронов с наибольшим |активации|, без групп."""
    (a,) = inputs
    keep = torch.zeros_like(a).scatter_(-1, a.abs().topk(n, dim=-1).indices, 1.0)
    return (a * keep,)


def ch05_moe_neurons():
    """Сколько нейронов SwiGLU работает на токене и группы нейронов как эксперты."""
    torch.manual_seed(0)
    tok, model = load("base")
    acts = [[] for _ in model.blocks]                     # активации входа down по слоям
    hooks = [b.ffn.down.register_forward_pre_hook(
        lambda m, inp, i=i: acts[i].append(inp[0][0])) for i, b in enumerate(model.blocks)]
    with torch.no_grad():
        for story in stories(20):
            model(story[None])
    for h in hooks:
        h.remove()
    acts = [torch.cat(a) for a in acts]                   # [токены, 768]
    print(f"токенов: {len(acts[0])} из 20 отложенных историй")

    print("доля нейронов, несущих 90 % суммы |активаций| токена:")
    for i, a in enumerate(acts):
        mass = a.abs().sort(dim=1, descending=True).values.cumsum(1)
        need = (mass < 0.9 * mass[:, -1:]).sum(1) + 1
        print(f"  слой {i}: {need.float().mean() / 768:.2f}")

    clustered = [balanced_kmeans(F.normalize(a.abs().T, dim=1)) for a in acts]
    random_split = [torch.randperm(768) % 4 for _ in acts]

    def ppl(hook):                                        # перплексия с крючком в каждом блоке
        hooks = [b.ffn.down.register_forward_pre_hook(hook(i)) for i, b in enumerate(model.blocks)]
        value = perplexity(model)
        for h in hooks:
            h.remove()
        return value

    route = lambda labels, k: lambda i: partial(oracle_route, label=labels[i], k=k)
    best = lambda n: lambda i: partial(top_neurons, n=n)
    print("перплексия на отложенных историях:")
    print(f"  исходная SwiGLU, все 768 нейронов   {perplexity(model):7.2f}")
    print(f"  группы k-средних, k=2 из 4          {ppl(route(clustered, 2)):7.2f}")
    print(f"  группы k-средних, k=1 из 4          {ppl(route(clustered, 1)):7.2f}")
    print(f"  случайные группы, k=2 из 4          {ppl(route(random_split, 2)):7.2f}")
    print(f"  случайные группы, k=1 из 4          {ppl(route(random_split, 1)):7.2f}")
    print(f"  свои 384 нейрона у каждого токена   {ppl(best(384)):7.2f}")
    print(f"  свои 192 нейрона у каждого токена   {ppl(best(192)):7.2f}")
    moe = dict(BOOKLM, d_ff=192, E=4, k=2)
    print(f"параметров на токен у смеси 4 x 192, k=2: "
          f"{moe_parameter_count(moe, active=True):,} из {moe_parameter_count(moe):,}")


EXPERIMENTS = {f.__name__: f for f in (ch05_config_json, ch05_sliding_window, ch05_moe_neurons)}


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    run(EXPERIMENTS)
