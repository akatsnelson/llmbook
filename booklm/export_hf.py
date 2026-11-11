"""Экспорт BookLM в формат Hugging Face как LlamaForCausalLM.

Архитектура BookLM — маленькая LLaMA: RMSNorm, SwiGLU, RoPE, GQA, связанные веса, без смещений.
Одно отличие в раскладке: книга поворачивает пары соседних координат (x[2k], x[2k+1]),
а LLaMA в transformers — пары из половин вектора (x[k], x[k + d_h/2]). Математика та же,
поэтому достаточно переставить строки W_Q и W_K внутри каждой головы. После экспорта
логиты двух реализаций сравниваются на одном входе.

Три правила главы 17 (черновая выгрузка hf/instruct их нарушала и разобрана там как пример;
выгрузка этим скриптом лежит в hf/instruct-fixed):
  * родитель записан суммой SHA-256, а не только путём;
  * логиты сверяются по уже сохранённому файлу (веса пишутся в BF16, а не по модели в памяти в FP32);
  * шаблон диалога начинается с <bos>, как вход, на котором училась инструктивная BookLM.

Запуск из корня репозитория (процессора хватает):

    python booklm/export_hf.py --ckpt models/booklm/booklm-base.pt --tokenizer models/booklm --out runs/hf-base
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from model import BookLM, BookLMConfig


def interleaved_to_halves(w, heads, d_head):
    # w: [heads*d_head, d_model]; строки головы: 0,1,2,...,d_head-1 (пары соседние)
    perm = torch.cat([torch.arange(0, d_head, 2), torch.arange(1, d_head, 2)])
    return w.view(heads, d_head, -1)[:, perm, :].reshape(heads * d_head, -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    state = torch.load(a.ckpt, map_location="cpu")
    state = state.get("model", state)
    cfg = BookLMConfig()
    ours = BookLM(cfg)
    ours.load_state_dict(state)
    ours.eval()
    d_head = cfg.d_model // cfg.heads

    hf_cfg = LlamaConfig(
        vocab_size=cfg.vocab, hidden_size=cfg.d_model, intermediate_size=cfg.d_ff,
        num_hidden_layers=cfg.layers, num_attention_heads=cfg.heads, num_key_value_heads=cfg.kv_heads,
        max_position_embeddings=cfg.max_len, rope_theta=cfg.rope_base, rms_norm_eps=cfg.norm_eps,
        tie_word_embeddings=True, attention_bias=False, mlp_bias=False,
        bos_token_id=1, eos_token_id=2, pad_token_id=0, dtype="bfloat16")
    hf = LlamaForCausalLM(hf_cfg)
    sd = {"model.embed_tokens.weight": state["embed.weight"], "model.norm.weight": state["norm.weight"]}
    for i in range(cfg.layers):
        p, q = f"blocks.{i}.", f"model.layers.{i}."
        sd[q + "input_layernorm.weight"] = state[p + "norm_1.weight"]
        sd[q + "post_attention_layernorm.weight"] = state[p + "norm_2.weight"]
        sd[q + "self_attn.q_proj.weight"] = interleaved_to_halves(state[p + "attention.q.weight"], cfg.heads, d_head)
        sd[q + "self_attn.k_proj.weight"] = interleaved_to_halves(state[p + "attention.k.weight"], cfg.kv_heads, d_head)
        sd[q + "self_attn.v_proj.weight"] = state[p + "attention.v.weight"]
        sd[q + "self_attn.o_proj.weight"] = state[p + "attention.out.weight"]
        sd[q + "mlp.gate_proj.weight"] = state[p + "ffn.gate.weight"]
        sd[q + "mlp.up_proj.weight"] = state[p + "ffn.up.weight"]
        sd[q + "mlp.down_proj.weight"] = state[p + "ffn.down.weight"]
    sd["lm_head.weight"] = state["embed.weight"]
    missing, unexpected = hf.load_state_dict(sd, strict=False)
    assert not unexpected and set(missing) <= {"lm_head.weight"}, (missing, unexpected)
    hf.eval()

    ids = torch.randint(0, cfg.vocab, (1, 64), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        ref = ours(ids)
        diff_fp32 = (hf(ids).logits - ref).abs().max().item()
    print(f"перестановка строк, FP32: наибольшее расхождение логитов {diff_fp32:.2e}")
    assert diff_fp32 < 1e-3, diff_fp32         # проверка самой перестановки W_Q, W_K

    out = Path(a.out)
    hf.to(torch.bfloat16).save_pretrained(out, safe_serialization=True)
    tok = PreTrainedTokenizerFast(tokenizer_file=str(Path(a.tokenizer) / "tokenizer.json"),
                                  bos_token="<bos>", eos_token="<eos>", pad_token="<pad>")
    tok.chat_template = (
        "{{ bos_token }}"
        "{% for m in messages %}{% if m['role'] == 'user' %}<|user|>{{ m['content'] }}"
        "{% else %}<|assistant|>{{ m['content'] }}<eos>{% endif %}{% endfor %}"
        "{% if add_generation_prompt %}<|assistant|>{% endif %}")
    tok.save_pretrained(out)

    # сверка тех байтов, которые получит сервер: читаем сохранённое, считаем в FP32
    saved = LlamaForCausalLM.from_pretrained(out, dtype=torch.float32).eval()
    with torch.no_grad():
        got = saved(ids).logits
    diff_saved = (got - ref).abs().max().item()
    flips = int((got.argmax(-1) != ref.argmax(-1)).sum())
    print(f"сохранённый файл (BF16): расхождение {diff_saved:.3f}, жадный токен другой в {flips} из {ids.numel()}")
    first = tok.apply_chat_template([{"role": "user", "content": "x"}], add_generation_prompt=True)
    first = first["input_ids"] if isinstance(first, dict) or hasattr(first, "keys") else first
    assert first[0] == tok.bos_token_id, first[:3]

    note = {"source_checkpoint": Path(a.ckpt).name,   # имя внутри models/booklm
            "source_sha256": hashlib.sha256(Path(a.ckpt).read_bytes()).hexdigest(),
            "dtype": "bfloat16",
            "max_logit_diff_fp32_permutation": diff_fp32,
            "max_logit_diff_saved": diff_saved,
            "greedy_flips_saved": flips,
            "checked_positions": ids.numel(),
            "chat_template_starts_with_bos": True}
    json.dump(note, open(out / "booklm_export.json", "w"), ensure_ascii=False, indent=1)
    print("сохранено:", out)


if __name__ == "__main__":
    main()
