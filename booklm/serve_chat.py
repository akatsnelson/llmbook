"""Окно чата с BookLM на одной странице. Без внешних зависимостей, кроме torch и tokenizers.

Запуск из корня репозитория (процессора хватает):

    python booklm/serve_chat.py --base models/booklm/booklm-base.pt \
        --instruct models/booklm/booklm-instruct.pt --tokenizer models/booklm

Откройте http://localhost:8080 (другой порт — флаг --port; на удалённой машине — через
ssh -L 8080:localhost:8080).
Можно выбрать модель (базовая продолжает текст, инструктивная выполняет просьбу),
температуру и top-k, а под ответом увидеть, из каких токенов он собран.
"""
import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
from tokenizers import Tokenizer

try:                                    # импорт как пакет: from booklm.sft_lora import ...
    from .model import BookLM, BookLMConfig
except ImportError:                     # запуск файлом: python booklm/<скрипт>.py
    from model import BookLM, BookLMConfig

PAGE = """<!doctype html><html lang="ru"><head><meta charset="utf-8"><title>BookLM</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
body{font-family:system-ui,sans-serif;max-width:760px;margin:24px auto;padding:0 16px;color:#1f2430;background:#fcfbf6}
h1{font-size:22px;margin:0 0 4px} .sub{color:#6a7285;margin-bottom:16px;font-size:14px}
textarea{width:100%;height:70px;font:inherit;padding:8px;border:1px solid #c9cfdb;border-radius:8px;box-sizing:border-box}
.row{display:flex;gap:12px;flex-wrap:wrap;align-items:center;margin:10px 0;font-size:14px}
button{background:#1f3a8a;color:#fff;border:0;border-radius:8px;padding:10px 18px;font:inherit;cursor:pointer;min-height:44px}
.out{white-space:pre-wrap;background:#fff;border:1px solid #e4e9f0;border-radius:8px;padding:14px;min-height:80px;line-height:1.55}
.toks span{display:inline-block;margin:1px;padding:1px 4px;border-radius:4px;background:#dcedff;font-size:13px}
.toks span:nth-child(2n){background:#fff0c2} .meta{color:#6a7285;font-size:13px;margin-top:6px}
</style></head><body>
<h1>BookLM</h1><div class="sub">6 622 464 параметра · учебная модель книги «Большие языковые модели»</div>
<label for="p">Начало текста или просьба</label>
<textarea id="p">Напиши сказку про кота, который боялся темноты.</textarea>
<div class="row">
<label><input type="radio" name="m" value="instruct" checked> выполняет просьбы</label>
<label><input type="radio" name="m" value="base"> продолжает текст</label>
<label>температура <input id="t" type="number" step="0.1" min="0" max="2" value="0.8" style="width:60px"></label>
<label>top-k <input id="k" type="number" min="0" max="8192" value="40" style="width:70px"></label>
<button id="go">Сгенерировать</button></div>
<div class="out" id="out"></div><div class="meta" id="meta"></div>
<details><summary>Из каких токенов собран ответ</summary><div class="toks" id="toks"></div></details>
<script>
document.getElementById('go').onclick = async () => {
  out.textContent = '…'; toks.innerHTML=''; meta.textContent='';
  const body = {prompt: p.value, model: document.querySelector('input[name=m]:checked').value,
                temperature: +t.value, top_k: +k.value};
  const r = await fetch('/generate', {method:'POST', body: JSON.stringify(body)});
  const j = await r.json();
  out.textContent = j.text;
  meta.textContent = j.n_tokens + ' токенов за ' + j.seconds.toFixed(2) + ' с';
  j.tokens.forEach(s => {const e=document.createElement('span'); e.textContent=s; toks.appendChild(e);});
};
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True); ap.add_argument("--instruct")
    ap.add_argument("--tokenizer", required=True); ap.add_argument("--port", type=int, default=8080)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = Tokenizer.from_file(str(Path(a.tokenizer) / "tokenizer.json"))
    models = {}
    for name, path in (("base", a.base), ("instruct", a.instruct)):
        if path:
            m = BookLM(BookLMConfig()).to(dev).eval()
            m.load_state_dict(torch.load(path, map_location=dev))
            models[name] = m
    sp = {s: tok.token_to_id(s) for s in ("<bos>", "<eos>", "<|user|>", "<|assistant|>")}

    class H(BaseHTTPRequestHandler):
        def _send(self, code, body, ctype):
            data = body.encode("utf-8")
            self.send_response(code); self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)

        def do_GET(self):
            self._send(200, PAGE, "text/html; charset=utf-8")

        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            name = req.get("model", "instruct") if req.get("model") in models else next(iter(models))
            m = models[name]
            text = req.get("prompt", "")[:1000]
            # Текст пользователя кодируем без служебных токенов: набранное «<|assistant|>»
            # не должно стать номером 4 и подделать ход модели (глава 18).
            tok.encode_special_tokens = True
            body_ids = tok.encode(text).ids
            tok.encode_special_tokens = False
            if name == "instruct":
                ids = [sp["<bos>"], sp["<|user|>"]] + body_ids + [sp["<|assistant|>"]]
            else:
                ids = [sp["<bos>"]] + body_ids
            x = torch.tensor([ids], device=dev)
            t0 = time.time()
            with torch.no_grad():
                g = m.generate(x, max_new=300, temperature=float(req.get("temperature", 0.8)),
                               top_k=int(req.get("top_k", 40)) or None, eos_id=sp["<eos>"])
            new = [i for i in g[0, len(ids):].tolist() if i != sp["<eos>"]]
            body = {"text": (text + tok.decode(new)) if name == "base" else tok.decode(new),
                    "tokens": [tok.decode([i]) for i in new], "n_tokens": len(new), "seconds": time.time() - t0}
            self._send(200, json.dumps(body, ensure_ascii=False), "application/json; charset=utf-8")

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", a.port), H)   # если порт занят, ошибка будет здесь
    print(f"BookLM: http://localhost:{a.port}  (модели: {', '.join(models)})", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
