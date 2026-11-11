"""Канареечный выпуск BookLM на одном ноутбуке: два сервера, прокси-маршрутизатор и записанная нагрузка.

Глава 17 «Выпуск и сопровождение модели», раздел 17.8 «Лабораторная работа», шаги 3–5. Всё работает на CPU, видеокарта не нужна.

  * Два сервера BookLM на http.server: v1 — инструктивная модель с правильным шаблоном диалога,
    v2 — те же веса, но сервер ставит после <|assistant|> лишний перевод строки (поломка шаблона
    из главы 10). У каждого есть /ready (проверка готовности: известный запрос и сумма весов) и
    /generate.
  * Прокси — маршрутизатор с рисунка 17.6: выбирает версию функцией
    choose_executor по номеру сессии, отправляет каждую пятую просьбу v1 теневой копией на v2 и
    сравнивает жадные токены, ведёт журнал с номером трассы, версией и SHA-256 весов.
  * Записанная нагрузка: 200 просьб из models/booklm/data/sft.jsonl с моментами отправки через 20 мс.
  * В конце — p50 и p99 времени ответа по версиям, ошибки, теневые совпадения и release_decision.

Запуск из корня репозитория: python labs/ch17/canary_proxy.py   (доля v2 — 20 %)
Версия v2 не проходит проверку готовности: python labs/ch17/canary_proxy.py --break-ready
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from booklm.lab import MODELS, load, prompt_ids, special  # noqa: E402
from labs.ch17.release import (canary_ok, choose_executor, percentile, release_decision,  # noqa: E402
                               sha256_of)

LOCK = threading.Lock()                                   # одна модель на два сервера: считаем по очереди


def make_backend(version, tok, model, weights_sha, extra_newline, ready):
    eos, nl = special(tok)["<eos>"], tok.encode("\n").ids

    def answer(prompt, max_new):
        ids = prompt_ids(tok, prompt, chat=True) + (nl if extra_newline else [])
        with LOCK, torch.no_grad():
            out = model.generate(torch.tensor([ids]), max_new=max_new, temperature=0, eos_id=eos)
        return out[0, len(ids):].tolist()

    known = answer("Расскажи сказку про кота", 8)        # эталон для проверки готовности

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, body):
            data = json.dumps(body, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):                                 # /ready
            ok = ready() and answer("Расскажи сказку про кота", 8) == known
            self._send(200 if ok else 503, {"version": version, "weights_sha256": weights_sha})

        def do_POST(self):                                # /generate
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            tokens = answer(req["prompt"], req.get("max_new", 16))
            self._send(200, {"version": version, "weights_sha256": weights_sha, "tokens": tokens})

        def log_message(self, *args):
            pass

    return Handler


def post(port, body, timeout=30):
    data = json.dumps(body, ensure_ascii=False).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/generate", data=data, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def is_ready(port):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/ready", timeout=10) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError):
        return False


def serve(handler, port):
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", type=float, default=0.2, help="доля сессий на v2")
    ap.add_argument("--break-ready", action="store_true", help="v2 не проходит проверку готовности")
    ap.add_argument("--requests", type=int, default=200)
    ap.add_argument("--port", type=int, default=8460)
    a = ap.parse_args()

    tok, model = load("instruct")
    weights_sha = sha256_of(MODELS / "booklm-instruct.pt")
    ports = {"v1": a.port + 1, "v2": a.port + 2}
    serve(make_backend("v1", tok, model, weights_sha, False, lambda: True), ports["v1"])
    serve(make_backend("v2", tok, model, weights_sha, True, lambda: not a.break_ready), ports["v2"])
    ready = {v: is_ready(p) for v, p in ports.items()}
    print("готовность:", ready)

    journal, shadow = [], {"same": 0, "total": 0}

    def proxy(i, session, prompt):
        """Маршрутизатор: версия по сессии, неготовая версия запросов не получает."""
        version = choose_executor({"session": session}, a.share if ready["v2"] else 0.0)
        body = {"prompt": prompt, "max_new": 16}
        t0 = time.perf_counter()
        try:
            out = post(ports[version], body)
            ok = True
        except (urllib.error.URLError, OSError):
            out, ok = {"tokens": [], "weights_sha256": None}, False
        seconds = time.perf_counter() - t0
        journal.append({"trace": f"t{i:04d}", "session": session, "version": version, "ok": ok,
                        "seconds": seconds, "weights": (out["weights_sha256"] or "")[:8]})
        if version == "v1" and ok and ready["v2"] and i % 5 == 0:   # теневая копия на v2
            twin = post(ports["v2"], body)
            shadow["total"] += 1
            shadow["same"] += twin["tokens"] == out["tokens"]

    with open(MODELS / "data" / "sft.jsonl", encoding="utf-8") as f:
        asks = [json.loads(line)["request"] for line in f][:a.requests]
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=8) as pool:      # разомкнутая нагрузка: моменты заданы заранее
        for i, prompt in enumerate(asks):
            time.sleep(max(0.0, start + 0.02 * i - time.perf_counter()))
            pool.submit(proxy, i, f"s{i}", prompt)

    print("журнал, первые записи:")
    for r in sorted(journal, key=lambda r: r["trace"])[:3]:
        print(f"  {r['trace']} {r['session']:<5} {r['version']} веса {r['weights']} "
              f"{r['seconds'] * 1000:6.1f} мс {'ok' if r['ok'] else 'ошибка'}")
    counts = {}
    for v in ("v1", "v2"):
        mine = [r for r in journal if r["version"] == v]
        errors = sum(not r["ok"] for r in mine)
        counts[v] = {"total": len(mine), "errors": errors}
        if mine:
            ms = [r["seconds"] * 1000 for r in mine]
            print(f"{v}: запросов {len(mine):3d}, ошибок {errors}, "
                  f"p50 {percentile(ms, 50):6.1f} мс, p99 {percentile(ms, 99):6.1f} мс")
        else:
            print(f"{v}: запросов 0")
    print(f"теневые копии: совпало {shadow['same']} из {shadow['total']}")
    v1, v2 = counts["v1"], counts["v2"]
    if v2["total"] == 0:
        print("решение: v2 не готова и запросов не получила")
        assert a.break_ready
        return
    baseline = v1["errors"] / max(v1["total"], 1)       # базовая доля — v1 в том же окне
    canary = {"errors": v2["errors"], "total": v2["total"],
              "shadow_ok": shadow["total"] > 0 and shadow["same"] == shadow["total"]}
    print("ошибки в норме:", canary_ok(canary["errors"], canary["total"], baseline))
    print("решение:", release_decision(a.share, canary, baseline, min_requests=20))


if __name__ == "__main__":
    main()
