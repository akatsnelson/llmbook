"""Проверки модуля выпуска из главы 17 «Выпуск и сопровождение модели».

Функции берутся из labs/ch17/release.py — того же модуля, который глава строит по разделам. Каждая
проверка здесь способна упасть: рядом с ней стоит вход, на котором старая или неверная версия функции
дала бы другой ответ (например, порог 0 в checks или ключ страницы без суммы весов).

Запуск из корня репозитория: python labs/ch17/release_checks.py
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import math
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from labs.ch17.release import (bucket, canary_ok, checks, choose_executor, make_manifest,  # noqa: E402
                               page_keys, pass_chance, percentile, release_decision, sha256_of,
                               trace, verify_lineage, verify_parents)


def quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def main() -> None:
    # SHA-256: эталон FIPS 180-4 при чтении по одному байту и по 4 КБ
    with tempfile.TemporaryDirectory() as tmp:
        sample = Path(tmp, "abc.txt")
        sample.write_bytes(b"abc")
        assert sha256_of(sample, chunk=1).startswith("ba7816bf8f01cfea")
        big = Path(tmp, "big.bin")
        big.write_bytes(bytes(range(256)) * 4097)            # больше одного куска 4 КБ
        assert sha256_of(big, chunk=4096) == hashlib.sha256(big.read_bytes()).hexdigest()

        # Родословная с двумя родителями: подмена базы ловится, хотя адаптер цел
        base, lora, merged = (Path(tmp, n) for n in ("base.pt", "lora.pt", "merged.pt"))
        base.write_bytes(b"base-weights")
        lora.write_bytes(b"lora-adapter")
        merged.write_bytes(b"merged-weights")
        files = {"base": base, "lora": lora, "merged": merged}
        manifests = {"lora": make_manifest(lora, "lora", {"base": base}, "sft_lora"),
                     "merged": make_manifest(merged, "merged", {"base": base, "lora": lora}, "merge")}
        assert set(manifests["merged"]["parents"]) == {"base", "lora"}
        assert verify_lineage(manifests, files, "merged") == ["merged", "lora"]
        base.write_bytes(b"base-weights, but other")             # подменили базу
        try:
            verify_parents(manifests["merged"], files)
            raise AssertionError("подменённая база принята")
        except ValueError as error:
            assert "родитель base" in str(error)

    # checks: порог 0 — настоящий порог; при «low and v < low» эта строка прошла бы
    limits = {"точность, %": (70.0, None), "потеряно запросов": (None, 0)}
    assert not quiet(checks, {"точность, %": 70.4, "потеряно запросов": 3}, limits)
    assert quiet(checks, {"точность, %": 70.4, "потеряно запросов": 0}, limits)
    try:
        quiet(checks, {"точность, %": 70.4}, limits)
        raise AssertionError("отсутствующий замер пропущен")
    except KeyError:
        pass

    # Трасса и перцентили
    log = [{"t": 7.8, "trace": "a"}, {"t": 0.0, "trace": "b"}, {"t": 0.0, "trace": "a"},
           {"t": 9.0, "trace": "a"}]
    assert [r["t"] for r in trace(log, "a")] == [0.0, 7.8, 9.0]
    ttft = [0.2] * 95 + [4.0] * 5
    assert percentile(ttft, 50) == 0.2 and percentile(ttft, 95) == 0.2   # 95-е место — быстрый
    assert percentile(ttft, 99) == 4.0                                   # 99-е — медленный
    # ранг ровно p при p, для которых p / 100 * 100 в двоичной записи чуть больше p
    assert all(percentile(list(range(1, 101)), p) == p for p in (7, 14, 28, 55, 56))
    try:
        import numpy as np
        assert abs(np.percentile(ttft, 95) - 0.39) < 1e-9                # numpy интерполирует
    except ImportError:
        pass

    # Маршрутизатор: доля около 5 %, одна сессия — одна версия, начатый запрос — своя версия
    picks = [choose_executor({"session": f"s{i}"}, 0.05) for i in range(20_000)]
    share = picks.count("v2") / len(picks)
    assert abs(share - 0.05) < 4 * math.sqrt(0.05 * 0.95 / 20_000), share
    assert len({choose_executor({"session": "s7"}, 0.05) for _ in range(5)}) == 1
    assert 0.0 <= bucket("x") < 1.0
    assert choose_executor({"session": "s3", "kv_version": "v2"}, 0.0) == "v2"
    assert choose_executor({"session": "s3"}, 0.0) == "v1"

    # Порог канарейки и его мощность: числа из главы
    assert canary_ok(18, 3600, 0.005) and canary_ok(27, 3600, 0.005)
    assert not canary_ok(40, 3600, 0.005)
    assert abs(pass_chance(0.005) - 0.983) < 0.001        # ложная тревога 1,7 %
    assert abs(pass_chance(0.0075) - 0.551) < 0.001       # в 1,5 раза хуже: проходит в 55 %
    assert abs(pass_chance(0.01) - 0.073) < 0.001         # вдвое хуже: в 7 %

    # Ключ страницы: без суммы весов страницы двух моделей совпали бы
    tokens = list(range(100, 164))
    sha_a, sha_b = hashlib.sha256(b"a").hexdigest(), hashlib.sha256(b"b").hexdigest()
    assert page_keys(tokens, "") == page_keys(tokens, "")
    assert all(x != y for x, y in zip(page_keys(tokens, sha_a), page_keys(tokens, sha_b)))
    assert len(page_keys(tokens, sha_a)) == 4

    # Решение о выпуске: поломка шаблона откатывает выпуск без единой лишней ошибки
    assert release_decision(0.05, {"errors": 9, "total": 1800, "shadow_ok": True}, 0.005) == ("ждём", 0.05)
    # выше порога, но окно не набрано — ждём; ошибок больше 27 — окно уже не спасти
    assert release_decision(0.05, {"errors": 20, "total": 1800, "shadow_ok": True}, 0.005) == ("ждём", 0.05)
    assert release_decision(0.05, {"errors": 30, "total": 1800, "shadow_ok": True}, 0.005) == ("откат", 0.0)
    assert release_decision(0.05, {"errors": 17, "total": 3600, "shadow_ok": True}, 0.005) == ("расширяем", 0.25)
    assert release_decision(1.0, {"errors": 17, "total": 3600, "shadow_ok": True}, 0.005) == ("выпущено", 1.0)
    assert release_decision(0.05, {"errors": 40, "total": 3600, "shadow_ok": True}, 0.005) == ("откат", 0.0)
    assert release_decision(0.05, {"errors": 15, "total": 3600, "shadow_ok": False}, 0.005) == ("откат", 0.0)

    print("release_checks: OK")


if __name__ == "__main__":
    main()
