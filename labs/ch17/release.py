"""Модуль выпуска из главы 17 «Выпуск и сопровождение модели».

Каждая функция здесь — та же, что в листинге главы; глава показывает
её по частям, а лабораторные (release_checks.py, canary_proxy.py, manifest_lineage.py) импортируют отсюда.

    from labs.ch17.release import sha256_of, verify_parents, checks, canary_ok, release_decision

Функции по разделам главы:
  * 17.1 родословная:   sha256_of, make_manifest, verify_parents, verify_lineage;
  * 17.2 три проверки:  checks;
  * 17.6 наблюдаемость: trace, percentile;
  * 17.7 канарейка:     bucket, choose_executor, canary_ok, pass_chance, page_keys, release_decision.

Все проверки громкие: испорченный родитель, отсутствующий замер или неизвестная версия — исключение,
а не тихое False или неверное число.

Запуск из корня репозитория: python labs/ch17/release.py   (только проверяет, что модуль загружается;
проверки самих функций — в release_checks.py)
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import struct
import subprocess
from pathlib import Path

# ----------------------------------------------------------------------------- родословная


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    """SHA-256 файла, прочитанного частями по chunk байт (многогигабайтные веса в память не грузим)."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        while block := f.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             cwd=Path(__file__).resolve().parent, timeout=5)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def make_manifest(artifact: Path, artifact_id: str, parents: dict[str, Path],
                  transform: str, checks_passed: dict | None = None) -> dict:
    """Пишет <файл>.manifest.json: сумма файла, суммы ВСЕХ родителей, код, окружение, проверки.

    parents: идентификатор родителя -> путь к его файлу. Сумму родителя считаем по файлу сейчас,
    а не берём из чужого манифеста, — так манифест ребёнка фиксирует байты, от которых он собран.
    """
    try:
        import torch
        torch_version = torch.__version__
    except ImportError:
        torch_version = None
    manifest = {
        "id": artifact_id,
        "file": Path(artifact).name,
        "format": Path(artifact).suffix.lstrip("."),
        "sha256": sha256_of(artifact),
        "parents": {pid: sha256_of(p) for pid, p in parents.items()},
        "transform": transform,
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "torch": torch_version,
        "checks": checks_passed or {},
    }
    Path(f"{artifact}.manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return manifest


def verify_parents(child: dict, files: dict) -> None:
    """Сверяет сумму каждого родителя из манифеста ребёнка с файлом; при расхождении — ValueError."""
    for parent_id, expected in child["parents"].items():
        if parent_id not in files:
            raise KeyError(f"{child['id']}: файл родителя {parent_id} не передан")
        actual = sha256_of(files[parent_id])
        if actual != expected:
            raise ValueError(f"{child['id']}: родитель {parent_id} "
                             f"{expected[:8]} != {actual[:8]}")


def verify_lineage(manifests: dict[str, dict], files: dict[str, Path], top: str) -> list[str]:
    """Идёт от узла top вверх по всем родителям и проверяет каждое звено. Возвращает порядок обхода."""
    seen: list[str] = []
    stack = [top]
    while stack:
        node = stack.pop()
        if node in seen or node not in manifests:      # корень (корпус) манифеста может не иметь
            continue
        seen.append(node)
        child = manifests[node]
        verify_parents(child, files)
        if sha256_of(files[node]) != child["sha256"]:
            raise ValueError(f"{node}: файл не совпадает с собственным манифестом")
        stack.extend(child["parents"])
    return seen


# ----------------------------------------------------------------------------- три проверки


def checks(measured: dict, limits: dict, quiet: bool = False) -> bool:
    """Сверяет замеры с порогами (не меньше, не больше). None — границы нет.

    Порог 0 — настоящий порог: проверка идёт через `is not None`, а не через истинность числа.
    Отсутствующий замер — KeyError, а не пропущенная строка.
    """
    passed = True
    for name, (low, high) in limits.items():
        if low is None and high is None:
            raise ValueError(f"{name}: у порога нет ни одной границы")
        v = measured[name]
        ok = ((low is None or v >= low) and
              (high is None or v <= high))
        passed = passed and ok
        if not quiet:
            mark = "да" if ok else "НЕТ"
            print(f"{name:<22} {v:>6.1f}  [{low}, {high}]  {mark}")
    return passed


# ----------------------------------------------------------------------------- наблюдаемость


def trace(log: list[dict], trace_id: str) -> list[dict]:
    """Записи одного запроса из общего журнала, по времени."""
    return sorted((r for r in log if r["trace"] == trace_id), key=lambda r: r["t"])


def percentile(values: list[float], p: float) -> float:
    """Перцентиль по правилу ближайшего ранга (numpy по умолчанию интерполирует — глава 14)."""
    if not values:
        raise ValueError("перцентиль пустого списка")
    ordered = sorted(values)
    # сначала умножаем: p / 100 * n при p = 7 даёт 7.000000000000001, и ранг уехал бы на единицу
    rank = math.ceil(p * len(ordered) / 100)
    return ordered[max(rank, 1) - 1]


# ----------------------------------------------------------------------------- канарейка


def bucket(session_id: str) -> float:
    """Число из [0, 1), постоянное для сессии: реплики одного диалога не скачут между версиями."""
    digest = hashlib.sha256(session_id.encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def choose_executor(request: dict, canary_share: float) -> str:
    """Версия для запроса. Начатый запрос пути B (внешний цикл держит кэш) идёт туда, где его кэш."""
    if request.get("kv_version"):
        return request["kv_version"]
    return "v2" if bucket(request["session"]) < canary_share else "v1"


def canary_ok(errors: int, total: int, baseline_rate: float, max_ratio: float = 1.5) -> bool:
    if total <= 0:
        raise ValueError("канарейка без запросов: решать не по чему")
    return errors / total <= baseline_rate * max_ratio


def pass_chance(rate: float, total: int = 3600, limit: int = 27) -> float:
    """P(ошибок <= limit) из total при доле ошибок rate — биномиальное распределение."""
    return sum(math.comb(total, k) * rate**k * (1 - rate)**(total - k) for k in range(limit + 1))


PAGE = 16


def page_keys(tokens: list[int], weights_sha256: str) -> list[bytes]:
    """Цепочка ключей страниц KV-кэша (глава 10), где в начало цепочки положена сумма весов.

    Пустая строка вместо суммы даёт ключи главы 10: страницы разных версий модели совпали бы.
    """
    keys, prev = [], bytes.fromhex(weights_sha256)
    for i in range(len(tokens) // PAGE):
        data = struct.pack(f"<{PAGE}H", *tokens[i * PAGE:(i + 1) * PAGE])
        prev = hashlib.sha256(prev + data).digest()
        keys.append(prev)
    return keys


STAGES = [0.05, 0.25, 1.0]
MIN_REQUESTS = 3600


def release_decision(share: float, canary: dict, baseline_rate: float,
                     stages: list[float] = STAGES, min_requests: int = MIN_REQUESTS) -> tuple[str, float]:
    """«ждём» / «расширяем» / «выпущено» / «откат» и новая доля канарейки."""
    errors, total = canary["errors"], canary["total"]
    window = max(total, min_requests)  # порог — на полное окно: ранний взгляд не даёт ложной тревоги
    if not canary["shadow_ok"] or not canary_ok(errors, window, baseline_rate):
        return "откат", 0.0
    if total < min_requests:
        return "ждём", share
    bigger = [s for s in stages if s > share]
    return ("расширяем", bigger[0]) if bigger else ("выпущено", share)


if __name__ == "__main__":
    print("OK: release.py")
