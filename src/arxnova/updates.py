"""Есть ли новая версия сервера — чтобы плеер сказал владельцу «есть обновление»: у сервера на
Pi без экрана иначе узнать об этом негде. Раз в 12 часов спрашиваем GitHub о последнем выпуске
(только номер версии, о медиатеке и слушателях — ничего). Выключается ARXNOVA_NO_UPDATE_CHECK=1."""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.request

URL = "https://api.github.com/repos/ArxSecretorum/arxnova/releases/latest"
EVERY_S = 12 * 3600

_lock = threading.Lock()
_state = {"latest": None, "at": 0.0, "busy": False}


def _fetch() -> None:
    try:
        req = urllib.request.Request(URL, headers={"Accept": "application/vnd.github+json", "User-Agent": "arxnova"})
        with urllib.request.urlopen(req, timeout=10) as r:
            tag = json.load(r).get("tag_name", "")
        with _lock:
            _state["latest"] = tag.lstrip("v") or None
    except Exception:  # нет сети — промолчим, спросим в следующий раз
        pass
    finally:
        with _lock:
            _state["busy"] = False


def latest() -> str | None:
    """Последняя известная версия; проверка — в фоне, ответ сервера не ждёт сеть."""
    if os.environ.get("ARXNOVA_NO_UPDATE_CHECK"):
        return None
    with _lock:
        due = not _state["busy"] and time.time() - _state["at"] > EVERY_S
        if due:
            _state["busy"], _state["at"] = True, time.time()
        got = _state["latest"]
    if due:
        threading.Thread(target=_fetch, name="update-check", daemon=True).start()
    return got


def _parts(v: str) -> tuple[int, ...]:
    out = []
    for p in v.split("."):
        digits = "".join(ch for ch in p if ch.isdigit())
        out.append(int(digits) if digits else 0)
    return tuple(out)


def newer(candidate: str, current: str) -> bool:
    return _parts(candidate) > _parts(current)
