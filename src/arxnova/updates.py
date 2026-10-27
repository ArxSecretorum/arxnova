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
_state = {"latest": None, "wheel": None, "at": 0.0, "busy": False}


def _fetch() -> None:
    try:
        req = urllib.request.Request(URL, headers={"Accept": "application/vnd.github+json", "User-Agent": "arxnova"})
        with urllib.request.urlopen(req, timeout=10) as r:
            rel = json.load(r)
        tag = rel.get("tag_name", "")
        wheel = next((a.get("browser_download_url") for a in rel.get("assets", [])
                      if str(a.get("name", "")).endswith(".whl")), None)
        with _lock:
            _state["latest"] = tag.lstrip("v") or None
            _state["wheel"] = wheel
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


# ---- самообновление ночью (ТЗ Nova, установка без хлопот) ----
# Сервер на Pi без экрана иначе навсегда остаётся на версии, с которой его поставили. Ночью, когда
# музыку никто не слушает, сервер ставит последний выпуск тем же uv и перезапускается: служба
# (systemd / launchd) поднимает его снова. Только для установки через uv (install.sh); окно
# приложения на Mac/Windows обновляется своим установщиком. Выключить: ARXNOVA_NO_AUTO_UPDATE=1.

NIGHT_HOURS = range(3, 6)
IDLE_S = 30 * 60
RESTART_CODE = 75  # ненулевой: служба перезапускает «упавший» процесс


def uv_tool_install() -> str | None:
    """Путь к uv, если сервер стоит инструментом uv; иначе None."""
    import shutil
    import sys
    from pathlib import Path

    if "/uv/tools/arxnova" not in sys.prefix.replace("\\", "/"):
        return None
    uv = shutil.which("uv") or str(Path.home() / ".local" / "bin" / "uv")
    return uv if Path(uv).exists() else None


def due_for_update(now_local, last_listen_s: float, current: str) -> str | None:
    """Ставить ли обновление сейчас: ночь, тишина, есть новее. Возвращает адрес пакета."""
    if os.environ.get("ARXNOVA_NO_AUTO_UPDATE") or now_local.hour not in NIGHT_HOURS:
        return None
    if time.time() - last_listen_s < IDLE_S:
        return None
    got = latest()
    with _lock:
        wheel = _state["wheel"]
    return wheel if got and wheel and newer(got, current) else None


def install_and_restart(uv: str, wheel: str) -> None:
    import subprocess

    print(f"Самообновление: ставлю {wheel.rsplit('/', 1)[-1]}", flush=True)
    r = subprocess.run([uv, "tool", "install", "--force", "--python", "3.12", wheel],
                       capture_output=True, text=True, timeout=900)
    if r.returncode != 0:
        print(f"Самообновление не удалось: {r.stderr.strip()[-300:]}", flush=True)
        return
    print("Самообновление поставлено — перезапускаюсь", flush=True)
    os._exit(RESTART_CODE)
