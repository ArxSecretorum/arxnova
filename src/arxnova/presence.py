"""Подтверждение нового устройства на самом сервере (решение владельца 27.10.2026).

Раньше пустить новое устройство могло только уже подключённое устройство владельца. Переустановил
плеер на единственном приборе — и подтверждать некому, а код с сервера у Pi без экрана не увидеть.
Теперь сервер подтверждает сам, тем способом, какой у него есть:

- Mac, Windows, Linux с монитором — окно на экране сервера: «Пустить …? Цифры NN»;
- Raspberry Pi без монитора — короткое нажатие кнопки питания на корпусе (пока запрос ждёт, сервер
  запрещает системе выключаться по этой кнопке);
- прочий Linux без монитора — «дом без хозяина»: если у сервера не осталось устройств владельца,
  запрос из домашней сети пускается своим устройством сам.

Устройства владельца по-прежнему получают запрос и могут ответить первыми."""

from __future__ import annotations

import os
import platform
import sqlite3
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable

BUTTON = Path("/dev/input/by-path/platform-pwr_button-event")
DRM = Path("/sys/class/drm")
EV_KEY, KEY_POWER = 1, 116
_EVENT = struct.Struct("llHHi")  # struct input_event: timeval, type, code, value
# Устройство владельца, которое не появлялось столько времени, считается ушедшим (для «дома без хозяина»).
OWNER_GONE_S = 30 * 24 * 3600


def kind() -> str:
    s = platform.system()
    if s == "Darwin":
        return "mac"
    if s == "Windows":
        return "windows"
    try:
        if "raspberry pi" in Path("/proc/device-tree/model").read_bytes().decode(errors="ignore").lower():
            return "pi"
    except OSError:
        pass
    return "linux"


def monitor_connected(drm: Path = DRM) -> bool:
    """Подключён ли к серверу монитор: состояние HDMI/DP-разъёмов, читается без прав root."""
    for status in drm.glob("card*-*/status"):
        try:
            if status.read_text().strip() == "connected":
                return True
        except OSError:
            pass
    return False


def display_env() -> dict | None:
    """Окружение, в котором окно сервера появится на мониторе (служба запущена без рабочего стола
    в переменных). None — показать окно негде."""
    if kind() in ("mac", "windows"):
        return dict(os.environ)
    env = dict(os.environ)
    run = env.get("XDG_RUNTIME_DIR") or f"/run/user/{os.getuid()}"
    if not env.get("WAYLAND_DISPLAY") and Path(run, "wayland-0").exists():
        env["WAYLAND_DISPLAY"] = "wayland-0"
        env.setdefault("XDG_RUNTIME_DIR", run)
    if not env.get("DISPLAY") and Path("/tmp/.X11-unix/X0").exists():
        env["DISPLAY"] = ":0"
    return env if env.get("DISPLAY") else None  # окно на Tk, ему нужен X (на Wayland — Xwayland)


def method(k: str | None = None, monitor: bool | None = None, display: bool | None = None) -> str:
    """Как сервер подтверждает сам: window — окно, button — кнопка Pi, home — дом без хозяина."""
    k = k or kind()
    if k in ("mac", "windows"):
        return "window"
    mon = monitor_connected() if monitor is None else monitor
    disp = (display_env() is not None) if display is None else display
    if mon and disp:
        return "window"
    if k == "pi":
        return "button"
    return "home"


def owners_gone(conn: sqlite3.Connection, now: float | None = None) -> bool:
    """Не осталось устройств владельца: ни одного неотозванного, которое появлялось бы за месяц."""
    now = time.time() if now is None else now
    row = conn.execute(
        "SELECT 1 FROM devices WHERE role = 'owner' AND revoked = 0 AND COALESCE(last_seen, created_at) > ? LIMIT 1",
        (int(now - OWNER_GONE_S),)).fetchone()
    return row is None


class ServerConfirm:
    """Следит за запросами доступа и подтверждает их на самом сервере.

    pending() — ждущие запросы (у каждого id, name, check); approve(id, role) — пустить."""

    def __init__(self, pending: Callable[[], list], approve: Callable[[str, str], None], server_name: str,
                 log: Callable[[str], None] = lambda s: print(s, flush=True)) -> None:
        self.pending, self.approve, self.server_name, self.log = pending, approve, server_name, log
        self._windows: dict[str, subprocess.Popen] = {}
        self._inhibit: subprocess.Popen | None = None
        self._button_started = False
        self._stop = threading.Event()

    def start(self) -> None:
        threading.Thread(target=self._loop, name="server-confirm", daemon=True).start()

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        while not self._stop.wait(1.0):
            try:
                self.tick()
            except Exception as e:  # noqa: BLE001 — наблюдатель не должен ронять сервер
                self.log(f"подтверждение на сервере: {e}")

    def tick(self) -> None:
        pend = self.pending()
        ids = {r.id for r in pend}
        for rid in [rid for rid in self._windows if rid not in ids]:  # ответили в другом месте
            self._close_window(rid)
        m = method() if pend else None
        if m == "window":
            for r in pend:
                if r.id not in self._windows:
                    self._open_window(r)
        if m == "button":
            self._hold_button()
        else:
            self._release_button()

    # ---- окно ----
    def _open_window(self, r) -> None:
        env = display_env()
        if env is None:
            return
        p = subprocess.Popen([sys.executable, "-m", "arxnova.confirm_window", self.server_name, r.name, r.check],
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env, text=True)
        self._windows[r.id] = p
        threading.Thread(target=self._await_window, args=(r.id, r.name, p), daemon=True).start()

    def _await_window(self, rid: str, name: str, p: subprocess.Popen) -> None:
        out = (p.communicate()[0] or "").strip()
        if self._windows.get(rid) is not p:
            return
        self._windows.pop(rid, None)
        if out in ("owner", "guest"):
            self.approve(rid, out)
            self.log(f"«{name}»: {'своё устройство' if out == 'owner' else 'гость'} — подтверждено на экране сервера")

    def _close_window(self, rid: str) -> None:
        p = self._windows.pop(rid, None)
        if p and p.poll() is None:
            p.terminate()

    # ---- кнопка Raspberry Pi ----
    def _hold_button(self) -> None:
        if self._inhibit is None or self._inhibit.poll() is not None:
            # Пока запрос ждёт, короткое нажатие не выключает Pi, а подтверждает.
            self._inhibit = subprocess.Popen(
                ["systemd-inhibit", "--what=handle-power-key", "--mode=block", "--who=Arx Nova",
                 "--why=подтверждение нового устройства", "sleep", "infinity"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.log("Новое устройство ждёт: нажмите один раз кнопку на корпусе Raspberry Pi, чтобы пустить его")
        if not self._button_started and BUTTON.exists():
            self._button_started = True
            threading.Thread(target=self._read_button, name="pi-button", daemon=True).start()

    def _release_button(self) -> None:
        if self._inhibit is not None:
            if self._inhibit.poll() is None:
                self._inhibit.terminate()
            self._inhibit = None

    def _read_button(self) -> None:
        try:
            with BUTTON.open("rb") as f:
                while not self._stop.is_set():
                    data = f.read(_EVENT.size)
                    if len(data) < _EVENT.size:
                        return
                    _, _, typ, code, value = _EVENT.unpack(data)
                    if typ == EV_KEY and code == KEY_POWER and value == 1:
                        self.on_button()
        except OSError as e:
            self.log(f"кнопка Raspberry Pi недоступна: {e}")
            self._button_started = False

    def on_button(self) -> None:
        if self._inhibit is None:
            return  # запроса нет — кнопка остаётся кнопкой выключения
        pend = self.pending()
        if pend:
            r = pend[0]
            self.approve(r.id, "owner")
            self.log(f"«{r.name}»: своё устройство — подтверждено кнопкой на Raspberry Pi")
