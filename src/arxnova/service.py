"""Автозапуск сервера (ТЗ, УСТ-4, ОД-3а): своя служба под каждую ОС, ставится одной командой.

- macOS — LaunchAgent пользователя: у него есть доступ к папкам с музыкой; перезапуск при сбое.
- Linux — служба systemd пользователя (`systemctl --user`), «linger», чтобы жила без входа.
- Windows — задание Планировщика при входе пользователя.
Права администратора не нужны. `arxnova service remove` убирает всё обратно.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

LABEL = "com.arxnova.server"


def _exe() -> list[str]:
    """Чем запускать сервер: само приложение (собранное установщиком), установленная команда
    arxnova, иначе этот же Python с модулем."""
    if getattr(sys, "frozen", False):
        return [sys.executable]
    found = shutil.which("arxnova")
    if found:
        return [found]
    return [sys.executable, "-c", "import arxnova; arxnova.main()"]


def _log_dir(data_dir: Path) -> Path:
    d = data_dir / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------- macOS ----------

def _plist_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"


def _plist(cmd: list[str], data_dir: Path) -> str:
    from xml.sax.saxutils import escape

    args = "".join(f"<string>{escape(a)}</string>" for a in cmd + ["--data-dir", str(data_dir), "serve"])
    logs = _log_dir(data_dir)
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>{LABEL}</string>
  <key>ProgramArguments</key><array>{args}</array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><dict><key>SuccessfulExit</key><false/></dict>
  <key>ThrottleInterval</key><integer>10</integer>
  <key>ProcessType</key><string>Standard</string>
  <key>StandardOutPath</key><string>{escape(str(logs / "server.log"))}</string>
  <key>StandardErrorPath</key><string>{escape(str(logs / "server.log"))}</string>
</dict></plist>
"""


def _launchctl(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["launchctl", *args], capture_output=True, text=True)


# ---------- Linux ----------

def _unit_path() -> Path:
    base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / "systemd" / "user" / "arxnova.service"


def _unit(cmd: list[str], data_dir: Path) -> str:
    import shlex

    exec_line = " ".join(shlex.quote(a) for a in cmd + ["--data-dir", str(data_dir), "serve"])
    return f"""[Unit]
Description=Arx Nova — сервер домашней медиатеки
After=network-online.target

[Service]
ExecStart={exec_line}
Restart=on-failure
RestartSec=10
Nice=0
IOSchedulingClass=best-effort

[Install]
WantedBy=default.target
"""


# ---------- общее ----------

def install(data_dir: Path) -> str:
    cmd = _exe()
    if sys.platform == "darwin":
        p = _plist_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        _launchctl("bootout", f"gui/{os.getuid()}/{LABEL}")
        p.write_text(_plist(cmd, data_dir), encoding="utf-8")
        r = _launchctl("bootstrap", f"gui/{os.getuid()}", str(p))
        if r.returncode != 0:
            raise RuntimeError(r.stderr.strip() or "launchctl bootstrap не удался")
        return f"автозапуск включён (LaunchAgent {p})"
    if sys.platform == "win32":
        tr = subprocess.list2cmdline(cmd + ["--data-dir", str(data_dir), "serve"])
        r = subprocess.run(["schtasks", "/Create", "/F", "/SC", "ONLOGON", "/RL", "LIMITED",
                            "/TN", "ArxNova", "/TR", tr], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(r.stderr.strip() or r.stdout.strip())
        subprocess.run(["schtasks", "/Run", "/TN", "ArxNova"], capture_output=True)
        return "автозапуск включён (Планировщик заданий: ArxNova, при входе)"
    p = _unit_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(_unit(cmd, data_dir), encoding="utf-8")
    subprocess.run(["systemctl", "--user", "daemon-reload"], capture_output=True)
    r = subprocess.run(["systemctl", "--user", "enable", "--now", "arxnova.service"], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip())
    # чтобы служба жила и без входа в систему (сервер на мини-ПК); не вышло — работает, пока вы вошли
    linger = subprocess.run(["loginctl", "enable-linger", os.environ.get("USER", "")], capture_output=True)
    note = "" if linger.returncode == 0 else "; работает, пока вы вошли в систему (enable-linger не разрешён)"
    return f"автозапуск включён (systemd --user arxnova.service){note}"


def restart() -> None:
    """Перезапуск службы (сменили папку с музыкой и т.п.)."""
    if sys.platform == "darwin":
        _launchctl("kickstart", "-k", f"gui/{os.getuid()}/{LABEL}")
    elif sys.platform == "win32":
        subprocess.run(["schtasks", "/End", "/TN", "ArxNova"], capture_output=True)
        subprocess.run(["schtasks", "/Run", "/TN", "ArxNova"], capture_output=True)
    else:
        subprocess.run(["systemctl", "--user", "restart", "arxnova.service"], capture_output=True)


def stop() -> None:
    """Остановить сервер до следующего входа в систему (автозапуск остаётся включённым)."""
    if sys.platform == "darwin":
        _launchctl("bootout", f"gui/{os.getuid()}/{LABEL}")
    elif sys.platform == "win32":
        subprocess.run(["schtasks", "/End", "/TN", "ArxNova"], capture_output=True)
    else:
        subprocess.run(["systemctl", "--user", "stop", "arxnova.service"], capture_output=True)


def start(data_dir: Path) -> None:
    """Запустить остановленный сервер; автозапуска нет — включить его."""
    if sys.platform == "darwin":
        p = _plist_path()
        if not p.exists():
            install(data_dir)
            return
        _launchctl("bootstrap", f"gui/{os.getuid()}", str(p))
        _launchctl("kickstart", f"gui/{os.getuid()}/{LABEL}")
    elif sys.platform == "win32":
        r = subprocess.run(["schtasks", "/Run", "/TN", "ArxNova"], capture_output=True)
        if r.returncode != 0:
            install(data_dir)
    else:
        r = subprocess.run(["systemctl", "--user", "start", "arxnova.service"], capture_output=True)
        if r.returncode != 0:
            install(data_dir)


def remove() -> str:
    if sys.platform == "darwin":
        _launchctl("bootout", f"gui/{os.getuid()}/{LABEL}")
        _plist_path().unlink(missing_ok=True)
        return "автозапуск выключен"
    if sys.platform == "win32":
        subprocess.run(["schtasks", "/End", "/TN", "ArxNova"], capture_output=True)
        subprocess.run(["schtasks", "/Delete", "/F", "/TN", "ArxNova"], capture_output=True)
        return "автозапуск выключен"
    subprocess.run(["systemctl", "--user", "disable", "--now", "arxnova.service"], capture_output=True)
    _unit_path().unlink(missing_ok=True)
    subprocess.run(["systemctl", "--user", "daemon-reload"], capture_output=True)
    return "автозапуск выключен"


def leave_background() -> None:
    """Сервер отдаёт музыку и разбирает медиатеку — он не фоновая задача. Под фоновым приоритетом
    (старые службы: ProcessType=Background на Mac, idle-ввод-вывод на Linux, задача планировщика
    с приоритетом 7 на Windows) macOS держала его на энергоэффективных ядрах с урезанным диском:
    обход большого диска шёл 15 МБ/с вместо ~80 (22.10.2026). Снимаем это при каждом запуске."""
    try:
        if sys.platform == "darwin":
            os.setpriority(4, 0, 0)  # PRIO_DARWIN_PROCESS: выйти из режима «фон»
        elif sys.platform == "win32":
            import ctypes

            k = ctypes.windll.kernel32
            k.SetPriorityClass(k.GetCurrentProcess(), 0x20)  # NORMAL_PRIORITY_CLASS
        else:
            if os.getpriority(os.PRIO_PROCESS, 0) > 0:
                os.setpriority(os.PRIO_PROCESS, 0, 0)
    except Exception:
        pass


def status() -> tuple[bool, str]:
    """(включён ли автозапуск, пояснение)."""
    if sys.platform == "darwin":
        r = _launchctl("print", f"gui/{os.getuid()}/{LABEL}")
        if r.returncode != 0:
            return False, "автозапуск не включён — `arxnova service install`"
        state = next((l.split("=", 1)[1].strip() for l in r.stdout.splitlines() if l.strip().startswith("state =")), "?")
        return True, f"LaunchAgent {LABEL}: {state}"
    if sys.platform == "win32":
        r = subprocess.run(["schtasks", "/Query", "/TN", "ArxNova"], capture_output=True, text=True)
        return (r.returncode == 0, "задание ArxNova есть" if r.returncode == 0 else "автозапуск не включён")
    r = subprocess.run(["systemctl", "--user", "is-active", "arxnova.service"], capture_output=True, text=True)
    active = r.stdout.strip()
    return (active == "active", f"systemd --user: {active or 'нет службы'}")
