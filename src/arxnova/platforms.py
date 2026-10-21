"""Профили платформ (ТЗ, ОД-3а): ядро связи одно, а то, в чём ОС отличаются, — здесь.

Сейчас сделан профиль macOS; Pi, Windows и Linux x86_64 — на своих этапах (§19). У остальных
платформ работают только общие действия (адрес, шлюз), без ОС-зависимых проверок.
"""

from __future__ import annotations

import ipaddress
import os
import re
import socket
import subprocess
import sys


class Profile:
    name = "общий"
    key = "generic"

    def local_ip(self) -> str | None:
        """Адрес компьютера в домашней сети: интерфейс маршрута по умолчанию."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("192.0.2.1", 9))  # пакет не уходит: только выбор интерфейса
                ip = s.getsockname()[0]
        except OSError:
            return None
        return None if ip.startswith("127.") else ip

    def gateway(self, local_ip: str | None) -> str | None:
        if not local_ip:
            return None
        # запасной вариант: первый адрес сети /24 (так у большинства домашних роутеров)
        return str(ipaddress.ip_network(f"{local_ip}/24", strict=False).network_address + 1)

    def global_ipv6(self) -> str | None:
        try:
            with socket.socket(socket.AF_INET6, socket.SOCK_DGRAM) as s:
                s.connect(("2001:4860:4860::8888", 9))
                ip = s.getsockname()[0]
        except OSError:
            return None
        a = ipaddress.ip_address(ip.split("%")[0])
        return ip if a.is_global else None

    def firewall(self, port: int) -> tuple[str, str] | None:
        """(статус, пояснение) или None, если проверки для платформы нет."""
        return None

    def keep_awake(self) -> "object | None":
        """Не давать системе засыпать, пока работает сервер. Возвращает то, что надо закрыть."""
        return None


class MacOS(Profile):
    name = "macOS"
    key = "macos"

    def gateway(self, local_ip):
        try:
            out = subprocess.run(["route", "-n", "get", "default"], capture_output=True, text=True, timeout=3).stdout
            m = re.search(r"gateway:\s*(\S+)", out)
            if m:
                ipaddress.IPv4Address(m.group(1))
                return m.group(1)
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
        return super().gateway(local_ip)

    FW = "/usr/libexec/ApplicationFirewall/socketfilterfw"

    def firewall(self, port):
        if not os.path.exists(self.FW):
            return None
        try:
            state = subprocess.run([self.FW, "--getglobalstate"], capture_output=True, text=True, timeout=3).stdout
            if "disabled" in state or "State = 0" in state:
                return "ok", "брандмауэр macOS выключен — входящие на порт не блокируются"
            block = subprocess.run([self.FW, "--getblockall"], capture_output=True, text=True, timeout=3).stdout
            if "set to enabled" in block.lower():
                return "fail", ("брандмауэр macOS блокирует все входящие — снимите «Блокировать все входящие "
                                "подключения» в Настройки → Сеть → Брандмауэр")
            app = os.path.realpath(sys.executable)
            res = subprocess.run([self.FW, "--getappblocked", app], capture_output=True, text=True, timeout=3).stdout
            if "permitted" not in res.lower():
                return "fail", f"брандмауэр macOS запрещает входящие для {app} — разрешите в Настройки → Сеть → Брандмауэр"
            return "ok", ("брандмауэр macOS включён; при первом запуске сервера macOS один раз спросит "
                          "«Разрешить входящие подключения» — ответьте «Разрешить»")
        except (OSError, subprocess.SubprocessError):
            return None

    def keep_awake(self):
        # caffeinate -s: система не засыпает, пока подключено питание; -w: пока жив сервер.
        try:
            return subprocess.Popen(["caffeinate", "-s", "-w", str(os.getpid())],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError:
            return None


def current() -> Profile:
    if sys.platform == "darwin":
        return MacOS()
    return Profile()
