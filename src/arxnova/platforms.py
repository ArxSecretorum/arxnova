"""Профили платформ (ТЗ, ОД-3а): ядро связи одно, а то, в чём ОС отличаются, — здесь.

Сделаны профили macOS и Windows; у Linux и Pi работают только общие действия (адрес, шлюз): у них
брандмауэр по умолчанию выключен, а ufw показывает состояние только с правами root.
"""

from __future__ import annotations

import ipaddress
import json
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


# Строки PowerShell — имена из API, а не перевод: вывод netsh на русской Windows разбирать нельзя.
_WIN_FW = r"""
$ErrorActionPreference = 'SilentlyContinue'
$net = Get-NetConnectionProfile | Where-Object { $_.IPv4Connectivity -eq 'Internet' } | Select-Object -First 1
if (-not $net) { $net = Get-NetConnectionProfile | Select-Object -First 1 }
$cat = [string]$net.NetworkCategory
$name = if ($cat -eq 'DomainAuthenticated') { 'Domain' } else { $cat }
$fw = if ($name) { Get-NetFirewallProfile -Name $name -PolicyStore ActiveStore }
function Rows($rules) { @($rules | Where-Object { [string]$_.Direction -eq 'Inbound' } | ForEach-Object {
  @{ action = [string]$_.Action; enabled = [string]$_.Enabled; profile = [string]$_.Profile } }) }
$port = [string]$args[0]
$apps = @(); foreach ($p in $args[1..($args.Count-1)]) {
  $apps += Rows (Get-NetFirewallApplicationFilter -Program $p -PolicyStore ActiveStore | Get-NetFirewallRule) }
$ports = Rows (Get-NetFirewallPortFilter -Protocol TCP -PolicyStore ActiveStore |
  Where-Object { $_.LocalPort -eq $port } | Get-NetFirewallRule)
@{ category = $name; enabled = [string]$fw.Enabled; rules = @($apps) + @($ports) } | ConvertTo-Json -Depth 4 -Compress
"""


def windows_firewall_verdict(data: dict, app: str) -> tuple[str, str] | None:
    """Что сказать о брандмауэре Windows по ответу PowerShell. None — данных нет, молчим."""
    category, enabled = data.get("category"), data.get("enabled")
    if not category or enabled not in ("True", "False"):
        return None
    if enabled == "False":
        return "ok", f"брандмауэр Windows выключен для этой сети ({category}) — входящие не блокируются"
    here = [r for r in data.get("rules") or [] if r.get("enabled") == "True"
            and (r.get("profile") == "Any" or category in str(r.get("profile")))]
    # у Windows запрещающее правило сильнее разрешающего
    if any(r.get("action") == "Block" for r in here):
        return "fail", (f"брандмауэр Windows запрещает входящие для {app} — снимите запрет: Безопасность Windows → "
                        "Брандмауэр → Разрешить работу с приложением")
    if any(r.get("action") == "Allow" for r in here):
        return "ok", f"брандмауэр Windows разрешает входящие к серверу в этой сети ({category})"
    return "fail", (f"брандмауэр Windows не пропускает входящие к серверу в этой сети ({category}): в домашней сети сервер "
                    "не найдётся сразу, снаружи — дольше, через канал. Разрешите: Безопасность Windows → Брандмауэр → "
                    f"Разрешить работу с приложением → {app}")


class Windows(Profile):
    name = "Windows"
    key = "windows"

    @staticmethod
    def apps() -> list[str]:
        """Кого пропускать: в окружении venv слушает настоящий python, а не его копия."""
        return list(dict.fromkeys(os.path.realpath(p) for p in (sys.executable, getattr(sys, "_base_executable", None)) if p))

    @staticmethod
    def read_firewall(port: int, apps: list[str]) -> dict:
        quoted = " ".join("'" + a.replace("'", "''") + "'" for a in apps)
        out = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", f"& {{ {_WIN_FW} }} {port} {quoted}"],
            capture_output=True, text=True, errors="replace", timeout=25, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout
        return json.loads(out)

    def firewall(self, port):
        apps = self.apps()
        try:
            return windows_firewall_verdict(self.read_firewall(port, apps), apps[0])
        except (OSError, ValueError, subprocess.SubprocessError):
            return None


def current() -> Profile:
    if sys.platform == "darwin":
        return MacOS()
    if sys.platform == "win32":
        return Windows()
    return Profile()
