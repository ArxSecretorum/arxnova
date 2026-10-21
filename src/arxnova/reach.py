"""Доступность сервера из интернета (ТЗ, ОД-3 и ОД-3а): проброс порта, внешний адрес, диагноз.

Раз в минуту: адрес компьютера и шлюз (сеть могла смениться, ноутбук — проснуться), продление
проброса на половине аренды, внешний адрес по STUN, диагноз. Состояние отдаётся плееру
(`/v1/net`) и `arxnova doctor`. Коды диагноза плеер переводит сам.
"""

from __future__ import annotations

import ipaddress
import socket
import ssl
import threading
import time
from dataclasses import dataclass, field

from arxnova import platforms, portmap, stun

CHECK_S = 60
CGNAT = ipaddress.ip_network("100.64.0.0/10")

# Коды диагноза → фраза для doctor (плеер переводит по коду)
DIAGNOSIS = {
    "public": "белый адрес, порт открыт на роутере — сервер доступен из интернета",
    "manual": "роутер не открывает порт сам; если порт проброшен вручную — доступ есть",
    "cgnat": "провайдер выдаёт серый адрес (CGNAT) — из интернета до роутера не достучаться; "
             "попросите у провайдера белый IP-адрес",
    "double_nat": "два роутера подряд — порт открыт только на ближнем; переведите модем провайдера "
                  "в режим моста или пробросьте порт и на нём",
    "offline": "нет связи с интернетом",
    "starting": "проверка ещё идёт",
}


def note_address(path, ip: str) -> None:
    """Дописывает «время адрес», только если адрес не тот, что в последней строке."""
    from pathlib import Path

    p = Path(path)
    last = ""
    if p.exists():
        lines = p.read_text().strip().splitlines()
        last = lines[-1].split()[-1] if lines else ""
    if ip != last:
        with p.open("a") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M')} {ip}\n")


def address_changes(path) -> tuple[int, str | None]:
    """(сколько раз адрес менялся, с какого времени ведётся журнал)."""
    from pathlib import Path

    p = Path(path)
    if not p.exists():
        return 0, None
    lines = [l for l in p.read_text().splitlines() if l.strip()]
    return max(0, len(lines) - 1), (" ".join(lines[0].split()[:2]) if lines else None)


def _is_public(ip: str | None) -> bool:
    try:
        return bool(ip) and ipaddress.ip_address(ip).is_global
    except ValueError:
        return False


@dataclass
class Status:
    profile: str = ""
    local_ip: str | None = None
    gateway: str | None = None
    ipv6: str | None = None
    method: str | None = None  # pcp | natpmp | upnp | None
    router_ip: str | None = None
    public_ip: str | None = None
    port: int = 0
    diagnosis: str = "starting"
    loopback: bool | None = None  # удалось ли зайти на свой внешний адрес изнутри сети
    errors: list[str] = field(default_factory=list)
    checked_at: int = 0

    def endpoint(self) -> str | None:
        """Адрес для плеера вне дома, если он вообще может работать."""
        if self.public_ip and self.diagnosis in ("public", "manual"):
            return f"{self.public_ip}:{self.port}"
        return None

    def as_json(self) -> dict:
        return {
            "profile": self.profile, "method": self.method, "diagnosis": self.diagnosis,
            "external": self.endpoint(), "port": self.port, "ipv6": bool(self.ipv6),
            "loopback": self.loopback, "checked_at": self.checked_at,
        }


class Reach:
    def __init__(self, port: int, profile: platforms.Profile | None = None, history=None, fingerprint: str | None = None):
        self.profile = profile or platforms.current()
        self.port = port
        # журнал смен внешнего адреса: часто ли провайдер его меняет (нужна ли служба адресов)
        self.history = history
        # объявление в торрент-сети DHT: плеер найдёт дом и после смены адреса (ОД-3)
        self.fingerprint = fingerprint
        self._announced_at = 0.0
        self._announced_ip: str | None = None
        self.dht_nodes = 0
        self.status = Status(profile=self.profile.name, port=port)
        self._map: portmap.Mapping | None = None
        self._mapped_at = 0.0
        self._map_for: tuple[str | None, str | None] = (None, None)
        self._lock = threading.Lock()
        self._stop = threading.Event()

    # -- цикл --
    def start(self) -> None:
        threading.Thread(target=self._loop, name="reach", daemon=True).start()

    def _loop(self) -> None:
        while True:
            try:
                self.refresh()
            except Exception as e:  # проверка сети не роняет сервер
                self.status.errors = [str(e)]
            if self._stop.wait(CHECK_S):
                return

    def close(self) -> None:
        self._stop.set()
        with self._lock:
            if self._map and self.status.gateway and self.status.local_ip:
                portmap.remove(self.status.gateway, self.status.local_ip, self._map)
                self._map = None

    def refresh(self) -> Status:
        with self._lock:
            p = self.profile
            local = p.local_ip()
            gw = p.gateway(local)
            errors: list[str] = []
            # сеть сменилась — старый проброс ничего не значит
            if (local, gw) != self._map_for:
                self._map = None
            due = self._map is None or (
                self._map.lifetime and time.monotonic() - self._mapped_at >= self._map.lifetime / 2)
            if local and gw and due:
                m, errors = portmap.add(gw, local, self.port, prev=self._map)
                if m:
                    self._map, self._mapped_at, self._map_for = m, time.monotonic(), (local, gw)
                else:
                    self._map = None
            router_ip = self._map.external_ip if self._map else None
            if gw and not router_ip:
                try:
                    router_ip = portmap.natpmp_external(gw)
                except (portmap.MapError, OSError):
                    pass
            public = stun.external_ip(local)
            st = Status(profile=p.name, local_ip=local, gateway=gw, ipv6=p.global_ipv6(),
                        method=self._map.method if self._map else None, router_ip=router_ip,
                        public_ip=public, port=self.port, errors=errors, checked_at=int(time.time()))
            st.diagnosis = self._diagnose(st)
            if st.endpoint():
                st.loopback = self._loopback(st.public_ip, self.port)
            if self.history and public:
                note_address(self.history, public)
            self.status = st
        self._announce(local, public)
        return st

    def _announce(self, local: str | None, public: str | None) -> None:
        """Раз в 15 минут и сразу после смены адреса — «я здесь» в DHT (вне блокировки:
        поиск узлов занимает секунды)."""
        from arxnova import dht

        if not (self.fingerprint and public and local):
            return
        due = time.monotonic() - self._announced_at >= dht.ANNOUNCE_EVERY_S or public != self._announced_ip
        if not due:
            return
        try:
            self.dht_nodes = dht.announce(self.fingerprint, self.port, local)
        except OSError:
            self.dht_nodes = 0
        self._announced_at, self._announced_ip = time.monotonic(), public

    @staticmethod
    def _diagnose(st: Status) -> str:
        if not st.public_ip:
            return "offline"
        if st.router_ip:
            ip = ipaddress.ip_address(st.router_ip)
            if ip in CGNAT:
                return "cgnat"
            if ip.is_private or st.router_ip != st.public_ip:
                return "double_nat"
        return "public" if st.method else "manual"

    @staticmethod
    def _loopback(ip: str, port: int) -> bool:
        """Заходим на свой внешний адрес изнутри. Не все роутеры это умеют (петля NAT), поэтому
        «нет» здесь не значит, что снаружи не работает; «да» — значит, проброс точно жив."""
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        try:
            with socket.create_connection((ip, port), timeout=2) as raw, ctx.wrap_socket(raw) as s:
                s.sendall(b"GET /v1/health HTTP/1.0\r\n\r\n")
                return b"arxnova" in s.recv(2048)
        except OSError:
            return False
