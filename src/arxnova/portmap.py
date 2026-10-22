"""Автоматический проброс порта на роутере (ТЗ, ОД-3 п. 3 и ОД-3а): PCP → NAT-PMP → UPnP-IGD.

Только стандартная библиотека: UDP к шлюзу (PCP, NAT-PMP — RFC 6887, RFC 6886) и SSDP + SOAP
(UPnP). Аренда — на LEASE_S, продление делает вызывающий; при остановке проброс снимается.
DMZ не используется никогда.
"""

from __future__ import annotations

import ipaddress
import os
import re
import socket
import struct
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from xml.etree import ElementTree as ET

LEASE_S = 3600
PMP_PORT = 5351
DESCRIPTION = "Arx Nova"


@dataclass
class Mapping:
    method: str  # pcp | natpmp | upnp
    external_ip: str | None
    external_port: int
    lifetime: int  # 0 — бессрочно (UPnP-роутер не умеет аренду), снимаем сами
    nonce: bytes = b""  # PCP
    control_url: str = ""  # UPnP
    service: str = ""  # UPnP
    udp: bool = False  # открыт ли и UDP-порт (канал пробивки NAT: тогда до него можно напрямую)
    udp_nonce: bytes = b""  # PCP, UDP-отображение


class MapError(Exception):
    pass


# ---------- PCP (RFC 6887) ----------

def _v4mapped(ip: str) -> bytes:
    return b"\x00" * 10 + b"\xff\xff" + socket.inet_aton(ip)


def _udp(gateway: str, payload: bytes, timeout: float = 0.6, tries: int = 3) -> bytes:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.settimeout(timeout)
        for _ in range(tries):
            s.sendto(payload, (gateway, PMP_PORT))
            try:
                while True:
                    data, (src, _) = s.recvfrom(1100)
                    if src == gateway:
                        return data
            except socket.timeout:
                continue
    raise MapError("шлюз не отвечает")


def pcp_map(gateway: str, local_ip: str, port: int, lifetime: int = LEASE_S, nonce: bytes | None = None,
            proto: int = 6) -> Mapping:
    """proto: 6 — TCP, 17 — UDP."""
    nonce = nonce or os.urandom(12)
    req = struct.pack("!BBHI16s", 2, 1, 0, lifetime, _v4mapped(local_ip))
    req += nonce + struct.pack("!B3xHH16s", proto, port, port, b"\x00" * 10 + b"\xff\xff" + b"\x00" * 4)
    data = _udp(gateway, req)
    if len(data) < 60 or data[0] != 2 or data[1] != 0x81:
        raise MapError("роутер не говорит PCP")
    result = data[3]
    if result != 0:
        raise MapError(f"PCP: код {result}")
    life = struct.unpack("!I", data[4:8])[0]
    if data[24:36] != nonce:
        raise MapError("PCP: чужой ответ")
    ext_port = struct.unpack("!H", data[42:44])[0]
    ext_ip = socket.inet_ntoa(data[56:60])
    return Mapping("pcp", ext_ip, ext_port, life, nonce=nonce)


# ---------- NAT-PMP (RFC 6886) ----------

def natpmp_external(gateway: str) -> str:
    data = _udp(gateway, b"\x00\x00")
    if len(data) < 12 or data[1] != 128:
        raise MapError("роутер не говорит NAT-PMP")
    if struct.unpack("!H", data[2:4])[0] != 0:
        raise MapError("NAT-PMP: ошибка адреса")
    return socket.inet_ntoa(data[8:12])


def natpmp_map(gateway: str, port: int, lifetime: int = LEASE_S, udp: bool = False) -> Mapping:
    ext_ip = natpmp_external(gateway)
    op = 1 if udp else 2  # RFC 6886: 1 — UDP, 2 — TCP
    data = _udp(gateway, struct.pack("!BBHHHI", 0, op, 0, port, port if lifetime else 0, lifetime))
    if len(data) < 16 or data[1] != 128 + op:
        raise MapError("NAT-PMP: странный ответ")
    code, _, internal, external, life = struct.unpack("!HIHHI", data[2:16])
    if code != 0:
        raise MapError(f"NAT-PMP: код {code}")
    return Mapping("natpmp", ext_ip, external, life)


# ---------- UPnP-IGD ----------

_WAN = ("urn:schemas-upnp-org:service:WANIPConnection:2", "urn:schemas-upnp-org:service:WANIPConnection:1",
        "urn:schemas-upnp-org:service:WANPPPConnection:1")


def upnp_find(local_ip: str, timeout: float = 2.0) -> tuple[str, str]:
    """(controlURL, тип службы) первого найденного интернет-шлюза."""
    msg = ("M-SEARCH * HTTP/1.1\r\nHOST: 239.255.255.250:1900\r\nMAN: \"ssdp:discover\"\r\nMX: 2\r\n"
           "ST: urn:schemas-upnp-org:device:InternetGatewayDevice:1\r\n\r\n").encode()
    locations: list[str] = []
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP) as s:
        s.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_IF, socket.inet_aton(local_ip))
        s.bind((local_ip, 0))
        s.settimeout(timeout)
        s.sendto(msg, ("239.255.255.250", 1900))
        try:
            while True:
                data, _ = s.recvfrom(4096)
                m = re.search(rb"(?im)^location:\s*(\S+)", data)
                if m:
                    loc = m.group(1).decode(errors="replace")
                    if loc not in locations:
                        locations.append(loc)
        except socket.timeout:
            pass
    for loc in locations:
        # описание отдаёт только роутер из домашней сети — чужие адреса не трогаем
        host = urllib.parse.urlsplit(loc).hostname or ""
        try:
            if not ipaddress.ip_address(host).is_private:
                continue
        except ValueError:
            continue
        try:
            xml = urllib.request.urlopen(loc, timeout=3).read(256 * 1024)
            root = ET.fromstring(xml)
        except Exception:
            continue
        for svc in root.iter():
            if not svc.tag.endswith("service"):
                continue
            kv = {c.tag.split("}")[-1]: (c.text or "").strip() for c in svc}
            if kv.get("serviceType") in _WAN and kv.get("controlURL"):
                return urllib.parse.urljoin(loc, kv["controlURL"]), kv["serviceType"]
    raise MapError("UPnP-шлюз не найден")


def _soap(url: str, service: str, action: str, args: dict[str, str]) -> dict[str, str]:
    body = "".join(f"<{k}>{v}</{k}>" for k, v in args.items())
    env = (f'<?xml version="1.0"?><s:Envelope xmlns:s="http://schemas.xmlsoap.org/soap/envelope/" '
           f's:encodingStyle="http://schemas.xmlsoap.org/soap/encoding/"><s:Body>'
           f'<u:{action} xmlns:u="{service}">{body}</u:{action}></s:Body></s:Envelope>').encode()
    req = urllib.request.Request(url, env, {"Content-Type": 'text/xml; charset="utf-8"',
                                            "SOAPAction": f'"{service}#{action}"'})
    try:
        raw = urllib.request.urlopen(req, timeout=4).read(64 * 1024)
    except urllib.error.HTTPError as e:
        raw = e.read(64 * 1024)
        code = re.search(rb"<errorCode>(\d+)</errorCode>", raw)
        raise MapError(f"UPnP {action}: ошибка {code.group(1).decode() if code else e.code}")
    out = {}
    for el in ET.fromstring(raw).iter():
        if el.text and len(el) == 0:
            out[el.tag.split("}")[-1]] = el.text.strip()
    return out


def upnp_map(local_ip: str, port: int, lifetime: int = LEASE_S, found: tuple[str, str] | None = None,
             proto: str = "TCP") -> Mapping:
    url, service = found or upnp_find(local_ip)
    args = {"NewRemoteHost": "", "NewExternalPort": str(port), "NewProtocol": proto,
            "NewInternalPort": str(port), "NewInternalClient": local_ip, "NewEnabled": "1",
            "NewPortMappingDescription": DESCRIPTION, "NewLeaseDuration": str(lifetime)}
    try:
        _soap(url, service, "AddPortMapping", args)
    except MapError as e:
        if "725" not in str(e):  # OnlyPermanentLeasesSupported
            raise
        lifetime = 0
        _soap(url, service, "AddPortMapping", {**args, "NewLeaseDuration": "0"})
    ext = None
    try:
        ext = _soap(url, service, "GetExternalIPAddress", {}).get("NewExternalIPAddress")
    except MapError:
        pass
    return Mapping("upnp", ext, port, lifetime, control_url=url, service=service)


# ---------- общее ----------

def add(gateway: str, local_ip: str, port: int, prev: Mapping | None = None) -> tuple[Mapping | None, list[str]]:
    """Пробует способы по порядку ОД-3а; при продлении — сначала тот, что сработал."""
    errors: list[str] = []
    order = ["pcp", "natpmp", "upnp"]
    if prev:
        order.remove(prev.method)
        order.insert(0, prev.method)
    for method in order:
        try:
            if method == "pcp":
                m = pcp_map(gateway, local_ip, port, nonce=prev.nonce if prev and prev.method == "pcp" else None)
            elif method == "natpmp":
                m = natpmp_map(gateway, port)
            else:
                found = (prev.control_url, prev.service) if prev and prev.method == "upnp" else None
                m = upnp_map(local_ip, port, found=found)
        except (MapError, OSError) as e:
            errors.append(f"{method}: {e}")
            continue
        _add_udp(gateway, local_ip, port, m, prev)
        return m, errors
    return None, errors


def _add_udp(gateway: str, local_ip: str, port: int, m: Mapping, prev: Mapping | None) -> None:
    """Тот же порт и для UDP — канала пробивки NAT (ОД-3 п. 4): роутер, который открывает порты
    сам, пустит плеер в канал напрямую, без встречного стука. Не вышло — не беда, есть пробивка."""
    try:
        if m.method == "pcp":
            nonce = prev.udp_nonce if prev and prev.method == "pcp" and prev.udp_nonce else None
            u = pcp_map(gateway, local_ip, port, nonce=nonce, proto=17)
            m.udp, m.udp_nonce = u.external_port == m.external_port, u.nonce
        elif m.method == "natpmp":
            m.udp = natpmp_map(gateway, port, udp=True).external_port == m.external_port
        else:
            upnp_map(local_ip, port, found=(m.control_url, m.service), proto="UDP")
            m.udp = True
    except (MapError, OSError):
        m.udp = False


def remove(gateway: str, local_ip: str, m: Mapping) -> None:
    try:
        if m.method == "pcp":
            pcp_map(gateway, local_ip, m.external_port, lifetime=0, nonce=m.nonce)
            if m.udp:
                pcp_map(gateway, local_ip, m.external_port, lifetime=0, nonce=m.udp_nonce, proto=17)
        elif m.method == "natpmp":
            natpmp_map(gateway, m.external_port, lifetime=0)
            if m.udp:
                natpmp_map(gateway, m.external_port, lifetime=0, udp=True)
        elif m.method == "upnp":
            for proto in ("TCP", "UDP") if m.udp else ("TCP",):
                _soap(m.control_url, m.service, "DeletePortMapping",
                      {"NewRemoteHost": "", "NewExternalPort": str(m.external_port), "NewProtocol": proto})
    except (MapError, OSError):
        pass

