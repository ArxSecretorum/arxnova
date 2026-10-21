"""Внешний адрес глазами интернета (STUN, RFC 5389) — для диагноза сети (ОД-3а).

Запрос уходит с домашнего адреса компьютера (сокет привязан к нему): на машине с VPN иначе
ответ показал бы адрес VPN, а не дома. Через STUN не идёт ничего, кроме этого запроса.
"""

from __future__ import annotations

import os
import socket
import struct

SERVERS = (("stun.l.google.com", 19302), ("stun.cloudflare.com", 3478))
MAGIC = 0x2112A442


def external_ip(local_ip: str | None = None, timeout: float = 1.5) -> str | None:
    for host, port in SERVERS:
        try:
            ip = _query(host, port, local_ip, timeout)
        except OSError:
            continue
        if ip:
            return ip
    return None


def _query(host: str, port: int, local_ip: str | None, timeout: float) -> str | None:
    addr = socket.getaddrinfo(host, port, socket.AF_INET, socket.SOCK_DGRAM)[0][4]
    tid = os.urandom(12)
    req = struct.pack("!HHI", 0x0001, 0, MAGIC) + tid
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.settimeout(timeout)
        if local_ip:
            s.bind((local_ip, 0))
        for _ in range(2):
            s.sendto(req, addr)
            try:
                data, _ = s.recvfrom(1024)
            except socket.timeout:
                continue
            if len(data) < 20 or data[8:20] != tid:
                continue
            return _parse(data)
    return None


def _parse(data: bytes) -> str | None:
    pos, end = 20, 20 + struct.unpack("!H", data[2:4])[0]
    mapped = None
    while pos + 4 <= min(end, len(data)):
        t, ln = struct.unpack("!HH", data[pos:pos + 4])
        v = data[pos + 4:pos + 4 + ln]
        if t == 0x0020 and len(v) >= 8 and v[1] == 1:  # XOR-MAPPED-ADDRESS, IPv4
            return socket.inet_ntoa(struct.pack("!I", struct.unpack("!I", v[4:8])[0] ^ MAGIC))
        if t == 0x0001 and len(v) >= 8 and v[1] == 1:  # MAPPED-ADDRESS
            mapped = socket.inet_ntoa(v[4:8])
        pos += 4 + ln + (-ln % 4)
    return mapped
