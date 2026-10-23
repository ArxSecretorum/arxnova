"""Как плеер находит сервер, если у дома сменился адрес (ТЗ, ОД-3): общая торрент-сеть DHT.

Своего сервера адресов нет ни у разработчика, ни у владельца. Сервер Nova раз в 15 минут
объявляет себя в Mainline DHT (BEP 5) — распределённой «книге», которую держат миллионы
торрент-клиентов, — под ключом, выведенным из отпечатка своего сертификата. Узлы DHT
запоминают адрес, с которого пришло объявление, — это и есть текущий внешний адрес дома.
Плеер по тому же ключу спрашивает DHT и получает адреса; подделка ничего не даёт — плеер
всё равно верит только закреплённому ключу сервера.

Ключ в DHT — SHA-1 от «arxnova:» + отпечаток: без отпечатка (его знают только сопряжённые
устройства и гости) по ключу не понять, чей это сервер. Через DHT идёт только «я здесь».
"""

from __future__ import annotations

import hashlib
import os
import socket
import struct
import time

BOOTSTRAP = (("router.bittorrent.com", 6881), ("dht.transmissionbt.com", 6881),
             ("router.utorrent.com", 6881), ("dht.libtorrent.org", 25401))
K = 8
ANNOUNCE_EVERY_S = 15 * 60  # узлы DHT забывают объявления примерно через 30 минут


def info_hash(fingerprint: str) -> bytes:
    return hashlib.sha1(b"arxnova:" + bytes.fromhex(fingerprint)).digest()


# ---------- bencode ----------

def benc(x) -> bytes:
    if isinstance(x, int):
        return b"i%de" % x
    if isinstance(x, str):
        x = x.encode()
    if isinstance(x, bytes):
        return b"%d:%s" % (len(x), x)
    if isinstance(x, list):
        return b"l" + b"".join(benc(i) for i in x) + b"e"
    if isinstance(x, dict):
        return b"d" + b"".join(benc(k) + benc(x[k]) for k in sorted(x)) + b"e"
    raise TypeError(type(x))


def bdec(data: bytes):
    def go(i):
        c = data[i:i + 1]
        if c == b"i":
            e = data.index(b"e", i)
            return int(data[i + 1:e]), e + 1
        if c == b"l":
            i, out = i + 1, []
            while data[i:i + 1] != b"e":
                v, i = go(i)
                out.append(v)
            return out, i + 1
        if c == b"d":
            i, out = i + 1, {}
            while data[i:i + 1] != b"e":
                k, i = go(i)
                v, i = go(i)
                out[k] = v
            return out, i + 1
        colon = data.index(b":", i)
        n = int(data[i:colon])
        return data[colon + 1:colon + 1 + n], colon + 1 + n

    v, _ = go(0)
    return v


# ---------- KRPC ----------

def _nodes(compact: bytes) -> list[tuple[bytes, tuple[str, int]]]:
    out = []
    for i in range(0, len(compact) - 25, 26):
        nid = compact[i:i + 20]
        ip = socket.inet_ntoa(compact[i + 20:i + 24])
        port = struct.unpack("!H", compact[i + 24:i + 26])[0]
        if port:
            out.append((nid, (ip, port)))
    return out


def _peers(values) -> list[tuple[str, int]]:
    out = []
    for v in values or []:
        if isinstance(v, bytes) and len(v) == 6:
            out.append((socket.inet_ntoa(v[:4]), struct.unpack("!H", v[4:])[0]))
    return out


class Dht:
    def __init__(self, bind_ip: str = "0.0.0.0", timeout: float = 0.8):
        self.id = os.urandom(20)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((bind_ip, 0))
        self.sock.settimeout(timeout)

    def close(self):
        self.sock.close()

    def _query(self, addrs: list[tuple[str, int]], q: str, a: dict) -> dict[tuple[str, int], dict]:
        """Шлёт запрос сразу нескольким узлам и собирает ответы до таймаута."""
        sent = {}
        for addr in addrs:
            t = os.urandom(2)
            sent[t] = addr
            try:
                self.sock.sendto(benc({"t": t, "y": "q", "q": q, "a": {"id": self.id, **a}}), addr)
            except OSError:
                pass
        replies: dict[tuple[str, int], dict] = {}
        start = time.monotonic()
        deadline = start + self.sock.gettimeout()
        # половина узлов мертва: двое последних молчунов не держат круг дольше 250 мс
        while len(sent) > (2 if len(addrs) >= 4 and time.monotonic() - start > 0.25 else 0) and time.monotonic() < deadline:
            try:
                data, src = self.sock.recvfrom(4096)
            except (socket.timeout, OSError):
                break
            try:
                m = bdec(data)
            except (ValueError, IndexError):
                continue
            if not isinstance(m, dict) or m.get(b"y") != b"r":
                continue
            addr = sent.pop(m.get(b"t"), None)
            if addr is not None:
                replies[addr] = m.get(b"r") or {}
        return replies

    def _bootstrap(self) -> list[tuple[str, int]]:
        out = []
        for host, port in BOOTSTRAP:
            try:
                out.append((socket.gethostbyname(host), port))
            except OSError:
                pass
        return out

    def lookup(self, target: bytes, rounds: int = 8, early: bool = True, keep: int = K) -> tuple[list[tuple[str, int]], list[tuple[tuple[str, int], bytes]]]:
        """Итеративный get_peers: (найденные адреса, ближайшие узлы с токенами для объявления).
        early=False — без выхода на первых найденных адресах: для объявления нужны именно
        ближайшие к ключу узлы (там его и ищут), а не те, на которых поиск остановился."""
        def dist(nid: bytes) -> int:
            return int.from_bytes(nid, "big") ^ int.from_bytes(target, "big")

        known: dict[tuple[str, int], bytes] = {a: b"\xff" * 20 for a in self._bootstrap()}
        asked: set[tuple[str, int]] = set()
        tokens: dict[tuple[str, int], bytes] = {}
        responders: dict[tuple[str, int], bytes] = {}
        peers: set[tuple[str, int]] = set()
        for _ in range(rounds):
            todo = [a for a, _ in sorted(known.items(), key=lambda kv: dist(kv[1])) if a not in asked][:K]
            if not todo:
                break
            asked.update(todo)
            for addr, r in self._query(todo, "get_peers", {"info_hash": target}).items():
                nid = r.get(b"id")
                if isinstance(nid, bytes) and len(nid) == 20:
                    responders[addr] = nid
                if isinstance(r.get(b"token"), bytes):
                    tokens[addr] = r[b"token"]
                peers.update(_peers(r.get(b"values")))
                for nid2, a2 in _nodes(r.get(b"nodes") or b""):
                    known.setdefault(a2, nid2)
            # нашлись адреса и дальше узлы не приближаются — хватит
            if early and peers and len(asked) >= 3 * K:
                break
            # сошлись: K ближайших живых (или не спрошенных) узлов уже спрошены
            nearest = sorted((a for a in known if a not in asked or a in responders), key=lambda a: dist(known[a]))[:K]
            if not early and all(a in asked for a in nearest):
                break
        closest = sorted(((a, tokens[a]) for a in tokens if a in responders), key=lambda at: dist(responders[at[0]]))[:keep]
        return sorted(peers), closest

    def announce(self, target: bytes, port: int) -> int:
        """Объявить себя: узлы запомнят наш внешний адрес и этот порт. Возвращает, сколько узлов приняли."""
        _, closest = self.lookup(target, early=False)
        ok = 0
        for addr, token in closest:
            r = self._query([addr], "announce_peer", {"info_hash": target, "port": port, "implied_port": 0, "token": token})
            ok += 1 if r else 0
        return ok


def announce(fingerprint: str, port: int, bind_ip: str = "0.0.0.0") -> int:
    d = Dht(bind_ip)
    try:
        return d.announce(info_hash(fingerprint), port)
    finally:
        d.close()


def find(fingerprint: str, bind_ip: str = "0.0.0.0", tries: int = 2) -> list[tuple[str, int]]:
    """Адреса, под которыми объявлялся сервер. Порт из DHT ненадёжен (встречаются узлы,
    переворачивающие байты), поэтому плеер берёт отсюда адрес, а порт знает сам."""
    for _ in range(tries):
        d = Dht(bind_ip)
        try:
            peers, _ = d.lookup(info_hash(fingerprint), rounds=12)
        finally:
            d.close()
        if peers:
            return peers
    return []
