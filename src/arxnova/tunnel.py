"""Пробивка NAT и свой защищённый канал поверх UDP (ТЗ, ОД-3 п. 4) — без ретранслятора и без
чьего-либо сервера.

Когда до дома не дойти по TCP (NAT провайдера, два роутера подряд), сервер и плеер
открывают друг другу прямой UDP-путь:

1. Встреча — через торрент-сеть DHT. Сервер объявляет СВОЙ UDP-сокет под ключом
   SHA-1(«arxnova-udp:»+отпечаток) с implied_port=1: узлы запоминают адрес и порт, под
   которыми NAT выпустил этот сокет наружу. Плеер объявляет свой сокет под ключом
   SHA-1(«arxnova-knock:»+отпечаток) и стучит серверу. Сервер раз в KNOCK_POLL_S смотрит,
   кто стучится, и стучит в ответ — оба NAT открываются навстречу.
2. Рукопожатие: эфемерный ECDH P-256; сервер подписывает рукопожатие ключом своего
   сертификата, плеер сверяет отпечаток сертификата с закреплённым (как у TLS) — подменить
   сервер нельзя. Ключи AES-256-GCM по HKDF, свои на каждое направление.
3. Внутри — запросы HTTP, разрезанные на фрагменты, с подтверждениями, повтором потерь и
   окном. Сервер исполняет их тем же приложением FastAPI, «как из интернета» (порт для
   интернета, адрес — настоящий адрес плеера): токены, гости, лимиты — те же.
4. Сессию определяет conn_id, а не адрес: плеер сменил сеть — канал живёт дальше.

Формат пакета: b"AN" | версия 1 | тип | conn_id (8) | тело.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import socket
import struct
import time
from dataclasses import dataclass, field

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from arxnova import dht

MAGIC = b"AN\x01"
T_PUNCH, T_HELLO, T_WELCOME, T_DATA = 1, 2, 3, 4
F_FRAG, F_ACK, F_PING = 1, 2, 4
FRAG_DATA = 1100  # данных во фрагменте: с заголовками пакет < 1280 (минимальный MTU пути)
SESSION_IDLE_S = 120
KNOCK_POLL_S = 10
ANNOUNCE_S = 15 * 60
SIG_CONTEXT = b"arxnova-tunnel-v1"


def udp_key(fp: str) -> bytes:
    return hashlib.sha1(b"arxnova-udp:" + bytes.fromhex(fp)).digest()


def knock_key(fp: str) -> bytes:
    return hashlib.sha1(b"arxnova-knock:" + bytes.fromhex(fp)).digest()


def derive(shared: bytes, nonce_c: bytes, nonce_s: bytes) -> tuple[bytes, bytes]:
    """(ключ плеер→сервер, ключ сервер→плеер)."""
    okm = HKDF(algorithm=hashes.SHA256(), length=64, salt=nonce_c + nonce_s,
               info=b"arxnova-tunnel-v1 keys").derive(shared)
    return okm[:32], okm[32:]


def transcript(conn_id: bytes, c_pub: bytes, nonce_c: bytes, s_pub: bytes, nonce_s: bytes) -> bytes:
    return SIG_CONTEXT + conn_id + c_pub + nonce_c + s_pub + nonce_s


def frame(kind: int, payload: bytes) -> bytes:
    return struct.pack("!BH", kind, len(payload)) + payload


def frames(data: bytes):
    pos = 0
    while pos + 3 <= len(data):
        kind, ln = struct.unpack("!BH", data[pos:pos + 3])
        yield kind, data[pos + 3:pos + 3 + ln]
        pos += 3 + ln


# ---------- надёжная доставка сообщений поверх датаграмм ----------

@dataclass
class Outgoing:
    frags: list[bytes]
    acked: list[bool]
    sent_at: list[float]
    done: asyncio.Future | None = None
    retx: list[bool] = field(default_factory=list)
    hi: int = -1  # старший подтверждённый фрагмент: всё, что раньше и давно в пути, — потеряно


@dataclass
class Incoming:
    count: int
    parts: dict[int, bytes] = field(default_factory=dict)
    delivered: bool = False


class Session:
    """Одна сессия плеера: шифрование, окно, повторы, сборка сообщений."""

    def __init__(self, conn_id: bytes, k_rx: bytes, k_tx: bytes, addr, send_raw, on_message, tx_dir: int):
        self.conn_id = conn_id
        self.rx, self.tx = AESGCM(k_rx), AESGCM(k_tx)
        self.tx_dir, self.rx_dir = tx_dir, 1 - tx_dir
        self.addr = addr
        self.send_raw = send_raw
        self.on_message = on_message
        self.seq = 0
        self.seen: set[int] = set()  # защита от повтора пакетов (окно последних номеров)
        self.out: dict[int, Outgoing] = {}
        self.inc: dict[int, Incoming] = {}
        self.cwnd = 64.0
        self.ssthresh = 512.0
        self.srtt = 0.1
        self.last_cut = 0.0
        self.backoff = 1.0
        self.retx_count = 0  # замер: сколько фрагментов повторено
        # темп (pacing): фрагменты уходят ровным потоком, а не залпом — точка доступа Wi-Fi
        # не копит очередь и не выбрасывает хвост пачки
        self.paced: list[tuple[Outgoing, int]] = []
        self.pace_credit = 0.0
        self.pace_at = time.monotonic()
        self.last_rx = time.monotonic()
        self.pending_acks: set[int] = set()
        self.hello = b""
        self.welcome = b""

    # шифрование пакета
    def seal(self, payload: bytes) -> bytes:
        self.seq += 1
        header = MAGIC + bytes([T_DATA]) + self.conn_id
        nonce = struct.pack("!IQ", self.tx_dir, self.seq)
        return header + struct.pack("!Q", self.seq) + self.tx.encrypt(nonce, payload, header)

    def open(self, packet: bytes) -> bytes | None:
        header = packet[:12]
        seq = struct.unpack("!Q", packet[12:20])[0]
        if seq in self.seen:
            return None
        try:
            plain = self.rx.decrypt(struct.pack("!IQ", self.rx_dir, seq), packet[20:], header)
        except Exception:
            return None
        self.seen.add(seq)
        if len(self.seen) > 4096:
            self.seen = {s for s in self.seen if s > seq - 2048}
        self.last_rx = time.monotonic()
        return plain

    def inflight(self) -> int:
        """Сколько фрагментов в пути: отправленные и не подтверждённые, кроме явно потерянных
        (за ними уже подтверждены более поздние). Запоздавшее подтверждение (Wi-Fi в экономии)
        потерей не считается — иначе окно не сдерживает отправку и канал тонет в повторах."""
        now = time.monotonic()
        gap = 1.25 * self.srtt + 0.01
        return sum(1 for o in self.out.values() for i, (a, t) in enumerate(zip(o.acked, o.sent_at))
                   if not a and (t < 0 or (t > 0 and not (i < o.hi and now - t > gap))))

    def _loss(self, now: float) -> None:
        # окно сужаем не чаще раза за круг и мягко (как CUBIC), а не вдвое на каждой потере
        # Wi-Fi и мобильная сеть теряют пакеты и без перегрузки: на редких потерях окно не душим
        # (пол — 64 пакета ≈ 70 КБ в пути), сжатие мягкое
        if now - self.last_cut > 2 * self.srtt:
            self.ssthresh = max(64.0, self.cwnd * 0.85)
            self.cwnd = self.ssthresh
            self.last_cut = now

    def send_message(self, stream: int, data: bytes) -> asyncio.Future:
        chunks = [data[i:i + FRAG_DATA] for i in range(0, len(data), FRAG_DATA)] or [b""]
        o = Outgoing([struct.pack("!III", stream, i, len(chunks)) + c for i, c in enumerate(chunks)],
                     [False] * len(chunks), [0.0] * len(chunks), asyncio.get_running_loop().create_future(),
                     [False] * len(chunks))
        self.out[stream] = o
        self.pump()
        return o.done

    def pump(self) -> None:
        """Отправить, что позволяет окно: сначала потерянные (быстрый повтор — позже
        отправленные уже дошли; или вышел таймаут), потом новые."""
        now = time.monotonic()
        rto = max(0.25, 3 * self.srtt) * self.backoff
        gap = 1.25 * self.srtt + 0.01
        budget = int(self.cwnd) - self.inflight()
        for o in self.out.values():
            for i, f in enumerate(o.frags):
                if o.acked[i]:
                    continue
                t = o.sent_at[i]
                sack_lost = t > 0 and i < o.hi and now - t > gap
                timed_out = t > 0 and now - t > rto
                lost = sack_lost or timed_out
                if timed_out and not sack_lost:
                    self.backoff = min(8.0, self.backoff * 2)  # подряд таймауты — ждём дольше
                if t < 0:
                    continue  # уже в очереди темпа
                if lost or (t == 0 and budget > 0):
                    self.paced.append((o, i))
                    o.sent_at[i] = -1.0  # в очереди; время отправки поставит pace()
                    if lost:
                        self.retx_count += 1
                        o.retx[i] = True
                        self._loss(now)
                    else:
                        budget -= 1

    def pace(self) -> None:
        """Отправить из очереди столько, сколько позволяет темп: окно за время круга (+25%)."""
        now = time.monotonic()
        rate = 1.25 * self.cwnd / max(0.01, self.srtt)  # пакетов в секунду
        self.pace_credit = min(self.pace_credit + (now - self.pace_at) * rate, 16.0)
        self.pace_at = now
        while self.paced and self.pace_credit >= 1.0:
            o, i = self.paced.pop(0)
            if o.acked[i]:
                continue
            self.send_raw(self.seal(frame(F_FRAG, o.frags[i])), self.addr)
            o.sent_at[i] = now
            self.pace_credit -= 1.0

    def on_packet(self, plain: bytes) -> None:
        for kind, p in frames(plain):
            if kind == F_FRAG and len(p) >= 12:
                stream, idx, count = struct.unpack("!III", p[:12])
                inc = self.inc.get(stream)
                if inc is None:
                    inc = self.inc[stream] = Incoming(count)
                if idx < count:
                    inc.parts[idx] = p[12:]
                self.pending_acks.add(stream)
                if not inc.delivered and len(inc.parts) == inc.count:
                    inc.delivered = True
                    self.on_message(stream, b"".join(inc.parts[i] for i in range(inc.count)))
            elif kind == F_ACK and len(p) >= 8:
                stream, count = struct.unpack("!II", p[:8])
                o = self.out.get(stream)
                if o is None:
                    continue
                bits = p[8:]
                now = time.monotonic()
                for i in range(min(count, len(o.frags))):
                    if not o.acked[i] and bits[i // 8] >> (i % 8) & 1:
                        o.acked[i] = True
                        o.hi = max(o.hi, i)
                        self.backoff = 1.0
                        if o.sent_at[i] > 0 and not o.retx[i]:  # время круга — только по непотерянным
                            self.srtt = 0.875 * self.srtt + 0.125 * (now - o.sent_at[i])
                        # разгон: до порога — +1 за подтверждение (окно удваивается за круг), дальше — плавно
                        self.cwnd = min(2048.0, self.cwnd + (1.0 if self.cwnd < self.ssthresh else 1.0 / self.cwnd))
                if all(o.acked):
                    self.out.pop(stream, None)
                    if o.done and not o.done.done():
                        o.done.set_result(True)
        self.flush_acks()
        self.pump()

    def flush_acks(self) -> None:
        payload = b""
        for stream in list(self.pending_acks):
            inc = self.inc.get(stream)
            if inc is None:
                continue
            bits = bytearray((inc.count + 7) // 8)
            for i in inc.parts:
                bits[i // 8] |= 1 << (i % 8)
            f = frame(F_ACK, struct.pack("!II", stream, inc.count) + bytes(bits))
            if len(payload) + len(f) > 1150:
                self.send_raw(self.seal(payload), self.addr)
                payload = b""
            payload += f
        self.pending_acks.clear()
        if payload:
            self.send_raw(self.seal(payload), self.addr)


# ---------- сервер ----------

def encode_response(status: int, headers: dict, body: bytes) -> bytes:
    h = json.dumps(headers).encode()
    return struct.pack("!HI", status, len(h)) + h + body


def decode_request(data: bytes) -> tuple[str, str, dict, bytes]:
    hlen = struct.unpack("!I", data[:4])[0]
    meta = json.loads(data[4:4 + hlen])
    return meta["m"], meta["p"], meta.get("h", {}), data[4 + hlen:]


class TunnelServer(asyncio.DatagramProtocol):
    def __init__(self, app, port: int, cert_der: bytes, key, fingerprint: str):
        self.app = app
        self.port = port
        self.cert_der = cert_der
        self.key = key
        self.fp = fingerprint
        self.transport: asyncio.DatagramTransport | None = None
        self.sessions: dict[bytes, Session] = {}
        self.dht_wait: dict[bytes, asyncio.Future] = {}
        self.knocked: dict[tuple, float] = {}
        self.hello_rate: dict[str, list[float]] = {}
        self.dht_nodes = 0
        self.peers_seen = 0

    # --- сеть ---
    def connection_made(self, transport):
        self.transport = transport

    send_errors = 0

    def error_received(self, exc):
        # macOS: очередь интерфейса полна (ENOBUFS) — asyncio пакет выбросил
        self.send_errors += 1

    def send_raw(self, data: bytes, addr) -> None:
        if self.transport is not None:
            self.transport.sendto(data, addr)

    def datagram_received(self, data: bytes, addr) -> None:
        if data.startswith(b"d1:"):  # ответ узла DHT
            try:
                m = dht.bdec(data)
            except (ValueError, IndexError):
                return
            fut = self.dht_wait.pop(m.get(b"t", b""), None) if isinstance(m, dict) else None
            if fut and not fut.done():
                fut.set_result(m.get(b"r") or {})
            return
        if not data.startswith(MAGIC) or len(data) < 12:
            return
        kind, conn_id = data[3], data[4:12]
        if kind == T_PUNCH:
            self.send_raw(MAGIC + bytes([T_PUNCH]) + conn_id + os.urandom(8), addr)
        elif kind == T_HELLO:
            self._hello(data, addr, conn_id)
        elif kind == T_DATA:
            s = self.sessions.get(conn_id)
            if s is None or len(data) < 20:
                return
            plain = s.open(data)
            if plain is None:
                return
            s.addr = addr  # плеер сменил сеть — продолжаем по новому адресу
            s.on_packet(plain)

    def _hello(self, data: bytes, addr, conn_id: bytes) -> None:
        body = data[12:]
        if len(body) < 65 + 16 + 20 or body[81:101] != udp_key(self.fp):
            return
        now = time.monotonic()
        recent = [t for t in self.hello_rate.get(addr[0], []) if now - t < 10] + [now]
        self.hello_rate[addr[0]] = recent
        if len(recent) > 20:  # подпись дорогая — не больше 20 рукопожатий за 10 с с адреса
            return
        c_pub, nonce_c = body[:65], body[65:81]
        old = self.sessions.get(conn_id)
        if old is not None and old.hello == c_pub + nonce_c:
            # повтор того же приветствия (ответ потерялся): тот же ответ, те же ключи
            old.addr = addr
            self.send_raw(old.welcome, addr)
            return
        s_priv = ec.generate_private_key(ec.SECP256R1())
        s_pub = s_priv.public_key().public_bytes(serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint)
        nonce_s = os.urandom(16)
        try:
            peer = ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256R1(), c_pub)
        except ValueError:
            return
        shared = s_priv.exchange(ec.ECDH(), peer)
        k_c2s, k_s2c = derive(shared, nonce_c, nonce_s)
        sig = self.key.sign(transcript(conn_id, c_pub, nonce_c, s_pub, nonce_s), ec.ECDSA(hashes.SHA256()))
        welcome = (MAGIC + bytes([T_WELCOME]) + conn_id + s_pub + nonce_s
                   + struct.pack("!H", len(self.cert_der)) + self.cert_der + struct.pack("!H", len(sig)) + sig)
        s = Session(conn_id, k_c2s, k_s2c, addr, self.send_raw,
                    lambda st, msg, cid=conn_id: self._on_request(cid, st, msg), tx_dir=1)
        s.hello, s.welcome = c_pub + nonce_c, welcome
        self.sessions[conn_id] = s
        # сертификат ECDSA P-256 + подпись — около 600 байт: в одну датаграмму помещается
        self.send_raw(welcome, addr)

    def _on_request(self, conn_id: bytes, stream: int, msg: bytes) -> None:
        asyncio.get_running_loop().create_task(self._serve(conn_id, stream, msg))

    async def _serve(self, conn_id: bytes, stream: int, msg: bytes) -> None:
        import httpx

        s = self.sessions.get(conn_id)
        if s is None:
            return
        try:
            method, path, headers, body = decode_request(msg)
            # «как из интернета»: порт для интернета, адрес — настоящий адрес плеера
            transport = httpx.ASGITransport(app=self.app, client=(s.addr[0], s.addr[1]))
            async with httpx.AsyncClient(transport=transport, base_url=f"https://tunnel:{self.port}") as c:
                r = await c.request(method, path, headers=headers, content=body)
            out = encode_response(r.status_code, {k: v for k, v in r.headers.items()
                                                  if k.lower() in ("content-type", "content-range", "content-length",
                                                                   "etag", "cache-control")}, r.content)
        except Exception as e:  # битый запрос — ответ 400, сессия живёт
            out = encode_response(400, {"content-type": "application/json"}, json.dumps({"detail": str(e)}).encode())
        t0, r0 = time.monotonic(), s.retx_count
        await s.send_message(stream, out)
        if os.environ.get("ARXNOVA_TUNNEL_DEBUG") and len(out) > 100_000:
            print(f"tunnel: {len(out) // 1024} KB за {int((time.monotonic() - t0) * 1000)} мс, повторов {s.retx_count - r0}, "
                  f"окно {int(s.cwnd)}, круг {int(s.srtt * 1000)} мс, отказов отправки {self.send_errors}", flush=True)

    # --- торрент-сеть: объявить себя, услышать стучащихся ---
    async def _query(self, addr, q: str, a: dict, my_id: bytes, timeout: float = 0.8) -> dict | None:
        t = os.urandom(2)
        fut = asyncio.get_running_loop().create_future()
        self.dht_wait[t] = fut
        self.send_raw(dht.benc({"t": t, "y": "q", "q": q, "a": {"id": my_id, **a}}), addr)
        try:
            return await asyncio.wait_for(fut, timeout)
        except asyncio.TimeoutError:
            self.dht_wait.pop(t, None)
            return None

    async def _lookup(self, target: bytes, my_id: bytes, rounds: int = 10):
        loop = asyncio.get_running_loop()
        known: dict = {}
        for h, p in dht.BOOTSTRAP:
            try:
                info = await loop.getaddrinfo(h, p, family=socket.AF_INET, type=socket.SOCK_DGRAM)
                known[info[0][4][:2]] = b"\xff" * 20
            except OSError:
                pass
        asked, tokens, ids, peers = set(), {}, {}, set()

        def dist(n: bytes) -> int:
            return int.from_bytes(n, "big") ^ int.from_bytes(target, "big")

        for _ in range(rounds):
            todo = [a for a, _ in sorted(known.items(), key=lambda kv: dist(kv[1])) if a not in asked][:dht.K]
            if not todo:
                break
            asked.update(todo)
            replies = await asyncio.gather(*(self._query(a, "get_peers", {"info_hash": target}, my_id) for a in todo))
            for a, r in zip(todo, replies):
                if not r:
                    continue
                nid = r.get(b"id")
                if isinstance(nid, bytes) and len(nid) == 20:
                    ids[a] = nid
                if isinstance(r.get(b"token"), bytes):
                    tokens[a] = r[b"token"]
                peers.update(dht._peers(r.get(b"values")))
                for n2, a2 in dht._nodes(r.get(b"nodes") or b""):
                    known.setdefault(a2, n2)
            if peers and len(asked) >= 3 * dht.K:
                break
        closest = sorted(((a, tokens[a]) for a in tokens if a in ids), key=lambda at: dist(ids[at[0]]))[:dht.K]
        return peers, closest

    async def announce_loop(self) -> None:
        my_id = os.urandom(20)
        while True:
            try:
                _, closest = await self._lookup(udp_key(self.fp), my_id)
                ok = 0
                for a, tok in closest:
                    # implied_port=1: узел запомнит порт, под которым NAT выпустил ЭТОТ сокет
                    r = await self._query(a, "announce_peer", {"info_hash": udp_key(self.fp), "port": self.port,
                                                               "implied_port": 1, "token": tok}, my_id)
                    ok += 1 if r is not None else 0
                self.dht_nodes = ok
            except Exception:
                pass
            await asyncio.sleep(ANNOUNCE_S if self.dht_nodes else 60)

    async def knock_loop(self) -> None:
        """Кто стучится — тому стучим в ответ: открываем свой NAT навстречу плееру."""
        my_id = os.urandom(20)
        while True:
            try:
                peers, _ = await self._lookup(knock_key(self.fp), my_id, rounds=8)
                now = time.monotonic()
                self.peers_seen = len(peers)
                for addr in peers:
                    if now - self.knocked.get(addr, 0) < 30:
                        continue
                    self.knocked[addr] = now
                    for _ in range(5):
                        self.send_raw(MAGIC + bytes([T_PUNCH]) + b"\0" * 8 + os.urandom(8), addr)
                        await asyncio.sleep(0.2)
            except Exception:
                pass
            # старые сессии — прочь
            now = time.monotonic()
            for cid in [c for c, s in self.sessions.items() if now - s.last_rx > SESSION_IDLE_S]:
                self.sessions.pop(cid, None)
            for s in list(self.sessions.values()):
                s.pump()
            await asyncio.sleep(KNOCK_POLL_S)

    async def retransmit_loop(self) -> None:
        last_pump = 0.0
        while True:
            now = time.monotonic()
            for s in list(self.sessions.values()):
                if now - last_pump > 0.02:
                    s.pump()
                s.pace()
            if now - last_pump > 0.02:
                last_pump = now
            await asyncio.sleep(0.002)


async def start(app, port: int, cert_path, key_path, fingerprint: str) -> TunnelServer:
    """Поднять канал на UDP-порту (тот же номер, что у TCP-порта для интернета)."""
    from cryptography import x509

    cert = x509.load_pem_x509_certificate(open(cert_path, "rb").read())
    key = serialization.load_pem_private_key(open(key_path, "rb").read(), password=None)
    loop = asyncio.get_running_loop()
    server = TunnelServer(app, port, cert.public_bytes(serialization.Encoding.DER), key, fingerprint)
    await loop.create_datagram_endpoint(lambda: server, local_addr=("0.0.0.0", port))
    if port:  # port=0 — проверки: без торрент-сети
        loop.create_task(server.announce_loop())
        loop.create_task(server.knock_loop())
    loop.create_task(server.retransmit_loop())
    return server


# ---------- клиент (для проверок и для `arxnova tunnel-test`; в плеере — то же на Kotlin) ----------

def encode_request(method: str, path: str, headers: dict, body: bytes = b"") -> bytes:
    meta = json.dumps({"m": method, "p": path, "h": headers}).encode()
    return struct.pack("!I", len(meta)) + meta + body


class TunnelClient(asyncio.DatagramProtocol):
    def __init__(self, fingerprint: str, drop: float = 0.0):
        self.fp = fingerprint
        self.drop = drop  # доля «потерянных» пакетов — только для проверок
        self.conn_id = os.urandom(8)
        self.transport = None
        self.session: Session | None = None
        self.welcome: asyncio.Future | None = None
        self.replies: dict[int, asyncio.Future] = {}
        self.next_stream = 1

    def connection_made(self, transport):
        self.transport = transport

    def send_raw(self, data: bytes, addr) -> None:
        if self.drop and random.random() < self.drop:
            return
        self.transport.sendto(data, addr)

    def datagram_received(self, data: bytes, addr) -> None:
        if self.drop and random.random() < self.drop:
            return
        if not data.startswith(MAGIC) or len(data) < 12 or data[4:12] != self.conn_id:
            return
        if data[3] == T_WELCOME and self.welcome and not self.welcome.done():
            self.welcome.set_result(data)
        elif data[3] == T_DATA and self.session:
            plain = self.session.open(data)
            if plain is not None:
                self.session.on_packet(plain)

    def _on_message(self, stream: int, msg: bytes) -> None:
        fut = self.replies.pop(stream, None)
        if fut and not fut.done():
            fut.set_result(msg)

    async def connect(self, addr, timeout: float = 10.0) -> None:
        from cryptography import x509

        c_priv = ec.generate_private_key(ec.SECP256R1())
        c_pub = c_priv.public_key().public_bytes(serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint)
        nonce_c = os.urandom(16)
        hello = MAGIC + bytes([T_HELLO]) + self.conn_id + c_pub + nonce_c + udp_key(self.fp)
        self.welcome = asyncio.get_running_loop().create_future()
        deadline = time.monotonic() + timeout
        while not self.welcome.done():
            if time.monotonic() > deadline:
                raise TimeoutError("сервер не ответил")
            self.send_raw(MAGIC + bytes([T_PUNCH]) + self.conn_id + os.urandom(8), addr)
            self.send_raw(hello, addr)
            await asyncio.wait([self.welcome], timeout=0.5)
        w = self.welcome.result()[12:]
        s_pub, nonce_s = w[:65], w[65:81]
        clen = struct.unpack("!H", w[81:83])[0]
        cert_der = w[83:83 + clen]
        slen = struct.unpack("!H", w[83 + clen:85 + clen])[0]
        sig = w[85 + clen:85 + clen + slen]
        if hashlib.sha256(cert_der).hexdigest() != self.fp.lower():
            raise PermissionError("ключ сервера не совпал")
        cert = x509.load_der_x509_certificate(cert_der)
        cert.public_key().verify(sig, transcript(self.conn_id, c_pub, nonce_c, s_pub, nonce_s), ec.ECDSA(hashes.SHA256()))
        shared = c_priv.exchange(ec.ECDH(), ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256R1(), s_pub))
        k_c2s, k_s2c = derive(shared, nonce_c, nonce_s)
        self.session = Session(self.conn_id, k_s2c, k_c2s, addr, self.send_raw, self._on_message, tx_dir=0)
        asyncio.get_running_loop().create_task(self._pump())

    async def _pump(self):
        n = 0
        while self.transport is not None and not self.transport.is_closing():
            if self.session:
                if n % 10 == 0:
                    self.session.pump()
                self.session.pace()
            n += 1
            await asyncio.sleep(0.002)

    async def request(self, method: str, path: str, headers: dict | None = None, body: bytes = b"",
                      timeout: float = 30.0) -> tuple[int, dict, bytes]:
        stream = self.next_stream
        self.next_stream += 2
        fut = asyncio.get_running_loop().create_future()
        self.replies[stream] = fut
        await self.session.send_message(stream, encode_request(method, path, headers or {}, body))
        msg = await asyncio.wait_for(fut, timeout)
        status, hlen = struct.unpack("!HI", msg[:6])
        return status, json.loads(msg[6:6 + hlen]), msg[6 + hlen:]
