"""Общий доступ к медиатеке (ТЗ, §15.1, ОД-1…ОД-10): приглашения, вход гостя, ограничения, отзыв.

Правила, которые держат правовые опоры (ОД-10), зашиты здесь, а не в настройках:
- приглашение поимённое и одноразовое, код живёт 24 ч;
- гостей не больше MAX_GUESTS = 5 при любой конфигурации (настройкой можно только меньше);
- одновременно слушают не больше MAX_STREAMS = 2 гостей;
- суточный объём на гостя — параметр.
"""

from __future__ import annotations

import hashlib
import secrets
import sqlite3
import threading
import time

from arxnova.auth import tidy_name
import urllib.parse

MAX_GUESTS = 5  # жёсткий потолок (ОД-7, ОД-10 п. 2)
MAX_STREAMS = 2
INVITE_TTL_S = 24 * 3600
STREAM_WINDOW_S = 30  # гость «слушает», если брал звук за последние 30 с
# Без похожих символов (O/0, I/1, L): код диктуют голосом и переписывают руками.
ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"
CODE_LEN = 8

# Защита от перебора (ОД-9): с одного адреса и со всех вместе.
IP_FAILS, IP_WINDOW_S = 5, 600
ALL_FAILS, ALL_WINDOW_S = 30, 3600
BLOCK_S = 3600


class GuestError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


def _h(code: str) -> bytes:
    return hashlib.sha256(normalize(code).encode()).digest()


def normalize(code: str) -> str:
    """Регистр, пробелы и дефисы не важны: «abcd-efgh» = «ABCDEFGH»."""
    return "".join(c for c in code.upper() if c.isalnum())


def limit(configured: int | None) -> int:
    return max(0, min(MAX_GUESTS, configured if configured is not None else MAX_GUESTS))


def occupied(conn: sqlite3.Connection) -> int:
    """Места: действующие гости + неиспользованные живые приглашения."""
    now = int(time.time())
    g = conn.execute("SELECT count(*) FROM devices WHERE role = 'guest' AND revoked = 0").fetchone()[0]
    i = conn.execute(
        "SELECT count(*) FROM invites WHERE used_at IS NULL AND revoked = 0 AND expires_at >= ?", (now,)
    ).fetchone()[0]
    return g + i


def create_invite(conn: sqlite3.Connection, guest_name: str, created_by: int | None, max_guests: int | None = None) -> dict:
    name = guest_name.strip()[:60]
    if not name:
        raise GuestError(400, "укажите, кому приглашение")
    cap = limit(max_guests)
    if occupied(conn) >= cap:
        raise GuestError(409, f"мест для гостей нет: не больше {cap} (семья и близкие)")
    code = "".join(secrets.choice(ALPHABET) for _ in range(CODE_LEN))
    now = int(time.time())
    with conn:
        cur = conn.execute(
            "INSERT INTO invites (code_hash, guest_name, created_by, created_at, expires_at) VALUES (?,?,?,?,?)",
            (_h(code), name, created_by, now, now + INVITE_TTL_S),
        )
    return {"id": cur.lastrowid, "code": code, "guest_name": name, "expires_at": now + INVITE_TTL_S}


def link(code: str, server: str, inviter: str, net: dict | None, lan_url: str | None) -> str:
    """Ссылка-приглашение: код + как достучаться до сервера + отпечаток его ключа. Без службы
    встречи адрес нужен в самой ссылке (ОД-3); отпечаток — чтобы гость сразу верил только
    этому серверу."""
    q = {"c": code, "n": server, "by": inviter}
    if net and net.get("tls"):
        q["fp"] = net["tls"]["fingerprint"]
        q["p"] = str(net["tls"]["port"])  # порт с TLS: и дома у друга гость идёт только с шифрованием
        if net.get("external"):
            q["h"] = net["external"]
    if lan_url:
        q["l"] = lan_url
    return "arxnova://join?" + urllib.parse.urlencode(q, quote_via=urllib.parse.quote)


class Throttle:
    """Ограничение попыток ввода кода (ОД-9): в памяти процесса."""

    def __init__(self):
        self.lock = threading.Lock()
        self.fails: dict[str, list[float]] = {}
        self.all: list[float] = []
        self.blocked: dict[str, float] = {}
        self.all_blocked_until = 0.0

    def check(self, ip: str) -> None:
        now = time.time()
        with self.lock:
            if now < self.all_blocked_until:
                raise GuestError(429, "вход по приглашениям временно закрыт: слишком много ошибок")
            if now < self.blocked.get(ip, 0):
                raise GuestError(429, "слишком много попыток с этого адреса, попробуйте через час")

    def fail(self, ip: str) -> None:
        now = time.time()
        with self.lock:
            f = [t for t in self.fails.get(ip, []) if now - t < IP_WINDOW_S] + [now]
            self.fails[ip] = f
            if len(f) >= IP_FAILS:
                self.blocked[ip] = now + BLOCK_S
            self.all = [t for t in self.all if now - t < ALL_WINDOW_S] + [now]
            if len(self.all) >= ALL_FAILS:
                self.all_blocked_until = now + BLOCK_S


def join(conn: sqlite3.Connection, code: str, device_name: str, ip: str, throttle: Throttle) -> tuple[int, str, sqlite3.Row]:
    """Код → токен гостя. Код одноразовый; после ввода место занято гостем."""
    throttle.check(ip)
    now = int(time.time())
    row = conn.execute(
        "SELECT * FROM invites WHERE code_hash = ? AND used_at IS NULL AND revoked = 0 AND expires_at >= ?",
        (_h(code), now),
    ).fetchone()
    if not row:
        throttle.fail(ip)
        raise GuestError(403, "приглашение неверное, уже использовано или истекло")
    token = secrets.token_urlsafe(32)
    with conn:
        cur = conn.execute(
            "INSERT INTO devices (name, role, token_hash, guest_name) VALUES (?, 'guest', ?, ?)",
            ((device_name or "устройство")[:80], hashlib.sha256(token.encode()).digest(), row["guest_name"]),
        )
        conn.execute("UPDATE invites SET used_at = ?, device_id = ? WHERE id = ?", (now, cur.lastrowid, row["id"]))
    return cur.lastrowid, token, row


def list_guests(conn: sqlite3.Connection) -> dict:
    now = int(time.time())
    guests = [
        {"id": r["id"], "name": r["guest_name"] or r["name"], "device": tidy_name(r["name"]), "last_seen": r["last_seen"],
         "since": r["created_at"],
         "bytes": conn.execute("SELECT coalesce(sum(bytes),0) FROM guest_usage WHERE device_id = ?", (r["id"],)).fetchone()[0]}
        for r in conn.execute("SELECT * FROM devices WHERE role = 'guest' AND revoked = 0 ORDER BY created_at")
    ]
    invites = [
        {"id": r["id"], "name": r["guest_name"], "expires_at": r["expires_at"]}
        for r in conn.execute(
            "SELECT * FROM invites WHERE used_at IS NULL AND revoked = 0 AND expires_at >= ? ORDER BY created_at", (now,))
    ]
    return {"guests": guests, "invites": invites}


def revoke_guest(conn: sqlite3.Connection, device_id: int) -> bool:
    """Отзыв (ОД-6): токен перестаёт действовать сразу; «Избранное», история и профиль гостя
    удаляются (они живут только в его событиях)."""
    r = conn.execute("SELECT id FROM devices WHERE id = ? AND role = 'guest' AND revoked = 0", (device_id,)).fetchone()
    if not r:
        return False
    with conn:
        conn.execute("UPDATE devices SET revoked = 1 WHERE id = ?", (device_id,))
        conn.execute("DELETE FROM events WHERE device_id = ?", (device_id,))
        conn.execute("DELETE FROM nova_requests WHERE device_id = ?", (device_id,))
        conn.execute("DELETE FROM device_catalog WHERE device_id = ?", (device_id,))
        conn.execute("DELETE FROM guest_usage WHERE device_id = ?", (device_id,))
    return True


def revoke_invite(conn: sqlite3.Connection, invite_id: int) -> bool:
    with conn:
        return conn.execute("UPDATE invites SET revoked = 1 WHERE id = ? AND used_at IS NULL", (invite_id,)).rowcount > 0


class Streams:
    """Кто из гостей сейчас слушает (ОД-7): не больше MAX_STREAMS одновременно. Гости уступают
    владельцу: пока владелец слушает сам, канал дома делится — гостю место только одно."""

    def __init__(self):
        self.lock = threading.Lock()
        self.last: dict[int, float] = {}
        self.owner_at = 0.0

    def owner_listening(self) -> None:
        self.owner_at = time.time()

    def admit(self, device_id: int) -> None:
        now = time.time()
        with self.lock:
            self.last = {d: t for d, t in self.last.items() if now - t < STREAM_WINDOW_S}
            cap = 1 if now - self.owner_at < STREAM_WINDOW_S else MAX_STREAMS
            if device_id not in self.last and len(self.last) >= cap:
                raise GuestError(429, "сейчас слушают другие — попробуйте чуть позже")
            self.last[device_id] = now


def charge(conn: sqlite3.Connection, device_id: int, nbytes: int, daily_mb: int) -> None:
    """Учёт объёма гостя за сутки; превышение — отказ до завтра (ОД-7)."""
    day = time.strftime("%Y-%m-%d")
    used = conn.execute("SELECT bytes FROM guest_usage WHERE device_id = ? AND day = ?", (device_id, day)).fetchone()
    used = used[0] if used else 0
    if daily_mb > 0 and used + nbytes > daily_mb * 1024 * 1024:
        raise GuestError(429, "суточный объём для гостя исчерпан — до завтра")
    with conn:
        conn.execute(
            "INSERT INTO guest_usage (device_id, day, bytes) VALUES (?,?,?) "
            "ON CONFLICT(device_id, day) DO UPDATE SET bytes = bytes + excluded.bytes",
            (device_id, day, nbytes),
        )
