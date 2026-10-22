"""Сопряжение (API-4): 6-значный код на 5 минут → токен устройства. Храним только хеши.

Сопряжение работает и через интернет (решение владельца 22.10.2026: медиатека — для доступа
из любой точки, локальная сеть — частный случай). Пока код действует, сервер объявляется в
торрент-сети под ключами, выведенными из кода (pair_keys); плеер с кодом находит его оттуда.
Ключ сервера плеер ещё не знает — сервер доказывает, что знает код: HMAC(секрет кода, nonce
плеера + отпечаток сертификата). Подставной сервер без кода такого не сделает, а код уходит
серверу только после проверки доказательства."""

from __future__ import annotations

import hashlib
import hmac
import secrets
import sqlite3
import time

CODE_TTL_S = 300
MAX_FAILED = 10  # неудачных попыток подряд — и ввод кодов блокируется на минуту (ОД-9)
MAX_FAILED_REMOTE = 5  # из интернета строже: 5 за минуту — за 5 минут жизни кода < 0,003 % угадать


def _h(s: str) -> bytes:
    return hashlib.sha256(s.encode()).digest()


def pair_secret(code: str) -> bytes:
    return hashlib.sha256(b"arxnova-pair:" + "".join(ch for ch in code if ch.isdigit()).encode()).digest()


def pair_keys(secret: bytes) -> dict[str, bytes]:
    """Ключи торрент-сети для поиска сервера по коду (20 байт, как info_hash)."""
    return {k: hashlib.sha1(k.encode() + b":" + secret).digest() for k in ("tcp", "udp", "knock")}


def pair_proof(secret: bytes, nonce: bytes, fingerprint: str) -> str:
    return hmac.new(secret, nonce + bytes.fromhex(fingerprint), hashlib.sha256).hexdigest()


def new_pair_code(conn: sqlite3.Connection, role: str = "owner") -> str:
    code = f"{secrets.randbelow(1_000_000):06d}"
    with conn:
        conn.execute("DELETE FROM pair_codes WHERE expires_at < ?", (int(time.time()),))
        conn.execute(
            "INSERT OR REPLACE INTO pair_codes (code_hash, role, expires_at, secret) VALUES (?,?,?,?)",
            (_h(code), role, int(time.time()) + CODE_TTL_S, pair_secret(code)),
        )
    return code


def active_secrets(conn: sqlite3.Connection) -> list[bytes]:
    return [r[0] for r in conn.execute(
        "SELECT secret FROM pair_codes WHERE used = 0 AND expires_at >= ? AND secret IS NOT NULL", (int(time.time()),))]


def secret_for_key(conn: sqlite3.Connection, tcp_key: bytes) -> bytes | None:
    for s in active_secrets(conn):
        if hmac.compare_digest(pair_keys(s)["tcp"], tcp_key):
            return s
    return None


class PairError(Exception):
    pass


_failed: list[float] = []


def redeem(conn: sqlite3.Connection, code: str, device_name: str, remote: bool = False) -> tuple[int, str]:
    now = time.time()
    recent = [t for t in _failed if now - t < 60]
    _failed[:] = recent
    if len(recent) >= (MAX_FAILED_REMOTE if remote else MAX_FAILED):
        raise PairError("слишком много попыток, подождите минуту")
    row = conn.execute(
        "SELECT role FROM pair_codes WHERE code_hash = ? AND used = 0 AND expires_at >= ?",
        (_h(code.strip()), int(now)),
    ).fetchone()
    if not row:
        _failed.append(now)
        raise PairError("код неверный или истёк")
    token = secrets.token_urlsafe(32)
    with conn:
        conn.execute("UPDATE pair_codes SET used = 1 WHERE code_hash = ?", (_h(code.strip()),))
        cur = conn.execute(
            "INSERT INTO devices (name, role, token_hash) VALUES (?,?,?)",
            (device_name[:80] or "устройство", row["role"], _h(token)),
        )
    return cur.lastrowid, token


def device_for_token(conn: sqlite3.Connection, token: str) -> sqlite3.Row | None:
    row = conn.execute(
        "SELECT * FROM devices WHERE token_hash = ? AND revoked = 0", (_h(token),)
    ).fetchone()
    if row:
        with conn:
            conn.execute("UPDATE devices SET last_seen = ? WHERE id = ?", (int(time.time()), row["id"]))
    return row
