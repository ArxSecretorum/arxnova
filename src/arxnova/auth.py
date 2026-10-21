"""Сопряжение (API-4): 6-значный код на 5 минут → токен устройства. Храним только хеши."""

from __future__ import annotations

import hashlib
import secrets
import sqlite3
import time

CODE_TTL_S = 300
MAX_FAILED = 10  # неудачных попыток подряд — и ввод кодов блокируется на минуту (ОД-9)


def _h(s: str) -> bytes:
    return hashlib.sha256(s.encode()).digest()


def new_pair_code(conn: sqlite3.Connection, role: str = "owner") -> str:
    code = f"{secrets.randbelow(1_000_000):06d}"
    with conn:
        conn.execute("DELETE FROM pair_codes WHERE expires_at < ?", (int(time.time()),))
        conn.execute(
            "INSERT OR REPLACE INTO pair_codes (code_hash, role, expires_at) VALUES (?,?,?)",
            (_h(code), role, int(time.time()) + CODE_TTL_S),
        )
    return code


class PairError(Exception):
    pass


_failed: list[float] = []


def redeem(conn: sqlite3.Connection, code: str, device_name: str) -> tuple[int, str]:
    now = time.time()
    recent = [t for t in _failed if now - t < 60]
    _failed[:] = recent
    if len(recent) >= MAX_FAILED:
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
