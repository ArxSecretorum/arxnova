"""Общий доступ к медиатеке (ТЗ, §15.1, ОД-1…ОД-10): приглашения, вход гостя, ограничения, отзыв.

Правила, которые держат правовые опоры (ОД-10), зашиты здесь, а не в настройках:
- приглашение поимённое и одноразовое, код живёт 24 ч;
- гостей не больше MAX_GUESTS = 5 при любой конфигурации (настройкой можно только меньше);
- одновременно слушают не больше MAX_STREAMS = 2 гостей;
- суточный объём на гостя — параметр.

Приглашение — короткий ключ (решение 0007): 15 случайных знаков и контрольный. В нём нет ни адреса,
ни отпечатка, ни имён: сервер объявляется в торрент-сети под метками, выведенными из ключа, и
доказывает гостю, что знает его (тот же путь, что у сопряжения по коду, см. auth).
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import sqlite3
import threading
import time

from arxnova.auth import pair_keys, tidy_name

MAX_GUESTS = 5  # жёсткий потолок (ОД-7, ОД-10 п. 2)
MAX_STREAMS = 2
INVITE_TTL_S = 24 * 3600
STREAM_WINDOW_S = 30  # гость «слушает», если брал звук за последние 30 с
# Без похожих символов (O/0, I/1, L): ключ диктуют голосом и переписывают руками. Знаков 31 — число
# простое, поэтому контрольный знак ловит любую одиночную опечатку и перестановку соседних.
ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"
KEY_LEN = 16  # 15 случайных (74 бита) + контрольный
# Свежая метка в торрент-сети находится хуже постоянной (замер 02.11.2026 с M9): первые минуты
# объявляемся чаще, дальше — как постоянная метка сервера (узлы помнят объявление около 30 мин).
ANNOUNCE_FAST_S, ANNOUNCE_FAST_FOR_S = 60, 10 * 60
ANNOUNCE_SLOW_S = 15 * 60
# Сколько после конца ключа сервер ещё отвечает, что с ним: гость с ключом, уже использованным на другом
# приборе, не ждёт минуту поиска. Старше — как будто ключа и не было; меток в торрент-сети не больше ANSWERING_MAX.
DEAD_KEEP_S = 3 * 24 * 3600
ANSWERING_MAX = 16

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


def check_symbol(payload: str) -> str:
    """Контрольный знак: сумма знаков с весами 1…16 делится на 31. Та же формула — в плеере (InviteKey)."""
    total = sum((i + 1) * ALPHABET.index(c) for i, c in enumerate(payload))
    return ALPHABET[(-total * pow(KEY_LEN, -1, len(ALPHABET))) % len(ALPHABET)]


def new_key() -> str:
    payload = "".join(secrets.choice(ALPHABET) for _ in range(KEY_LEN - 1))
    return payload + check_symbol(payload)


def key_ok(code: str) -> bool:
    n = normalize(code)
    return len(n) == KEY_LEN and all(c in ALPHABET for c in n) and check_symbol(n[:-1]) == n[-1]


def pretty(code: str) -> str:
    """Как ключ показывать человеку: группами по 4."""
    n = normalize(code)
    return "-".join(n[i:i + 4] for i in range(0, len(n), 4))


def invite_secret(code: str) -> bytes:
    """Из него — метки в торрент-сети и доказательство (auth.pair_keys, auth.pair_proof)."""
    return hashlib.sha256(b"arxnova-invite:" + normalize(code).encode()).digest()


def answering(conn: sqlite3.Connection) -> list[tuple[bytes, int, str]]:
    """(секрет, когда создано, состояние) ключей, на которые сервер отвечает: живые — всегда, мёртвые — ещё
    DEAD_KEEP_S. Состояния: active, used, revoked, expired. Созданные до ключей секрета не имеют."""
    now = int(time.time())

    def state(r) -> str:
        return "revoked" if r[3] else "used" if r[2] is not None else "expired" if r[4] < now else "active"

    rows = conn.execute(
        "SELECT secret, created_at, used_at, revoked, expires_at FROM invites WHERE secret IS NOT NULL AND expires_at >= ? "
        "ORDER BY (used_at IS NULL AND revoked = 0 AND expires_at >= ?) DESC, created_at DESC, id DESC LIMIT ?",
        (now - DEAD_KEEP_S, now, ANSWERING_MAX)).fetchall()
    return [(r[0], r[1], state(r)) for r in rows]


def active_secrets(conn: sqlite3.Connection) -> list[tuple[bytes, int]]:
    """(секрет, когда создано) живых приглашений."""
    return [(s, created) for s, created, st in answering(conn) if st == "active"]


def secret_for_key(conn: sqlite3.Connection, tcp_key: bytes) -> tuple[bytes, str] | None:
    """(секрет, состояние) ключа по его метке."""
    for secret, _, state in answering(conn):
        if hmac.compare_digest(pair_keys(secret)["tcp"], tcp_key):
            return secret, state
    return None


def announce_due(age_s: float, since_last_s: float | None, alive: bool = True) -> bool:
    """Пора ли объявить метку приглашения: сразу, в первые минуты часто, потом редко. Мёртвый ключ — всегда редко."""
    if since_last_s is None:
        return True
    return since_last_s >= (ANNOUNCE_FAST_S if alive and age_s < ANNOUNCE_FAST_FOR_S else ANNOUNCE_SLOW_S)


def forget_dead_keys(conn: sqlite3.Connection) -> None:
    """Открытый ключ нужен, только пока приглашение ждёт (повторный показ владельцу)."""
    with conn:
        conn.execute("UPDATE invites SET key_text = NULL WHERE key_text IS NOT NULL AND expires_at < ?", (int(time.time()),))


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
    code = new_key()
    now = int(time.time())
    with conn:
        cur = conn.execute(
            "INSERT INTO invites (code_hash, guest_name, created_by, created_at, expires_at, secret, key_text) VALUES (?,?,?,?,?,?,?)",
            (_h(code), name, created_by, now, now + INVITE_TTL_S, invite_secret(code), code),
        )
    return {"id": cur.lastrowid, "code": code, "guest_name": name, "expires_at": now + INVITE_TTL_S}


def link(code: str) -> str:
    """Ссылка для плеера: только ключ. Адрес и отпечаток гость узнаёт у самого сервера (решение 0007)."""
    return f"arxnova://join?k={normalize(code)}"


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
        conn.execute("UPDATE invites SET used_at = ?, device_id = ?, key_text = NULL WHERE id = ?", (now, cur.lastrowid, row["id"]))
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
        {"id": r["id"], "name": r["guest_name"], "expires_at": r["expires_at"], "key": r["key_text"]}
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
        return conn.execute("UPDATE invites SET revoked = 1, key_text = NULL WHERE id = ? AND used_at IS NULL", (invite_id,)).rowcount > 0


class Streams:
    """Кто из гостей сейчас слушает (ОД-7): не больше MAX_STREAMS одновременно. Гости уступают
    владельцу: пока владелец слушает сам, канал дома делится — гостю место только одно."""

    def __init__(self):
        self.lock = threading.Lock()
        self.last: dict[int, float] = {}
        self.owner_at = 0.0
        self.stream_at = 0.0

    def owner_listening(self) -> None:
        self.owner_at = time.time()

    def stream_asked(self) -> None:
        """Поток попросил следующие треки: слушают, даже если звук играет с устройства (скачанные треки,
        поток «только с устройства») и кусков у сервера не просят."""
        self.stream_at = time.time()

    def last_listen_at(self) -> float:
        """Когда в последний раз было видно, что слушают: кусок звука (владелец или гость) или запрос следующих треков
        потока; 0 — с запуска не было. По нему самообновление и выкладка новой версии ждут тишины: перезапуск посреди
        прослушивания обрывает поток (docs/bugs/2026-11-06-stream-stall-on-deploy.md)."""
        with self.lock:
            return max([self.owner_at, self.stream_at, *self.last.values()], default=0.0)

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
