"""Подключение своего устройства без кода и без камеры (решение владельца 26.10.2026: у DAP нет
камеры, у Pi нет экрана). Новое устройство в домашней сети жмёт «Подключить» — запрос уходит на
все подключённые устройства владельца; владелец жмёт «Разрешить» — устройство подключено.

Защита: запрос можно поставить только из домашней сети; на обоих экранах — одинаковые две цифры
для сверки (чужое устройство в том же Wi-Fi с тем же именем не проскочит незамеченным); запрос
живёт 10 минут; одновременно — не больше нескольких. Ответ (токен) забирает только тот, кто
поставил запрос: у него секрет запроса. Хранится в памяти: после перезапуска сервера запросы
пропадают — новое устройство просто попросит снова."""

from __future__ import annotations

import hmac
import secrets
import sqlite3
import threading
import time
from dataclasses import dataclass, field

from arxnova import auth, guests

TTL_S = 600
MAX_PENDING = 5


class AccessError(Exception):
    def __init__(self, status: int, msg: str):
        super().__init__(msg)
        self.status = status


@dataclass
class Request:
    id: str
    secret: str
    name: str
    check: str  # две цифры для сверки на обоих экранах
    created: float = field(default_factory=time.time)
    status: str = "pending"  # pending → approved | denied
    device_id: int | None = None
    token: str | None = None
    role: str | None = None


class AccessRequests:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._items: dict[str, Request] = {}

    def _expire(self, now: float) -> None:
        for k in [k for k, r in self._items.items() if now - r.created > TTL_S]:
            del self._items[k]

    def ask(self, name: str) -> Request:
        now = time.time()
        with self._lock:
            self._expire(now)
            if sum(r.status == "pending" for r in self._items.values()) >= MAX_PENDING:
                raise AccessError(429, "слишком много запросов — подождите несколько минут")
            r = Request(secrets.token_hex(8), secrets.token_urlsafe(24), (name.strip() or "устройство")[:80],
                        str(10 + secrets.randbelow(90)))  # без «00» и «07»: две цифры читаются сразу
            self._items[r.id] = r
            return r

    def poll(self, rid: str, secret: str) -> Request:
        with self._lock:
            self._expire(time.time())
            r = self._items.get(rid)
            if r is None or not hmac.compare_digest(r.secret.encode(), secret.encode()):
                raise AccessError(404, "запрос не найден или истёк — попросите снова")
            if r.status != "pending":
                del self._items[rid]  # ответ забирают один раз: токен больше нигде не лежит
            return r

    def pending(self) -> list[Request]:
        with self._lock:
            self._expire(time.time())
            return sorted((r for r in self._items.values() if r.status == "pending"), key=lambda r: r.created)

    def answer(self, conn: sqlite3.Connection, rid: str, allow: bool, role: str = "guest",
               max_guests: int | None = None) -> Request:
        """Владелец решает, КЕМ пустить: своим устройством (полный доступ: скачивание, папка на
        сервере, гости) или гостем (лимиты гостя, без скачивания). Друг у вас в Wi-Fi попросит
        доступ — его пускают гостем, а не «своим» (замечание владельца 26.10.2026)."""
        with self._lock:
            self._expire(time.time())
            r = self._items.get(rid)
            if r is None or r.status != "pending":
                raise AccessError(404, "запрос не найден или уже обработан")
            if allow:
                if role not in ("owner", "guest"):
                    raise AccessError(400, "роль — owner или guest")
                if role == "guest":
                    if guests.occupied(conn) >= guests.limit(max_guests):
                        raise AccessError(409, "мест для гостей нет — отзовите кого-нибудь")
                    r.device_id, r.token = auth.add_device(conn, r.name, "guest", guest_name=r.name)
                else:
                    r.device_id, r.token = auth.add_device(conn, r.name, "owner")
                r.role = role
                r.status = "approved"
            else:
                r.status = "denied"
            return r
