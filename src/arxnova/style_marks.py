"""Стиль трека, поставленный владельцем (решение 0024).

Стиль сервер решает сам — по совокупности свидетельств (style_evidence, решение 0023). Остаются пограничные треки, где
модели слышат одно, а люди называют другое: оригинальный «Kuos» Banco de Gaia по описанию автора — транс, обе модели
слышат хаус. Для них — слово владельца: оно применяется после всех свидетельств.

Поправка относится к записи: все издания одной песни (альбом, сингл, сборник) получают тот же стиль. Действует
последнее по времени действие над любым изданием; последним было снятие — поправки нет. Снятая поправка хранится
строкой с пустой меткой: файл медиатеки (sidecar) не вернёт прежнюю.

Каталог (nova.Library) правится дважды: при сборке — apply(), и на месте, сразу после запроса плеера, — set_live():
ответы меняются тем же запросом, а размеры стилей, полки и поток догоняют фоновой пересборкой."""

from __future__ import annotations

import sqlite3
import time
from collections import Counter


def rows(conn: sqlite3.Connection) -> dict[int, tuple[str, int]]:
    """track_id → (метка модели или пусто, когда поставлена)."""
    try:
        return {r[0]: (r[1], r[2]) for r in conn.execute("SELECT track_id, label, ts FROM style_marks")}
    except sqlite3.OperationalError:  # база прежней версии, открытая без миграций (замеры на копии)
        return {}


def put(conn: sqlite3.Connection, track_id: int, label: str, ts: int | None = None) -> int:
    """Поставить (label) или снять (пусто). Вызывать внутри транзакции. Время не идёт назад: «побеждает позднее»
    не должно зависеть от того, что часы сервера поправили."""
    ts = int(time.time() * 1000) if ts is None else ts
    last = conn.execute("SELECT max(ts) FROM style_marks").fetchone()[0]
    if last is not None and ts <= last:
        ts = last + 1
    conn.execute("INSERT OR REPLACE INTO style_marks (track_id, label, ts) VALUES (?,?,?)", (track_id, label, ts))
    return ts


def title(label: str) -> tuple[str, str]:
    """Метка модели → жанр и стиль, как их видит плеер."""
    from arxnova.nova import STYLE_TITLES

    genre, _, style = label.partition("---")
    return genre, STYLE_TITLES.get((genre, style), style)


def resolve(lib, genre: str, style: str) -> str | None:
    """Жанр и стиль, как их видит плеер, → метка модели; None — такого стиля сервер не знает."""
    want = (genre.strip().casefold(), style.strip().casefold())
    for label in lib.style_labels:
        g, s = title(label)
        if (g.casefold(), s.casefold()) == want:
            return label
    return None


def choices(lib) -> list[dict]:
    """Все стили, которые можно поставить, и сколько треков медиатеки сейчас с каждым."""
    size = Counter((t.genre, t.style) for t in lib.tracks.values())
    seen, out = set(), []
    for label in lib.style_labels:
        key = title(label)
        if key not in seen:
            seen.add(key)
            out.append({"genre": key[0], "style": key[1], "tracks": size.get(key, 0)})
    return sorted(out, key=lambda x: (x["genre"], x["style"]))


def _set(lib, info, label: str) -> None:
    lib.style_auto.setdefault(info.id, (info.genre, info.style))  # что решила автоматика — чтобы вернуть без пересборки
    info.genre, info.style = title(label)
    lib._style_genre.setdefault(info.style, info.genre)


def apply(lib, marks: dict[int, tuple[str, int]]) -> None:
    """При сборке каталога, после слияния изданий: стиль владельца — на все издания записи."""
    lib.style_auto = {}
    known = set(lib.style_labels)
    last: dict[bytes, tuple[int, str]] = {}  # запись → последнее действие над любым её изданием
    for tid, (label, ts) in marks.items():
        info = lib.tracks.get(tid)
        if info is not None and ts >= last.get(info.recording_key, (-1, ""))[0]:
            last[info.recording_key] = (ts, label)
    want = {rk: label for rk, (_, label) in last.items() if label in known}  # метку другой модели пропускаем, не падаем
    if not want:
        return
    if not hasattr(lib, "_style_genre"):
        lib._style_genre = {}
    for info in lib.tracks.values():
        label = want.get(info.recording_key)
        if label:
            _set(lib, info, label)


def set_live(lib, track_id: int, label: str) -> None:
    """В работающем каталоге, сразу: поставить (label) или снять (пусто) у всех изданий записи."""
    for tid in lib.editions.get(lib.tracks[track_id].recording_key) or [track_id]:
        info = lib.tracks[tid]
        if label:
            _set(lib, info, label)
        elif tid in lib.style_auto:
            info.genre, info.style = lib.style_auto.pop(tid)
    lib.__dict__.pop("_facets_cache", None)  # числа у жанров и стилей считались по прежним подписям
