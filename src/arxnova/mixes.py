"""Заготовленные подборки (ТЗ, ФТ-4.7, П-5): что играть, когда сервера не видно.

Для прибора — K подборок по N треков только из того, что лежит на нём самом (каталог прибора,
П-2), поэтому без связи они играют с его памяти, без кэша. Затравки — недавно игравшие и
давно не игравшие лайкнутые треки; подборка строится тем же подбором «похожие на трек», с
правилами разнесения и «не альбомами». Пересчёт — не чаще раза в сутки или когда меняется
каталог прибора.
"""

from __future__ import annotations

import hashlib
import json
import random
import sqlite3
import time

from arxnova import nova

K_MIXES = 6
N_TRACKS = 25
MIN_ON_DEVICE = 10  # меньше — подборки не имеют смысла
REBUILD_S = 24 * 3600
LIKED_IDLE_MS = 30 * 24 * 3600 * 1000  # лайкнутые, не игравшие месяц — тоже затравки


def catalog_stamp(conn: sqlite3.Connection, device_id: int) -> str:
    h = hashlib.sha256()
    for (m,) in conn.execute("SELECT md5 FROM device_catalog WHERE device_id = ? ORDER BY md5", (device_id,)):
        h.update(m)
    return h.hexdigest()[:16]


def on_device(conn: sqlite3.Connection, device_id: int) -> set[int]:
    return {r[0] for r in conn.execute(
        "SELECT t.id FROM tracks t JOIN device_catalog c ON c.md5 = t.md5 AND c.device_id = ?", (device_id,))}


def seeds(events: list[sqlite3.Row], allowed: set[int], liked: set[int], now_ms: int, rng: random.Random) -> list[int]:
    """Недавние прослушивания (свежие первыми), затем давно не игравшие лайкнутые, затем случайные."""
    out: list[int] = []
    last_play: dict[int, int] = {}
    for e in reversed(events):
        t = e["track_id"]
        if t is None or t not in allowed:
            continue
        if e["event_type"] in ("listen", "replay", "like"):
            last_play.setdefault(t, e["ts"])
            if t not in out and len(out) < K_MIXES // 2:
                out.append(t)
    idle_liked = [t for t in liked if t in allowed and now_ms - last_play.get(t, 0) > LIKED_IDLE_MS and t not in out]
    rng.shuffle(idle_liked)
    out += idle_liked[: K_MIXES - len(out)]
    rest = [t for t in allowed if t not in out]
    rng.shuffle(rest)
    out += rest[: K_MIXES - len(out)]
    return out[:K_MIXES]


def build(lib: "nova.Library", conn: sqlite3.Connection, device_id: int, listener_ids: list[int],
          now_ms: int | None = None, rng: random.Random | None = None) -> list[dict]:
    now_ms = now_ms or int(time.time() * 1000)
    rng = rng or random.Random(device_id * 7919 + now_ms // 86_400_000)  # одна раскладка на сутки
    allowed = {t for t in on_device(conn, device_id) if t in lib.tracks}
    if len(allowed) < MIN_ON_DEVICE:
        return []
    events = nova.load_events(conn, listener_ids)
    prof = nova.Profile(lib, events, now_ms)
    out = []
    used_seeds: set[int] = set()
    for seed in seeds(events, allowed, set(prof.liked), now_ms, rng):
        if seed in used_seeds:
            continue
        used_seeds.add(seed)
        hist = [seed]
        picks_meta: list[tuple[int, str | None]] = [(seed, None)]
        while len(hist) < min(N_TRACKS, len(allowed)):
            picks = nova.next_tracks(lib, prof, current=hist[-1], history=hist, n=1, now_ms=now_ms,
                                     allowed=allowed, rng=rng, mode=nova.MODE_TRACK, seed=seed)
            if not picks or picks[0].track_id in hist:
                break
            hist.append(picks[0].track_id)
            picks_meta.append((picks[0].track_id, picks[0].reason))
        if len(hist) >= 5:
            out.append({"seed": seed, "tracks": [{"id": t, "reason": r} for t, r in picks_meta]})
    return out


def get(conn: sqlite3.Connection, lib: "nova.Library", device_id: int, listener_ids: list[int]) -> tuple[list[dict], int]:
    """Подборки прибора из кэша; устарели (сутки) или сменился каталог — собираются заново."""
    stamp = catalog_stamp(conn, device_id)
    row = conn.execute("SELECT built_at, stamp, data FROM device_mixes WHERE device_id = ?", (device_id,)).fetchone()
    now = int(time.time())
    if row and row["stamp"] == stamp and now - row["built_at"] < REBUILD_S:
        return json.loads(row["data"]), row["built_at"]
    data = build(lib, conn, device_id, listener_ids)
    with conn:
        conn.execute("INSERT OR REPLACE INTO device_mixes (device_id, built_at, stamp, data) VALUES (?,?,?,?)",
                     (device_id, now, stamp, json.dumps(data)))
    return data, now
