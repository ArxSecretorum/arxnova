"""Жанры из MusicBrainz — только если владелец включил (config.musicbrainz; решение 23.10.2026).

Наружу уходят имя исполнителя и название альбома (поиск) и идентификаторы MusicBrainz — больше
ничего; не чаще запроса в секунду, с User-Agent и адресом проекта, как требует MusicBrainz. Ответ
— жанры и теги с числом голосов — кэшируется в базе (и едет с медиатекой, см. sidecar): альбом
спрашивается один раз, ненайденный — снова через 30 дней.

Жанры в MusicBrainz ставят люди, и на уровне исполнителя они размыты: «liquid funk» стоит и у
Calibre, и у Mistabishi. Поэтому это не ответ, а свидетельство — его сводят с оценкой по звуку
(см. shelves). Список жанров MusicBrainz — CC0; привязки жанров к релизам — CC BY-NC-SA 3.0
(для платного продукта нужна лицензия MetaBrainz)."""

from __future__ import annotations

import json
import sqlite3
import time
import urllib.parse
import urllib.request

from arxnova import __version__

API = "https://musicbrainz.org/ws/2"
UA = f"ArxNova/{__version__} (https://github.com/ArxSecretorum/arxnova)"
PAUSE_S = 1.1
RETRY_MISS_S = 30 * 86400
MIN_SCORE = 95

SCHEMA = """CREATE TABLE IF NOT EXISTS mb_cache (
    kind TEXT NOT NULL, key TEXT NOT NULL, mbid TEXT, genres TEXT NOT NULL, at INTEGER NOT NULL,
    PRIMARY KEY (kind, key))"""


class Client:
    def __init__(self, fetch=None):
        self._last = 0.0
        self._fetch = fetch or self._http

    def _http(self, url: str) -> dict:
        req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.loads(r.read())

    def get(self, path: str, **params) -> dict:
        wait = PAUSE_S - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.monotonic()
        return self._fetch(f"{API}/{path}?{urllib.parse.urlencode({**params, 'fmt': 'json'})}")


def _quote(s: str) -> str:
    return '"' + s.replace("\\", " ").replace('"', " ") + '"'


def _votes(d: dict) -> dict[str, int]:
    """Жанры и теги вместе (теги ловят «liquid drum and bass», которого нет среди жанров)."""
    out: dict[str, int] = {}
    for g in (d.get("genres") or []) + (d.get("tags") or []):
        name = g.get("name", "").strip().lower()
        if name:
            out[name] = max(out.get(name, 0), int(g.get("count", 0)) or 1)
    return out


def lookup_album(c: Client, artist: str, album: str) -> tuple[str | None, dict[str, int]]:
    q = f"releasegroup:{_quote(album)}" + (f" AND artist:{_quote(artist)}" if artist else "")
    r = c.get("release-group/", query=q, limit=3)
    best = next((x for x in r.get("release-groups", []) if int(x.get("score", 0)) >= MIN_SCORE), None)
    if not best:
        return None, {}
    d = c.get(f"release-group/{best['id']}", inc="genres+tags")
    return best["id"], _votes(d)


def lookup_artist(c: Client, artist: str) -> tuple[str | None, dict[str, int]]:
    r = c.get("artist/", query=f"artist:{_quote(artist)}", limit=3)
    best = next((x for x in r.get("artists", []) if int(x.get("score", 0)) >= MIN_SCORE), None)
    if not best:
        return None, {}
    d = c.get(f"artist/{best['id']}", inc="genres+tags")
    return best["id"], _votes(d)


VARIOUS = {"various artists", "various", "va", "v.a.", "сборник", "разные исполнители"}


def refresh(conn: sqlite3.Connection, client: Client | None = None, progress=None, should_stop=None) -> int:
    """Спросить MusicBrainz о новых альбомах и исполнителях медиатеки. Возвращает число запросов."""
    conn.execute(SCHEMA)
    c = client or Client()
    now = int(time.time())
    cached = {(r[0], r[1]): (r[2], r[3]) for r in conn.execute("SELECT kind, key, mbid, at FROM mb_cache")}

    def fresh(kind, key):
        hit = cached.get((kind, key))
        return hit is not None and (hit[0] is not None or now - hit[1] < RETRY_MISS_S)

    # сначала альбомы; исполнитель нужен, только если его альбом не нашёлся (иначе — тысячи лишних
    # запросов: у сборников и фитов исполнителей в разы больше, чем альбомов)
    albums: dict[tuple[str, str], set[str]] = {}
    for aa, a, alb in conn.execute("SELECT albumartist, artist, album FROM tracks"):
        who = (aa or a or "").strip()
        if alb and who:
            albums.setdefault((who, alb.strip()), set()).add(_primary(a or aa or ""))
    todo = [("album", f"{w}\t{al}", (w, al)) for w, al in sorted(albums) if not fresh("album", f"{w}\t{al}")]

    def artists_todo():
        found = {r[0] for r in conn.execute("SELECT key FROM mb_cache WHERE kind = 'album' AND mbid IS NOT NULL")}
        need = set()
        for (w, al), track_artists in albums.items():
            if f"{w}\t{al}" in found:
                continue
            need |= {x for x in ({_primary(w)} if w.lower() not in VARIOUS else track_artists) if x}
        cached_now = {r[0] for r in conn.execute("SELECT key FROM mb_cache WHERE kind = 'artist'")}
        return [("artist", x, (x,)) for x in sorted(need - {v for v in VARIOUS}) if x not in cached_now or not fresh("artist", x)]

    done = 0
    todo_artists_added = False
    while True:
        if not todo:
            if todo_artists_added:
                break
            todo, todo_artists_added = artists_todo(), True
            if not todo:
                break
        kind, key, args = todo.pop(0)
        if should_stop and should_stop():
            break
        try:
            if kind == "album":
                mbid, votes = lookup_album(c, args[0], args[1]) if args[0].lower() not in VARIOUS else \
                    lookup_album(c, "", args[1])
            else:
                mbid, votes = lookup_artist(c, args[0])
        except Exception:
            continue  # сеть или 503 — спросим в следующий раз
        with conn:
            conn.execute("INSERT OR REPLACE INTO mb_cache VALUES (?,?,?,?,?)", (kind, key, mbid, json.dumps(votes), now))
        done += 1
        if progress and done % 20 == 0:
            progress(f"  … MusicBrainz {done}, осталось {len(todo)}")
    return done


def _primary(artist: str) -> str:
    """Основной исполнитель без фитов: «A feat. B», «A & B», «A, B» → «A»."""
    import re

    return re.split(r"\s+(?:feat\.?|ft\.?|featuring|vs\.?|x)\s+|\s*[,&;/]\s*", artist.strip(), maxsplit=1, flags=re.I)[0].strip()


def track_genres(conn: sqlite3.Connection) -> dict[int, dict[str, int]]:
    """track_id → жанры: альбома, а если у альбома пусто — исполнителя."""
    conn.execute(SCHEMA)
    cache = {(k, key): json.loads(g) for k, key, g in conn.execute("SELECT kind, key, genres FROM mb_cache")}
    out = {}
    for tid, aa, a, alb in conn.execute("SELECT id, albumartist, artist, album FROM tracks"):
        who = (aa or a or "").strip()
        g = cache.get(("album", f"{who}\t{(alb or '').strip()}")) or {}
        if not g:
            for x in (a, aa):
                g = cache.get(("artist", _primary(x or ""))) or cache.get(("artist", (x or "").strip())) or {}
                if g:
                    break
        if g:
            out[tid] = g
    return out


def evidence(conn: sqlite3.Connection, pos: set[str], neg: set[str]) -> tuple[dict, dict]:
    """Свидетельства «за / против» полки по трекам: (по альбому, по исполнителю), каждое —
    track_id → (голоса за, голоса против). Только найденные в MusicBrainz записи."""
    conn.execute(SCHEMA)
    cache = {(k, key): json.loads(g) for k, key, g in conn.execute(
        "SELECT kind, key, genres FROM mb_cache WHERE mbid IS NOT NULL")}

    def votes(g):
        return sum(v for k, v in g.items() if k in pos), sum(v for k, v in g.items() if k in neg)

    album, artist = {}, {}
    for tid, aa, a, alb in conn.execute("SELECT id, albumartist, artist, album FROM tracks"):
        who = (aa or a or "").strip()
        g = cache.get(("album", f"{who}\t{(alb or '').strip()}"))
        if g:
            album[tid] = votes(g)
        for x in (a, aa):
            g = cache.get(("artist", _primary(x or "")))
            if g:
                artist[tid] = votes(g)
                break
    return album, artist
