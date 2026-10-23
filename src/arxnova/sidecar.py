"""Разбор медиатеки — рядом с самой медиатекой: `<папка>/.arxnova/analysis.sqlite`.

Разбор звука (EffNet, фасеты Demucs, темп, тональность, CLAP) — часы работы на каждую тысячу
треков. В базе сервера он привязан к номеру трека, а номер свой на каждом компьютере. Здесь он
хранится по ключу звука — (MD5 сигнала, начало трека в листе CUE): ключ не зависит ни от пути,
ни от системы. Диск с музыкой подключили к другому серверу (Mac, Windows, Linux, Pi) — обход
находит готовый разбор и берёт его, а не разбирает заново (решение владельца 23.10.2026).

Сюда же — разметка мостов (КО-1), история и отметки владельца («нравится», «не предлагать»,
причины пропусков) и настройки потока (смелость, порог шлюза, запреты, свои сценарии): новый
сервер подбирает по вашему вкусу сразу. История гостей, гости и ключи сервера в медиатеку не
пишутся: диск могут подключить к чужому компьютеру (решение владельца 23.10.2026).

Файл — SQLite с обычным журналом (без WAL: на exFAT и сетевых дисках так надёжнее), каждый
выгруз — одна транзакция: выдернули диск посреди записи — останется прежнее состояние. Папка
только для чтения — молча пропускаем."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

DIRNAME = ".arxnova"
FILENAME = "analysis.sqlite"
FORMAT = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS emb (
    md5 BLOB NOT NULL, cue_ms INTEGER NOT NULL, model TEXT NOT NULL,
    vector BLOB NOT NULL, styles BLOB, windows INTEGER NOT NULL,
    PRIMARY KEY (md5, cue_ms, model));
CREATE TABLE IF NOT EXISTS facet (
    md5 BLOB NOT NULL, cue_ms INTEGER NOT NULL, facet TEXT NOT NULL, model TEXT NOT NULL,
    vector BLOB NOT NULL, PRIMARY KEY (md5, cue_ms, facet));
CREATE TABLE IF NOT EXISTS feat (
    md5 BLOB NOT NULL, cue_ms INTEGER NOT NULL, version INTEGER NOT NULL,
    bpm REAL, bpm_conf REAL, bpm_src TEXT, musical_key INTEGER, key_conf REAL,
    loudness_db REAL, onset_rate REAL, vocal_frac REAL, vocal_ratio REAL,
    PRIMARY KEY (md5, cue_ms));
CREATE TABLE IF NOT EXISTS clap (
    md5 BLOB NOT NULL, cue_ms INTEGER NOT NULL, vector BLOB NOT NULL,
    PRIMARY KEY (md5, cue_ms));
CREATE TABLE IF NOT EXISTS pcm (
    relpath TEXT NOT NULL, size INTEGER NOT NULL, md5 BLOB NOT NULL, PRIMARY KEY (relpath, size));
CREATE TABLE IF NOT EXISTS blob (name TEXT PRIMARY KEY, data BLOB NOT NULL);
CREATE TABLE IF NOT EXISTS event (
    event_id TEXT PRIMARY KEY, ts INTEGER NOT NULL, md5 BLOB NOT NULL, cue_ms INTEGER NOT NULL,
    event_type TEXT NOT NULL, played_ratio_pct INTEGER, track_length_s INTEGER,
    is_organic INTEGER NOT NULL, session_id INTEGER, prev_md5 BLOB, prev_cue INTEGER,
    transition_type TEXT, recording_key BLOB, skip_reason TEXT);
CREATE TABLE IF NOT EXISTS label (
    a_md5 BLOB NOT NULL, a_cue INTEGER NOT NULL, b_md5 BLOB NOT NULL, b_cue INTEGER NOT NULL,
    label TEXT NOT NULL, ts INTEGER NOT NULL,
    PRIMARY KEY (a_md5, a_cue, b_md5, b_cue));
"""

def index_files(cfg) -> dict[str, Path]:
    """Файлы, которые едут вместе с разбором: векторы текстов сценариев (без них сценарии не
    собрать без модели CLAP) и свои сценарии владельца."""
    return {"scenario_text.npz": Path(cfg.index_dir) / "laion-clap-music" / "scenario_text.npz",
            "scenarios.toml": Path(cfg.data_dir) / "scenarios.toml"}


FEAT_COLS = ("version", "bpm", "bpm_conf", "bpm_src", "musical_key", "key_conf", "loudness_db",
             "onset_rate", "vocal_frac", "vocal_ratio")


def path(root: Path) -> Path:
    return Path(root) / DIRNAME / FILENAME


def _open(root: Path, create: bool) -> sqlite3.Connection | None:
    p = path(root)
    if not p.exists() and not create:
        return None
    try:
        p.parent.mkdir(exist_ok=True)
        c = sqlite3.connect(p, timeout=15)
        c.execute("PRAGMA journal_mode=DELETE")
        c.execute("PRAGMA synchronous=FULL")
        c.executescript(SCHEMA)
        c.execute("INSERT OR IGNORE INTO meta VALUES ('format', ?)", (str(FORMAT),))
        c.commit()
        return c
    except (sqlite3.Error, OSError):  # диск только для чтения, сеть отвалилась — без файла
        return None


def _keys_under(conn: sqlite3.Connection, root: Path) -> dict[int, tuple[bytes, int]]:
    """track_id → ключ звука для треков, чьи файлы лежат в этой папке."""
    prefix = str(Path(root)).rstrip(os.sep) + os.sep
    # файл в базе привязан к одному треку, а образ с листом CUE — это много треков одного звука:
    # трек в этой медиатеке, если здесь лежит файл с его звуком (MD5)
    here = {bytes(md5) for md5, p in conn.execute(
        "SELECT t.md5, f.path FROM files f JOIN tracks t ON t.id = f.track_id") if p.startswith(prefix)}
    return {tid: (bytes(md5), cue_ms) for tid, md5, cue_ms in conn.execute("SELECT id, md5, cue_ms FROM tracks")
            if bytes(md5) in here}


def export(conn: sqlite3.Connection, roots: list[Path], files: dict[str, Path] | None = None) -> int:
    """Дописать в файлы медиатек разбор, которого там ещё нет. files — небольшие файлы индекса,
    без которых разбор не собрать (векторы текстов сценариев: считать их — модель CLAP на 2 ГБ).
    Возвращает число новых строк."""
    added = 0
    for root in roots:
        keys = _keys_under(conn, root)
        if not keys:
            continue
        sc = _open(root, create=True)
        if sc is None:
            continue
        try:
            with sc:
                for name, fp in (files or {}).items():
                    if fp.exists():
                        data = fp.read_bytes()
                        old = sc.execute("SELECT data FROM blob WHERE name = ?", (name,)).fetchone()
                        if old is None or bytes(old[0]) != data:
                            sc.execute("INSERT OR REPLACE INTO blob VALUES (?,?)", (name, data))
            have = {t: set() for t in ("emb", "facet", "clap")}
            for md5, cue, model in sc.execute("SELECT md5, cue_ms, model FROM emb"):
                have["emb"].add((bytes(md5), cue, model))
            for md5, cue, facet in sc.execute("SELECT md5, cue_ms, facet FROM facet"):
                have["facet"].add((bytes(md5), cue, facet))
            for md5, cue in sc.execute("SELECT md5, cue_ms FROM clap"):
                have["clap"].add((bytes(md5), cue))
            feats = {(bytes(r[0]), r[1]): tuple(r[2:]) for r in
                     sc.execute(f"SELECT md5, cue_ms, {', '.join(FEAT_COLS)} FROM feat")}
            with sc:
                for tid, model, vector, styles, windows in conn.execute(
                        "SELECT track_id, model, vector, styles, windows FROM embeddings"):
                    k = keys.get(tid)
                    if k and (*k, model) not in have["emb"]:
                        sc.execute("INSERT OR REPLACE INTO emb VALUES (?,?,?,?,?,?)", (*k, model, vector, styles, windows))
                        added += 1
                for tid, facet, model, vector in conn.execute(
                        "SELECT track_id, facet, model, vector FROM facet_vectors"):
                    k = keys.get(tid)
                    if k and (*k, facet) not in have["facet"]:
                        sc.execute("INSERT OR REPLACE INTO facet VALUES (?,?,?,?,?)", (*k, facet, model, vector))
                        added += 1
                for row in conn.execute(f"SELECT track_id, {', '.join(FEAT_COLS)} FROM track_features"):
                    k = keys.get(row[0])
                    # темп пересчитывается (по долям вместо ударных) — признаки сравниваем целиком
                    if k and feats.get(k) != tuple(row[1:]):
                        sc.execute(f"INSERT OR REPLACE INTO feat VALUES (?,?,{','.join('?' * len(FEAT_COLS))})",
                                   (*k, *row[1:]))
                        added += 1
                for tid, vector in conn.execute("SELECT track_id, vector FROM clap_vectors"):
                    k = keys.get(tid)
                    if k and k not in have["clap"]:
                        sc.execute("INSERT OR REPLACE INTO clap VALUES (?,?,?)", (*k, vector))
                        added += 1
                # MD5 звука, посчитанный декодированием (в заголовке не записан): новый сервер
                # не декодирует весь диск заново (~0,5 с на файл)
                prefix = str(Path(root)).rstrip(os.sep) + os.sep
                have_pcm = {(r[0], r[1]) for r in sc.execute("SELECT relpath, size FROM pcm")}
                for fpath, size, md5 in conn.execute(
                        "SELECT f.path, f.size, t.md5 FROM files f JOIN tracks t ON t.id = f.track_id "
                        "WHERE t.md5_source = 'pcm'"):
                    if fpath.startswith(prefix):
                        rel = fpath[len(prefix):].replace(os.sep, "/")
                        if (rel, size) not in have_pcm:
                            sc.execute("INSERT OR REPLACE INTO pcm VALUES (?,?,?)", (rel, size, md5))
                            have_pcm.add((rel, size))
                have_ev = {r[0] for r in sc.execute("SELECT event_id FROM event")}
                for r in conn.execute(
                        """SELECT e.event_id, e.ts, e.track_id, e.event_type, e.played_ratio_pct, e.track_length_s,
                                  e.is_organic, e.session_id, e.prev_track_id, e.transition_type, e.recording_key,
                                  e.skip_reason
                           FROM events e JOIN devices d ON d.id = e.device_id
                           WHERE d.role = 'owner' AND e.track_id IS NOT NULL"""):
                    k = keys.get(r[2])
                    if not k or r[0] in have_ev:
                        continue
                    prev = keys.get(r[8]) if r[8] is not None else None
                    sc.execute("INSERT OR IGNORE INTO event VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                               (r[0], r[1], *k, r[3], r[4], r[5], r[6], r[7],
                                prev[0] if prev else None, prev[1] if prev else None, r[9], r[10], r[11]))
                    added += 1
                for a, b, label, ts in conn.execute("SELECT a, b, label, ts FROM bridge_labels"):
                    ka, kb = keys.get(a), keys.get(b)
                    if ka and kb:
                        sc.execute("INSERT OR REPLACE INTO label VALUES (?,?,?,?,?,?)", (*ka, *kb, label, ts))
        except (sqlite3.Error, OSError):
            pass  # диск выдернули посреди выгрузки — транзакция откатится, база сервера цела
        finally:
            sc.close()
    return added


def import_missing(conn: sqlite3.Connection, roots: list[Path], files: dict[str, Path] | None = None) -> int:
    """Взять из файлов медиатек разбор для треков, у которых его в базе сервера нет (и файлы
    индекса, которых нет на этом сервере). Возвращает число треков, получивших хоть что-то."""
    got: set[int] = set()
    for root in roots:
        sc = _open(root, create=False)
        if sc is None:
            continue
        try:
            for name, fp in (files or {}).items():
                row = sc.execute("SELECT data FROM blob WHERE name = ?", (name,)).fetchone()
                if row is not None and not fp.exists():
                    fp.parent.mkdir(parents=True, exist_ok=True)
                    fp.write_bytes(bytes(row[0]))
            keys = _keys_under(conn, root)
            if not keys:
                continue
            by_key = {k: t for t, k in keys.items()}
            has_emb = {(r[0], r[1]) for r in conn.execute("SELECT track_id, model FROM embeddings")}
            has_facet = {(r[0], r[1]) for r in conn.execute("SELECT track_id, facet FROM facet_vectors")}
            has_feat = {r[0] for r in conn.execute("SELECT track_id FROM track_features")}
            has_clap = {r[0] for r in conn.execute("SELECT track_id FROM clap_vectors")}
            with conn:
                for md5, cue, model, vector, styles, windows in sc.execute("SELECT * FROM emb"):
                    t = by_key.get((bytes(md5), cue))
                    if t is not None and (t, model) not in has_emb:
                        conn.execute("INSERT OR REPLACE INTO embeddings (track_id, model, vector, styles, windows) "
                                     "VALUES (?,?,?,?,?)", (t, model, vector, styles, windows))
                        got.add(t)
                for md5, cue, facet, model, vector in sc.execute("SELECT * FROM facet"):
                    t = by_key.get((bytes(md5), cue))
                    if t is not None and (t, facet) not in has_facet:
                        conn.execute("INSERT OR REPLACE INTO facet_vectors (track_id, facet, model, vector) "
                                     "VALUES (?,?,?,?)", (t, facet, model, vector))
                        got.add(t)
                for row in sc.execute(f"SELECT md5, cue_ms, {', '.join(FEAT_COLS)} FROM feat"):
                    t = by_key.get((bytes(row[0]), row[1]))
                    if t is not None and t not in has_feat:
                        conn.execute(f"INSERT OR REPLACE INTO track_features (track_id, {', '.join(FEAT_COLS)}) "
                                     f"VALUES (?,{','.join('?' * len(FEAT_COLS))})", (t, *row[2:]))
                        got.add(t)
                for md5, cue, vector in sc.execute("SELECT * FROM clap"):
                    t = by_key.get((bytes(md5), cue))
                    if t is not None and t not in has_clap:
                        conn.execute("INSERT OR REPLACE INTO clap_vectors (track_id, vector) VALUES (?,?)", (t, vector))
                        got.add(t)
                evs = [r for r in sc.execute("SELECT * FROM event")]
                if evs:
                    have_ev = {r[0] for r in conn.execute("SELECT event_id FROM events")}
                    evs = [r for r in evs if r[0] not in have_ev and by_key.get((bytes(r[2]), r[3])) is not None]
                if evs:
                    dev = _carried_device(conn)
                    for (eid, ts, md5, cue, et, pct, length, organic, sess, pmd5, pcue, trans, rkey, reason) in evs:
                        t = by_key[(bytes(md5), cue)]
                        prev = by_key.get((bytes(pmd5), pcue)) if pmd5 is not None else None
                        conn.execute(
                            """INSERT OR IGNORE INTO events (event_id, device_id, ts, md5, cue_ms, track_id, event_type,
                               played_ratio_pct, track_length_s, is_organic, session_id, prev_track_id,
                               transition_type, recording_key, skip_reason) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                            (eid, dev, ts, md5, cue, t, et, pct, length, organic, sess, prev, trans, rkey, reason))
                        got.add(t)
                for a_md5, a_cue, b_md5, b_cue, label, ts in sc.execute("SELECT * FROM label"):
                    a, b = by_key.get((bytes(a_md5), a_cue)), by_key.get((bytes(b_md5), b_cue))
                    if a is not None and b is not None:
                        conn.execute("INSERT OR IGNORE INTO bridge_labels (a, b, label, ts) VALUES (?,?,?,?)",
                                     (a, b, label, ts))
        except (sqlite3.Error, OSError):
            pass
        finally:
            sc.close()
    return len(got)


CARRIED_NAME = "История из медиатеки"


def _carried_device(conn: sqlite3.Connection) -> int:
    """Служебное устройство владельца для перенесённой истории: войти с него нельзя (токена нет,
    устройство отозвано), но вкус владельца собирается со всех его устройств — и с этого."""
    row = conn.execute("SELECT id FROM devices WHERE role = 'owner' AND name = ? AND revoked = 1",
                       (CARRIED_NAME,)).fetchone()
    if row:
        return row[0]
    cur = conn.execute("INSERT INTO devices (name, role, token_hash, revoked) VALUES (?, 'owner', ?, 1)",
                       (CARRIED_NAME, os.urandom(32)))
    return cur.lastrowid


SETTINGS = ("boldness", "gate_pct", "gate_forbid")


def sync_settings(cfg, roots: list[Path]) -> bool:
    """Настройки потока едут с медиатекой. Сервер, где их не трогали (значения по умолчанию),
    берёт их из медиатеки; иначе свои пишет туда. True — настройки этого сервера изменились."""
    import json

    from arxnova.config import Config

    default = Config(data_dir=cfg.data_dir)
    mine = {k: getattr(cfg, k) for k in SETTINGS}
    untouched = all(mine[k] == getattr(default, k) for k in SETTINGS)
    changed = False
    for root in roots:
        sc = _open(root, create=not untouched)
        if sc is None:
            continue
        try:
            row = sc.execute("SELECT value FROM meta WHERE key = 'settings'").fetchone()
            theirs = json.loads(row[0]) if row else None
            if untouched and theirs and theirs != mine:
                for k in SETTINGS:
                    if k in theirs:
                        setattr(cfg, k, theirs[k])
                cfg.save()
                changed, untouched = True, False
                mine = {k: getattr(cfg, k) for k in SETTINGS}
            elif not untouched and theirs != mine:
                with sc:
                    sc.execute("INSERT OR REPLACE INTO meta VALUES ('settings', ?)", (json.dumps(mine),))
        except (sqlite3.Error, OSError, ValueError):
            pass
        finally:
            sc.close()
    return changed


def pcm_md5s(root: Path) -> dict[tuple[str, int], bytes]:
    """(путь от корня медиатеки через «/», размер) → MD5 звука, уже посчитанный другим сервером."""
    sc = _open(root, create=False)
    if sc is None:
        return {}
    try:
        return {(r[0], r[1]): bytes(r[2]) for r in sc.execute("SELECT relpath, size, md5 FROM pcm")}
    except sqlite3.Error:
        return {}
    finally:
        sc.close()
