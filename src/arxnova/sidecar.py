"""Разбор медиатеки — рядом с самой медиатекой: `<папка>/.arxnova/analysis.sqlite`.

Разбор звука (EffNet, MAEST, темп, признаки, CLAP) — часы работы на каждую тысячу
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
CREATE TABLE IF NOT EXISTS feat_more (  -- своя таблица: серверы прежней версии пишут в feat позиционно, лишний столбец сломал бы им запись
    md5 BLOB NOT NULL, cue_ms INTEGER NOT NULL, female_p REAL, female_stem_p REAL, PRIMARY KEY (md5, cue_ms));
CREATE TABLE IF NOT EXISTS voc (  -- CLAP выделенного голоса (решение 0014)
    md5 BLOB NOT NULL, cue_ms INTEGER NOT NULL, vector BLOB NOT NULL, PRIMARY KEY (md5, cue_ms));
CREATE TABLE IF NOT EXISTS maest (
    md5 BLOB NOT NULL, cue_ms INTEGER NOT NULL, activations BLOB NOT NULL, embedding BLOB NOT NULL,
    windows INTEGER NOT NULL, PRIMARY KEY (md5, cue_ms));
CREATE TABLE IF NOT EXISTS clap (
    md5 BLOB NOT NULL, cue_ms INTEGER NOT NULL, vector BLOB NOT NULL,
    PRIMARY KEY (md5, cue_ms));
CREATE TABLE IF NOT EXISTS pcm (
    relpath TEXT NOT NULL, size INTEGER NOT NULL, md5 BLOB NOT NULL, PRIMARY KEY (relpath, size));
CREATE TABLE IF NOT EXISTS mb (
    kind TEXT NOT NULL, key TEXT NOT NULL, mbid TEXT, genres TEXT NOT NULL, at INTEGER NOT NULL,
    PRIMARY KEY (kind, key));
CREATE TABLE IF NOT EXISTS blob (name TEXT PRIMARY KEY, data BLOB NOT NULL);
CREATE TABLE IF NOT EXISTS event (
    event_id TEXT PRIMARY KEY, ts INTEGER NOT NULL, md5 BLOB NOT NULL, cue_ms INTEGER NOT NULL,
    event_type TEXT NOT NULL, played_ratio_pct INTEGER, track_length_s INTEGER,
    is_organic INTEGER NOT NULL, session_id INTEGER, prev_md5 BLOB, prev_cue INTEGER,
    transition_type TEXT, recording_key BLOB, skip_reason TEXT);
CREATE TABLE IF NOT EXISTS event_more (  -- своя таблица: файл читают и серверы прежней версии, а они ждут в event ровно 14 столбцов
    event_id TEXT PRIMARY KEY, left_by TEXT, seeks_back INTEGER, seeks_forward INTEGER);
CREATE TABLE IF NOT EXISTS label (
    a_md5 BLOB NOT NULL, a_cue INTEGER NOT NULL, b_md5 BLOB NOT NULL, b_cue INTEGER NOT NULL,
    label TEXT NOT NULL, ts INTEGER NOT NULL,
    PRIMARY KEY (a_md5, a_cue, b_md5, b_cue));
"""

def index_files(cfg) -> dict[str, Path]:
    """Файлы, которые едут вместе с разбором: векторы текстов сценариев (без них сценарии не
    собрать без модели CLAP) и свои сценарии владельца."""
    return {"scenario_text.npz": Path(cfg.index_dir) / "laion-clap-music" / "scenario_text.npz",
            "maest_labels.json": Path(cfg.index_dir) / "maest_labels.json",
            "scenarios.toml": Path(cfg.data_dir) / "scenarios.toml"}


FEAT_COLS = ("version", "bpm", "bpm_conf", "bpm_src", "musical_key", "key_conf", "loudness_db",
             "onset_rate", "vocal_frac", "vocal_ratio")


def path(root: Path) -> Path:
    return Path(root) / DIRNAME / FILENAME


def _tidy(sc: sqlite3.Connection) -> None:
    """То же, что db.tidy_storage, в файле медиатеки: иначе убранное вернётся при следующем импорте, а файл не похудеет."""
    from arxnova.embed import MIX_SCHEME
    from arxnova.models import EFFNET

    with sc:
        changed = _facets_to_numbers(sc)
        changed += sc.execute("UPDATE maest SET embedding = x'' WHERE length(embedding) > 0").rowcount
        changed += sc.execute("""DELETE FROM emb WHERE model = ? AND EXISTS (
                                 SELECT 1 FROM emb m WHERE m.md5 = emb.md5 AND m.cue_ms = emb.cue_ms AND m.model = ?)""",
                              (EFFNET.id, f"{EFFNET.id}@{MIX_SCHEME}")).rowcount
    if changed:
        sc.execute("VACUUM")


def _facets_to_numbers(sc: sqlite3.Connection) -> int:
    """То же, что db.facets_to_numbers (решение 0014): голос по стему вокала — в feat_more, векторы фасетов — прочь.
    Саму таблицу facet не удаляем: её читают серверы прежней версии."""
    import numpy as np

    from arxnova import sound_tags

    rows = sc.execute("""SELECT f.md5, f.cue_ms, f.vector FROM facet f WHERE f.facet = 'vocal' AND NOT EXISTS (
                         SELECT 1 FROM feat_more m WHERE m.md5 = f.md5 AND m.cue_ms = f.cue_ms AND m.female_p IS NOT NULL)""").fetchall()
    rows = [(md5, cue, np.frombuffer(v, np.float32)) for md5, cue, v in rows if len(v) == sound_tags.VOICE_DIM * 4]
    if rows:
        p = sound_tags.female_probability(np.stack([v for *_, v in rows]))
        sc.executemany("INSERT INTO feat_more (md5, cue_ms, female_p) VALUES (?,?,?) "
                       "ON CONFLICT (md5, cue_ms) DO UPDATE SET female_p = excluded.female_p",
                       [(md5, cue, float(x)) for x, (md5, cue, _) in zip(p, rows)])
    return sc.execute("DELETE FROM facet").rowcount


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
            _tidy(sc)
            with sc:
                for name, fp in (files or {}).items():
                    if fp.exists():
                        data = fp.read_bytes()
                        old = sc.execute("SELECT data FROM blob WHERE name = ?", (name,)).fetchone()
                        if old is None or bytes(old[0]) != data:
                            sc.execute("INSERT OR REPLACE INTO blob VALUES (?,?)", (name, data))
            have = {t: set() for t in ("emb", "clap", "voc")}
            for md5, cue, model in sc.execute("SELECT md5, cue_ms, model FROM emb"):
                have["emb"].add((bytes(md5), cue, model))
            for md5, cue in sc.execute("SELECT md5, cue_ms FROM clap"):
                have["clap"].add((bytes(md5), cue))
            for md5, cue in sc.execute("SELECT md5, cue_ms FROM voc"):
                have["voc"].add((bytes(md5), cue))
            have["maest"] = {(bytes(r[0]), r[1]) for r in sc.execute("SELECT md5, cue_ms FROM maest")}
            feats = {(bytes(r[0]), r[1]): tuple(r[2:]) for r in
                     sc.execute(f"SELECT md5, cue_ms, {', '.join(FEAT_COLS)} FROM feat")}
            female = {(bytes(r[0]), r[1]): tuple(r[2:]) for r in sc.execute("SELECT md5, cue_ms, female_p, female_stem_p FROM feat_more")}
            from arxnova.embed import MIX_SCHEME
            from arxnova.models import EFFNET

            by_excerpts = f"{EFFNET.id}@{MIX_SCHEME}"
            replaced = {r[0] for r in conn.execute("SELECT track_id FROM embeddings WHERE model = ?", (by_excerpts,))}
            with sc:
                for tid, model, vector, styles, windows in conn.execute(
                        "SELECT track_id, model, vector, styles, windows FROM embeddings"):
                    k = keys.get(tid)
                    if model == EFFNET.id and tid in replaced:
                        continue  # иначе _tidy убирает его из файла, а следующая выгрузка снова кладёт
                    if k and (*k, model) not in have["emb"]:
                        sc.execute("INSERT OR REPLACE INTO emb VALUES (?,?,?,?,?,?)", (*k, model, vector, styles, windows))
                        added += 1
                for row in conn.execute(f"SELECT track_id, {', '.join(FEAT_COLS)} FROM track_features"):
                    k = keys.get(row[0])
                    # темп пересчитывается (по долям вместо ударных) — признаки сравниваем целиком
                    if k and feats.get(k) != tuple(row[1:]):
                        sc.execute(f"INSERT OR REPLACE INTO feat VALUES (?,?,{','.join('?' * len(FEAT_COLS))})",
                                   (*k, *row[1:]))
                        added += 1
                for tid, p, ps in conn.execute("""SELECT track_id, female_p, female_stem_p FROM track_features
                                                  WHERE female_p IS NOT NULL OR female_stem_p IS NOT NULL"""):
                    k = keys.get(tid)
                    if k and female.get(k) != (p, ps):
                        sc.execute("INSERT OR REPLACE INTO feat_more (md5, cue_ms, female_p, female_stem_p) VALUES (?,?,?,?)", (*k, p, ps))
                        added += 1
                for tid, acts, emb, windows in conn.execute("SELECT track_id, activations, embedding, windows FROM maest"):
                    k = keys.get(tid)
                    if k and k not in have["maest"]:
                        sc.execute("INSERT OR REPLACE INTO maest VALUES (?,?,?,?,?)", (*k, acts, emb, windows))
                        added += 1
                for tid, vector in conn.execute("SELECT track_id, vector FROM clap_vectors"):
                    k = keys.get(tid)
                    if k and k not in have["clap"]:
                        sc.execute("INSERT OR REPLACE INTO clap VALUES (?,?,?)", (*k, vector))
                        added += 1
                for tid, vector in conn.execute("SELECT track_id, vector FROM clap_voc"):
                    k = keys.get(tid)
                    if k and k not in have["voc"]:
                        sc.execute("INSERT OR REPLACE INTO voc VALUES (?,?,?)", (*k, vector))
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
                # ответы MusicBrainz (по именам — от путей не зависят): другой сервер не спрашивает снова
                from arxnova import mbgenres

                conn.execute(mbgenres.SCHEMA)
                for row in conn.execute("SELECT kind, key, mbid, genres, at FROM mb_cache"):
                    sc.execute("INSERT OR REPLACE INTO mb VALUES (?,?,?,?,?)", tuple(row))
                forgotten = [(r[0],) for r in conn.execute("SELECT event_id FROM forgotten_events")]
                sc.executemany("DELETE FROM event WHERE event_id = ?", forgotten)
                sc.executemany("DELETE FROM event_more WHERE event_id = ?", forgotten)
                have_ev = {r[0] for r in sc.execute("SELECT event_id FROM event")}
                for r in conn.execute(
                        """SELECT e.event_id, e.ts, e.track_id, e.event_type, e.played_ratio_pct, e.track_length_s,
                                  e.is_organic, e.session_id, e.prev_track_id, e.transition_type, e.recording_key,
                                  e.skip_reason, e.left_by, e.seeks_back, e.seeks_forward
                           FROM events e JOIN devices d ON d.id = e.device_id
                           WHERE d.role = 'owner' AND e.track_id IS NOT NULL"""):
                    k = keys.get(r[2])
                    if not k or r[0] in have_ev:
                        continue
                    prev = keys.get(r[8]) if r[8] is not None else None
                    sc.execute("INSERT OR IGNORE INTO event VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                               (r[0], r[1], *k, r[3], r[4], r[5], r[6], r[7],
                                prev[0] if prev else None, prev[1] if prev else None, r[9], r[10], r[11]))
                    if r[12] is not None or r[13] is not None or r[14] is not None:
                        sc.execute("INSERT OR IGNORE INTO event_more VALUES (?,?,?,?)", (r[0], r[12], r[13], r[14]))
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
            has_feat = {r[0] for r in conn.execute("SELECT track_id FROM track_features")}
            has_clap = {r[0] for r in conn.execute("SELECT track_id FROM clap_vectors")}
            from arxnova.embed import MIX_SCHEME
            from arxnova.models import EFFNET

            by_excerpts = f"{EFFNET.id}@{MIX_SCHEME}"
            in_file = {(bytes(r[0]), r[1]) for r in sc.execute("SELECT md5, cue_ms FROM emb WHERE model = ?", (by_excerpts,))}
            with conn:
                for md5, cue, model, vector, styles, windows in sc.execute("SELECT * FROM emb"):
                    t = by_key.get((bytes(md5), cue))
                    if model == EFFNET.id and ((bytes(md5), cue) in in_file or (t, by_excerpts) in has_emb):
                        continue  # вектор по целому треку заменён векторами по отрывкам (db.tidy_storage)
                    if t is not None and (t, model) not in has_emb:
                        conn.execute("INSERT OR REPLACE INTO embeddings (track_id, model, vector, styles, windows) "
                                     "VALUES (?,?,?,?,?)", (t, model, vector, styles, windows))
                        got.add(t)
                for row in sc.execute(f"SELECT md5, cue_ms, {', '.join(FEAT_COLS)} FROM feat"):
                    t = by_key.get((bytes(row[0]), row[1]))
                    if t is not None and t not in has_feat:
                        conn.execute(f"INSERT OR REPLACE INTO track_features (track_id, {', '.join(FEAT_COLS)}) "
                                     f"VALUES (?,{','.join('?' * len(FEAT_COLS))})", (t, *row[2:]))
                        got.add(t)
                # голос по стему вокала: числом (feat_more) или, в файле прежней версии, ещё вектором фасета
                no_female = {r[0] for r in conn.execute("SELECT track_id FROM track_features WHERE female_p IS NULL")}
                if no_female:
                    import numpy as np

                    from arxnova import sound_tags

                    for md5, cue, p in sc.execute("SELECT md5, cue_ms, female_p FROM feat_more WHERE female_p IS NOT NULL"):
                        t = by_key.get((bytes(md5), cue))
                        if t in no_female:
                            conn.execute("UPDATE track_features SET female_p = ? WHERE track_id = ?", (p, t))
                            no_female.discard(t)
                            got.add(t)
                    for md5, cue, p in sc.execute("SELECT md5, cue_ms, female_stem_p FROM feat_more WHERE female_stem_p IS NOT NULL"):
                        t = by_key.get((bytes(md5), cue))
                        if t is not None:
                            n = conn.execute("UPDATE track_features SET female_stem_p = ? WHERE track_id = ? AND female_stem_p IS NULL",
                                             (p, t)).rowcount
                            if n:
                                got.add(t)
                    old = [(by_key.get((bytes(md5), cue)), v)
                           for md5, cue, v in sc.execute("SELECT md5, cue_ms, vector FROM facet WHERE facet = 'vocal'")]
                    old = [(t, np.frombuffer(v, np.float32)) for t, v in old  # длина — до чтения: битая строка не роняет обход
                           if t in no_female and len(v) == sound_tags.VOICE_DIM * 4]
                    if old:
                        p = sound_tags.female_probability(np.stack([v for _, v in old]))
                        conn.executemany("UPDATE track_features SET female_p = ? WHERE track_id = ?",
                                         [(float(x), t) for x, (t, _) in zip(p, old)])
                        got.update(t for t, _ in old)
                has_maest = {r[0] for r in conn.execute("SELECT track_id FROM maest")}
                for md5, cue, acts, emb, windows in sc.execute("SELECT * FROM maest"):
                    t = by_key.get((bytes(md5), cue))
                    if t is not None and t not in has_maest:
                        conn.execute("INSERT OR REPLACE INTO maest (track_id, activations, embedding, windows) "
                                     "VALUES (?,?,?,?)", (t, acts, emb, windows))
                        got.add(t)
                for md5, cue, vector in sc.execute("SELECT * FROM clap"):
                    t = by_key.get((bytes(md5), cue))
                    if t is not None and t not in has_clap:
                        conn.execute("INSERT OR REPLACE INTO clap_vectors (track_id, vector) VALUES (?,?)", (t, vector))
                        got.add(t)
                has_voc = {r[0] for r in conn.execute("SELECT track_id FROM clap_voc")}
                for md5, cue, vector in sc.execute("SELECT * FROM voc"):
                    t = by_key.get((bytes(md5), cue))
                    if t is not None and t not in has_voc:
                        conn.execute("INSERT OR REPLACE INTO clap_voc (track_id, vector) VALUES (?,?)", (t, vector))
                        got.add(t)
                from arxnova import mbgenres

                conn.execute(mbgenres.SCHEMA)
                for row in sc.execute("SELECT kind, key, mbid, genres, at FROM mb"):
                    conn.execute("INSERT OR IGNORE INTO mb_cache VALUES (?,?,?,?,?)", tuple(row))
                evs = [r for r in sc.execute("SELECT * FROM event")]
                if evs:
                    have_ev = {r[0] for r in conn.execute("SELECT event_id FROM events")}
                    have_ev |= {r[0] for r in conn.execute("SELECT event_id FROM forgotten_events")}  # удалённое не возвращать
                    evs = [r for r in evs if r[0] not in have_ev and by_key.get((bytes(r[2]), r[3])) is not None]
                if evs:
                    dev = _carried_device(conn)
                    more = {r[0]: r[1:] for r in sc.execute("SELECT * FROM event_more")}
                    for (eid, ts, md5, cue, et, pct, length, organic, sess, pmd5, pcue, trans, rkey, reason) in evs:
                        left_by, back, forward = more.get(eid, (None, None, None))
                        t = by_key[(bytes(md5), cue)]
                        prev = by_key.get((bytes(pmd5), pcue)) if pmd5 is not None else None
                        conn.execute(
                            """INSERT OR IGNORE INTO events (event_id, device_id, ts, md5, cue_ms, track_id, event_type,
                               played_ratio_pct, track_length_s, is_organic, session_id, prev_track_id,
                               transition_type, recording_key, skip_reason, left_by, seeks_back, seeks_forward)
                               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                            (eid, dev, ts, md5, cue, t, et, pct, length, organic, sess, prev, trans, rkey, reason, left_by, back, forward))
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


def forget_lead_in(roots: list[Path], md5: bytes, first_ms: int) -> None:
    """Образ разрезан заново: первый трек теперь с нуля (scan._forget_lead_ins). Числа прежней паузы лежат под тем же
    ключом (md5, 0), и обход вернул бы их первому треку."""
    for root in roots:
        sc = _open(root, create=False)
        if sc is None:
            continue
        try:
            with sc:
                for table in ("emb", "facet", "feat", "feat_more", "voc", "maest", "clap"):
                    sc.execute(f"DELETE FROM {table} WHERE md5 = ? AND cue_ms IN (0, ?)", (md5, first_ms))
                sc.execute("UPDATE event SET cue_ms = 0 WHERE md5 = ? AND cue_ms = ?", (md5, first_ms))
        except sqlite3.Error:
            pass
        finally:
            sc.close()


def forget_events(roots: list[Path], ids: list[str]) -> None:
    """Удалённая история уходит и из файлов медиатек. Диск не подключён — уберёт следующий выгруз."""
    for root in roots:
        sc = _open(root, create=False)
        if sc is None:
            continue
        try:
            with sc:
                sc.executemany("DELETE FROM event WHERE event_id = ?", [(i,) for i in ids])
                sc.executemany("DELETE FROM event_more WHERE event_id = ?", [(i,) for i in ids])
        except (sqlite3.Error, OSError):
            pass
        finally:
            sc.close()


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


SETTINGS = ("boldness", "bridge_pct", "gate_forbid")


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
            # чужие ключи (прежний gate_pct) не считаются различием: иначе вечная перезапись на каждом обходе
            if untouched and theirs and any(k in theirs and theirs[k] != mine[k] for k in SETTINGS):
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
