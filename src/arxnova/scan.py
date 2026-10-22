"""Обход медиатеки: каждый FLAC → ключ и теги → база. Повторный обход трогает только
новые и изменённые файлы; работа возобновляемая, единица работы — файл (ФТ-2.6)."""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from arxnova import cue, flac, keys

VARIOUS = {"various artists", "various", "va", "v.a.", "сборник", "разные исполнители"}


@dataclass
class ScanReport:
    seen: int = 0
    added: int = 0
    unchanged: int = 0
    duplicates: int = 0  # тот же звук по другому пути
    pcm_md5: int = 0  # MD5 в файле был нулевым — посчитали по звуку
    removed: int = 0
    errors: list[tuple[str, str]] = field(default_factory=list)


def iter_flac(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for name in filenames:
            if name.lower().endswith(".flac") and not name.startswith("._"):
                yield Path(dirpath) / name


def _is_compilation(info: flac.FlacInfo) -> bool:
    if info.tag("COMPILATION") in ("1", "true", "yes"):
        return True
    return keys.normalize(info.tag("ALBUMARTIST")) in VARIOUS


def _sibling_cue_sig(path: Path) -> str | None:
    """Подпись .cue рядом с файлом (имя, размер, время) — без чтения FLAC. Меняется, когда
    лист появился, поменялся или исчез: тогда файл переразбирается."""
    found = cue.find_for(path)
    return found[2] if found else None


def scan(conn: sqlite3.Connection, roots: list[Path], progress=None) -> ScanReport:
    rep = ScanReport()
    known = {
        row["path"]: (row["size"], row["mtime_ns"], row["cue_sig"])
        for row in conn.execute("SELECT path, size, mtime_ns, cue_sig FROM files")
    }
    present: set[str] = set()

    def changed():
        for root in roots:
            for path in iter_flac(root):
                rep.seen += 1
                p = str(path)
                present.add(p)
                st = path.stat()
                k = known.get(p)
                if k and k[:2] == (st.st_size, st.st_mtime_ns):
                    sig = _sibling_cue_sig(path)
                    # встроенный лист живёт в самом файле — раз файл не менялся, не менялся и он
                    if sig == k[2] or (sig is None and k[2] == "embedded"):
                        rep.unchanged += 1
                        continue
                yield path, st

    # Тяжёлое (заголовок, MD5 звука, лист CUE) — в отдельном потоке на файл вперёд, запись в базу —
    # по порядку в этом потоке: чтение следующего файла идёт, пока пишется предыдущий
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=SCAN_WORKERS) as pool:
        window: list = []
        it = changed()

        def fill():
            while len(window) < SCAN_WORKERS * 2:
                nxt = next(it, None)
                if nxt is None:
                    return
                window.append((nxt[0], nxt[1], pool.submit(_probe, nxt[0])))

        fill()
        while window:
            path, st, fut = window.pop(0)
            fill()
            try:
                _index_file(conn, path, st, rep, fut.result(), moved=str(path) not in known)
            except (flac.FlacError, OSError, RuntimeError) as e:
                rep.errors.append((str(path), str(e)))
            if progress:
                progress(rep)
    gone = [p for p in known if p not in present]
    with conn:
        for p in gone:
            conn.execute("DELETE FROM files WHERE path = ?", (p,))
        # трек без единого файла больше не в медиатеке (треки образа держатся за файл через MD5)
        conn.execute(
            """DELETE FROM tracks WHERE md5 NOT IN (
                   SELECT r.md5 FROM files f JOIN tracks r ON r.id = f.track_id)"""
        )
    rep.removed = len(gone)
    return rep


def _upsert_track(conn, rep, *, md5, source, info, cue_ms, cue_end_ms, duration, title, artist,
                  albumartist, album, date, genre, tracknumber, compilation) -> int:
    basis = keys.recording_basis(
        mb_recording=info.tag("MUSICBRAINZ_TRACKID") if cue_ms == 0 and cue_end_ms < 0 else None,
        artist=artist, title=title,
    )
    known = conn.execute(
        "SELECT recording_key, duration FROM tracks WHERE recording_basis = ? AND NOT (md5 = ? AND cue_ms = ?)",
        (basis, md5, cue_ms),
    ).fetchall()
    rec = keys.match_recording(basis, duration, known) or keys.new_recording_key(basis, duration)
    row = conn.execute("SELECT id FROM tracks WHERE md5 = ? AND cue_ms = ?", (md5, cue_ms)).fetchone()
    values = (rec, basis, info.sample_rate, info.bits, info.channels, duration, cue_end_ms, artist,
              albumartist, album, title, date, genre, tracknumber, info.tag("MUSICBRAINZ_TRACKID"),
              info.tag("MUSICBRAINZ_ALBUMID"), int(compilation))
    if row:
        conn.execute(
            """UPDATE tracks SET recording_key=?, recording_basis=?, sample_rate=?, bits=?, channels=?,
                   duration=?, cue_end_ms=?, artist=?, albumartist=?, album=?, title=?, date=?, genre_raw=?,
                   tracknumber=?, mb_recording=?, mb_album=?, compilation=? WHERE id=?""",
            (*values, row["id"]),
        )
        return row["id"]
    cur = conn.execute(
        """INSERT INTO tracks (md5, cue_ms, md5_source, recording_key, recording_basis, sample_rate, bits,
               channels, duration, cue_end_ms, artist, albumartist, album, title, date, genre_raw,
               tracknumber, mb_recording, mb_album, compilation)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (md5, cue_ms, source, *values),
    )
    rep.added += 1
    # события, пришедшие раньше трека, связываются с ним теперь (ТЗ, §7: track_id null)
    conn.execute(
        "UPDATE events SET track_id = ?, recording_key = ? WHERE md5 = ? AND cue_ms = ? AND track_id IS NULL",
        (cur.lastrowid, rec, md5, cue_ms),
    )
    return cur.lastrowid


# Один поток чтения: на жёстком диске 1 поток — 80 МБ/с, 2 — 57, 3 — 40 (головка мечется между
# файлами; у половины веб-рипов MD5 в заголовке нулевой, и файл декодируется целиком)
SCAN_WORKERS = 1


def _probe(path: Path):
    """Всё, что читает файл, — без базы (идёт в пуле потоков)."""
    info = flac.read_flac(path)
    md5, source = info.md5, "streaminfo"
    if info.md5_is_zero:
        md5, source = flac.pcm_md5(path, info.bits), "pcm"
    return info, md5, source, cue.find_for(path, info.tag("CUESHEET"))


def _index_file(conn: sqlite3.Connection, path: Path, st: os.stat_result, rep: ScanReport, probe=None,
                moved: bool = False):
    info, md5, source, found = probe or _probe(path)
    if source == "pcm":
        rep.pcm_md5 += 1
    with conn:
        # файл или его лист изменились — границы треков могли сдвинуться, анализ звука заново.
        # Новый путь (медиатеку перенесли на другой диск, скопировали альбом) — тот же звук и те же
        # треки по (md5, начало): разбор остаётся. Раньше перенос 472 треков на внешний диск
        # выбрасывал их разбор целиком (22.10.2026)
        if not moved:
            conn.execute("DELETE FROM embeddings WHERE track_id IN (SELECT id FROM tracks WHERE md5 = ?)", (md5,))
        dup = conn.execute(
            "SELECT 1 FROM files f JOIN tracks r ON r.id = f.track_id WHERE r.md5 = ? AND f.path != ?",
            (md5, str(path)),
        ).fetchone()
        if dup and not conn.execute("SELECT 1 FROM files WHERE path = ?", (str(path),)).fetchone():
            rep.duplicates += 1
        if found:
            sheet, tracks, sig = found
            total_ms = int(info.duration * 1000)
            album = sheet.title or info.tag("ALBUM")
            album_artist = sheet.performer or info.tag("ALBUMARTIST") or info.tag("ARTIST")
            performers = {t.performer for t in tracks if t.performer}
            compilation = len(performers) >= 4 or keys.normalize(album_artist) in VARIOUS
            starts = [t.start_ms for t in tracks]
            first_id = None
            for i, t in enumerate(tracks):
                end = starts[i + 1] if i + 1 < len(tracks) else total_ms
                tid = _upsert_track(
                    conn, rep, md5=md5, source=source, info=info, cue_ms=t.start_ms,
                    cue_end_ms=end if i + 1 < len(tracks) else -1,
                    duration=max(0.0, (end - t.start_ms) / 1000.0),
                    title=t.title or f"Track {t.number}", artist=t.performer or album_artist,
                    albumartist=sheet.performer or info.tag("ALBUMARTIST"), album=album,
                    date=sheet.date or info.tag("DATE"), genre=sheet.genre or info.tag("GENRE"),
                    tracknumber=str(t.number), compilation=compilation,
                )
                if t.start_ms == 0:
                    first_id = tid
            if first_id is None:  # первый трек начинается не с нуля (скрытый пролог) — держатель файла
                first_id = _upsert_track(
                    conn, rep, md5=md5, source=source, info=info, cue_ms=0, cue_end_ms=starts[0],
                    duration=starts[0] / 1000.0, title=f"Track 0", artist=album_artist,
                    albumartist=sheet.performer, album=album, date=sheet.date, genre=sheet.genre,
                    tracknumber="0", compilation=compilation,
                )
                starts = [0] + starts
            q = ",".join("?" * len(starts))
            conn.execute(f"DELETE FROM tracks WHERE md5 = ? AND cue_ms NOT IN ({q})", (md5, *starts))
            track_id = first_id
        else:
            sig = None
            track_id = _upsert_track(
                conn, rep, md5=md5, source=source, info=info, cue_ms=0, cue_end_ms=-1,
                duration=info.duration, title=info.tag("TITLE"), artist=info.tag("ARTIST"),
                albumartist=info.tag("ALBUMARTIST"), album=info.tag("ALBUM"), date=info.tag("DATE"),
                genre=info.tag("GENRE"), tracknumber=info.tag("TRACKNUMBER"),
                compilation=_is_compilation(info),
            )
            # лист убрали — треки образа уходят, остаётся файл целиком
            conn.execute("DELETE FROM tracks WHERE md5 = ? AND cue_ms != 0", (md5,))
        conn.execute(
            """INSERT INTO files (path, track_id, size, mtime_ns, cue_sig) VALUES (?,?,?,?,?)
               ON CONFLICT(path) DO UPDATE SET track_id=excluded.track_id,
                   size=excluded.size, mtime_ns=excluded.mtime_ns, cue_sig=excluded.cue_sig""",
            (str(path), track_id, st.st_size, st.st_mtime_ns, sig),
        )
