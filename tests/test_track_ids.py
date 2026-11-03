"""Номер удалённого трека никому не достаётся, события на него не переходят к чужому звуку (03.11.2026)."""

import shutil
import sqlite3
import subprocess

import pytest

from arxnova import config, db, scan

needs_ffmpeg = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="нужен ffmpeg")


def flac(path, freq):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"sine=frequency={freq}:duration=1",
                    "-c:a", "flac", "-metadata", f"title=T{freq}", "-metadata", "artist=A", str(path)], check=True)
    return path


def library(tmp_path):
    lib = tmp_path / "lib"
    lib.mkdir()
    cfg = config.Config(data_dir=tmp_path / "data", libraries=[lib])
    cfg.ensure_dirs()
    return lib, db.connect(cfg.db_path)


def track_id(conn, title):
    r = conn.execute("SELECT id FROM tracks WHERE title = ?", (title,)).fetchone()
    return r["id"] if r else None


def add_event(conn, tid, event_id="e1"):
    row = conn.execute("SELECT md5, cue_ms, recording_key FROM tracks WHERE id = ?", (tid,)).fetchone()
    with conn:
        conn.execute("INSERT OR IGNORE INTO devices (id, name, token_hash, role, created_at) VALUES (1, 'd', 'x', 'owner', 0)")
        conn.execute("INSERT INTO events (event_id, device_id, ts, md5, cue_ms, track_id, event_type, recording_key) "
                     "VALUES (?, 1, 1, ?, ?, ?, 'skip', ?)", (event_id, row["md5"], row["cue_ms"], tid, row["recording_key"]))


@needs_ffmpeg
def test_deleted_track_number_is_not_reused_and_history_does_not_move(tmp_path):
    lib, conn = library(tmp_path)
    files = {"T300": flac(lib / "a.flac", 300), "T400": flac(lib / "b.flac", 400)}
    scan.scan(conn, [lib])
    # удаляем трек с самым большим номером: его SQLite выдал бы снова. Какой он, зависит от порядка обхода на этом диске
    top = conn.execute("SELECT id, title FROM tracks ORDER BY id DESC LIMIT 1").fetchone()
    old, title, src = top["id"], top["title"], files[top["title"]]
    keep = tmp_path / "gone.keep"
    shutil.copy(src, keep)
    add_event(conn, old)

    src.unlink()
    scan.scan(conn, [lib])
    assert track_id(conn, title) is None
    e = conn.execute("SELECT track_id FROM events WHERE event_id = 'e1'").fetchone()
    assert e["track_id"] is None, "событие удалённого трека не должно держать его номер"

    flac(lib / "c.flac", 500)  # новый звук
    scan.scan(conn, [lib])
    assert track_id(conn, "T500") > old
    assert conn.execute("SELECT track_id FROM events WHERE event_id = 'e1'").fetchone()["track_id"] is None

    shutil.copy(keep, src)  # тот же звук вернулся: история снова с ним, номер новый
    scan.scan(conn, [lib])
    back = track_id(conn, title)
    assert back not in (old, track_id(conn, "T500"))
    assert conn.execute("SELECT track_id FROM events WHERE event_id = 'e1'").fetchone()["track_id"] == back


@needs_ffmpeg
def test_migration_repairs_events_on_someone_elses_track(tmp_path):
    lib, conn = library(tmp_path)
    flac(lib / "a.flac", 300)
    flac(lib / "b.flac", 400)
    scan.scan(conn, [lib])
    a, b = track_id(conn, "T300"), track_id(conn, "T400")
    add_event(conn, a, "right")
    add_event(conn, a, "foreign")
    with conn:
        # как до правки: номер занят другим звуком
        conn.execute("UPDATE events SET track_id = ? WHERE event_id = 'foreign'", (b,))
        conn.execute("UPDATE events SET prev_track_id = 99999 WHERE event_id = 'right'")
        conn.execute("DROP TRIGGER track_deleted")
        conn.execute("DROP TABLE track_id_floor")
        conn.execute("DROP INDEX events_prev_track")
        conn.execute(f"PRAGMA user_version = {len(db.MIGRATIONS) - 1}")
    db.migrate(conn)
    rows = {r["event_id"]: r for r in conn.execute("SELECT * FROM events")}
    assert rows["right"]["track_id"] == a and rows["foreign"]["track_id"] is None
    assert rows["right"]["prev_track_id"] is None
    assert len(rows) == 2, "строки событий не удаляются"
    assert conn.execute("SELECT id FROM track_id_floor").fetchone()[0] == max(a, b)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == len(db.MIGRATIONS)


@needs_ffmpeg
def test_cue_region_removed_keeps_event_detached(tmp_path):
    """Лист убрали: треки образа ушли, их события не должны достаться тому, кто займёт номер."""
    lib, conn = library(tmp_path)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=frequency=300:duration=8",
                    "-c:a", "flac", str(lib / "img.flac")], check=True)
    sheet = lib / "img.cue"
    sheet.write_text('FILE "img.flac" WAVE\n  TRACK 01 AUDIO\n    TITLE "P1"\n    INDEX 01 00:00:00\n'
                     '  TRACK 02 AUDIO\n    TITLE "P2"\n    INDEX 01 00:04:00\n')
    scan.scan(conn, [lib])
    p2 = track_id(conn, "P2")
    add_event(conn, p2, "cue")
    sheet.unlink()
    scan.scan(conn, [lib])
    assert track_id(conn, "P2") is None
    assert conn.execute("SELECT track_id FROM events WHERE event_id = 'cue'").fetchone()["track_id"] is None


def test_old_database_gets_the_migration(tmp_path):
    conn = db.connect(tmp_path / "x.sqlite3")
    assert conn.execute("PRAGMA user_version").fetchone()[0] == len(db.MIGRATIONS)
    assert isinstance(conn.execute("SELECT id FROM track_id_floor").fetchone()[0], int)
    assert isinstance(conn, sqlite3.Connection)
