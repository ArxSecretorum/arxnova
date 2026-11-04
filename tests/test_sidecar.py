"""Разбор, история и настройки едут вместе с медиатекой (sidecar): диск подключили к другому
серверу — ничего не разбирается заново и вкус владельца на месте."""

import shutil
import sqlite3
import time

import pytest
from fastapi.testclient import TestClient

from arxnova import auth, config, db, scan, sidecar
from arxnova.server import create_app
from tests.test_flac_keys import make_flac, needs_ffmpeg, zero_md5


def _server(tmp_path, name, lib):
    cfg = config.Config(data_dir=tmp_path / name, libraries=[lib])
    cfg.ensure_dirs()
    conn = db.connect(cfg.db_path)
    scan.scan(conn, [lib])
    return cfg, conn


@needs_ffmpeg
def test_analysis_history_settings_travel_with_library(tmp_path, monkeypatch):
    lib = tmp_path / "disk" / "Music"
    lib.mkdir(parents=True)
    make_flac(lib / "a.flac", freq=300.0, ARTIST="A", TITLE="One")
    make_flac(lib / "b.flac", freq=500.0, ARTIST="B", TITLE="Two")
    src = make_flac(tmp_path / "c_src.flac", freq=700.0, ARTIST="C", TITLE="Three")
    (lib / "sub").mkdir()
    zero_md5(src, lib / "sub" / "c.flac")  # веб-рип: MD5 в заголовке не записан
    cfg_a, a = _server(tmp_path, "server_a", lib)
    ids = [r[0] for r in a.execute("SELECT id FROM tracks ORDER BY id")]
    with a:
        for t in ids:
            a.execute("INSERT INTO embeddings VALUES (?,?,?,?,?)", (t, "effnet-discogs-1", b"v" * 8, b"s" * 8, 3))
            a.execute("INSERT INTO facet_vectors VALUES (?,?,?,?)", (t, "vocal", "effnet-discogs-1", b"f" * 8))
            a.execute("INSERT INTO track_features (track_id, version, bpm, bpm_src) VALUES (?,1,128.0,'beat_this')", (t,))
            a.execute("INSERT INTO clap_vectors VALUES (?,?)", (t, b"c" * 8))
            a.execute("INSERT INTO maest VALUES (?,?,?,?)", (t, b"a" * 8, b"e" * 8, 3))
        a.execute("INSERT INTO devices (name, role, token_hash) VALUES ('M9', 'owner', x'01')")
        a.execute("INSERT INTO devices (name, role, token_hash) VALUES ('guest', 'guest', x'02')")
        md5 = {t: bytes(m) for t, m in a.execute("SELECT id, md5 FROM tracks")}
        a.execute("INSERT INTO events (event_id, device_id, ts, md5, track_id, event_type, prev_track_id) "
                  "VALUES ('e1', 1, 1000, ?, ?, 'like', ?)", (md5[ids[0]], ids[0], ids[1]))
        a.execute("INSERT INTO events (event_id, device_id, ts, md5, track_id, event_type) "
                  "VALUES ('g1', 2, 2000, ?, ?, 'listen')", (md5[ids[1]], ids[1]))
        a.execute("INSERT INTO bridge_labels VALUES (?,?, 'good', 5)", (ids[0], ids[1]))
    (cfg_a.data_dir / "scenarios.toml").write_text("[my]\ntitle = 'Моё'\n")
    cfg_a.boldness, cfg_a.bridge_pct = 0.8, 70.0
    cfg_a.save()
    assert sidecar.export(a, [lib], sidecar.index_files(cfg_a)) > 0
    sidecar.sync_settings(cfg_a, [lib])

    # тот же диск — на другом компьютере, по другому пути
    moved = tmp_path / "other_mount" / "Music"
    shutil.copytree(lib, moved)
    # MD5 звука уже посчитан сервером A — B не декодирует файл заново
    monkeypatch.setattr(scan.flac, "pcm_md5", lambda *a, **k: pytest.fail("файл декодирован заново"))
    cfg_b, b = _server(tmp_path, "server_b", moved)
    assert b.execute("SELECT count(*) FROM tracks WHERE md5_source = 'pcm'").fetchone()[0] == 1
    got = sidecar.import_missing(b, [moved], sidecar.index_files(cfg_b))
    assert got == 3
    for table in ("embeddings", "facet_vectors", "track_features", "clap_vectors", "maest"):
        assert b.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == 3, table
    assert b.execute("SELECT bpm, bpm_src FROM track_features LIMIT 1").fetchone()[:] == (128.0, "beat_this")
    # история владельца — да (на служебном устройстве владельца), гостя — нет
    ev = b.execute("SELECT e.event_id, d.role, d.revoked, e.prev_track_id IS NOT NULL FROM events e "
                   "JOIN devices d ON d.id = e.device_id").fetchall()
    assert [tuple(r) for r in ev] == [("e1", "owner", 1, 1)]
    assert b.execute("SELECT count(*) FROM bridge_labels").fetchone()[0] == 1
    assert (cfg_b.data_dir / "scenarios.toml").read_text().startswith("[my]")
    assert sidecar.sync_settings(cfg_b, [moved])
    assert (cfg_b.boldness, cfg_b.bridge_pct) == (0.8, 70.0)
    # повтор ничего не дублирует, а экспорт обратно не плодит события
    assert sidecar.import_missing(b, [moved], sidecar.index_files(cfg_b)) == 0
    assert b.execute("SELECT count(*) FROM events").fetchone()[0] == 1
    sidecar.export(b, [moved], sidecar.index_files(cfg_b))
    sc = __import__("sqlite3").connect(sidecar.path(moved))
    assert sc.execute("SELECT count(*) FROM event").fetchone()[0] == 1


def test_readonly_library_is_skipped(tmp_path):
    """Медиатека только для чтения (сетевой диск) — без файла разбора и без ошибок."""
    cfg = config.Config(data_dir=tmp_path / "d", libraries=[tmp_path / "nope" / "x"])
    cfg.ensure_dirs()
    conn = db.connect(cfg.db_path)
    assert sidecar.import_missing(conn, cfg.libraries) == 0
    assert sidecar.export(conn, cfg.libraries) == 0


def _forget_setup(tmp_path):
    """Сервер с одним событием владельца, уже выгруженным в файл медиатеки."""
    lib = tmp_path / "disk" / "Music"
    lib.mkdir(parents=True)
    make_flac(lib / "a.flac", freq=300.0, ARTIST="A", TITLE="One")
    cfg, conn = _server(tmp_path, "server", lib)
    client = TestClient(create_app(cfg))
    tok = client.post("/v1/pair", json={"code": auth.new_pair_code(conn), "name": "M9"}).json()["token"]
    h = {"Authorization": f"Bearer {tok}"}
    tid = conn.execute("SELECT id FROM tracks").fetchone()[0]
    client.post("/v1/events", headers=h, json={"events": [{"event_id": "forget-me-0001", "ts": 1, "track_id": tid, "event_type": "skip"}]})
    sidecar.export(conn, [lib])
    assert _in_file(lib) == 1
    return lib, conn, client, h


def _in_file(lib) -> int:
    sc = sqlite3.connect(sidecar.path(lib))
    try:
        return sc.execute("SELECT count(*) FROM event").fetchone()[0]
    finally:
        sc.close()


@needs_ffmpeg
def test_deleted_history_leaves_library_file_and_does_not_return(tmp_path):
    """«Удалить мою историю»: событие уходит и из файла медиатеки, иначе обход вернёт его (03.11.2026)."""
    lib, conn, client, h = _forget_setup(tmp_path)
    assert client.delete("/v1/devices/me/history", headers=h).json()["deleted"] == 1
    deadline = time.monotonic() + 5
    while _in_file(lib) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert _in_file(lib) == 0
    sidecar.import_missing(conn, [lib])
    assert conn.execute("SELECT count(*) FROM events").fetchone()[0] == 0


@needs_ffmpeg
def test_history_deleted_while_disk_is_away_does_not_return(tmp_path, monkeypatch):
    """Диск не подключён, пока историю удаляют: обход её не вернёт, а выгруз уберёт из файла."""
    lib, conn, client, h = _forget_setup(tmp_path)
    with monkeypatch.context() as away:
        away.setattr(sidecar, "_open", lambda *a, **k: None)
        assert client.delete("/v1/devices/me/history", headers=h).json()["deleted"] == 1
        time.sleep(0.2)  # удаление из файла идёт в своём потоке
    assert _in_file(lib) == 1, "диска не было: файл на месте"
    sidecar.import_missing(conn, [lib])
    assert conn.execute("SELECT count(*) FROM events").fetchone()[0] == 0
    sidecar.export(conn, [lib])
    assert _in_file(lib) == 0
