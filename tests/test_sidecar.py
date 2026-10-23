"""Разбор, история и настройки едут вместе с медиатекой (sidecar): диск подключили к другому
серверу — ничего не разбирается заново и вкус владельца на месте."""

import shutil

import pytest

from arxnova import config, db, scan, sidecar

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
    cfg_a.boldness, cfg_a.gate_pct = 0.8, 95.0
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
    assert (cfg_b.boldness, cfg_b.gate_pct) == (0.8, 95.0)
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
