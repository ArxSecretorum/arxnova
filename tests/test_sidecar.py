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
            a.execute("INSERT INTO track_features (track_id, version, bpm, bpm_src, female_p, female_stem_p) "
                      "VALUES (?,1,128.0,'beat_this',0.8,0.7)", (t,))
            a.execute("INSERT INTO clap_voc VALUES (?,?)", (t, b"w" * 8))
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
    for table in ("embeddings", "track_features", "clap_vectors", "clap_voc", "maest"):
        assert b.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == 3, table
    assert b.execute("SELECT bpm, bpm_src, female_p, female_stem_p FROM track_features LIMIT 1").fetchone()[:] == (128.0, "beat_this", 0.8, 0.7)
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


def _library_with_vectors(tmp_path):
    """Сервер и медиатека: у трека вектор по целому файлу, по отрывкам и эмбеддинг MAEST, как в базах до решения 0013."""
    from arxnova import db
    from arxnova.embed import MIX_SCHEME
    from arxnova.models import EFFNET

    lib = tmp_path / "Music"
    lib.mkdir()
    make_flac(lib / "a.flac", freq=300.0, ARTIST="A", TITLE="One")
    make_flac(lib / "b.flac", freq=500.0, ARTIST="B", TITLE="Two")
    cfg, conn = _server(tmp_path, "server", lib)
    ids = [r[0] for r in conn.execute("SELECT id FROM tracks")]
    mixed = f"{EFFNET.id}@{MIX_SCHEME}"
    with conn:
        for t in ids:
            conn.execute("INSERT INTO embeddings VALUES (?,?,?,?,?)", (t, EFFNET.id, b"v" * 8, b"s" * 8, 3))
            conn.execute("INSERT INTO embeddings VALUES (?,?,?,?,?)", (t, mixed, b"w" * 8, b"s" * 8, 3))
            conn.execute("INSERT INTO maest VALUES (?,?,?,?)", (t, b"a" * 8, b"e" * 8, 3))
    return lib, cfg, conn, db, EFFNET, mixed


@needs_ffmpeg
def test_vectors_removed_by_cleanup_do_not_come_back_from_the_library_file(tmp_path):
    """Решение 0013: db.tidy_storage убирает векторы по целому треку, а файл медиатеки возвращал их при каждом запуске: база
    толкалась туда-обратно, VACUUM и пересборка индекса на каждый старт."""
    lib, cfg, conn, db, EFFNET, mixed = _library_with_vectors(tmp_path)
    assert sidecar.export(conn, [lib], sidecar.index_files(cfg)) > 0
    assert db.tidy_storage(conn) > 0
    assert sidecar.import_missing(conn, [lib], sidecar.index_files(cfg)) == 0
    assert conn.execute("SELECT count(*) FROM embeddings WHERE model = ?", (EFFNET.id,)).fetchone()[0] == 0
    assert db.tidy_storage(conn) == 0, "второй запуск не должен ничего убирать"
    sidecar.export(conn, [lib], sidecar.index_files(cfg))
    sc = sqlite3.connect(sidecar.path(lib))
    assert sc.execute("SELECT count(*) FROM emb WHERE model = ?", (EFFNET.id,)).fetchone()[0] == 0, "и файл медиатеки худеет"
    assert sc.execute("SELECT count(*) FROM emb WHERE model = ?", (mixed,)).fetchone()[0] == 2
    assert sc.execute("SELECT count(*) FROM maest WHERE length(embedding) > 0").fetchone()[0] == 0


@needs_ffmpeg
def test_whole_track_vector_still_comes_when_nothing_replaces_it(tmp_path):
    """Старый файл медиатеки на новом сервере: отрывков нет, вектор по целому файлу пока единственный."""
    lib, cfg, conn, db, EFFNET, mixed = _library_with_vectors(tmp_path)
    with conn:
        conn.execute("DELETE FROM embeddings WHERE model = ?", (mixed,))
    assert sidecar.export(conn, [lib], sidecar.index_files(cfg)) > 0
    with conn:
        conn.execute("DELETE FROM embeddings")
    assert sidecar.import_missing(conn, [lib], sidecar.index_files(cfg)) == 2
    assert conn.execute("SELECT count(*) FROM embeddings WHERE model = ?", (EFFNET.id,)).fetchone()[0] == 2


@needs_ffmpeg
def test_how_a_track_was_left_travels_with_the_library(tmp_path):
    lib, cfg, conn, db, EFFNET, mixed = _library_with_vectors(tmp_path)
    tid = conn.execute("SELECT id FROM tracks LIMIT 1").fetchone()[0]
    md5 = conn.execute("SELECT md5 FROM tracks WHERE id = ?", (tid,)).fetchone()[0]
    with conn:
        conn.execute("INSERT INTO devices (name, role, token_hash) VALUES ('M9', 'owner', x'01')")
        conn.execute("INSERT INTO events (event_id, device_id, ts, md5, track_id, event_type, left_by, seeks_back, seeks_forward) "
                     "VALUES ('moved-001', 1, 1000, ?, ?, 'skip', 'back', 2, 1)", (md5, tid))
    sidecar.export(conn, [lib], sidecar.index_files(cfg))
    with conn:
        conn.execute("DELETE FROM events")
    assert sidecar.import_missing(conn, [lib], sidecar.index_files(cfg)) >= 1
    row = conn.execute("SELECT left_by, seeks_back, seeks_forward FROM events WHERE event_id = 'moved-001'").fetchone()
    assert tuple(row) == ("back", 2, 1)


@needs_ffmpeg
def test_library_file_stays_readable_for_a_server_of_the_previous_version(tmp_path):
    """Второй сервер старой версии на том же диске или откат: он читает из event ровно 14 столбцов. Новые поля — в своей таблице."""
    lib, cfg, conn, db, EFFNET, mixed = _library_with_vectors(tmp_path)
    tid = conn.execute("SELECT id FROM tracks LIMIT 1").fetchone()[0]
    md5 = conn.execute("SELECT md5 FROM tracks WHERE id = ?", (tid,)).fetchone()[0]
    with conn:
        conn.execute("INSERT INTO devices (name, role, token_hash) VALUES ('M9', 'owner', x'01')")
        conn.execute("INSERT INTO events (event_id, device_id, ts, md5, track_id, event_type, left_by) VALUES ('kept-0001', 1, 1000, ?, ?, 'skip', 'back')", (md5, tid))
    sidecar.export(conn, [lib], sidecar.index_files(cfg))
    sc = sqlite3.connect(sidecar.path(lib))
    assert len(sc.execute("SELECT * FROM event").fetchone()) == 14
    assert sc.execute("SELECT left_by FROM event_more").fetchone()[0] == "back"
    sc.execute("DROP TABLE event_more")  # файл от прежней версии: таблицы нет
    sc.commit()
    sc.close()
    with conn:
        conn.execute("DELETE FROM events")
    assert sidecar.import_missing(conn, [lib], sidecar.index_files(cfg)) >= 1
    assert conn.execute("SELECT event_type, left_by FROM events WHERE event_id = 'kept-0001'").fetchone()[:] == ("skip", None)


@needs_ffmpeg
def test_cleaning_the_library_file_does_not_repeat_when_some_track_has_no_new_vector(tmp_path):
    """Уборка в базе ждёт, пока вектор по отрывкам появится у всех. Выгрузка не должна снова класть то, что уборка файла только что убрала."""
    lib, cfg, conn, db, EFFNET, mixed = _library_with_vectors(tmp_path)
    ids = [r[0] for r in conn.execute("SELECT id FROM tracks")]
    with conn:
        conn.execute("DELETE FROM embeddings WHERE track_id = ? AND model = ?", (ids[0], mixed))
    assert db.tidy_storage(conn) > 0
    assert conn.execute("SELECT count(*) FROM embeddings WHERE model = ?", (EFFNET.id,)).fetchone()[0] == 2, "вектор по целому треку ещё нужен"
    sidecar.export(conn, [lib], sidecar.index_files(cfg))
    sidecar.export(conn, [lib], sidecar.index_files(cfg))
    sc = sqlite3.connect(sidecar.path(lib))
    assert sc.execute("SELECT count(*) FROM emb WHERE model = ?", (EFFNET.id,)).fetchone()[0] == 1


def _voice_vectors(n):
    import numpy as np

    rng = np.random.default_rng(4)
    return rng.normal(size=(n, 1280)).astype(np.float32)


@needs_ffmpeg
def test_facet_vectors_become_voice_number(tmp_path):
    """Решение 0014: вектор стема вокала переводится в вероятность женского голоса, затем все векторы фасетов удаляются —
    и в базе, и в файле медиатеки, причём старый файл с векторами даёт число и новому серверу."""
    import numpy as np

    from arxnova import sound_tags

    lib = tmp_path / "disk" / "Music"
    lib.mkdir(parents=True)
    make_flac(lib / "a.flac", freq=300.0, ARTIST="A", TITLE="One")
    make_flac(lib / "b.flac", freq=500.0, ARTIST="B", TITLE="Two")
    cfg_a, a = _server(tmp_path, "server_a", lib)
    ids = [r[0] for r in a.execute("SELECT id FROM tracks ORDER BY id")]
    voc = _voice_vectors(len(ids))
    # файл медиатеки, записанный сервером прежней версии: векторы фасетов есть, числа голоса нет
    sc = sidecar._open(lib, create=True)
    md5 = {t: bytes(m) for t, m in a.execute("SELECT id, md5 FROM tracks")}
    with sc:
        for t, v in zip(ids, voc):
            sc.execute("INSERT INTO facet VALUES (?,?,?,?,?)", (md5[t], 0, "vocal", "effnet-discogs-1", v.tobytes()))
            sc.execute("INSERT INTO facet VALUES (?,?,?,?,?)", (md5[t], 0, "rhythm", "effnet-discogs-1", v.tobytes()))
            sc.execute("INSERT INTO feat (md5, cue_ms, version, bpm) VALUES (?,0,1,120.0)", (md5[t],))
    sc.close()
    want = sound_tags.female_probability(voc)
    got = sidecar.import_missing(a, [lib])
    assert got == 2
    p = dict(a.execute("SELECT track_id, female_p FROM track_features"))
    assert np.allclose([p[t] for t in ids], want, atol=1e-6)
    assert a.execute("SELECT count(*) FROM facet_vectors").fetchone()[0] == 0  # векторы в базу не идут
    sidecar.export(a, [lib])  # уборка файла: число — в feat_more, векторы — прочь
    sc = sqlite3.connect(sidecar.path(lib))
    assert sc.execute("SELECT count(*) FROM facet").fetchone()[0] == 0
    assert np.allclose(sorted(r[0] for r in sc.execute("SELECT female_p FROM feat_more")), sorted(want), atol=1e-6)
    sc.close()

    # база сервера с векторами (разобрано до 0014): уборка переводит их в число и удаляет
    cfg_b, b = _server(tmp_path, "server_b", lib)
    bids = [r[0] for r in b.execute("SELECT id FROM tracks ORDER BY id")]
    with b:
        for t, v in zip(bids, voc):
            b.execute("INSERT INTO track_features (track_id, version, bpm) VALUES (?,1,120.0)", (t,))
            for facet in ("rhythm", "instr", "vocal"):
                b.execute("INSERT INTO facet_vectors VALUES (?,?,?,?)", (t, facet, "effnet-discogs-1", v.tobytes()))
    db.tidy_storage(b)
    assert b.execute("SELECT count(*) FROM facet_vectors").fetchone()[0] == 0
    p = dict(b.execute("SELECT track_id, female_p FROM track_features"))
    assert np.allclose([p[t] for t in bids], want, atol=1e-6)
