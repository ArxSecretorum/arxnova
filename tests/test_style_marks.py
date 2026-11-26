"""Стиль трека, поставленный владельцем (решение 0024): слово владельца — поверх автоматики. Виден сразу, переживает
пересборку каталога, действует на все издания записи, едет вместе с медиатекой. Повод — оригинальный «Kuos» Banco de Gaia
(26.11.2026): по людям транс, обе модели слышат хаус, местными свидетельствами не лечится."""

import shutil
import sqlite3
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from arxnova import auth, config, db, indexer, models, scan, sidecar
from arxnova.server import create_app
from tests.test_flac_keys import make_flac
from tests.test_server_nova import fake_model

DNB, HOUSE = {"genre": "Electronic", "style": "Drum n Bass"}, {"genre": "Electronic", "style": "Deep House"}


def library(root):
    """Два альбома DnB по 4 трека и одна песня ещё и синглом — второе издание той же записи."""
    lib = root / "lib"
    cfg = config.Config(data_dir=root / "data", libraries=[lib])
    cfg.ensure_dirs()
    fake_model(cfg)
    plan = [("a", "Album A", "Artist A", f"Alpha Song {i}") for i in range(4)]
    plan += [("b", "Album B", "Artist B", f"Beta Song {i}") for i in range(4)]
    plan += [("single", "Alpha Song 0 (Single)", "Artist A", "Alpha Song 0")]
    for i, (folder, album, artist, title) in enumerate(plan):
        (lib / folder).mkdir(parents=True, exist_ok=True)
        make_flac(lib / folder / f"{i:02d}.flac", seconds=0.3, freq=200 + 13 * i, ARTIST=artist, ALBUMARTIST=artist,
                  ALBUM=album, TITLE=title, DATE="2020")
    conn = db.connect(cfg.db_path, threaded=True)
    scan.scan(conn, [lib])
    rng = np.random.default_rng(0)
    center = rng.normal(size=64) * 3
    for r in conn.execute("SELECT id FROM tracks ORDER BY id").fetchall():
        sty = np.zeros(400, np.float32)
        sty[0] = 1.0  # всё — Drum n Bass
        conn.execute("INSERT INTO embeddings VALUES (?,?,?,?,?)",
                     (r["id"], models.EFFNET.id, (center + rng.normal(size=64) * 0.2).astype(np.float32).tobytes(), sty.tobytes(), 1))
    conn.commit()
    indexer.export_matrices(cfg, conn)
    return cfg, conn, lib


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    if not shutil.which("ffmpeg"):
        pytest.skip("нужен ffmpeg")
    cfg, conn, lib = library(tmp_path_factory.mktemp("marks"))
    app = create_app(cfg)
    client = TestClient(app)
    tok = client.post("/v1/pair", json={"code": auth.new_pair_code(conn), "name": "владелец"}).json()["token"]
    return cfg, conn, app, client, {"Authorization": f"Bearer {tok}"}


def track(client, h, title, album=None) -> dict:
    found = client.get("/v1/library/search", headers=h, params={"q": title}).json()["tracks"]
    return next(t for t in found if t["title"] == title and (album is None or t["album"] == album))


def rebuilt(app) -> None:
    """Каталог собран заново из базы — как после перезапуска сервера."""
    app.state.nova.forget_library()


def wait_until(cond, timeout=10.0):
    deadline = time.monotonic() + timeout
    while not cond():
        assert time.monotonic() < deadline, "не дождались"
        time.sleep(0.02)


def test_owner_style_shows_at_once_and_survives_rebuild(env):
    cfg, conn, app, client, h = env
    t = track(client, h, "Beta Song 1")
    assert t["style"] == "Drum n Bass" and "style_manual" not in t
    r = client.put(f"/v1/tracks/{t['id']}/style", json=HOUSE, headers=h)
    assert r.status_code == 200
    assert r.json() == {"id": t["id"], **HOUSE, "style_manual": True}
    for _ in range(2):  # тем же каталогом сразу — и собранным заново
        now = track(client, h, "Beta Song 1")
        assert (now["genre"], now["style"], now.get("style_manual")) == ("Electronic", "Deep House", True)
        neighbour = track(client, h, "Beta Song 2")  # поправка — слово об одном треке, альбом остаётся при своём
        assert neighbour["style"] == "Drum n Bass" and "style_manual" not in neighbour
        listed = client.get("/v1/library/styles/Deep House", headers=h).json()["tracks"]
        assert [x["id"] for x in listed] == [t["id"]]
        rebuilt(app)


def test_clearing_returns_the_automatic_style(env):
    cfg, conn, app, client, h = env
    t = track(client, h, "Beta Song 2")
    client.put(f"/v1/tracks/{t['id']}/style", json=HOUSE, headers=h)
    r = client.delete(f"/v1/tracks/{t['id']}/style", headers=h)
    assert r.status_code == 200 and r.json() == {"id": t["id"], **DNB}
    for _ in range(2):
        now = track(client, h, "Beta Song 2")
        assert now["style"] == "Drum n Bass" and "style_manual" not in now
        rebuilt(app)
    assert client.delete(f"/v1/tracks/{t['id']}/style", headers=h).json() == {"id": t["id"], **DNB}  # снимать нечего — не ошибка


def test_all_editions_of_a_recording_change_together(env):
    cfg, conn, app, client, h = env
    album, single = track(client, h, "Alpha Song 0", "Album A"), track(client, h, "Alpha Song 0", "Alpha Song 0 (Single)")
    assert album["id"] != single["id"]
    client.put(f"/v1/tracks/{album['id']}/style", json=HOUSE, headers=h)
    for _ in range(2):
        for where in ("Album A", "Alpha Song 0 (Single)"):
            now = track(client, h, "Alpha Song 0", where)
            assert (now["style"], now.get("style_manual")) == ("Deep House", True), where
        rebuilt(app)
    assert client.delete(f"/v1/tracks/{single['id']}/style", headers=h).json()["style"] == "Drum n Bass"  # снять — с любого издания
    for _ in range(2):
        for where in ("Album A", "Alpha Song 0 (Single)"):
            now = track(client, h, "Alpha Song 0", where)
            assert now["style"] == "Drum n Bass" and "style_manual" not in now, where
        rebuilt(app)


def test_unknown_style_unknown_track_and_guest_are_refused(env):
    cfg, conn, app, client, h = env
    t = track(client, h, "Beta Song 3")
    assert client.put(f"/v1/tracks/{t['id']}/style", json={"genre": "Electronic", "style": "Нет такого"}, headers=h).status_code == 422
    assert client.put("/v1/tracks/999999/style", json=HOUSE, headers=h).status_code == 404
    guest = client.post("/v1/pair", json={"code": auth.new_pair_code(conn), "name": "гость"}).json()
    with conn:
        conn.execute("UPDATE devices SET role = 'guest' WHERE id = ?", (guest["device_id"],))
    g = {"Authorization": f"Bearer {guest['token']}"}
    assert client.put(f"/v1/tracks/{t['id']}/style", json=HOUSE, headers=g).status_code == 403
    assert client.delete(f"/v1/tracks/{t['id']}/style", headers=g).status_code == 403
    assert track(client, g, "Beta Song 3")["style"] == "Drum n Bass"
    assert client.get("/v1/library/styles", headers=g).status_code == 200  # список стилей гостю виден


def test_choices_are_every_style_the_server_knows(env):
    cfg, conn, app, client, h = env
    items = client.get("/v1/library/styles", headers=h).json()["items"]
    assert len(items) == 400
    by = {(x["genre"], x["style"]): x["tracks"] for x in items}
    on_dnb = sum(1 for t in client.get("/v1/library/styles/Drum n Bass", headers=h).json()["tracks"])
    assert by[("Electronic", "Drum n Bass")] == on_dnb > 0
    assert ("Jazz", "Contemporary Jazz") in by and by[("Jazz", "Contemporary Jazz")] == 0  # стиля в медиатеке нет — поставить можно


def _file_marks(lib) -> dict:
    sc = sqlite3.connect(sidecar.path(lib))
    try:
        return {(bytes(m), c): (label, ts) for m, c, label, ts in sc.execute("SELECT md5, cue_ms, label, ts FROM style")}
    except sqlite3.OperationalError:
        return {}
    finally:
        sc.close()


def test_mark_travels_with_the_library(env, tmp_path):
    cfg, conn, app, client, h = env
    lib = cfg.libraries[0]
    t = track(client, h, "Beta Song 0")
    key = (bytes.fromhex(t["md5"]), t["cue_ms"])
    client.put(f"/v1/tracks/{t['id']}/style", json=HOUSE, headers=h)
    # в файл медиатеки поправка уходит сразу, а не со следующим разбором новых треков
    wait_until(lambda: _file_marks(lib).get(key, ("",))[0] == "Electronic---Deep House")

    other = config.Config(data_dir=tmp_path / "data", libraries=[lib])  # другой сервер, тот же диск
    other.ensure_dirs()
    c2 = db.connect(other.db_path)
    scan.scan(c2, [lib])
    assert sidecar.import_missing(c2, [lib]) >= 1
    got = c2.execute("SELECT m.label FROM style_marks m JOIN tracks t ON t.id = m.track_id WHERE t.md5 = ? AND t.cue_ms = ?", key).fetchone()
    assert got is not None and got[0] == "Electronic---Deep House"

    client.delete(f"/v1/tracks/{t['id']}/style", headers=h)  # сняли позже — второй сервер тоже снимает
    wait_until(lambda: _file_marks(lib).get(key, ("?",))[0] == "")
    sidecar.import_missing(c2, [lib])
    assert c2.execute("SELECT m.label FROM style_marks m JOIN tracks t ON t.id = m.track_id WHERE t.md5 = ?", (key[0],)).fetchone()[0] == ""
    # а старая поправка из файла не перебивает более новую в базе
    with c2:
        c2.execute("UPDATE style_marks SET label = 'Jazz---Contemporary Jazz', ts = ts + 1000")
    sidecar.import_missing(c2, [lib])
    assert c2.execute("SELECT label FROM style_marks").fetchone()[0] == "Jazz---Contemporary Jazz"
    sidecar.export(c2, [lib])
    assert _file_marks(lib)[key][0] == "Jazz---Contemporary Jazz"
    c2.close()


def _settled(st) -> bool:
    """Ни одна пересборка каталога не идёт и не ждёт."""
    if st._lib_built != st._lib_generation or not st._lib_lock.acquire(blocking=False):
        return False
    st._lib_lock.release()
    return True


def test_second_edit_during_rebuild_shows_at_once(env, monkeypatch):
    """Гонка, пойманная прогоном на macOS 26.11.2026: владелец поправил стиль и тут же поправил ещё раз. Первая пересборка
    каталога уже сверилась, что правок не было, но каталог ещё не подменила; вторая правка легла на прежний каталог, а
    отвечать начал новый — со стилем от первой правки, до конца следующей пересборки (на Pi это 20 с)."""
    import threading

    from arxnova import nova

    cfg, conn, app, client, h = env
    st = app.state.nova
    t = track(client, h, "Beta Song 3")
    wait_until(lambda: _settled(st))
    armed, closing, go, second = threading.Event(), threading.Event(), threading.Event(), threading.Event()
    builds = []

    class SlowClose(sqlite3.Connection):
        def close(self):  # сборка каталога закрывает своё соединение последним шагом — здесь её и придержим
            if armed.is_set() and threading.current_thread().name.startswith("library-"):
                armed.clear()
                closing.set()
                assert go.wait(10)
            super().close()

    real_connect, real_library = sqlite3.connect, nova.Library

    def library(*a, **k):
        builds.append(1)
        if len(builds) >= 2:  # вторая пересборка ждёт: смотрим, что отвечает сервер между пересборками
            assert second.wait(10)
        return real_library(*a, **k)

    monkeypatch.setattr(sqlite3, "connect", lambda *a, **k: real_connect(*a, **{"factory": SlowClose, **k}))
    monkeypatch.setattr(nova, "Library", library)
    armed.set()
    client.put(f"/v1/tracks/{t['id']}/style", json=HOUSE, headers=h)
    assert closing.wait(10), "первая пересборка не дошла до конца"
    client.put(f"/v1/tracks/{t['id']}/style", json={"genre": "Jazz", "style": "Contemporary Jazz"}, headers=h)
    assert track(client, h, "Beta Song 3")["style"] == "Contemporary Jazz"
    go.set()
    wait_until(lambda: len(builds) >= 2)  # первая пересборка закончена, вторая стоит на пороге
    try:
        assert track(client, h, "Beta Song 3")["style"] == "Contemporary Jazz", "после пересборки отвечает каталог без второй правки"
    finally:
        second.set()
    wait_until(lambda: _settled(st))
    assert track(client, h, "Beta Song 3")["style"] == "Contemporary Jazz"
    client.delete(f"/v1/tracks/{t['id']}/style", headers=h)
    wait_until(lambda: _settled(st))
