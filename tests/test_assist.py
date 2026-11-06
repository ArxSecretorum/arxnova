"""Разбор медиатеки на другом компьютере по локальной сети (решение 0009)."""

import base64
import hashlib
import json
import sqlite3
import sys

import numpy as np
import pytest
from fastapi.testclient import TestClient

from arxnova import analysis, assist, config, db, facets, models, nova, scan, sidecar, tlsid
from arxnova.assist import BadFile, LinkDown, LinkError
from arxnova.reach import Reach
from arxnova.server import create_app
from tests.test_flac_keys import make_flac, needs_ffmpeg


def _b64(dim, value=0.5):
    return base64.b64encode(np.full(dim, value, np.float32).tobytes()).decode()


def _item(md5, cue=0):
    return {"md5": md5, "cue_ms": cue,
            "maest": {"a": _b64(assist.MAEST_CLASSES), "w": 3},
            "feat": {"version": facets.VERSION, "bpm": 120.0, "bpm_conf": 0.9, "bpm_src": "beat_this", "musical_key": None,
                     "key_conf": None, "loudness_db": -14.0, "onset_rate": 3.0, "vocal_frac": None, "vocal_ratio": None},
            "clap": _b64(assist.CLAP_DIM)}


@pytest.fixture(autouse=True)
def _fresh_helper():
    analysis.helper.update(name="", seen=0.0)
    yield
    analysis.helper.update(name="", seen=0.0)
    analysis.take_results_flag()


def _server(tmp_path):
    lib = tmp_path / "lib"
    (lib / "album").mkdir(parents=True)
    make_flac(lib / "album" / "a.flac", freq=300.0, ARTIST="A", TITLE="One")
    make_flac(lib / "album" / "b.flac", freq=500.0, ARTIST="B", TITLE="Two")
    (lib / "secret.txt").write_text("не музыка")
    cfg = config.Config(data_dir=tmp_path / "data", libraries=[lib], remote_port=23056)
    cfg.ensure_dirs()
    conn = db.connect(cfg.db_path)
    scan.scan(conn, [lib])
    conn.close()
    ident = tlsid.ensure(cfg.data_dir, "дом")
    app = create_app(cfg, Reach(23056), ident.fingerprint)
    home = TestClient(app, base_url="http://testserver:8765", client=("192.168.0.50", 40000))
    away = TestClient(app, base_url="https://testserver:23056", client=("192.168.0.50", 40000))
    owner = {"Authorization": "Bearer " + home.post("/v1/setup/claim", json={"name": "M9"}).json()["token"]}
    return cfg, app, home, away, owner


def _grant(home, owner, answer_role="guest"):
    r = home.post("/v1/access/request", json={"name": "Mac · разбор", "kind": "assist"}).json()
    # в плеере нет кнопки «разбор»: что бы ни выбрал владелец, такой запрос даёт только роль assist
    assert home.post(f"/v1/access/requests/{r['id']}", json={"allow": True, "role": answer_role}, headers=owner).status_code == 200
    st = home.get(f"/v1/access/request/{r['id']}", params={"secret": r["secret"]}).json()
    assert st["role"] == "assist"
    return {"Authorization": f"Bearer {st['token']}"}


@needs_ffmpeg
@pytest.mark.parametrize("answer", ["guest", "owner"])
def test_token_reads_only_assist_api(tmp_path, answer):
    cfg, app, home, away, owner = _server(tmp_path)
    h = _grant(home, owner, answer)
    assert home.get("/v1/assist/pending", headers=h).status_code == 200
    for path in ("/v1/library/artists", "/v1/devices/me", "/v1/nova/facets", "/v1/devices"):
        assert home.get(path, headers=h).status_code == 403, path
    # из интернета токен не работает, а токен владельца не подходит к разбору
    assert away.get("/v1/assist/pending", headers=h).status_code == 403
    assert home.get("/v1/assist/pending", headers=owner).status_code == 403
    assert home.get("/v1/assist/pending").status_code == 401


@needs_ffmpeg
def test_owner_sees_and_revokes_helper(tmp_path):
    cfg, app, home, away, owner = _server(tmp_path)
    h = _grant(home, owner)
    dev = next(d for d in home.get("/v1/devices", headers=owner).json()["devices"] if "разбор" in d["name"])
    assert home.delete(f"/v1/devices/{dev['id']}", headers=owner).status_code == 200
    assert home.get("/v1/assist/pending", headers=h).status_code == 401


@needs_ffmpeg
def test_file_only_from_library_with_checksum(tmp_path):
    cfg, app, home, away, owner = _server(tmp_path)
    h = _grant(home, owner)
    batch = home.get("/v1/assist/pending", headers=h).json()
    assert batch["total"] == 2 and len(batch["files"]) == 2
    path = batch["files"][0]["path"]
    r = home.get("/v1/assist/file", params={"path": path}, headers=h)
    assert r.status_code == 200
    assert r.headers["x-arx-sha256"] == hashlib.sha256(r.content).hexdigest()
    for bad in (str(cfg.data_dir / "arxnova.toml"), str(cfg.libraries[0] / "secret.txt"), "/etc/passwd"):
        assert home.get("/v1/assist/file", params={"path": bad}, headers=h).status_code == 404, bad


@needs_ffmpeg
def test_results_fill_only_what_is_missing(tmp_path):
    cfg, app, home, away, owner = _server(tmp_path)
    h = _grant(home, owner)
    files = home.get("/v1/assist/pending", headers=h).json()["files"]
    good, other = files[0]["tracks"][0], files[1]["tracks"][0]
    broken = _item(other["md5"])
    broken["maest"]["a"] = _b64(3)  # не та длина
    nan = _item(other["md5"])
    nan["clap"] = _b64(assist.CLAP_DIM, float("nan"))
    r = home.post("/v1/assist/results", json={"items": [_item(good["md5"]), broken, nan, _item("00" * 16)], "last": True},
                  headers=h).json()
    assert r == {"saved": 1, "rejected": 2, "unknown": 1}
    assert analysis.take_results_flag()
    # отклонённый трек отложен на сутки, как неразобранный: иначе помощник гонял бы ту же пачку по кругу
    left = home.get("/v1/assist/pending", headers=h).json()
    assert left["total"] == 0
    c = sqlite3.connect(cfg.db_path)
    # у отклонённого трека в базе ни кусочка
    assert c.execute("SELECT count(*) FROM maest").fetchone()[0] == 1
    assert c.execute("SELECT count(*) FROM track_features").fetchone()[0] == 1
    assert c.execute("SELECT count(*) FROM facet_vectors").fetchone()[0] == 0
    # повтор не перезаписывает: посчитанное раньше остаётся
    again = _item(good["md5"])
    again["clap"] = _b64(assist.CLAP_DIM, 0.9)
    home.post("/v1/assist/results", json={"items": [again]}, headers=h)
    v = np.frombuffer(c.execute("SELECT vector FROM clap_vectors").fetchone()[0], np.float32)
    assert float(v[0]) == 0.5


@needs_ffmpeg
def test_failed_tracks_wait_a_day_instead_of_looping(tmp_path):
    cfg, app, home, away, owner = _server(tmp_path)
    h = _grant(home, owner)
    files = home.get("/v1/assist/pending", headers=h).json()["files"]
    t = files[0]["tracks"][0]
    home.post("/v1/assist/results", json={"failed": [t]}, headers=h)
    left = home.get("/v1/assist/pending", headers=h).json()
    assert left["total"] == 1 and t["md5"] not in json.dumps(left["files"])


def test_listing_keeps_cue_image_whole_and_respects_limits(tmp_path):
    d = tmp_path / "x"
    d.mkdir()
    (d / "image.flac").write_bytes(b"0" * 100)
    (d / "image.cue").write_text("cue")
    (d / "._image.cue").write_text("мусор macOS")
    for n in "abc":
        (d / f"{n}.flac").write_bytes(b"0" * 10)
    rows = [(bytes([i]) * 16, i * 1000, str(d / "image.flac")) for i in range(5)]
    rows += [(bytes([50 + i]) * 16, 0, str(d / f"{n}.flac")) for i, n in enumerate("abc")]
    out = assist.listing(rows, set(), limit=6)
    # образ не разрезан: файл с пятью треками и один соседний; лист CUE рядом
    assert [len(f["tracks"]) for f in out["files"]] == [5, 1]
    assert out["total"] == 8
    assert [c["name"] for c in out["cues"]] == ["image.cue"]
    assert {f["dir"] for f in out["files"]} == {out["cues"][0]["dir"]}
    assert len(assist.listing(rows, set(), limit=1)["files"]) == 1  # больше лимита, но файл один — идёт
    assert len(assist.listing(rows, set(), limit=50, max_bytes=105)["files"]) == 1


def test_maest_labels_without_model(tmp_path):
    """Сервер без модели на 350 МБ пользуется стилями, которые посчитал другой компьютер."""
    cfg = config.Config(data_dir=tmp_path / "data")
    cfg.ensure_dirs()
    d = cfg.index_dir / models.EFFNET.id
    d.mkdir(parents=True)
    effnet = np.array([[1.0, 0.0], [0.0, 1.0]], np.float32)
    acts = np.zeros((2, assist.MAEST_CLASSES), np.float32)
    acts[0, 2] = 1.0
    np.save(d / "maest.npy", acts)
    np.save(d / "maest_has.npy", np.array([1, 0], np.uint8))
    assert nova._maest_styles(cfg, d, effnet, ["A", "B"], 2)[1] == ["A", "B"]  # названий нет — как раньше
    assist.save_extras(cfg, ["A", "B", "C"] + [f"S{i}" for i in range(516)], None)
    # трек с MAEST — по нему, без MAEST — по EffNet в тех же колонках
    out, labels = nova._maest_styles(cfg, d, effnet, ["A", "B"], 2)
    assert labels[2] == "C" and out[0].argmax() == 2 and out[1].argmax() == 1
    assert "maest_labels.json" in sidecar.index_files(cfg)  # едет вместе с диском
    assist.save_extras(cfg, ["мало"], None)  # мусор не затирает настоящие названия
    assert len(models.maest_labels_for(cfg.models_dir, cfg.index_dir)) == 519


def test_scenario_text_needed_and_validated(tmp_path):
    cfg = config.Config(data_dir=tmp_path / "data")
    cfg.ensure_dirs()
    assert assist.needs(cfg) == {"labels": True, "scenario_text": True}
    from arxnova import scenarios

    texts, ids = scenarios._texts(scenarios.load_defs(cfg))
    good = tmp_path / "good.npz"
    np.savez(good, ids=np.array(ids), text=np.zeros((len(ids), 512), np.float32), prompts=np.array(texts))
    assist.save_extras(cfg, None, base64.b64encode(good.read_bytes()).decode())
    assert assist.needs(cfg)["scenario_text"] is False
    with pytest.raises(ValueError):
        assist.save_extras(cfg, None, base64.b64encode(b"not npz").decode())
    assert assist.needs(cfg)["scenario_text"] is False  # старый файл на месте
    (cfg.data_dir / "scenarios.toml").write_text('[[scenario]]\nid = "own"\nprompt = "my own"\n')
    assert assist.needs(cfg)["scenario_text"] is True  # владелец добавил свой сценарий


def test_own_deep_analysis_waits_for_helper(tmp_path):
    cfg = config.Config(data_dir=tmp_path / "data")
    analysis.helper.update(name="Mac", seen=analysis.time.time())
    assert analysis.helper_active()
    assert analysis.run_deep(cfg, None) is False  # база не тронута: conn=None не падает


# ---------- компьютер-помощник ----------

class FakeLink:
    """Сервер с медиатекой для Worker: настоящее приложение через TestClient."""

    home = None
    corrupt: dict = {}  # путь → сколько раз отдать файл битым
    down_after = None

    def __init__(self, addr, port, token=None):
        self.token = token

    def _h(self):
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    def call(self, path, params=None, body=None, method=None, timeout=60):
        if body is not None:
            r = FakeLink.home.post(path, params=params, json=body, headers=self._h())
        else:
            r = FakeLink.home.get(path, params=params, headers=self._h())
        if r.status_code >= 400:
            raise LinkError(r.status_code)
        return r.json()

    def download(self, path, dest):
        if FakeLink.down_after == 0:
            raise LinkDown("обрыв")
        r = FakeLink.home.get("/v1/assist/file", params={"path": path}, headers=self._h())
        if r.status_code >= 400:
            raise LinkError(r.status_code)
        if FakeLink.corrupt.get(path, 0) > 0:
            FakeLink.corrupt[path] -= 1
            raise BadFile(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(r.content)
        if FakeLink.down_after is not None:
            FakeLink.down_after -= 1


def fake_batch(cfg, work, on_line):
    """Настоящий обход рабочей папки и подставные числа: проверяем дорогу, а не модели."""
    wcfg = config.load(work)
    wcfg.ensure_dirs()
    conn = db.connect(wcfg.db_path)
    scan.scan(conn, wcfg.libraries)
    on_line("  … стили 1/2")
    for (tid,) in conn.execute("SELECT id FROM tracks").fetchall():
        with conn:
            conn.execute("INSERT INTO maest VALUES (?,?,?,?)", (tid, np.full(519, 0.1, np.float32).tobytes(), np.full(768, 0.2, np.float32).tobytes(), 3))
            conn.execute("INSERT INTO clap_vectors VALUES (?,?)", (tid, np.full(512, 0.3, np.float32).tobytes()))
            conn.execute("""INSERT INTO track_features (track_id, version, bpm, bpm_conf, bpm_src, loudness_db, onset_rate)
                VALUES (?,?,?,?,?,?,?)""", (tid, facets.VERSION, 100.0, 0.5, "beat_this", -10.0, 2.0))
    conn.close()


def _worker(tmp_path, home, batch=fake_batch, on_battery=lambda: False, monkeypatch=None):
    FakeLink.home, FakeLink.corrupt, FakeLink.down_after = home, {}, None
    mine = config.Config(data_dir=tmp_path / "helper", server_name="Mac")
    mine.ensure_dirs()
    monkeypatch.setattr(analysis, "installed", lambda cfg: True)
    w = assist.Worker(mine, link=FakeLink, batch=batch, on_battery=on_battery)
    w.blip_s = 0
    w.configure(True, "дом", "192.168.0.101", 8765)
    return w


def _approve(w, home, owner):
    assert w.tick() == 3
    st = w.state
    assert st["stage"] == "asking" and st["check"]
    req = home.get("/v1/access/requests", headers=owner).json()["requests"][0]
    assert req["check"] == st["check"] and "разбор" in req["name"]
    home.post(f"/v1/access/requests/{req['id']}", json={"allow": True, "role": "guest"}, headers=owner)
    assert w.tick() == 0  # токен получен
    assert w.settings["token"]


@needs_ffmpeg
def test_worker_full_round(tmp_path, monkeypatch):
    cfg, app, home, away, owner = _server(tmp_path)
    w = _worker(tmp_path, home, monkeypatch=monkeypatch)
    _approve(w, home, owner)
    assert w.tick() == 0
    assert w.done == 2 and not w.work.exists()  # файлы удалены
    c = sqlite3.connect(cfg.db_path)
    for table, n in (("maest", 2), ("clap_vectors", 2), ("track_features", 2)):
        assert c.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == n, table
    assert analysis.take_results_flag()  # сервер пересоберёт индекс
    # названия стилей недоступны (модели на этом компьютере нет) — пачка всё равно дошла; в очереди пусто
    assert w.tick() == assist.IDLE_S and w.state["stage"] == "idle"
    assert analysis.helper["name"] == "Mac"
    assert app.state.assist_worker.snapshot()["on"] is False  # настройки сервера и помощника раздельны


@needs_ffmpeg
def test_worker_keys_come_from_server_not_from_local_md5(tmp_path, monkeypatch):
    """У дорожек с потерями MD5 считается по звуку и зависит от версии декодера."""
    cfg, app, home, away, owner = _server(tmp_path)
    w = _worker(tmp_path, home, monkeypatch=monkeypatch)
    _approve(w, home, owner)

    def batch(mine, work, on_line):
        fake_batch(mine, work, on_line)
        c = sqlite3.connect(work / "db" / "arxnova.sqlite3")
        with c:
            c.execute("UPDATE tracks SET md5 = randomblob(16)")
        c.close()

    w.batch = batch
    w.tick()
    assert sqlite3.connect(cfg.db_path).execute("SELECT count(*) FROM maest").fetchone()[0] == 2


@needs_ffmpeg
def test_corrupt_file_never_reaches_analysis(tmp_path, monkeypatch):
    cfg, app, home, away, owner = _server(tmp_path)
    seen = []

    def batch(mine, work, on_line):
        seen.extend(p.name for p in (work / "lib").rglob("*.flac"))
        fake_batch(mine, work, on_line)

    w = _worker(tmp_path, home, batch=batch, monkeypatch=monkeypatch)
    _approve(w, home, owner)
    files = home.get("/v1/assist/pending", headers={"Authorization": f"Bearer {w.settings['token']}"}).json()["files"]
    FakeLink.corrupt = {files[0]["path"]: 99, files[1]["path"]: 1}  # первый битый всегда, второй — только раз
    assert w.tick() == 0
    assert seen == [files[1]["name"]]  # второй после повтора дошёл, первый в разбор не попал
    assert w.done == 1
    left = home.get("/v1/assist/pending", headers={"Authorization": f"Bearer {w.settings['token']}"}).json()
    assert left["files"] == []  # и не ходит по кругу: снова через сутки


@needs_ffmpeg
def test_link_download_checks_checksum(tmp_path):
    """Настоящий Link: файл с неверной суммой не остаётся на диске ни под каким именем."""
    import http.server
    import threading

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("X-Arx-Sha256", "0" * 64)
            self.send_header("Content-Length", "4")
            self.end_headers()
            self.wfile.write(b"data")

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        dest = tmp_path / "d" / "x.flac"
        with pytest.raises(BadFile):
            assist.Link("127.0.0.1", srv.server_address[1]).download("p", dest)
        assert list((tmp_path / "d").iterdir()) == []
    finally:
        srv.shutdown()


@needs_ffmpeg
def test_link_down_mid_batch_keeps_tracks_in_queue(tmp_path, monkeypatch):
    cfg, app, home, away, owner = _server(tmp_path)
    w = _worker(tmp_path, home, monkeypatch=monkeypatch)
    _approve(w, home, owner)
    FakeLink.down_after = 1
    with pytest.raises(LinkDown):
        w.tick()
    assert not w.work.exists()
    FakeLink.down_after = None
    assert home.get("/v1/assist/pending", headers={"Authorization": f"Bearer {w.settings['token']}"}).json()["total"] == 2


@needs_ffmpeg
def test_network_blip_loses_neither_files_nor_results(tmp_path, monkeypatch):
    """Wi-Fi пропадает на минуты: файл докачивается заново, а посчитанное ждёт связи и не выбрасывается."""
    cfg, app, home, away, owner = _server(tmp_path)
    runs = []

    def batch(mine, work, on_line):
        runs.append(1)
        fake_batch(mine, work, on_line)

    w = _worker(tmp_path, home, batch=batch, monkeypatch=monkeypatch)
    _approve(w, home, owner)
    blips = {"file": 2, "send": 3}

    class Blippy(FakeLink):
        def download(self, path, dest):
            if blips["file"]:
                blips["file"] -= 1
                raise LinkDown("обрыв")
            super().download(path, dest)

        def call(self, path, params=None, body=None, method=None, timeout=60):
            if path == "/v1/assist/results" and blips["send"]:
                blips["send"] -= 1
                raise LinkDown("обрыв")
            return super().call(path, params, body, method, timeout)

    w.link = Blippy
    assert w.tick() == 0
    assert len(runs) == 1 and w.done == 2  # считали один раз


@needs_ffmpeg
def test_batch_crash_does_not_postpone_tracks(tmp_path, monkeypatch):
    cfg, app, home, away, owner = _server(tmp_path)

    def boom(mine, work, on_line):
        raise RuntimeError("торч упал")

    w = _worker(tmp_path, home, batch=boom, monkeypatch=monkeypatch)
    _approve(w, home, owner)
    assert w.tick() == assist.RETRY_S
    assert w.state["stage"] == "error" and "торч" in w.state["detail"]
    assert home.get("/v1/assist/pending", headers={"Authorization": f"Bearer {w.settings['token']}"}).json()["total"] == 2


@needs_ffmpeg
def test_worker_waits_for_power_and_components(tmp_path, monkeypatch):
    cfg, app, home, away, owner = _server(tmp_path)
    w = _worker(tmp_path, home, on_battery=lambda: True, monkeypatch=monkeypatch)
    _approve(w, home, owner)
    assert w.tick() == assist.OFFLINE_S and w.state["stage"] == "battery" and w.state["left"] == 2
    monkeypatch.setattr(analysis, "installed", lambda c: False)
    assert w.tick() == assist.OFFLINE_S and w.state["stage"] == "components"


@needs_ffmpeg
def test_denied_and_revoked_do_not_ask_in_a_loop(tmp_path, monkeypatch):
    cfg, app, home, away, owner = _server(tmp_path)
    w = _worker(tmp_path, home, monkeypatch=monkeypatch)
    w.tick()
    rid = home.get("/v1/access/requests", headers=owner).json()["requests"][0]["id"]
    home.post(f"/v1/access/requests/{rid}", json={"allow": False}, headers=owner)
    w.tick()
    assert w.state["stage"] == "denied"
    w.tick()
    assert w.state["stage"] == "denied"
    assert home.get("/v1/access/requests", headers=owner).json()["requests"] == []  # нового запроса нет
    w.configure(True, retry=True)
    w.tick()
    assert w.state["stage"] == "asking"
    rid = home.get("/v1/access/requests", headers=owner).json()["requests"][0]["id"]
    home.post(f"/v1/access/requests/{rid}", json={"allow": True}, headers=owner)
    w.tick()
    assert w.settings["token"]
    dev = next(d for d in home.get("/v1/devices", headers=owner).json()["devices"] if "разбор" in d["name"])
    home.delete(f"/v1/devices/{dev['id']}", headers=owner)  # владелец отключил в плеере
    w.tick()
    assert w.settings["token"] is None and w.state["stage"] == "denied"


def test_old_server_gets_no_request(tmp_path, monkeypatch):
    """Старый сервер не знает роли: тот же запрос дал бы гостя или полный доступ."""
    asked = []

    class Old(FakeLink):
        def call(self, path, params=None, body=None, method=None, timeout=60):
            asked.append(path)
            return {"name": "arxnova", "version": "0.2.50", "server": "дом"}

    mine = config.Config(data_dir=tmp_path / "helper", server_name="Mac")
    w = assist.Worker(mine, link=Old, batch=fake_batch, on_battery=lambda: False)
    w.configure(True, "дом", "192.168.0.101", 8765)
    assert w.tick() == assist.RETRY_S and w.state["stage"] == "old_server"
    assert asked == ["/v1/health"]


@pytest.mark.parametrize("lang", ["RU", "EN"])
def test_every_status_has_text(lang):
    """Строка состояния в окне: у каждой стадии есть текст, и подстановка не падает."""
    from types import SimpleNamespace

    app = pytest.importorskip("arxnova.app")
    me = SimpleNamespace(t=getattr(app, lang))
    stages = {"choose", "offline", "old_server", "asking", "denied", "not_lan", "idle", "components", "battery", "busy",
              "fetch", "work", "send", "error"}
    for stage in stages:
        for confirm in ("window", "button", "device"):
            text = app.App._assist_text(me, {"name": "дом", "found": [], "state": {
                "stage": stage, "name": "дом", "left": 5, "n": 2, "i": 1, "detail": "x", "check": "47", "confirm": confirm}})
            assert text and "{" not in text, stage
    assert app.App._assist_text(me, {"state": {"stage": "off"}}) == ""


def test_worker_settings_survive_restart_and_token_is_private(tmp_path):
    mine = config.Config(data_dir=tmp_path / "helper", server_name="Mac")
    w = assist.Worker(mine, link=FakeLink, batch=fake_batch)
    w.configure(True, "дом", "192.168.0.101", 8765)
    w.settings["token"] = "секрет"
    w.configure(True, "дом", "192.168.0.102", 8765)  # тот же компьютер с новым адресом — разрешение остаётся
    assert w.settings["token"] == "секрет"
    if sys.platform != "win32":  # на Windows доступ задаётся не этими битами
        assert (w.path.stat().st_mode & 0o777) == 0o600
    again = assist.Worker(mine, link=FakeLink, batch=fake_batch)
    assert again.settings["on"] is True and again.settings["addr"] == "192.168.0.102"
    again.configure(True, "другой", "192.168.0.200", 8765)  # другой компьютер — другое разрешение
    assert again.settings["token"] is None


@needs_ffmpeg
def test_new_client_stays_readable_for_a_server_of_the_previous_version(tmp_path):
    """Сервер прежней версии отклонял трек без поля e, и пачка считалась по кругу: новый клиент шлёт нули той же длины."""
    import base64

    from arxnova import db as _db
    from arxnova import scan as _scan
    from tests.test_flac_keys import make_flac

    lib = tmp_path / "Music"
    lib.mkdir()
    make_flac(lib / "a.flac", freq=300.0, ARTIST="A", TITLE="One")
    work = tmp_path / "work"
    (work / "db").mkdir(parents=True)
    conn = _db.connect(work / "db" / "arxnova.sqlite3")
    _scan.scan(conn, [lib])
    tid, md5 = conn.execute("SELECT id, md5 FROM tracks").fetchone()
    with conn:
        conn.execute("INSERT INTO maest VALUES (?,?,?,?)", (tid, b"a" * (assist.MAEST_CLASSES * 4), b"", 3))
    conn.close()
    batch = {"files": [{"path": "server/a.flac", "tracks": [{"md5": bytes(md5).hex(), "cue_ms": 0}]}]}
    items, _ = assist.collect(work, batch, {"server/a.flac": lib / "a.flac"})
    assert len(base64.b64decode(items[0]["maest"]["e"])) == assist.MAEST_DIM * 4


@needs_ffmpeg
def test_old_helper_numbers_accepted_voice_vector_becomes_number(tmp_path):
    """Помощник прежней версии (с Demucs): признаки версии 1 — те же числа, что у разобранных до 0014, берём; из вектора стема
    вокала — только вероятность женского голоса, сами векторы фасетов не храним."""
    from arxnova import sound_tags

    cfg, app, home, away, owner = _server(tmp_path)
    h = _grant(home, owner)
    good = home.get("/v1/assist/pending", headers=h).json()["files"][0]["tracks"][0]
    old = _item(good["md5"])
    old["feat"].update(version=1, vocal_frac=0.6, vocal_ratio=0.3)
    voc = np.random.default_rng(2).normal(size=1280).astype(np.float32)
    old["facets"] = {k: base64.b64encode(voc.tobytes()).decode() for k in ("rhythm", "instr", "vocal")}
    assert home.post("/v1/assist/results", json={"items": [old]}, headers=h).json()["saved"] == 1
    c = sqlite3.connect(cfg.db_path)
    version, vf, female = c.execute("SELECT version, vocal_frac, female_p FROM track_features").fetchone()
    assert version == 1 and vf == 0.6 and abs(female - float(sound_tags.female_probability(voc[None])[0])) < 1e-6
    assert c.execute("SELECT count(*) FROM facet_vectors").fetchone()[0] == 0


@needs_ffmpeg
def test_helper_sends_voice_numbers_and_clap_of_voice(tmp_path):
    """Решение 0014: вероятность голоса по стему 8 отрывков и CLAP выделенного голоса доходят до сервера с медиатекой."""
    cfg, app, home, away, owner = _server(tmp_path)
    h = _grant(home, owner)
    good = home.get("/v1/assist/pending", headers=h).json()["files"][0]["tracks"][0]
    it = _item(good["md5"])
    it["feat"]["female_stem_p"] = 0.83
    it["voc"] = _b64(assist.CLAP_DIM, 0.02)
    assert home.post("/v1/assist/results", json={"items": [it]}, headers=h).json()["saved"] == 1
    c = sqlite3.connect(cfg.db_path)
    assert c.execute("SELECT female_stem_p FROM track_features").fetchone()[0] == 0.83
    v = np.frombuffer(c.execute("SELECT vector FROM clap_voc").fetchone()[0], np.float32)
    assert len(v) == assist.CLAP_DIM and abs(float(v[0]) - 0.02) < 1e-6


def test_relisted_old_track_gets_voice_number_of_new_helper(tmp_path):
    """Трек разобран прежним кодом, помощник новой версии прислал его заново (из-за MAEST или CLAP): число голоса по стему
    отрывков дописывается, остальные признаки не меняются; уже записанное число не перезаписывается."""
    cfg, app, home, away, owner = _server(tmp_path)
    h = _grant(home, owner)
    good = home.get("/v1/assist/pending", headers=h).json()["files"][0]["tracks"][0]
    c = sqlite3.connect(cfg.db_path)
    tid = c.execute("SELECT id FROM tracks WHERE md5 = ?", (bytes.fromhex(good["md5"]),)).fetchone()[0]
    with c:
        c.execute("INSERT INTO track_features (track_id, version, bpm, vocal_frac, female_p) VALUES (?,1,120,0.6,0.2)", (tid,))
    it = _item(good["md5"])
    it["feat"].update(female_stem_p=0.83, vocal_frac=0.9)
    assert home.post("/v1/assist/results", json={"items": [it]}, headers=h).json()["saved"] == 1
    assert c.execute("SELECT version, vocal_frac, female_p, female_stem_p FROM track_features").fetchone() == (1, 0.6, 0.2, 0.83)
    it["feat"]["female_stem_p"] = 0.1
    home.post("/v1/assist/results", json={"items": [it]}, headers=h)
    assert c.execute("SELECT female_stem_p FROM track_features").fetchone()[0] == 0.83
