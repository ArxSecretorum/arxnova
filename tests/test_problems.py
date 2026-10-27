"""Сервер сам говорит плееру, что не так (у сервера без экрана больше негде): диск с медиатекой
отключён, нет модели, есть новая версия."""

from fastapi.testclient import TestClient

from arxnova import config, tlsid, updates
from arxnova.reach import Reach
from arxnova.server import create_app


def test_problems_in_health(tmp_path, monkeypatch):
    monkeypatch.setattr(updates, "latest", lambda: "9.9.9")
    lib = tmp_path / "disk" / "Music"
    lib.mkdir(parents=True)
    cfg = config.Config(data_dir=tmp_path / "data", libraries=[lib], remote_port=23056)
    cfg.ensure_dirs()
    app = create_app(cfg, Reach(23056), tlsid.ensure(cfg.data_dir, "дом").fingerprint)
    home = TestClient(app, base_url="http://testserver:8765", client=("192.168.0.5", 1))
    codes = {p["code"] for p in home.get("/v1/health").json()["problems"]}
    assert codes == {"no_model", "update"}  # диск на месте — о нём ни слова
    lib.rmdir()  # диск отключили
    codes = {p["code"] for p in home.get("/v1/health").json()["problems"]}
    assert "library_missing" in codes


def test_version_compare():
    assert updates.newer("0.2.40", "0.2.37") and not updates.newer("0.2.37", "0.2.37")
    assert updates.newer("0.10.0", "0.9.9")


def test_models_consent_from_player(tmp_path, monkeypatch):
    """Согласие на модель — из плеера владельца: текст лицензии с сервера, установка по нажатию."""
    from arxnova import models
    installed = []
    monkeypatch.setattr(models, "install", lambda root, m, progress=None, **kw: installed.append(m.id))
    cfg = config.Config(data_dir=tmp_path / "data", remote_port=23056)
    cfg.ensure_dirs()
    app = create_app(cfg, Reach(23056), tlsid.ensure(cfg.data_dir, "дом").fingerprint)
    home = TestClient(app, base_url="http://testserver:8765", client=("192.168.0.5", 1))
    assert home.get("/v1/setup/models").status_code == 401
    h = {"Authorization": "Bearer " + home.post("/v1/setup/claim", json={"name": "M9"}).json()["token"]}
    m = home.get("/v1/setup/models", headers=h).json()["models"][0]
    assert m["id"] == "effnet-discogs-1" and "CC BY-NC-SA" in m["license"] and m["consent"] and not m["installed"]
    assert home.post("/v1/setup/models/effnet-discogs-1", headers=h).json()["state"] == "installing"
    import time
    for _ in range(50):
        if installed:
            break
        time.sleep(0.02)
    assert installed == ["effnet-discogs-1"]
    assert home.post("/v1/setup/models/nope", headers=h).status_code == 404


def test_self_update_only_at_quiet_night(monkeypatch):
    from datetime import datetime
    import time as _t
    monkeypatch.delenv("ARXNOVA_NO_AUTO_UPDATE", raising=False)
    monkeypatch.setattr(updates, "latest", lambda: "9.9.9")
    monkeypatch.setitem(updates._state, "wheel", "https://x/arxnova-9.9.9-py3-none-any.whl")
    night, day = datetime(2026, 10, 28, 4, 0), datetime(2026, 10, 28, 14, 0)
    quiet, listening = _t.time() - 3600, _t.time() - 60
    assert updates.due_for_update(night, quiet, "0.2.38").endswith(".whl")
    assert updates.due_for_update(day, quiet, "0.2.38") is None  # днём — нет
    assert updates.due_for_update(night, listening, "0.2.38") is None  # слушают — нет
    assert updates.due_for_update(night, quiet, "9.9.9") is None  # уже последняя
    monkeypatch.setenv("ARXNOVA_NO_AUTO_UPDATE", "1")
    assert updates.due_for_update(night, quiet, "0.2.38") is None
