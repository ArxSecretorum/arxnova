"""Подтверждение нового устройства на самом сервере (решение владельца 27.10.2026): окно на
Mac/Windows/Linux с монитором, кнопка Raspberry Pi без монитора, «дом без хозяина» на прочем
Linux без монитора."""

from fastapi.testclient import TestClient

from arxnova import config, presence, tlsid
from arxnova.reach import Reach
from arxnova.server import create_app


def test_method_by_platform_and_monitor():
    assert presence.method("mac", monitor=False, display=False) == "window"
    assert presence.method("windows", monitor=False, display=False) == "window"
    assert presence.method("pi", monitor=True, display=True) == "window"
    assert presence.method("linux", monitor=True, display=True) == "window"
    assert presence.method("pi", monitor=False, display=True) == "button"
    assert presence.method("pi", monitor=True, display=False) == "button"  # окно показать негде
    assert presence.method("linux", monitor=False, display=True) == "home"


def test_monitor_from_drm_status(tmp_path):
    for name, status in (("card1-HDMI-A-1", "disconnected"), ("card1-Writeback-1", "unknown")):
        (tmp_path / name).mkdir()
        (tmp_path / name / "status").write_text(status + "\n")
    assert not presence.monitor_connected(tmp_path)
    (tmp_path / "card1-HDMI-A-2").mkdir()
    (tmp_path / "card1-HDMI-A-2" / "status").write_text("connected\n")
    assert presence.monitor_connected(tmp_path)


class _Req:
    def __init__(self, rid, name="M9", check="23"):
        self.id, self.name, self.check = rid, name, check


def test_button_approves_only_while_a_request_waits():
    approved = []
    sc = presence.ServerConfirm(lambda: [_Req("a")], lambda rid, role: approved.append((rid, role)), "дом",
                                log=lambda s: None)
    sc.on_button()  # запрета на выключение нет — запроса «не ждали», кнопка остаётся кнопкой
    assert approved == []
    sc._inhibit = object()  # сервер держит запрет, пока запрос ждёт
    sc.on_button()
    assert approved == [("a", "owner")]


def _app(tmp_path):
    cfg = config.Config(data_dir=tmp_path / "data", remote_port=23056)
    cfg.ensure_dirs()
    app = create_app(cfg, Reach(23056), tlsid.ensure(cfg.data_dir, "дом").fingerprint)
    home = TestClient(app, base_url="http://testserver:8765", client=("192.168.0.50", 40000))
    home.post("/v1/setup/claim", json={"name": "M9"})
    return app, home


def test_request_says_where_to_confirm(tmp_path, monkeypatch):
    _, home = _app(tmp_path)
    monkeypatch.setattr(presence, "method", lambda *a, **k: "button")
    r = home.post("/v1/access/request", json={"name": "M9"}).json()
    assert r["confirm"] == "button"
    assert home.get(f"/v1/access/request/{r['id']}", params={"secret": r["secret"]}).json()["status"] == "pending"


def test_home_without_owner_lets_the_device_in(tmp_path, monkeypatch):
    app, home = _app(tmp_path)
    monkeypatch.setattr(presence, "method", lambda *a, **k: "home")
    # владелец недавно был — ждём его подтверждения
    r = home.post("/v1/access/request", json={"name": "M9"}).json()
    assert r["confirm"] == "device"
    assert home.get(f"/v1/access/request/{r['id']}", params={"secret": r["secret"]}).json()["status"] == "pending"
    # устройство владельца не появлялось больше месяца — ушло; новое из дома становится своим
    monkeypatch.setattr(presence, "OWNER_GONE_S", -60)
    r = home.post("/v1/access/request", json={"name": "M9"}).json()
    got = home.get(f"/v1/access/request/{r['id']}", params={"secret": r["secret"]}).json()
    assert got["status"] == "approved" and got["role"] == "owner" and got["token"]
