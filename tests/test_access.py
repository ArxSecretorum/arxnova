"""Своё устройство без кода: запрос из домашней сети → «Разрешить» на устройстве владельца
(решение владельца 26.10.2026); код для своего устройства — из плеера владельца."""

from fastapi.testclient import TestClient

from arxnova import config, tlsid
from arxnova.reach import Reach
from arxnova.server import create_app


def _setup(tmp_path):
    cfg = config.Config(data_dir=tmp_path / "data", remote_port=23056)
    cfg.ensure_dirs()
    ident = tlsid.ensure(cfg.data_dir, "дом")
    app = create_app(cfg, Reach(23056), ident.fingerprint)
    home = TestClient(app, base_url="http://testserver:8765", client=("192.168.0.50", 40000))
    away = TestClient(app, base_url="https://testserver:23056", client=("192.168.0.50", 40000))
    token = home.post("/v1/setup/claim", json={"name": "M9"}).json()["token"]
    return home, away, {"Authorization": f"Bearer {token}"}


def test_request_approved_by_owner(tmp_path):
    home, away, owner = _setup(tmp_path)
    assert away.post("/v1/access/request", json={"name": "vivo"}).status_code == 403  # только из дома
    r = home.post("/v1/access/request", json={"name": "vivo"}).json()
    rid, secret, check = r["id"], r["secret"], r["check"]
    assert len(check) == 2
    assert home.get(f"/v1/access/request/{rid}", params={"secret": secret}).json()["status"] == "pending"
    assert home.get(f"/v1/access/request/{rid}", params={"secret": "чужой"}).status_code == 404

    assert home.get("/v1/access/requests").status_code == 401  # видит только владелец
    pend = home.get("/v1/access/requests", headers=owner).json()["requests"]
    assert [(p["name"], p["check"]) for p in pend] == [("vivo", check)]  # те же цифры, что у нового
    assert home.post(f"/v1/access/requests/{rid}", json={"allow": True, "role": "owner"}, headers=owner).json()["status"] == "approved"

    got = home.get(f"/v1/access/request/{rid}", params={"secret": secret}).json()
    assert got["status"] == "approved" and got["token"]
    me = home.get("/v1/devices/me", headers={"Authorization": f"Bearer {got['token']}"}).json()
    assert me["role"] == "owner" and me["name"] == "vivo"
    # токен отдаётся один раз
    assert home.get(f"/v1/access/request/{rid}", params={"secret": secret}).status_code == 404
    assert home.get("/v1/access/requests", headers=owner).json()["requests"] == []


def test_request_denied(tmp_path):
    home, _, owner = _setup(tmp_path)
    r = home.post("/v1/access/request", json={"name": "чужой"}).json()
    home.post(f"/v1/access/requests/{r['id']}", json={"allow": False}, headers=owner)
    got = home.get(f"/v1/access/request/{r['id']}", params={"secret": r["secret"]}).json()
    assert got["status"] == "denied" and "token" not in got


def test_no_request_before_owner_and_flood_limit(tmp_path):
    cfg = config.Config(data_dir=tmp_path / "data", remote_port=23056)
    cfg.ensure_dirs()
    app = create_app(cfg, Reach(23056), tlsid.ensure(cfg.data_dir, "дом").fingerprint)
    home = TestClient(app, base_url="http://testserver:8765", client=("192.168.0.50", 40000))
    assert home.post("/v1/access/request", json={"name": "x"}).status_code == 409  # забирают без запроса
    home.post("/v1/setup/claim", json={"name": "M9"})
    codes = [home.post("/v1/access/request", json={"name": f"d{i}"}).status_code for i in range(7)]
    assert codes[:5] == [200] * 5 and codes[5:] == [429, 429]


def test_owner_issues_pair_code(tmp_path):
    home, away, owner = _setup(tmp_path)
    assert home.post("/v1/pair/code").status_code == 401
    code = home.post("/v1/pair/code", headers=owner).json()["code"]
    assert len(code) == 6
    r = away.post("/v1/pair", json={"code": code, "name": "телефон вне дома"})
    assert r.status_code == 200 and r.json()["token"]


def test_guest_by_request_gets_guest_rights(tmp_path):
    """Друг у вас в Wi-Fi попросил доступ — пускаем гостем: гость не скачивает и не видит папки."""
    home, _, owner = _setup(tmp_path)
    r = home.post("/v1/access/request", json={"name": "телефон Пети"}).json()
    home.post(f"/v1/access/requests/{r['id']}", json={"allow": True, "role": "guest"}, headers=owner)
    got = home.get(f"/v1/access/request/{r['id']}", params={"secret": r["secret"]}).json()
    assert got["role"] == "guest"
    g = {"Authorization": f"Bearer {got['token']}"}
    assert home.get("/v1/devices/me", headers=g).json()["role"] == "guest"
    assert home.get("/v1/downloads/plan", params={"ids": "1"}, headers=g).status_code == 403
    assert home.get("/v1/setup/folders", headers=g).status_code == 403
    names = [x["name"] for x in home.get("/v1/guests", headers=owner).json()["guests"]]
    assert names == ["телефон Пети"]  # гость — в списке гостей, в лимите


def test_default_answer_is_guest(tmp_path):
    """Без явного role сервер пускает гостем: полный доступ — только по явному выбору."""
    home, _, owner = _setup(tmp_path)
    r = home.post("/v1/access/request", json={"name": "x"}).json()
    home.post(f"/v1/access/requests/{r['id']}", json={"allow": True}, headers=owner)
    assert home.get(f"/v1/access/request/{r['id']}", params={"secret": r["secret"]}).json()["role"] == "guest"


def test_owner_sees_and_revokes_own_devices(tmp_path):
    home, _, owner = _setup(tmp_path)
    r = home.post("/v1/access/request", json={"name": "vivo"}).json()
    home.post(f"/v1/access/requests/{r['id']}", json={"allow": True, "role": "owner"}, headers=owner)
    tok = home.get(f"/v1/access/request/{r['id']}", params={"secret": r["secret"]}).json()["token"]
    devs = home.get("/v1/devices", headers=owner).json()["devices"]
    assert [(d["name"], d["me"]) for d in devs] == [("M9", True), ("vivo", False)]
    me_id = devs[0]["id"]
    assert home.delete(f"/v1/devices/{me_id}", headers=owner).status_code == 400  # себя — нельзя
    assert home.delete(f"/v1/devices/{devs[1]['id']}", headers=owner).status_code == 200
    assert home.get("/v1/devices/me", headers={"Authorization": f"Bearer {tok}"}).status_code == 401
