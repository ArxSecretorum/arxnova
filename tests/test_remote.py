"""Связь вне дома (ОД-3, ОД-3а): ключ сервера, порт для интернета, разбор STUN, диагноз сети.
Сеть не нужна: роутер и STUN не трогаются."""

import os
import socket
import sys
import struct

from fastapi.testclient import TestClient

from arxnova import auth, config, db, reach, stun, tlsid
from arxnova.reach import Reach, Status
from arxnova.server import create_app


def test_identity_created_once_and_private(tmp_path):
    a = tlsid.ensure(tmp_path, "дом")
    b = tlsid.ensure(tmp_path, "дом")
    assert a.fingerprint == b.fingerprint and len(a.fingerprint) == 64
    if sys.platform != "win32":  # права POSIX; в Windows доступ задаёт профиль пользователя
        assert (a.key.stat().st_mode & 0o077) == 0  # ключ — только владельцу


def test_stun_xor_mapped_address():
    ip, magic = "5.9.10.11", stun.MAGIC
    xip = struct.unpack("!I", socket.inet_aton(ip))[0] ^ magic
    attr = struct.pack("!HHBBHI", 0x0020, 8, 0, 1, 50565 ^ (magic >> 16), xip)
    msg = struct.pack("!HHI", 0x0101, len(attr), magic) + b"\0" * 12 + attr
    assert stun._parse(msg) == ip


def _st(router, public, method="natpmp"):
    return Status(router_ip=router, public_ip=public, method=method, port=23056)


def test_diagnosis():
    d = Reach._diagnose
    assert d(_st("5.9.10.11", "5.9.10.11")) == "public"
    assert d(_st(None, "5.9.10.11", None)) == "manual"
    assert d(_st("100.72.1.5", "5.6.7.8")) == "cgnat"
    assert d(_st("192.168.1.10", "5.6.7.8")) == "double_nat"
    assert d(_st("1.2.3.4", None)) == "offline"
    ok = _st("5.9.10.11", "5.9.10.11")
    ok.diagnosis = d(ok)
    assert ok.endpoint() == "5.9.10.11:23056"
    s = _st("100.72.1.5", "5.6.7.8")
    s.diagnosis = "cgnat"
    assert s.endpoint() is None  # недостижимый адрес плееру не сообщаем


def test_internet_port_rules(tmp_path):
    cfg = config.Config(data_dir=tmp_path, remote_port=23056)
    cfg.ensure_dirs()
    ident = tlsid.ensure(tmp_path, "дом")
    r = Reach(23056)
    r.status = _st("5.9.10.11", "5.9.10.11")
    r.status.diagnosis = "public"
    app = create_app(cfg, r, ident.fingerprint)
    home = TestClient(app, base_url="http://testserver:8765")
    away = TestClient(app, base_url="https://testserver:23056")
    conn = db.connect(cfg.db_path)

    # сопряжение из интернета (решение владельца 22.10.2026): сначала сервер доказывает знание кода
    code = auth.new_pair_code(conn)
    secret = auth.pair_secret(code)
    nonce = os.urandom(16)
    hello = away.post("/v1/pair/hello", json={"key": auth.pair_keys(secret)["tcp"].hex(), "nonce": nonce.hex()}).json()
    assert hello["fingerprint"] == ident.fingerprint
    assert hello["proof"] == auth.pair_proof(secret, nonce, ident.fingerprint)
    # красное: подставной сервер без кода доказательство не сделает (чужой секрет → другой HMAC)
    assert hello["proof"] != auth.pair_proof(auth.pair_secret("000000" if code != "000000" else "111111"), nonce, ident.fingerprint)
    # ключ не от действующего кода — сервер молчит про себя
    assert away.post("/v1/pair/hello", json={"key": "00" * 20, "nonce": nonce.hex()}).status_code == 404
    res = away.post("/v1/pair", json={"code": code, "name": "M6"}).json()
    assert res["net"]["tls"] == {"port": 23056, "fingerprint": ident.fingerprint}
    assert res["net"]["external"] == "5.9.10.11:23056"
    token = res["token"]

    # из интернета без токена — только «я здесь», без сведений о медиатеке
    assert away.get("/v1/health").json() == {"name": "arxnova", "version": away.get("/v1/health").json()["version"]}
    assert "tracks" in home.get("/v1/health").json()
    # с токеном — всё как дома
    h = {"Authorization": f"Bearer {token}"}
    assert "tracks" in away.get("/v1/health", headers=h).json()
    assert "tracks" not in away.get("/v1/health", headers={"Authorization": "Bearer stranger"}).json()
    assert away.get("/v1/net", headers=h).json()["diagnosis"] == "public"
    assert away.get("/v1/library/albums", headers=h).status_code == 200
    assert away.get("/v1/library/albums").status_code == 401


def test_remote_port_picked_once(tmp_path):
    cfg = config.Config(data_dir=tmp_path)
    assert cfg.pick_remote_port() and 20000 <= cfg.remote_port < 60000
    cfg.save()
    again = config.load(tmp_path)
    assert again.remote_port == cfg.remote_port and not again.pick_remote_port()


def test_internet_port_hides_service_pages_and_throttles(tmp_path):
    cfg = config.Config(data_dir=tmp_path, remote_port=23056)
    cfg.ensure_dirs()
    app = create_app(cfg, None, "ab" * 32)
    home = TestClient(app, base_url="http://testserver:8765")
    away = TestClient(app, base_url="https://testserver:23056")
    assert home.get("/openapi.json").status_code == 200  # дома описание API доступно (API-1)
    for path in ("/openapi.json", "/docs", "/redoc"):
        assert away.get(path).status_code == 404
    bad = {"Authorization": "Bearer guess"}
    codes = [away.get("/v1/library/albums", headers=bad).status_code for _ in range(25)]
    assert codes[:20] == [401] * 20 and set(codes[20:]) == {429}
    assert home.get("/v1/library/albums", headers=bad).status_code == 401  # дома не блокируем
    # тот же адрес (роутер, оператор), но действующий токен — проходит: владельца не отрезаем
    conn = db.connect(cfg.db_path)
    code = auth.new_pair_code(conn)
    tok = home.post("/v1/pair", json={"code": code, "name": "M6"}).json()["token"]
    assert away.get("/v1/library/albums", headers={"Authorization": f"Bearer {tok}"}).status_code == 200


def test_address_history(tmp_path):
    log = tmp_path / "address_history.log"
    for ip in ("1.1.1.1", "1.1.1.1", "2.2.2.2", "2.2.2.2"):
        reach.note_address(log, ip)
    n, since = reach.address_changes(log)
    assert n == 1 and since


def test_dht_bencode_and_key():
    from arxnova import dht

    msg = {"t": b"aa", "y": "q", "q": "get_peers", "a": {"id": b"x" * 20, "port": 23056, "l": [1, b"z"]}}
    back = dht.bdec(dht.benc(msg))
    assert back[b"a"][b"port"] == 23056 and back[b"a"][b"l"] == [1, b"z"]
    k = dht.info_hash("ab" * 32)
    assert len(k) == 20 and k != dht.info_hash("ac" * 32)
    assert dht._peers([socket.inet_aton("1.2.3.4") + struct.pack("!H", 23056), b"bad"]) == [("1.2.3.4", 23056)]


def test_find_cover(tmp_path):
    from arxnova.server import find_cover

    album = tmp_path / "Artist" / "2020 - Album"
    album.mkdir(parents=True)
    assert find_cover(album) is None
    (tmp_path / "Artist" / "artist.jpg").write_bytes(b"x")
    assert find_cover(album).name == "artist.jpg"  # уровнем выше
    (album / "scan_back.jpg").write_bytes(b"x")
    (album / "Front Cover.JPG").write_bytes(b"x")
    assert find_cover(album).name == "Front Cover.JPG"


def test_local_endpoints_loopback_only(tmp_path):
    cfg = config.Config(data_dir=tmp_path, remote_port=23056)
    cfg.ensure_dirs()
    r = Reach(23056)
    r.status = _st("5.9.10.11", "5.9.10.11")
    r.refresh = lambda force=False: r.status  # без сети
    app = create_app(cfg, r, "ab" * 32)
    local = TestClient(app, base_url="http://testserver:8765", client=("127.0.0.1", 50000))
    lan = TestClient(app, base_url="http://testserver:8765", client=("192.168.0.50", 50000))
    away = TestClient(app, base_url="https://testserver:23056", client=("127.0.0.1", 50000))
    assert local.get("/v1/local/net").json()["public_ip"] == "5.9.10.11"
    assert local.post("/v1/local/net/check").status_code == 200
    assert lan.get("/v1/local/net").status_code == 403  # чужой в сети — нет
    assert away.get("/v1/local/net").status_code == 403  # из интернета — нет
