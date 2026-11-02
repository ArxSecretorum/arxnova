"""Приглашение — короткий ключ вместо ссылки (решение 0007): в ключе нет ни адреса, ни отпечатка, ни
имён; сервер находится по ключу в торрент-сети и доказывает, что знает его, — как при сопряжении
по коду. Старые приглашения (8 знаков) продолжают работать."""

import hashlib
import os
import time

from fastapi.testclient import TestClient

from arxnova import auth, config, db, guests, tlsid
from arxnova.reach import Reach, Status
from arxnova.server import create_app


def _env(tmp_path):
    cfg = config.Config(data_dir=tmp_path, remote_port=23056, owner_name="Рома")
    cfg.ensure_dirs()
    ident = tlsid.ensure(tmp_path, "дом")
    r = Reach(23056)
    r.status = Status(local_ip="192.168.0.5", public_ip="5.9.10.11", router_ip="5.9.10.11", port=23056, diagnosis="public")
    app = create_app(cfg, r, ident.fingerprint)
    home = TestClient(app, base_url="http://testserver:8765")
    away = TestClient(app, base_url="https://testserver:23056")
    conn = db.connect(cfg.db_path)
    token = home.post("/v1/pair", json={"code": auth.new_pair_code(conn), "name": "мой"}).json()["token"]
    return ident, home, away, conn, {"Authorization": f"Bearer {token}"}


def _hello(client, key: str, nonce: bytes):
    tcp = auth.pair_keys(guests.invite_secret(key))["tcp"]
    return client.post("/v1/pair/hello", json={"key": tcp.hex(), "nonce": nonce.hex()})


def test_key_is_short_and_carries_a_check_symbol():
    key = guests.new_key()
    assert len(key) == 16 and set(key) <= set(guests.ALPHABET) and guests.key_ok(key)
    assert guests.key_ok(guests.pretty(key).lower())  # регистр и дефисы не важны
    assert guests.pretty("ABCDEFGHJKMNPQR2") == "ABCD-EFGH-JKMN-PQR2"
    # тот же вектор — в InviteTest плеера: контрольный знак считается одинаково с обеих сторон
    assert guests.key_ok("ABCDEFGHJKMNPQR2") and not guests.key_ok("ABCDEFGHJKMNPQR3")
    assert not guests.key_ok("ABCD2345")  # старый код из 8 знаков — не ключ


def test_check_symbol_catches_every_single_typo_and_adjacent_swap():
    key = "K7QM9FHTXW4PR2N" + guests.check_symbol("K7QM9FHTXW4PR2N")
    assert guests.key_ok(key)
    for i in range(16):
        for c in guests.ALPHABET:
            if c != key[i]:
                assert not guests.key_ok(key[:i] + c + key[i + 1:]), (i, c)
        if i < 15 and key[i] != key[i + 1]:
            assert not guests.key_ok(key[:i] + key[i + 1] + key[i] + key[i + 2:]), i


def test_invite_link_holds_nothing_but_the_key(tmp_path):
    ident, home, away, conn, h = _env(tmp_path)
    inv = home.post("/v1/invites", json={"name": "Аня"}, headers=h).json()
    assert guests.key_ok(inv["code"])
    assert inv["link"] == f"arxnova://join?k={inv['code']}"  # ни адреса, ни отпечатка, ни имён
    assert ident.fingerprint not in inv["link"] and "5.9.10.11" not in inv["link"]


def test_guest_finds_server_by_key_and_joins(tmp_path):
    ident, home, away, conn, h = _env(tmp_path)
    key = home.post("/v1/invites", json={"name": "Аня"}, headers=h).json()["code"]
    nonce = os.urandom(16)
    hello = _hello(away, key, nonce).json()
    # сервер доказал, что знает ключ, и сказал всё, что раньше лежало в ссылке
    assert hello["fingerprint"] == ident.fingerprint
    assert hello["proof"] == auth.pair_proof(guests.invite_secret(key), nonce, ident.fingerprint)
    assert hello["server"] and hello["owner"] == "Рома"
    assert hello["net"]["tls"] == {"port": 23056, "fingerprint": ident.fingerprint}
    assert hello["net"]["external"] == "5.9.10.11:23056"
    assert hello["lan"] == "http://192.168.0.5:8765"
    # красное: подставной сервер без ключа доказательство не сделает
    other = guests.invite_secret("K7QM9FHTXW4PR2NZ")
    assert hello["proof"] != auth.pair_proof(other, nonce, ident.fingerprint)
    # чужой ключ — сервер молчит о себе
    assert _hello(away, "K7QM9FHTXW4PR2NZ", nonce).status_code == 404

    j = away.post("/v1/join", json={"code": guests.pretty(key).lower(), "name": "телефон"}).json()
    assert j["role"] == "guest" and j["guest_name"] == "Аня" and j["owner"] == "Рома"
    # ключ одноразовый: после входа он не живой, а сервер только говорит, что он использован
    assert guests.active_secrets(conn) == []
    assert _hello(away, key, nonce).json()["state"] == "used"


def test_revoked_and_expired_invites_stop_announcing(tmp_path):
    ident, home, away, conn, h = _env(tmp_path)
    a = home.post("/v1/invites", json={"name": "Аня"}, headers=h).json()
    b = home.post("/v1/invites", json={"name": "Боря"}, headers=h).json()
    assert len(guests.active_secrets(conn)) == 2
    home.delete(f"/v1/invites/{a['id']}", headers=h)
    assert len(guests.active_secrets(conn)) == 1
    with conn:
        conn.execute("UPDATE invites SET expires_at = ? WHERE id = ?", (int(time.time()) - 1, b["id"]))
    assert guests.active_secrets(conn) == []


def test_old_eight_symbol_invite_still_joins(tmp_path):
    """Приглашение, созданное до обновления: секрета у него нет, в DHT оно не объявляется, но вход
    по старой ссылке работает, пока оно живо."""
    ident, home, away, conn, h = _env(tmp_path)
    now = int(time.time())
    with conn:
        conn.execute("INSERT INTO invites (code_hash, guest_name, created_at, expires_at) VALUES (?,?,?,?)",
                     (hashlib.sha256(b"ABCD2345").digest(), "Старый", now, now + 3600))
    assert guests.active_secrets(conn) == []
    assert away.post("/v1/join", json={"code": "abcd-2345", "name": "x"}).json()["guest_name"] == "Старый"


def test_pair_code_hello_does_not_reveal_owner(tmp_path):
    """Шестизначный код короток: по нему сервер не раскрывает ни имя владельца, ни адреса."""
    ident, home, away, conn, h = _env(tmp_path)
    secret = auth.pair_secret(auth.new_pair_code(conn))
    r = away.post("/v1/pair/hello", json={"key": auth.pair_keys(secret)["tcp"].hex(), "nonce": os.urandom(16).hex()}).json()
    assert r["fingerprint"] == ident.fingerprint and "owner" not in r and "net" not in r and "lan" not in r


def test_values_match_the_player():
    """Те же значения — в InviteKeyTest плеера: иначе гость не найдёт сервер и не сойдётся доказательство."""
    s = guests.invite_secret("ABCDEFGHJKMNPQR2")
    assert s.hex() == "85ffda8b9418a9bb97fed06dbd6a7c189e1c0f0682c8ac3ff961b7f24ff221a0"
    assert auth.pair_keys(s)["tcp"].hex() == "009373bc10d0feb2e133d764f70d0983cde9dd38"


def test_invite_secret_is_separate_from_pair_codes():
    assert guests.invite_secret("ABCDEFGHJKMNPQR2") != auth.pair_secret("ABCDEFGHJKMNPQR2")
    # регистр и дефисы не меняют секрет: гость мог набрать ключ как угодно
    assert guests.invite_secret("abcd-efgh-jkmn-pqr2") == guests.invite_secret("ABCDEFGHJKMNPQR2")


def test_announce_schedule_is_dense_at_first_then_sparse():
    """Свежая метка в торрент-сети находится хуже (замер 02.11.2026, M9): первые минуты — часто,
    потом как у постоянной метки сервера."""
    assert guests.announce_due(age_s=0, since_last_s=None)
    assert not guests.announce_due(age_s=30, since_last_s=30)
    assert guests.announce_due(age_s=70, since_last_s=70)
    assert not guests.announce_due(age_s=3600, since_last_s=70)
    assert guests.announce_due(age_s=3600, since_last_s=16 * 60)


def test_owner_can_see_a_waiting_key_again_and_only_while_it_waits(tmp_path):
    """Закрыл карточку — ключ не должен пропасть (замечание 02.11.2026). Но только владельцу и пока он ждёт."""
    ident, home, away, conn, h = _env(tmp_path)
    a = home.post("/v1/invites", json={"name": "Аня"}, headers=h).json()
    b = home.post("/v1/invites", json={"name": "Боря"}, headers=h).json()
    c = home.post("/v1/invites", json={"name": "Вера"}, headers=h).json()
    waiting = {i["name"]: i["key"] for i in home.get("/v1/guests", headers=h).json()["invites"]}
    assert waiting == {"Аня": a["code"], "Боря": b["code"], "Вера": c["code"]}

    stored = lambda i: conn.execute("SELECT key_text FROM invites WHERE id = ?", (i,)).fetchone()[0]
    assert stored(a["id"]) == a["code"]
    j = away.post("/v1/join", json={"code": a["code"], "name": "телефон"}).json()
    assert stored(a["id"]) is None  # вошли — открытого ключа больше нет
    home.delete(f"/v1/invites/{b['id']}", headers=h)
    assert stored(b["id"]) is None  # отменили — тоже
    with conn:
        conn.execute("UPDATE invites SET expires_at = ? WHERE id = ?", (int(time.time()) - 1, c["id"]))
    guests.forget_dead_keys(conn)
    assert stored(c["id"]) is None  # истёк — тоже

    # гостю список недоступен вовсе
    assert home.get("/v1/guests", headers={"Authorization": f"Bearer {j['token']}"}).status_code == 403


def test_dead_key_is_answered_with_its_state_and_proof(tmp_path):
    """Гость не должен ждать минуту, чтобы узнать, что ключ уже использован (второй прибор — обычное дело)."""
    ident, home, away, conn, h = _env(tmp_path)
    used = home.post("/v1/invites", json={"name": "Аня"}, headers=h).json()
    revoked = home.post("/v1/invites", json={"name": "Боря"}, headers=h).json()
    expired = home.post("/v1/invites", json={"name": "Вера"}, headers=h).json()
    away.post("/v1/join", json={"code": used["code"], "name": "телефон"})
    home.delete(f"/v1/invites/{revoked['id']}", headers=h)
    with conn:
        conn.execute("UPDATE invites SET expires_at = ? WHERE id = ?", (int(time.time()) - 1, expired["id"]))

    for inv, state in ((used, "used"), (revoked, "revoked"), (expired, "expired")):
        nonce = os.urandom(16)
        r = _hello(away, inv["code"], nonce).json()
        assert r["state"] == state
        # ответу можно верить: сервер доказал, что знает этот ключ
        assert r["proof"] == auth.pair_proof(guests.invite_secret(inv["code"]), nonce, ident.fingerprint)
        # мёртвому ключу адреса и имя владельца ни к чему
        assert "net" not in r and "lan" not in r and "owner" not in r
        # и войти по нему нельзя
        assert away.post("/v1/join", json={"code": inv["code"], "name": "x"}).status_code == 403
        away.app.state.throttle = guests.Throttle()

    # живой ключ — как раньше, состояние «active»
    live = home.post("/v1/invites", json={"name": "Гена"}, headers=h).json()
    assert _hello(away, live["code"], os.urandom(16)).json()["state"] == "active"


def test_dead_key_is_answered_only_for_a_while(tmp_path):
    ident, home, away, conn, h = _env(tmp_path)
    inv = home.post("/v1/invites", json={"name": "Аня"}, headers=h).json()
    with conn:
        conn.execute("UPDATE invites SET expires_at = ? WHERE id = ?", (int(time.time()) - guests.DEAD_KEEP_S + 60, inv["id"]))
    assert _hello(away, inv["code"], os.urandom(16)).json()["state"] == "expired"
    with conn:
        conn.execute("UPDATE invites SET expires_at = ? WHERE id = ?", (int(time.time()) - guests.DEAD_KEEP_S - 60, inv["id"]))
    assert _hello(away, inv["code"], os.urandom(16)).status_code == 404


def test_only_live_keys_take_the_tunnel_dead_ones_only_a_port_label(tmp_path):
    """Стук каждые 12 с нужен только живому ключу; мёртвый — под меткой порта и редко, чтобы гость узнал о нём."""
    ident, home, away, conn, h = _env(tmp_path)
    live = home.post("/v1/invites", json={"name": "Аня"}, headers=h).json()
    dead = home.post("/v1/invites", json={"name": "Боря"}, headers=h).json()
    home.delete(f"/v1/invites/{dead['id']}", headers=h)
    assert [s for s, _ in guests.active_secrets(conn)] == [guests.invite_secret(live["code"])]
    states = {s: st for s, _, st in guests.answering(conn)}
    assert states == {guests.invite_secret(live["code"]): "active", guests.invite_secret(dead["code"]): "revoked"}
    # мёртвому — только редкое расписание, сколько бы ему ни было минут
    assert not guests.announce_due(age_s=30, since_last_s=70, alive=False)
    assert guests.announce_due(age_s=30, since_last_s=16 * 60, alive=False)


def test_list_of_answered_keys_is_bounded(tmp_path):
    """Метки в торрент-сети не должны копиться: кто-то может создавать и отменять приглашения подряд."""
    ident, home, away, conn, h = _env(tmp_path)
    for i in range(guests.ANSWERING_MAX + 5):
        inv = home.post("/v1/invites", json={"name": f"g{i}"}, headers=h).json()
        home.delete(f"/v1/invites/{inv['id']}", headers=h)
    assert len(guests.answering(conn)) == guests.ANSWERING_MAX
