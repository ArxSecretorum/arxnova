"""Сервер и подбор на синтетической медиатеке: настоящие FLAC (ffmpeg), эмбеддинги
подставлены вручную — так поведение правил проверяется без модели."""

import json
import random
import shutil

import numpy as np
import pytest
from fastapi.testclient import TestClient

from arxnova import auth, config, db, indexer, models, nova, scan
from arxnova.server import create_app
from tests.test_flac_keys import make_flac, needs_ffmpeg

STYLES = ["Electronic---Drum n Bass", "Electronic---Deep House", "Jazz---Contemporary Jazz"]


def fake_model(cfg):
    d = models.model_dir(cfg.models_dir, models.EFFNET)
    d.mkdir(parents=True, exist_ok=True)
    for f in models.EFFNET.files:
        (d / f.name).write_bytes(b"")
    classes = STYLES + [f"Other---S{i}" for i in range(400 - len(STYLES))]
    (d / "discogs-effnet-bsdynamic-1.json").write_text(json.dumps({"classes": classes}))
    (d / "license.json").write_text("{}")


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    if not shutil.which("ffmpeg"):
        pytest.skip("нужен ffmpeg")
    root = tmp_path_factory.mktemp("nova")
    lib = root / "lib"
    cfg = config.Config(data_dir=root / "data", libraries=[lib])
    cfg.ensure_dirs()
    fake_model(cfg)
    plan = []  # (папка, альбом, исполнитель альбома, исполнитель, стиль)
    for a in range(3):  # DnB: три альбома по 6 треков от трёх исполнителей
        for t in range(6):
            plan.append((f"dnb{a}", f"DnB Album {a}", f"Artist{a}", f"Artist{a}", 0))
    for t in range(8):  # DnB-сборник: 8 разных исполнителей
        plan.append(("va", "DnB Compilation", "Various Artists", f"VA{t}", 0))
    for t in range(10):  # хаус: 10 исполнителей-одиночек
        plan.append(("house", f"House {t}", f"House{t}", f"House{t}", 1))
    for t in range(6):
        plan.append(("jazz", "Jazz Album", "JazzMan", f"Jazz{t}", 2))
    for i, (folder, album, aa, artist, _) in enumerate(plan):
        (lib / folder).mkdir(parents=True, exist_ok=True)
        make_flac(lib / folder / f"{i:02d}.flac", seconds=0.3, freq=200 + 13 * i,
                  ARTIST=artist, ALBUMARTIST=aa, ALBUM=album, TITLE=f"T{i}", DATE="2020")
    conn = db.connect(cfg.db_path, threaded=True)
    scan.scan(conn, [lib])
    rng = np.random.default_rng(0)
    centers = rng.normal(size=(3, 64)) * 3
    album_off = {}
    rows = conn.execute("SELECT t.id, t.album, t.artist FROM tracks t ORDER BY id").fetchall()
    path_of = {r[0]: r[1] for r in conn.execute("SELECT track_id, path FROM files")}
    for r in rows:
        i = int(path_of[r["id"]].rsplit("/", 1)[1][:2])
        style = plan[i][4]
        # треки одного альбома очень близки друг к другу — соблазн играть альбомом
        off = album_off.setdefault(r["album"], rng.normal(size=64) * 1.5)
        v = centers[style] + off + rng.normal(size=64) * 0.2
        sty = np.zeros(400, np.float32)
        sty[style] = 1.0
        conn.execute("INSERT INTO embeddings VALUES (?,?,?,?,?)",
                     (r["id"], models.EFFNET.id, v.astype(np.float32).tobytes(), sty.tobytes(), 1))
    conn.commit()
    indexer.export_matrices(cfg, conn)
    app = create_app(cfg)
    client = TestClient(app)
    code = auth.new_pair_code(conn)
    tok = client.post("/v1/pair", json={"code": code, "name": "test"}).json()["token"]
    return cfg, conn, client, {"Authorization": f"Bearer {tok}"}


@needs_ffmpeg
def test_pairing(env):
    cfg, conn, client, h = env
    assert client.get("/v1/nova/facets").status_code == 401
    assert client.post("/v1/pair", json={"code": "000000", "name": "x"}).status_code == 403
    code = auth.new_pair_code(conn)
    r = client.post("/v1/pair", json={"code": code, "name": "phone"})
    assert r.status_code == 200
    # код одноразовый
    assert client.post("/v1/pair", json={"code": code, "name": "again"}).status_code == 403
    tok = r.json()["token"]
    hh = {"Authorization": f"Bearer {tok}"}
    assert client.get("/v1/library/albums", headers=hh).status_code == 200
    with conn:
        conn.execute("UPDATE devices SET revoked = 1 WHERE id = ?", (r.json()["device_id"],))
    assert client.get("/v1/library/albums", headers=hh).status_code == 401


@needs_ffmpeg
def test_events_idempotent(env):
    cfg, conn, client, h = env
    tid = conn.execute("SELECT id FROM tracks LIMIT 1").fetchone()[0]
    batch = {"events": [{"event_id": f"event-{i:04d}", "ts": 1000 + i, "track_id": tid,
                         "event_type": "listen", "played_ratio_pct": 100} for i in range(5)]}
    r1 = client.post("/v1/events", json=batch, headers=h).json()
    r2 = client.post("/v1/events", json=batch, headers=h).json()
    assert r1["accepted"] == 5 and r2 == {"accepted": 0, "duplicates": 5, "rejected": 0}
    bad = {"events": [{"event_id": "bad-0001", "ts": 1, "track_id": tid, "event_type": "explode"}]}
    assert client.post("/v1/events", json=bad, headers=h).json()["rejected"] == 1


def _stream(lib, start, steps=40, seed=1, prof=None, mode=nova.MODE_TRACK):
    prof = prof or nova.Profile(lib, [], 0)
    hist = [start]
    rng = random.Random(seed)
    for _ in range(steps):
        picks = nova.next_tracks(lib, prof, current=hist[-1], history=hist, n=1, rng=rng, mode=mode, seed=start)
        if not picks:
            break
        hist.append(picks[0].track_id)
    return hist


@needs_ffmpeg
def test_no_albums_in_stream(env, monkeypatch):
    cfg, conn, client, h = env
    lib = nova.Library(cfg, conn)
    start = next(t.id for t in lib.tracks.values() if t.album == "DnB Album 0")

    def consecutive_same_album(hist):
        return sum(1 for a, b in zip(hist, hist[1:])
                   if lib.tracks[a].album_key == lib.tracks[b].album_key and not lib.tracks[a].compilation)

    # красное: без правил альбома и исполнителя поток уходит в альбом
    monkeypatch.setattr(nova, "_album_ok", lambda *a: True)
    monkeypatch.setattr(nova, "_artist_ok", lambda *a: True)
    assert consecutive_same_album(_stream(lib, start, 10)) > 0
    monkeypatch.undo()
    # зелёное
    hist = _stream(lib, start, 30)
    assert len(hist) > 10
    assert consecutive_same_album(hist) == 0
    per_album = {}
    for t in hist:
        if not lib.tracks[t].compilation:
            per_album[lib.tracks[t].album_key] = per_album.get(lib.tracks[t].album_key, 0) + 1
    assert max(per_album.values()) <= nova.ALBUM_PER_SESSION
    # исполнитель не повторяется чаще 1 раза в K
    for i, t in enumerate(hist):
        for prev in hist[max(0, i - nova.ARTIST_GAP) : i]:
            assert not (lib.tracks[t].artists & lib.tracks[prev].artists)


@needs_ffmpeg
def test_modes_my_vs_track(env):
    """«Свой поток» идёт от вкуса, а не от текущего трека; «Похожие» держатся выбранного трека."""
    cfg, conn, client, h = env
    lib = nova.Library(cfg, conn)
    jazz = [t.id for t in lib.tracks.values() if t.style == "Contemporary Jazz"]
    house = next(t.id for t in lib.tracks.values() if t.style == "Deep House")
    now = 10**13
    evs = [{"track_id": j, "event_type": "like", "ts": now - 1000, "is_organic": 1, "played_ratio_pct": None,
            "skip_reason": None} for j in jazz[:3]]
    prof = nova.Profile(lib, evs, now)
    # красное: в старой логике (опора — предыдущий трек) после хауса шёл бы хаус
    track_mode = nova.next_tracks(lib, prof, current=house, history=[house], n=3, mode=nova.MODE_TRACK, seed=house, now_ms=now)
    assert all(lib.tracks[p.track_id].style == "Deep House" for p in track_mode)
    assert all("похоже на" in p.reason for p in track_mode)
    # зелёное: «Свой поток» после хауса ведёт к тому, что вы лайкали (джаз)
    my = nova.next_tracks(lib, prof, current=house, history=[house], n=3, mode=nova.MODE_MY, now_ms=now)
    assert sum(lib.tracks[p.track_id].style == "Contemporary Jazz" for p in my) >= 2
    assert not any("похоже на" in p.reason for p in my)


@needs_ffmpeg
def test_track_mode_does_not_drift(env):
    """«Похожие на трек»: опора не сдвигается на предыдущий трек — 20 шагов, все из жанра опоры."""
    cfg, conn, client, h = env
    lib = nova.Library(cfg, conn)
    start = next(t.id for t in lib.tracks.values() if t.style == "Contemporary Jazz")
    hist = _stream(lib, start, 20)
    assert all(lib.tracks[t].genre == lib.tracks[start].genre for t in hist)


@needs_ffmpeg
def test_replay_outweighs_skip(env):
    """Вернулись к пропущенному и дослушали: трек и его стиль получают баллы, пропуск отменён."""
    cfg, conn, client, h = env
    lib = nova.Library(cfg, conn)
    jazz = next(t.id for t in lib.tracks.values() if t.style == "Contemporary Jazz")
    ev = lambda et, ts: {"track_id": jazz, "event_type": et, "ts": ts, "is_organic": 1,
                         "played_ratio_pct": 5 if et == "skip" else 100, "skip_reason": None}
    only_skip = nova.Profile(lib, [ev("skip", 1)], 10)
    # красное: просто пропуск — трек уходит в отрицательный вектор
    assert only_skip.neg.any() and not only_skip.long.any()
    # зелёное: пропуск, затем возврат и дослушивание — отрицательного нет, вкус к джазу есть
    back = nova.Profile(lib, [ev("skip", 1), ev("replay", 5)], 10)
    assert not back.neg.any() and back.long.any()
    lst = nova.Profile(lib, [ev("listen", 5)], 10)
    assert np.linalg.norm(back.long) > np.linalg.norm(lst.long)  # возврат весит больше простого прослушивания


@needs_ffmpeg
def test_single_artist_style_does_not_go_silent(env):
    """Стиль, где все треки одного исполнителя: поток не замолкает, но альбомом подряд не идёт."""
    cfg, conn, client, h = env
    lib = nova.Library(cfg, conn)
    # хаус: 10 треков, у каждого свой исполнитель; подменим всех на одного
    house = [t for t in lib.tracks.values() if t.style == "Deep House"]
    for t in house:
        t.artists = frozenset({"one artist"})
    prof = nova.Profile(lib, [], 0)
    cur = house[0].id
    picks = nova.next_tracks(lib, prof, current=cur, history=[cur], n=3, filters=nova.Filters(styles=["Deep House"]))
    assert len(picks) == 3
    assert all(lib.tracks[p.track_id].style == "Deep House" for p in picks)
    lib2 = nova.Library(cfg, conn)  # исходные исполнители — правило «1 на K» действует как раньше
    hist = _stream(lib2, cur, 8)
    for i, t in enumerate(hist):
        for prev in hist[max(0, i - nova.ARTIST_GAP) : i]:
            assert not (lib2.tracks[t].artists & lib2.tracks[prev].artists)


@needs_ffmpeg
def test_compilation_exempt(env):
    cfg, conn, client, h = env
    lib = nova.Library(cfg, conn)
    va = [t for t in lib.tracks.values() if t.album == "DnB Compilation"]
    assert all(t.compilation for t in va)
    hist = [va[0].id, va[1].id, va[2].id]
    assert nova._album_ok(lib, va[3].id, hist)  # сборник может идти подряд
    dnb = [t for t in lib.tracks.values() if t.album == "DnB Album 1"]
    assert not nova._album_ok(lib, dnb[1].id, [dnb[0].id])


@needs_ffmpeg
def test_stream_stays_in_genre_and_filters(env):
    cfg, conn, client, h = env
    lib = nova.Library(cfg, conn)
    start = next(t.id for t in lib.tracks.values() if t.style == "Deep House")
    hist = _stream(lib, start, 6)
    assert all(lib.tracks[t].style == "Deep House" for t in hist)
    # фильтр стиля через API
    r = client.post("/v1/nova/next", headers=h, json={"mode": "track", "seed_track": start, "current_track": start,
                                                       "history": [start], "filters": {"styles": ["Contemporary Jazz"]}, "n": 3}).json()
    assert r["tracks"] and all(t["style"] == "Contemporary Jazz" for t in r["tracks"])
    assert all(t["reason"] for t in r["tracks"])
    assert all(t["why"] in ("liked", "similar", "new", "taste") for t in r["tracks"])
    assert all(t.get("seed_title") for t in r["tracks"])  # режим «Похожие» — с названием опоры


@needs_ffmpeg
def test_library_and_audio_range(env):
    cfg, conn, client, h = env
    albums = client.get("/v1/library/albums", headers=h).json()
    assert albums["total"] >= 5
    va = next(a for a in albums["items"] if a["title"] == "DnB Compilation")
    full = client.get(f"/v1/library/albums/{va['id']}", headers=h).json()
    assert len(full["tracks"]) == 8
    top = client.get("/v1/library/folders", headers=h).json()
    assert {f["name"] for f in top["folders"]} >= {"dnb0", "va", "jazz"}
    inner = client.get("/v1/library/folders", params={"path": "jazz"}, headers=h).json()
    assert len(inner["tracks"]) == 6
    assert client.get("/v1/library/search", params={"q": "jazz"}, headers=h).json()["tracks"]
    tid = full["tracks"][0]["id"]
    r = client.get(f"/v1/tracks/{tid}/audio", headers={**h, "Range": "bytes=0-99"})
    assert r.status_code == 206 and len(r.content) == 100 and r.content[:4] == b"fLaC"
    assert client.get(f"/v1/tracks/{tid}/audio").status_code == 401


@needs_ffmpeg
def test_marks_and_dislike_excluded(env):
    """Дизлайк сильнее пропуска: трек больше не подаётся; отмена возвращает его."""
    cfg, conn, client, h = env
    lib = nova.Library(cfg, conn)
    dnb = [t.id for t in lib.tracks.values() if t.style == "Drum n Bass"]
    seed = dnb[0]
    first = client.post("/v1/nova/next", headers=h, json={"current_track": seed, "history": [seed], "n": 1}).json()["tracks"][0]["id"]
    ev = lambda i, t, k: {"event_id": f"mark-{k}-{i:04d}", "ts": 10_000 + i, "track_id": t, "event_type": k}
    client.post("/v1/events", headers=h, json={"events": [ev(1, first, "dislike")]})
    m = client.get("/v1/devices/me/marks", headers=h).json()
    assert first in m["disliked"]
    for _ in range(5):
        r = client.post("/v1/nova/next", headers=h, json={"current_track": seed, "history": [seed], "n": 4}).json()
        assert first not in [t["id"] for t in r["tracks"]]
    client.post("/v1/events", headers=h, json={"events": [ev(2, first, "undislike"), ev(3, first, "like")]})
    m = client.get("/v1/devices/me/marks", headers=h).json()
    assert first in m["liked"] and first not in m["disliked"]
    fav = client.get("/v1/library/favorites", headers=h).json()["tracks"]
    assert fav[0]["id"] == first
    # гость не видит избранное владельца
    with conn:
        conn.execute("INSERT INTO pair_codes (code_hash, role, expires_at) VALUES (?, 'guest', ?)",
                     (auth._h("424242"), 10**10))
    gtok = client.post("/v1/pair", json={"code": "424242", "name": "guest"}).json()["token"]
    gh = {"Authorization": f"Bearer {gtok}"}
    assert client.get("/v1/library/favorites", headers=gh).json()["tracks"] == []


@needs_ffmpeg
def test_delete_history(env):
    cfg, conn, client, h = env
    tid = conn.execute("SELECT id FROM tracks LIMIT 1").fetchone()[0]
    client.post("/v1/events", headers=h, json={"events": [{"event_id": "hist-del-0001", "ts": 1, "track_id": tid, "event_type": "like"}]})
    dev = conn.execute("SELECT device_id FROM events WHERE event_id = 'hist-del-0001'").fetchone()[0]
    assert conn.execute("SELECT count(*) FROM events WHERE device_id = ?", (dev,)).fetchone()[0] > 0
    assert client.delete("/v1/devices/me/history", headers=h).json()["deleted"] > 0
    assert conn.execute("SELECT count(*) FROM events WHERE device_id = ?", (dev,)).fetchone()[0] == 0


@needs_ffmpeg
def test_event_before_track_links_later(env, tmp_path):
    """Событие по MD5 трека, которого ещё нет на сервере, хранится и связывается, когда трек появится."""
    cfg, conn, client, h = env
    new = make_flac(tmp_path / "new.flac", freq=1234.0, ARTIST="Late", TITLE="Arrival")
    from arxnova import flac as F
    md5 = F.read_flac(new).md5.hex()
    ev = {"events": [{"event_id": "late-event-0001", "ts": 5, "md5": md5, "event_type": "like"}]}
    assert client.post("/v1/events", json=ev, headers=h).json()["accepted"] == 1
    row = conn.execute("SELECT track_id FROM events WHERE event_id = 'late-event-0001'").fetchone()
    assert row[0] is None
    shutil.copy(new, cfg.libraries[0] / "new.flac")
    scan.scan(conn, cfg.libraries)
    row = conn.execute("SELECT track_id FROM events WHERE event_id = 'late-event-0001'").fetchone()
    assert row[0] is not None


# ---- общий доступ (§15.1, приёмка §21, п. 19–21, 27) ----

def _guest(client, h, name="Аня"):
    inv = client.post("/v1/invites", json={"name": name}, headers=h)
    assert inv.status_code == 200, inv.text
    j = client.post("/v1/join", json={"code": inv.json()["code"].lower(), "name": "phone"})
    assert j.status_code == 200, j.text
    return j.json()["device_id"], {"Authorization": f"Bearer {j.json()['token']}"}


def _clear_guests(conn, client=None):
    from arxnova import guests

    if client is not None:
        client.app.state.throttle = guests.Throttle()
        client.app.state.streams = guests.Streams()
    with conn:
        conn.execute("UPDATE devices SET revoked = 1 WHERE role = 'guest'")
        conn.execute("UPDATE invites SET revoked = 1")


@needs_ffmpeg
def test_invite_join_rights(env):
    cfg, conn, client, h = env
    _clear_guests(conn, client)
    inv = client.post("/v1/invites", json={"name": "Аня"}, headers=h).json()
    assert len(inv["code"]) == 8 and not set(inv["code"]) & set("O0I1")
    assert inv["link"].startswith("arxnova://join?") and inv["code"] in inv["link"]
    j = client.post("/v1/join", json={"code": inv["code"][:4] + "-" + inv["code"][4:], "name": "phone"}).json()
    assert j["role"] == "guest" and j["guest_name"] == "Аня"
    # код одноразовый
    assert client.post("/v1/join", json={"code": inv["code"], "name": "x"}).status_code == 403
    g = {"Authorization": f"Bearer {j['token']}"}
    # гость слушает и смотрит медиатеку, но доступом не управляет (ОД-5)
    assert client.get("/v1/library/albums", headers=g).status_code == 200
    assert client.post("/v1/invites", json={"name": "Боря"}, headers=g).status_code == 403
    assert client.get("/v1/guests", headers=g).status_code == 403
    assert client.get("/v1/devices/me", headers=g).json()["role"] == "guest"
    lst = client.get("/v1/guests", headers=h).json()
    assert [x["name"] for x in lst["guests"]] == ["Аня"] and lst["limit"] == 5


@needs_ffmpeg
def test_revoke_guest_deletes_everything(env):
    """Приёмка п. 19: после отзыва следующий запрос гостя отклонён, его данные удалены."""
    cfg, conn, client, h = env
    _clear_guests(conn, client)
    gid, g = _guest(client, h)
    tid = conn.execute("SELECT id FROM tracks LIMIT 1").fetchone()[0]
    client.post("/v1/events", json={"events": [{"event_id": "guest-like-1", "ts": 5, "track_id": tid, "event_type": "like"}]}, headers=g)
    assert client.get("/v1/library/favorites", headers=g).json()["tracks"]
    assert client.get(f"/v1/tracks/{tid}/audio", headers=g).status_code == 200
    assert client.delete(f"/v1/guests/{gid}", headers=h).status_code == 200
    assert client.get("/v1/library/albums", headers=g).status_code == 401
    assert client.get(f"/v1/tracks/{tid}/audio", headers=g).status_code == 401
    assert conn.execute("SELECT count(*) FROM events WHERE device_id = ?", (gid,)).fetchone()[0] == 0
    assert conn.execute("SELECT count(*) FROM guest_usage WHERE device_id = ?", (gid,)).fetchone()[0] == 0


@needs_ffmpeg
def test_guest_does_not_touch_owner_profile(env):
    """Приёмка п. 20: 100 лайков гостя — векторы владельца не изменились."""
    cfg, conn, client, h = env
    _clear_guests(conn, client)
    lib = nova.Library(cfg, conn)
    owner_ids = [r[0] for r in conn.execute("SELECT id FROM devices WHERE role = 'owner'")]
    before = nova.Profile(lib, nova.load_events(conn, owner_ids), 10_000)
    gid, g = _guest(client, h)
    tids = [r[0] for r in conn.execute("SELECT id FROM tracks")]
    ev = [{"event_id": f"glike-{i:04d}", "ts": 9_000 + i, "track_id": tids[i % len(tids)], "event_type": "like"} for i in range(100)]
    assert client.post("/v1/events", json={"events": ev}, headers=g).json()["accepted"] == 100
    after = nova.Profile(lib, nova.load_events(conn, owner_ids), 10_000)
    assert np.array_equal(before.long, after.long) and np.array_equal(before.neg, after.neg)
    # «Избранное» у каждого своё
    assert not set(client.get("/v1/devices/me/marks", headers=h).json()["liked"]) >= set(tids[:5])
    assert set(client.get("/v1/devices/me/marks", headers=g).json()["liked"]) >= set(tids[:5])


@needs_ffmpeg
def test_join_bruteforce_blocked(env):
    """Приёмка п. 21: после N ошибок адрес блокируется, действующий код не угадан."""
    cfg, conn, client, h = env
    _clear_guests(conn, client)
    inv = client.post("/v1/invites", json={"name": "Цель"}, headers=h).json()
    codes = [r.status_code for r in (client.post("/v1/join", json={"code": f"ZZZZ{i:04d}"}) for i in range(10))]
    assert codes[:5] == [403] * 5 and set(codes[5:]) == {429}
    # заблокирован адрес — даже верный код с него не проходит
    assert client.post("/v1/join", json={"code": inv["code"]}).status_code == 429


@needs_ffmpeg
def test_guest_cap_is_hard(env):
    """Приёмка п. 27: шестое приглашение отклоняется при любой конфигурации."""
    cfg, conn, client, h = env
    _clear_guests(conn, client)
    cfg.max_guests = 100
    try:
        for i in range(5):
            assert client.post("/v1/invites", json={"name": f"g{i}"}, headers=h).status_code == 200
        r = client.post("/v1/invites", json={"name": "шестой"}, headers=h)
        assert r.status_code == 409 and "5" in r.json()["detail"]
    finally:
        cfg.max_guests = 5


@needs_ffmpeg
def test_guest_streams_and_volume(env):
    cfg, conn, client, h = env
    _clear_guests(conn, client)
    tid = conn.execute("SELECT id FROM tracks LIMIT 1").fetchone()[0]
    gs = [_guest(client, h, f"g{i}")[1] for i in range(3)]
    assert client.get(f"/v1/tracks/{tid}/audio", headers=gs[0]).status_code == 200
    assert client.get(f"/v1/tracks/{tid}/audio", headers=gs[1]).status_code == 200
    assert client.get(f"/v1/tracks/{tid}/audio", headers=gs[2]).status_code == 429  # третий гость ждёт
    assert client.get(f"/v1/tracks/{tid}/audio", headers=h).status_code == 200  # владельца не ограничиваем
    # владелец слушает — гостям уступают: место одно, и оно уже занято первым гостем
    client.app.state.streams = __import__("arxnova.guests", fromlist=["Streams"]).Streams()
    client.get(f"/v1/tracks/{tid}/audio", headers=h)
    assert client.get(f"/v1/tracks/{tid}/audio", headers=gs[0]).status_code == 200
    assert client.get(f"/v1/tracks/{tid}/audio", headers=gs[1]).status_code == 429
    cfg.guest_daily_mb = 0
    try:
        gid = client.get("/v1/devices/me", headers=gs[0]).json()["id"]
        with conn:
            conn.execute("INSERT OR REPLACE INTO guest_usage VALUES (?, strftime('%Y-%m-%d','now','localtime'), ?)", (gid, 10 << 30))
        cfg.guest_daily_mb = 1024
        assert client.get(f"/v1/tracks/{tid}/audio", headers=gs[0]).status_code == 429
    finally:
        cfg.guest_daily_mb = 4096


@needs_ffmpeg
def test_inviter_name_only_by_owner_choice(env):
    """«Кто пригласил» — только имя, которое владелец указал сам; из ОС ничего не берётся."""
    cfg, conn, client, h = env
    _clear_guests(conn, client)
    cfg.owner_name = ""
    inv = client.post("/v1/invites", json={"name": "Аня"}, headers=h).json()
    from urllib.parse import parse_qs, urlsplit

    assert parse_qs(urlsplit(inv["link"]).query).get("by", [""]) == [""]
    j = client.post("/v1/join", json={"code": inv["code"]}).json()
    assert j["owner"] == ""
    inv = client.post("/v1/invites", json={"name": "Боря", "owner_name": "Рома"}, headers=h).json()
    assert client.post("/v1/join", json={"code": inv["code"]}).json()["owner"] == "Рома"
    assert client.get("/v1/guests", headers=h).json()["owner_name"] == "Рома"
    cfg.owner_name = ""


@needs_ffmpeg
def test_play_folder_recursive(env):
    cfg, conn, client, h = env
    root = client.get("/v1/library/folders", headers=h).json()
    allr = client.get("/v1/library/folders/tracks", headers=h).json()["tracks"]
    assert len(allr) == conn.execute("SELECT count(*) FROM tracks").fetchone()[0]
    one = root["folders"][0]["name"]
    sub = client.get("/v1/library/folders/tracks", params={"path": one}, headers=h).json()["tracks"]
    assert 0 < len(sub) < len(allr) and len(sub) == root["folders"][0]["tracks"]


@needs_ffmpeg
def test_mixes_only_from_device(env):
    """ФТ-4.7: подборки только из треков прибора, по 25 и меньше, без двух треков альбома подряд."""
    cfg, conn, client, h = env
    rows = conn.execute("SELECT id, md5, album FROM tracks ORDER BY id").fetchall()
    on = rows[::2]  # на приборе — половина медиатеки
    r = client.put("/v1/devices/me/catalog", json=[bytes(x["md5"]).hex() for x in on], headers=h).json()
    assert r["matched"] == len(on)
    with conn:
        conn.execute("DELETE FROM device_mixes")
    out = client.get("/v1/devices/me/mixes", headers=h).json()
    assert out["mixes"], out
    ids = {x["id"] for x in on}
    albums = {x["id"]: x["album"] for x in rows}
    for m in out["mixes"]:
        tr = [t["id"] for t in m["tracks"]]
        assert set(tr) <= ids and 5 <= len(tr) <= 25 and len(set(tr)) == len(tr)
        assert all(t["md5"] and "cue_ms" in t for t in m["tracks"])
    # кэш: повторный запрос не пересобирает
    again = client.get("/v1/devices/me/mixes", headers=h).json()
    assert again["built_at"] == out["built_at"]
    # мало треков на приборе — подборок нет
    client.put("/v1/devices/me/catalog", json=[bytes(on[0]["md5"]).hex()], headers=h)
    assert client.get("/v1/devices/me/mixes", headers=h).json()["mixes"] == []


@needs_ffmpeg
def test_similar_is_positive(env):
    """«Похожие» на треке — положительный сигнал, а не пропуск."""
    cfg, conn, client, h = env
    lib = nova.Library(cfg, conn)
    tid = next(iter(lib.tracks))
    ev = [{"event_id": "sim-0001", "ts": 1000, "track_id": tid, "event_type": "similar"}]
    assert client.post("/v1/events", json={"events": ev}, headers=h).json()["accepted"] == 1
    w = nova.Profile._weight({"event_type": "similar", "is_organic": 1, "played_ratio_pct": None})
    assert w > 0


def test_parallel_requests_share_connection_safely(env):
    """Обработчики идут в пуле потоков, соединение с базой одно: без замка параллельные
    /v1/health и списки ломали курсор (500 «NoneType is not subscriptable»)."""
    from concurrent.futures import ThreadPoolExecutor

    _, _, client, h = env

    def hit(i):
        r = client.get("/v1/health") if i % 2 else client.get("/v1/library/albums", headers=h)
        return r.status_code

    with ThreadPoolExecutor(8) as pool:
        codes = list(pool.map(hit, range(400)))
    assert set(codes) == {200}


def test_health_does_not_build_library(env):
    """«Жив ли сервер» отвечает, не строя модель подбора (на большой медиатеке — секунды;
    окно приложения ждёт 5 с и раньше перезапускало сервер по кругу)."""
    cfg, _, client, _ = env
    st = client.app.state.nova
    st._lib = None
    called = []
    orig = st.library
    st.library = lambda: called.append(1) or orig()
    try:
        h = client.get("/v1/health").json()
    finally:
        st.library = orig
    assert not called and h["indexed"] > 0


def test_sound_counts_within_style(env):
    """Числа у фильтров звука — как у фильтра: внутри выбранного стиля их не больше, чем всего,
    а список стилей остаётся полным."""
    _, _, client, h = env
    whole = client.get("/v1/nova/facets?min_tracks=1", headers=h).json()
    style = whole["styles"][0]["name"]
    inside = client.get(f"/v1/nova/facets?min_tracks=1&styles={style}", headers=h).json()
    assert inside["styles"] == whole["styles"]
    for k in ("yes", "no"):
        assert inside["sound"]["vocal"][k] <= whole["sound"]["vocal"][k]
    assert sum(b["tracks"] for b in inside["sound"]["energy"]) <= sum(b["tracks"] for b in whole["sound"]["energy"])
    assert inside["sound"]["analysed"] <= whole["sound"]["analysed"]
    other = client.get("/v1/nova/facets?min_tracks=1&styles=Nonexistent", headers=h).json()
    assert other["sound"]["analysed"] == 0
