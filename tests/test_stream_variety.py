"""Память о прослушанном и разброс разбора (30.10.2026): назавтра поток не повторяет вчерашний,
незнакомым трекам достаётся доля сессии; разбор идёт вразброс по медиатеке; одна запись в двух
изданиях считается один раз."""

import random

import numpy as np

from arxnova import indexer, models, nova
from tests.test_server_nova import env, needs_ffmpeg  # noqa: F401 — фикстура медиатеки

DAY = 86_400_000


def _session(lib, prof, now_ms, steps=10, seed=3, familiar=0.5):
    hist: list[int] = []
    rng = random.Random(seed)
    for _ in range(steps):
        picks = nova.next_tracks(lib, prof, current=hist[-1] if hist else None, history=hist, n=1,
                                 now_ms=now_ms, rng=rng, mode=nova.MODE_MY, start=not hist,
                                 filters=nova.Filters(familiar=familiar))
        if not picks:
            break
        hist.append(picks[0].track_id)
    return hist


def _listened(tids, ts):
    return [{"track_id": t, "event_type": "listen", "ts": ts, "is_organic": 1, "played_ratio_pct": 100,
             "skip_reason": None, "prev_track_id": None, "device_id": 1} for t in tids]


def _dnb(lib):
    """Треки DnB по альбомам: три альбома по 6 и сборник из 8 (правило «не альбомами» пускает в
    сессию не больше 2 с альбома — значит, за сессию доступны 6 альбомных + 8 со сборника)."""
    albums = [t.id for t in lib.tracks.values() if t.style == "Drum n Bass" and not t.compilation]
    va = [t.id for t in lib.tracks.values() if t.style == "Drum n Bass" and t.compilation]
    return sorted(albums), sorted(va)


@needs_ffmpeg
def test_next_day_does_not_replay_yesterday(env):
    cfg, conn, client, h = env
    lib = nova.Library(cfg, conn)
    albums, va = _dnb(lib)
    taste = _listened(albums[::6], 0)  # вкус — DnB (по треку с каждого альбома, давно)
    day1 = _session(lib, nova.Profile(lib, taste, 30 * DAY), now_ms=30 * DAY, steps=6, seed=4)
    assert len(day1) == 6 and all(t in albums + va for t in day1), day1
    # красное: без памяти о вчерашнем та же случайность даёт ту же сессию (шаг в шаг)
    again = _session(lib, nova.Profile(lib, taste, 31 * DAY), now_ms=31 * DAY, steps=6, seed=4)
    assert again == day1
    # зелёное: вчерашнее помнится — назавтра при той же случайности играет другое, и всё ещё DnB
    prof = nova.Profile(lib, taste + _listened(day1, 30 * DAY), 31 * DAY)
    day2 = _session(lib, prof, now_ms=31 * DAY, steps=6, seed=4)
    assert len(day2) == 6 and all(t in albums + va for t in day2), day2
    assert len(set(day1) & set(day2)) <= 3, (day1, day2)  # повторы «в тему» допустимы, но не вся сессия


@needs_ffmpeg
def test_repeat_in_theme_is_allowed(env):
    """Вчерашний трек, который лучше всех продолжает то, что играет сейчас, не отбрасывается."""
    cfg, conn, client, h = env
    lib = nova.Library(cfg, conn)
    albums, va = _dnb(lib)
    # сборник — своя «тема» (треки сборника близки друг к другу); вчера слушали весь сборник,
    # кроме одного трека, сегодня с него и начали
    prof = nova.Profile(lib, _listened(va[1:], 30 * DAY), 31 * DAY)
    rng = random.Random(1)
    picks = nova.next_tracks(lib, prof, current=va[0], history=[va[0]], n=3, now_ms=31 * DAY, rng=rng, mode=nova.MODE_MY)
    assert picks and picks[0].track_id in va[1:], [(p.track_id, round(p.score, 3)) for p in picks]


@needs_ffmpeg
def test_unplayed_tracks_get_their_share(env):
    cfg, conn, client, h = env
    lib = nova.Library(cfg, conn)
    albums, va = _dnb(lib)
    heard = albums + va[:4]  # слушано давно всё, кроме 4 треков сборника
    prof = nova.Profile(lib, _listened(heard, 10 * DAY), 40 * DAY)
    sess = _session(lib, prof, now_ms=40 * DAY, steps=8)
    new = [t for t in sess if t not in prof.played]
    assert len(new) >= 2, sess  # посередине ползунка — половина незнакомого; без прибавки хватило бы и знакомых
    assert all(t in va[4:] for t in new)


@needs_ffmpeg
def test_slider_extremes_are_strict(env):
    """«Новое» до упора — только неслушанное, пока оно есть; «знакомое» до упора — только слушанное."""
    cfg, conn, client, h = env
    lib = nova.Library(cfg, conn)
    albums, va = _dnb(lib)
    heard = albums + va[:4]
    prof = nova.Profile(lib, _listened(heard, 10 * DAY), 40 * DAY)
    only_new = _session(lib, prof, now_ms=40 * DAY, steps=4, familiar=0.0)
    assert only_new and all(t in va[4:] for t in only_new), only_new
    only_known = _session(lib, prof, now_ms=40 * DAY, steps=6, familiar=1.0)
    assert len(only_known) == 6 and all(t in heard for t in only_known), only_known
    # незнакомое кончилось — поток не молчит, а играет что есть
    more = _session(lib, prof, now_ms=40 * DAY, steps=8, familiar=0.0)
    assert len(more) == 8, more


@needs_ffmpeg
def test_pending_order_spreads_over_library(env):
    cfg, conn, client, h = env
    rows = indexer.pending(conn, "никакой-модели")
    ids = [r[0] for r in rows]
    assert len(ids) > 20 and ids != sorted(ids)  # не альбом за альбомом
    assert ids == [r[0] for r in indexer.pending(conn, "никакой-модели")]  # детерминированно
    assert sorted(ids) == sorted(set(ids))


@needs_ffmpeg
def test_same_recording_vector_is_copied(env):
    cfg, conn, client, h = env
    a, b = [r[0] for r in conn.execute("SELECT id FROM tracks ORDER BY id LIMIT 2")]
    with conn:
        conn.execute("UPDATE tracks SET recording_key = (SELECT recording_key FROM tracks WHERE id = ?) WHERE id = ?", (a, b))
        conn.execute("INSERT OR REPLACE INTO embeddings VALUES (?,?,?,?,?)",
                     (a, indexer.mix_id(), np.ones(4, np.float32).tobytes(), np.ones(2, np.float32).tobytes(), 1))
    assert indexer.copy_same_recording(conn, indexer.mix_id()) == 1
    got = conn.execute("SELECT vector FROM embeddings WHERE track_id = ? AND model = ?", (b, indexer.mix_id())).fetchone()
    assert got and np.frombuffer(got[0], np.float32).tolist() == [1, 1, 1, 1]
    assert indexer.copy_same_recording(conn, indexer.mix_id()) == 0  # второй раз копировать нечего
    with conn:
        conn.execute("DELETE FROM embeddings WHERE model = ?", (indexer.mix_id(),))
