"""Пропуск после перемотки вперёд — отказ, а не «почти дослушал» (решение 0020).

16.11.2026: Ulver «1969» — пропуск на 61 % при ~40 с звука (две перемотки вперёд). Доля после перемотки — место, докуда домотали,
а не сколько слушали; сервер засчитал «принят» и обнулил серию пропусков."""

from __future__ import annotations

from arxnova import db, nova


def test_skip_after_seeking_forward_is_a_refusal(tmp_path):
    conn = db.connect(tmp_path / "t.sqlite3")
    conn.execute("INSERT INTO devices (id, name, role, token_hash) VALUES (1, 'm9', 'owner', 'x')")
    rows = [("a", 61, 2, "skip"), ("b", 61, None, "skip"), ("c", 100, 3, "listen")]
    for i, (eid, pct, fwd, et) in enumerate(rows):
        conn.execute("INSERT INTO events (event_id, device_id, ts, md5, event_type, played_ratio_pct, is_organic, seeks_forward) "
                     "VALUES (?, 1, ?, ?, ?, ?, 0, ?)", (eid, i, bytes(16), et, pct, fwd))
    got = {r["event_id"]: r["played_ratio_pct"] for r in nova.load_events(conn, 1)}
    assert got == {"a": nova.LATE_SKIP_PCT - 1, "b": 61, "c": 100}
    assert {r["event_id"]: r["played_ratio_pct"] for r in nova.session_events(conn, [1])}["a"] == nova.LATE_SKIP_PCT - 1
