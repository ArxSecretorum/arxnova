"""Фильтры запроса — в журнал и на табло (ревизия разбора 07.11.2026, docs/notes/analysis-consumers-2026-11-07.md): Demucs и CLAP
служат только фильтру «с вокалом», плиткам голоса и сценариям, а сколько раз ими пользуются, измерить было нечем."""

import json

from arxnova import config, db, nova, stats

NOW = 10_000_000_000


def test_filters_used_lists_only_what_is_set():
    assert nova.Filters().used() == {}
    f = nova.Filters(vocal="yes", bpm=(100.0, 120.0), energy=(0.2, 0.8), scenario="focus", styles=["Ambient", "Dub"], genres=["Jazz"])
    assert f.used() == {"vocal": "yes", "bpm": [100.0, 120.0], "energy": [0.2, 0.8], "scenario": "focus", "styles": 2, "genres": 1}


def test_report_counts_requests_with_filters(tmp_path):
    cfg = config.Config(data_dir=tmp_path / "data", libraries=[])
    cfg.ensure_dirs()
    conn = db.connect(cfg.db_path)
    rows = [
        {"filters": {"vocal": "yes", "bpm": [100.0, 120.0]}},
        {"filters": {"scenario": "focus"}},
        {"start": True},  # без фильтров
        None,  # журнал стёрт
    ]
    for i, detail in enumerate(rows):
        conn.execute(
            "INSERT INTO nova_requests (device_id, ts, seed_track, result, detail, is_start) VALUES (1, ?, NULL, '[]', ?, 0)",
            (NOW - i * 1000, json.dumps(detail) if detail else None),
        )
    conn.execute(  # чужой прибор и старый запрос не считаются
        "INSERT INTO nova_requests (device_id, ts, seed_track, result, detail, is_start) VALUES (2, ?, NULL, '[]', ?, 0)",
        (NOW, json.dumps({"filters": {"vocal": "no"}})),
    )
    conn.execute(
        "INSERT INTO nova_requests (device_id, ts, seed_track, result, detail, is_start) VALUES (1, ?, NULL, '[]', ?, 0)",
        (NOW - 3 * 86_400_000, json.dumps({"filters": {"vocal": "no"}})),
    )
    conn.commit()
    r = stats.report(conn, [1], days=1, now_ms=NOW + 1)
    assert r["filters"] == {"requests": 4, "with_filters": 2, "vocal": 1, "bpm": 1, "energy": 0, "scenario": 1, "styles": 0, "genres": 0}
    text = stats.format_report(r)
    assert "Фильтры в запросах" in text and "вокал 1" in text and "сценарий 1" in text
