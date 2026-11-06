"""Таблица «кто слушает вместе» и чередование (решение 0015, владелец 06.11.2026: «Да, готовь»).

Таблица ищет исполнителя так же, как её собирала сводка дампа; служебные имена не исполнители. В потоке рукав один на запрос,
журнал хранит рукав и выбор другого рукава при том же жребии; без рукава подбор прежний. Отчёт сравнивает только места, где
рукава разошлись."""

from __future__ import annotations

import importlib.util
import json
import pathlib
import random

import numpy as np
import pytest

from arxnova import db, listeners, nova, stats
from tests.test_bridges_stream import NOW, _event, _listened, make_library

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _unit(m):
    return m / np.linalg.norm(m, axis=-1, keepdims=True)


def _lb_norm():
    path = ROOT / "bench" / "eval_goal" / "lb_norm.py"
    if not path.exists():  # публичный выпуск идёт без bench/: сверка со сводкой дампа — только в рабочем репозитории
        pytest.skip("нет bench/eval_goal/lb_norm.py")
    spec = importlib.util.spec_from_file_location("lb_norm", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


NAMES = ["Ólafur Arnalds", "The Prodigy", "S.P.Y", "Nu:Tone", "Camo & Krooked", "London Elektricity, Elsa Esmeralda",
         "DJ Marky feat. Stamina MC", "Calibre (ft. DRS)", "Мельница", "Ёлка", "Йорш", "Björk", "Sunn O)))", "Øystein Sevåg",
         "A x B", "AC/DC", "Florence + The Machine", "Various Artists", "", "山岡晃"]


def test_normalization_matches_the_dump_aggregation():
    """Ключ строки таблицы — хэш имени после свёртки сводки дампа: разойдутся — исполнители перестанут находиться."""
    ref = _lb_norm()
    for raw in NAMES:
        assert listeners.artist_key(raw) == ref.artist_key(raw), raw
        assert listeners.artist_parts(raw) == ref.artist_parts(raw), raw
        assert listeners.h64(listeners.artist_key(raw)) == ref.h64(ref.artist_key(raw)), raw


def _table(vectors: dict[str, np.ndarray]) -> listeners.Table:
    items = sorted((listeners.h64(listeners.key(k)), np.round(_unit(v) * 127).astype(np.int8)) for k, v in vectors.items())
    return listeners.Table(np.array([h for h, _ in items], np.uint64), np.stack([v for _, v in items]))


def test_lookup_whole_name_then_participants_and_never_service_names():
    rng = np.random.default_rng(1)
    v = {name: rng.normal(size=8) for name in ("Calibre", "Lenzman", "Camo", "Krooked", "Various Artists")}
    t = _table(v)
    cos = lambda a, b: float(a @ _unit(b))
    assert cos(t.vector("Calibre"), v["Calibre"]) > 0.99
    assert cos(t.vector("Calibre feat. DRS"), v["Calibre"]) > 0.99  # гость не автор
    both = t.vector("Camo & Krooked")  # сочетания в таблице нет — среднее участников
    assert cos(both, _unit(v["Camo"]) + _unit(v["Krooked"])) > 0.99
    assert t.vector("Various Artists") is None  # служебное имя: вектор сборников бессмыслен
    assert t.vector("Nobody Here") is None and t.vector(None) is None and t.vector("  ") is None
    m, ok, rows = t.library([(None, "Lenzman"), ("Nobody", "Calibre"), ("Various Artists", None), ("Camo & Krooked", None)])
    assert ok.tolist() == [True, True, False, True]
    assert rows[0] == t.lookup("Lenzman")[1] and len(rows[0]) == 1 and rows[2] == frozenset()
    assert rows[3] == t.lookup("Camo")[1] | t.lookup("Krooked")[1]  # сочетание — строки участников
    assert cos(m[0], v["Lenzman"]) > 0.99 and cos(m[1], v["Calibre"]) > 0.99


def test_shipped_table_knows_the_owners_scene():
    """Файл таблицы едет с сервером: без него чередование молча выключено. Соседи — как на проверке дампа."""
    t = listeners.Table.load()
    assert t is not None and t.vecs.shape[1] == 128 and len(t.keys) > 100_000
    calibre, lenzman, wardruna = (t.vector(x) for x in ("Calibre", "Lenzman", "Wardruna"))
    assert calibre @ lenzman > calibre @ wardruna + 0.2
    assert t.vector("Various Artists") is None


def _with_listen(lib, scenes=6, dims=16, seed=11):
    rng = np.random.default_rng(seed)
    centers = _unit(rng.normal(size=(scenes, dims)))
    scene = rng.integers(0, scenes, size=len(lib.ids))
    lib.listen = _unit(centers[scene] + 0.3 * rng.normal(size=(len(lib.ids), dims))).astype(np.float32)
    lib.listen_ok = np.ones(len(lib.ids), bool)
    lib.listen_rows = [frozenset({i}) for i in range(len(lib.ids))]  # у каждого трека свой исполнитель в таблице
    return lib, scene


def _run(lib, listen, batches=25, seed=3):
    rng, events, hist, traces, picks = random.Random(seed), [], [], [], []
    base = _listened(range(1, 41), NOW - 86_400_000)
    now = NOW
    for _ in range(batches):
        mood = nova.Mood.from_events(lib, list(reversed(events)), now)
        prof = nova.Profile(lib, base + events, now)
        trace: dict = {}
        batch = nova.next_tracks(lib, prof, current=hist[-1] if hist else None, history=hist[-150:], n=3, now_ms=now, rng=rng,
                                 mode=nova.MODE_MY, start=not hist, mood=mood, trace=trace, listen=listen)
        assert batch
        traces.append((trace, batch, prof))
        for p in batch:
            picks.append(p)
            hist.append(p.track_id)
            now += 60_000
            events.append(_event(p.track_id, "listen", now, 100))
    return traces, picks


def test_without_trial_the_stream_is_unchanged():
    lib, _ = _with_listen(make_library())
    plain = [p.track_id for p in _run(make_library(), None)[1]]
    traces, picks = _run(lib, None)
    assert [p.track_id for p in picks] == plain  # таблица есть, но без рукава выбор тот же
    assert all("listen" not in t for t, _, _ in traces) and all(p.phi is None for p in picks)


def test_zero_weight_arms_choose_the_same(monkeypatch):
    """Выбор другого рукава считается при том же жребии: при нулевом весе он обязан совпасть — иначе расхождения были бы шумом."""
    monkeypatch.setattr(nova, "LISTEN_W", 0.0)
    lib, _ = _with_listen(make_library())
    traces, _ = _run(lib, "B")
    assert all(t["listen"]["arm"] == "B" and t["listen"]["main"] == t["listen"]["counter"] for t, _, _ in traces)


def test_arm_b_leans_to_what_is_heard_together_and_the_journal_keeps_both():
    """С таблицей выбранное ближе по слушателям к сессии и вкусу; журнал хранит рукав, оба выбора и близости."""
    lib, _ = _with_listen(make_library())
    closeness = {}
    for arm in ("A", "B"):
        traces, _ = _run(lib, arm, batches=30)
        main = [p for t, batch, _ in traces for p in batch if p.track_id in t["listen"]["main"]]
        closeness[arm] = np.mean([sum(p.phi) for p in main])
        assert all(t["listen"]["arm"] == arm and len(t["listen"]["main"]) and len(t["listen"]["counter"]) for t, _, _ in traces)
        # свой main в журнале — основной стиль сессии; чередование его не затирает
        assert all(isinstance(t.get("main"), str) for t, _, _ in traces if "leaving" in t)
        assert all(("listen" in p.parts) == (arm == "B") for p in main)
        t, batch, prof = traces[-1]
        rec = nova.journal(lib, prof, batch, t)
        assert rec["listen"] == t["listen"] and all("phi" in x for x in rec["picks"] if x["t"] in t["listen"]["main"])
    assert closeness["B"] > closeness["A"] + 0.1, closeness


def test_trial_draws_an_arm_per_request():
    lib, _ = _with_listen(make_library())
    prof = nova.Profile(lib, _listened(range(1, 41), NOW - 86_400_000), NOW)
    rng = random.Random(5)
    arms = []
    for _ in range(200):
        trace: dict = {}
        nova.next_tracks(lib, prof, current=None, history=[], n=3, now_ms=NOW, rng=rng, mode=nova.MODE_MY, start=True,
                         trace=trace, listen="trial")
        arms.append(trace["listen"]["arm"])
    assert 70 <= arms.count("B") <= 130, arms.count("B")


def test_trial_report_compares_only_where_the_arms_differ(tmp_path):
    conn = db.connect(tmp_path / "t.sqlite3")
    conn.execute("PRAGMA foreign_keys=OFF")
    t0 = NOW
    reqs = [(1, "A", [10, 11], [12, 11]), (2, "B", [20, 21, 30], [22, 23])]
    for rid, arm, main, counter in reqs:
        conn.execute("INSERT INTO nova_requests (id, device_id, ts, result, detail) VALUES (?,?,?,?,?)",
                     (rid, 1, t0, "[]", json.dumps({"v": "drift-1", "main": "Drum n Bass", "listen": {"arm": arm, "w": 0.1, "main": main, "counter": counter}})))
    ev = [(10, "listen", 100, 0, 1), (11, "skip", 5, 0, 1),  # A: разошлось и дослушано; совпало — не в сравнении
          (20, "skip", 5, 0, 2), (21, "listen", 100, 0, 2),  # B: разошлось — ранний пропуск и дослушано
          (30, "listen", 100, 0, 2),  # третьего места у другого рукава нет — не в сравнении
          (12, "listen", 100, 1, 1)]  # выбор слушателя — не в сравнении
    for i, (tid, et, pct, organic, rid) in enumerate(ev):
        conn.execute("INSERT INTO events (event_id, device_id, ts, md5, track_id, event_type, played_ratio_pct, is_organic, nova_request_id)"
                     " VALUES (?,?,?,?,?,?,?,?,?)", (f"e{i}", 1, t0 + 60_000 * (i + 1), b"x", tid, et, pct, organic, rid))
    conn.commit()
    r = stats.trial_report(conn, [1], now_ms=t0 + 86_400_000)
    assert r["requests"] == 2
    assert r["A"] == {"differ": 1, "same": 1, "done": 1.0, "early": 0.0}
    assert r["B"] == {"differ": 2, "same": 0, "done": 0.5, "early": 0.5}
    assert "done_diff" not in r  # мало показов — разность не печатается
    assert "B — с таблицей" in stats.format_trial(r)


@pytest.mark.parametrize("value", [True, False])
def test_config_switch_round_trips(tmp_path, value):
    from arxnova import config

    cfg = config.Config(data_dir=tmp_path)
    cfg.listen_trial = value
    cfg.save()
    assert config.load(tmp_path).listen_trial is value


def test_installers_carry_the_table():
    """Установщики собирает PyInstaller: файлы данных пакета он берёт, только если их назвать."""
    spec = (ROOT / "packaging" / "arxnova.spec").read_text()
    assert 'collect_data_files("arxnova")' in spec


def _mood(lib, plays, now):
    """plays: (трек, событие, доля, время) — как события устройства, новые сверху."""
    events = [{"track_id": t, "event_type": et, "played_ratio_pct": pct, "ts": ts, "is_organic": 0, "skip_reason": None,
               "left_by": None, "prev_track_id": None, "device_id": 1} for t, et, pct, ts in plays]
    return nova.Mood.from_events(lib, list(reversed(events)), now)


def test_start_has_no_session_closeness():
    """Плеер на старте шлёт в истории прошлые запуски (до 400 треков): своего запуска у старта ещё нет, φ_сессия — 0."""
    lib, _ = _with_listen(make_library())
    hist = list(range(1, 41))
    mood = _mood(lib, [(t, "listen", 100, NOW - 3_600_000 + 60_000 * i) for i, t in enumerate(hist)], NOW)
    prof = nova.Profile(lib, _listened(hist, NOW - 3_600_000), NOW)
    trace: dict = {}
    picks = nova.next_tracks(lib, prof, current=hist[-1], history=hist, n=4, now_ms=NOW, rng=random.Random(2), mode=nova.MODE_MY,
                             start=True, mood=mood, trace=trace, listen="B")
    assert picks and trace["listen"]["arm"] == "B"
    assert all(p.phi[0] == 0.0 for p in picks if p.phi), [p.phi for p in picks]
    assert any(p.phi and p.phi[1] != 0.0 for p in picks)  # вкус на старте есть


def test_same_artist_is_a_shared_table_row_not_a_similar_name():
    """«Тот же исполнитель» — общая строка таблицы: Esbjörn и Esbjorn Svensson Trio — одна строка, хотя имена в медиатеке разные."""
    lib, _ = _with_listen(make_library())
    u, w = np.eye(16, dtype=np.float32)[0], np.eye(16, dtype=np.float32)[1]
    played, other, same_row, twin = 1, 2, 3, 4  # у twin тот же вектор, что у played, но своя строка таблицы
    for t, v, row in ((played, u, 900), (other, w, 901), (same_row, u, 900), (twin, u, 902)):
        lib.listen[lib.row[t]] = v
        lib.listen_rows[lib.row[t]] = frozenset({row})
    center = nova._listen_center(lib, {played: 1.0, other: 1.0})
    assert nova._listen_phi(lib, same_row, center) == pytest.approx(0.0, abs=1e-6)  # своя строка вычтена: остаётся w
    assert nova._listen_phi(lib, twin, center) == pytest.approx(2 ** -0.5, abs=1e-6)  # чужая строка не вычитается


def test_run_boundary_ignores_a_track_repeated_inside_the_run():
    """[P1, X, P2 | R1, R2, X, R3]: повтор X внутри запуска не отрезает R1 и R2 от центра сессии (и от дрейфа)."""
    lib = make_library()
    P1, X, P2, R1, R2, R3 = 1, 2, 3, 4, 5, 6
    hist = [P1, X, P2, R1, R2, X, R3]
    plays = [(t, "listen", 100, NOW - 600_000 + 60_000 * i) for i, t in enumerate(hist)]
    mood = _mood(lib, plays, NOW)
    since = nova._run_since(hist, 3, mood)
    assert nova._heard_in_run(hist[3:], mood, since) == [R1, R2, X, R3]


def test_editions_count_once_in_taste():
    """Масса прослушиваний копируется на все издания записи: в центре вкуса запись — один раз, по лучшему изданию."""
    import types

    lib = types.SimpleNamespace(best_edition={10: 10, 11: 10})
    prof = types.SimpleNamespace(fam_mass={10: 1.0, 11: 1.0, 20: 0.5}, liked={30}, disliked={20})
    assert nova._listen_taste_weights(lib, prof) == {10: 1.0, 11: 0.0, 30: nova.LISTEN_LIKE_W}


def test_trial_report_counts_outcomes_the_way_the_player_sends_them(tmp_path):
    """Плеер шлёт лайк без запроса, «Похожие» и возврат — как свой выбор без запроса, «Не предлагать» — с запросом и без пропуска
    (NovaStreamer.kt). Исход показа — как в табло: возврат, лайк и «Похожие» — дослушан, «Не предлагать» — ранний пропуск."""
    conn = db.connect(tmp_path / "t.sqlite3")
    conn.execute("PRAGMA foreign_keys=OFF")
    conn.execute("INSERT INTO nova_requests (id, device_id, ts, result, detail) VALUES (?,?,?,?,?)",
                 (1, 1, NOW, "[]", json.dumps({"v": "drift-1", "listen": {"arm": "B", "w": 0.1, "main": [40, 41, 42, 43],
                                                                           "counter": [50, 51, 52, 53]}})))
    ev = [(40, "skip", 5, 0, 1), (40, "replay", 100, 1, None),  # бросил и вернулся дослушать
          (41, "skip", 50, 0, 1), (41, "like", None, 0, None),  # лайк без запроса
          (42, "dislike", None, 0, 1),  # «Не предлагать»: пропуска после него нет
          (43, "skip", 30, 0, 1), (43, "similar", None, 1, None)]  # «Похожие» — как свой выбор
    for i, (tid, et, pct, organic, rid) in enumerate(ev):
        conn.execute("INSERT INTO events (event_id, device_id, ts, md5, track_id, event_type, played_ratio_pct, is_organic, nova_request_id)"
                     " VALUES (?,?,?,?,?,?,?,?,?)", (f"p{i}", 1, NOW + 30_000 * (i + 1), b"x", tid, et, pct, organic, rid))
    conn.commit()
    r = stats.trial_report(conn, [1], now_ms=NOW + 86_400_000)
    assert r["B"] == {"differ": 4, "same": 0, "done": 0.75, "early": 0.25}, r
    # табло прежнее: те же показы, «Не предлагать» в нём не считается
    assert stats.report(conn, [1], days=2, now_ms=NOW + 86_400_000)["total"] == [0, 0, 0, 0, 3]
