"""Шлюз М6 (ФТ-6.1–6.8) на синтетической медиатеке. Приёмка §21, п. 5: заведомо плохая пара —
близкий микс, но несовпавший вокал — без ФТ-6.2 проходит, со шлюзом отклонена с причиной."""

from __future__ import annotations

import numpy as np
import pytest

from arxnova import gate as G

COLS = ["bpm", "bpm_conf", "musical_key", "key_conf", "loudness_db", "onset_rate", "vocal_frac", "vocal_ratio"]
DIM = 16


def _unit(v):
    return v / np.linalg.norm(v)


def make(rng_seed=0, n_per=30):
    """Два стиля: «A» и «B», по n_per треков. Трек 0 стиля A и трек 0 стиля B — мост: близки
    миксом и всеми фасетами. Трек 1 стиля B — ловушка: микс как у моста, вокал другой."""
    rng = np.random.default_rng(rng_seed)
    ca, cb = _unit(rng.normal(size=DIM)), _unit(rng.normal(size=DIM))
    n = 2 * n_per
    styles = ["A"] * n_per + ["B"] * n_per
    mix = np.stack([_unit((ca if s == "A" else cb) + 0.35 * rng.normal(size=DIM)) for s in styles])
    fac = {f: np.stack([_unit((ca if s == "A" else cb) + 0.35 * rng.normal(size=DIM)) for s in styles]) for f in G.FACETS}
    a0, b0, trap = 0, n_per, n_per + 1
    mid = _unit(ca + cb)
    for i in (a0, b0, trap):
        mix[i] = _unit(mid + 0.01 * rng.normal(size=DIM))
    for f in G.FACETS:
        for i in (a0, b0):
            fac[f][i] = _unit(mid + 0.01 * rng.normal(size=DIM))
        fac[f][trap] = fac[f][b0]
    fac["vocal"][trap] = _unit(-mid)  # совсем другой голос
    feats = np.zeros((n, len(COLS)), np.float32)
    feats[:, 0] = 120.0
    feats[:, 1] = 0.9
    feats[:, 4] = -10.0
    feats[:, 5] = 4.0
    feats[:, 6] = 0.6  # у всех есть вокал
    scores = np.zeros((n, 3), np.float32)
    scores[:, 0] = 0.6
    scores[:, 1] = 0.1
    return mix, styles, scores, fac, feats, (a0, b0, trap)


def build(mix, styles, scores, fac, feats):
    # микс в Library уже нормирован и центрирован; здесь — как есть (единичные векторы)
    return G.Gate(mix.astype(np.float32), styles, scores, fac, feats, COLS)


def test_bridge_passes_and_trap_rejected_on_vocal(monkeypatch):
    mix, styles, scores, fac, feats, (a0, b0, trap) = make()
    g = build(mix, styles, scores, fac, feats)
    assert g.check(a0, b0, boldness=0.5).ok
    v = g.check(a0, trap, boldness=0.5)
    assert not v.ok and v.failed == G.COND_FACET and v.detail == "vocal"
    # красное: без ФТ-6.2 (фасеты не проверяются) ловушка проходит — проверка что-то различает
    monkeypatch.setattr(G, "FACETS", ())
    assert build(mix, styles, scores, fac, feats).check(a0, trap, boldness=0.5).ok


def test_tempo_half_double_and_mismatch():
    mix, styles, scores, fac, feats, (a0, b0, _) = make()
    feats[b0, 0] = 240.0  # двойной темп — совместим
    assert build(mix, styles, scores, fac, feats).check(a0, b0).ok
    feats[b0, 0] = 140.0  # 120 → 140: несовместим
    v = build(mix, styles, scores, fac, feats).check(a0, b0)
    assert v.failed == G.COND_TEMPO


def test_vocal_presence_must_match():
    mix, styles, scores, fac, feats, (a0, b0, _) = make()
    feats[b0, 6] = 0.0  # инструментал
    assert build(mix, styles, scores, fac, feats).check(a0, b0).failed == G.COND_VOCAL


def test_unconfident_classifier_blocks():
    mix, styles, scores, fac, feats, (a0, b0, _) = make()
    scores[b0] = [0.3, 0.28, 0.0]  # два стиля почти поровну
    assert build(mix, styles, scores, fac, feats).check(a0, b0).failed == G.COND_CONF


def test_no_facets_blocks_cross_style():
    mix, styles, scores, fac, feats, (a0, b0, _) = make()
    g = G.Gate(mix.astype(np.float32), styles, scores, {}, feats, COLS)
    assert g.check(a0, b0).failed == G.COND_NOFACETS


def test_forbidden_pair():
    mix, styles, scores, fac, feats, (a0, b0, _) = make()
    g = G.Gate(mix.astype(np.float32), styles, scores, fac, feats, COLS, forbid={frozenset(("A", "B"))})
    assert g.check(a0, b0).failed == G.COND_PAIR


def test_boldness_moves_threshold():
    mix, styles, scores, fac, feats, _ = make()
    g = build(mix, styles, scores, fac, feats)
    assert g.threshold("mix", g.pct(1.0)) < g.threshold("mix", g.pct(0.0))
    assert g.pct(1.0) < g.pct(0.5) < g.pct(0.0)


@pytest.mark.parametrize("hist,ok", [
    (["A"] * 10, True),
    (["A"] * 5 + ["B"] * 2, False),  # переход только что — держимся стиля
    (["A"] * 5 + ["B"] * G.N_CROSS, True),
])
def test_one_cross_per_n(hist, ok):
    assert G.cross_allowed_now(hist) is ok


def _stub_lib():
    from types import SimpleNamespace as NS

    tracks = {1: NS(style="A", recording_key=b"r1"), 2: NS(style="B", recording_key=b"r2"),
              3: NS(style="A", recording_key=b"r3")}
    return NS(vec=np.eye(3, DIM, dtype=np.float32), row={1: 0, 2: 1, 3: 2}, tracks=tracks)


def _skip(tid, prev, reason=None, pct=5):
    return {"track_id": tid, "event_type": "skip", "ts": 1000, "is_organic": 0, "played_ratio_pct": pct,
            "skip_reason": reason, "prev_track_id": prev}


def test_skip_feedback_tightens_only_named_facet():
    from arxnova import nova

    lib = _stub_lib()
    p = nova.Profile(lib, [_skip(2, 1, "vocal")], 2000)
    assert p.pair_tighten[frozenset(("A", "B"))] == {"vocal": nova.GATE_STEP}
    p = nova.Profile(lib, [_skip(2, 1, "genre")], 2000)
    assert set(p.pair_tighten[frozenset(("A", "B"))]) == {"mix", "rhythm", "instr", "vocal"}
    # «не сейчас» шлюз не трогает (ФТ-8.5) — красное было бы: любой пропуск ужесточает пару
    assert nova.Profile(lib, [_skip(2, 1, "not_now")], 2000).pair_tighten == {}
    # внутри стиля «не тот ритм» — штраф паре записей, а не пороги
    p = nova.Profile(lib, [_skip(3, 1, "rhythm", pct=60)], 2000)
    assert p.pair_tighten == {} and p.pair_penalty[(1, 3)] == 1


def test_tired_doubles_repeat_interval():
    from arxnova import nova

    lib = _stub_lib()
    hour = 3_600_000
    listen = {"track_id": 1, "event_type": "listen", "ts": 1, "is_organic": 0, "played_ratio_pct": 100,
              "skip_reason": None, "prev_track_id": None}
    now = (nova.REPEAT_HOURS + 1) * hour
    assert nova._recording_ok(lib, 1, [], nova.Profile(lib, [listen], now), now)
    tired = dict(_skip(1, None, "tired"), ts=1)
    assert not nova._recording_ok(lib, 1, [], nova.Profile(lib, [listen, tired], now), now)


def test_filters_vocal_bpm_energy():
    from types import SimpleNamespace as NS

    from arxnova import nova

    mix, styles, scores, fac, feats, _ = make()
    feats[:10, 6] = 0.0  # первые 10 — инструменталы
    feats[:, 0] = 87.0
    feats[20:, 0] = 174.0
    g = build(mix, styles, scores, fac, feats)
    lib = NS(gate=g, row={i: i for i in range(len(styles))},
             tracks={i: NS(style=s, genre="E", recording_key=bytes([i])) for i, s in enumerate(styles)})
    no_vocal = nova.Filters(vocal="no")
    assert {t for t in lib.tracks if nova._filter_ok(lib, t, no_vocal)} == set(range(10))
    # 87 BPM попадает в «быстрый» диапазон через удвоение (174), а 174 — напрямую
    fast = nova.Filters(bpm=(150.0, 180.0))
    assert all(nova._filter_ok(lib, t, fast) for t in lib.tracks)
    slow = nova.Filters(bpm=(60.0, 100.0))
    assert {t for t in lib.tracks if nova._filter_ok(lib, t, slow)} == set(range(len(styles)))  # 174/2 = 87
    mid = nova.Filters(bpm=(100.0, 130.0))
    assert not any(nova._filter_ok(lib, t, mid) for t in lib.tracks)
    # неизвестная энергия под фильтр не попадает
    g.energy[:] = np.nan
    assert not any(nova._filter_ok(lib, t, nova.Filters(energy=(0.0, 1.0))) for t in lib.tracks)
    assert nova.filter_count(lib, no_vocal) == 10


def test_scenario_filter_membership():
    from types import SimpleNamespace as NS

    from arxnova import nova

    mix, styles, scores, fac, feats, _ = make()
    lib = NS(gate=build(mix, styles, scores, fac, feats), row={i: i for i in range(len(styles))},
             tracks={i: NS(style=s, genre="E", recording_key=bytes([i])) for i, s in enumerate(styles)},
             scenarios={"run": {"title": "Для бега", "members": {1, 2, 3}}})
    f = nova.Filters(scenario="run")
    assert {t for t in lib.tracks if nova._filter_ok(lib, t, f)} == {1, 2, 3}
    # неизвестный сценарий ничего не пропускает (а не всё подряд)
    assert not any(nova._filter_ok(lib, t, nova.Filters(scenario="nope")) for t in lib.tracks)


def test_skip_feedback_infers_previous_track_without_prev_id():
    """Плеер не прислал prev_track_id — предыдущий трек берётся из порядка событий прибора."""
    from arxnova import nova

    lib = _stub_lib()
    listen = {"track_id": 1, "event_type": "listen", "ts": 900, "is_organic": 0, "played_ratio_pct": 100,
              "skip_reason": None, "prev_track_id": None, "device_id": 7}
    skip = dict(_skip(2, None, "vocal"), device_id=7)
    p = nova.Profile(lib, [listen, skip], 2000)
    assert p.pair_tighten[frozenset(("A", "B"))] == {"vocal": nova.GATE_STEP}
    # на другом приборе — не связано
    other = dict(_skip(2, None, "vocal"), device_id=8)
    assert nova.Profile(lib, [listen, other], 2000).pair_tighten == {}


def test_tempo_from_beats_not_quantized():
    """Темп по долям — среднее ровных интервалов: 129 BPM на кадрах по 20 мс не превращается в 130,4."""
    from arxnova import facets

    period = 60 / 129
    beats = np.round(np.arange(0, 60, period) / 0.02) * 0.02  # доли, привязанные к кадрам модели
    bpm, conf = facets.tempo_from_beats(lambda audio, sr: (beats, None), np.zeros((2, 10)))
    assert abs(bpm - 129) < 0.3 and conf > 0.9
    assert facets.tempo_from_beats(lambda audio, sr: (beats[:5], None), np.zeros((2, 10))) is None


def test_tempo_from_beats_folds_octaves():
    from arxnova import facets

    fake = lambda bpm: (lambda audio, sr: (np.arange(0, 60, 60 / bpm), None))  # noqa: E731
    assert abs(facets.tempo_from_beats(fake(60), np.zeros((2, 10)))[0] - 120) < 0.5
    assert abs(facets.tempo_from_beats(fake(217), np.zeros((2, 10)))[0] - 108.5) < 0.5
    assert abs(facets.tempo_from_beats(fake(174), np.zeros((2, 10)))[0] - 174) < 0.5


def test_album_tiebreak_prefers_album_majority():
    """Почти вровень — побеждает стиль, преобладающий на альбоме; уверенные треки не трогаем."""
    from arxnova.nova import _album_tiebreak

    #                 Instr  TripHop Downtempo
    sc = np.array([[0.31, 0.30, 0.28],   # неуверенный трек альбома A
                   [0.05, 0.10, 0.80],   # уверенный Downtempo, альбом A
                   [0.02, 0.05, 0.70],   # уверенный Downtempo, альбом A
                   [0.90, 0.05, 0.02],   # уверенный Instrumental, альбом B
                   [0.31, 0.30, 0.28]])  # неуверенный, альбом C без уверенных — остаётся как есть
    top = sc.argmax(1)
    out = _album_tiebreak(sc, top, {0: "A", 1: "A", 2: "A", 3: "B", 4: "C"})
    assert list(out) == [2, 2, 2, 0, 0]


def test_mb_fallback_only_when_unsure_and_plausible(tmp_path):
    """Модель не уверена — стиль из MusicBrainz, но только правдоподобный для самой модели."""
    import json as _json

    from arxnova import config, db, mbgenres
    from arxnova.nova import _mb_fallback

    cfg = config.Config(data_dir=tmp_path)
    cfg.musicbrainz = True
    conn = db.connect(tmp_path / "d.sqlite3")
    conn.execute(mbgenres.SCHEMA)
    with conn:
        for tid, artist in ((1, "Big Band"), (2, "Sure"), (3, "Wrong")):
            conn.execute("INSERT INTO tracks (id, md5, md5_source, recording_key, recording_basis, sample_rate, bits, "
                         "channels, duration, artist, albumartist, album, title) VALUES (?, ?, 'streaminfo', ?, 'x', "
                         "44100, 16, 2, 100, ?, ?, 'A', 'T')", (tid, bytes([tid]), bytes([tid]), artist, artist))
        conn.execute("INSERT INTO mb_cache VALUES ('artist', 'Big Band', 'x', ?, 0)", (_json.dumps({"big band": 1, "drum and bass": 1}),))
        conn.execute("INSERT INTO mb_cache VALUES ('artist', 'Sure', 'x', ?, 0)", (_json.dumps({"rock": 5}),))
        conn.execute("INSERT INTO mb_cache VALUES ('artist', 'Wrong', 'x', ?, 0)", (_json.dumps({"doom metal": 5}),))
    labels = ["Electronic---Drum n Bass", "Funk / Soul---Neo Soul", "Jazz---Big Band", "Rock---Doom Metal"]
    sc = np.array([[0.08, 0.12, 0.03, 0.0],    # не уверена: MB → DnB (выше у модели, чем Big Band)
                   [0.90, 0.01, 0.01, 0.0],    # уверена — не трогаем
                   [0.01, 0.15, 0.01, 0.001]])  # MB «doom metal», но модель его почти исключает — не трогаем
    out = _mb_fallback(cfg, conn, np.array([1, 2, 3]), sc, sc.argmax(1), labels)
    assert list(out) == [0, 0, 1]
