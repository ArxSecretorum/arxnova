"""Модель слушателя (решение 0016): поток меняет курс с первого пропуска, в потоке держится рядом, учится на истории.
Медиатека синтетическая (как в test_bridges_stream): основной стиль и два небольших."""

from __future__ import annotations

import itertools
import json
import random

import numpy as np

from arxnova import config, listener as LM, nova
from tests.test_bridges_stream import NOW, _event, _listened, make_library


def test_outcome_of_a_play():
    assert LM.outcome_of(["done"]) == 1
    assert LM.outcome_of(["late"]) == 1
    assert LM.outcome_of(["early"]) == -1 and LM.outcome_of(["mid"]) == -1
    assert LM.outcome_of(["plus", "early"]) == 1  # лайкнул и пошёл дальше — принят
    assert LM.outcome_of(["plus", "dislike"]) == -1  # последний дизлайк важнее лайка
    assert LM.outcome_of(["none"]) == 0 and LM.outcome_of([]) == 0  # «назад», свой выбор: не понять


def test_fit_follows_the_data_and_without_data_keeps_the_prior():
    m = LM.Model()
    assert np.allclose(LM.fit_rows(m, np.zeros((0, len(LM.FEATURES))), np.zeros(0)).beta, LM.PRIOR_MU)
    rng = np.random.default_rng(1)
    X = rng.normal(size=(600, len(LM.FEATURES)))
    truth = np.array([-1.0, 1.5, 0.0, -1.0, 0.0, 1.2, 0.0, 0.0, 0.0, 0.0])
    p = 1 / (1 + np.exp(-(truth[0] + X @ truth[1:])))
    y = (rng.random(600) < p).astype(float)
    fitted = LM.fit_rows(LM.Model(), X, y).beta
    i = {k: n + 1 for n, k in enumerate(LM.FEATURES)}
    assert fitted[i["prev_skip"]] > 0.8 and fitted[i["taste"]] < -0.5 and fitted[i["flee"]] > 0.6


def _near_far(lib, anchor):
    """Свои соседи якоря (основной стиль, ближайшие) и чужие треки."""
    a = lib.row[anchor]
    main = [t for t in lib.tracks if lib.tracks[t].style == "Main" and t != anchor]
    main.sort(key=lambda t: -float(lib.vec[lib.row[t]] @ lib.vec[a]))
    far = [t for t in lib.tracks if lib.tracks[t].style != "Main"]
    return np.array([lib.row[t] for t in main[:20]]), np.array([lib.row[t] for t in far[:20]])


def test_after_a_skip_similar_is_risky_after_a_listen_similar_is_safe():
    """История владельца: после пропуска похожий пропущен в 68 %, непохожий в 42 %; после дослушанного похожий 5 %, непохожий 11 %."""
    lib = make_library()
    m = LM.Model()
    anchor = 7
    near, far = _near_far(lib, anchor)
    accepted = [LM.Play(lib.row[3], 1), LM.Play(lib.row[5], 1), LM.Play(lib.row[anchor], 1)]
    refused = accepted[:-1] + [LM.Play(lib.row[anchor], -1)]
    acc_near, acc_far = (m.p_skip(LM.features(lib, m, accepted, r)).mean() for r in (near, far))
    ref_near, ref_far = (m.p_skip(LM.features(lib, m, refused, r)).mean() for r in (near, far))
    assert acc_near < acc_far, (acc_near, acc_far)
    assert ref_near > ref_far, (ref_near, ref_far)


def _session(lib, last_kind):
    main = [t for t in lib.tracks if lib.tracks[t].style == "Main"]
    now, evs = NOW, []
    for t in main[50:56]:
        now += 300_000
        evs.append(_event(t, "listen", now))
    now += 60_000
    evs.append(_event(main[56], "skip", now, 10) if last_kind == "skip" else _event(main[56], "listen", now))
    return main[50:57], evs, now


def _next(lib, hist, evs, now, seed, trace=None):
    mood = nova.Mood.from_events(lib, list(reversed(evs)), now)
    prof = nova.Profile(lib, _listened(range(1, 41), NOW - 86_400_000) + evs, now)
    return nova.next_tracks(lib, prof, current=hist[-1], history=hist, n=3, now_ms=now, rng=random.Random(seed),
                            mode=nova.MODE_MY, mood=mood, model="listener", trace=trace)


def test_first_skip_turns_the_stream_away_and_a_listen_keeps_it_close():
    """Владелец 06.11.2026: «много надо скипнуть, чтоб выкинуло». Один пропуск — следующий трек уже в стороне."""
    lib = make_library()
    sims = {}
    for kind in ("skip", "listen"):
        hist, evs, now = _session(lib, kind)
        last = lib.vec[lib.row[hist[-1]]]
        sims[kind] = np.mean([float(lib.vec[lib.row[_next(lib, hist, evs, now, s)[0].track_id]] @ last) for s in range(12)])
    assert sims["skip"] < sims["listen"] - 0.05, sims


def test_skip_contingency_points_away_from_the_playing_track():
    """Запасной ход: плеер ставит его сразу после пропуска, без второго пропуска и без нового запроса."""
    lib = make_library()
    hist, evs, now = _session(lib, "listen")
    evs = evs[:-1]  # последний трек ещё играет: событий по нему нет
    away, main_pick = [], []
    for seed in range(10):
        trace: dict = {}
        picks = _next(lib, hist, evs, now, seed, trace)
        cur = lib.vec[lib.row[hist[-1]]]
        alt = trace["on_skip"][hist[-1]]
        assert alt.track_id not in {p.track_id for p in picks} and alt.track_id not in hist
        assert set(trace["on_skip"]) >= {hist[-1], *(p.track_id for p in picks)}
        away.append(float(lib.vec[lib.row[alt.track_id]] @ cur))
        main_pick.append(float(lib.vec[lib.row[picks[0].track_id]] @ cur))
    assert np.mean(away) < np.mean(main_pick) - 0.05, (np.mean(away), np.mean(main_pick))


def test_journal_has_predictions_and_stays_json():
    lib = make_library()
    hist, evs, now = _session(lib, "listen")
    trace: dict = {}
    picks = _next(lib, hist, evs, now, 1, trace)
    prof = nova.Profile(lib, _listened(range(1, 41), NOW - 86_400_000) + evs, now)
    rec = nova.journal(lib, prof, picks, trace)
    json.dumps(rec)
    assert rec["v"] == nova.LISTENER_VERSION
    assert all(0.0 < p["q"] < 1.0 for p in rec["picks"])
    assert set(rec["listener"]["beta"]) == {"bias", *LM.FEATURES}
    assert rec["on_skip"]


def test_passive_listener_moves_on_by_itself_but_keeps_home():
    """Без действий поток не стоит на месте вечно (усталость), но дом держится. Пробы раз в 16 больше не нужны."""
    lib = make_library()
    base = _listened(range(1, 41), NOW - 86_400_000)
    rng, hist, events, picks, now = random.Random(3), [], [], [], NOW
    while len(picks) < 150:
        mood = nova.Mood.from_events(lib, list(reversed(events)), now)
        prof = nova.Profile(lib, base + events, now)
        batch = nova.next_tracks(lib, prof, current=hist[-1] if hist else None, history=hist[-150:], n=3, now_ms=now, rng=rng,
                                 mode=nova.MODE_MY, start=not hist, mood=mood, model="listener")
        assert batch, "поток не должен замолкать"
        for p in batch:
            picks.append(p)
            hist.append(p.track_id)
            now += 300_000
            events.append(_event(p.track_id, "listen", now))
    styles = [lib.tracks[p.track_id].style for p in picks]
    share = styles.count("Main") / len(styles)
    runs = [(k, len(list(g))) for k, g in itertools.groupby(styles)]
    assert 0.5 <= share <= 0.9, (share, runs)
    # путешествие, а не россыпь по одному треку: вне дома хотя бы один отрезок, как в сете
    assert any(k != "Main" and n >= 10 for k, n in runs), runs
    assert all(p.why != "spice" for p in picks)


def test_stream_model_setting_round_trips(tmp_path):
    cfg = config.Config(data_dir=tmp_path)
    cfg.ensure_dirs()
    assert cfg.stream_model == "listener"
    cfg.stream_model = "drift"
    cfg.save()
    assert config.load(tmp_path).stream_model == "drift"


def test_stats_reads_predictions_and_contingency_from_the_journal():
    from arxnova import stats

    detail = json.dumps({"v": nova.LISTENER_VERSION, "picks": [{"t": 5, "q": 0.2}, {"t": 6, "q": 0.7}],
                         "on_skip": {"4": {"t": 9, "q": 0.1}}})
    assert stats._predicted(detail, 6) == (0.7, False)
    assert stats._predicted(detail, 9) == (0.1, True)  # пришёл запасным ходом
    assert stats._predicted(detail, 77) == (None, False) and stats._predicted(None, 5) == (None, False)
    assert stats._auc([(0.9, 1), (0.8, 1), (0.1, 0), (0.2, 0)]) == 1.0
    assert stats._auc([(0.5, 1), (0.5, 0)]) == 0.5
    assert stats._auc([(0.3, 0)]) is None


# ---- находки независимой проверки 06.11.2026 ----

def _listened_session(lib, k=6):
    main = [t for t in lib.tracks if lib.tracks[t].style == "Main"]
    now, evs = NOW, []
    for t in main[50:50 + k]:
        now += 300_000
        evs.append(_event(t, "listen", now))
    return main, evs, now


def test_contingency_after_dislike_does_not_lean_on_the_disliked_track():
    """«Не предлагать» на играющем D: запасной ход для следующего трека не должен держаться рядом с D (раньше D становился привязкой)."""
    lib = make_library()
    near, hop = LM.scale_of(lib)
    main, evs, now = _listened_session(lib)
    D = main[56]
    hist = main[50:57]
    now += 20_000
    evs.append(_event(D, "dislike", now))
    close = []
    for seed in range(8):
        trace: dict = {}
        picks = _next(lib, hist, evs, now, seed, trace)
        alt = trace["on_skip"].get(picks[0].track_id)
        if alt is not None:
            close.append(float(lib.vec[lib.row[alt.track_id]] @ lib.vec[lib.row[D]]))
    assert close and np.mean(close) < hop, (np.mean(close), hop)


def test_switched_track_without_events_does_not_break_the_skip_streak():
    """Плеер подменил B запасным ходом U: B остаётся в истории без событий. Это не «принят»: пропуск U — второй подряд после A,
    шаг уводит от обоих без привязки, а не держится рядом с B (выбранным ещё до пропуска A, рядом с A)."""
    lib = make_library()
    acc, A, B, U = (lib.row[t] for t in (3, 7, 8, 9))
    step = LM.Step.of(lib, [LM.Play(acc, 1), LM.Play(A, -1), LM.Play(B, 0), LM.Play(U, -1)])
    assert step.rows[0] is None and set(step.rows[1]) == {U, A}, step.rows  # два пропуска подряд: привязка не действует (Step.allows)
    # а играющий трек без событий — по-прежнему «пока заходит»: шаг рядом с ним
    flow = LM.Step.of(lib, [LM.Play(acc, 1), LM.Play(A, -1), LM.Play(B, 0)])
    assert flow.rows[0] == B


def test_relaxed_fallback_with_listener_still_finds_new_tracks():
    """Запасной путь (relaxed) идёт без модели: ему нужны и случайные новые кандидаты, иначе «только новое» давало 0 %."""
    lib = make_library()
    main, evs, now = _listened_session(lib)
    base = _listened(range(1, 401), NOW - 86_400_000)  # весь основной стиль знаком: новое только в дальних стилях
    mood = nova.Mood.from_events(lib, list(reversed(evs)), now)
    prof = nova.Profile(lib, base + evs, now)
    picks = nova.next_tracks(lib, prof, current=main[55], history=main[50:56], n=3, now_ms=now, rng=random.Random(1), mode=nova.MODE_MY,
                             mood=mood, model="listener", relaxed=True, filters=nova.Filters(familiar=0.0))
    assert picks and all(prof.newness(p.track_id) >= 0.5 for p in picks), [prof.newness(p.track_id) for p in picks]


def test_small_allowed_set_does_not_silence_the_listener_stream():
    """Только треки устройства (allowed): отбор лучших по модели раньше шёл до проверки allowed и оставлял пустой ответ."""
    lib = make_library()
    allowed = set(range(1, 520, 13))  # 40 треков из 520
    base = _listened(range(1, 41), NOW - 86_400_000)
    rng, hist, events, now = random.Random(2), [], [], NOW
    for _ in range(10):
        mood = nova.Mood.from_events(lib, list(reversed(events)), now)
        prof = nova.Profile(lib, base + events, now)
        batch = nova.next_tracks(lib, prof, current=hist[-1] if hist else None, history=hist[-150:], n=3, now_ms=now, rng=rng,
                                 mode=nova.MODE_MY, start=not hist, mood=mood, model="listener", allowed=allowed)
        assert batch, f"замолчал после {len(hist)} треков"
        assert all(p.track_id in allowed for p in batch)
        for p in batch:
            hist.append(p.track_id)
            now += 300_000
            events.append(_event(p.track_id, "listen", now))


def test_like_during_a_play_is_not_in_the_taste_of_that_same_play():
    """Лайк пишется до «дослушал»: при обучении он не должен попадать во вкус раньше признаков этого же показа (утечка будущего)."""
    lib = make_library()
    main = [t for t in lib.tracks if lib.tracks[t].style == "Main"]
    now, plain, liked = NOW, [], []
    for i, t in enumerate(main[:12]):
        now += 300_000
        e = {**_event(t, "listen", now), "nova_request_id": 1, "left_by": "next"}
        plain.append(e)
        if i == 11:
            liked.append({**_event(t, "like", now - 1000), "nova_request_id": 1})
        liked.append(e)
    xp, _, _, _ = LM.training_rows(lib, plain)
    xl, _, _, _ = LM.training_rows(lib, liked)
    i = LM.FEATURES.index("taste")
    assert abs(xp[-1, i] - xl[-1, i]) < 1e-6, (xp[-1, i], xl[-1, i])
