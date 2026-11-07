"""Маршрут и свежее (решение 0017): постояв на месте, поток идёт к цели — стилю без истории или со свежим, — через переходные треки.
Владелец 07.11.2026: «У нас столько классной музыки, я добавил недавно новую, а Нова даёт либо джаз либо ДнБ»."""

from __future__ import annotations

import random
from collections import Counter
from types import SimpleNamespace

import numpy as np

from arxnova import listener as LM, nova
from tests.test_bridges_stream import NOW, _event, _listened, make_library

HOME_LEAVE_MEDIAN = 24  # калибровка 07.11.2026: у этого слушателя на 0.2.55 медиана первого ухода из дома — 21-й трек, с ошибкой удержания — 34-й


def _plays(outcomes):
    return [LM.Play(i, o) for i, o in enumerate(outcomes)]


def test_stand_is_the_last_run_of_one_style_and_a_skip_after_it_cancels_the_route():
    twelve = ["A"] * 12
    assert LM.route_origin(twelve + ["B", "C"], _plays([1] * 14)) == (0, 11, 11)  # стояли, теперь в пути
    assert LM.route_origin(twelve[:-1] + ["B"], _plays([1] * 12)) is None  # одиннадцать — ещё не стоянка
    assert LM.route_origin(twelve + ["B"], _plays([1] * 12 + [-1])) is None  # пропуск в пути: решает логика пропуска
    assert LM.route_origin(twelve, _plays([1] * 12), home="A") is None  # дома стоянка вдвое длиннее
    assert LM.route_origin(["A"] * 24, _plays([1] * 24), home="A") == (0, 23, 23)


def test_a_skip_inside_a_stand_brings_the_departure_closer():
    """Пропуск в стоянке засчитывается за 1 + ROUTE_SKIP_W трека: кто пропускает каждый четвёртый трек дома, уходит после 16, а не 24."""
    every4 = [-1 if k % 4 == 3 else 1 for k in range(17)]
    assert LM.route_origin(["H"] * 17, _plays(every4), home="H") == (0, 16, 15)
    assert LM.route_origin(["H"] * 15, _plays(every4[:15]), home="H") is None
    assert LM.route_origin(["A"] * 12 + ["B"], _plays([1] * 11 + [-1, 1])) == (0, 11, 11)  # 11 + пропуск (1 + 2): набралось на нём
    assert LM.route_origin(["A"] * 12 + ["B", "B"], _plays([1] * 12 + [1, -1])) is None  # пропуск в пути


def _stub(home="Home", stats=None, refused=()):
    rows = {s: np.arange(40) for s in ["Home", "Tried", "Skipped", "Untried", "Fresh", "Tiny", "OneArtist"]}
    rows["Tiny"] = np.arange(LM.ROUTE_MIN - 1)  # в мелком стиле поток упрётся в правила повторов: не цель
    capacity = {"OneArtist": 3}  # 40 треков одного исполнителя: при правиле исполнителя 12 подряд не встать
    lib = SimpleNamespace(route_styles=lambda: rows, route_capacity=lambda s: capacity.get(s, 40))
    stats = stats or {"Home": (60, 20), "Tried": (10, 4), "Skipped": (2, 30)}
    prof = SimpleNamespace(home_style=lambda lib: home, style_pos={s: p for s, (p, _) in stats.items()},
                           style_neg={s: n for s, (_, n) in stats.items()}, style_refused=lambda s: s in refused)
    return lib, prof


def test_destination_prefers_unheard_and_fresh_styles_and_rarely_picks_a_skipped_one():
    lib, prof = _stub()
    fresh = {"Fresh": 1.0}
    got = Counter(nova._route_to(lib, prof, (o, o + 1), {"Home"}, fresh, "Home") for o in range(600))
    assert got["Home"] == 0  # со стоянки дома домой не идут
    assert got["Tiny"] == 0
    assert got["OneArtist"] == 0  # стоять там нельзя: исполнитель не чаще раза в ARTIST_GAP
    assert got["Skipped"] < 0.05 * 600, got
    assert got["Fresh"] > got["Untried"] > got["Skipped"], got


def test_from_a_stand_away_from_home_the_stream_returns_home_as_often_as_home_is_accepted():
    lib, prof = _stub()
    share = 60 / (60 + 10 + 2)  # доля дома среди принятого
    got = Counter(nova._route_to(lib, prof, (o, o + 1), {"Home", "Untried"}, {}, "Untried") for o in range(800))
    want = min(LM.ROUTE_HOME_MAX, max(LM.ROUTE_HOME_MIN, share))
    assert abs(got["Home"] / 800 - want) < 0.06, (got["Home"] / 800, want)


def test_destination_stays_the_same_while_the_stream_walks_to_it():
    """Все запросы одного перехода видят одну цель: счёт стилей берётся на момент жребия (pos/neg), поэтому треки, принятые в пути,
    — и цели, и переходных стилей — её не двигают (независимая проверка 07.11.2026: по живому счёту цель менялась в 89 из 200)."""
    lib, prof = _stub()
    pos0, neg0 = dict(prof.style_pos), dict(prof.style_neg)
    moved = 0
    for o in range(200):
        first = nova._route_to(lib, prof, (o, o + 1), {"Home"}, {}, "Home", pos0, neg0)
        live = SimpleNamespace(**{**vars(prof), "style_pos": {**prof.style_pos, **{s: 2.0 for s in ["Untried", "Fresh", "Tried"]}}})
        assert nova._route_to(lib, live, (o, o + 1), {"Home"}, {}, "Home", pos0, neg0) == first
        moved += nova._route_to(lib, live, (o, o + 1), {"Home"}, {}, "Home") != first
    assert moved, "по живому счёту цель не сдвинулась ни разу — проверка ничего не проверяет"


def test_fresh_share_counts_unheard_fresh_tracks_as_of_the_draw():
    """Доля свежего в стиле — неслушанное на момент жребия: стиль, где всё свежее уже прослушано, надбавки не тянет."""
    lib = make_library()
    side = [t for t in lib.tracks if lib.tracks[t].style == "Side"]
    lib.added_at = {t: (NOW // 1000 - DAY if t in side else 1000) for t in lib.tracks}
    lib.first_import_end = 2000
    prof = SimpleNamespace(played=set())
    assert nova._fresh_share(lib, prof, NOW // 1000)["Side"] == 1.0
    assert nova._fresh_share(lib, prof, NOW // 1000)["Main"] == 0.0
    assert nova._fresh_share(lib, prof, NOW // 1000, played=set(side[: len(side) // 2])) == {**nova._fresh_share(lib, prof, NOW // 1000), "Side": 0.5}


DAY = 86_400


def test_freshness_is_for_unheard_tracks_added_after_the_first_big_import_and_fades():
    today = 121 * DAY
    lib = SimpleNamespace(added_at={1: 100 * DAY, 2: today - DAY, 3: today - DAY, 4: today - (LM.FRESH_DAYS + 1) * DAY, 5: today + 5 * DAY},
                          first_import_end=100 * DAY + 600)
    prof = SimpleNamespace(newness=lambda t: 0.0 if t == 3 else 1.0)
    assert nova._freshness(lib, prof, 1, today) == 0.0  # из первого разбора — не новое в медиатеке
    assert 0.9 < nova._freshness(lib, prof, 2, today) <= 1.0  # вчерашнее
    assert nova._freshness(lib, prof, 3, today) == 0.0  # уже слушал
    assert nova._freshness(lib, prof, 4, today) == 0.0  # больше FRESH_DAYS назад
    assert nova._freshness(lib, prof, 5, today) == 1.0  # дата добавления впереди часов сервера — не больше единицы


def test_first_import_is_the_first_unbroken_burst_even_across_midnight():
    """Первый большой разбор — от первого трека, пока между добавлениями нет перерыва дольше LM.FIRST_IMPORT_GAP_H часов.
    Независимая проверка 07.11.2026: «самый частый день» ломался, если разбор пересёк полночь или новый диск больше первого."""
    t0 = 300 * DAY - 120  # за две минуты до полуночи UTC
    burst = [t0 + 3 * k for k in range(2000)]  # 100 минут через полночь
    later = [t0 + 10 * DAY + k for k in range(5000)]  # новый диск больше первого разбора
    assert nova.first_import_end(burst + later) == burst[-1]
    assert nova.first_import_end([]) == 0


def _walk(lib, hist, evs, now, seed):
    tr = {}
    mood = nova.Mood.from_events(lib, list(reversed(evs)), now)
    prof = nova.Profile(lib, _listened(range(1, 41), NOW - 86_400_000) + evs, now)
    picks = nova.next_tracks(lib, prof, current=hist[-1], history=hist, n=3, now_ms=now, rng=random.Random(seed),
                             mode=nova.MODE_MY, mood=mood, model="listener", trace=tr)
    return picks, tr


def test_after_a_long_stand_at_home_the_stream_sets_off_and_a_skip_on_the_way_cancels_it():
    lib = make_library()
    main = [t for t in lib.tracks if lib.tracks[t].style == "Main"][60:60 + 2 * LM.DWELL_START]
    now, evs = NOW, []
    for t in main:
        now += 300_000
        evs.append(_event(t, "listen", now))
    picks, tr = _walk(lib, main, evs, now, 1)
    assert tr.get("route", {}).get("to") in {"Side", "Far"}, tr.get("route")
    assert picks and "route" in picks[0].parts
    to = tr["route"]["to"]
    first = picks[0].track_id
    now += 300_000
    _, tr2 = _walk(lib, main + [first], evs + [_event(first, "listen", now)], now, 2)
    if lib.tracks[first].style != to:  # ещё в пути — цель та же
        assert tr2.get("route", {}).get("to") == to, (to, tr2.get("route"))
    _, tr3 = _walk(lib, main + [first], evs + [_event(first, "skip", now, 10)], now, 2)
    assert "route" not in tr3, tr3.get("route")


def _stand(lib, k, length=2 * LM.DWELL_START):
    main = [t for t in lib.tracks if lib.tracks[t].style == "Main"]
    tracks = main[60 + length * k:60 + length * (k + 1)]
    now, evs = NOW, []
    for t in tracks:
        now += 300_000
        evs.append(_event(t, "listen", now))
    return tracks, evs, now


def _ask(lib, hist, evs, now, seed, filters=None, **kw):
    tr = {}
    mood = nova.Mood.from_events(lib, list(reversed(evs)), now)
    prof = nova.Profile(lib, _listened(range(1, 41), NOW - 86_400_000) + evs, now)
    picks = nova.next_tracks(lib, prof, current=hist[-1], history=hist, n=3, now_ms=now, rng=random.Random(seed),
                             mode=nova.MODE_MY, mood=mood, model="listener", trace=tr, filters=filters, **kw)
    return picks, tr, prof


def test_on_the_way_the_batch_ends_closest_to_the_target():
    """Плеер берёт пачку целиком, а следующий запрос идёт от последнего трека очереди: в пути он ближе всех к цели
    (независимая проверка 07.11.2026: без этого маршрут кончался в соседнем стиле — 13 стоянок в цели из 49)."""
    lib = make_library()
    for k in range(6):
        stand, evs, now = _stand(lib, k)
        picks, tr, _ = _ask(lib, stand, evs, now, k)
        c = lib.style_closeness(tr["route"]["to"])
        close = [float(c[lib.row[p.track_id]]) for p in picks]
        assert close == sorted(close), (tr["route"], close)


def test_a_target_track_mid_batch_is_not_arrival_and_the_target_does_not_change():
    """Пришли — когда последний трек очереди в стиле цели. Трек цели посреди пачки и переходный трек, который зашёл,
    цель не меняют: жребий тянется по счётам на начало стоянки (независимая проверка: цель менялась в 89 из 200 стоянок)."""
    lib = make_library()
    for k in range(10):
        stand, evs, now = _stand(lib, k)
        _, tr, _ = _ask(lib, stand, evs, now, k)
        to = tr["route"]["to"]
        target = next(t for t in lib.tracks if lib.tracks[t].style == to)
        other = next(t for t in lib.tracks if lib.tracks[t].style not in (to, "Main"))
        evs2 = evs + [_event(target, "listen", now + 300_000), _event(other, "listen", now + 600_000)]
        _, tr2, _ = _ask(lib, stand + [target, other], evs2, now + 600_000, k)
        assert tr2.get("route", {}).get("to") == to, (k, to, tr2.get("route"))
        _, tr3, _ = _ask(lib, stand + [other, target], evs2, now + 600_000, k)
        assert "route" not in tr3, tr3.get("route")  # последний — цель: пришли


def test_a_stand_that_cannot_leave_drops_the_route_after_route_steps():
    """Шаги пути считаются от жребия стоянки: цель недостижима — стоянка тянется, и через ROUTE_STEPS маршрут снят,
    дальше «пора дальше» как в 0016 (независимая проверка: при «только знакомом» стоянка дома росла до 65–80 треков)."""
    lib = make_library()
    stand, evs, now = _stand(lib, 0, 2 * LM.DWELL_START + LM.ROUTE_STEPS + 1)
    _, tr, _ = _ask(lib, stand, evs, now, 1)
    assert "route" not in tr, tr.get("route")


def test_only_familiar_slider_gets_no_new_tracks_on_the_way():
    """Цель — только стиль, где хватает треков, проходящих ползунок, фильтры и «только треки устройства»: в Side и Far знакомых нет,
    маршрута нет и новых треков в пути нет (независимая проверка: при «только знакомом» шли новые из пути)."""
    lib = make_library()
    for k in range(6):
        stand, evs, now = _stand(lib, k)
        picks, tr, prof = _ask(lib, stand, evs, now, k, filters=nova.Filters(familiar=1.0))
        assert "route" not in tr, tr.get("route")
        assert all(prof.newness(p.track_id) < 0.5 for p in picks), [prof.newness(p.track_id) for p in picks]


def _walk_listener(lib, react, seed, batches):
    base = _listened(range(1, 41), NOW - 86_400_000)
    rng, evs, hist, picks, now = random.Random(seed), [], [], [], NOW
    for _ in range(batches):
        mood = nova.Mood.from_events(lib, list(reversed(evs)), now)
        prof = nova.Profile(lib, base + evs, now)
        batch = nova.next_tracks(lib, prof, current=hist[-1] if hist else None, history=hist[-150:], n=3, now_ms=now, rng=rng,
                                 mode=nova.MODE_MY, start=not hist, mood=mood, model="listener")
        for p in batch:
            picks.append(p)
            hist.append(p.track_id)
            now += 60_000
            et, pct = react(lib.tracks[p.track_id], len(picks))
            evs.append(_event(p.track_id, et, now, pct))
    return picks


def test_a_listener_who_skips_some_home_tracks_leaves_home_as_soon_as_before():
    """Дома стоянка вдвое длиннее только для маршрута: удержание места считает настоящие треки, иначе дома оно включено
    с первого трека, и кто пропускает часть ДнБ, уходит из дома вдвое позже — «много надо скипнуть» (жалоба 06.11)."""
    lib = make_library()
    firsts = []
    for seed in range(6):
        picks = _walk_listener(lib, lambda t, k: ("skip", 10) if t.style == "Main" and k % 4 == 0 else ("listen", 100), seed, 12)
        firsts.append(next((i for i, p in enumerate(picks) if lib.tracks[p.track_id].style != "Main"), len(picks)))
    assert sorted(firsts)[len(firsts) // 2] <= HOME_LEAVE_MEDIAN, firsts


def _fresh_walk(monkeypatch, **kw):
    """Вся медиатека добавлена вчера, после первого разбора. Короткие и длинные стоянки (с маршрутом): части неслушанных треков в выборе
    и доля свежего, с которой выбиралась цель маршрута."""
    lib = make_library()
    lib.added_at = {t: NOW // 1000 - 86_400 for t in lib.tracks}
    lib.first_import_end = 0
    shares, real = [], nova._route_to
    monkeypatch.setattr(nova, "_route_to", lambda *a, **k: (shares.append(a[4]), real(*a, **k))[1])
    parts = []
    for k in range(4):
        for length in (6, 2 * LM.DWELL_START):
            stand, evs, now = _stand(lib, k, length)
            picks, _, prof = _ask(lib, stand, evs, now, k, **kw)
            parts += [p.parts for p in picks if prof.newness(p.track_id) >= 0.5]
    assert parts and shares, "неслушанных в выборе или маршрутов не было — проверка ничего не проверила"
    return parts, shares


def test_fresh_is_off_by_default_even_for_a_track_added_yesterday(monkeypatch):
    """Свежее (решение 0017) — только по настройке fresh_boost: владелец 07.11.2026 «не уверен в этом». Без неё трек, добавленный вчера,
    надбавки не получает, и цель маршрута выбирается без доли свежего в стиле."""
    parts, shares = _fresh_walk(monkeypatch)
    assert not any("fresh" in p for p in parts), [p for p in parts if "fresh" in p][:2]
    assert not any(any(s.values()) for s in shares), shares[:2]


def test_with_fresh_boost_unheard_recent_tracks_get_their_bonus_and_pull_the_route(monkeypatch):
    parts, shares = _fresh_walk(monkeypatch, fresh=True)
    assert all(p.get("fresh", 0.0) > 0 for p in parts), parts
    assert any(any(s.values()) for s in shares), shares[:2]


def test_fresh_boost_setting_is_off_by_default_and_round_trips(tmp_path):
    from arxnova import config

    cfg = config.Config(data_dir=tmp_path)
    assert cfg.fresh_boost is False
    cfg.save()
    assert config.load(tmp_path).fresh_boost is False
    cfg.fresh_boost = True
    cfg.save()
    assert config.load(tmp_path).fresh_boost is True


def test_on_skip_alternative_ignores_the_route_pull():
    """Пропуск в пути снимает маршрут: запасной ход считается без части «к цели»."""
    lib = make_library()
    main = [t for t in lib.tracks if lib.tracks[t].style == "Main"]
    hist = main[60:66]
    state = [LM.Play(lib.row[t], 1) for t in hist]
    m = LM.Model()
    pulled = nova.Pick(main[200], "", 5.0, parts={"route": 4.9})
    plain = nova.Pick(main[201], "", 3.0, parts={})
    batch = [nova.Pick(main[202], "", 2.0, parts={})]
    alt = nova._on_skip(lib, nova.Profile(lib, [], NOW), m, state, m.beta, [pulled, plain] + batch, batch, hist, NOW)
    assert alt[batch[0].track_id].track_id == plain.track_id, alt


def test_on_the_way_the_place_is_not_held():
    """В пути место не держится: иначе после первого переходного трека поток стоял бы в его стиле 12 треков, а не шёл к цели.
    В журнале маршрута — «hold»; на старте маршрута стоянка длинная и удержания нет в любом случае, различает второй запрос."""
    lib = make_library()
    for k in range(4):
        stand, evs, now = _stand(lib, k)
        _, tr, _ = _ask(lib, stand, evs, now, k)
        assert tr["route"]["cands"] > 0 and tr["route"]["hold"] is False, tr["route"]
        to = tr["route"]["to"]
        other = next(t for t in lib.tracks if lib.tracks[t].style not in (to, "Main"))
        _, tr2, _ = _ask(lib, stand + [other], evs + [_event(other, "listen", now + 300_000)], now + 300_000, k)
        assert tr2["route"]["to"] == to and tr2["route"]["hold"] is False, tr2["route"]


def test_a_listener_without_history_gets_no_fresh_pull():
    """У нового слушателя неслушанное всё: надбавка за недавно добавленное без истории выключена и растёт с принятым."""
    assert nova._fresh_scale({}) == 0.0
    assert nova._fresh_scale({"A": LM.FRESH_FULL_AT / 2}) == 0.5
    assert nova._fresh_scale({"A": 3 * LM.FRESH_FULL_AT, "B": -5.0}) == 1.0
