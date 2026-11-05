"""Дом — обычный стиль слушателя. Владелец 05.11.2026: «дом тоже должен уметь дрейфовать и/или сохранять пограничные состояния,
но к дрейфу дома надо отнестись очень осторожно, необходим очевидный уход пользователем от привычного» (решение 0013)."""

from __future__ import annotations

import random
import time
from collections import Counter

from arxnova import drift as D
from arxnova import nova
from tests.test_bridges_stream import EARLY, FOUR_STYLES, LIKE, LISTEN, NOW, _drift, _session, _event, HEARD, SKIPPED, make_library

DAY_MS = 86_400_000
HABIT = [Counter({"A": 20})] * 30


def _primary(days):
    return next(iter(D.home_of(days)))


def test_one_long_evening_does_not_move_home():
    home = D.home_of(HABIT + [Counter({"B": 300})])
    assert list(home) == ["A"] and home["A"] > 0.9


def test_a_few_tracks_are_not_a_vote_of_a_whole_day():
    assert list(D.home_of(HABIT + [Counter({"B": 1})] * 30)) == ["A"]


def test_obvious_departure_gives_a_second_home_first_and_then_a_new_main_one():
    away = [Counter({"B": 20})]
    assert list(D.home_of(HABIT + away * 5)) == ["A"], "пять дней — ещё не привычка"
    assert list(D.home_of(HABIT + away * 6)) == ["A", "B"], "пограничное: дом двойной, главный прежний"
    assert list(D.home_of(HABIT + away * 15)) == ["A", "B"]
    assert list(D.home_of(HABIT + away * 16)) == ["B", "A"], "главный сменился, а старый остался вторым"
    assert list(D.home_of(HABIT + away * 40)) == ["B"]


def test_partial_departure_needs_longer():
    mostly = [Counter({"B": 14, "A": 6})]
    assert _primary(HABIT + mostly * 25) == "A"
    assert _primary(HABIT + mostly * 40) == "B"


def test_living_between_two_styles_holds_and_the_main_one_does_not_flicker():
    both = [Counter({"A": 20}), Counter({"B": 20})] * 40
    for n in range(1, len(both) + 1):
        assert _primary(HABIT + both[:n]) == "A", n
    home = D.home_of(HABIT + both)
    assert list(home) == ["A", "B"] and abs(home["A"] - home["B"]) < 0.1


def test_steady_mix_settles_at_its_proportion():
    home = D.home_of(HABIT + [Counter({"A": 12, "B": 8})] * 100)
    assert list(home) == ["A", "B"] and abs(home["B"] - 0.4) < 0.02


def test_one_evening_of_a_new_listener_is_not_a_second_home():
    assert list(D.home_of([Counter({"A": 40}), Counter({"B": 40})])) == ["A"]
    assert list(D.home_of([Counter({"A": 40}), Counter({"B": 40})] * 3)) == ["A", "B"]


def test_small_interest_is_not_a_home():
    assert list(D.home_of(HABIT + [Counter({"A": 17, "B": 3})] * 100)) == ["A"]


def test_there_is_no_habit_yet_at_the_start():
    assert _primary([Counter({"B": 10})]) == "B"
    assert _primary([Counter({"B": 10})] + [Counter({"A": 10})] * 2) == "B"
    assert _primary([Counter({"B": 10})] + [Counter({"A": 10})] * 3) == "A"
    assert D.home_of([]) == {} and D.home_of([Counter()]) == {}


def _morning(ts):
    """Утро дня слушателя: тесты не должны зависеть от часового пояса машины."""
    offset = time.localtime(ts / 1000).tm_gmtoff
    return (nova.listening_day(ts, offset) * 86_400 + 4 * 3600 - offset + 3 * 3600) * 1000


def _days(lib, plan, start=NOW - 200 * DAY_MS):
    """plan: [(стиль, треков в день, дней)]."""
    start = _morning(start)
    events, day, used = [], 0, Counter()
    for style, per_day, n in plan:
        pool = [t.id for t in lib.tracks.values() if t.style == style]
        for _ in range(n):
            for k in range(per_day):
                events.append(_event(pool[used[style] % len(pool)], "listen", start + day * DAY_MS + k * 60_000))
                used[style] += 1
            day += 1
    return events


def test_one_evening_does_not_move_home_in_the_real_profile():
    """Было: 300 треков за один присест меняли дом, потому что дом был просто самым частым стилем."""
    lib = make_library(FOUR_STYLES)
    prof = nova.Profile(lib, _days(lib, [("Main", 10, 20), ("Far", 300, 1)]), NOW)
    assert prof.home_style(lib) == "Main" and "Far" not in prof.homes


def test_a_break_changes_nothing():
    lib = make_library(FOUR_STYLES)
    events = _days(lib, [("Main", 10, 20), ("Far", 10, 8)])
    soon = nova.Profile(lib, events, events[-1]["ts"] + DAY_MS)
    years = nova.Profile(lib, events, events[-1]["ts"] + 900 * DAY_MS)
    assert soon.homes == years.homes and list(soon.homes)[0] == "Main"


def test_likes_weigh_more_than_passive_hearing():
    lib = make_library(FOUR_STYLES)
    far = [t.id for t in lib.tracks.values() if t.style == "Far"]
    start = _morning(NOW - 40 * DAY_MS)
    habit = _days(lib, [("Main", 10, 25)], start)
    quiet, active = [], []
    for d in range(12):
        ts = start + (25 + d) * DAY_MS
        for k in range(10):
            quiet.append(_event(far[k], "listen", ts + k))
            active.append(_event(far[k], "listen", ts + k))
        main = [t.id for t in lib.tracks.values() if t.style == "Main"]
        for k in range(10):
            quiet.append(_event(main[k + 50], "listen", ts + 100 + k))
            active.append(_event(main[k + 50], "listen", ts + 100 + k))
        active += [_event(far[k], "like", ts + 200 + k) for k in range(4)]
    assert "Far" not in nova.Profile(lib, habit + quiet, NOW).homes
    assert "Far" in nova.Profile(lib, habit + active, NOW).homes


def _start_styles(lib, prof, seeds=60):
    got = Counter()
    for seed in range(seeds):
        picks = nova.next_tracks(lib, prof, current=None, history=[], n=3, now_ms=NOW, rng=random.Random(seed), mode=nova.MODE_MY, start=True)
        got[lib.tracks[picks[0].track_id].style] += 1
    return got


def test_session_starts_at_home_and_a_double_home_shares_the_starts():
    lib = make_library((("Main", 400), ("Side", 60), ("Far", 300), ("Mid", 60)))
    one = nova.Profile(lib, _days(lib, [("Main", 10, 25)]), NOW)
    assert _start_styles(lib, one)["Main"] == 60
    both = nova.Profile(lib, _days(lib, [("Main", 10, 1), ("Far", 10, 1)] * 15), NOW)
    assert set(both.homes) == {"Main", "Far"}, both.homes
    got = _start_styles(lib, both)
    assert got["Main"] >= 15 and got["Far"] >= 15 and got["Main"] + got["Far"] == 60, got


def test_second_home_counts_as_home_for_the_session():
    """Мягкие пропуски не выгоняют из него, а на возврат хватает двух лайков."""
    style_of = {i: "a" for i in range(1, 7)}
    mid = lambda ts: ("skip", 30, False, None, None, ts)  # noqa: E731
    events = {i: [mid(i)] for i in (5, 6)}
    prior = lambda main: {"Main": 0.5, "b": 0.5}  # noqa: E731
    follow = lambda *homes: D.follow(list(style_of), style_of.get, lambda t: events.get(t, []), prior, "Main", 0.5, homes)  # noqa: E731
    assert follow().leaving == "a"
    assert follow("a").leaving is None
    like = lambda ts: ("like", 100, False, None, None, ts)  # noqa: E731
    listen = lambda ts: ("listen", 100, False, None, None, ts)  # noqa: E731
    steps = {1: "Main", 2: "Main", 3: "Main", 4: "b", 5: "b"}
    actions = {4: [like(4), listen(5)], 5: [like(6), listen(7)]}
    went = lambda *homes: D.follow(list(steps), steps.get, lambda t: actions.get(t, []), lambda main: {"b": 0.5, "c": 0.5}, "Main", 0.5, homes)  # noqa: E731
    assert went().main == "Main" and went("b").main == "b"


def test_leaving_one_home_goes_to_the_other():
    lib = make_library(FOUR_STYLES)
    for seed in range(1, 25):
        events = _days(lib, [("Main", 10, 1), ("Far", 10, 1)] * 15)
        prof = nova.Profile(lib, events, NOW)
        assert set(prof.homes) == {"Main", "Far"}
        pool = [t.id for t in lib.tracks.values() if t.style == "Main"]
        hist = pool[100:110]
        now = NOW
        session = []
        for t in hist[:-2]:
            now += 60_000
            session.append(_event(t, "listen", now))
        for t in hist[-2:]:
            now += 60_000
            session.append(_event(t, "skip", now, 3))
        prof = nova.Profile(lib, events + session, now)
        mood = nova.Mood.from_events(lib, list(reversed(session)), now)
        picks = nova.next_tracks(lib, prof, current=hist[-1], history=hist, n=3, now_ms=now, rng=random.Random(seed), mode=nova.MODE_MY, mood=mood)
        assert lib.tracks[picks[0].track_id].style == "Far", seed


def test_borderline_inside_a_session_holds_while_it_is_accepted():
    """Два лайка: половина потока в новом направлении. Слушатель не возражает — так и остаётся, не растёт и не гаснет."""
    base = [("Main", [LISTEN])] * 3 + [("a", [LIKE, LISTEN])] * 2
    start = _drift(base)
    assert start.main == "Main" and 0.35 < start.share("a") < 0.6
    long = _drift(base + [("Main", [LISTEN]), ("a", [LISTEN])] * 60)
    assert long.main == "Main" and abs(long.share("a") - start.share("a")) < 0.05, long.share("a")
    assert long.explore == start.explore


def test_borderline_fades_after_a_refusal_or_when_probes_stop_coming():
    base = [("Main", [LISTEN])] * 3 + [("a", [LIKE, LISTEN])] * 2
    start = _drift(base).share("a")
    assert _drift(base + [("a", [EARLY])]).share("a") < start * 0.7
    assert _drift(base + [("a", [EARLY])] * 2).share("a") < start / 3
    assert _drift(base + [("Main", [LISTEN])] * 120).share("a") < 0.15


def test_probes_without_a_like_do_not_hold_anything():
    passive = _drift([("Main", [LISTEN])] * 3 + [("a", [LISTEN]), ("Main", [LISTEN])] * 40)
    assert abs(passive.explore - 1 / 16) < 1e-6


def test_journal_shows_home():
    lib = make_library(FOUR_STYLES)
    picks, prof = _session(lib, [("Main", HEARD)] * 6 + [("Main", SKIPPED)], 1)
    assert picks and prof.homes


def test_a_tail_without_the_like_keeps_the_listener_where_he_is():
    """Старый плеер шлёт 30 треков: лайк, с которого начался сдвиг, из них выпал."""
    style_of = {i: "Main" if i <= 2 else "a" for i in range(1, 31)}
    d = D.follow(list(style_of), style_of.get, lambda t: [], lambda main: {"Main": 0.5, "b": 0.5}, "Main", 0.5)
    assert d.main == "a"
    probes = {i: "a" if i % 16 == 0 else "Main" for i in range(1, 31)}
    d = D.follow(list(probes), probes.get, lambda t: [], lambda main: {"a": 0.5, "b": 0.5}, "Main", 0.5)
    assert d.main == "Main"


def test_alternating_stream_stays_borderline_and_head_says_where_it_began():
    steps = {i: "Main" if i <= 3 or i % 2 == 0 else "a" for i in range(1, 60)}
    like = lambda ts: ("like", 100, False, None, None, ts)  # noqa: E731
    listen = lambda ts: ("listen", 100, False, None, None, ts)  # noqa: E731
    events = {t: [listen(t * 10)] for t in steps}
    events[5] = [like(49), listen(50)]
    events[7] = [like(69), listen(70)]
    d = D.follow(list(steps), steps.get, lambda t: events[t], lambda main: {"a": 0.5, "b": 0.5}, "Main", 0.5)
    assert d.main == "Main" and d.share("a") > 0.35


def test_shift_survives_a_pause_in_the_same_queue():
    """Таймера нет: вернулся к той же очереди через час, и сдвиг на месте."""
    lib = make_library(FOUR_STYLES)
    side = [t.id for t in lib.tracks.values() if t.style == "Side"]
    main = [t.id for t in lib.tracks.values() if t.style == "Main"]
    hist = main[50:54] + side[:3]
    events = [_event(t, "listen", NOW - 7_200_000 + i * 60_000) for i, t in enumerate(main[50:54])]
    for i, t in enumerate(side[:3]):
        events.append(_event(t, "like", NOW - 7_000_000 + i * 60_000))
        events.append(_event(t, "listen", NOW - 7_000_000 + i * 60_000 + 1))
    prof = nova.Profile(lib, events, NOW)
    mood = nova.Mood.from_events(lib, list(reversed(events)), NOW)
    assert mood.by_track[side[0]], "пауза дольше сессии, а действия помним"
    trace = {}
    nova.next_tracks(lib, prof, current=hist[-1], history=hist, n=3, now_ms=NOW, rng=random.Random(1), mode=nova.MODE_MY, mood=mood, trace=trace)
    assert trace["main"] == "Side"


# Находки независимой проверки 05.11.2026

def test_new_stream_start_cuts_yesterday_out_of_the_session():
    """Вчера Ambient×27, сегодня «Свой поток» с DnB: плеер шлёт весь хвост очереди, и основным становился вчерашний стиль."""
    import sqlite3

    from arxnova import db

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    db.migrate(conn)
    lib = make_library(FOUR_STYLES)
    far = [t.id for t in lib.tracks.values() if t.style == "Far"][:27]
    main = [t.id for t in lib.tracks.values() if t.style == "Main"][100:104]
    picks = [nova.Pick(t, "почему", 0.5, "taste") for t in main]
    nova.log_request(conn, 1, None, [nova.Pick(far[-1], "прогрев", 0.5, "taste")], None, {"start": True})  # совпало со старым: найдётся позже
    nova.log_request(conn, 1, None, [nova.Pick(999_999, "прогрев", 0.5, "taste")], None, {"start": True})  # прогрев, не пригодился
    nova.log_request(conn, 1, None, picks, None, {"start": True})
    nova.log_request(conn, 1, None, [nova.Pick(999_998, "прогрев", 0.5, "taste")], None, {"start": True})
    history = far + main
    assert nova.stream_start(conn, 1, history) == 27
    assert nova.stream_start(conn, 2, history) == 0, "другое устройство"
    prof = nova.Profile(lib, _days(lib, [("Main", 10, 20)]), NOW)
    trace = {}
    nova.next_tracks(lib, prof, current=history[-1], history=history, n=3, now_ms=NOW, rng=random.Random(1), mode=nova.MODE_MY, trace=trace, run_from=27)
    assert trace["main"] == "Main"


def test_undone_like_and_dislike_do_not_count():
    lib = make_library(FOUR_STYLES)
    side = [t.id for t in lib.tracks.values() if t.style == "Side"][0]
    events = [_event(side, "like", NOW - 300), _event(side, "unlike", NOW - 200), _event(side, "listen", NOW - 100)]
    mood = nova.Mood.from_events(lib, list(reversed(events)), NOW)
    assert [e[0] for e in mood.by_track[side]] == ["listen"]
    events = [_event(side, "dislike", NOW - 300), _event(side, "undislike", NOW - 200)]
    mood = nova.Mood.from_events(lib, list(reversed(events)), NOW)
    assert side not in mood.by_track and not mood.refused.get("Side", 0.0) > 0
    # лайк, снятие и снова лайк — лайк есть
    events = [_event(side, "like", NOW - 300), _event(side, "unlike", NOW - 200), _event(side, "like", NOW - 100)]
    assert [e[0] for e in nova.Mood.from_events(lib, list(reversed(events)), NOW).by_track[side]] == ["like"]


def test_like_then_skip_on_a_probe_is_a_like():
    """На снимке 6 из 16 лайков кончились пропуском: лайк и ранний пропуск давали 1 из 20, то есть почти ничего."""
    base = [("Main", [LISTEN])] * 3
    liked = _drift(base + [("a", [LIKE])]).share("a")
    assert liked > 0.1
    assert abs(_drift(base + [("a", [LIKE, EARLY])]).share("a") - liked) < 1e-9
    moved = base + [("a", [LIKE, EARLY])] * 3
    assert _drift(moved).main == "a"


def test_queued_probe_does_not_answer_a_request_to_leave():
    """Джаз, два пропуска, следующим в очереди уже стояла проба Ambient. Она не должна становиться основным стилем мимо принятых направлений."""
    steps = [("Main", [LISTEN])] * 3 + [("a", [LIKE, LISTEN])] * 3 + [("a", [EARLY])] * 2
    assert _drift(steps + [("b", [])]).leaving == "a", "ещё не прозвучала"
    heard = _drift(steps + [("b", [LISTEN])])
    assert heard.main == "b" and heard.leaving is None, "приняли — шаг вышел"
    refused = _drift(steps + [("b", [EARLY])])
    assert refused.leaving == "a", "пропустили и её — просьба уйти всё ещё в силе"


def test_going_back_and_skipping_the_same_track_is_one_refusal():
    from arxnova import drift as D

    early = lambda ts: ("skip", 3, False, None, None, ts)  # noqa: E731
    style_of = {1: "a", 2: "a", 3: "a", 4: "a"}
    events = {4: [early(1), early(2)]}
    d = D.follow([1, 2, 3, 4], style_of.get, lambda t: events.get(t, []), lambda main: {"b": 1.0}, "Main", 0.5)
    assert d.refusals == 1 and d.leaving is None
    events = {4: [early(1), early(3)], 3: [early(2)]}
    d = D.follow([1, 2, 3, 4], style_of.get, lambda t: events.get(t, []), lambda main: {"b": 1.0}, "Main", 0.5)
    assert d.refusals == 2 and d.leaving == "a", "а два разных трека — два отказа"


def test_back_button_and_own_browsing_do_not_punish_the_session():
    lib = make_library(FOUR_STYLES)
    side = [t.id for t in lib.tracks.values() if t.style == "Side"][:3]
    back = {**_event(side[0], "skip", NOW - 300, 3), "left_by": "back"}
    own = {**_event(side[1], "skip", NOW - 200, 3), "is_organic": 1}
    mood = nova.Mood.from_events(lib, [own, back], NOW)
    assert mood.refused.get("Side", 0.0) == 0.0 and side[0] not in mood.skipped and side[1] not in mood.skipped
    prof = nova.Profile(lib, [back], NOW)
    assert prof.style_neg.get("Side", 0.0) == 0.0 and prof.skip_mass.get(side[0], 0.0) == 0.0, "«назад» — вообще не отказ"
    prof = nova.Profile(lib, [own], NOW)
    assert prof.style_neg.get("Side", 0.0) > 0.0, "долгая память помнит и свои пропуски: без них на снимке Dubstep снова пошёл в пробы"


def test_direction_prior_is_not_shared_between_profiles():
    """Кэш был по id профиля, а id после сборки мусора повторяются: доля могла достаться чужому профилю."""
    lib = make_library(FOUR_STYLES)
    one = nova.Profile(lib, _days(lib, [("Main", 10, 20)]), NOW)
    two = nova.Profile(lib, _days(lib, [("Far", 10, 20)]), NOW)
    assert lib.direction_prior(one, "Side") != lib.direction_prior(two, "Side")
    assert "_direction_prior" in one.__dict__ and "_prior_cache" not in lib.__dict__


def test_accepted_direction_is_not_cut_for_being_recent():
    """Деление на «уже пробовали» резало и то, что слушатель лайкнул."""
    lib = make_library(FOUR_STYLES)
    prof = nova.Profile(lib, _days(lib, [("Main", 10, 20)]), NOW)
    mood = nova.Mood.from_events(lib, [], NOW)
    side = [t.id for t in lib.tracks.values() if t.style == "Side"][0]
    hist = list(range(1, 11)) + [side] + list(range(11, 16))
    drift = D.Drift("Main", "Main", 0.2, 1 / 16, {"Side": 0.5, "Far": 0.5}, accepted={"Side"}, played=15, probes=[(10, "Side")])
    got = Counter()
    for seed in range(300):
        p = nova._probe(lib, prof, hist[-1], hist, nova.Filters(), random.Random(seed), mood, drift, None, NOW, set(), lambda s: True)
        got[lib.tracks[p.track_id].style] += 1
    assert got["Side"] > 0.8 * got["Far"], got


def test_no_boldness_still_leaves_to_the_best_direction():
    """При смелости 0 все доли нулевые, и max брал первый ключ, а не лучшее направление."""
    lib = make_library(FOUR_STYLES)
    prof = nova.Profile(lib, _days(lib, [("Main", 10, 20)]), NOW)
    side = [t.id for t in lib.tracks.values() if t.style == "Side"][:8]
    events = [_event(t, "listen", NOW - 1000 + i) for i, t in enumerate(side[:6])] + [_event(t, "skip", NOW - 500 + i, 3) for i, t in enumerate(side[6:])]
    mood = nova.Mood.from_events(lib, list(reversed(events)), NOW)
    prof = nova.Profile(lib, _days(lib, [("Main", 10, 20)]) + events, NOW)
    for seed in range(8):
        picks = nova.next_tracks(lib, prof, current=side[-1], history=side, n=2, now_ms=NOW, rng=random.Random(seed), mode=nova.MODE_MY, mood=mood,
                                 filters=nova.Filters(familiar=0.5, boldness=0.0))
        assert lib.tracks[picks[0].track_id].style == "Main", seed


def test_slider_edges_hold_when_the_main_style_has_nothing_of_that_kind():
    """0 % нового в сессии Nordic: 70 новых из 121, потому что откат отдавал не тот разряд, хотя в других стилях нужный есть."""
    lib = make_library(FOUR_STYLES)
    main = [t.id for t in lib.tracks.values() if t.style == "Main"]
    far = [t.id for t in lib.tracks.values() if t.style == "Far"]
    habit = [_event(t, "listen", NOW - 40 * DAY_MS - i) for i, t in enumerate(main[:150])] + [_event(t, "listen", NOW - 40 * DAY_MS - 1000 - i) for i, t in enumerate(main[:150])]
    hist = far[:6]
    session = [_event(t, "listen", NOW - 900_000 + i * 60_000) for i, t in enumerate(hist)]
    prof = nova.Profile(lib, habit + session, NOW)
    mood = nova.Mood.from_events(lib, list(reversed(session)), NOW)
    assert prof.newness(far[10]) >= 0.5
    picks = nova.next_tracks(lib, prof, current=hist[-1], history=hist, n=3, now_ms=NOW, rng=random.Random(2), mode=nova.MODE_MY, mood=mood,
                             filters=nova.Filters(familiar=1.0, boldness=0.5))
    assert picks and all(prof.newness(p.track_id) < 0.5 for p in picks), [(p.why, lib.tracks[p.track_id].style) for p in picks]


def test_kind_used_up_by_recent_plays_leads_home_and_not_all_over_the_library():
    """Разряд в стиле есть, но всё только что звучало: откат выбирал самое просроченное, и поток скакал по стилям с каждым треком."""
    lib = make_library(FOUR_STYLES)
    main = [t.id for t in lib.tracks.values() if t.style == "Main"]
    side = [t.id for t in lib.tracks.values() if t.style == "Side"]
    far = [t.id for t in lib.tracks.values() if t.style == "Far"]
    old = [_event(t, "listen", NOW - 40 * DAY_MS - i) for i, t in enumerate(main[:100] + far[:10])] * 2
    hist = main[100:106] + side[:9]  # вся знакомая Side — это её последние треки
    session = [_event(t, "listen", NOW - 900_000 + i * 60_000) for i, t in enumerate(main[100:106])]
    for i, t in enumerate(side[:9]):
        session += [_event(t, "like", NOW - 500_000 + i * 2_000), _event(t, "listen", NOW - 500_000 + i * 2_000 + 1)]
    prof = nova.Profile(lib, old + session, NOW)
    mood = nova.Mood.from_events(lib, list(reversed(session)), NOW)
    trace: dict = {}
    picks = nova.next_tracks(lib, prof, current=hist[-1], history=hist, n=3, now_ms=NOW, rng=random.Random(4), mode=nova.MODE_MY, mood=mood,
                             filters=nova.Filters(familiar=1.0), trace=trace)
    assert trace["main"] == "Side"
    assert all(prof.newness(p.track_id) < 0.5 for p in picks)
    assert [lib.tracks[p.track_id].style for p in picks] == ["Main"] * 3, [(p.why, lib.tracks[p.track_id].style) for p in picks]


def test_the_stream_keeps_its_mood_on_a_fallback():
    """Откат терял настрой: журнал говорил одно, а поток играл другое (основной Nordic, а отдан DnB)."""
    lib = make_library(FOUR_STYLES)
    side = [t.id for t in lib.tracks.values() if t.style == "Side"]
    main = [t.id for t in lib.tracks.values() if t.style == "Main"][50:56]
    hist = main + side[:3]
    events = [_event(t, "listen", NOW - 900_000 + i) for i, t in enumerate(main)]
    for i, t in enumerate(side[:3]):
        events += [_event(t, "like", NOW - 800_000 + i * 2), _event(t, "listen", NOW - 800_000 + i * 2 + 1)]
    prof = nova.Profile(lib, _days(lib, [("Main", 10, 20)]) + events, NOW)
    mood = nova.Mood.from_events(lib, list(reversed(events)), NOW)
    # вся Side уже знакома и звучала только что — строгому отбору в этом стиле нечего предложить, а настрой должен дойти до отката
    seen = []
    real = nova.next_tracks

    def spy(*a, **k):
        seen.append(k.get("mood"))
        return real(*a, **k)

    nova.next_tracks = spy
    try:
        spy(lib, prof, current=hist[-1], history=hist, n=3, now_ms=NOW, rng=random.Random(1), mode=nova.MODE_MY, mood=mood, filters=nova.Filters(familiar=1.0))
    finally:
        nova.next_tracks = real
    assert len(seen) >= 1 and all(m is mood for m in seen), [m is mood for m in seen]


def test_sound_tile_is_not_a_trap():
    """Плитка chill: Contemporary Jazz×43, сколько ни пропускай. Плитка охватывает разные стили: отказ уводит в другой стиль, но внутри неё."""
    lib = make_library(FOUR_STYLES)
    side = [t.id for t in lib.tracks.values() if t.style == "Side"][:8]
    events = [_event(t, "listen", NOW - 1000 + i) for i, t in enumerate(side[:6])] + [_event(t, "skip", NOW - 500 + i, 3) for i, t in enumerate(side[6:])]
    habit = _days(lib, [("Main", 10, 20)])
    prof = nova.Profile(lib, habit + events, NOW)
    mood = nova.Mood.from_events(lib, list(reversed(events)), NOW)
    tile = nova.Filters(vocal="yes", boldness=0.5)
    picks = nova.next_tracks(lib, prof, current=side[-1], history=side, n=3, now_ms=NOW, rng=random.Random(1), mode=nova.MODE_MY, mood=mood, filters=tile)
    assert picks and lib.tracks[picks[0].track_id].style != "Side"
    assert all(nova._filter_ok(lib, p.track_id, tile) for p in picks)


def test_journal_probe_chance_matches_how_often_the_direction_is_taken():
    """prob в журнале был условным на одну случайную выборку: трек с prob 0,46–0,49 реально выпадал в 1,3 % случаев."""
    lib = make_library(FOUR_STYLES)
    prof = nova.Profile(lib, _days(lib, [("Main", 10, 20)]), NOW)
    mood = nova.Mood.from_events(lib, [], NOW)
    hist = list(range(1, 11))
    drift = D.Drift("Main", "Main", 1 / 16, 1 / 16, {"Side": 0.7, "Far": 0.2, "Mid": 0.1})
    got, said = Counter(), {}
    for seed in range(600):
        p = nova._probe(lib, prof, hist[-1], hist, nova.Filters(), random.Random(seed), mood, drift, None, NOW, set(), lambda s: True)
        style = lib.tracks[p.track_id].style
        got[style] += 1
        said.setdefault(style, []).append(p.prob)
    for style, n in got.items():
        assert abs(n / 600 - sum(said[style]) / len(said[style])) < 0.12, (style, n / 600, sum(said[style]) / len(said[style]))
    assert 0.0 == D.Drift("Main", "Main", 0.1, 0.1).gap_chance(5) and D.Drift("Main", "Main", 0.1, 0.1).gap_chance(13) == 1.0


def test_main_picks_do_not_pretend_to_have_a_chance():
    lib = make_library(FOUR_STYLES)
    picks, prof = _session(lib, [("Main", HEARD)] * 6, 1)
    entry = nova.journal(lib, prof, picks, {})
    assert all(x["prob"] is None and x["slot"] is None and x["track"] is None for x in entry["picks"] if x["why"] != "spice")
    assert nova._sig(0.000123456) == 0.000123 and nova._sig(None) is None


def test_a_map_is_not_built_from_a_few_tracks(tmp_path):
    """Настоящий разбор одного трека: UMAP падал на n_neighbors, и с ним вся выгрузка индекса. На медиатеке в пару треков."""
    import numpy as np

    from arxnova import config, indexer, models

    cfg = config.Config(data_dir=tmp_path / "data", libraries=[])
    cfg.ensure_dirs()
    d = cfg.index_dir / models.EFFNET.id
    d.mkdir(parents=True)
    np.save(d / "mix.npy", np.random.default_rng(1).normal(size=(2, 8)).astype(np.float32))
    assert indexer.export_map(cfg) is None


# Находки второй проверки 05.11.2026

def test_a_single_track_on_the_first_day_does_not_hold_home():
    """Один трек Tribal, на другой день 25 треков DnB: дом Tribal с долей 0,11, и 30 запусков из 30 начинались в нём."""
    home = D.home_of([Counter({"Tribal": 1}), Counter({"DnB": 25})])
    assert next(iter(home)) == "DnB"
    assert next(iter(D.home_of([Counter({"A": 20})] * 4 + [Counter({"B": 20})] * 2))) == "A", "а сложившаяся привычка держится"


def test_a_tiny_style_is_not_where_a_session_starts():
    lib = make_library((("Main", 300), ("Tiny", 12), ("Far", 200)))
    tiny = [t.id for t in lib.tracks.values() if t.style == "Tiny"]
    far = [t.id for t in lib.tracks.values() if t.style == "Far"]
    start = _morning(NOW - 12 * DAY_MS)
    events = []
    for d in range(8):
        ids = tiny[:10] if d % 2 == 0 else far[d * 2:d * 2 + 8]
        events += [_event(t, "listen", start + d * DAY_MS + i) for i, t in enumerate(ids)]
    prof = nova.Profile(lib, events, NOW)
    assert {"Tiny", "Far"} <= set(prof.homes), prof.homes
    assert _start_styles(lib, prof, seeds=30)["Far"] == 30


def test_a_used_up_tiny_style_hands_over_to_a_normal_stream():
    """Известен один маленький стиль: основным он оставался все 60 треков, и каждый выбирался запасным путём, без шлюза и проб."""
    lib = make_library((("Main", 300), ("Tiny", 12), ("Far", 60)))
    tiny = [t.id for t in lib.tracks.values() if t.style == "Tiny"]
    hist = tiny[:12]
    now = NOW - 7_200_000
    events = []
    for t in hist:
        now += 60_000
        events.append(_event(t, "listen", now))
    rng = random.Random(1)
    picks = []
    for _ in range(14):
        now += 180_000
        prof = nova.Profile(lib, events, now)
        mood = nova.Mood.from_events(lib, list(reversed(events)), now)
        batch = nova.next_tracks(lib, prof, current=hist[-1], history=hist, n=3, now_ms=now, rng=rng, mode=nova.MODE_MY, mood=mood)
        assert batch
        for p in batch:
            picks.append(p)
            hist.append(p.track_id)
            events.append(_event(p.track_id, "listen", now))
    styles = [lib.tracks[p.track_id].style for p in picks]
    assert styles[-12:].count("Tiny") == 0, styles
    assert sum(p.why == "restyle" for p in picks[-20:]) == 0, [p.why for p in picks]
    assert len(set(styles[-12:])) <= 2, "не скачет по стилям"


def test_direction_shares_stay_a_proper_distribution_with_few_styles():
    """Два дома при малом числе стилей давали {H1: 0,84, H2: 0,21, Other: −0,05}, и выбор пробы падал."""
    lib = make_library((("H1", 300), ("H2", 40), ("Other", 40), ("Main", 40)))
    h1 = [t.id for t in lib.tracks.values() if t.style == "H1"][:60]
    h2 = [t.id for t in lib.tracks.values() if t.style == "H2"][:30]
    events = []
    start = _morning(NOW - 30 * DAY_MS)
    for d in range(12):
        ids = h1[d * 4:d * 4 + 8] if d % 2 == 0 else h2[(d * 2) % 20:(d * 2) % 20 + 8]
        events += [_event(t, "listen", start + d * DAY_MS + i) for i, t in enumerate(ids)]
    prof = nova.Profile(lib, events, NOW)
    for main in ("Main", "Other"):
        prior = lib.direction_prior(prof, main)
        assert all(x >= 0.0 for x in prior.values()) and abs(sum(prior.values()) - 1.0) < 1e-6, prior


def test_what_was_main_before_a_drift_does_not_count_as_tried():
    """Дом занимал 20 из 30 последних мест, и деление на «уже пробовали» оставляло ему 6,5 % проб вместо половины."""
    style_of = {i: "Main" if i <= 20 else "a" for i in range(1, 31)}
    like = lambda ts: ("like", 100, False, None, None, ts)  # noqa: E731
    listen = lambda ts: ("listen", 100, False, None, None, ts)  # noqa: E731
    events = {i: [listen(i * 10)] for i in style_of}
    for i in (21, 22, 23):
        events[i] = [like(i * 10 - 1), listen(i * 10)]
    d = D.follow(list(style_of), style_of.get, lambda t: events[t], lambda main: {"Main": 0.5, "b": 0.5} if main == "a" else {"a": 0.5, "b": 0.5}, "Main", 0.5)
    assert d.main == "a" and d.tried()["Main"] == 0
    style_of[31] = "Main"
    events[31] = [listen(310)]
    d = D.follow(list(style_of), style_of.get, lambda t: events[t], lambda main: {"Main": 0.5, "b": 0.5} if main == "a" else {"a": 0.5, "b": 0.5}, "Main", 0.5)
    assert d.tried()["Main"] == 1, "а настоящая проба считается"


def test_stream_start_looks_at_the_first_track_of_the_batch():
    """Другой трек пачки звучал раньше в этом же хвосте: брали его старое место, и в сессию попадал вчерашний день."""
    import sqlite3

    from arxnova import db

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    db.migrate(conn)
    nova.log_request(conn, 1, None, [nova.Pick(50, "", 0.5, "taste"), nova.Pick(5, "", 0.5, "taste")], None, {"start": True})
    assert nova.stream_start(conn, 1, [1, 2, 5, 3, 4, 50]) == 5
    assert nova.stream_start(conn, 1, [1, 2, 5, 3, 4]) == 0, "первый трек ещё не звучал"



def test_actions_from_before_the_run_do_not_count_for_its_first_track():
    """Вчерашний лайк трека, что открыл сегодняшний поток: на старте сразу 1 из 6 в его стиль."""
    lib = make_library(FOUR_STYLES)
    side = [t.id for t in lib.tracks.values() if t.style == "Side"][0]
    main = [t.id for t in lib.tracks.values() if t.style == "Main"][100:104]
    before = [t.id for t in lib.tracks.values() if t.style == "Main"][200]
    history = [before, side, *main]
    events = [_event(side, "like", NOW - 86_400_000), _event(before, "listen", NOW - 86_300_000)]
    events += [_event(t, "listen", NOW - 600_000 + i * 60_000) for i, t in enumerate(history[1:])]
    prof = nova.Profile(lib, events, NOW)
    mood = nova.Mood.from_events(lib, list(reversed(events)), NOW)
    trace = {}
    nova.next_tracks(lib, prof, current=history[-1], history=history, n=1, now_ms=NOW, rng=random.Random(1), mode=nova.MODE_MY, mood=mood, trace=trace, run_from=1)
    assert abs(trace["explore"] - 1 / 16) < 0.01, trace["explore"]


def test_session_actions_count_even_if_the_device_clock_is_far_behind_the_server():
    """Сверка времени устройства со временем запроса сервера молча выбрасывала все лайки, если часы устройства отстают."""
    lib = make_library(FOUR_STYLES)
    side = [t.id for t in lib.tracks.values() if t.style == "Side"][:3]
    main = [t.id for t in lib.tracks.values() if t.style == "Main"][100:106]
    before = [t.id for t in lib.tracks.values() if t.style == "Main"][200]
    history = [before, *main, *side]
    behind = NOW - 3 * 3_600_000  # часы устройства отстают на три часа
    events = [_event(before, "listen", behind - 1_000_000)]
    events += [_event(t, "listen", behind - 500_000 + i) for i, t in enumerate(main)]
    for i, t in enumerate(side):
        events += [_event(t, "like", behind - 400_000 + i * 2), _event(t, "listen", behind - 400_000 + i * 2 + 1)]
    prof = nova.Profile(lib, _days(lib, [("Main", 10, 20)]) + events, NOW)
    mood = nova.Mood.from_events(lib, list(reversed(events)), NOW)
    trace = {}
    nova.next_tracks(lib, prof, current=history[-1], history=history, n=1, now_ms=NOW, rng=random.Random(1), mode=nova.MODE_MY, mood=mood, trace=trace, run_from=1)
    assert trace["main"] == "Side"


def test_rest_of_a_leaving_batch_does_not_repeat_the_track_at_the_end_of_the_queue():
    """Дозагрузка шлёт последний трек очереди, которого нет в истории: остаток пачки ухода мог его повторить."""
    lib = make_library((("Main", 34), ("Side", 40)))
    side = [t.id for t in lib.tracks.values() if t.style == "Side"][:8]
    main = [t.id for t in lib.tracks.values() if t.style == "Main"]
    events = [_event(t, "listen", NOW - 1000 + i) for i, t in enumerate(side[:6])] + [_event(t, "skip", NOW - 500 + i, 3) for i, t in enumerate(side[6:])]
    prof = nova.Profile(lib, _days(lib, [("Main", 4, 8)]) + events, NOW)
    mood = nova.Mood.from_events(lib, list(reversed(events)), NOW)
    for seed in range(40):
        last = main[seed % 30]
        picks = nova.next_tracks(lib, prof, current=last, history=side, n=3, now_ms=NOW, rng=random.Random(seed), mode=nova.MODE_MY, mood=mood)
        assert last not in [p.track_id for p in picks], seed


def test_dislike_after_a_like_wins():
    """Лайк и «не предлагать» в одном показе считались лайком: плеер при дизлайке снятие лайка не пишет."""
    base = [("Main", [LISTEN])] * 3
    dislike = ("dislike", 100, False, None, None)
    both = _drift(base + [("a", [LIKE, dislike])])
    alone = _drift(base + [("a", [dislike])])
    assert abs(both.share("a") - alone.share("a")) < 1e-9 and both.share("a") < 0.01
    assert _drift(base + [("a", [dislike, LIKE])]).share("a") > 0.1, "передумал в другую сторону"


def test_an_evening_across_midnight_is_one_listening_day():
    offset = 3 * 3600
    evening = (20_000 * 86_400 - offset + 23 * 3600) * 1000
    assert nova.listening_day(evening, offset) == nova.listening_day(evening + 2 * 3_600_000, offset)
    assert nova.listening_day(evening, offset) != nova.listening_day(evening + 8 * 3_600_000, offset)


def test_slot_is_the_chance_of_this_very_place():
    d = D.Drift("Main", "Main", 0.1, 0.1)
    assert d.gap_chance(10) == 0.5 and abs(d.gap_chance(11, first=False) - 0.2) < 1e-9
    assert d.gap_chance(5, first=False) == 0.0


def test_borderline_survives_a_long_session_on_the_new_player():
    """Лайки выпадали из 150 последних событий, и граница гасла примерно на 150-м треке."""
    lib = make_library(FOUR_STYLES)
    main = [t.id for t in lib.tracks.values() if t.style == "Main"]
    side = [t.id for t in lib.tracks.values() if t.style == "Side"]
    hist, events = [], []
    now = NOW - 40_000_000
    for i in range(320):
        now += 60_000
        t = side[i % 60] if i >= 4 and i % 2 == 0 else main[100 + i % 300]
        hist.append(t)
        if i in (4, 6):
            events.append(_event(t, "like", now - 1000))
        events.append(_event(t, "listen", now))
    mood = nova.Mood.from_events(lib, list(reversed(events)), now)
    prof = nova.Profile(lib, _days(lib, [("Main", 10, 30)]) + events, now)
    assert next(iter(prof.homes)) == "Main"
    trace = {}
    nova.next_tracks(lib, prof, current=hist[-1], history=hist, n=1, now_ms=now, rng=random.Random(1), mode=nova.MODE_MY, mood=mood, trace=trace)
    assert trace["main"] == "Main" and trace["explore"] > 0.4, (trace["main"], trace["explore"])
