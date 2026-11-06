"""Межжанровые переходы в потоке (владелец 04.11.2026: «сколько часов слушаю, 0 межжанровых
переходов»). Медиатека как у него: один стиль подавляет остальные. Поток обязан выходить из основного
стиля через мост, держаться нового стиля и возвращаться. На прежнем шлюзе мостов было ноль."""

from __future__ import annotations

import itertools
import random

import numpy as np

from arxnova import gate as G
from arxnova import nova

COLS = ["bpm", "bpm_conf", "musical_key", "key_conf", "loudness_db", "onset_rate", "vocal_frac", "vocal_ratio"]
DIM = 32
NOW = 1_800_000_000_000


def _unit(m):
    return m / np.linalg.norm(m, axis=-1, keepdims=True)


def make_library(sizes=(("Main", 400), ("Side", 60), ("Far", 60))) -> nova.Library:
    """Основной стиль и два небольших; центры стилей близки настолько, что соседи через границу есть,
    но своего всегда ближе, чем чужого (как DnB в медиатеке владельца)."""
    rng = np.random.default_rng(5)
    base = _unit(rng.normal(size=DIM))
    centers = {name: _unit(base + 0.35 * rng.normal(size=DIM)) for name, _ in sizes}
    styles = [name for name, k in sizes for _ in range(k)]
    n = len(styles)
    mix = _unit(np.stack([centers[s] + 0.22 * rng.normal(size=DIM) for s in styles])).astype(np.float32)
    feats = np.zeros((n, len(COLS)), np.float32)
    feats[:, 0] = 170.0
    feats[:, 1] = 0.9
    feats[:, 4] = -10.0
    feats[:, 5] = 4.0
    feats[:, 6] = 0.6
    scores = np.zeros((n, 3), np.float32)
    scores[:, 0], scores[:, 1] = 0.6, 0.1

    lib = nova.Library.__new__(nova.Library)
    lib.ids = np.arange(1, n + 1)
    lib.row = {i + 1: i for i in range(n)}
    lib.vec = mix
    lib.hub = nova.hub_scores(mix)
    lib.style_scores = scores
    lib.style_labels = ["Electronic---A", "Electronic---B", "Electronic---C"]
    lib.dj_tools = set()
    lib.shelves, lib.shelf_genre, lib.scenarios, lib.bpm_of = {}, {}, {}, {}
    lib.tracks = {
        i + 1: nova.TrackInfo(i + 1, f"Artist {i}", f"Title {i}", f"Album {i}", frozenset({f"artist {i}"}), f"album {i}",
                              False, bytes(str(i), "ascii"), 300.0, "Electronic", styles[i])
        for i in range(n)
    }
    lib._style_genre = {s: "Electronic" for s in styles}
    from collections import Counter

    lib.style_size = Counter(styles)
    lib.main_style = sizes[0][0]
    lib.gate = G.Gate(mix, styles, scores, feats, COLS)
    return lib


def _listened(tids, ts):
    return [{"track_id": t, "event_type": "listen", "ts": ts, "is_organic": 1, "played_ratio_pct": 100,
             "skip_reason": None, "prev_track_id": None, "device_id": 1} for t in tids]


def _stream(lib, total=160, seed=3, filters=None):
    """Как плеер: пачками по 4, опора — последний трек пачки, история — последние 30."""
    prof = nova.Profile(lib, _listened(range(1, 41), NOW - 86_400_000), NOW)
    rng, hist, picks = random.Random(seed), [], []
    start = True
    while len(picks) < total:
        batch = nova.next_tracks(lib, prof, current=hist[-1] if hist else None, history=hist[-30:], n=4, now_ms=NOW,
                                 rng=rng, mode=nova.MODE_MY, start=start, filters=filters)
        assert batch, "поток не должен замолкать"
        start = False
        picks += batch
        hist += [p.track_id for p in batch]
    return picks


def test_passive_listener_stays_in_the_main_style_with_rare_probes():
    """Владелец 05.11.2026: чтоб дрейф состоялся, слушатель сам должен сместить акцент. Без действий — основной стиль и пробы."""
    lib = make_library()
    ls = Listener(lib, _heard)
    ls.step(80)
    styles = ls.styles()
    runs = [(k, len(list(g))) for k, g in itertools.groupby(styles)]
    assert styles.count("Main") / len(styles) >= 0.85, runs
    assert all(n == 1 for k, n in runs if k != "Main"), runs
    probes = [p for p in ls.picks if p.why == "spice"]
    assert 8 <= len(probes) <= 30, len(probes)
    assert all(p.why not in ("bridge", "restyle", "leave") for p in ls.picks)


def _likes_probes_of(style):
    return lambda t, p: ("like", 100) if t.style == style and p.why == "spice" else ("listen", 100)


def test_likes_on_probes_shift_the_main_style_step_by_step():
    """1 из 16 → чаще → основной; после этого обратно по сроку не выталкивает."""
    lib = make_library()
    ls = Listener(lib, _likes_probes_of("Side"))
    ls.step(60)
    styles = ls.styles()
    last = styles[-40:]
    assert last.count("Side") / len(last) >= 0.7, [(k, len(list(g))) for k, g in itertools.groupby(styles)]
    first_run = next(i for i, s in enumerate(styles) if s == "Side")
    assert first_run >= 3, "сначала основной стиль"
    tail = Listener(lib, _heard)  # лайков больше нет, а стиль остаётся
    tail.events, tail.hist, tail.picks, tail.now = ls.events, ls.hist, ls.picks, ls.now
    before = len(tail.picks)
    tail.step(10)
    after = [lib.tracks[p.track_id].style for p in tail.picks[before:]]
    assert after.count("Side") / len(after) >= 0.7, after


def test_leaving_goes_through_a_bridge_when_there_is_one():
    """Мост проверен от того трека основного стиля, от которого уходим."""
    lib = make_library()
    main = [t.id for t in lib.tracks.values() if t.style == "Main"]
    now, evs = NOW, []
    for t in main[:6]:
        now += 60_000
        evs.append(_event(t, "listen", now))
    for t in main[6:8]:
        now += 60_000
        evs.append(_event(t, "skip", now, 3))
    hist = main[:8]
    bridges = 0
    for seed in range(8):
        picks = _next(lib, _listened(main[20:60], NOW - 86_400_000), evs, hist, now, seed=seed)
        if picks[0].why == "bridge":
            bridges += 1
            assert lib.gate.check(lib.row[hist[-1]], lib.row[picks[0].track_id], pct=lib.gate.pct(0.9)).ok
    assert bridges, "мост есть, его и берём"


def test_no_bridge_on_the_first_tracks_of_a_session():
    lib = make_library()
    picks = _stream(lib, total=8)
    assert all(p.why != "bridge" for p in picks[:8])


def _event(tid, et, ts, pct=100, reason=None):
    return {"track_id": tid, "event_type": et, "ts": ts, "is_organic": 0, "played_ratio_pct": pct,
            "skip_reason": reason, "prev_track_id": None, "device_id": 1}


class Listener:
    """Прогон потока с откликом: что слушатель делает с каждым треком, сервер видит сразу."""

    def __init__(self, lib, react, seed=3, filters=None):
        self.lib, self.react, self.filters = lib, react, filters
        self.base = _listened(range(1, 41), NOW - 86_400_000)
        self.rng = random.Random(seed)
        self.events, self.hist, self.picks = [], [], []
        self.now = NOW

    def step(self, batches=1):
        for _ in range(batches):
            mood = nova.Mood.from_events(self.lib, list(reversed(self.events)), self.now)
            prof = nova.Profile(self.lib, self.base + self.events, self.now)  # как на сервере: профиль идёт за событиями
            batch = nova.next_tracks(self.lib, prof, current=self.hist[-1] if self.hist else None,
                                     history=self.hist[-150:], n=3, now_ms=self.now, rng=self.rng, mode=nova.MODE_MY,
                                     start=not self.hist, mood=mood, filters=self.filters)
            for p in batch:
                self.picks.append(p)
                self.hist.append(p.track_id)
                self.now += 60_000
                et, pct = self.react(self.lib.tracks[p.track_id], p)
                self.events.append(_event(p.track_id, et, self.now, pct))

    def styles(self):
        return [self.lib.tracks[p.track_id].style for p in self.picks]


def _heard(t, p):
    return "listen", 100


def test_two_early_skips_lead_out_of_the_main_style_right_away():
    lib = make_library()
    main = [t.id for t in lib.tracks.values() if t.style == "Main"]
    now, evs = NOW, []
    for t in main[:6]:
        now += 60_000
        evs.append(_event(t, "listen", now))
    for t in main[6:8]:
        now += 60_000
        evs.append(_event(t, "skip", now, 3))
    for seed in range(6):
        picks = _next(lib, _listened(main[20:60], NOW - 86_400_000), evs, main[:8], now, seed=seed)
        assert lib.tracks[picks[0].track_id].style != "Main", [p.why for p in picks]


def test_skipped_probes_of_a_style_fade_and_others_remain():
    """Пропускает пробы Side — их почти не остаётся, а Far звучит; основной стиль не меняется."""
    lib = make_library()
    ls = Listener(lib, lambda t, p: ("skip", 3) if t.style == "Side" else ("listen", 100))
    ls.step(100)
    styles = ls.styles()
    assert styles.count("Far") > styles.count("Side"), (styles.count("Far"), styles.count("Side"))
    assert styles[-30:].count("Main") >= 20


def test_refusing_the_main_style_goes_where_probes_were_accepted():
    lib = make_library()
    main = [t.id for t in lib.tracks.values() if t.style == "Main"]
    far = [t.id for t in lib.tracks.values() if t.style == "Far"]
    now, evs = NOW, []
    hist = main[:6] + far[:1] + main[6:8]
    for t in main[:6]:
        now += 60_000
        evs.append(_event(t, "listen", now))
    now += 60_000
    evs.append(_event(far[0], "like", now))
    evs.append(_event(far[0], "listen", now))
    for t in main[6:8]:
        now += 60_000
        evs.append(_event(t, "skip", now, 3))
    got = [lib.tracks[_next(lib, _listened(main[20:60], NOW - 86_400_000), evs, hist, now, seed=s)[0].track_id].style for s in range(6)]
    assert got.count("Far") >= 5, got


def test_spice_is_a_single_track_close_in_feel_and_stream_goes_on():
    lib = make_library()
    picks = _stream(lib, total=240, seed=5)
    spices = [i for i, p in enumerate(picks) if p.why == "spice"]
    assert 5 <= len(spices) <= 40, len(spices)
    g = lib.gate
    for i in spices:
        if not 0 < i < len(picks) - 1:
            continue
        before, x, after = (lib.tracks[picks[j].track_id] for j in (i - 1, i, i + 1))
        assert before.style == after.style != x.style, "после подмеса поток идёт как шёл"
        ra, rx = lib.row[before.id], lib.row[x.id]
        assert abs(g.energy[ra] - g.energy[rx]) <= G.FEEL_ENERGY_TOL and g.vocal[ra] == g.vocal[rx]
    assert all(b - a >= 5 for a, b in zip(spices, spices[1:]))


def test_skipped_spice_pushes_the_next_one_away():
    lib = make_library()
    gaps = {"heard": [], "skipped": []}
    for seed in range(8):
        for name, react in (("heard", lambda t, p: ("listen", 100)),
                            ("skipped", lambda t, p: ("skip", 3) if p.why == "spice" else ("listen", 100))):
            ls = Listener(lib, react, seed=seed)
            ls.step(60)
            idx = [i for i, p in enumerate(ls.picks) if p.why == "spice"]
            gaps[name] += list(np.diff(idx))
    assert np.mean(gaps["skipped"]) > np.mean(gaps["heard"]) * 1.2, {k: float(np.mean(v)) for k, v in gaps.items()}


def test_chosen_style_or_sound_filter_is_played_as_chosen():
    """Выбрал стиль нарочно — ни мостов, ни подмеса, даже если всё пропускает."""
    lib = make_library()
    for filters in (nova.Filters(styles=["Side"]), nova.Filters(genres=["Electronic"])):
        picks = _stream(lib, total=60, filters=filters)
        assert all(p.why not in ("bridge", "spice") for p in picks)
    picks = _stream(lib, total=60, filters=nova.Filters(styles=["Side"]))
    assert {lib.tracks[p.track_id].style for p in picks} == {"Side"}


def test_similar_to_track_mode_has_no_mixing():
    lib = make_library()
    prof = nova.Profile(lib, _listened(range(1, 41), NOW - 86_400_000), NOW)
    hist, rng = [], random.Random(1)
    for _ in range(30):
        batch = nova.next_tracks(lib, prof, current=hist[-1] if hist else 1, history=hist[-30:], n=4, now_ms=NOW, rng=rng,
                                 mode=nova.MODE_TRACK, seed=1)
        assert all(p.why != "spice" for p in batch)
        hist += [p.track_id for p in batch]


def test_long_memory_style_that_never_lands_is_not_offered():
    lib = make_library()
    side = [t.id for t in lib.tracks.values() if t.style == "Side"][:6]
    events = _listened(range(1, 41), NOW - 86_400_000) + [_event(t, "dislike", NOW - 3_600_000) for t in side]
    prof = nova.Profile(lib, events, NOW)
    assert prof.style_refused("Side") and not prof.style_refused("Far")
    assert prof.style_accept("Main") > 0.5 > prof.style_accept("Side")


def _offers_from(lib, prof, style, target, tries=300):
    """Сколько раз в target ведёт мост или подмес, когда поток долго идёт в style."""
    own = [t.id for t in lib.tracks.values() if t.style == style]
    hits = 0
    for seed in range(tries):
        rng = random.Random(seed)
        hist = rng.sample(own, 20)
        batch = nova.next_tracks(lib, prof, current=hist[-1], history=hist, n=3, now_ms=NOW, rng=rng, mode=nova.MODE_MY,
                                 filters=nova.Filters(familiar=0.0))
        hits += any(lib.tracks[p.track_id].style == target and p.why in ("bridge", "spice") for p in batch)
    return hits


def test_unloved_style_is_rare_and_only_from_a_close_style():
    """Не запрет: из близкого стиля отвергнутый иногда звучит, но редко; из любимого — никогда."""
    lib = make_library()
    side = [t.id for t in lib.tracks.values() if t.style == "Side"][:6]
    events = _listened(range(1, 41), NOW - 86_400_000) + [_event(t, "dislike", NOW - 3_600_000) for t in side]
    prof = nova.Profile(lib, events, NOW)
    assert prof.style_refused("Side")
    calm = nova.Profile(lib, _listened(range(1, 41), NOW - 86_400_000), NOW)
    usual = _offers_from(lib, calm, "Far", "Side")
    rare = _offers_from(lib, prof, "Far", "Side")
    assert usual > 5, usual
    assert rare < usual / 3, (rare, usual)
    # не запрет: когда редкий случай выпал и другого пути нет, стиль звучит
    from arxnova import drift as D

    side_only = D.Drift("Far", "Main", 0.1, 0.1, {"Side": 1.0})
    hist = [t.id for t in lib.tracks.values() if t.style == "Far"][:5]
    allowed = nova._probe(lib, prof, hist[-1], hist, nova.Filters(), random.Random(1), nova.Mood(), side_only, None, NOW, set(), lambda s: True)
    assert allowed is not None and lib.tracks[allowed.track_id].style == "Side"
    assert nova._probe(lib, prof, hist[-1], hist, nova.Filters(), random.Random(1), nova.Mood(), side_only, None, NOW, set(), lambda s: False) is None


def test_unloved_style_not_close_is_not_offered():
    lib = make_library()
    lib.styles_close = lambda a, b: False
    side = [t.id for t in lib.tracks.values() if t.style == "Side"][:6]
    events = _listened(range(1, 41), NOW - 86_400_000) + [_event(t, "dislike", NOW - 3_600_000) for t in side]
    assert _offers_from(lib, nova.Profile(lib, events, NOW), "Far", "Side") == 0


def test_slider_works_when_the_profile_refreshes_during_the_session():
    """Дослушанное становится знакомым по ходу, а доля нового всё равно держится у ползунка (было 0,96 / 1 / 1)."""
    lib = make_library()
    low, mid, high = (sum(x[3:]) / len(x[3:]) for x in (_served_days(lib, v, days=14) for v in (0.2, 0.5, 0.8)))
    assert low < mid < high, (low, mid, high)
    assert abs(low - 0.2) < 0.12 and abs(mid - 0.5) < 0.12 and abs(high - 0.8) < 0.12, (low, mid, high)


def _lib_all_heard():
    lib = make_library()
    now = NOW
    evs = []
    for t in lib.tracks:  # каждый трек слышан: половина один раз год назад, половина часто и недавно
        if t % 2:
            evs.append(_event(t, "listen", now - 330 * 86_400_000))
        else:
            evs += [_event(t, "listen", now - k * 86_400_000) for k in (40, 60, 90)]
    return lib, evs


def test_when_the_whole_library_is_heard_new_is_what_was_forgotten():
    """«В конце концов я ознакомлюсь с 100 % медиатеки — что будет новым?» Самое забытое, и его хватает."""
    lib, evs = _lib_all_heard()
    prof = nova.Profile(lib, evs, NOW)
    pool = [t for t in lib.tracks if prof.newness(t) >= 0.5]
    assert len(pool) >= nova.NEW_POOL_MIN
    assert sum(1 for t in pool if t % 2) > 0.8 * len(pool), "новым считается забытое, а не частое и свежее"
    assert all(prof.forgotten(t) for t in pool)
    rng, hist = random.Random(1), []
    f = nova.Filters(familiar=0.0)
    for _ in range(20):
        batch = nova.next_tracks(lib, prof, current=hist[-1] if hist else None, history=hist[-30:], filters=f, n=3,
                                 now_ms=NOW, rng=rng, mode=nova.MODE_MY, start=not hist)
        assert batch, "правая часть ползунка не пустеет"
        hist += [p.track_id for p in batch]
    assert sum(1 for t in hist if t in pool) / len(hist) > 0.8
    assert {p.why for p in nova.next_tracks(lib, prof, current=hist[-1], history=hist[-30:], filters=f, n=3, now_ms=NOW,
                                            rng=rng, mode=nova.MODE_MY)} & {"forgotten", "bridge", "spice"}


def test_an_unheard_track_is_still_simply_new():
    lib = make_library()
    prof = nova.Profile(lib, _listened(range(1, 41), NOW - 86_400_000), NOW)
    assert prof.new_mass == -1.0
    assert prof.newness(200) == 1.0 and not prof.forgotten(200)
    assert prof.newness(5) < 0.5


def test_rest_scales_with_known_pool_and_pace():
    lib = make_library()
    day = 86_400_000
    small = nova.Profile(lib, [_event(t, "listen", NOW - (t % 14) * day) for t in range(1, 41)], NOW)
    big = nova.Profile(lib, [_event(t, "listen", NOW - t * day // 5) for t in range(1, 301)], NOW)
    fast = nova.Profile(lib, [_event(t, "listen", NOW - k * day // 20) for k, t in enumerate(list(range(1, 41)) * 7)], NOW)
    assert big.rest_days(0.5) > small.rest_days(0.5), "знакомого больше — повтор реже"
    assert fast.rest_days(0.5) < small.rest_days(0.5), "слушает больше — цикл короче"
    assert small.rest_days(0.2) > small.rest_days(0.8), "знакомого в потоке меньше — каждый трек реже"


def test_small_pool_limits_the_familiar_share_until_it_grows():
    """Пул мал — знакомого столько, сколько он выдержит без заезженности; вырос — предел уходит."""
    lib = make_library()
    day = 86_400_000
    plays = [_event(t, "listen", NOW - k * day // 8) for k, t in enumerate(list(range(1, 11)) * 11)]  # 8 в день, 10 треков
    tiny = nova.Profile(lib, plays, NOW)
    assert tiny.feasible_known(0.5) < 0.5, "10 треков не выдержат половину потока при 8 треках в день"
    grown = nova.Profile(lib, plays + _listened(range(11, 301), NOW - 30 * day), NOW)
    assert grown.feasible_known(0.5) == 0.5
    assert tiny.min_cycle() > 0 and tiny.feasible_known(0.5) * tiny.plays_per_day * tiny.min_cycle() <= tiny.known_count + 1e-6


def test_interval_budget_adds_up_to_the_slider():
    """Темпы треков 1/Tᵢ складываются ровно в P·s — столько знакомых и прозвучит в день."""
    lib = make_library()
    day = 86_400_000
    evs = [_event(t, "listen", NOW - t * day // 8) for t in range(1, 101)] + [_event(t, "like", NOW - day) for t in (3, 9)]
    prof = nova.Profile(lib, evs, NOW)
    share = 0.5
    cycle = prof.rest_days(share)
    known = [t for t in lib.tracks if prof.newness(t) < 0.5]
    budget = sum(1.0 / (cycle * prof.mean_rate * prof.interval_factor(lib, t)) for t in known)
    assert abs(budget - prof.plays_per_day * share) / (prof.plays_per_day * share) < 0.02, (budget, prof.plays_per_day)
    assert prof.interval_factor(lib, 3) < prof.interval_factor(lib, 4), "лайкнутый приходит чаще"


def test_each_batch_keeps_the_slider_within_one_track():
    """Пачка не уходит целиком в один разряд: на каждом месте — чего не хватает."""
    lib = make_library()
    prof = nova.Profile(lib, _listened(range(1, 201), NOW - 10 * 86_400_000), NOW)
    prof.share_count, prof.share_new = 0.0, 0.0  # прошлое не влияет: смотрим только эту пачку
    known = [t for t in range(1, 201)]
    rng = random.Random(3)
    hist = rng.sample(known, 6)
    batch = nova.next_tracks(lib, prof, current=hist[-1], history=hist, n=3, now_ms=NOW, rng=rng, mode=nova.MODE_MY,
                             filters=nova.Filters(familiar=0.5))
    marks = [prof.newness(p.track_id) >= 0.5 for p in batch if p.why not in ("bridge", "spice")]
    assert any(marks), "история вся знакомая — нужно новое"
    hist2 = [t for t in range(300, 306)]
    batch2 = nova.next_tracks(lib, prof, current=hist2[-1], history=hist2, n=3, now_ms=NOW, rng=rng, mode=nova.MODE_MY,
                              filters=nova.Filters(familiar=0.5))
    assert any(prof.newness(p.track_id) < 0.5 for p in batch2), "история вся новая — нужно знакомое"


def _served_days(lib, new_share, days=25, per_day=12, seed=4):
    """Доля новых по дням, профиль обновляется по ходу, как на сервере; поток идёт днями с паузами."""
    base = _listened(range(1, 121), NOW - 86_400_000)
    rng, evs, shares = random.Random(seed), [], []
    f = nova.Filters(familiar=1.0 - new_share)
    for d in range(days):
        now, hist, start, today = NOW + d * 86_400_000, [], True, []
        while len(today) < per_day:
            prof = nova.Profile(lib, base + evs, now)
            mood = nova.Mood.from_events(lib, list(reversed(evs))[:150], now)
            for p in nova.next_tracks(lib, prof, current=hist[-1] if hist else None, history=hist[-30:], filters=f, n=3,
                                      now_ms=now, rng=rng, mode=nova.MODE_MY, start=start, mood=mood):
                today.append(prof.newness(p.track_id) >= 0.5)
                hist.append(p.track_id)
                now += 300_000
                evs.append(_event(p.track_id, "listen", now))
            start = False
        shares.append(sum(today) / len(today))
    return shares


def test_share_converges_and_does_not_drift_as_the_pool_grows():
    """Сигма-дельта: средняя доля сходится к ползунку и на поздних днях не уплывает."""
    lib = make_library()
    shares = _served_days(lib, 0.5)
    settled = shares[3:]  # первые дни отрабатывается прежняя история
    assert abs(sum(settled) / len(settled) - 0.5) < 0.04, shares
    assert abs(sum(settled[-8:]) / 8 - 0.5) < 0.06, shares


def test_new_tracks_are_candidates_in_a_small_style():
    """В небольшом стиле новинка находится, даже когда в общую случайную выборку она не попала."""
    lib = make_library()
    side = [t.id for t in lib.tracks.values() if t.style == "Side"]
    prof = nova.Profile(lib, _listened(range(1, 121), NOW - 86_400_000) + _listened(side[:6], NOW - 86_400_000), NOW)
    prof.share_count, prof.share_new = 100.0, 0.0  # нового очень не хватает
    rng = random.Random(1)
    batch = nova.next_tracks(lib, prof, current=side[5], history=side[:6], n=3, now_ms=NOW, rng=rng, mode=nova.MODE_MY,
                             filters=nova.Filters(familiar=0.5))
    assert all(prof.newness(p.track_id) >= 0.5 and lib.tracks[p.track_id].style == "Side" for p in batch)


def test_style_is_not_left_just_because_the_needed_kind_ran_out():
    """Срока и запаса больше нет: уходят только по действию слушателя, долг ползунка отдаётся позже (решение 0013)."""
    lib = make_library()
    side = [t.id for t in lib.tracks.values() if t.style == "Side"]
    main = [t.id for t in lib.tracks.values() if t.style == "Main"]
    day = 86_400_000
    prof = nova.Profile(lib, _listened(main[:60], NOW - 60 * day) + _listened(side, NOW - 40 * day), NOW)
    prof.share_count, prof.share_new = 100.0, 20.0  # нового сильно не хватает
    rng = random.Random(2)
    hist = rng.sample(side, 5)
    batch = nova.next_tracks(lib, prof, current=hist[-1], history=hist, n=3, now_ms=NOW, rng=rng, mode=nova.MODE_MY,
                             filters=nova.Filters(familiar=0.5))
    assert lib.tracks[batch[0].track_id].style == "Side", [lib.tracks[p.track_id].style for p in batch]


def _in_foreign_style(lib, skips, style="Side"):
    """Сессия идёт в чужом стиле (он основной), слушатель дослушал четыре трека, затем пропуски skips: [(доля, причина)]."""
    side = [t.id for t in lib.tracks.values() if t.style == style]
    main = [t.id for t in lib.tracks.values() if t.style == "Main"]
    now, evs, hist = NOW, [], []
    for t in side[:4]:
        now += 60_000
        evs.append(_event(t, "listen", now))
        hist.append(t)
    for t, (pct, reason) in zip(side[4:], skips):
        now += 60_000
        evs.append(_event(t, "skip", now, pct, reason))
        hist.append(t)
    base = _listened(main[20:60], NOW - 86_400_000)
    return base, evs, hist, now


def _next(lib, base, evs, hist, now, seed=3, n=3):
    mood = nova.Mood.from_events(lib, list(reversed(evs)), now)
    prof = nova.Profile(lib, base + evs, now)
    return nova.next_tracks(lib, prof, current=hist[-1], history=hist[-30:], n=n, now_ms=now, rng=random.Random(seed),
                            mode=nova.MODE_MY, mood=mood)


def test_two_skips_of_any_depth_leave_a_foreign_style_at_once():
    """Владелец 05.11.2026: «поток ушёл в другой жанр, застревает, нужно много скипать, чтоб выбраться»."""
    lib = make_library()
    for seed in range(4):
        picks = _next(lib, *_in_foreign_style(lib, [(30, None), (40, None)]), seed=seed)
        assert picks and lib.tracks[picks[0].track_id].style != "Side", [p.why for p in picks]


def test_one_skip_in_a_foreign_style_is_not_yet_a_refusal():
    lib = make_library()
    picks = _next(lib, *_in_foreign_style(lib, [(3, None)]))
    assert lib.tracks[picks[0].track_id].style == "Side", [p.why for p in picks]


def test_leaving_does_not_need_a_bridge():
    """Мост через шлюз найти не удалось: слушатель всё равно уходит, а не пропускает стиль до конца."""
    lib = make_library()
    side = np.array([t.style == "Side" for t in lib.tracks.values()])
    lib.gate.bpm[side] = 100.0  # темп не сводится ни с чем: шлюз закрыт
    for seed in range(4):
        picks = _next(lib, *_in_foreign_style(lib, [(3, None), (3, None)]), seed=seed)
        assert picks and lib.tracks[picks[0].track_id].style != "Side", [p.why for p in picks]
        assert picks[0].why == "leave", picks[0].why
        assert "Side" in picks[0].reason


def test_leaving_does_not_walk_to_a_bridge_inside_the_refused_style():
    """От этого трека моста нет, от соседнего есть. Подойти к мосту ещё одним треком того же стиля нельзя: просили уйти."""
    lib = make_library()
    base, evs, hist, now = _in_foreign_style(lib, [(3, None), (3, None)])
    lib.gate.bpm[lib.row[hist[-1]]] = 100.0
    for seed in range(4):
        picks = _next(lib, base, evs, hist, now, seed=seed)
        assert picks and lib.tracks[picks[0].track_id].style != "Side", [p.why for p in picks]


def test_home_style_is_not_left_on_two_mild_skips():
    lib = make_library()
    main = [t.id for t in lib.tracks.values() if t.style == "Main"]
    now, evs = NOW, []
    for t in main[:4]:
        now += 60_000
        evs.append(_event(t, "listen", now))
    for t in main[4:6]:
        now += 60_000
        evs.append(_event(t, "skip", now, 30))
    picks = _next(lib, _listened(main[20:60], NOW - 86_400_000), evs, main[:6], now)
    assert lib.tracks[picks[0].track_id].style == "Main", [p.why for p in picks]


def test_liked_track_skipped_later_is_not_a_refusal():
    """Лайкнул и ушёл на следующий: «сюда можно», это не отказ стилю."""
    lib = make_library()
    base, evs, hist, now = _in_foreign_style(lib, [(30, None), (30, None)])
    evs.insert(-1, _event(hist[-1], "like", now - 1000))
    picks = _next(lib, base, evs, hist, now)
    assert lib.tracks[picks[0].track_id].style == "Side", [p.why for p in picks]


def test_after_leaving_the_stream_goes_on_normally():
    """Уход — один шаг: дальше обычные правила, а не вся пачка мимо шлюза."""
    lib = make_library()
    side = np.array([t.style == "Side" for t in lib.tracks.values()])
    lib.gate.bpm[side] = 100.0
    picks = _next(lib, *_in_foreign_style(lib, [(3, None), (3, None)]), n=3)
    assert picks[0].why == "leave"
    assert all(p.why != "leave" for p in picks[1:]), [p.why for p in picks]


def test_back_button_and_own_browsing_are_not_refusals():
    lib = make_library()
    base, evs, hist, now = _in_foreign_style(lib, [(3, None), (3, None)])
    for e in evs[-2:]:
        e["left_by"] = "back"
    assert lib.tracks[_next(lib, base, evs, hist, now)[0].track_id].style == "Side"
    base, evs, hist, now = _in_foreign_style(lib, [(3, None), (3, None)])
    for e in evs[-2:]:
        e["is_organic"] = 1
    assert lib.tracks[_next(lib, base, evs, hist, now)[0].track_id].style == "Side"


def _drift(steps, boldness=0.5, directions=None):
    """steps: [(стиль, [(тип, доля, сам, причина, как ушёл), …]), …]; у каждого показа свой номер трека."""
    from arxnova import drift as D

    base = directions or {s: 0.1 for s in "abcdefghij"}
    prior = lambda main: {k: v for k, v in {**base, "Main": 0.3}.items() if k != main}  # noqa: E731
    style_of = {i: s for i, (s, _) in enumerate(steps)}
    events = {i: [(*x, 1000 * i + k) for k, x in enumerate(e)] for i, (_, e) in enumerate(steps)}
    return D.follow(list(style_of), style_of.get, lambda t: events.get(t, []), prior, "Main", boldness)


LISTEN, LIKE, EARLY = ("listen", 100, False, None, None), ("like", 100, False, None, None), ("skip", 3, False, None, None)


def test_drift_share_grows_by_steps_and_the_main_style_moves():
    """Шаги из концепции: 1 из 16, чаще, половина, основной."""
    steps = [("Main", [LISTEN])] * 3
    d0 = _drift(steps)
    assert abs(d0.explore - 1 / 16) < 0.01 and d0.main == "Main"
    one = _drift(steps + [("a", [LIKE, LISTEN])])
    two = _drift(steps + [("a", [LIKE, LISTEN])] * 2)
    assert d0.share("a") < 0.02 < 0.05 < one.share("a") < 0.25 < two.share("a"), (d0.share("a"), one.share("a"), two.share("a"))
    assert one.main == "Main"
    three = _drift(steps + [("a", [LIKE, LISTEN])] * 3)
    assert three.main == "a"


def test_completed_probe_hardly_counts_and_passive_listening_changes_nothing():
    steps = [("Main", [LISTEN])] * 3
    many = _drift(steps + [("a", [LISTEN]), ("Main", [LISTEN]), ("Main", [LISTEN])] * 6)
    assert many.main == "Main" and many.share("a") < 0.1
    assert abs(_drift([("Main", [LISTEN])] * 100).explore - 1 / 16) < 1e-6


def test_own_choice_in_another_style_moves_the_session():
    own = ("listen", 100, True, None, None)
    d = _drift([("Main", [LISTEN])] * 3 + [("a", [own])] * 3)
    assert d.main == "a"


def test_boldness_sets_the_probe_share():
    from arxnova import drift as D

    assert D.probe_share(0.0) == 0.0 and D.probe_share(0.5) == 1 / 16 and D.probe_share(1.0) > 0.15


def _with_editions(lib, a, b, bits=(16, 24)):
    rows = {t: {"bits": 16, "sample_rate": 44100} for t in lib.tracks}
    rows[a], rows[b] = {"bits": bits[0], "sample_rate": 44100}, {"bits": bits[1], "sample_rate": 96000}
    lib.tracks[a].artist = lib.tracks[b].artist = "Same"
    lib.tracks[a].title = lib.tracks[b].title = "Same Song"
    lib._merge_editions(rows)


def test_editions_are_one_song_and_the_best_one_plays():
    lib = make_library()
    _with_editions(lib, 5, 6)
    assert lib.tracks[5].recording_key == lib.tracks[6].recording_key
    assert lib.best_edition[5] == 6 == lib.best_edition[6]
    prof = nova.Profile(lib, _listened([5], NOW - 86_400_000), NOW)
    assert prof.newness(6) < 0.5, "слышал одно издание — второе не новое"
    rng = random.Random(1)
    seen = {p.track_id for _ in range(30) for p in nova.next_tracks(lib, prof, current=None, history=[], n=4, now_ms=NOW, rng=rng,
                                                                    mode=nova.MODE_MY, start=True)}
    assert 5 not in seen


def test_excerpt_from_a_mix_is_not_an_edition():
    lib = make_library()
    lib.tracks[6].duration = 100.0
    _with_editions(lib, 5, 6)
    assert lib.tracks[5].recording_key != lib.tracks[6].recording_key


def test_like_fades_after_two_skips_but_comes_back_with_a_new_like():
    lib = make_library()
    evs = [_event(7, "like", NOW - 5000), _event(7, "skip", NOW - 4000, 3), _event(7, "skip", NOW - 3000, 3)]
    assert 7 not in nova.Profile(lib, evs, NOW).liked
    assert 7 in nova.Profile(lib, evs + [_event(7, "like", NOW - 2000)], NOW).liked
    assert 7 in nova.Profile(lib, evs[:2], NOW).liked


def test_weak_tempo_is_unknown_and_long_mix_is_out_of_the_stream():
    lib = make_library()
    lib.dj_tools.add(9)
    prof = nova.Profile(lib, _listened(range(1, 41), NOW - 86_400_000), NOW)
    rng = random.Random(2)
    got = {p.track_id for _ in range(20) for p in nova.next_tracks(lib, prof, current=None, history=[], n=4, now_ms=NOW, rng=rng,
                                                                   mode=nova.MODE_MY, start=True)}
    assert 9 not in got


def test_a_like_means_the_same_in_a_library_with_many_styles():
    """Прогон на снимке: 21 направление, у каждого по 3–10 %. Лайк умножал долю в 10 раз и давал 1 из 25 вместо 1 из 5."""
    probes = [("Main", [LISTEN])] * 3
    for count in (3, 21, 60):
        styles = {f"s{i}": 1.0 / count for i in range(count)}
        one = _drift(probes + [("s0", [LIKE, LISTEN])], directions=styles)
        two = _drift(probes + [("s0", [LIKE, LISTEN])] * 2, directions=styles)
        three = _drift(probes + [("s0", [LIKE, LISTEN])] * 3, directions=styles)
        assert 0.12 < one.share("s0") < 0.25 and one.main == "Main", (count, one.share("s0"))
        assert 0.35 < two.share("s0") < 0.6 and two.main == "Main", (count, two.share("s0"))
        assert three.main == "s0", count


def test_after_a_like_other_directions_get_little_until_it_is_forgotten():
    styles = {f"s{i}": 1.0 / 21 for i in range(21)}
    probes = [("Main", [LISTEN])] * 3
    one = _drift(probes + [("s0", [LIKE, LISTEN])], directions=styles)
    assert one.directions["s0"] > 0.75
    later = _drift(probes + [("s0", [LIKE, LISTEN])] + [("Main", [LISTEN])] * 200, directions=styles)
    assert later.main == "Main" and later.directions["s0"] < 0.15 and abs(later.explore - 1 / 16) < 0.02


def test_probes_keep_the_slider_edges():
    """Прогон на снимке: при «только знакомое» 4 % нового, и всё это пробы. Направление без нужного разряда не пробуем."""
    lib = make_library()
    side = [t.id for t in lib.tracks.values() if t.style == "Side"]
    known = set(range(1, 161)) | set(side[:20])
    only_known = Listener(lib, _heard, filters=nova.Filters(familiar=1.0, boldness=1.0))
    only_known.base = _listened(sorted(known), NOW - 30 * 86_400_000)
    only_known.step(30)
    probes = [p.track_id for p in only_known.picks if p.why == "spice"]
    assert probes and all(t in known for t in probes), [t for t in probes if t not in known]
    assert {lib.tracks[t].style for t in probes} == {"Side"}, "в Far знакомого нет, туда не пробуем"
    only_new = Listener(lib, _heard, filters=nova.Filters(familiar=0.0, boldness=1.0))
    only_new.base = list(only_known.base)
    only_new.step(30)
    probes = [p.track_id for p in only_new.picks if p.why == "spice"]
    assert probes and not [t for t in probes if t in known]


def test_only_familiar_rotates_as_widely_as_the_middle_of_the_slider():
    """На краю была своя очередь из 24 лучших: на снимке 111 разных на 233 места, а при 20 % нового 193 на 201.
    На игрушечной медиатеке это не воспроизводится, тест лишь держит ширину; доказательство — bench/eval_goal/audit_stream.py."""
    lib = make_library()
    ls = Listener(lib, _heard, filters=nova.Filters(familiar=1.0, boldness=0.0))
    ls.base = _listened(range(1, 161), NOW - 30 * 86_400_000)
    ls.step(40)
    played = [p.track_id for p in ls.picks]
    assert len(played) == 120 and len(set(played)) >= 110, len(set(played))


def test_a_style_probed_a_moment_ago_waits_for_others():
    """Пробы одного стиля подряд ничего не дают: дослушанная проба почти ничего не говорит, а другие ещё не пробовали."""
    from arxnova import drift as D

    lib = make_library()
    prof = nova.Profile(lib, _listened(range(1, 41), NOW - 86_400_000), NOW)
    mood = nova.Mood.from_events(lib, [], NOW)
    side = next(t.id for t in lib.tracks.values() if t.style == "Side")
    hist = list(range(1, 11)) + [side] + list(range(11, 16))
    drift = D.Drift("Main", "Main", 1 / 16, 1 / 16, {"Side": 0.5, "Far": 0.5}, played=15, probes=[(10, "Side")])
    got = {"Side": 0, "Far": 0}
    for seed in range(300):
        p = nova._probe(lib, prof, hist[-1], hist, nova.Filters(), random.Random(seed), mood, drift, None, NOW, set(), lambda s: True)
        got[lib.tracks[p.track_id].style] += 1
    assert got["Far"] > 1.4 * got["Side"] > 0, got


def test_gate_decisions_do_not_overwrite_the_choice_journal():
    """Строка шлюза в цикле записывалась в ту же переменную, и в журнал уходило её содержимое."""
    import json
    import sqlite3

    from arxnova import db

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    db.migrate(conn)
    verdict = G.Verdict(True, None, 0.7, "")
    pick = nova.Pick(1, "почему", 0.5, "taste", prob=0.25)
    req = nova.log_request(conn, 1, None, [pick], [(1, 2, verdict)], {"v": nova.ALGO_VERSION, "picks": [{"t": 1}]})
    row = conn.execute("SELECT detail FROM nova_requests WHERE id = ?", (req,)).fetchone()
    assert json.loads(row[0]) == {"v": nova.ALGO_VERSION, "picks": [{"t": 1}]}


def test_a_listener_who_calmed_down_is_not_led_away():
    """Два отказа — просьба уйти. Но если следующий трек этого же стиля дослушан, просьбы больше нет (0011: дослушал — успокоился)."""
    steps = [("Main", [LISTEN])] * 6 + [("a", [LIKE, LISTEN])] * 3  # основной теперь a
    asked = _drift(steps + [("a", [EARLY])] * 2)
    assert asked.main == "a" and asked.leaving == "a"
    calm = _drift(steps + [("a", [EARLY])] * 2 + [("a", [LISTEN])])
    assert calm.main == "a" and calm.leaving is None


def test_events_of_an_earlier_play_do_not_count_for_a_repeat():
    """Трек уже звучал в этой сессии и снова встал в очередь. Его старое «дослушал» сбрасывало два отказа, и выход занимал 4 пропуска."""
    from arxnova import drift as D

    style_of = {t: "a" for t in range(1, 6)}
    done = lambda ts: ("listen", 100, False, None, None, ts)  # noqa: E731
    early = lambda ts: ("skip", 3, False, None, None, ts)  # noqa: E731
    events = {1: [done(1)], 2: [done(2)], 3: [early(3)], 4: [early(4)]}
    prior = lambda main: {"b": 1.0}  # noqa: E731
    d = D.follow([1, 2, 3, 4, 1], style_of.get, lambda t: events.get(t, []), prior, "Main", 0.5)
    assert d.main == "a" and d.leaving == "a", "повтор ещё не звучал, просьба уйти остаётся"
    events[1] = [done(1), early(5)]  # а этот повтор уже пропущен: действие относится ко второму показу
    d = D.follow([1, 2, 3, 1], style_of.get, lambda t: events.get(t, []), prior, "Main", 0.5)
    assert d.refusals == 2 and d.leaving == "a"
    events = {1: [done(1), done(2)]}  # подряд два раза: каждый показ получает своё
    assert [len(x) for x in D._by_play([1, 1], lambda t: events.get(t, []))] == [1, 1]
    # прошлый показ вообще не попал в хвост истории: плеер шлёт только последние треки
    events = {3: [early(3)], 4: [early(4)], 2: [done(1)]}
    d = D.follow([3, 4, 2], style_of.get, lambda t: events.get(t, []), prior, "Main", 0.5)
    assert d.refusals == 2 and d.leaving == "a"


def test_a_like_after_the_track_ended_still_counts_for_it():
    from arxnova import drift as D

    style_of = {1: "Main", 2: "Main", 3: "Main", 4: "b", 5: "Main"}
    events = {4: [("listen", 100, False, None, None, 4), ("like", 100, False, None, None, 5)],
              5: [("listen", 100, False, None, None, 6)]}
    prior = lambda main: {"b": 0.04, **{f"s{i}": 0.04 for i in range(24)}}  # noqa: E731
    d = D.follow([1, 2, 3, 4, 5], style_of.get, lambda t: events.get(t, []), prior, "Main", 0.5)
    assert d.directions["b"] > 0.7


FOUR_STYLES = (("Main", 400), ("Side", 60), ("Far", 60), ("Mid", 60))


def _session(lib, steps, seed):
    """Сессия по шагам: (стиль, что сделали). Возвращает следующие треки."""
    used, events, hist = {}, [], []
    now = NOW
    for style, actions in steps:
        pool = [t.id for t in lib.tracks.values() if t.style == style]
        tid = pool[40 + used.get(style, 0)]  # первые 40 Main уже слушаны в прошлом
        used[style] = used.get(style, 0) + 1
        hist.append(tid)
        for et, pct in actions:
            now += 60_000
            events.append(_event(tid, et, now, pct))
    mood = nova.Mood.from_events(lib, list(reversed(events)), now)
    prof = nova.Profile(lib, _listened(range(1, 41), NOW - 86_400_000) + events, now)
    return nova.next_tracks(lib, prof, current=hist[-1], history=hist[-30:], n=3, now_ms=now, rng=random.Random(seed),
                            mode=nova.MODE_MY, mood=mood), prof


HEARD, LIKED, SKIPPED = [("listen", 100)], [("like", 100), ("listen", 100)], [("skip", 3)]
DRIFTED = [("Main", HEARD)] * 6 + [("Side", LIKED)] * 3 + [("Side", HEARD)] * 6


def test_two_refusals_after_a_drift_lead_home_every_time():
    """Владелец 05.11.2026: как вернуться к основному стилю после дрейфа. Случайность уводила в малоизвестный стиль в 7 % случаев."""
    lib = make_library(FOUR_STYLES)
    for seed in range(1, 25):
        picks, _ = _session(lib, DRIFTED + [("Side", SKIPPED)] * 2, seed)
        assert lib.tracks[picks[0].track_id].style == "Main", seed


def test_an_accepted_probe_beats_home_when_refusing_a_drifted_style():
    lib = make_library(FOUR_STYLES)
    for seed in range(1, 13):
        picks, _ = _session(lib, DRIFTED + [("Far", LIKED)] + [("Side", SKIPPED)] * 2, seed)
        assert lib.tracks[picks[0].track_id].style == "Far", seed


def test_a_style_refused_a_moment_ago_is_not_where_refusal_leads():
    """Ушёл из A, пропустил B — и снова A: на снимке путь Folk, Modern Classical, Folk, 6 пропусков до дома."""
    lib = make_library(FOUR_STYLES)
    for seed in range(1, 25):
        picks, _ = _session(lib, [("Main", HEARD)] * 8 + [("Main", SKIPPED)] * 2 + [("Side", SKIPPED)] * 2, seed)
        assert lib.tracks[picks[0].track_id].style not in ("Main", "Side"), seed


def test_a_refused_style_can_return_later_and_after_a_change_of_mind():
    from arxnova import drift as D

    steps = [("Main", [LISTEN])] * 3 + [("a", [LIKE, LISTEN])] * 3 + [("a", [EARLY])] * 2 + [("b", [LISTEN])]
    fresh = _drift(steps)
    assert fresh.main == "b" and fresh.refused_lately("a")
    assert not _drift(steps + [("b", [LISTEN])] * D.REFUSED_FRESH).refused_lately("a")
    assert not _drift(steps + [("a", [LIKE, LISTEN])]).refused_lately("a")


def test_half_of_the_probes_after_a_drift_go_home_for_a_one_style_listener():
    lib = make_library(FOUR_STYLES)
    solo = nova.Profile(lib, _listened(range(1, 41), NOW - 86_400_000), NOW)
    assert solo.home_style(lib) == "Main"
    prior = lib.direction_prior(solo, "Side")
    assert abs(prior["Main"] - 0.5) < 1e-6 and abs(sum(prior.values()) - 1.0) < 1e-6
    side = [t.id for t in lib.tracks.values() if t.style == "Side"][:30]
    far = [t.id for t in lib.tracks.values() if t.style == "Far"][:30]
    wide = nova.Profile(lib, _listened(list(range(1, 41)) + side + far, NOW - 86_400_000), NOW)
    assert wide.home_style(lib) == "Main"
    prior = lib.direction_prior(wide, "Side")
    assert 0.25 < prior["Main"] < 0.5, prior
    assert "Main" not in lib.direction_prior(solo, "Main")


def test_two_likes_on_home_probes_bring_the_stream_back():
    steps = [("Main", [LISTEN])] * 3 + [("a", [LIKE, LISTEN])] * 3 + [("a", [LISTEN])] * 3
    assert _drift(steps).main == "a"
    one = _drift(steps + [("Main", [LIKE, LISTEN])])
    assert one.main == "a" and one.share("Main") > 0.15
    assert _drift(steps + [("Main", [LIKE, LISTEN])] * 2).main == "Main"
    new = _drift(steps + [("b", [LIKE, LISTEN])] * 2)
    assert new.main == "a", "в новое по-прежнему три"


def test_skipping_half_of_a_drifted_style_is_a_request_to_leave_too():
    """Через один: подряд двух отказов нет, неусидчивость гаснет, и такой слушатель считался довольным."""
    moved = [("Main", [LISTEN])] * 3 + [("a", [LIKE, LISTEN])] * 3
    assert _drift(moved + [("a", [EARLY]), ("a", [LISTEN]), ("a", [EARLY]), ("a", [LISTEN])]).leaving is None
    assert _drift(moved + [("a", [EARLY]), ("a", [LISTEN]), ("a", [EARLY]), ("a", [LISTEN]), ("a", [EARLY])]).leaving == "a"
    rare = moved + [("a", [EARLY])] + [("a", [LISTEN])] * 3 + [("a", [EARLY])] + [("a", [LISTEN])] * 3 + [("a", [EARLY])]
    assert _drift(rare).leaving is None, "один из четырёх — не просьба"
    home = [("Main", [LISTEN])] * 3 + [("Main", [EARLY]), ("Main", [LISTEN])] * 3
    assert _drift(home).leaving is None, "дома порог выше"


def test_leaving_home_itself_still_varies():
    """Из дома известного места нет: каждый раз в одно и то же было бы хуже."""
    lib = make_library(FOUR_STYLES)
    got = set()
    for seed in range(1, 25):
        picks, _ = _session(lib, [("Main", HEARD)] * 8 + [("Main", SKIPPED)] * 2, seed)
        got.add(lib.tracks[picks[0].track_id].style)
    assert "Main" not in got and len(got) >= 2, got
