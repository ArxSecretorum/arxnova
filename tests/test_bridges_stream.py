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
    fac = {f: _unit(mix + 0.05 * rng.normal(size=mix.shape)).astype(np.float32) for f in G.FACETS}
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
    lib.gate = G.Gate(mix, styles, scores, fac, feats, COLS)
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


def test_stream_leaves_the_main_style_through_bridges_and_comes_back():
    lib = make_library()
    picks = _stream(lib)
    styles = [lib.tracks[p.track_id].style for p in picks]
    bridges = [p for p in picks if p.why == "bridge"]
    assert len(bridges) >= 3, f"мостов {len(bridges)}: {[(k, len(list(g))) for k, g in itertools.groupby(styles)]}"
    assert {"Side", "Far"} & set(styles)
    assert styles[-40:].count("Main") > 0, "из чужих стилей поток возвращается домой"
    assert all(p.why != "restyle" for p in picks), "смены стиля — мостами, а не от безысходности"


def test_stream_stays_after_a_bridge_and_does_not_flicker():
    lib = make_library()
    styles = G.without_islands([lib.tracks[p.track_id].style for p in _stream(lib)])  # подмес стиль не меняет
    runs = [(k, len(list(g))) for k, g in itertools.groupby(styles)]
    # первый и последний участки могут быть обрезаны началом и концом прогона
    # стиль без знакомых треков покидается раньше обычных 8, когда знакомого не хватает, — но не сразу
    for style, length in runs[1:-1]:
        assert length >= (12 if style == "Main" else G.M_STICK), runs


def test_bridge_is_a_real_gate_pass_from_the_previous_track():
    """Мост проверен от того трека, что перед ним в очереди, а не от опоры запроса."""
    lib = make_library()
    picks = _stream(lib)
    pct = lib.gate.pct(0.5)
    checked = 0
    for prev, p in zip(picks, picks[1:]):
        if p.why == "bridge":
            assert lib.gate.check(lib.row[prev.track_id], lib.row[p.track_id], pct=pct).ok
            checked += 1
    assert checked


def test_no_bridge_on_the_first_tracks_of_a_session():
    lib = make_library()
    picks = _stream(lib, total=8)
    assert all(p.why != "bridge" for p in picks[:8])


def _event(tid, et, ts, pct=100, reason=None):
    return {"track_id": tid, "event_type": et, "ts": ts, "is_organic": 0, "played_ratio_pct": pct,
            "skip_reason": reason, "prev_track_id": None, "device_id": 1}


class Listener:
    """Прогон потока с откликом: что слушатель делает с каждым треком, сервер видит сразу."""

    def __init__(self, lib, react, seed=3):
        self.lib, self.react = lib, react
        self.base = _listened(range(1, 41), NOW - 86_400_000)
        self.rng = random.Random(seed)
        self.events, self.hist, self.picks = [], [], []
        self.now = NOW

    def step(self, batches=1):
        for _ in range(batches):
            mood = nova.Mood.from_events(self.lib, list(reversed(self.events)), self.now)
            prof = nova.Profile(self.lib, self.base + self.events, self.now)  # как на сервере: профиль идёт за событиями
            batch = nova.next_tracks(self.lib, prof, current=self.hist[-1] if self.hist else None,
                                     history=self.hist[-30:], n=3, now_ms=self.now, rng=self.rng, mode=nova.MODE_MY,
                                     start=not self.hist, mood=mood)
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


def test_two_early_skips_lead_out_of_the_style_right_away():
    """Спокойно из основного стиля раньше 12 треков не уйти; два ранних пропуска — уходим сейчас."""
    lib = make_library()
    skipping = {"on": False}
    ls = Listener(lib, lambda t, p: ("skip", 3) if skipping["on"] else ("listen", 100))
    ls.step(2)  # 6 треков, срок ещё не подошёл
    assert set(ls.styles()) <= {"Main"} or ls.picks[0].why != "bridge"
    assert all(p.why != "bridge" for p in ls.picks)
    skipping["on"] = True
    ls.step(1)  # эта пачка пропущена целиком
    skipping["on"] = False
    before = len(ls.picks)
    ls.step(2)
    assert any(p.why == "bridge" for p in ls.picks[before:]), [p.why for p in ls.picks]


def test_rejected_style_is_avoided_for_the_session():
    """Пропускает всё в Side — поток уходит оттуда и больше туда не ведёт, а в Far ведёт."""
    lib = make_library()
    ls = Listener(lib, lambda t, p: ("skip", 3) if t.style == "Side" else ("listen", 100))
    ls.step(70)
    styles = G.without_islands(ls.styles())
    runs = [(k, len(list(g))) for k, g in itertools.groupby(styles)]
    side = [n for k, n in runs if k == "Side"]
    # настрой тает за часы (на проде дальше держит долгая память профиля), но каждый заход короткий
    assert len(side) <= 4 and all(n <= 6 for n in side), runs
    assert "Far" in styles, runs


def test_like_on_a_foreign_track_keeps_the_stream_there():
    """Лайк в чужом стиле: домой не тянет, держимся дольше обычных 8 треков."""
    lib = make_library()
    ls = Listener(lib, lambda t, p: ("like", 100) if t.style != "Main" and p.why == "bridge" else ("listen", 100))
    ls.step(60)
    runs = [(k, len(list(g))) for k, g in itertools.groupby(G.without_islands(ls.styles()))]
    away = [n for k, n in runs[1:-1] if k != "Main"]
    assert away and max(away) >= 8 + nova.LIKE_STAY - 3, runs


def test_spice_is_a_single_track_close_in_feel_and_stream_goes_on():
    lib = make_library()
    picks = _stream(lib, total=240, seed=5)
    spices = [i for i, p in enumerate(picks) if p.why == "spice"]
    assert 5 <= len(spices) <= 40, len(spices)
    g = lib.gate
    for i in spices:
        before, x, after = (lib.tracks[picks[j].track_id] for j in (i - 1, i, i + 1))
        assert before.style == after.style != x.style, "после подмеса поток идёт как шёл"
        ra, rx = lib.row[before.id], lib.row[x.id]
        assert abs(g.energy[ra] - g.energy[rx]) <= G.FEEL_ENERGY_TOL and g.vocal[ra] == g.vocal[rx]
    assert all(b - a >= nova.SPICE_GAP for a, b in zip(spices, spices[1:]))


def test_skipped_spice_pushes_the_next_one_away():
    lib = make_library()
    gaps = {}
    for name, react in (("heard", lambda t, p: ("listen", 100)),
                        ("skipped", lambda t, p: ("skip", 3) if p.why == "spice" else ("listen", 100))):
        ls = Listener(lib, react, seed=7)
        ls.step(100)
        idx = [i for i, p in enumerate(ls.picks) if p.why == "spice"]
        gaps[name] = np.mean(np.diff(idx)) if len(idx) > 1 else 1e9
    assert gaps["skipped"] > gaps["heard"], gaps


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
    assert usual > 30, usual
    assert 0 < rare < usual / 3, (rare, usual)
    assert _offers_from(lib, prof, "Main", "Side") == 0


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


def test_stream_leaves_a_style_where_the_needed_kind_ran_out():
    """Весь Side уже слышан, а нового не хватает: поток не сидит там положенные 8 треков и туда не ведёт."""
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
    assert any(lib.tracks[p.track_id].style != "Side" for p in batch), [lib.tracks[p.track_id].style for p in batch]
