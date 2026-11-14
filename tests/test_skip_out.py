"""Уход из стиля по пропускам и начало сессии не только дома (решение 0018).

Владелец 14.11.2026: «очень странно ведёт себя поток… многие жанры вообще ещё не слышал, зато он мне постоянно один ДнБ пихает»,
«перелистывания по 2-3-4 трека ДнБ подряд не дает результатов». Журнал Pi за 7–13.11: 149 из 229 показов ДнБ, все 13 сессий
начались с ДнБ, после двух-трёх пропусков подряд поток выбирал соседний ДнБ — запрещены были только треки рядом с пропущенными."""

from __future__ import annotations

import random
from collections import Counter

from arxnova import listener as LM, nova
from tests.test_bridges_stream import NOW, _event, _listened, make_library

DAY = 86_400_000


def _main(lib):
    return [t for t in lib.tracks if lib.tracks[t].style == "Main"]


def _home_history(lib, days=10):
    """Десять дней одного Main: дом — Main, как ДнБ у владельца."""
    main = _main(lib)
    evs = []
    for d in range(days):
        for k in range(12):
            evs.append(_event(main[(d * 12 + k) % len(main)], "listen", NOW - (days - d) * DAY + k * 300_000))
    return evs


def _ask(lib, hist, evs, now, seed, start=False):
    tr = {}
    mood = nova.Mood.from_events(lib, list(reversed(evs)), now)
    prof = nova.Profile(lib, _listened(range(1, 41), NOW - 30 * DAY) + evs, now)
    picks = nova.next_tracks(lib, prof, current=None if start else hist[-1], history=[] if start else hist, n=3, now_ms=now,
                             rng=random.Random(seed), mode=nova.MODE_MY, mood=mood, model="listener", trace=tr, start=start)
    return picks, tr, prof


def _session_with_skips(lib, seed, listened=6, skipped=2):
    """Сессия в Main: сначала принятые, потом пропуски подряд."""
    main = _main(lib)
    rng = random.Random(seed)
    tracks = rng.sample(main[200:], listened + skipped)
    now, evs = NOW, _home_history(lib)
    for k, t in enumerate(tracks):
        now += 40_000 if k >= listened else 300_000
        evs.append(_event(t, "skip" if k >= listened else "listen", now, 15 if k >= listened else 100))
    return tracks, evs, now


def test_two_skips_in_a_row_in_one_style_leave_that_style():
    """Два пропуска подряд в стиле, где стоит поток, — «не этот стиль сейчас»: следующий трек из другого стиля."""
    lib = make_library()
    out = 0
    for seed in range(8):
        hist, evs, now = _session_with_skips(lib, seed)
        picks, tr, _ = _ask(lib, hist, evs, now, seed)
        assert picks
        out += lib.tracks[picks[0].track_id].style != "Main"
    assert out == 8, out


def test_tracks_queued_before_the_skips_do_not_pull_the_stream_back():
    """Как в плеере: запрос идёт от последнего трека очереди, а там ещё два Main, выбранные до пропусков и не слышанные. 13.11.2026
    после двух пропусков ДнБ поток выбирал от такого трека «пока заходит» — снова ДнБ."""
    lib = make_library()
    out = 0
    for seed in range(8):
        hist, evs, now = _session_with_skips(lib, seed)
        stale = random.Random(seed + 50).sample([t for t in _main(lib)[200:] if t not in hist], 2)
        picks, tr, _ = _ask(lib, hist + stale, evs, now + 1000, seed)
        out += lib.tracks[picks[0].track_id].style != "Main"
        assert tr.get("leave", {}).get("now"), tr.get("leave")
    assert out == 8, out


def test_a_listen_after_leaving_ends_it():
    """Приняли трек — уход кончился: дальше решает обычный поток (держаться нового места или вернуться, если приняли Main)."""
    lib = make_library()
    hist, evs, now = _session_with_skips(lib, 0)
    t = next(t for t in _main(lib)[200:] if t not in hist)
    _, before, _ = _ask(lib, hist + [t], evs, now + 1000, 0)
    assert before.get("leave", {}).get("now"), before.get("leave")  # пока не принят — уходим
    _, tr, _ = _ask(lib, hist + [t], evs + [_event(t, "listen", now + 300_000)], now + 300_000, 0)
    assert not tr.get("leave", {}).get("now"), tr.get("leave")


def test_spares_for_batch_tracks_after_one_skip_stay_in_the_style():
    """Запасной ход трека пачки — на момент, когда его пропустят: играющий и треки перед ним к тому времени прослушаны. Один прошлый
    пропуск плюс этот — не серия: запасной остаётся в стиле, как после одного пропуска (независимая проверка 14.11.2026: было 20 из 20 вон)."""
    lib = make_library()
    stay = total = 0
    for seed in range(8):
        hist, evs, now = _session_with_skips(lib, seed, skipped=1)
        playing = random.Random(seed + 100).choice([t for t in _main(lib)[200:] if t not in hist])
        picks, tr, _ = _ask(lib, hist + [playing], evs, now + 1000, seed)
        for p in picks:
            alt = (tr.get("on_skip") or {}).get(p.track_id)
            if alt is not None and lib.tracks[p.track_id].style == "Main":
                total += 1
                stay += lib.tracks[alt.track_id].style == "Main"
    assert total >= 8 and stay >= 0.7 * total, (stay, total)


def test_while_the_spare_of_another_style_plays_the_stream_follows_it():
    """Запасной ход уже увёл (играет трек другого стиля, ещё без событий): поток идёт от него, а не ищет от пропусков по всей медиатеке,
    и плееру больше не «уводим» — перескакивать нечего (независимая проверка 14.11.2026)."""
    lib = make_library()
    for seed in range(4):
        hist, evs, now = _session_with_skips(lib, seed)
        spare = random.Random(seed).choice([t for t in lib.tracks if lib.tracks[t].style == "Side"])
        picks, tr, _ = _ask(lib, hist + [spare], evs, now + 1000, seed)
        assert not tr.get("leave", {}).get("now"), tr.get("leave")
        assert all(lib.tracks[p.track_id].style != "Main" for p in picks), [lib.tracks[p.track_id].style for p in picks]


def test_leaving_with_device_only_tracks_still_fills_the_batch():
    """«Только треки устройства»: кандидаты ухода отбираются из годных запросу, а не 60 ближайших, из которых годных единицы
    (независимая проверка 14.11.2026: пачки по 1–2 трека и уход в случайный стиль)."""
    lib = make_library()
    for seed in range(4):
        hist, evs, now = _session_with_skips(lib, seed)
        rng = random.Random(seed)
        # других стилей на устройстве хватает на цель (не меньше ROUTE_MIN в стиле), но это малая часть ближайших к последнему принятому
        allowed = set(hist) | set(rng.sample(_main(lib), 120)) | set(rng.sample([t for t in lib.tracks if lib.tracks[t].style != "Main"], 44))
        tr = {}
        mood = nova.Mood.from_events(lib, list(reversed(evs)), now)
        prof = nova.Profile(lib, _listened(range(1, 41), NOW - 30 * DAY) + evs, now)
        picks = nova.next_tracks(lib, prof, current=hist[-1], history=hist, n=3, now_ms=now, rng=random.Random(seed), mode=nova.MODE_MY,
                                 mood=mood, model="listener", trace=tr, allowed=allowed)
        assert len(picks) == 3 and all(p.track_id in allowed for p in picks), [p.track_id for p in picks]
        assert lib.tracks[picks[0].track_id].style != "Main", [lib.tracks[p.track_id].style for p in picks]


def test_one_skip_is_not_yet_leaving():
    """Один пропуск — шаг в сторону внутри места, как раньше: в Main остаётся большинство."""
    lib = make_library()
    stay = 0
    for seed in range(8):
        hist, evs, now = _session_with_skips(lib, seed, skipped=1)
        picks, _, _ = _ask(lib, hist, evs, now, seed)
        stay += lib.tracks[picks[0].track_id].style == "Main"
    assert stay >= 6, stay


def test_after_one_skip_the_spare_for_a_second_skip_already_leaves_the_style():
    """Запасной ход играющего трека считается для состояния «его тоже пропустили»: после одного пропуска это второй подряд —
    запасной трек из другого стиля, и плеер уводит сразу, без нового запроса."""
    lib = make_library()
    out = total = 0
    for seed in range(8):
        hist, evs, now = _session_with_skips(lib, seed, skipped=1)
        playing = random.Random(seed + 100).choice([t for t in _main(lib)[200:] if t not in hist])
        _, tr, _ = _ask(lib, hist + [playing], evs, now + 1000, seed)
        alt = (tr.get("on_skip") or {}).get(playing)
        if alt is not None:
            total += 1
            out += lib.tracks[alt.track_id].style != "Main"
    assert total >= 6 and out == total, (out, total)


def test_session_start_is_not_always_home():
    """Начало сессии — не всегда дом: иначе стили, до которых маршрут не дошёл за вечер, не звучат никогда."""
    lib = make_library()
    evs = _home_history(lib)
    styles = Counter()
    runs = 120
    for seed in range(runs):
        picks, _, prof = _ask(lib, [], evs, NOW + seed * DAY // runs, seed, start=True)
        styles[lib.tracks[picks[0].track_id].style] += 1
    back = nova._home_back(prof.style_pos, prof.home_style(lib))
    assert LM.ROUTE_HOME_MIN < back <= LM.ROUTE_HOME_MAX  # у этого слушателя доля дома своя, а не край: проверка различит ошибку
    assert abs(styles["Main"] / runs - back) < 0.12, (styles, back)  # дома — с вероятностью возврата домой со стоянки
    assert styles.most_common(1)[0][0] == "Main", styles  # дом остаётся самым частым началом, но не единственным


def test_stand_in_a_small_style_is_what_it_can_play_in_a_row():
    """Проба (решение 0019): в мелком стиле стоянка — сколько его треков можно поставить подряд, а не 12, которых там нет."""
    plays = [LM.Play(i, 1) for i in range(5)]
    assert LM.route_origin(["S"] * 5, plays, need_of=lambda s: 5) == (0, 4, 4)
    assert LM.route_origin(["S"] * 4, plays[:4], need_of=lambda s: 5) is None


def test_stand_need_follows_artists_and_capacity():
    lib = make_library((("Main", 400), ("Side", 60), ("Far", 60)))
    assert lib.stand_need("Main") == LM.DWELL_START
    one = make_library((("Main", 400), ("Solo", 20)))
    for t, info in one.tracks.items():  # все треки Solo — один исполнитель: подряд больше одного не поставить
        if info.style == "Solo":
            one.tracks[t] = info.__class__(**{**info.__dict__, "artist": "Solo", "artists": frozenset({"solo"})})
    one.__dict__.pop("_stand_need", None); one.__dict__.pop("_route_capacity", None)
    assert one.stand_need("Solo") == 1


def _probe_lib():
    """Main (дом), Side — знакомый, Far — незнакомый, Tiny — четыре трека одного альбома (подряд не поставить и двух)."""
    lib = make_library((("Main", 400), ("Side", 60), ("Far", 60), ("Tiny", 4)))
    for t, info in lib.tracks.items():
        if info.style == "Tiny":
            lib.tracks[t] = info.__class__(**{**info.__dict__, "album": "Tiny One", "album_key": "tiny one"})
    for k in ("_stand_need", "_route_capacity"):
        lib.__dict__.pop(k, None)
    side = [t for t in lib.tracks if lib.tracks[t].style == "Side"][:8]
    evs = _home_history(lib) + [_event(t, "listen", NOW - 2 * DAY + k * 300_000) for k, t in enumerate(side)]
    return lib, evs


def test_probe_stand_is_short_in_an_unknown_style_and_normal_in_known_home_and_liked():
    lib, evs = _probe_lib()
    mood = nova.Mood.from_events(lib, list(reversed(evs)), NOW)
    prof = nova.Profile(lib, _listened(range(1, 41), NOW - 30 * DAY) + evs, NOW)
    stand = nova._Stand(lib, prof, mood, NOW - 1000)
    assert stand.probe("Far") and stand.need("Far") == LM.PROBE_STAND
    assert not stand.probe("Side") and stand.need("Side") == LM.DWELL_START  # знакомый — обычная стоянка
    assert not stand.probe("Main") and stand.need("Main") == LM.DWELL_START  # дом — никогда не проба
    assert not stand.target("Tiny") and stand.need("Tiny") == LM.DWELL_START  # не цель — и стоянкой не бывает
    far = next(t for t in lib.tracks if lib.tracks[t].style == "Far")
    liked = evs + [_event(far, "like", NOW)]
    stand2 = nova._Stand(lib, nova.Profile(lib, _listened(range(1, 41), NOW - 30 * DAY) + liked, NOW),
                         nova.Mood.from_events(lib, list(reversed(liked)), NOW + 1), NOW - 1000)
    assert not stand2.probe("Far") and stand2.need("Far") == LM.DWELL_START  # лайк в пробе — остаться


def test_known_count_does_not_shrink_with_plays_of_this_session():
    """Сервер отдаёт профиль с отставанием на пачку: «знаком ли стиль» — по счёту профиля на начало сессии, а не «всё минус показы сессии»
    (независимая проверка 14.11.2026: знакомый стиль становился пробой посреди стоянки)."""
    lib, evs = _probe_lib()
    side = [t for t in lib.tracks if lib.tracks[t].style == "Side"][20:23]
    session = [_event(t, "skip", NOW + k * 40_000, 30) for k, t in enumerate(side)]  # пропуски посередине: в счёте стиля с весом 0
    prof = nova.Profile(lib, _listened(range(1, 41), NOW - 30 * DAY) + evs, NOW)  # профиль без этих событий — как с отставанием
    mood = nova.Mood.from_events(lib, list(reversed(evs + session)), NOW + 200_000)
    assert not nova._Stand(lib, prof, mood, NOW - 1000).probe("Side")


def test_a_single_track_of_a_style_that_cannot_be_a_target_is_not_a_stand():
    """Переходный трек стиля, где подряд не поставить и двух, — не стоянка: маршрут не перехватывается (проверка 14.11.2026: Black Metal
    ×1 отправлял поток домой)."""
    lib, evs = _probe_lib()
    stand = nova._Stand(lib, nova.Profile(lib, evs, NOW), nova.Mood.from_events(lib, list(reversed(evs)), NOW), NOW - 1000)
    styles = ["Main"] * 12 + ["Far", "Tiny"]
    plays = [LM.Play(i, 1) for i in range(14)]
    assert LM.route_origin(styles, plays, "Main", need_of=stand) == (0, 11, 11)  # стоянка — по-прежнему дома, маршрут не сбит


def test_after_a_probe_the_next_target_is_not_home():
    """После пробы — сразу к следующей цели: ни «домой» по доле дома, ни дом выборкой Томпсона (у дома история большая и хорошая)."""
    from tests.test_route import _stub
    lib, prof = _stub()
    got = Counter(nova._route_to(lib, prof, (o, o + 1), {"Untried"}, {}, "Untried", home_back=False) for o in range(300))
    assert got["Home"] == 0 and sum(got.values()) == 300, got
    with_home = Counter(nova._route_to(lib, prof, (o, o + 1), {"Untried"}, {}, "Untried") for o in range(300))
    assert with_home["Home"] > 0, with_home  # контроль: без пробы дом достижим


def test_stand_at_home_is_as_long_as_elsewhere():
    """Дома стоянка той же длины, что в других стилях: двойная держала ДнБ, пока короткая сессия не кончится."""
    assert LM.route_origin(["H"] * LM.DWELL_START, [LM.Play(i, 1) for i in range(LM.DWELL_START)], home="H") == \
        (0, LM.DWELL_START - 1, LM.DWELL_START - 1)
