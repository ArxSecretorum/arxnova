"""Авто-режим «Своего потока» (решение 0020): дом — область дослушанного, кольца, примесь и широта, авто и ручной.

Владелец 16.11.2026: «ты меня решил задушить драм н бэйсом», «мне нужны умные пиздатые алгоритмы для авто-режима!!!! а также шкалы,
как мы оговаривали раньше!». Про дом: «Он должен быть разнообразным, чтоб не надоедал.... Но построенным на дослушивании»."""

from __future__ import annotations

import random

from arxnova import auto as A, listener as LM, nova
from tests.test_bridges_stream import NOW, _event, _listened, make_library

DAY = 86_400_000


def _main(lib):
    return [t for t in lib.tracks if lib.tracks[t].style == "Main"]


def _history(lib, n=60, style="Main"):
    tracks = [t for t in lib.tracks if lib.tracks[t].style == style][:n]
    return [_event(t, "listen", NOW - DAY + k * 300_000) for k, t in enumerate(tracks)]


def _home(lib, evs=None):
    return A.home_of(lib, evs if evs is not None else _history(lib), NOW)


def _play(lib, tid, outcome):
    return LM.Play(lib.row[tid], outcome)


def test_no_home_without_enough_listening():
    """Принятого меньше HOME_MIN_ROWS — дома нет, авто молчит, поток прежний (маршрут, стоянки)."""
    lib = make_library()
    assert A.home_of(lib, _history(lib, n=5), NOW) is None
    prof = nova.Profile(lib, _history(lib, n=5), NOW)
    hist = _main(lib)[100:104]
    tr = {}
    nova.next_tracks(lib, prof, current=hist[-1], history=hist, n=3, now_ms=NOW, rng=random.Random(1), mode=nova.MODE_MY,
                     model="listener", trace=tr, auto={"mix": None, "breadth": None})
    assert "auto" not in tr


def test_home_is_built_from_listening_and_is_wide():
    """Дом — где слушали: почти весь Main (в том числе неслышанные соседи), Far — в основном вне дома."""
    lib = make_library()
    home = _home(lib)
    main = _main(lib)
    far = [t for t in lib.tracks if lib.tracks[t].style == "Far"]
    in_home = sum(home.zone[lib.row[t]] == 0 for t in main) / len(main)
    far_home = sum(home.zone[lib.row[t]] == 0 for t in far) / len(far)
    assert in_home > 0.5, in_home  # слушали 60 из 400 — дом с соседями вдвое шире слушанного
    assert far_home < in_home, (far_home, in_home)
    assert set(home.zone.tolist()) - {0} <= set(range(1, A.RINGS + 1))


def test_home_moves_with_what_is_accepted():
    """Дом движется сам: дослушанное в Far за последний месяц делает Far домом (уголок), без настройки."""
    lib = make_library()
    far = [t for t in lib.tracks if lib.tracks[t].style == "Far"]
    before = _home(lib)
    after = _home(lib, _history(lib) + [_event(t, "listen", NOW - 3_600_000 + k * 1000) for k, t in enumerate(far[:30])])
    share = lambda h: sum(h.zone[lib.row[t]] == 0 for t in far) / len(far)  # noqa: E731
    assert share(after) > share(before), (share(before), share(after))


def _corner_tracks(lib, home, c, k=8):
    return [t for t in _main(lib) if home.zone[lib.row[t]] == 0 and home.corner[lib.row[t]] == c][:k]


def test_two_refusals_in_a_corner_go_to_another_far_corner():
    lib = make_library()
    home = _home(lib)
    c = int(home.corner[lib.row[_main(lib)[0]]])
    tr = _corner_tracks(lib, home, c, 6)
    state = [_play(lib, t, 1) for t in tr[:4]] + [_play(lib, t, -1) for t in tr[4:6]]
    for seed in range(6):
        plan = A.decide(home, state, 1, random.Random(seed))
        z, c2 = plan.slots[0]
        assert z == 0 and c2 != c, plan.slots


def test_four_home_refusals_leave_home_and_tell_the_player():
    lib = make_library()
    home = _home(lib)
    homes = [t for t in _main(lib) if home.zone[lib.row[t]] == 0]
    state = [_play(lib, t, 1) for t in homes[:4]] + [_play(lib, t, -1) for t in homes[10:14]]
    plan = A.decide(home, state, 3, random.Random(1))
    assert plan.leaving and all(z > 0 for z, _ in plan.slots), plan.slots


def test_two_refusals_outside_go_back_home():
    lib = make_library()
    home = _home(lib)
    homes = [t for t in _main(lib) if home.zone[lib.row[t]] == 0]
    outs = [t for t in lib.tracks if home.zone[lib.row[t]] > 0]
    state = [_play(lib, t, 1) for t in homes[:4]] + [_play(lib, t, -1) for t in outs[:2]]
    plan = A.decide(home, state, 2, random.Random(1))
    assert all(z == 0 for z, _ in plan.slots), plan.slots


def test_a_style_refused_twice_is_closed_until_accepted():
    lib = make_library()
    main = _main(lib)
    style_of = lambda r: lib.tracks[int(lib.ids[r])].style  # noqa: E731
    state = [_play(lib, main[0], 1), _play(lib, main[1], -1), _play(lib, main[2], -1)]
    assert A.closed_styles(state, style_of) == {"Main"}
    assert A.closed_styles(state + [_play(lib, main[3], 1)], style_of) == frozenset()


def test_manual_mix_holds_the_share():
    """Ручной: примесь 0 — только дом, 1 — вне дома, как только позволяет вылазка."""
    lib = make_library()
    home = _home(lib)
    homes = [t for t in _main(lib) if home.zone[lib.row[t]] == 0]
    state = [_play(lib, t, 1) for t in homes[:3]]
    for seed in range(5):
        assert all(z == 0 for z, _ in A.decide(home, state, 3, random.Random(seed), mix=0.0).slots)
        assert all(z > 0 for z, _ in A.decide(home, state, 3, random.Random(seed), mix=1.0).slots)


def test_manual_breadth_limits_how_far():
    lib = make_library()
    home = _home(lib)
    homes = [t for t in _main(lib) if home.zone[lib.row[t]] == 0]
    state = [_play(lib, t, 1) for t in homes[:3]]
    for seed in range(8):
        plan = A.decide(home, state, 3, random.Random(seed), mix=1.0, breadth=1)
        assert all(z == 1 for z, _ in plan.slots) and plan.breadth == 1, plan.slots


def test_narrow_breadth_widens_itself_when_too_few_tracks():
    """Узкая широта и большая примесь: в кольце 1 годных мало — поток сам расширяется и говорит об этом."""
    lib = make_library()
    home = _home(lib)
    homes = [t for t in _main(lib) if home.zone[lib.row[t]] == 0]
    plan = A.decide(home, [_play(lib, t, 1) for t in homes[:3]], 2, random.Random(1), mix=1.0, breadth=1,
                    count_out=lambda w: 10 * w)
    assert plan.widened and plan.breadth >= 3, plan


def test_stream_in_auto_mode_reports_state_and_closes_a_refused_style():
    """Сквозной: «Свой поток» в авто — состояние в журнале запроса, а дважды пропущенный стиль (как ДнБ 16.11) не возвращается
    следующим треком."""
    lib = make_library(sizes=(("Main", 400), ("Side", 120), ("Far", 120)))
    main = _main(lib)
    evs = _history(lib, n=80)
    hist = main[200:208]
    now = NOW
    for k, t in enumerate(hist):
        now += 40_000 if k >= 6 else 300_000
        evs.append(_event(t, "skip" if k >= 6 else "listen", now, 15 if k >= 6 else 100))
    out = 0
    for seed in range(6):
        prof = nova.Profile(lib, _listened(range(1, 41), NOW - 30 * DAY) + evs, now)
        tr = {}
        picks = nova.next_tracks(lib, prof, current=hist[-1], history=hist, n=3, now_ms=now, rng=random.Random(seed),
                                 mode=nova.MODE_MY, mood=nova.Mood.from_events(lib, list(reversed(evs)), now), model="listener",
                                 trace=tr, auto={"mix": None, "breadth": None})
        assert picks and "auto" in tr and "Main" in tr["auto"]["closed"], tr.get("auto")
        out += lib.tracks[picks[0].track_id].style != "Main"
    assert out == 6, out



def test_leave_label_without_auto_mode_is_kept():
    """Без авто-режима (stream_auto = false, гость, мало истории) уход 0018 подписан «вы пропускали «Main»»: независимая проверка
    16.11.2026 нашла, что блок подписи оказался внутри ветки авто-режима и без неё не выполнялся."""
    from tests.test_skip_out import _ask, _session_with_skips
    lib = make_library()
    labelled = 0
    for seed in range(6):
        hist, evs, now = _session_with_skips(lib, seed)
        picks, tr, _ = _ask(lib, hist, evs, now, seed)
        labelled += any(p.why == "leave" and "Main" in p.reason for p in picks)
    assert labelled == 6, labelled


def test_style_change_in_auto_mode_keeps_energy_smooth():
    """Смена стиля в тему (17.11.2026): на истории владельца смену стиля дослушивали при скачке энергии в медиане 0,17, пропускали —
    при 0,38. Когда Main закрыт пропусками и поток уходит в другой стиль, первый трек нового стиля — со скачком энергии не больше
    AUTO_CHANGE_ENERGY от играющего (такие в кандидатах есть: энергия разбросана по всей шкале)."""
    import numpy as np
    lib = make_library(sizes=(("Main", 400), ("Side", 120), ("Far", 120)))
    lib.gate.energy = np.random.default_rng(11).uniform(0.0, 1.0, len(lib.ids)).astype(np.float32)
    main = _main(lib)
    evs = _history(lib, n=80)
    hist = main[200:208]
    now = NOW
    for k, t in enumerate(hist):
        now += 40_000 if k >= 6 else 300_000
        evs.append(_event(t, "skip" if k >= 6 else "listen", now, 15 if k >= 6 else 100))
    jumps = []
    for seed in range(8):
        prof = nova.Profile(lib, _listened(range(1, 41), NOW - 30 * DAY) + evs, now)
        picks = nova.next_tracks(lib, prof, current=hist[-1], history=hist, n=3, now_ms=now, rng=random.Random(seed),
                                 mode=nova.MODE_MY, mood=nova.Mood.from_events(lib, list(reversed(evs)), now), model="listener",
                                 trace={}, auto={"mix": None, "breadth": None})
        first = picks[0].track_id
        assert lib.tracks[first].style != "Main"
        jumps.append(abs(float(lib.gate.energy[lib.row[first]] - lib.gate.energy[lib.row[hist[-1]]])))
    assert max(jumps) <= getattr(nova, "AUTO_CHANGE_ENERGY", 0.2), jumps  # 0,2 — для проверки на коде до правки
