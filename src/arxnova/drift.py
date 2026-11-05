"""Дрейф между стилями: куда смещается сессия и кто решает (решение 0013).

Владелец 05.11.2026: подмес — проба, «чтоб дрейф состоялся, пользователь сам должен захотеть сместить акцент». Поэтому
таймеров нет. У каждой сессии свой основной стиль M. Остальное — доля разведки w и веса направлений φ:

    π(M) = 1 − w,   π(c) = w·φ(c)

После каждого показа работают экспоненциальные веса с возвратом к исходной картине (fixed share, Herbster, Warmuth 1998):

    φ(c) ← (1 − α)·φ(c)·exp(g(c)) / Z + α·φ⁰(c),   logit w ← logit w + g_w,   w ← w + α·(w⁰ − w)

g — логарифм отношения правдоподобия для действия слушателя. Обычное дослушивание почти ничего не сдвигает: так кончается
90 % показов, в нём нет сведений о желании. Лайк, возврат, «Похожие», свой выбор двигают сильно, пропуск отталкивает. Для
плюса g не постоянное: шансы направления становятся не меньше 4:1. Иначе в медиатеке с двадцатью стилями у лайка нет силы:
доля 3 % от него вырастает только до 24 %, а в маленькой он сразу решал бы всё.
Когда π(c) > π(M), основным становится c: это и есть дрейф. Два отказа основного стиля подряд или половина последних шести
вразнобой — уход сейчас, не дожидаясь весов; из любимого стиля порог выше (ранние пропуски). Нормы подобраны под шаги
1 из 16 → 1 из 5 → половина → основной. Домой короче: половина проб после дрейфа и две отметки вместо трёх.

Состояние не хранится: оно каждый раз собирается по истории сессии, поэтому его нельзя рассинхронизировать с журналом."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field

PROBE_SHARE = 1 / 16  # доля проб при средней смелости
BOLD_POWER = 1.4  # смелость 1 → примерно 1 из 6, 0 → проб нет
PROBE_SHARE_MAX = 0.3
REVERT = 0.02  # α: без подкрепления сдвиг забывается за десятки треков (половина за 35)
# Множители веса направления и шансов разведки
PLUS_DIRECTION, PLUS_EXPLORE = 10.0, 3.75
PLUS_DIRECTION_ODDS = 4.0  # после плюса на это направление идёт минимум 4/5 проб: сила жеста не зависит от числа стилей
PLUS_EXPLORE_HOME = 6.0  # домой короче, чем в новое: стиль слушателя и так известен, две отметки вместо трёх
DONE_DIRECTION = 1.15
MID_DIRECTION, MID_EXPLORE = 0.5, 0.8
EARLY_DIRECTION, EARLY_EXPLORE = 0.2, 0.5
DISLIKE_DIRECTION = 0.05
MAIN_SKIP_EXPLORE = 1.3  # пропускают основной — пробуем другое чаще
# Уход из основного стиля. Из чужого легче: два отказа подряд любой глубины; из любимого — неусидчивость 2 (два ранних)
REFUSALS_AWAY = 2
RESTLESS_LEAVE = 2.0
SESSION_HEAD = 3  # в первых треках потока проб нет (nova.PROBE_MIN_TRACKS): они и говорят, с чего он начался
SETTLED_WINDOW, SETTLED_TRACKS = 8, 6  # столько треков подряд в одном стиле: поток уже там, даже если причина выпала из хвоста истории
SPREAD_WINDOW, SPREAD_REFUSALS = 6, 3  # вне дома: половина последних треков основного стиля пропущена вразнобой — тоже просьба уйти
REFUSED_FRESH = 10  # от какого стиля только что ушли по отказам, туда столько треков не идём: иначе отказ сразу возвращает обратно
HOLD_GAPS = 2.0  # принятое направление держится, пока его пробы приходят и дослушиваются: столько промежутков между пробами
GAP_JITTER = 0.25  # ±: пробы не ходят по часам
EARLY_SKIP_PCT, LATE_SKIP_PCT = 15, 60
REFUSALS = ("early", "mid", "dislike")
ACCEPTS = ("plus", "done", "late")
# Дом: обычный стиль слушателя, отдельно от сессии и гораздо медленнее (владелец 05.11.2026: «дом тоже должен уметь дрейфовать и/или
# сохранять пограничные состояния, но к дрейфу дома надо отнестись очень осторожно, необходим очевидный уход от привычного»)
HOME_RATE = 1 / 20  # день целиком в другом стиле сдвигает долю на 5 %: вторым домом стиль становится примерно за 6 таких дней, главным за 16
HOME_FULL_DAY = 8.0  # день из нескольких треков считается долей дня: случайный трек не голос
HOME_SECOND_IN, HOME_SECOND_OUT = 0.25, 0.18  # второй дом: вход и выход порознь, чтобы граница не дрожала
HOME_SECOND_DAYS = 3.0  # и стиль должен быть заметен не один день: у нового слушателя один вечер не второй дом
HOME_SWITCH = 1.25  # главный меняется, когда другой стиль заметно обошёл его
HOME_SWITCH_DAYS = 3.0  # и обгонял его день за день хотя бы столько раз
HOME_HABIT_DAYS = 3.0  # пока истории меньше, привычки нет: один случайный трек в первый день не должен держать дом три дня
TRIED_WINDOW = 30


def home_of(days: Iterable[Mapping[str, float]]) -> dict[str, float]:
    """Дом по дням прослушивания, главный стиль первым: {стиль: доля}. Второй стиль в ответе — пограничное состояние.

        H(c) ← (1 − λ)·H(c) + λ·s_d(c),   λ = v_d·max(1/20, 1/дней)

    s_d — доля стиля в дне, v_d — полнота дня. Каждый день — один голос, сколько бы треков ни прозвучало, поэтому один длинный вечер
    дом не сдвигает, а пока истории мало, привычки ещё нет и оценка гибкая. Дни без музыки ничего не меняют: таймера нет. Живёт
    между двумя стилями — доли сходятся к его пропорции и держатся, а главный не дрожит: для смены нужен заметный обгон."""
    shares: dict[str, float] = {}
    primary: str | None = None
    second: set[str] = set()
    led: Counter[str] = Counter()
    present: Counter[str] = Counter()
    seen = 0.0
    for day in days:
        total = sum(day.values())
        if total <= 0.0:
            continue
        full = min(1.0, total / HOME_FULL_DAY)
        seen += full
        rate = full * max(HOME_RATE, 1.0 / seen)
        for style in set(shares) | set(day):
            shares[style] = (1.0 - rate) * shares.get(style, 0.0) + rate * day.get(style, 0.0) / total
            if day.get(style, 0.0) >= HOME_SECOND_IN * total:
                present[style] += full
        if primary is None:
            primary = max(shares, key=shares.get)
        lead = max(day, key=day.get)
        if lead != primary and day[lead] > day.get(primary, 0.0):
            led[lead] += full
        best = max(shares, key=shares.get)
        if best != primary and shares[best] >= HOME_SWITCH * shares[primary] and (seen < HOME_HABIT_DAYS or led[best] >= HOME_SWITCH_DAYS):
            primary = best
            led.clear()
        for style, x in shares.items():
            if x >= HOME_SECOND_IN and present[style] >= HOME_SECOND_DAYS:
                second.add(style)
            elif x < HOME_SECOND_OUT:
                second.discard(style)
    if primary is None:
        return {}
    second.discard(primary)
    return {primary: shares[primary], **{s: shares[s] for s in sorted(second, key=shares.get, reverse=True)}}


def probe_share(boldness: float) -> float:
    b = min(1.0, max(0.0, boldness))
    return min(PROBE_SHARE_MAX, PROBE_SHARE * (2.0 * b) ** BOLD_POWER)


def kind(event_type: str, pct: float, organic: bool, reason: str | None = None, left_by: str | None = None) -> str:
    """Что действие значит для желания: plus, done, late, mid, early, dislike или none.
    Свой выбор — сильный плюс; пролистывание своего альбома и уход кнопкой «назад» — не отказ стилю."""
    if event_type in ("like", "replay", "similar"):
        return "plus"
    if event_type == "listen":
        return "plus" if organic else "done"
    if event_type == "dislike":
        return "dislike"
    if event_type != "skip" or organic or left_by == "back" or reason == "tired":
        return "none"
    if reason in ("genre", "not_now"):
        return "early"
    if reason in ("rhythm", "vocal"):
        return "mid"
    return "early" if pct < EARLY_SKIP_PCT else "mid" if pct < LATE_SKIP_PCT else "late"


def _odds(p: float, k: float) -> float:
    if p <= 0.0:
        return 0.0
    o = p / max(1.0 - p, 1e-9) * k
    return o / (1.0 + o)


@dataclass
class Drift:
    main: str
    home: str
    explore: float  # w
    base_explore: float  # w⁰
    directions: dict[str, float] = field(default_factory=dict)  # φ
    prior: Callable[[str], dict[str, float]] = lambda style: {}
    leaving: str | None = None  # основной стиль, от которого отказались: следующий трек уводит из него
    refusals: int = 0
    restless: float = 0.0
    moves: list[str] = field(default_factory=list)  # смены основного за сессию
    recent: list[int] = field(default_factory=list)  # последние показы основного стиля: 1 — отказ
    left_at: dict[str, int] = field(default_factory=dict)  # откуда уходили по отказам и на каком показе
    played: int = 0
    homes: frozenset[str] = frozenset()  # дом: обычный стиль и, в пограничном состоянии, второй
    accepted: set[str] = field(default_factory=set)  # направления с плюсом в этой сессии
    held_until: int = 0
    last_refused: int | None = None  # трек последнего отказа: вернулся и снова ушёл — это один отказ, а не два
    probes: list[tuple[int, str]] = field(default_factory=list)  # показы не основного стиля с последней смены основного

    def refused_lately(self, style: str) -> bool:
        at = self.left_at.get(style)
        return at is not None and self.played - at < REFUSED_FRESH

    def tried(self) -> Counter[str]:
        """Какие стили пробовали недавно. Только пробы: что было основным до смены, в счёт не идёт, иначе домой почти нечем вернуться."""
        return Counter(style for at, style in self.probes if self.played - at < TRIED_WINDOW)

    def share(self, style: str) -> float:
        if style == self.main:
            return 1.0 - self.explore
        return self.explore * self.directions.get(style, 0.0)

    def gap(self, rng) -> float:
        """Через сколько треков следующая проба."""
        if self.explore <= 0.0 or not self.directions:
            return float("inf")
        return rng.uniform(1.0 - GAP_JITTER, 1.0 + GAP_JITTER) / self.explore

    def gap_chance(self, since: int, first: bool = True) -> float:
        """Шанс, что проба стоит именно на этом месте пачки: промежуток один на пачку, поэтому дальше первого места — только его часть."""
        if self.explore <= 0.0:
            return 0.0
        low, high = (1.0 - GAP_JITTER) / self.explore, (1.0 + GAP_JITTER) / self.explore
        reached = lambda s: min(1.0, max(0.0, (s - low) / (high - low)))  # noqa: E731
        return reached(since) if first else reached(since) - reached(since - 1)

    def _move_to(self, style: str) -> None:
        if self.leaving is not None:
            self.left_at[self.leaving] = self.played
        self.main = style
        self.explore = self.base_explore
        self.directions = dict(self.prior(style))
        self.leaving = None
        self.refusals = 0
        self.restless = 0.0
        self.recent = []
        self.probes = []
        self.accepted = set()
        self.held_until = 0
        self.moves.append(style)

    def _revert(self) -> None:
        if self.played < self.held_until:
            return  # слушатель принимает это направление: пограничное состояние не растёт и не гаснет само
        self.explore += REVERT * (self.base_explore - self.explore)
        base = self.prior(self.main)
        if base:
            self.directions = {c: (1.0 - REVERT) * self.directions.get(c, 0.0) + REVERT * x for c, x in base.items()}

    def _on_main(self, k: str) -> None:
        if k in REFUSALS:
            self.refusals += 1
            self.restless += {"early": 1.0, "mid": 0.5, "dislike": 1.5}[k]
            self.explore = _odds(self.explore, MAIN_SKIP_EXPLORE)
        elif k == "plus":
            self.refusals = 0
            self.restless = max(0.0, self.restless - 1.5)
        elif k in ("done", "late"):
            self.refusals = 0
            self.restless *= 0.5

    def _on_probe(self, style: str, k: str) -> None:
        d = self.directions
        if style not in d:
            d[style] = min(d.values(), default=1.0)  # стиль не из списка (маленький), но сыграл: теперь есть
        if k == "plus":
            d[style] = max(d[style] * PLUS_DIRECTION, PLUS_DIRECTION_ODDS * (sum(d.values()) - d[style]))
            self.explore = _odds(self.explore, PLUS_EXPLORE_HOME if style in self.homes else PLUS_EXPLORE)
            self.left_at.pop(style, None)  # передумал
            self.accepted.add(style)
        elif k == "done":
            d[style] *= DONE_DIRECTION
        elif k == "mid":
            d[style] *= MID_DIRECTION
            self.explore = _odds(self.explore, MID_EXPLORE)
        elif k == "early":
            d[style] *= EARLY_DIRECTION
            self.explore = _odds(self.explore, EARLY_EXPLORE)
        elif k == "dislike":
            d[style] *= DISLIKE_DIRECTION
            self.explore = _odds(self.explore, EARLY_EXPLORE)
        if k in ("mid", "early", "dislike"):
            self.accepted.discard(style)
            self.held_until = 0
        elif k in ("plus", "done") and style in self.accepted and self.explore > 0.0:
            self.held_until = self.played + 1 + math.ceil(HOLD_GAPS / self.explore)
        total = sum(d.values())
        if total > 0:
            for c in d:
                d[c] /= total
        if self.share(style) > self.share(self.main):
            self._move_to(style)

    def _asks_out(self) -> bool:
        if self.main in self.homes:
            return self.restless >= RESTLESS_LEAVE
        return (self.restless >= RESTLESS_LEAVE or self.refusals >= REFUSALS_AWAY
                or sum(self.recent[-SPREAD_WINDOW:]) >= SPREAD_REFUSALS)


def _by_play(tracks: list[int], events_of: Callable[[int], list]) -> list[list[tuple]]:
    """Действия по показам. Трек может звучать в сессии второй раз, а его действия лежат одним списком. Показу достаётся то, что
    случилось не раньше конца предыдущего и до его собственного конца (дослушал или ушёл). Более старое — прошлый показ, которого
    может не быть в хвосте истории. Лайк после конца остаётся этому показу, если дальше повтора нет."""
    left = {t: list(events_of(t)) for t in set(tracks)}
    last = {t: i for i, t in enumerate(tracks)}
    plays: list[list[tuple]] = []
    after = None
    for i, t in enumerate(tracks):
        rest = left[t]
        while rest and after is not None and rest[0][5] < after:
            rest.pop(0)
        play: list[tuple] = []
        while rest:
            e = rest.pop(0)
            play.append(e)
            if e[0] in ("listen", "skip"):
                after = e[5]
                break
        if last[t] == i:
            play += rest
        plays.append(play)
    return plays


def follow(hist: Iterable[int], style_of: Callable[[int], str], events_of: Callable[[int], list],
           prior: Callable[[str], dict[str, float]], home: str, boldness: float, homes: Iterable[str] = ()) -> Drift:
    """Состояние дрейфа по сессии. events_of(трек) — действия по нему: (тип, доля, свой выбор, причина, как ушёл, время)."""
    tracks = list(hist)
    plays = _by_play(tracks, events_of)
    head = Counter(style_of(t) for t in tracks[:SESSION_HEAD]).most_common(1)[0][0] if tracks else home
    d = _replay(tracks, plays, head, style_of, prior, home, boldness, homes)
    if tracks:
        # Хвост истории у старых плееров короче потока: лайк, с которого начался сдвиг, мог из него выпасть
        settled, count = Counter(style_of(t) for t in tracks[-SETTLED_WINDOW:]).most_common(1)[0]
        if count >= SETTLED_TRACKS and settled != d.main:
            d = _replay(tracks, plays, settled, style_of, prior, home, boldness, homes)
    return d


def _replay(tracks: list[int], plays: list[list[tuple]], first: str, style_of: Callable[[int], str],
            prior: Callable[[str], dict[str, float]], home: str, boldness: float, homes: Iterable[str]) -> Drift:
    w0 = probe_share(boldness)
    d = Drift(first, home, w0, w0, dict(prior(first)), prior, homes=frozenset({home, *homes}))
    for i, (tid, actions) in enumerate(zip(tracks, plays)):
        style = style_of(tid)
        d.played = i
        kinds = [kind(et, pct, organic, reason, left_by) for et, pct, organic, reason, left_by, _ in actions]
        if "dislike" in kinds:  # лайкнул, а потом «не предлагать»: важно последнее. Снятие лайка плеер при этом не пишет
            kinds = kinds[len(kinds) - 1 - kinds[::-1].index("dislike"):]
        # Лайкнул и пошёл дальше — лайк, а не отказ, и на основном стиле, и на пробе (6 из 16 лайков на снимке кончились пропуском)
        refusal = None
        if "plus" not in kinds and tid != d.last_refused:
            refusal = "dislike" if "dislike" in kinds else next((k for k in kinds if k in REFUSALS), None)
        accepted = any(k in ACCEPTS for k in kinds)
        # Просили уйти, а следующий трек другого стиля: шаг вышел, только если его приняли. Проба, что стояла в очереди ещё до отказов,
        # и есть ещё не прозвучавшая, а куда идти, решает просьба: сначала принятые пробы, потом дом.
        if d.leaving is not None and style != d.leaving and accepted:
            d._move_to(style)
        d._revert()
        for k in [k for k in kinds if k in ACCEPTS] + ([refusal] if refusal else []):
            if style == d.main:
                d._on_main(k)
            else:
                d._on_probe(style, k)
        if refusal:
            d.last_refused = tid
        elif accepted:
            d.last_refused = None
        if style != d.main:
            d.probes.append((i, style))
        if style == d.main:
            if refusal:
                d.recent.append(1)
            elif accepted:
                d.recent.append(0)
            # дослушал следующий трек этого же стиля — успокоился, просьбы уйти больше нет
            d.leaving = d.main if d._asks_out() else None if accepted else d.leaving
    return d
