"""Авто-режим «Своего потока» (решение 0020): дом — область дослушанного, вне дома — кольца по удалённости звучания,
две ручки — «примесь» (какая доля потока вне дома) и «широта» (до какого кольца можно уходить).

Дом. Всё, что слушатель принял (дослушал, лайк, возврат, свой выбор), с затуханием HOME_HALF_LIFE_DAYS, — несколько уголков на
карте звучания (сферические k-средних по lib.vec, вес по исходу). Расстояние трека до дома d = 1 − близость к ближайшему уголку.
Граница дома — перцентиль HOME_PCT расстояний самих принятых: дом широкий, в нём и неслышанные соседи дослушанного. Владелец
16.11.2026: «Я не уверен что дом должен быть узким..... Он должен быть разнообразным, чтоб не надоедал.... Но построенным на
дослушивании, да». Дом движется сам: принятое далеко становится новым уголком, забытый уголок тает.

Кольца. Всё вне дома делится по d на RINGS колец равной численности: 1 — у самого дома, RINGS — дальше всего. Порядок
«похожести жанров» на истории владельца — монетка (проверка 14.11.2026), а близость треков по звуку угадывает дослушивание
(0,589), поэтому кольца — по звуку, а не по ярлыкам стилей.

Авто. Дом, каждый уголок и каждое кольцо — бета-распределения «принят / отказ» на показах потока этого слушателя (затухание
ZONE_HALF_LIFE_DAYS) плюс текущая сессия с весом SESSION_W. Кольцо выбирается выборкой Томпсона в пределах широты: где данных мало,
распределение широкое, и поток туда заглядывает сам — так разведывается неуслышанное. Примесь растёт от отказов дому и падает от
отказов вне дома; два отказа дому подряд (с последнего принятого, любые стили) — уходим из дома, пока что-то не примут; два отказа
вне дома подряд — домой и широта на кольцо уже. Принятое в кольце r расширяет широту до r + 1.
"""
from __future__ import annotations

import math
import random
import threading
from dataclasses import dataclass, field

import numpy as np

from arxnova import drift as D
from arxnova import listener as LM

HOME_K = 12  # уголков дома: мелкие уголки, где дослушано хоть несколько треков, не растворяются в главном
HOME_PCT = 95.0  # граница дома — перцентиль расстояний принятых до своих уголков
HOME_HALF_LIFE_DAYS = 30.0
HOME_TTL_MS = 3_600_000  # дом пересчитывается не чаще раза в час
STATS_CUT_MS = 3 * 3_600_000  # показы моложе — это текущая сессия, в счёт зон истории не идут
HOME_MIN_ROWS = 20  # меньше принятых треков — дома ещё нет, авто молчит (поток как раньше)
RINGS = 5
ZONE_HALF_LIFE_DAYS = 14.0
ZONE_PRIOR = (2.0, 2.0)  # (принят, отказ) — пока данных нет, кольцо считается «может зайти» и поток туда заглядывает
SESSION_W = 2.0
MIX_MIN, MIX_MAX = 0.1, 0.7
MIX_STEP = 0.1  # за каждый отказ дому среди последних SESSION_WINDOW домашних показов
SESSION_WINDOW = 6
LEAVE_REFUSALS = 2  # отказов дому подряд — в другой, далёкий уголок дома
LEAVE_HOME = 4  # отказов дому подряд (два уголка отвергнуты) — из дома
BACK_REFUSALS = 2  # отказов вне дома подряд — домой, широта уже
CORNER_REFUSALS = 2  # отказов в уголке за сессию — до конца сессии туда не идём (если есть куда)
STYLE_REFUSALS = 2  # отказов стилю после последнего принятого в нём — до принятого он закрыт в любой зоне
CORNER_STAND = 4  # треков подряд в уголке дома, потом — к соседнему уголку: дом разнообразный, но течёт плавно (с 6 за сессию
#                   звучало 3,6 уголка, один — 60 % дома: прогон 16.11.2026)
OUT_LEN = 3  # вылазка из дома — столько треков в кольце
BREADTH_START = 3  # кольца 1…3 открыты всегда, дальние — когда данные говорят «заходит» или там приняли трек
BREADTH_MIN = 1
WIDE_MIN_TRACKS = 30  # в пределах широты годных треков меньше — расширяемся сами (защита от «узко и много»)
KMEANS_ITERS = 15


@dataclass
class Home:
    corners: np.ndarray  # k × dim, единичные
    mass: np.ndarray  # вес принятого в каждом уголке
    dist: np.ndarray  # d по всем строкам медиатеки
    corner: np.ndarray  # ближайший уголок каждой строки
    zone: np.ndarray  # 0 — дом, 1…RINGS — кольцо
    edge: float  # граница дома по d
    stats: dict  # зона → [принят, отказ] из истории (с затуханием); ("c", k) — уголок дома


@dataclass
class Plan:
    mix: float
    breadth: int
    manual: bool
    widened: bool
    slots: list[tuple[int, int | None]]  # на каждое место пачки: (зона, уголок для дома)
    leaving: bool = False  # уходим из дома после отказов
    back: bool = False  # возвращаемся домой после отказов вне дома
    why: dict = field(default_factory=dict)
    closed_styles: frozenset = frozenset()  # стили, отвергнутые в сессии: STYLE_REFUSALS отказов после последнего принятого в них

    def summary(self) -> dict:
        return {"mix": round(self.mix, 3), "breadth": self.breadth, "manual": self.manual, "widened": self.widened,
                "leaving": self.leaving, "back": self.back, "slots": [z for z, _ in self.slots],
                "closed": sorted(self.closed_styles), **self.why}


def closed_styles(state: list[LM.Play], style_of) -> frozenset:
    """Стили, которые слушатель в этой сессии отверг: STYLE_REFUSALS отказов после последнего принятого трека этого стиля.
    Дом и кольца — по звуку, а устают от жанра: ДнБ лежит и в доме, и в ближних кольцах, и без этого после ухода из дома
    поток возвращался к ДнБ через кольцо (прогон 16.11.2026: 3,5 ДнБ из 5 после ухода)."""
    refused: dict[str, int] = {}
    done: set[str] = set()
    for p in reversed(state):
        s = style_of(p.row)
        if not s or s in done:
            continue
        if p.outcome > 0:
            done.add(s)
        elif p.outcome < 0:
            refused[s] = refused.get(s, 0) + 1
    return frozenset(s for s, n in refused.items() if n >= STYLE_REFUSALS)


def start_corner(home: Home, rng: random.Random) -> int:
    """С какого уголка дома начать сессию: по корню из дослушанного — главный уголок чаще, но не всегда (0018: все сессии
    владельца начинались с ДнБ)."""
    return rng.choices(range(len(home.corners)), weights=[max(x, 1e-6) ** 0.25 for x in home.mass])[0]


def _kmeans(X: np.ndarray, w: np.ndarray, k: int) -> np.ndarray:
    """Сферические k-средних с весами, начало как в k-means++ на постоянном зерне: один и тот же дом на одних и тех же данных."""
    rng = np.random.default_rng(0)
    first = int(np.argmax(w))
    C = [X[first]]
    for _ in range(1, k):
        d = 1.0 - np.max(X @ np.stack(C).T, axis=1)
        p = np.clip(d, 0.0, None) * w
        if p.sum() <= 0:
            break
        C.append(X[int(rng.choice(len(X), p=p / p.sum()))])
    C = np.stack(C).astype(np.float32)
    Xw = (X * w[:, None]).astype(np.float32)
    for _ in range(KMEANS_ITERS):
        # суммы по уголкам одним умножением (метки → разреженная матрица), а не проходом по каждому: на Pi пересчёт дома
        # давал хвост запроса 270 мс (замер 16.11.2026)
        lab = np.argmax(X @ C.T, axis=1)
        onehot = np.zeros((len(C), len(X)), dtype=np.float32)
        onehot[lab, np.arange(len(X))] = 1.0
        S = onehot @ Xw
        n = np.linalg.norm(S, axis=1, keepdims=True)
        new = np.where(n > 0, S / np.maximum(n, 1e-12), C)
        if np.allclose(new, C, atol=1e-5):
            break
        C = new
    return C.astype(np.float32)


def _weighted_pct(x: np.ndarray, w: np.ndarray, pct: float) -> float:
    o = np.argsort(x)
    cw = np.cumsum(w[o])
    return float(x[o][min(len(x) - 1, int(np.searchsorted(cw, cw[-1] * pct / 100.0)))])


def home_of(lib, events, now_ms: int) -> Home | None:
    """Дом и счёт зон по истории. None — принятого слишком мало: авто молчит."""
    acc: dict[int, float] = {}
    shows: list[tuple[int, int, float]] = []  # (строка, исход, вес) — показы потока
    for e in events:
        tid = e["track_id"]
        if tid is None or tid not in lib.row:
            continue
        et = e["event_type"]
        if et not in ("listen", "skip", "like", "replay", "dislike"):
            continue
        organic = bool(e["is_organic"])
        k = D.kind(et, e["played_ratio_pct"] or 0, organic, e["skip_reason"], LM._get(e, "left_by"))
        o = LM.outcome_of([k])
        age = max(0.0, (now_ms - e["ts"]) / 86_400_000)
        row = lib.row[tid]
        if o > 0:
            acc[row] = acc.get(row, 0.0) + (3.0 if et == "like" else 1.0) * 0.5 ** (age / HOME_HALF_LIFE_DAYS)
        # счёт зон — без последних STATS_CUT_MS: текущая сессия добавляется в decide с весом SESSION_W, а дом пересчитывается раз в
        # час посреди сессии — иначе её показы считались бы дважды и примесь со широтой прыгали (независимая проверка 16.11.2026)
        if (o != 0 and et in ("listen", "skip", "dislike") and not organic and LM._get(e, "nova_request_id")
                and now_ms - e["ts"] > STATS_CUT_MS):
            shows.append((row, o, 0.5 ** (age / ZONE_HALF_LIFE_DAYS)))
    if len(acc) < HOME_MIN_ROWS:
        return None
    rows = np.array(list(acc), dtype=int)
    w = np.array([acc[r] for r in rows], dtype=float)
    X = lib.vec[rows].astype(np.float32)
    C = _kmeans(X, w, min(HOME_K, max(1, len(rows) // 3)))
    sim = lib.vec @ C.T
    corner = np.argmax(sim, axis=1)
    dist = 1.0 - sim[np.arange(len(corner)), corner]
    edge = _weighted_pct(dist[rows], w, HOME_PCT)
    zone = np.zeros(len(dist), dtype=int)
    out = np.flatnonzero(dist > edge)
    if len(out):
        rank = np.empty(len(out), dtype=int)
        rank[np.argsort(dist[out], kind="stable")] = np.arange(len(out))
        zone[out] = 1 + (rank * RINGS) // len(out)
    mass = np.zeros(len(C))
    np.add.at(mass, corner[rows], w)
    stats: dict = {}
    for row, o, wt in shows:
        for key in ((int(zone[row]),) + ((("c", int(corner[row])),) if zone[row] == 0 else ())):
            s = stats.setdefault(key, [0.0, 0.0])
            s[0 if o > 0 else 1] += wt
    return Home(C, mass, dist, corner, zone, float(edge), stats)


def home_for(lib, prof, now_ms: int) -> Home | None:
    """Дом слушателя. Профиль пересобирается с каждым новым событием, а дом меняется за дни: пересчёт — раз в HOME_TTL_MS
    (на Mac сборка дома 18 мс, на Pi в разы дольше — каждый запрос не потянет). Текущая сессия в выборе учитывается сразу
    (decide смотрит на её показы), из истории тают только старые показы. Ключ — устройства слушателя: у гостя свой дом."""
    devices = tuple(sorted({e["device_id"] for e in prof._events[-200:] if LM._get(e, "device_id") is not None}))
    memo = lib.__dict__.setdefault("_auto_home_memo", {})
    got = memo.get(devices)
    if got is None:  # самый первый запрос после запуска сервера — ждём
        got = memo[devices] = (now_ms, home_of(lib, prof._events, now_ms))
    elif not (0 <= now_ms - got[0] < HOME_TTL_MS) and devices not in _rebuilding:
        # устарел — пересчёт в фоне, а запрос идёт с прежним домом: на Pi сборка дома 150 мс (умножение матриц), в запросе это хвост
        # 270 мс (замер 16.11.2026)
        _rebuilding.add(devices)
        events = list(prof._events)

        def rebuild():
            try:
                memo[devices] = (now_ms, home_of(lib, events, now_ms))
            finally:
                _rebuilding.discard(devices)

        threading.Thread(target=rebuild, name="arxnova-home", daemon=True).start()
    return got[1]


_rebuilding: set = set()


def _beta(stat: list[float] | None, sess: tuple[float, float] = (0.0, 0.0)) -> tuple[float, float]:
    a, b = ZONE_PRIOR
    if stat:
        a, b = a + stat[0], b + stat[1]
    return a + SESSION_W * sess[0], b + SESSION_W * sess[1]


def _mean(ab: tuple[float, float]) -> float:
    return ab[0] / (ab[0] + ab[1])


def _bounds(ab: tuple[float, float], z: float = 1.28) -> tuple[float, float]:
    a, b = ab
    m = a / (a + b)
    sd = math.sqrt(a * b / ((a + b) ** 2 * (a + b + 1)))
    return m - z * sd, m + z * sd


def session_counts(home: Home, state: list[LM.Play]) -> dict:
    """Счёт этой сессии по зонам и уголкам: [принят, отказ]."""
    out: dict = {}
    for p in state:
        if p.outcome == 0:
            continue
        z = int(home.zone[p.row])
        for key in ((z,) + ((("c", int(home.corner[p.row])),) if z == 0 else ())):
            s = out.setdefault(key, [0.0, 0.0])
            s[0 if p.outcome > 0 else 1] += 1
    return out


def refusals_since_accepted(home: Home, state: list[LM.Play]) -> tuple[int, int, list[int]]:
    """С последнего принятого трека (треки без событий серию не рвут): отказов дому, вне дома и зоны отказов вне дома."""
    h = o = 0
    rings: list[int] = []
    for p in reversed(state):
        if p.outcome > 0:
            break
        if p.outcome < 0:
            z = int(home.zone[p.row])
            if z == 0:
                h += 1
            else:
                o += 1
                rings.append(z)
    return h, o, rings


def decide(home: Home, state: list[LM.Play], n: int, rng: random.Random, *, mix: float | None = None,
           breadth: int | None = None, count_out=None, style_of=None) -> Plan:
    """Куда пойдут следующие n мест. mix и breadth — ручной режим (ползунки), None — авто. count_out(w) — сколько годных треков
    вне дома в кольцах 1…w: для защиты от узкой широты."""
    manual = mix is not None or breadth is not None
    sess = session_counts(home, state)
    h_ab = _beta(home.stats.get(0), tuple(sess.get(0, (0, 0))))
    ring_ab = {r: _beta(home.stats.get(r), tuple(sess.get(r, (0, 0)))) for r in range(1, RINGS + 1)}
    home_ref, out_ref, out_rings = refusals_since_accepted(home, state)
    why: dict = {"p_home": round(_mean(h_ab), 3)}

    # широта
    if breadth is not None:
        w = int(min(RINGS, max(BREADTH_MIN, breadth)))
    else:
        p_home_ = _mean(h_ab)
        w = BREADTH_START
        for r in range(BREADTH_START + 1, RINGS + 1):  # дальнее кольцо открыто, если там заходит почти как дома
            if sum(home.stats.get(r, (0, 0))) >= 3 and _mean(ring_ab[r]) >= p_home_ - 0.15:
                w = r
        for p in state:  # принятое в сессии в кольце r — можно на кольцо дальше
            z = int(home.zone[p.row])
            if p.outcome > 0 and z > 0:
                w = max(w, min(RINGS, z + 1))
        if out_ref >= BACK_REFUSALS and out_rings and max(out_rings) >= w - 1:
            w = max(BREADTH_MIN, min(w, max(out_rings)) - 1)
    widened = False
    if count_out is not None:
        while w < RINGS and count_out(w) < WIDE_MIN_TRACKS:
            w += 1
            widened = True

    # примесь
    if mix is not None:
        m = float(min(1.0, max(0.0, mix)))
    else:
        p_out = max(_mean(ring_ab[r]) for r in range(1, w + 1))
        p_home = _mean(h_ab)
        m = 0.1 + 0.6 * p_out / max(1e-6, p_out + p_home)
        recent_home = [p for p in state if p.outcome != 0 and home.zone[p.row] == 0][-SESSION_WINDOW:]
        recent_out = [p for p in state if p.outcome != 0 and home.zone[p.row] > 0][-SESSION_WINDOW:]
        m += MIX_STEP * sum(1 for p in recent_home if p.outcome < 0)
        m -= MIX_STEP * sum(1 for p in recent_out if p.outcome < 0)
        m = float(min(MIX_MAX, max(MIX_MIN, m)))

    # Серия отказов (с последнего принятого, любые стили): два отказа в уголке дома — в далёкий уголок; LEAVE_HOME отказов дому —
    # из дома; два отказа вне дома — домой (если дом сам не отвергнут)
    k = len(home.corners)
    closed = {c for c in range(k) if sess.get(("c", c), (0, 0))[1] >= CORNER_REFUSALS}
    series_corners = []
    for p in reversed(state):
        if p.outcome > 0:
            break
        if p.outcome < 0 and home.zone[p.row] == 0:
            series_corners.append(int(home.corner[p.row]))
    leaving = home_ref >= LEAVE_HOME or (home_ref >= LEAVE_REFUSALS and len(closed) >= k)
    back = back_now = not leaving and out_ref >= BACK_REFUSALS
    jump = not leaving and home_ref >= LEAVE_REFUSALS  # в другой, далёкий от отвергнутых уголок

    played_c: dict[int, int] = {}
    for p in state:
        if home.zone[p.row] == 0:
            played_c[int(home.corner[p.row])] = played_c.get(int(home.corner[p.row]), 0) + 1
    cc = home.corners @ home.corners.T  # близость уголков: переходы внутри дома — к соседним, после отказов — к далёким

    def corner_pick(cur: int | None, far_from: list[int]) -> int:
        weights = []
        for c in range(k):
            if c == cur or c in closed or c in far_from:
                weights.append(0.0)
                continue
            # корень четвёртой степени из дослушанного: главный уголок чаще, но мелкие не тонут. С корнем квадратным и квадратом
            # близости поток ходил по соседним уголкам ДнБ — слушатель без действий получал 82 % ДнБ (прогон 16.11.2026, v3)
            wt = max(home.mass[c], 1e-6) ** 0.25 / (1.0 + played_c.get(c, 0))
            if far_from:
                wt *= max(1e-3, 1.0 - float(np.max(cc[c, far_from])))
            elif cur is not None:
                wt *= max(1e-3, (1.0 + float(cc[c, cur])) / 2.0)
            weights.append(wt)
        if sum(weights) <= 0:
            weights = [math.sqrt(max(x, 1e-6)) if c not in closed else 0.0 for c, x in enumerate(home.mass)]
            if sum(weights) <= 0:
                weights = [1.0] * k
        return rng.choices(range(k), weights=weights)[0]

    def ring_pick(avoid: set[int]) -> int:
        draws = {r: rng.betavariate(*ring_ab[r]) for r in range(1, w + 1) if r not in avoid}
        if not draws:
            draws = {r: rng.betavariate(*ring_ab[r]) for r in range(1, w + 1)}
        return max(draws, key=draws.get)

    # Где поток сейчас: место последнего трека и сколько треков подряд там (дом — уголок, вне дома — вылазка)
    places = [(int(home.zone[p.row]), int(home.corner[p.row]) if home.zone[p.row] == 0 else None) for p in state]
    cur = places[-1] if places else None
    corner_run = home_run = out_run = 0
    if cur is not None:
        for z, c in reversed(places):
            if (z == 0) != (cur[0] == 0):
                break
            if z == 0:
                home_run += 1
            else:
                out_run += 1
        for pl in reversed(places):
            if pl != cur:
                break
            corner_run += 1 if cur[0] == 0 else 0
    last_home_corner = next((c for z, c in reversed(places) if z == 0), None)
    # стоянка дома до вылазки: доля вылазок = примесь (вылазка — OUT_LEN треков)
    home_stay = (10 ** 6 if m <= 0 else max(1, round(OUT_LEN * (1.0 - m) / m)) + rng.randint(-1, 1))

    slots: list[tuple[int, int | None]] = []
    for i in range(n):
        if leaving:
            avoid = set(out_rings) if out_ref else set()
            r = cur[0] if cur and cur[0] > 0 and i > 0 else ring_pick(avoid)
            slot = (r, None)
        elif back or cur is None:
            slot = (0, last_home_corner if last_home_corner is not None and last_home_corner not in closed
                    else corner_pick(None, []))
        elif jump and i == 0:
            slot = (0, corner_pick(None, list(dict.fromkeys(series_corners))))
        elif cur[0] > 0:  # вылазка: держимся кольца OUT_LEN треков (отказ — другое кольцо), потом домой к последнему уголку
            if (out_run < OUT_LEN or m >= 0.99) and w >= 1:  # примесь 100 % (ручной) — домой не возвращаемся
                refused_last = bool(state) and i == 0 and state[-1].outcome < 0
                slot = (ring_pick({cur[0]}) if refused_last else min(cur[0], w), None)
            else:
                slot = (0, last_home_corner if last_home_corner is not None and last_home_corner not in closed
                        else corner_pick(None, []))
        elif home_run >= home_stay and w >= 1:
            slot = (ring_pick(set()), None)
        elif corner_run >= CORNER_STAND or cur[1] in closed:
            slot = (0, corner_pick(cur[1], []))
        else:
            slot = cur
        slots.append(slot)
        # дальше по пачке — как если бы место уже заняли
        if cur is not None and slot[0] == 0 and cur[0] == 0 and slot[1] == cur[1]:
            corner_run += 1; home_run += 1
        elif slot[0] == 0:
            corner_run = 1; home_run = home_run + 1 if cur and cur[0] == 0 else 1; out_run = 0
            last_home_corner = slot[1]
        else:
            out_run = out_run + 1 if cur and cur[0] > 0 else 1; home_run = corner_run = 0
        cur = slot
        jump = back = False
    return Plan(m, w, manual, widened, slots, leaving, back_now, why,
                closed_styles(state, style_of) if style_of is not None else frozenset())
