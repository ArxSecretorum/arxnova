"""Модель слушателя (решение 0016): с какой вероятностью трек сейчас пропустят.

Вместо правил дрейфа — одна оценка. Пропуск записывается как y = 1, остальные исходы как 0:
    P(пропуск | трек, момент) = σ(β · x).
x — что модель знает о треке и моменте:
    prev_skip   прошлый трек сессии пропущен: вы ищете другое (после пропуска 55 % пропусков, после дослушанного 8 %)
    warmup      в сессии меньше двух сыгранных: начало, вы ищете своё (25 %)
    taste       близость к вашему вкусу за всё время
    stay        близость к прошлому треку, если его приняли: в потоке держаться рядом (похожий — 5 %, непохожий — 11 %)
    flee        близость к прошлому треку, если его пропустили: уходить в сторону (похожий — 68 %, непохожий — 42 %)
    repel       близость к прежним пропускам этой сессии, тающая с давностью
    fatigue     сколько похожего уже прозвучало: усталость от одного места (после 15+ треков стиля снова 27 %)
    trans       куда вы обычно шли из таких мест: после дослушанного — дальше по потоку, после пропуска — куда сбегали
    tempo_jump  скачок темпа (половинный и двойной темп — не скачок)
Числа в скобках — история владельца на 06.11.2026 (303 показа потока). Веса учатся на истории слушателя: байесовская
логистическая регрессия, априорные веса — эти наблюдения и прямая просьба владельца о плавных переходах, разброс
по Лапласу. Пока данных мало, решают априорные веса, с данными — история. Выбор трека — выборка Томпсона: веса
берутся случайно из апостериорного распределения, поэтому где модель не уверена, она пробует, а где уверена — играет
наверняка.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from arxnova import drift as D

FEATURES = ("prev_skip", "warmup", "taste", "stay", "flee", "repel", "fatigue", "trans", "tempo_jump")
# Априорные веса в стандартизованных единицах, первым — свободный член; + — к пропуску
PRIOR_MU = np.array([-1.5, 1.0, 0.5, -0.4, -0.4, 0.6, 0.3, 0.3, -0.3, 0.1])
PRIOR_SD = np.array([2.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
# Масштабы признаков, пока своих данных мало (минимум MIN_ROWS показов): по истории владельца 06.11.2026.
# Пробовал и число пропусков среди трёх последних: на той же проверке 0,69 против 0,70 без него, вес около нуля — убран
DEFAULT_MEAN = np.array([0.12, 0.18, 0.5, 0.38, 0.06, 0.05, 0.74, 0.35, 0.25])
DEFAULT_SD = np.array([0.32, 0.39, 0.24, 0.27, 0.17, 0.12, 0.88, 0.21, 0.49])
MIN_ROWS = 60
SESSION_GAP_MS = 30 * 60 * 1000  # пауза дольше — новая сессия (только для обучения: в потоке сессия — запуск)
WINDOW = 30  # сколько последних треков сессии видит модель: 0,9³⁰ ≈ 0,04, дальше вклад исчезает (с 20 усталость слабее, поток липнет к дому)
DECAY = 0.7  # прежние пропуски тают по давности
FATIGUE_DECAY = 0.9  # усталость: десять последних треков весят примерно как все
NEAR = 0.45  # «то же место», если медиатека мала для своей шкалы: у владельца соседние треки сессий похожи в среднем на 0,48, случайная пара — на 0,04
NEAR_Q, HOP_Q = 0.9, 0.7  # шкала по самой медиатеке: «то же место» — выше 90 % случайных пар, «соседнее» — выше 70 %
WARMUP = 2
PAIRS_MAX = 400  # память переходов: последние столько пар
TEMPO_SCALE = 20.0  # уд/мин на единицу признака
TEMPO_MIN_CONF = 0.3  # как у шлюза


def outcome_of(kinds: list[str]) -> int:
    """Исход показа по действиям на нём: +1 принят, −1 отказ, 0 не понять. Последний дизлайк важнее лайка,
    лайкнул и ушёл — принят (как в drift._replay)."""
    if "dislike" in kinds:
        kinds = kinds[len(kinds) - 1 - kinds[::-1].index("dislike"):]
        if "plus" not in kinds:
            return -1
    if "plus" in kinds:
        return 1
    if any(k in ("early", "mid") for k in kinds):
        return -1
    if any(k in ("done", "late") for k in kinds):
        return 1
    return 0


@dataclass
class Play:
    row: int
    outcome: int  # +1 принят, −1 отказ, 0 ещё играет или не понять


@dataclass
class Model:
    beta: np.ndarray = field(default_factory=lambda: PRIOR_MU.copy())  # MAP, первым свободный член
    cov: np.ndarray = field(default_factory=lambda: np.diag(PRIOR_SD ** 2))
    mean: np.ndarray = field(default_factory=lambda: DEFAULT_MEAN.copy())
    sd: np.ndarray = field(default_factory=lambda: DEFAULT_SD.copy())
    taste: np.ndarray | None = None  # единичный вектор вкуса: принятые треки и лайки (втрое)
    flow: list[tuple[int, int]] = field(default_factory=list)  # (откуда, куда): принят → принят
    escape: list[tuple[int, int]] = field(default_factory=list)  # пропущен → принят
    rows: int = 0
    skips: int = 0

    def sample(self, rng) -> np.ndarray:
        """Веса для одного запроса (выборка Томпсона)."""
        try:
            z = np.array([rng.gauss(0.0, 1.0) for _ in range(len(self.beta))])
            return self.beta + np.linalg.cholesky(self.cov) @ z
        except np.linalg.LinAlgError:
            return self.beta.copy()

    def logit(self, x: np.ndarray, beta: np.ndarray | None = None) -> np.ndarray:
        """Лог-шансы пропуска для строк признаков x (n × len(FEATURES))."""
        b = self.beta if beta is None else beta
        return b[0] + ((x - self.mean) / self.sd) @ b[1:]

    def p_skip(self, x: np.ndarray, beta: np.ndarray | None = None) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.logit(x, beta)))

    def weights(self) -> dict[str, float]:
        return {"bias": round(float(self.beta[0]), 3), **{k: round(float(w), 3) for k, w in zip(FEATURES, self.beta[1:])}}


def scale_of(lib) -> tuple[float, float]:
    """Пороги «то же место» и «соседнее место» по самой медиатеке: перцентили близости случайных пар. У владельца 06.11.2026 —
    около 0,48 и 0,23; на другой медиатеке векторы ложатся иначе, и постоянный порог там ничего не значит."""
    cached = getattr(lib, "_listener_scale", None)
    if cached is None:
        n = len(lib.ids)
        if n < 50:
            cached = (NEAR, HOP_FLOOR)
        else:
            rng = np.random.default_rng(0)
            i, j = rng.integers(0, n, 4000), rng.integers(0, n, 4000)
            sims = np.einsum("ij,ij->i", lib.vec[i], lib.vec[j])
            cached = (float(np.quantile(sims, NEAR_Q)), float(np.quantile(sims, HOP_Q)))
        try:
            lib._listener_scale = cached
        except AttributeError:
            pass
    return cached


def _unit(v: np.ndarray) -> np.ndarray | None:
    """Единичный вектор в одинарной точности, как векторы медиатеки: иначе numpy при умножении молча копирует всю матрицу
    кандидатов в двойную (на Pi это заметная доля запроса)."""
    n = float(np.linalg.norm(v))
    return (v / n).astype(np.float32) if n > 0 else None


class Sims:
    """Близости кандидатов к трекам: считаются один раз на запрос. Запасной ход проверяет четыре состояния «если пропустят»
    над теми же кандидатами, и без этого близости пересчитывались каждый раз (на Pi 06.11.2026 запрос шёл 158 мс против 52 у дрейфа)."""

    def __init__(self, lib, rows: np.ndarray):
        self.lib, self.rows = lib, np.asarray(rows, dtype=int)
        self.V = lib.vec[self.rows]
        self._cols: dict[int, np.ndarray] = {}
        self._taste: tuple[int, np.ndarray] | None = None

    def col(self, row: int) -> np.ndarray:
        c = self._cols.get(row)
        if c is None:
            c = self._cols[row] = self.V @ self.lib.vec[row]
        return c

    def cols(self, rows: list[int]) -> np.ndarray:
        """Близости ко многим трекам сразу: недостающие — одним умножением матриц, а не проходом на каждый (на Pi это в разы)."""
        need = [r for r in dict.fromkeys(rows) if r not in self._cols]
        if need:
            block = self.V @ self.lib.vec[need].T
            for k, r in enumerate(need):
                self._cols[r] = block[:, k]
        return np.stack([self._cols[r] for r in rows], axis=1)

    def subset(self, idx: np.ndarray) -> "Sims":
        """Часть кандидатов с уже посчитанными близостями: запасному ходу нужен только пул лучших, а не все кандидаты запроса."""
        sub = Sims.__new__(Sims)
        sub.lib, sub.rows, sub.V = self.lib, self.rows[idx], self.V[idx]
        sub._cols = {r: c[idx] for r, c in self._cols.items()}
        sub._taste = (self._taste[0], self._taste[1][idx]) if self._taste is not None else None
        return sub

    def taste(self, vec: np.ndarray) -> np.ndarray:
        if self._taste is None or self._taste[0] != id(vec):
            self._taste = (id(vec), self.V @ vec)
        return self._taste[1]


def features(lib, model: Model, state: list[Play], rows: np.ndarray, plays_in_session: int | None = None,
             sims: Sims | None = None) -> np.ndarray:
    """Признаки кандидатов (строки lib.vec) после сессии state. plays_in_session — сколько показов в сессии всего
    (state может быть хвостом); sims — близости тех же кандидатов, посчитанные раньше в этом запросе (тогда rows — его строки)."""
    sims = sims or Sims(lib, rows)
    rows = sims.rows
    x = np.zeros((len(rows), len(FEATURES)))
    if not len(rows):
        return x
    V = sims.V
    near = scale_of(lib)[0]
    col = {k: i for i, k in enumerate(FEATURES)}
    finished = sum(1 for p in state if p.outcome != 0) if plays_in_session is None else plays_in_session
    x[:, col["warmup"]] = 1.0 if finished < WARMUP else 0.0
    if model.taste is not None:
        x[:, col["taste"]] = sims.taste(model.taste)
    recent = state[-WINDOW:]
    if not recent:
        return x
    S = sims.cols([p.row for p in recent])  # n × L
    last = recent[-1]
    ages = np.arange(len(recent) - 1, -1, -1, dtype=float)
    if last.outcome < 0:
        x[:, col["prev_skip"]] = 1.0
        x[:, col["flee"]] = S[:, -1]
    else:
        x[:, col["stay"]] = S[:, -1]
    older = np.array([p.outcome < 0 for p in recent[:-1]] + [False])
    if older.any():
        x[:, col["repel"]] = np.max(S[:, older] * DECAY ** ages[older], axis=1)
    x[:, col["fatigue"]] = np.sum(np.clip((S - near) / (1.0 - near), 0.0, None) * FATIGUE_DECAY ** ages, axis=1)
    pairs = model.escape if last.outcome < 0 else model.flow
    if pairs:
        src = np.array([a for a, _ in pairs]); dst = np.array([b for _, b in pairs])
        w = np.clip(lib.vec[src] @ lib.vec[last.row] - near, 0.0, None)
        if w.sum() > 0:  # (V·Dᵀ)·w = V·(Dᵀ·w): одна близость к взвешенной точке «куда шли» вместо сотен
            x[:, col["trans"]] = V @ (lib.vec[dst].T @ w.astype(np.float32)) / float(w.sum())
    g = getattr(lib, "gate", None)
    if g is not None and getattr(g, "bpm", None) is not None:
        b0, c0 = float(g.bpm[last.row]), float(g.bpm_conf[last.row])
        if b0 > 0 and c0 >= TEMPO_MIN_CONF:
            bc, cc = g.bpm[rows], g.bpm_conf[rows]
            ok = (bc > 0) & (cc >= TEMPO_MIN_CONF)
            jump = np.min(np.abs(np.stack([bc * 0.5, bc, bc * 2.0]) - b0), axis=0) / TEMPO_SCALE
            x[ok, col["tempo_jump"]] = np.minimum(jump[ok], 3.0)
    return x


def _fit(X: np.ndarray, y: np.ndarray, mean: np.ndarray, sd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """MAP и ковариация по Лапласу: логистическая регрессия с нормальным априорным распределением."""
    Z = np.hstack([np.ones((len(X), 1)), (X - mean) / sd])
    prec = 1.0 / PRIOR_SD ** 2

    def loss(w: np.ndarray) -> float:
        a = Z @ w
        return float(np.sum(np.logaddexp(0.0, a) - y * a) + 0.5 * np.sum(prec * (w - PRIOR_MU) ** 2))

    w = PRIOR_MU.copy()
    cur = loss(w)
    for _ in range(100):
        p = 1.0 / (1.0 + np.exp(-np.clip(Z @ w, -30.0, 30.0)))
        g = Z.T @ (p - y) + prec * (w - PRIOR_MU)
        H = Z.T @ (Z * (p * (1.0 - p))[:, None]) + np.diag(prec)
        step = np.linalg.solve(H, g)
        t = 1.0
        while t > 1e-4:  # чистый Ньютон на этих данных раскачивался (06.11.2026): шаг только с убыванием потерь
            nxt = loss(w - t * step)
            if nxt <= cur:
                break
            t *= 0.5
        w, cur = w - t * step, min(cur, nxt)
        if np.max(np.abs(t * step)) < 1e-7:
            break
    p = 1.0 / (1.0 + np.exp(-np.clip(Z @ w, -30.0, 30.0)))
    H = Z.T @ (Z * (p * (1.0 - p))[:, None]) + np.diag(prec)
    return w, np.linalg.inv(H)


def training_rows(lib, events) -> tuple[np.ndarray, np.ndarray, list[int], Model]:
    """Показы потока с исходом и признаками, какими они были в момент показа: только прошлое. Возвращает и время показов,
    и модель с вкусом и памятью переходов на конец истории, но без весов."""
    m = Model()
    taste = np.zeros(lib.vec.shape[1])
    sessions: dict = {}
    marks: dict = {}
    X, y, T = [], [], []
    for e in events:
        tid = e["track_id"]
        if tid is None or tid not in lib.row:
            continue
        et, dev, ts = e["event_type"], e["device_id"], e["ts"]
        row = lib.row[tid]
        if et == "like":  # во вкус — после признаков своего показа: лайк пишется до «дослушал», иначе утечка будущего
            marks.setdefault((dev, tid), []).append("plus")
            continue
        if et not in ("listen", "skip", "replay", "dislike"):
            continue
        # «Не предлагать» во время игры плеер пишет без пропуска: это конец показа с отказом, как и при выдаче (session_state)
        s = sessions.get(dev)
        if s is None or ts - s["last"] > SESSION_GAP_MS:
            s = sessions[dev] = {"plays": [], "last": ts}
        organic = bool(e["is_organic"])
        k = D.kind(et, e["played_ratio_pct"] or 0, organic, e["skip_reason"], _get(e, "left_by"))
        kinds = marks.pop((dev, tid), []) + [k]
        o = outcome_of(kinds)
        if et != "replay" and _get(e, "nova_request_id") and not organic and o != 0:
            m.taste = _unit(taste)
            X.append(features(lib, m, s["plays"], np.array([row]))[0])
            y.append(1.0 if o < 0 else 0.0)
            T.append(ts)
        if "plus" in kinds and et != "dislike":
            taste += 3.0 * lib.vec[row]  # лайк весит как три принятых
        if o > 0:
            taste += lib.vec[row]
            prev = s["plays"][-1] if s["plays"] else None
            if prev is not None and prev.outcome != 0:
                (m.flow if prev.outcome > 0 else m.escape).append((prev.row, row))
                del m.flow[:-PAIRS_MAX], m.escape[:-PAIRS_MAX]
        s["plays"].append(Play(row, o))
        s["last"] = ts
    m.taste = _unit(taste)
    return np.array(X).reshape(-1, len(FEATURES)), np.array(y), T, m


def fit_rows(m: Model, X: np.ndarray, y: np.ndarray) -> Model:
    """Веса по строкам обучения: масштабы свои от MIN_ROWS показов, иначе по умолчанию."""
    m.rows, m.skips = len(y), int(y.sum()) if len(y) else 0
    if len(y) >= MIN_ROWS:
        m.mean = X.mean(axis=0)
        m.sd = np.where(X.std(axis=0) > 1e-6, X.std(axis=0), DEFAULT_SD)
    if len(y):
        m.beta, m.cov = _fit(X, y, m.mean, m.sd)
    return m


def fit_from_events(lib, events) -> Model:
    X, y, _, m = training_rows(lib, events)
    return fit_rows(m, X, y)




HOP_FLOOR = 0.25  # «соседнее место», если медиатека мала для своей шкалы
# Пора дальше: первые DWELL_START треков одного стиля подряд шаг маленький, дальше за DWELL_SPAN треков расширяется до соседнего места.
# История владельца 06.11.2026: с 3-го по 14-й трек стиля подряд пропущено 6–12 %, с 15-го — 27 % (4 из 15, намёк, а не доказательство)
DWELL_START, DWELL_SPAN = 12, 8


@dataclass
class Step:
    """Шаг поиска (решение 0016). Пока заходит — шаги маленькие: трек из того же места, что прошлый, а усталость сама толкает
    к краю этого места, и поток идёт. После пропуска — шаг в сторону от пропущенного, рядом с последним принятым. Пропуски
    подряд — от всех пропущенных, без привязки: поиск шире. Модель выбирает внутри разрешённого шага."""
    last: np.ndarray | None  # прошлый трек, если его приняли или он ещё играет
    fled: list[np.ndarray]  # пропущенные подряд в конце сессии
    anchor: np.ndarray | None  # последний принятый до них
    near: float = NEAR  # пока заходит: следующий из того же места (плавный переход, просьба владельца 06.11.2026); после пропуска — не ближе этого к пропущенному
    hop: float = HOP_FLOOR  # после первого пропуска: не дальше этого от последнего принятого — соседнее место, а не прыжок
    rows: tuple = ()  # строки тех же треков (прошлый, пропущенные, привязка): близости к ним берутся из Sims

    @classmethod
    def of(cls, lib, state: list[Play], dwell: int = 0) -> "Step | None":
        """dwell — сколько треков одного стиля подряд в конце сессии: чем дольше на месте, тем шире шаг."""
        if not state:
            return None
        # Серия пропусков в конце сессии. Трек без событий не в конце — подменённый запасным ходом или выбранный в списке —
        # не «принят»: серию не рвёт и привязкой не становится (независимая проверка 06.11.2026: иначе второй запасной ход подряд
        # держался рядом с треком, выбранным ещё до первого пропуска). Последний без событий — ещё играет: пока заходит.
        streak, i = [], len(state) - 1
        while i >= 0:
            p = state[i]
            if p.outcome > 0 or (p.outcome == 0 and i == len(state) - 1):
                break
            if p.outcome < 0:
                streak.append(p.row)
            i -= 1
        near, hop = scale_of(lib)
        if not streak:
            wide = min(1.0, max(0.0, (dwell - DWELL_START + 1) / DWELL_SPAN))
            return cls(lib.vec[state[-1].row], [], None, near - wide * (near - hop), hop, (state[-1].row, (), None))
        anchor = next((p.row for p in reversed(state[:i + 1]) if p.outcome > 0), None)
        return cls(None, [lib.vec[r] for r in streak], lib.vec[anchor] if anchor is not None else None, near, hop,
                   (None, tuple(streak), anchor))

    def allows_many(self, sims: "Sims") -> np.ndarray:
        """То же для всех кандидатов запроса разом, по готовым близостям."""
        last, fled, anchor = self.rows
        if last is not None:
            return sims.col(last) >= self.near
        ok = np.ones(len(sims.rows), dtype=bool)
        for r in fled:
            ok &= sims.col(r) < self.near
        if len(fled) == 1 and anchor is not None:
            ok &= sims.col(anchor) >= self.hop
        return ok

    def allows(self, v: np.ndarray) -> bool:
        if self.last is not None:
            return float(v @ self.last) >= self.near
        if any(float(v @ r) >= self.near for r in self.fled):
            return False
        if len(self.fled) == 1 and self.anchor is not None:
            return float(v @ self.anchor) >= self.hop
        return True


def session_state(lib, tracks: list[int], events_of) -> list[Play]:
    """Сессия для выбора: треки этого запуска по порядку, исходы по действиям на каждом показе (drift._by_play).
    Трек без действий — ещё играет или стоит в очереди: исход 0."""
    tracks = [t for t in tracks if t in lib.row]
    plays = D._by_play(tracks, events_of)
    out = []
    for t, actions in zip(tracks, plays):
        kinds = [D.kind(et, pct, organic, reason, left_by) for et, pct, organic, reason, left_by, _ in actions]
        out.append(Play(lib.row[t], outcome_of(kinds)))
    return out


def _get(e, name: str):
    try:
        return e[name]
    except (KeyError, IndexError):
        return None
