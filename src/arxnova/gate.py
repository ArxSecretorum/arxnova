"""Межжанровый шлюз М6 (ФТ-6.1–6.9) — ключевой модуль этапа 2.

Переход в другой стиль допускается, только если выполнены ВСЕ условия: близость полного микса
(6.1), каждого фасета по отдельности (6.2 — сильный фасет слабый не компенсирует), темп с учётом
half/double-time, энергия и наличие вокала (6.3), уверенность классификатора в новом стиле (6.4),
взаимное соседство среди чужих стилей (6.5), разрешённая пара стилей (6.7). Пороги — перцентили
близости к ближайшим трекам ЧУЖИХ стилей (6.6), ползунок смелости их сдвигает (6.9).

Почему не внутристилевые пороги, как было до 04.11.2026: похожесть внутри стиля выше почти любой
межстилевой, и с ней шлюз не открылся ни разу (медиатека владельца, 69 % DnB: 274 запроса потока за
23.10–04.11, 0 мостов; из 364 треков DnB в выборке мост не нашёлся ни у одного). Взаимное соседство
по всем стилям тоже не работало: у трека DnB все ближайшие — DnB. См. docs/decisions/0010.

«Жанр» шлюза — стиль Discogs: в домашней медиатеке почти всё — один жанр Discogs (Electronic),
а слушатель различает именно стили (Drum n Bass → Downtempo — это переход)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

FACETS = ("rhythm", "instr", "vocal")
K_CROSS = 100  # ФТ-6.5, 6.6: столько ближайших треков чужих стилей у каждого трека
PCT_DEFAULT = 50.0  # лучшая половина этих соседей; после разметки — из калибровки (КО-2)
BOLD_SPAN = 40.0  # ФТ-6.9: смелость 0…1 сдвигает перцентиль на ±20 вокруг калиброванного
PCT_LIMITS = (20.0, 90.0)
NEVER_BRIDGE_TO = {"Acapella"}  # голая вокальная дорожка в потоке не играет — мост в неё бессмыслен
TEMPO_TOL = 0.06  # ±6 % (с удвоением/половиной)
TEMPO_MIN_CONF = 0.3
ENERGY_TOL = 0.25  # разница в долях медиатеки (перцентильная шкала энергии)
VOCAL_FRAC = 0.2  # вокал звучит хотя бы в 20 % звучащего времени
VOCAL_RATIO = 0.05
CONF_MIN = 0.15  # активация лучшего стиля
CONF_MARGIN = 1.3  # и во столько раз выше лучшего другого стиля
ENERGY_JUMP = 0.45  # ФТ-7.2: скачок энергии больше этого — запрещён и внутри стиля
# Проба: порог только отсекает несовместимое, порядок задаёт близость. Строже нельзя: при 40-м перцентиле и условии на темп
# из DnB находилось 1,7 стиля на трек, поток не разведывал почти ничего (замер 05.11.2026); теперь 7
FEEL_ENERGY_TOL = 0.3
FEEL_TEMPO_TOL = 0.15  # совпал темп (с половиной и удвоением) — прибавка FEEL_TEMPO_BONUS, а не условие: один трек не сводится
FEEL_TEMPO_BONUS = 0.1
FEEL_PCT = 20.0

# условия по порядку проверки; коды пишутся в журнал шлюза (ФТ-8.3)
COND_MIX, COND_FACET, COND_TEMPO, COND_ENERGY, COND_VOCAL = "mix", "facet", "tempo", "energy", "vocal"
COND_CONF, COND_MUTUAL, COND_PAIR, COND_NOFACETS = "confidence", "mutual", "pair", "no_facets"


@dataclass
class Verdict:
    ok: bool
    failed: str | None = None  # код условия, на котором отказ
    mix: float = 0.0
    facets: dict[str, float] = field(default_factory=dict)  # близость по фасетам
    detail: str = ""


def _normed(m: np.ndarray) -> np.ndarray:
    m = m.astype(np.float32).copy()
    live = np.abs(m).sum(axis=1) > 0
    if live.any():
        m[live] -= m[live].mean(axis=0)
    m /= np.linalg.norm(m, axis=1, keepdims=True) + 1e-9
    m[~live] = 0
    return m


def _cross_neighbors(v: np.ndarray, style_ids: np.ndarray, k: int, closed: np.ndarray | None = None,
                     block: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Ближайшие k треков ЧУЖИХ стилей блоками ≤ 256 строк, индексы int32 (ФТ-2.7). Сверху ближние;
    где чужих меньше k — индекс −1. closed — куда мост не ведёт. Возвращает (индексы, близости)."""
    n = len(v)
    k = min(k, max(n - 1, 0))
    idx = np.full((n, k), -1, np.int32)
    sim = np.full((n, k), -np.inf, np.float32)
    if k == 0:
        return idx, sim
    for s in range(0, n, block):
        sims = v[s:s + block] @ v.T
        sims[style_ids[s:s + block, None] == style_ids[None, :]] = -np.inf
        if closed is not None:
            sims[:, closed] = -np.inf
        top = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        vals = np.take_along_axis(sims, top, axis=1)
        order = np.argsort(-vals, axis=1)
        top, vals = np.take_along_axis(top, order, axis=1), np.take_along_axis(vals, order, axis=1)
        top[~np.isfinite(vals)] = -1
        idx[s:s + block], sim[s:s + block] = top, vals
    return idx, sim


class Gate:
    def __init__(self, mix: np.ndarray, styles: list[str], style_scores: np.ndarray,
                 facet_mats: dict[str, np.ndarray], features: np.ndarray | None, feature_cols: list[str],
                 forbid: set[frozenset[str]] | None = None, allow_only: set[frozenset[str]] | None = None,
                 pct_center: float = PCT_DEFAULT):
        """mix — нормированные векторы микса (строки как в Library); styles — стиль каждой строки."""
        n = len(mix)
        self.n = n
        self.mix = mix
        self.styles = styles
        self.forbid = forbid or set()
        self.pct_center = pct_center
        self.allow_only = allow_only
        self.facet = {f: _normed(facet_mats[f]) for f in FACETS if f in facet_mats}
        self.has_facets = np.zeros(n, bool)
        if len(self.facet) == len(FACETS):
            live = [np.abs(self.facet[f]).sum(axis=1) > 0 for f in FACETS]
            self.has_facets = np.all(live, axis=0) if live else np.ones(n, bool)
        col = {c: i for i, c in enumerate(feature_cols)}
        f = features if features is not None else np.full((n, len(feature_cols)), np.nan, np.float32)
        self.bpm = f[:, col["bpm"]] if "bpm" in col else np.full(n, np.nan)
        self.bpm_conf = f[:, col["bpm_conf"]] if "bpm_conf" in col else np.zeros(n)
        vf = f[:, col["vocal_frac"]] if "vocal_frac" in col else np.full(n, np.nan)
        vr = f[:, col["vocal_ratio"]] if "vocal_ratio" in col else np.full(n, 1.0)
        # Голос есть, если он звучит заметную часть времени и слышен: при энергии голоса меньше 5 % это отбивки, а не вокал
        # (775 треков: по CLAP они ближе к инструменталам, замер 05.11.2026)
        self.vocal = np.where(np.isnan(vf), -1, ((vf >= VOCAL_FRAC) & ~((vr > 0) & (vr < VOCAL_RATIO))).astype(int))  # -1 — неизвестно
        self.energy = self._energy(f, col)
        # уверенность классификатора (ФТ-6.4): лучший стиль заметно выше лучшего другого
        sc = np.sort(style_scores, axis=1)
        self.confident = (sc[:, -1] >= CONF_MIN) & (sc[:, -1] >= CONF_MARGIN * sc[:, -2])
        # ближайшие чужих стилей (ФТ-6.5, 6.6) — раз при сборке медиатеки
        names = {s: i for i, s in enumerate(sorted(set(styles)))}
        self.style_ids = np.fromiter((names[s] for s in styles), np.int32, n)
        closed = np.fromiter((s in NEVER_BRIDGE_TO for s in styles), bool, n)
        self.closed = closed
        self.cross, cross_sim = _cross_neighbors(mix, self.style_ids, K_CROSS, closed)
        self._cross_sorted = np.sort(self.cross, axis=1)  # для быстрой проверки «b среди соседей a»
        self._pool = self._cross_sims(cross_sim)
        self._thresholds: dict[tuple[str, float], float] = {}

    @staticmethod
    def _energy(f: np.ndarray, col: dict[str, int]) -> np.ndarray:
        """Энергия 0–1: перцентиль суммы z-оценок громкости и плотности ударов по медиатеке."""
        if "loudness_db" not in col or "onset_rate" not in col:
            return np.full(len(f), np.nan)
        loud, ons = f[:, col["loudness_db"]], f[:, col["onset_rate"]]
        ok = ~(np.isnan(loud) | np.isnan(ons))
        e = np.full(len(f), np.nan)
        if ok.sum() < 2:
            return e
        z = (loud[ok] - loud[ok].mean()) / (loud[ok].std() + 1e-9) + (ons[ok] - ons[ok].mean()) / (ons[ok].std() + 1e-9)
        srt = np.sort(z)  # средний ранг: у равных треков — одинаковая энергия
        e[ok] = (np.searchsorted(srt, z, "left") + np.searchsorted(srt, z, "right")) / 2 / ok.sum()
        return e

    @staticmethod
    def pair_sims(m: np.ndarray, a: np.ndarray, b: np.ndarray, chunk: int = 4096) -> np.ndarray:
        """Близости пар строк (a[i], b[i]) кусками: m[a] целиком — это копия матрицы на каждую
        пару (80 000 пар × 1680 = 537 МБ, и вторая такая же для b) — на Pi с потолком памяти
        сервер падал с MemoryError при сборке медиатеки (30.10.2026), и список папок не открывался."""
        out = np.empty(len(a), np.float32)
        for s in range(0, len(a), chunk):
            out[s:s + chunk] = np.einsum("ij,ij->i", m[a[s:s + chunk]], m[b[s:s + chunk]])
        return out

    def _cross_sims(self, cross_sim: np.ndarray) -> dict[str, np.ndarray]:
        """Распределение близостей к ближайшим чужих стилей (ФТ-6.6): порог «лучшая половина»
        отвечает на вопрос «насколько этот трек из другого стиля действительно близок»."""
        keep = self.cross >= 0
        a = np.repeat(np.arange(self.n), keep.sum(axis=1))
        b = self.cross[keep]
        pool = {"mix": cross_sim[keep]}
        both = self.has_facets[a] & self.has_facets[b]
        for f, m in self.facet.items():
            pool[f] = self.pair_sims(m, a[both], b[both])
        return pool

    def pct(self, boldness: float) -> float:
        b = min(1.0, max(0.0, boldness))
        return min(PCT_LIMITS[1], max(PCT_LIMITS[0], self.pct_center + (0.5 - b) * BOLD_SPAN))

    def threshold(self, kind: str, pct: float) -> float:
        """Считается раз на перцентиль: check вызывается сотни раз на запрос потока."""
        key = (kind, float(pct))
        if key not in self._thresholds:
            d = self._pool.get(kind)
            self._thresholds[key] = float(np.percentile(d, pct)) if d is not None and len(d) else float("inf")
        return self._thresholds[key]

    def is_cross_neighbor(self, a: int, b: int) -> bool:
        row = self._cross_sorted[a]
        i = int(np.searchsorted(row, b))
        return i < len(row) and int(row[i]) == b

    def pair_allowed(self, sa: str, sb: str) -> bool:
        p = frozenset((sa, sb))
        if p in self.forbid:
            return False
        return self.allow_only is None or p in self.allow_only

    def check(self, a: int, b: int, *, boldness: float = 0.5, tighten: dict[str, float] | None = None,
              pct: float | None = None) -> Verdict:
        """a, b — строки медиатеки. tighten — ужесточение перцентиля по условию для этой пары
        стилей (ФТ-8.1, ФТ-8.5): {"mix": +3, "rhythm": +5, …}."""
        tighten = tighten or {}
        base = pct if pct is not None else self.pct(boldness)
        v = Verdict(False)
        v.mix = float(self.mix[a] @ self.mix[b])
        if not self.pair_allowed(self.styles[a], self.styles[b]):
            v.failed = COND_PAIR
            return v
        if v.mix < self.threshold("mix", min(99.5, base + tighten.get("mix", 0))):
            v.failed = COND_MIX
            return v
        if not (self.has_facets[a] and self.has_facets[b]):
            v.failed = COND_NOFACETS
            return v
        both_vocal = self.vocal[a] == 1 and self.vocal[b] == 1
        for f in FACETS:
            s = float(self.facet[f][a] @ self.facet[f][b])
            v.facets[f] = s
        for f in FACETS:
            if f == "vocal" and not both_vocal:
                continue  # у инструменталов вокал не сравнивается (наличие — ниже, ФТ-6.3)
            if v.facets[f] < self.threshold(f, min(99.5, base + tighten.get(f, 0))):
                v.failed, v.detail = COND_FACET, f
                return v
        if not self.tempo_ok(a, b):
            v.failed = COND_TEMPO
            return v
        ea, eb = self.energy[a], self.energy[b]
        if np.isnan(ea) or np.isnan(eb) or abs(ea - eb) > ENERGY_TOL:
            v.failed = COND_ENERGY
            return v
        if self.vocal[a] < 0 or self.vocal[a] != self.vocal[b]:
            v.failed = COND_VOCAL
            return v
        if not self.confident[b]:  # куда идём: стиль нового трека должен быть определён уверенно
            v.failed = COND_CONF
            return v
        if not (self.is_cross_neighbor(a, b) and self.is_cross_neighbor(b, a)):
            v.failed = COND_MUTUAL
            return v
        v.ok = True
        return v

    def feel_candidates(self, a: int, k: int = 40) -> list[tuple[int, float]]:
        """Проба: треки ЧУЖИХ стилей, близкие к a по динамике и подаче, а не по жанру: энергия, наличие вокала, ритм и голос
        (у инструменталов — звучание), совпавший темп прибавляет. Близость всего микса и соседство не нужны: это один трек."""
        if not (self.has_facets[a] and len(self.facet) == len(FACETS)) or self.vocal[a] < 0:
            return []
        rhythm = self.facet["rhythm"] @ self.facet["rhythm"][a]
        second = "vocal" if self.vocal[a] == 1 else "instr"
        delivery = self.facet[second] @ self.facet[second][a]
        ok = self.has_facets & (self.style_ids != self.style_ids[a]) & ~self.closed & (self.vocal == self.vocal[a])
        ok &= rhythm >= self.threshold("rhythm", FEEL_PCT)
        ok &= delivery >= self.threshold(second, FEEL_PCT)
        with np.errstate(invalid="ignore", divide="ignore"):
            ok &= np.abs(self.energy - self.energy[a]) <= FEEL_ENERGY_TOL
            r = self.bpm / self.bpm[a] if self.bpm[a] > 0 else np.full(self.n, np.nan)
            tempo = sum(np.abs(r * m - 1) <= FEEL_TEMPO_TOL for m in (1.0, 2.0, 0.5)) > 0
        rows = np.flatnonzero(ok)
        if not len(rows):
            return []
        feel = 0.6 * rhythm[rows] + 0.4 * delivery[rows] + FEEL_TEMPO_BONUS * tempo[rows]
        top = np.argsort(-feel)[:k]
        return [(int(rows[i]), float(feel[i])) for i in top]

    def tempo_ok(self, a: int, b: int) -> bool:
        ta, tb = self.bpm[a], self.bpm[b]
        if np.isnan(ta) or np.isnan(tb) or ta <= 0 or tb <= 0:
            return False
        if self.bpm_conf[a] < TEMPO_MIN_CONF or self.bpm_conf[b] < TEMPO_MIN_CONF:
            return False
        r = tb / ta
        return any(abs(r * m - 1) <= TEMPO_TOL for m in (1.0, 2.0, 0.5))


def energy_jump(g: "Gate", a: int, b: int) -> bool:
    ea, eb = g.energy[a], g.energy[b]
    return not (np.isnan(ea) or np.isnan(eb)) and abs(ea - eb) > ENERGY_JUMP


def without_islands(styles: list[str]) -> list[str]:
    """Одиночный подмес посреди стиля (A A X A A) — не смена стиля: поток шёл и идёт в A."""
    return [s for i, s in enumerate(styles)
            if not (0 < i < len(styles) - 1 and styles[i - 1] == styles[i + 1] != s)]


def bridge_reason(v: Verdict) -> str:
    """Человеческая причина межжанрового перехода (ФТ-8.2): самые близкие фасеты."""
    names = {"rhythm": "по ритму", "instr": "по звучанию", "vocal": "по вокалу"}
    best = sorted(((s, f) for f, s in v.facets.items() if f in names), reverse=True)[:2]
    return "мост: близок " + " и ".join(names[f] for _, f in best) if best else "мост"
