"""Межжанровый шлюз М6 (ФТ-6.1–6.9) — ключевой модуль этапа 2.

Переход в другой стиль допускается, только если выполнены ВСЕ условия: близость полного микса
(6.1), каждого фасета по отдельности (6.2 — сильный фасет слабый не компенсирует), темп с учётом
half/double-time, энергия и наличие вокала (6.3), уверенность классификатора (6.4), взаимное
соседство (6.5), разрешённая пара стилей (6.7). Пороги — перцентили внутристилевых соседей
медиатеки (6.6), ползунок смелости сдвигает их в откалиброванном диапазоне (6.9).

«Жанр» шлюза — стиль Discogs: в домашней медиатеке почти всё — один жанр Discogs (Electronic),
а слушатель различает именно стили (Drum n Bass → Downtempo — это переход)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

FACETS = ("rhythm", "instr", "vocal")
K_MUTUAL = 25  # ФТ-6.5: B в топ-K соседей A и наоборот
INTRA_FULL = 400  # стиль до стольких треков — все пары; больше — выборка
INTRA_SAMPLE = 80_000  # пар из большого стиля
PCT_DEFAULT = 90.0  # «не хуже верхних 10 % внутристилевых»; после разметки — из калибровки (КО-2)
BOLD_SPAN = 14.0  # ФТ-6.9: смелость 0…1 сдвигает перцентиль на ±7 вокруг калиброванного
PCT_LIMITS = (75.0, 99.0)
TEMPO_TOL = 0.06  # ±6 % (с удвоением/половиной)
TEMPO_MIN_CONF = 0.3
ENERGY_TOL = 0.25  # разница в долях медиатеки (перцентильная шкала энергии)
VOCAL_FRAC = 0.2  # вокал звучит хотя бы в 20 % звучащего времени
CONF_MIN = 0.15  # активация лучшего стиля
CONF_MARGIN = 1.3  # и во столько раз выше лучшего другого стиля
ENERGY_JUMP = 0.45  # ФТ-7.2: скачок энергии больше этого — запрещён и внутри стиля
KEY_MIN_CONF = 0.05  # тональность ниже этой уверенности в сведении не учитывается
N_CROSS = 8  # ФТ-6.8: не больше одного перехода на N треков
M_STICK = 4  # после перехода — M треков в новом стиле

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


def _topk(v: np.ndarray, k: int, block: int = 256) -> np.ndarray:
    """Топ-K соседей по косинусу блоками ≤ 256 строк, индексы int32 (ФТ-2.7)."""
    n = len(v)
    k = min(k, n - 1)
    out = np.zeros((n, max(k, 0)), np.int32)
    for s in range(0, n, block):
        sims = v[s:s + block] @ v.T
        for i in range(len(sims)):
            sims[i, s + i] = -np.inf
        out[s:s + block] = np.argpartition(-sims, k - 1, axis=1)[:, :k] if k > 0 else out[s:s + block]
    return out


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
        self.vocal = np.where(np.isnan(vf), -1, (vf >= VOCAL_FRAC).astype(int))  # -1 — неизвестно
        self.key = f[:, col["musical_key"]] if "musical_key" in col else np.full(n, np.nan)
        self.key_conf = f[:, col["key_conf"]] if "key_conf" in col else np.zeros(n)
        self.energy = self._energy(f, col)
        # уверенность классификатора (ФТ-6.4): лучший стиль заметно выше лучшего другого
        sc = np.sort(style_scores, axis=1)
        self.confident = (sc[:, -1] >= CONF_MIN) & (sc[:, -1] >= CONF_MARGIN * sc[:, -2])
        # взаимное соседство по миксу (ФТ-6.5), считается раз при сборке медиатеки
        nn = _topk(mix, K_MUTUAL)
        self.neighbors = [set(map(int, r)) for r in nn]
        # распределение близостей внутристилевых соседей (ФТ-6.6)
        self._intra = self._intra_sims()

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

    def _intra_sims(self) -> dict[str, np.ndarray]:
        """Распределение близостей пар внутри стиля (ФТ-6.6): «межстилевой сосед не хуже верхних
        10 % внутристилевых» — все пары стиля, а не только ближайшие соседи (так шлюз закрывался
        совсем: 0 мостов на медиатеке владельца). В больших стилях — случайная выборка пар."""
        rng = np.random.default_rng(7)
        by_style: dict[str, list[int]] = {}
        for i, s in enumerate(self.styles):
            by_style.setdefault(s, []).append(i)
        acc: dict[str, list[np.ndarray]] = {"mix": [], **{f: [] for f in self.facet}}
        for rows in by_style.values():
            n = len(rows)
            if n < 3:
                continue
            idx = np.array(rows)
            if n <= INTRA_FULL:
                a, b = np.triu_indices(n, 1)
            else:
                a, b = rng.integers(0, n, INTRA_SAMPLE), rng.integers(0, n, INTRA_SAMPLE)
                keep = a != b
                a, b = a[keep], b[keep]
            a, b = idx[a], idx[b]
            acc["mix"].append(self.pair_sims(self.mix, a, b))
            for f, m in self.facet.items():
                both = self.has_facets[a] & self.has_facets[b]
                acc[f].append(self.pair_sims(m, a[both], b[both]))
        return {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in acc.items()}

    def pct(self, boldness: float) -> float:
        b = min(1.0, max(0.0, boldness))
        return min(PCT_LIMITS[1], max(PCT_LIMITS[0], self.pct_center + (0.5 - b) * BOLD_SPAN))

    def threshold(self, kind: str, pct: float) -> float:
        d = self._intra.get(kind)
        return float(np.percentile(d, pct)) if d is not None and len(d) else float("inf")

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
        if not (self.confident[a] and self.confident[b]):
            v.failed = COND_CONF
            return v
        if b not in self.neighbors[a] or a not in self.neighbors[b]:
            v.failed = COND_MUTUAL
            return v
        v.ok = True
        return v

    def tempo_ok(self, a: int, b: int) -> bool:
        ta, tb = self.bpm[a], self.bpm[b]
        if np.isnan(ta) or np.isnan(tb) or ta <= 0 or tb <= 0:
            return False
        if self.bpm_conf[a] < TEMPO_MIN_CONF or self.bpm_conf[b] < TEMPO_MIN_CONF:
            return False
        r = tb / ta
        return any(abs(r * m - 1) <= TEMPO_TOL for m in (1.0, 2.0, 0.5))


def camelot_pos(key: int) -> tuple[int, str]:
    from arxnova.facets import camelot

    c = camelot(int(key))
    return int(c[:-1]), c[-1]


def harmonic(g: "Gate", a: int, b: int) -> float | None:
    """ФТ-7.1, круг Camelot: 1 — та же клетка, 0,8 — соседняя или параллельная, 0 — далеко;
    None — тональность не определена уверенно."""
    if np.isnan(g.key[a]) or np.isnan(g.key[b]) or g.key_conf[a] < KEY_MIN_CONF or g.key_conf[b] < KEY_MIN_CONF:
        return None
    (na, la), (nb, lb) = camelot_pos(g.key[a]), camelot_pos(g.key[b])
    if (na, la) == (nb, lb):
        return 1.0
    step = min((na - nb) % 12, (nb - na) % 12)
    if (la == lb and step == 1) or (la != lb and step == 0):
        return 0.8
    return 0.0


def energy_jump(g: "Gate", a: int, b: int) -> bool:
    ea, eb = g.energy[a], g.energy[b]
    return not (np.isnan(ea) or np.isnan(eb)) and abs(ea - eb) > ENERGY_JUMP


def cross_allowed_now(styles_hist: list[str]) -> bool:
    """ФТ-6.8: не больше одного перехода на N_CROSS треков и закрепление на M_STICK после него.
    styles_hist — стили сыгранных треков по порядку."""
    last = [s for s in styles_hist if s]
    if len(last) < 2:
        return True
    changes = [i for i in range(1, len(last)) if last[i] != last[i - 1]]
    if not changes:
        return True
    since = len(last) - changes[-1]  # треков в текущем стиле после последнего перехода
    return since >= max(M_STICK, N_CROSS)


def bridge_reason(v: Verdict) -> str:
    """Человеческая причина межжанрового перехода (ФТ-8.2): самые близкие фасеты."""
    names = {"rhythm": "по ритму", "instr": "по звучанию", "vocal": "по вокалу"}
    best = sorted(((s, f) for f, s in v.facets.items() if f in names), reverse=True)[:2]
    return "мост: близок " + " и ".join(names[f] for _, f in best) if best else "мост"
