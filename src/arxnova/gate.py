"""Межжанровый шлюз М6 (ФТ-6.1–6.9) — ключевой модуль этапа 2.

Переход в другой стиль допускается, только если выполнены ВСЕ условия: близость полного микса
(6.1), темп с учётом half/double-time, энергия и наличие вокала (6.3), уверенность классификатора
в новом стиле (6.4), взаимное соседство среди чужих стилей (6.5), разрешённая пара стилей (6.7).
Пороги — перцентили близости к ближайшим трекам ЧУЖИХ стилей (6.6), ползунок смелости их сдвигает (6.9).

Близости фасетов Demucs по отдельности (6.2: ритм, инструменты, вокал) больше нет (решение 0014): на настоящих
переходах чужих слушателей между стилями формула по ним угадывала хуже простого микса (−0,030 AUC, дамп
ListenBrainz, 06.11.2026), а векторы занимали половину базы и 80 % времени разбора.

Почему не внутристилевые пороги, как было до 04.11.2026: похожесть внутри стиля выше почти любой
межстилевой, и с ней шлюз не открылся ни разу (медиатека владельца, 69 % DnB: 274 запроса потока за
23.10–04.11, 0 мостов; из 364 треков DnB в выборке мост не нашёлся ни у одного). Взаимное соседство
по всем стилям тоже не работало: у трека DnB все ближайшие — DnB. См. docs/decisions/0010.

«Жанр» шлюза — стиль Discogs: в домашней медиатеке почти всё — один жанр Discogs (Electronic),
а слушатель различает именно стили (Drum n Bass → Downtempo — это переход)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

K_CROSS = 100  # ФТ-6.5, 6.6: столько ближайших треков чужих стилей у каждого трека
PCT_DEFAULT = 50.0  # лучшая половина этих соседей; после разметки — из калибровки (КО-2)
BOLD_SPAN = 40.0  # ФТ-6.9: смелость 0…1 сдвигает перцентиль на ±20 вокруг калиброванного
PCT_LIMITS = (20.0, 90.0)
NEVER_BRIDGE_TO = {"Acapella"}  # голая вокальная дорожка в потоке не играет — мост в неё бессмыслен
TEMPO_TOL = 0.06  # ±6 % (с удвоением/половиной)
TEMPO_MIN_CONF = 0.3
ENERGY_TOL = 0.25  # разница в долях медиатеки (перцентильная шкала энергии)
VOCAL_FRAC = 0.2  # вокал звучит хотя бы в 20 % звучащего времени
CONF_MIN = 0.15  # активация лучшего стиля
CONF_MARGIN = 1.3  # и во столько раз выше лучшего другого стиля
ENERGY_JUMP = 0.45  # ФТ-7.2: скачок энергии больше этого — запрещён и внутри стиля
# Проба: порог только отсекает несовместимое, порядок задаёт близость микса. Строже нельзя: при 40-м перцентиле и условии на темп
# из DnB находилось 1,7 стиля на трек, поток не разведывал почти ничего (замер 05.11.2026)
FEEL_ENERGY_TOL = 0.3
FEEL_TEMPO_TOL = 0.15  # совпал темп (с половиной и удвоением) — прибавка FEEL_TEMPO_BONUS, а не условие: один трек не сводится
FEEL_TEMPO_BONUS = 0.1
# Порог близости микса для пробы. По фасетам был 20-й перцентиль; по миксу при нём из DnB достижимо 10 чужих стилей на трек
# вместо 16, и разведка сужалась до Jungle и UK Garage. При 10-м — 15 стилей, как было (замер на снимке 06.11.2026)
FEEL_PCT = 10.0

# условия по порядку проверки; коды пишутся в журнал шлюза (ФТ-8.3). «no_facets» — прежний код «звук трека не разобран»:
# остаётся тем же, чтобы старые журналы и страница калибровки читались одинаково
COND_MIX, COND_TEMPO, COND_ENERGY, COND_VOCAL = "mix", "tempo", "energy", "vocal"
COND_CONF, COND_MUTUAL, COND_PAIR, COND_UNKNOWN = "confidence", "mutual", "pair", "no_facets"


@dataclass
class Verdict:
    ok: bool
    failed: str | None = None  # код условия, на котором отказ
    mix: float = 0.0
    detail: str = ""


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
                 features: np.ndarray | None, feature_cols: list[str], vocal: np.ndarray | None = None,
                 forbid: set[frozenset[str]] | None = None, allow_only: set[frozenset[str]] | None = None,
                 pct_center: float = PCT_DEFAULT):
        """mix — нормированные векторы микса (строки как в Library); styles — стиль каждой строки. vocal — готовое наличие
        вокала (1 есть, 0 нет, −1 неизвестно): Library берёт его из доли по стему или из CLAP; без него — по доле в features."""
        n = len(mix)
        self.n = n
        self.mix = mix
        self.styles = styles
        self.forbid = forbid or set()
        self.pct_center = pct_center
        self.allow_only = allow_only
        col = {c: i for i, c in enumerate(feature_cols)}
        f = features if features is not None else np.full((n, len(feature_cols)), np.nan, np.float32)
        self.bpm = f[:, col["bpm"]] if "bpm" in col else np.full(n, np.nan)
        self.bpm_conf = f[:, col["bpm_conf"]] if "bpm_conf" in col else np.zeros(n)
        vf = f[:, col["vocal_frac"]] if "vocal_frac" in col else np.full(n, np.nan)
        # Голос есть, если звучит хотя бы в VOCAL_FRAC звучащего времени. Правило «энергия голоса меньше 5 % — отбивки, а не вокал»
        # (05.11.2026) снято: по известной правде (трек певца — вокал есть, «Instrumental» — нет; 1523 трека) оно снимало вокал
        # у 68 треков, и у 63 из них вокал есть (bench/eval_goal/vocal_truth.py, 06.11.2026)
        self.vocal = (np.asarray(vocal, int) if vocal is not None
                      else np.where(np.isnan(vf), -1, (vf >= VOCAL_FRAC).astype(int)))  # -1 — неизвестно
        self.has_features = self.vocal >= 0  # звук трека разобран; без этого межстилевой переход закрыт
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

    def _cross_sims(self, cross_sim: np.ndarray) -> dict[str, np.ndarray]:
        """Распределение близостей к ближайшим чужих стилей (ФТ-6.6): порог «лучшая половина»
        отвечает на вопрос «насколько этот трек из другого стиля действительно близок»."""
        return {"mix": cross_sim[self.cross >= 0]}

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
        if not (self.has_features[a] and self.has_features[b]):
            v.failed = COND_UNKNOWN
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
        """Проба: треки ЧУЖИХ стилей, близкие к a по звучанию, с той же подачей: тот же вокал (есть или нет) и близкая энергия,
        совпавший темп прибавляет. Соседство не нужно: это один трек. Порядок — по близости микса: на настоящих переходах
        чужих слушателей между стилями она угадывала лучше прежней формулы по ритму и голосу Demucs (решение 0014)."""
        if not self.has_features[a]:
            return []
        sim = self.mix @ self.mix[a]
        ok = self.has_features & (self.style_ids != self.style_ids[a]) & ~self.closed & (self.vocal == self.vocal[a])
        ok &= sim >= self.threshold("mix", FEEL_PCT)
        with np.errstate(invalid="ignore", divide="ignore"):
            ok &= np.abs(self.energy - self.energy[a]) <= FEEL_ENERGY_TOL
            r = self.bpm / self.bpm[a] if self.bpm[a] > 0 else np.full(self.n, np.nan)
            tempo = sum(np.abs(r * m - 1) <= FEEL_TEMPO_TOL for m in (1.0, 2.0, 0.5)) > 0
        rows = np.flatnonzero(ok)
        if not len(rows):
            return []
        feel = sim[rows] + FEEL_TEMPO_BONUS * tempo[rows]
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
    """Человеческая причина межжанрового перехода (ФТ-8.2). Раньше называла самые близкие фасеты Demucs; их больше нет (решение 0014)."""
    return "мост: близок по звучанию"
