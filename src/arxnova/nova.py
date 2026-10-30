"""Подбор следующего трека: профиль вкуса (М3), кандидаты и ранжирование (М4–М5) с жёсткими
правилами ФТ-4.4 и ФТ-4.9 и причинами выбора (ФТ-8.2).

Поток держится стиля текущего трека; в другой стиль — только через межжанровый шлюз М6
(gate.py), не чаще одного перехода на N треков (ФТ-6.8).
"""

from __future__ import annotations

import json
import math
import random
import re
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from arxnova import gate as G
from arxnova import keys, models
from arxnova.config import Config

# Стартовые параметры (НФТ-7: позже — в TOML)
# «similar» — нажали «Похожие» на треке: хочу такое — как возврат к треку, а не пропуск
EVENT_WEIGHTS = {"like": 3.0, "replay": 2.0, "similar": 2.0, "listen": 1.0, "skip_early": -1.0, "dislike": -3.0}
# «replay» — вернулись назад к треку и дослушали: сильнее простого прослушивания, отменяет прежний пропуск
ORGANIC_BOOST = 1.5
EARLY_SKIP_PCT = 15
SESSION_LEN = 10
HALF_LIFE_DAYS = 90.0
REPEAT_HOURS = 12  # запись не повторяется в пределах N часов (ФТ-4.4)
ARTIST_GAP = 4  # исполнитель — не чаще 1 раза в K треков (ФТ-4.4)
ALBUM_GAP = 20  # альбом — не чаще 1 трека на L (ФТ-4.9)
ALBUM_PER_SESSION = 2  # и не больше A за сессию
COMPILATION_MIN_ARTISTS = 4  # от 4 разных исполнителей на 10 треков — сборник
KNN = 120
MMR_WEIGHT = 0.35  # штраф за похожесть на последние треки сессии (разнообразие)
START_POOL = 40  # среди скольких лучших по вкусу выбирается начало «Своего потока»
START_TEMP = 0.05  # мягкость выбора: чем меньше, тем сильнее перевес к самым близким
FACET_MIN_TRACKS = 50  # фильтр/сценарий показываем, если под него набирается столько записей
# Готовые полосы для плеера (ФТ-4.10): энергия — доля медиатеки, темп — BPM с half/double-time
ENERGY_BANDS = {"calm": (0.0, 0.34), "medium": (0.33, 0.67), "high": (0.66, 1.0)}
BPM_BANDS = {"slow": (60.0, 100.0), "mid": (100.0, 130.0), "fast": (130.0, 180.0)}
GATE_STEP = 3.0  # на столько перцентилей ужесточается условие шлюза за пропуск после перехода (ФТ-8.1)
GATE_TIGHTEN_MAX = 9.0
PAIR_PENALTY = 0.3  # штраф паре записей за «не тот ритм / вокал» после внутристилевого перехода (ФТ-8.5)
TEMPO_BONUS = 0.05  # М7, ФТ-7.1: совместимый темп (с удвоением/половиной)
KEY_BONUS = 0.06  # и тональность по кругу Camelot
TIRED_MULT = 2  # «надоел»: интервал повтора записи удваивается за каждый такой пропуск (ФТ-8.5)
# Память о прослушанном (владелец, 30.10.2026: «час-два в день, а назавтра то же самое при 6 тыс.
# треков»). Жёсткий запрет повтора — REPEAT_HOURS; дальше запись не запрещена, но штрафуется:
# вчерашняя почти на REPEAT_W, через полмесяца — почти ничем; часто игравшие — ещё и за частоту.
REPEAT_W = 0.3
REPEAT_HALF_LIFE_DAYS = 4.0
PLAYS_W = 0.04  # × log(1 + прослушиваний записи)
# Повтор «в тему» (владелец): штраф за недавнее снимается, если трек продолжает то, что играет
# сейчас, — по близости к последним трекам сессии он среди лучших (верхний децильный хвост
# кандидатов — без штрафа, медиана и ниже — штраф целиком). Просто «из вкуса» — уступает новому.
FIT_LO_PCT, FIT_HI_PCT = 50, 90
# Доля незнакомого в сессии = положение ползунка «знакомое ↔ новое» (1 − familiar): посередине
# половина, до упора «новое» — только неслушанное (пока оно есть), до упора «знакомое» — только
# слушанное (владелец 30.10.2026: «выкрутил на 100 — знакомых быть не должно»). Пока доли не
# хватает, неслушанные треки получают прибавку NEW_W; крайние положения — строго, отбором.
NEW_W = 0.25
NEW_STRICT = 0.99  # от этой доли — только новое; симметрично 1 − NEW_STRICT — только знакомое
# Следующий трек — не строго лучший, а случайный среди PICK_POOL лучших с перевесом к близким
# (как у старта «Своего потока»): две сессии в одном вкусе не повторяют друг друга шаг в шаг.
PICK_POOL = 6
PICK_TEMP = 0.03


def split_artists(s: str | None) -> frozenset[str]:
    s = keys.normalize(s)
    parts = re.split(r"\s*(?:&|,|;|\bfeat\.?|\bft\.?|\bx\b|\bvs\.?|\band\b)\s*", s)
    return frozenset(p for p in parts if p)


ACAPELLA_TITLE = re.compile(r"[(\[]\s*(?:a\s*c+a+p+e+l+l+a+s?|\w*apell?as?)\s*[)\]]\s*$|\ba\s+cappella\b", re.I)


def is_acapella(title: str | None, genre_tag: str | None) -> bool:
    """Вокальная дорожка без музыки — инструмент диджея (сборники «Hospital Accapellas»):
    по названию «… (Accapella)», «(Keyapella)», «A Cappella» или по тегу жанра. Модель жанров
    в ней не слышит инструментов и ставила что попало — Folk, House, Reggaeton (замечание
    владельца 26.10.2026). Сведение «инструментал + акапелла» из диджейского микса — музыка."""
    t = title or ""
    tag = (genre_tag or "").lower()
    if " + " in t or "instrumental" in t.lower():
        return False
    return bool(ACAPELLA_TITLE.search(t)) or "acap" in tag or "a cappella" in tag or "accapell" in tag


@dataclass
class TrackInfo:
    id: int
    artist: str | None
    title: str | None
    album: str | None
    artists: frozenset[str]
    album_key: str
    compilation: bool
    recording_key: bytes
    duration: float
    genre: str  # жанр Discogs по модели («Electronic»)
    style: str  # стиль Discogs по модели («Drum n Bass»)


@dataclass
class Pick:
    track_id: int
    reason: str
    score: float
    why: str = ""  # код причины для перевода в плеере: liked | similar | new | taste | bridge | restyle
    seed_title: str | None = None
    facets: list[str] = field(default_factory=list)  # мост: самые близкие фасеты (ritm, instr, vocal)


@dataclass
class Filters:
    genres: list[str] = field(default_factory=list)  # жанры Discogs
    styles: list[str] = field(default_factory=list)  # стили Discogs
    familiar: float = 0.5  # 0 — только новое, 1 — только знакомое (ФТ-4.3)
    boldness: float = 0.5  # смелость подмешивания (ФТ-6.9): сдвигает пороги шлюза
    vocal: str | None = None  # ФТ-4.10: "yes" — с вокалом, "no" — инструментал, None — не важно
    bpm: tuple[float, float] | None = None  # диапазон темпа
    bpm_exact: bool = False  # True — темп как есть (плитки «медленно/быстро»), иначе с half/double-time
    energy: tuple[float, float] | None = None  # 0–1, доля медиатеки (перцентиль энергии)
    scenario: str | None = None  # id сценария (ФТ-4.10)

    def narrow(self) -> bool:
        return bool(self.genres or self.styles or self.vocal or self.bpm or self.energy or self.scenario)


# Стили Discogs с двусмысленными названиями: без жанра «Instrumental» читается как «без вокала»,
# а это инструментальный хип-хоп (замечание владельца 23.10.2026 про Izzamuzzic)
STYLE_TITLES = {
    ("Hip Hop", "Instrumental"): "Instrumental Hip Hop",
    ("Classical", "Contemporary"): "Contemporary Classical",
    ("Classical", "Modern"): "Modern Classical",
    ("Classical", "Romantic"): "Romantic Classical",
    ("Electronic", "Experimental"): "Experimental Electronic",
    ("Rock", "Experimental"): "Experimental Rock",
    ("Jazz", "Big Band"): "Big Band Jazz",
    ("Pop", "Ballad"): "Pop Ballad",
}


NEAR = 0.85  # стили «почти вровень»: второй не меньше 85 % первого


def _album_tiebreak(scores: np.ndarray, top: np.ndarray, album_of: dict[int, str]) -> np.ndarray:
    """Трек, у которого несколько стилей почти вровень, получает тот из них, что преобладает на его
    альбоме среди уверенно размеченных треков. Иначе подпись решал случай: LTJ Bukem «Unconditional
    Love» — Instrumental 0,31 / Trip Hop 0,30 / Downtempo 0,28 — стал «Instrumental Hip Hop», хотя
    альбом — Downtempo и DnB (замечание владельца 23.10.2026)."""
    order = np.sort(scores, axis=1)
    sure = order[:, -2] < NEAR * order[:, -1]
    votes: dict[str, Counter] = {}
    for i, ak in album_of.items():
        if sure[i]:
            votes.setdefault(ak, Counter())[int(top[i])] += 1
    out = top.copy()
    for i, ak in album_of.items():
        if sure[i] or ak not in votes:
            continue
        near = np.flatnonzero(scores[i] >= NEAR * scores[i, top[i]])
        best = max(near, key=lambda j: (votes[ak][int(j)], scores[i, j]))
        if votes[ak][int(best)] > 0:
            out[i] = best
    return out


LOW = 0.2  # лучшая оценка ниже — модель не знает; подсказка — жанры MusicBrainz
MB_SYNONYMS = {"drum and bass": "Drum n Bass", "drum & bass": "Drum n Bass", "dnb": "Drum n Bass",
               "hip hop": "Boom Bap", "trip hop": "Trip Hop", "neo soul": "Neo Soul", "dubstep": "Dubstep",
               "uk garage": "UK Garage", "2-step": "UK Garage"}


def _norm_style(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower().replace("&", "n").replace(" and ", "n"))


def _mb_fallback(cfg, conn, ids, scores: np.ndarray, top: np.ndarray, labels: list[str]) -> np.ndarray:
    """Модель не уверена (лучшая оценка < LOW) — стиль из жанров MusicBrainz альбома, а если у
    альбома пусто — исполнителя: первый по голосам, который есть среди стилей модели. London
    Elektricity Big Band «Remember The Future (Live)»: оценки 0,08–0,12, первым был Neo Soul; у
    исполнителя в MusicBrainz «drum and bass» — значит DnB (замечание владельца 23.10.2026)."""
    low = scores.max(axis=1) < LOW
    if not low.any() or not getattr(cfg, "musicbrainz", False):
        return top
    from arxnova import mbgenres

    g = mbgenres.track_genres(conn)
    index = {_norm_style(lab.split("---")[1]): j for j, lab in enumerate(labels)}
    out = top.copy()
    for i in np.flatnonzero(low):
        cands = {index[k] for name in g.get(int(ids[i]), {})
                 if (k := _norm_style(MB_SYNONYMS.get(name, name))) in index}
        # из подсказанных — тот, что выше у самой модели, и не абсурдный для неё (≥ ¼ лучшей оценки):
        # ложное совпадение альбома в MusicBrainz («rock» у Deep House) так не проходит
        ok = [j for j in cands if scores[i, j] >= 0.25 * scores[i, top[i]]]
        if ok:
            out[i] = max(ok, key=lambda j: scores[i, j])
    return out


# Семейства стилей — порядок карусели: родственные рядом (замечание владельца 26.10.2026: «стили
# идут вразброс, листать неудобно»). Чего нет в таблице — семейство по жанру Discogs.
FAMILIES = {
    "dnb": ("Drum n Bass", "Liquid Funk", "Jungle", "Halftime", "Neurofunk", "Drumfunk", "Breakcore"),
    "bass": ("Dubstep", "UK Garage", "Bassline", "Grime", "Speed Garage", "2-Step", "Juke", "Footwork", "Breakbeat",
             "Breaks", "Big Beat", "Electro", "Future Bass"),
    "chill": ("Downtempo", "Ambient", "Chillwave", "Trip Hop", "Future Jazz", "Broken Beat", "Acid Jazz", "Dub",
              "Experimental Electronic", "IDM", "Leftfield", "Lo-Fi", "Ambient House", "Nu Jazz"),
    "house": ("House", "Deep House", "Tech House", "Progressive House", "Electro House", "Tropical House", "Garage House",
              "Disco", "Nu-Disco", "Techno", "Deep Techno", "Minimal", "Acid", "Trance", "Tech Trance", "Psy-Trance",
              "Progressive Trance", "Hard Trance", "Hardstyle", "Hardcore", "Happy Hardcore", "Synthwave"),
}
FAMILY_OF = {st: fam for fam, sts in FAMILIES.items() for st in sts}
GENRE_FAMILY = {"Hip Hop": "hiphop", "Rock": "rock", "Jazz": "jazz", "Funk / Soul": "soul", "Classical": "classical",
                "Pop": "pop", "Folk, World, & Country": "folk", "Reggae": "reggae", "Latin": "latin", "Blues": "blues",
                "Stage & Screen": "screen", "Vocal": "vocal"}


MAIN_STYLE_MIN = 30


def style_family(genre: str, style: str) -> str:
    return FAMILY_OF.get(style) or GENRE_FAMILY.get(genre) or "other"


DNB_FAMILY = {"Drum n Bass", "Jungle", "Halftime", "Dubstep", "Breakbeat", "Breakcore", "Neurofunk", "Drumfunk"}


def _dnb_tempo_fix(ids, styles, top, labels, bpm_of: dict, tag_of: dict) -> np.ndarray:
    """Трек на темпе DnB (165–185 по долям), в тегах файла — DnB или джангл, и модель видит
    Drum n Bass среди трёх первых стилей, а поставила не DnB (Glitch, Chiptune, Ambient,
    Downtempo, Trap…) — значит, Drum n Bass. Модель путала атмосферный и мелодичный DnB со
    спокойной музыкой: Camo & Krooked «Helios» — Chiptune, LTJ Bukem «Rhodes To Freedom» —
    Downtempo (замер 26.10.2026: 23 трека из 5939, все — DnB). Хаус-ремиксы на 125–135 BPM и
    спокойные интро на альбомах DnB не трогаются — у них не тот темп."""
    short = [lab.split("---")[-1] for lab in labels]
    if "Drum n Bass" not in short:
        return top
    dnb = short.index("Drum n Bass")
    top = top.copy()
    for i, tid in enumerate(ids):
        if short[top[i]] in DNB_FAMILY:
            continue
        b = bpm_of.get(int(tid))
        tag = (tag_of.get(int(tid)) or "").lower()
        if not b or not 165 <= b <= 185 or ("drum" not in tag and "jungle" not in tag):
            continue
        if int((styles[i] > styles[i, dnb]).sum()) < 3:
            top[i] = dnb
    return top


def _maest_styles(cfg: Config, d, effnet: np.ndarray, effnet_labels: list[str], n: int):
    """Стили — по MAEST-519, где он посчитан (точнее Discogs-EffNet); у остальных треков —
    оценки EffNet, переложенные в 519 классов по имени (все 400 стилей EffNet есть среди 519).
    Без MAEST — как было."""
    mp, hp = d / "maest.npy", d / "maest_has.npy"
    if not (mp.exists() and hp.exists() and models.is_installed(cfg.models_dir, models.MAEST)):
        return effnet, effnet_labels
    acts, has = np.load(mp), np.load(hp).astype(bool)
    if len(acts) != n or len(has) != n:
        return effnet, effnet_labels
    labels = models.maest_labels(cfg.models_dir)
    pos = {lab: i for i, lab in enumerate(labels)}
    out = np.zeros((n, len(labels)), np.float32)
    cols = [(j, pos[lab]) for j, lab in enumerate(effnet_labels) if lab in pos]
    src, dst = [c[0] for c in cols], [c[1] for c in cols]
    out[:, dst] = effnet[:, src]
    out[has] = acts[has]
    return out, labels


# Вектор похожести = звук (EffNet, центрированный) ⊕ STYLE_WEIGHT · стили (MAEST-519, без него —
# 400 стилей EffNet). На 1500 треках медиатеки владельца (30.10.2026): соседи того же стиля
# 0,53 → 0,65 при весе 0,5, исполнитель и альбом не хуже (0,224 → 0,231; 0,195 → 0,198), темп
# почти тот же (0,685 → 0,677); при весе 1,0 стиль 0,71, но исполнитель, альбом и темп уже
# проседают. Пороги шлюза перцентильные — к новому вектору они подстраиваются сами.
STYLE_WEIGHT = 0.5


def similarity_vectors(mix: np.ndarray, styles: np.ndarray, weight: float = STYLE_WEIGHT) -> np.ndarray:
    """Единичные векторы для поиска соседей: центрированный звук ⊕ взвешенные стили."""
    v = np.asarray(mix, np.float32) - np.asarray(mix, np.float32).mean(axis=0)
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
    if weight > 0 and styles is not None and len(styles) == len(v):
        s = np.asarray(styles, np.float32)
        s = s / (np.linalg.norm(s, axis=1, keepdims=True) + 1e-9)
        v = np.hstack([v, weight * s])
        v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
    return v


HUB_K = 10  # по скольким соседям считается «хабовость» трека
# Треки-«хабы» близки ко всем сразу и лезут в соседи к каждому, а 1,5 % треков не попадают в
# соседи никогда. Поправка CSLS (Conneau et al., 2018): из близости кандидата вычитается половина
# его средней близости к своим HUB_K соседям. На 1500 треках (30.10.2026): покрытие соседями
# 0,985 → 1,000, доля выдачи у 1 % самых частых 0,035 → 0,022, стиль соседей 0,645 → 0,638.
HUB_WEIGHT = 0.5


def hub_scores(vec: np.ndarray, k: int = HUB_K, block: int = 256) -> np.ndarray:
    """Средняя близость каждого трека к его k ближайшим (без себя); блоками по 256 строк (ОП-3)."""
    n = len(vec)
    out = np.zeros(n, np.float32)
    if n <= 1:
        return out
    kk = min(k, n - 1)
    for s in range(0, n, block):
        sim = vec[s:s + block] @ vec.T
        sim[np.arange(sim.shape[0]), np.arange(s, s + sim.shape[0])] = -np.inf
        out[s:s + block] = np.sort(sim, axis=1)[:, -kk:].mean(axis=1)
    return out


class Library:
    """Матрицы и метаданные в памяти. Пересобирается после индексации."""

    def __init__(self, cfg: Config, conn: sqlite3.Connection):
        self.cfg = cfg
        d = cfg.index_dir / models.EFFNET.id
        self.ids = np.load(d / "ids.npy")
        mix = np.load(d / "mix.npy").astype(np.float32)
        self.row = {int(t): i for i, t in enumerate(self.ids)}
        labels = models.style_labels(cfg.models_dir)
        styles = np.load(d / "styles.npy")
        styles, labels = _maest_styles(cfg, d, styles, labels, len(self.ids))
        self.vec = similarity_vectors(mix, styles, getattr(cfg, "style_weight", STYLE_WEIGHT))
        self.hub = hub_scores(self.vec)
        top = styles.argmax(axis=1)
        self.style_scores = styles
        self.style_labels = labels
        rows = {
            r["id"]: r
            for r in conn.execute(
                "SELECT id, artist, albumartist, album, title, date, mb_album, compilation, recording_key, duration, genre_raw FROM tracks"
            )
        }
        self.dj_tools: set[int] = set()  # акапеллы: в медиатеке есть, в поток не идут
        album_artists: dict[str, set[frozenset[str]]] = {}
        album_size: Counter[str] = Counter()
        tmp = {}
        album_of = {}
        for i, tid in enumerate(self.ids):
            r = rows[int(tid)]
            if r["mb_album"]:
                album_of[i] = "mb:" + r["mb_album"]
            else:
                album_of[i] = f"{keys.normalize(r['albumartist'] or r['artist'])}|{keys.normalize(r['album'])}|{(r['date'] or '')[:4]}"
        top = _album_tiebreak(styles, top, album_of)
        top = _mb_fallback(cfg, conn, self.ids, styles, top, labels)
        bpm_all = {r[0]: r[1] for r in conn.execute("SELECT track_id, bpm FROM track_features")}
        top = _dnb_tempo_fix(self.ids, styles, top, labels, bpm_all, {t: r["genre_raw"] for t, r in rows.items()})
        for i, tid in enumerate(self.ids):
            r = rows[int(tid)]
            ak = album_of[i]
            genre, _, style = labels[top[i]].partition("---")
            style = STYLE_TITLES.get((genre, style), style)
            if is_acapella(r["title"], r["genre_raw"]):
                genre, style = "Vocal", "Acapella"
                self.dj_tools.add(int(tid))
            arts = split_artists(r["artist"])
            album_artists.setdefault(ak, set()).add(arts)
            album_size[ak] += 1
            tmp[int(tid)] = (r, ak, genre, style, arts)
        self.tracks: dict[int, TrackInfo] = {}
        for tid, (r, ak, genre, style, arts) in tmp.items():
            n_art, n = len(album_artists[ak]), album_size[ak]
            comp = bool(r["compilation"]) or (n >= 4 and n_art * 10 >= COMPILATION_MIN_ARTISTS * n)
            self.tracks[tid] = TrackInfo(
                tid, r["artist"], r["title"], r["album"], arts, ak, comp,
                r["recording_key"], r["duration"], genre, style,
            )
        self.gate = self._load_gate(d)
        # полки, которых нет среди стилей модели (Liquid Funk), — правилом (см. shelves)
        from arxnova import shelves

        bpm_of = {r[0]: r[1] for r in conn.execute("SELECT track_id, bpm FROM track_features")}
        self.bpm_of = bpm_of  # темп — и плееру: значок играющего трека «пляшет» в такт
        style_of = {t: info.style for t, info in self.tracks.items()}
        self.shelves: dict[str, set[int]] = {}
        self.shelf_genre: dict[str, str] = {}
        from arxnova import mbgenres

        for sh in shelves.SHELVES:
            mb_al, mb_ar = mbgenres.evidence(conn, set(sh.mb_pos), set(sh.mb_neg)) if sh.mb_pos else ({}, {})
            m = shelves.members(sh, self.ids, self.style_scores, labels, style_of, bpm_of, mb_al, mb_ar)
            if m:
                self.shelves[sh.name], self.shelf_genre[sh.name] = m, sh.genre
        # сценарии (ФТ-4.10): принадлежность считается при индексации (`arxnova scenarios`)
        self.scenarios: dict[str, dict] = {}
        sp = d / "scenarios.json"
        if sp.exists():
            for sid, s in json.loads(sp.read_text()).items():
                self.scenarios[sid] = {"title": s["title"], "members": {int(t) for t in s["tracks"]}}

    def _load_gate(self, d) -> G.Gate:
        """Шлюз по фасетам, если они посчитаны (`arxnova facets`); без них межстилевые
        переходы закрыты — лучше меньше мостов, чем неуместные (КО-2)."""
        facet_mats = {}
        for f in G.FACETS:
            fp = d / f"facet_{f}.npy"
            if fp.exists():
                m = np.load(fp)
                if len(m) == len(self.ids):
                    facet_mats[f] = m
        feats, cols = None, []
        if (d / "features.npy").exists() and (d / "features.json").exists():
            feats = np.load(d / "features.npy")
            cols = json.loads((d / "features.json").read_text())["columns"]
            if len(feats) != len(self.ids):
                feats, cols = None, []
        styles = [self.tracks[int(t)].style for t in self.ids]
        forbid = {frozenset(p) for p in getattr(self.cfg, "gate_forbid", []) if len(p) == 2}
        return G.Gate(self.vec, styles, self.style_scores, facet_mats, feats, cols, forbid=forbid,
                      pct_center=getattr(self.cfg, "gate_pct", G.PCT_DEFAULT))

    def has_facets(self) -> bool:
        return bool(self.gate.has_facets.any())

    def facets(self, min_tracks: int = FACET_MIN_TRACKS, styles: list[str] | None = None) -> dict:
        """styles — числа у настроений и фильтров звука считаются внутри этих стилей (как у
        фильтра: выбран Halftime — «для работы» показывает, сколько Halftime для работы). Список
        стилей — всегда по всей медиатеке.

        Результат запоминается: медиатека между пересборками не меняется, а подсчёт по всем трекам
        держал Python — и канал через интернет в это время не отправлял пакеты (профиль сервера
        26.10.2026: две трети времени под нагрузкой — здесь, скачивание ползло)."""
        key = (min_tracks, tuple(sorted(styles)) if styles else ())
        cache = self.__dict__.setdefault("_facets_cache", {})
        if key not in cache:
            if len(cache) > 64:
                cache.clear()
            cache[key] = self._facets(min_tracks, styles)
        return cache[key]

    def _facets(self, min_tracks: int, styles: list[str] | None) -> dict:
        streamable = [t for t in self.tracks.values() if t.id not in self.dj_tools]  # акапеллы — не в поток
        g = Counter(t.genre for t in streamable)
        s = Counter((t.genre, t.style) for t in streamable)
        gt = self.gate
        inside = set(self.tracks) if not styles else \
            {t for t in self.tracks if _filter_ok(self, t, Filters(styles=list(styles)))}
        mask = np.array([int(t) in inside for t in self.ids])
        known = ~np.isnan(gt.energy) & mask
        sound = {
            "vocal": {"yes": int(((gt.vocal == 1) & mask).sum()), "no": int(((gt.vocal == 0) & mask).sum())},
            "energy": [{"name": k, "from": lo, "to": hi,
                        "tracks": int(((gt.energy >= lo) & (gt.energy <= hi) & mask).sum())}
                       for k, (lo, hi) in ENERGY_BANDS.items()],
            # темп как есть (bpm_exact): то же, что фильтр, но разом по массиву, а не трек за треком
            "bpm": [{"name": k, "from": lo, "to": hi,
                     "tracks": int(((gt.bpm >= lo) & (gt.bpm <= hi) & mask).sum())}
                    for k, (lo, hi) in BPM_BANDS.items()],
            "analysed": int(known.sum()),
        }
        scen = []
        for sid, sc in self.scenarios.items():
            n = len({self.tracks[t].recording_key for t in sc["members"] if t in inside})
            # внутри стиля стили маленькие — показываем всё, что не пусто
            if n >= (1 if styles else min_tracks):
                scen.append({"id": sid, "title": sc["title"], "tracks": n})
        for name, m in self.shelves.items():  # полка показывается как стиль рядом с остальными
            s[(self.shelf_genre[name], name)] = sum(1 for t in m if t in self.tracks)
        # одно название — один стиль (Grime у электроники и у хип-хопа); порядок — по семействам:
        # самое большое семейство первым, внутри — по числу треков
        merged: dict[str, list] = {}
        for (genre, name), n in s.items():
            cur = merged.setdefault(name, [genre, 0])
            cur[1] += n
        fam_total: Counter[str] = Counter()
        for name, (genre, n) in merged.items():
            fam_total[style_family(genre, name)] += n
        order = sorted(merged.items(), key=lambda kv: (-fam_total[style_family(kv[1][0], kv[0])],
                                                       style_family(kv[1][0], kv[0]), -kv[1][1]))
        return {
            "genres": [{"name": k, "tracks": n} for k, n in g.most_common() if n >= min_tracks],
            # main — в карусель (от 0,5 % медиатеки и не меньше 30 треков), остальные — в «Ещё…»:
            # хвост по 5–30 треков (Juke, Baroque…) мешал листать
            "styles": [{"genre": genre, "name": name, "tracks": n, "family": style_family(genre, name),
                        "main": n >= max(MAIN_STYLE_MIN, len(streamable) // 200)}
                       for name, (genre, n) in order if n >= min_tracks],
            "total": len(streamable),
            "sound": sound,
            "scenarios": scen,
            "min_tracks": min_tracks,
        }


class Profile:
    """Вкус устройства (М3): сессионный, долгосрочный и отрицательный векторы."""

    def __init__(self, lib: Library, events: list[sqlite3.Row], now_ms: int):
        dim = lib.vec.shape[1]
        self.long = np.zeros(dim, np.float32)
        self.neg = np.zeros(dim, np.float32)
        self.session = np.zeros(dim, np.float32)
        self.liked: set[int] = set()
        self.disliked: set[int] = set()
        self.last_played: dict[bytes, int] = {}  # recording_key → ts
        self.plays: Counter[bytes] = Counter()  # recording_key → сколько раз слушали (listen/replay)
        self.played: set[int] = set()
        # обратная связь шлюза (ФТ-8.1, ФТ-8.5): пара стилей → {условие: + перцентилей}
        self.pair_tighten: dict[frozenset, dict[str, float]] = {}
        self.pair_penalty: Counter[tuple[int, int]] = Counter()  # (предыдущий, пропущенный)
        self.tired: Counter[bytes] = Counter()  # recording_key → сколько раз «надоел»
        recent = []
        # предыдущий трек на том же приборе: плеер не всегда присылает prev_track_id, а без него
        # шлюз не видит, что пропуск был сразу после перехода в другой стиль (ФТ-8.1, ФТ-8.5)
        last_by_dev: dict = {}
        # вернулись и дослушали — прежние ранние пропуски этого трека не считаются отрицательными
        replayed_at: dict[int, int] = {}
        for e in events:
            if e["event_type"] == "replay" and e["track_id"] is not None:
                replayed_at[e["track_id"]] = max(replayed_at.get(e["track_id"], 0), e["ts"])
        for e in events:
            tid = e["track_id"]
            if tid is None or tid not in lib.row:
                continue
            w = self._weight(e)
            et = e["event_type"]
            if et == "like":
                self.liked.add(tid)
            elif et == "unlike":
                self.liked.discard(tid)
            elif et == "dislike":
                self.disliked.add(tid)
            elif et == "undislike":
                self.disliked.discard(tid)
            if et == "skip" and tid in replayed_at and e["ts"] <= replayed_at[tid]:
                w = 0.0  # пропуск, который потом «отменили» возвратом
            elif et == "skip":
                self._skip_feedback(lib, e, tid, last_by_dev.get(_field(e, "device_id")))
            if et in ("listen", "skip", "replay", "similar"):
                last_by_dev[_field(e, "device_id")] = tid
                self.played.add(tid)
                rk = lib.tracks[tid].recording_key
                self.last_played[rk] = max(self.last_played.get(rk, 0), e["ts"])
                if et in ("listen", "replay"):
                    self.plays[rk] += 1
                recent.append((tid, w))
            if w == 0:
                continue
            decay = 0.5 ** ((now_ms - e["ts"]) / 86_400_000 / HALF_LIFE_DAYS)
            v = lib.vec[lib.row[tid]]
            not_now = e["skip_reason"] == "not_now"
            if w > 0:
                self.long += w * decay * v
            elif not not_now:  # «не сейчас» — только сессионный сигнал (ФТ-3.3)
                self.neg += -w * decay * v
        for tid, w in recent[-SESSION_LEN:]:
            self.session += max(w, 0.2) * lib.vec[lib.row[tid]]

    def _skip_feedback(self, lib: "Library", e, tid: int, prev_seen: int | None = None) -> None:
        reason = _field(e, "skip_reason")
        early = (_field(e, "played_ratio_pct") or 0) < EARLY_SKIP_PCT
        if reason == "tired":
            self.tired[lib.tracks[tid].recording_key] += 1
            return
        prev = _field(e, "prev_track_id") or prev_seen
        if reason == "not_now" or prev is None or prev not in lib.tracks:
            return
        sa, sb = lib.tracks[prev].style, lib.tracks[tid].style
        if sa != sb:  # пропуск сразу после межстилевого перехода
            if reason in ("rhythm", "vocal"):
                conds = [reason]  # только этот фасет этой пары (ФТ-8.5)
            elif reason == "genre" or early:
                conds = ["mix", "rhythm", "instr", "vocal"]  # вся пара (ФТ-8.1)
            else:
                return
            t = self.pair_tighten.setdefault(frozenset((sa, sb)), {})
            for c in conds:
                t[c] = min(GATE_TIGHTEN_MAX, t.get(c, 0.0) + GATE_STEP)
        elif reason in ("rhythm", "vocal"):
            self.pair_penalty[(prev, tid)] += 1

    @staticmethod
    def _weight(e) -> float:
        et = e["event_type"]
        if et == "like":
            w = EVENT_WEIGHTS["like"]
        elif et == "dislike":
            w = EVENT_WEIGHTS["dislike"]
        elif et == "listen":
            w = EVENT_WEIGHTS["listen"]
        elif et in ("replay", "similar"):
            w = EVENT_WEIGHTS[et]
        elif et == "skip":
            pct = e["played_ratio_pct"] or 0
            w = EVENT_WEIGHTS["skip_early"] if pct < EARLY_SKIP_PCT else 0.0
        else:
            return 0.0
        return w * (ORGANIC_BOOST if e["is_organic"] else 1.0)


def _field(e, name: str):
    try:
        return e[name]
    except (KeyError, IndexError):
        return None


def _norm(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def _varied(scored: list, rng: random.Random, pool: int = PICK_POOL, temp: float = PICK_TEMP) -> list:
    """Порядок среди pool лучших — случайный с перевесом к более близким; остальные как есть."""
    if len(scored) <= 1:
        return scored
    head = list(scored[:pool])
    top = head[0].score
    weights = [float(np.exp((p.score - top) / temp)) for p in head]
    order = []
    while head:
        i = rng.choices(range(len(head)), weights=weights)[0]
        order.append(head.pop(i))
        weights.pop(i)
    return order + list(scored[pool:])


def _album_ok(lib: Library, tid: int, history: list[int]) -> bool:
    """ФТ-4.9: не два трека альбома подряд, не чаще 1 на ALBUM_GAP, не больше
    ALBUM_PER_SESSION за сессию; сборники под правило не попадают."""
    t = lib.tracks[tid]
    if t.compilation:
        return True
    same = [h for h in history if h in lib.tracks and lib.tracks[h].album_key == t.album_key]
    if not same:
        return True
    if len(same) >= ALBUM_PER_SESSION:
        return False
    return all(h not in history[-ALBUM_GAP:] for h in same)


def _artist_ok(lib: Library, tid: int, history: list[int], gap: int = ARTIST_GAP) -> bool:
    if gap <= 0:
        return True
    arts = lib.tracks[tid].artists
    return not any(lib.tracks[h].artists & arts for h in history[-gap:] if h in lib.tracks)


def _recording_ok(lib: Library, tid: int, history: list[int], prof: Profile, now_ms: int) -> bool:
    rk = lib.tracks[tid].recording_key
    if any(lib.tracks[h].recording_key == rk for h in history if h in lib.tracks):
        return False
    last = prof.last_played.get(rk)
    hours = REPEAT_HOURS * TIRED_MULT ** min(prof.tired.get(rk, 0), 6)  # «надоел» — реже (ФТ-8.5)
    return not last or now_ms - last > hours * 3_600_000


def _filter_ok(lib: Library, tid: int, f: Filters) -> bool:
    """Фильтр — жёсткое условие отбора (ФТ-4.10). Трек без посчитанных признаков под фильтр
    по вокалу, темпу и энергии не попадает: неизвестное — не «подходит»."""
    t = lib.tracks[tid]
    if f.styles and t.style not in f.styles and not any(tid in lib.shelves.get(x, ()) for x in f.styles):
        return False
    if f.genres and t.genre not in f.genres:
        return False
    if f.scenario:
        sc = getattr(lib, "scenarios", {}).get(f.scenario)
        if sc is None or tid not in sc["members"]:
            return False
    if f.vocal or f.bpm or f.energy:
        g, r = lib.gate, lib.row[tid]
        if f.vocal and g.vocal[r] != (1 if f.vocal == "yes" else 0):
            return False
        if f.bpm:
            b = g.bpm[r]
            mults = (1.0,) if f.bpm_exact else (1.0, 2.0, 0.5)
            if np.isnan(b) or not any(f.bpm[0] <= b * m <= f.bpm[1] for m in mults):
                return False
        if f.energy:
            e = g.energy[r]
            if np.isnan(e) or not f.energy[0] <= e <= f.energy[1]:
                return False
    return True


def filter_count(lib: Library, f: Filters) -> int:
    """Сколько записей под фильтром: меньше FACET_MIN_TRACKS — плеер честно говорит «мало треков»."""
    return len({lib.tracks[t].recording_key for t in lib.tracks if _filter_ok(lib, t, f)})


MODE_MY = "my"  # «Свой поток»: от вкуса слушателя, не от текущего трека (как «Моя волна»)
MODE_TRACK = "track"  # «Похожие на трек»: опора — выбранный трек, поток от него не уплывает


def next_tracks(
    lib: Library,
    prof: Profile,
    *,
    current: int | None,
    history: list[int],
    filters: Filters | None = None,
    n: int = 4,
    now_ms: int | None = None,
    allowed: set[int] | None = None,
    rng: random.Random | None = None,
    mode: str = MODE_MY,
    seed: int | None = None,
    gate_log: list | None = None,
    relaxed: bool = False,
    start: bool = False,
    strict_new: bool = True,
) -> list[Pick]:
    """Следующий трек и запасные (ФТ-4.6). history — треки сессии по порядку, current —
    последний; allowed — что доступно этому прибору (ФТ-4.5), None — всё.

    MODE_MY: опора — профиль вкуса (долгосрочный + сессионный вектор, лайки); что играло
    перед запуском, на выбор почти не влияет. MODE_TRACK: опора — seed, один и тот же на
    весь поток (не предыдущий трек — иначе поток цепочкой уплывает от выбранного).

    Смена стиля относительно опоры (MODE_TRACK) или последнего трека (MODE_MY) — только через
    шлюз М6; gate_log получает решения шлюза для журнала (ФТ-8.3). Если стили выбраны фильтром,
    выбор слушателя важнее шлюза."""
    f = filters or Filters()
    now_ms = now_ms or int(time.time() * 1000)
    rng = rng or random.Random()
    hist = [h for h in history if h in lib.tracks]
    if current is not None and current in lib.tracks and (not hist or hist[-1] != current):
        hist.append(current)
    if mode == MODE_TRACK and (seed is None or seed not in lib.tracks):
        seed = hist[-1] if hist else None
    anchor = seed if mode == MODE_TRACK else None

    long = _norm(prof.long) if prof.long.any() else None
    sess = _norm(prof.session) if prof.session.any() else None
    neg = _norm(prof.neg) if prof.neg.any() else None
    liked_vec = _norm(np.mean([lib.vec[lib.row[t]] for t in prof.liked if t in lib.row], axis=0)) if prof.liked else None
    # вкус: долгосрочный вектор важнее сессии; без истории — лайки; совсем без данных — нет опоры
    parts = [(0.6, long), (0.4, sess), (0.5, liked_vec)]
    taste = None
    if any(v is not None for _, v in parts):
        taste = _norm(sum(w * v for w, v in parts if v is not None))
    target = lib.vec[lib.row[anchor]] if anchor is not None else taste
    # Вкус — не точка, а область: средний вектор всех предпочтений указывает в одно место, и рядом
    # с ним всегда одни и те же треки (замечание владельца 27.10.2026). Старт «Своего потока»
    # идёт от случайной опоры внутри вкуса — одного из любимых или слушанных треков (лайкнутые
    # вдвое чаще), слегка притянутой к общему вкусу, чтобы не уйти в случайный угол.
    if start and anchor is None and taste is not None:
        pool = [t for t in prof.liked if t in lib.row] * 2 + [t for t in prof.played if t in lib.row]
        if pool:
            pivot = lib.vec[lib.row[rng.choice(pool)]]
            target = taste = _norm(0.7 * pivot + 0.3 * taste)

    # кандидаты (ФТ-4.1)
    cands: dict[int, str] = {}
    if target is not None:
        for j in np.argsort(-(lib.vec @ target - HUB_WEIGHT * lib.hub))[: KNN * 2]:
            tid = int(lib.ids[j])
            if tid != anchor:
                cands.setdefault(tid, "near" if anchor is not None else "taste")
    for tid in prof.liked:
        cands.setdefault(tid, "liked")
    if f.narrow():  # выбран стиль или фильтр — в кандидатах всё подходящее, а не случайная выборка
        for tid in lib.tracks:
            if _filter_ok(lib, tid, f):
                cands.setdefault(tid, "style")
    never = [t for t in lib.tracks if t not in prof.played]
    rng.shuffle(never)
    for tid in never[:200]:
        cands.setdefault(tid, "new")
    if not cands:
        cands = {t: "new" for t in lib.tracks}

    # от чего считается переход: выбранный трек «Похожих» или последний сыгранный
    # «Свой поток» стартует от вкуса: трек, игравший до запуска, переход не задаёт — шлюз
    # включается, когда поток уже идёт (в сессии от двух треков)
    # start — первый запрос запущенного потока: что играло до него, переход не задаёт
    ref = anchor if anchor is not None else (hist[-1] if len(hist) >= 2 and not start else None)
    ref_style = lib.tracks[ref].style if ref is not None else None
    gated = ref is not None and not (f.styles or f.genres) and not relaxed
    cross_now = G.cross_allowed_now([lib.tracks[h].style for h in hist])
    verdicts: dict[int, G.Verdict] = {}
    prev = hist[-1] if hist and (anchor is not None or not start) else None  # с чем сводится следующий трек (М7)
    recent_vecs = [lib.vec[lib.row[h]] for h in hist[-5:]]
    # уместность в текущей сессии (0…1): насколько трек продолжает последние сыгранные —
    # повтор «в тему» не штрафуется (см. FIT_LO_PCT)
    fit_of: dict[int, float] = {}
    if recent_vecs:
        ctx = _norm(np.sum(recent_vecs, axis=0))
        sims = {tid: float(lib.vec[lib.row[tid]] @ ctx) for tid in cands if tid in lib.row}
        if len(sims) > 2:
            lo, hi = np.percentile(list(sims.values()), [FIT_LO_PCT, FIT_HI_PCT])
            fit_of = {t: float(np.clip((c - lo) / (hi - lo + 1e-6), 0.0, 1.0)) for t, c in sims.items()}
    # нехватка незнакомого в этой сессии: 1 — ещё ничего нового не было, 0 — цель достигнута
    target_new = min(1.0, max(0.0, 1.0 - f.familiar))
    share_new = (sum(1 for h in hist if h not in prof.played) / len(hist)) if hist else 0.0
    new_deficit = max(0.0, target_new - share_new) / max(target_new, 0.05)
    # и обратно: нового уже больше цели — прибавку получают знакомые. Без этого при 4 % слушанной
    # медиатеки ползунок «50 %» давал 77 % нового (проверка на профиле владельца 30.10.2026)
    known_deficit = max(0.0, share_new - target_new) / max(1.0 - target_new, 0.05) if hist else 0.0
    # крайние положения ползунка — отбором, не прибавкой: только незнакомое / только знакомое,
    # если такие кандидаты вообще есть (кончились — играет что есть, см. strict_new ниже)
    if strict_new and (target_new >= NEW_STRICT or target_new <= 1 - NEW_STRICT):
        want_known = target_new <= 1 - NEW_STRICT
        keep = {t: src for t, src in cands.items() if ((t in prof.played or t in prof.liked) == want_known)}
        if keep:
            cands = keep

    scored: list[Pick] = []
    for tid, src in cands.items():
        if tid in prof.disliked or (allowed is not None and tid not in allowed):
            continue
        if tid in lib.dj_tools:  # голая вокальная дорожка посреди потока никому не нужна
            continue
        if not _filter_ok(lib, tid, f):
            continue
        bridge = None
        if gated and lib.tracks[tid].style != ref_style:
            if not cross_now:  # ФТ-6.8: переход был недавно — держимся стиля
                continue
            pair = frozenset((ref_style, lib.tracks[tid].style))
            v = lib.gate.check(lib.row[ref], lib.row[tid], boldness=f.boldness, tighten=prof.pair_tighten.get(pair))
            verdicts[tid] = v
            if not v.ok:
                continue
            bridge = v
        if not (_recording_ok(lib, tid, hist, prof, now_ms) and _album_ok(lib, tid, hist)):
            continue
        v = lib.vec[lib.row[tid]]
        s = 0.0
        hub = HUB_WEIGHT * float(lib.hub[lib.row[tid]])  # поправка CSLS к близости (см. hub_scores)
        if anchor is not None:
            s += 1.2 * (float(v @ target) - hub)
            if taste is not None:
                s += 0.3 * (float(v @ taste) - hub)
        elif taste is not None:
            s += 1.0 * (float(v @ taste) - hub)
        else:
            s += rng.random() * 0.1  # нет ни опоры, ни истории — первые треки знакомят с медиатекой
        if neg is not None:
            s -= 0.8 * float(v @ neg)
        if ref is not None and (ref, tid) in prof.pair_penalty:
            s -= PAIR_PENALTY * prof.pair_penalty[(ref, tid)]
        if prev is not None and not relaxed:  # сведение с предыдущим треком (М7)
            ra, rb = lib.row[prev], lib.row[tid]
            if G.energy_jump(lib.gate, ra, rb):
                continue  # ФТ-7.2: резкий скачок энергии
            if lib.gate.tempo_ok(ra, rb):
                s += TEMPO_BONUS
            h = G.harmonic(lib.gate, ra, rb)
            if h is not None:
                s += KEY_BONUS * h
        if recent_vecs:  # MMR: близко к стилю, но не копия последних треков
            s -= MMR_WEIGHT * max(float(v @ r) for r in recent_vecs)
        rk = lib.tracks[tid].recording_key
        last = prof.last_played.get(rk)
        if last is not None:  # память о прослушанном: недавнее и частое — реже, кроме повтора «в тему»
            days = max(0.0, (now_ms - last) / 86_400_000)
            s -= (REPEAT_W * 0.5 ** (days / REPEAT_HALF_LIFE_DAYS) * (1.0 - fit_of.get(tid, 0.0))
                  + PLAYS_W * float(np.log1p(prof.plays.get(rk, 0))))
        elif new_deficit > 0:
            s += NEW_W * new_deficit
        if last is not None and known_deficit > 0:
            s += NEW_W * known_deficit
        known = tid in prof.played or tid in prof.liked
        s += (f.familiar - 0.5) * (0.4 if known else -0.4)
        if tid in prof.liked:
            s += 0.15
        why = _why(tid, anchor, prof)
        seed_title = lib.tracks[anchor].title if anchor is not None else None
        reason = _reason(lib, tid, anchor, src, prof)
        if relaxed and ref is not None and lib.tracks[tid].style != ref_style:
            why, reason = "restyle", f"{lib.tracks[tid].style} · в «{ref_style}» больше нечего предложить"
        bridge_facets: list[str] = []
        if bridge is not None:
            why = "bridge"
            seed_title = lib.tracks[ref].title
            bridge_facets = [f for _, f in sorted(((x, f) for f, x in bridge.facets.items()), reverse=True)[:2]]
            reason = f"{lib.tracks[tid].style} · {G.bridge_reason(bridge)} с «{lib.tracks[ref].title or 'предыдущим'}»"
        scored.append(Pick(tid, reason, s, why, seed_title, bridge_facets))

    if gate_log is not None and ref is not None:
        gate_log.extend((ref, tid, v) for tid, v in verdicts.items())
    scored.sort(key=lambda p: -p.score)
    if start and anchor is None and len(scored) > n:
        # Старт «Своего потока»: строгий порядок по вкусу давал каждый раз одни и те же первые треки
        # (живая медиатека 27.10.2026: 3 разных первых трека на 10 запусков). Порядок среди лучших
        # START_POOL — случайный с перевесом к более близким: начало разное, но всё ещё «твоё».
        pool = scored[:START_POOL]
        top = pool[0].score
        weights = [math.exp((p.score - top) / START_TEMP) for p in pool]
        order = []
        while pool:
            i = rng.choices(range(len(pool)), weights=weights)[0]
            order.append(pool.pop(i))
            weights.pop(i)
        scored = order + scored[START_POOL:]
    # Разнесение исполнителей ослабляется, только если иначе тишина (узкий стиль одного
    # исполнителя): сначала «не чаще 1 на K», потом «не подряд», потом без ограничения.
    # Правила альбома и повтора записи не ослабляются никогда.
    if anchor is None and not start:
        scored = _varied(scored, rng)
    # Доля нового — отбором, а не только прибавкой: нового в сессии больше цели — сначала знакомые
    # кандидаты, меньше — сначала незнакомые (внутри каждой половины порядок по скору). Прибавка
    # одна не дотягивала: при 4 % слушанной медиатеки «50 %» давало 65–77 % нового.
    if hist and 0.0 < target_new < 1.0:
        want_known = share_new > target_new
        known_of = {p.track_id: (p.track_id in prof.played or p.track_id in prof.liked) for p in scored}
        scored = [p for p in scored if known_of[p.track_id] == want_known] + \
                 [p for p in scored if known_of[p.track_id] != want_known]
    for gap in (ARTIST_GAP, 1, 0):
        out: list[Pick] = []
        seq = list(hist)
        for p in scored:  # запасные тоже не нарушают правил друг относительно друга
            if (_artist_ok(lib, p.track_id, seq, gap) and _album_ok(lib, p.track_id, seq)
                    and _recording_ok(lib, p.track_id, seq, prof, now_ms)):
                out.append(p)
                seq.append(p.track_id)
            if len(out) >= n:
                break
        if out:
            return out
    if strict_new and not scored and (target_new >= NEW_STRICT or target_new <= 1 - NEW_STRICT):
        # строгий отбор оставил только то, что правила не пропускают — без строгости
        return next_tracks(lib, prof, current=current, history=history, filters=filters, n=n, now_ms=now_ms,
                           allowed=allowed, rng=rng, mode=mode, seed=seed, gate_log=gate_log, relaxed=relaxed,
                           start=start, strict_new=False)
    if not relaxed and mode == MODE_MY and (gated or prev is not None):
        # шлюз и сведение отсекли всё (маленький стиль исчерпан): лучше честная смена стиля,
        # чем тишина — причина говорит об этом прямо
        return next_tracks(lib, prof, current=current, history=history, filters=filters, n=n, now_ms=now_ms,
                           allowed=allowed, rng=rng, mode=mode, seed=seed, gate_log=None, relaxed=True, start=start)
    return []


def _why(tid: int, anchor: int | None, prof: "Profile") -> str:
    """Код причины — плеер сам переводит его на язык интерфейса."""
    if tid in prof.liked:
        return "liked"
    if anchor is not None:
        return "similar"
    if tid not in prof.played:
        return "new"
    return "taste"


def _reason(lib: Library, tid: int, anchor: int | None, src: str, prof: "Profile") -> str:
    """Подпись под треком: стиль и короткое человеческое «почему» (ФТ-8.2)."""
    t = lib.tracks[tid]
    if tid in prof.liked:
        why = "вы его лайкали"
    elif anchor is not None:
        why = f"похоже на «{lib.tracks[anchor].title or 'выбранный трек'}»"
    elif tid not in prof.played:
        why = "новое для вас"
    else:
        why = "по вашему вкусу"
    return f"{t.style} · {why}"


def load_events(conn: sqlite3.Connection, device_ids: int | list[int]) -> list[sqlite3.Row]:
    ids = [device_ids] if isinstance(device_ids, int) else list(device_ids)
    q = ",".join("?" * len(ids))
    return conn.execute(f"SELECT * FROM events WHERE device_id IN ({q}) ORDER BY ts", ids).fetchall()


GATE_LOG_DAYS = 90  # срок построчного журнала шлюза (ФТ-8.3)


def log_request(conn: sqlite3.Connection, device_id: int, seed: int | None, picks: list[Pick],
                gate_log: list | None = None) -> int:
    """Запрос и решения шлюза: построчно — принятые и отклонённые после близости микса,
    остальные отказы (микс, пара стилей) — счётчиками, иначе ~400 строк на шаг (ФТ-8.3)."""
    now = int(time.time() * 1000)
    counts: Counter[str] = Counter()
    rows = []
    for a, b, v in gate_log or []:
        counts["accepted" if v.ok else v.failed] += 1
        if v.ok or v.failed not in (G.COND_MIX, G.COND_PAIR):
            detail = json.dumps({"facet": v.detail, **{k: round(x, 4) for k, x in v.facets.items()}})
            rows.append((now, a, b, int(v.ok), v.failed, detail, round(v.mix, 4)))
    with conn:
        cur = conn.execute(
            "INSERT INTO nova_requests (device_id, ts, seed_track, result, gate_counts) VALUES (?,?,?,?,?)",
            (device_id, now, seed,
             json.dumps([{"track": p.track_id, "reason": p.reason, "score": round(p.score, 4)} for p in picks], ensure_ascii=False),
             json.dumps(counts) if counts else None),
        )
        if rows:
            conn.executemany(
                "INSERT INTO gate_log (request_id, ts, from_track, to_track, accepted, failed, detail, mix) VALUES (?,?,?,?,?,?,?,?)",
                [(cur.lastrowid, *r) for r in rows])
            conn.execute("DELETE FROM gate_log WHERE ts < ?", (now - GATE_LOG_DAYS * 86_400_000,))
    return cur.lastrowid
