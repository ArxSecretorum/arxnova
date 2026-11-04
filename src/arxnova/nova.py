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
TIRED_HOURS = 12  # «надоел»: запись не звучит TIRED_HOURS × 2^раз (ФТ-8.5) — запрет только по прямому сигналу
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
# треков»): недавняя запись штрафуется — вчерашняя почти на REPEAT_W, через полмесяца почти ничем;
# часто игравшие — ещё и за частоту. Общего запрета по часам больше нет — его заменила очередь ниже.
# Очередь знакомого (владелец 31.10.2026: «то, что прослушано, встаёт в очередь; выдача — по стилю, но
# сначала старое»; решение 0006). Среди ROT_POOL лучших по звучанию знакомых первым идёт самое
# просроченное: дни с последнего звучания / интервал трека. Интервал: лайк укорачивает вдвое, частые
# прослушивания — до PLAYS_INTERVAL_MIN (не сильнее: лайков мало, им нельзя занять поток), пропуски
# удлиняют, «надоел» удваивает. Подходящее к тому, что играет сейчас, — чуть вперёд (повтор «в тему»).
ROT_POOL = 24
LIKED_INTERVAL = 0.5
PLAYS_INTERVAL_MIN = 0.7
ROT_FIT_W = 0.5
ROT_NEVER_DAYS = 365.0  # лайкнутое, но ни разу не звучавшее, — давно просрочено
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
# Ползунок масштабируется под медиатеку и темп (владелец 04.11.2026), без постоянных «раз в N дней».
# K — знакомых треков, P — дослушиваний в день, s — доля знакомого на ползунке, aᵢ — вес трека
# (лайк выше, пропускаемый ниже), ā — средний вес. Тогда:
#   цикл      C  = K / (P·s)            — за сколько дней при этом темпе прозвучит всё знакомое;
#   интервал  Tᵢ = C · ā / aᵢ           — сумма 1/Tᵢ равна P·s: темпы складываются ровно в долю ползунка;
#   очередь   uᵢ = возраст / Tᵢ         — первым звучит самый просроченный (uᵢ = 1 — точно в срок);
#   пауза     uᵢ выше REST_FLOOR-квантиля пула — самые свежие REST_FLOOR пула ждут, пока есть кем заменить.
#             По рангу, а не по абсолюту: в молодом пуле всё звучало недавно, и порог «четверть интервала»
#             не прошёл бы никто (при 80 % нового выходило 98 %).
# Маленький пул: s ≤ K / (P·Cₘᵢₙ(L)) — цикл не короче разумного для медиатеки из L треков, Cₘᵢₙ растёт
# с логарифмом L. Всё пересчитывается при каждой сборке профиля: пул растёт — интервалы растягиваются сами.
PACE_DAYS = 14  # темп — дослушиваний в день за столько последних дней
REST_MIN_DAYS, REST_MAX_DAYS = 0.5, 365.0
REST_FLOOR = 0.25  # доля самых свежих в пуле знакомого, что ждёт своей очереди
# «В конце концов я ознакомлюсь с 100 % медиатеки — что будет новым?» Новое — относительно: сначала
# неслушанное, а когда его меньше, чем на NEW_POOL_DAYS при этом темпе, — самое забытое: меньше всего и
# давнее всего слушанное. Правая часть ползунка не пустеет никогда.
NEW_POOL_DAYS = 30
NEW_POOL_MIN = 30
PACE_MIN_DAYS = 3  # один длинный вечер — ещё не темп
# Доля держится как в сигма-дельта-модуляторе: копится не «сколько нового в этой сессии», а ошибка за всю
# недавнюю историю, с утечкой. Выбор каждого места по знаку ошибки: e = S − s·N, где S = Σ wₖ·xₖ, N = Σ wₖ,
# xₖ — был ли k-й от конца трек новым, wₖ = 2^(−k/H). Пока доля выполнима, |S/N − s| ≤ 1/N: ошибка вчерашнего
# дня отрабатывается сегодня, а не забывается. H — сутки прослушивания: столько же «помнит» и смена ползунка.
# Утечка и по времени: wₖ·2^(−возраст/τ) — долг месячной давности никого не касается.
SHARE_MEMORY_MIN = 12  # не короче стольких треков (H)
SHARE_MEMORY_DAYS = 3.0  # τ
NEW_AS_MASS = 0.515  # знакомость 0,3 — середина между FAM_NEW и FAM_KNOWN, та же граница, что newness ≥ 0,5
NEW_STRICT = 0.99  # от этой доли — только новое; симметрично 1 − NEW_STRICT — только знакомое
# Ступени знакомости (владелец 31.10.2026: «умнее алгоритмы и больше ступеней градации»). Знакомость
# трека — 0…1: лайк — целиком; каждое прослушивание прибавляет, а со временем знакомость тает
# (FAM_HALF_LIFE_DAYS). Пропуск её не даёт: пропущенное — не «знакомое». Одно свежее прослушивание —
# знакомый трек, одно ~10 месяцев назад — снова новый; между FAM_NEW и FAM_KNOWN — наполовину.
FAM_HALF_LIFE_DAYS = 180.0
FAM_KNOWN = 0.4
FAM_NEW = 0.2
LATE_SKIP_PCT = 60  # пропуск после 60 % — почти дослушал: половина прослушивания для знакомости
# Ступени пропуска: чем раньше пропустил, тем сильнее трек уходит вниз — штраф самому треку, не запрет;
# тает за NEG_HALF_LIFE_DAYS. Дизлайк — по-прежнему запрет: прямое «не предлагай» (решение 0003).
SKIP_STEPS = ((10, -1.0), (30, -0.6), (60, -0.25), (85, -0.1))  # (дослушал меньше %, вес)
NEG_HALF_LIFE_DAYS = 60.0
SKIP_TRACK_W = 0.5  # предел штрафа треку за пропуски
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
BRIDGE_AWAY_MAX = 2  # столько чужих стилей подряд, потом мост ведёт в любимый

# Настрой сессии (владелец 04.11.2026): пропуски и дизлайки — «хватит этого», лайк — «сюда можно».
# Сессия обрывается паузой; считается по свежей истории при каждом запросе, не по кэшу профиля.
SESSION_GAP_MS = 45 * 60_000
SESSION_EVENTS = 150
RESTLESS_LEAVE = 2.0  # столько «неусидчивости» в текущем стиле — уходим, не дожидаясь срока
RESTLESS_BOLD = 0.2  # и на столько смелее выбираем, куда: за единицу, не больше двух
REJECT_STYLE = 2.0  # столько отказов стилю за сессию — в эту сессию туда не идём
REJECT_HOME = 4.0  # от любимого стиля так легко не отказываются: пара случайных пропусков — не «надоел»
REFUSE_FADE = 0.99  # отказы тают с каждым новым действием: через несколько часов можно снова попробовать
# Стиль, что устойчиво не заходит, не запрещён (владелец 04.11.2026): предлагается крайне редко и когда
# уместен — поток уже ушёл в близкую к нему музыку. Не из любимого стиля: там отказ и сложился.
UNLOVED_CHANCE = 0.1
STYLE_NEAR = 3  # близкие стили: то же семейство или столько ближайших по звуку
SAME_FAMILY_CHANCE = 0.25  # мост к родственнику (DnB → Jungle) — редко: для слушателя это почти тот же стиль
LIKE_STAY = 6  # лайк или возврат в текущем стиле: на столько треков дольше не уводим
INVITE_FRESH = 4  # лайк чужого трека зовёт в его стиль в ближайшие столько треков
SESSION_NEG_W = 0.5  # поток отходит от только что пропущенного и внутри стиля
# Подмес: одиночный трек другого стиля, близкий по динамике и подаче; поток после него идёт как шёл
SPICE_GAP = 5  # не чаще, чем раз в столько треков
SPICE_CHANCE = 0.6  # с такой вероятностью на запрос (≈ раз в 15–20 треков), неусидчивость поднимает
SPICE_TOP = 5  # случайно среди стольких лучших: не один и тот же трек каждый раз


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
    labels = models.maest_labels_for(cfg.models_dir, cfg.index_dir)
    if not (mp.exists() and hp.exists() and labels):
        return effnet, effnet_labels
    acts, has = np.load(mp), np.load(hp).astype(bool)
    if len(acts) != n or len(has) != n:
        return effnet, effnet_labels
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
        self._style_genre = {t.style: t.genre for t in self.tracks.values()}
        self.style_size = Counter(t.style for t in self.tracks.values())
        self.main_style = self.style_size.most_common(1)[0][0] if self.style_size else ""
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

    def styles_close(self, a: str, b: str) -> bool:
        """Близкие стили: одно семейство или среди STYLE_NEAR ближайших по среднему звуку."""
        if self.style_family_of(a) == self.style_family_of(b):
            return True
        near = self.__dict__.get("_style_near")
        if near is None:
            ids = self.gate.style_ids
            names = {int(i): self.gate.styles[int(np.argmax(ids == i))] for i in np.unique(ids)}
            order = sorted(names)
            centers = _norm(np.stack([self.vec[ids == i].mean(axis=0) for i in order]))
            sims = centers @ centers.T
            np.fill_diagonal(sims, -np.inf)
            near = {names[i]: {names[order[j]] for j in np.argsort(-sims[k])[:STYLE_NEAR]} for k, i in enumerate(order)}
            self._style_near = near
        return b in near.get(a, ()) or a in near.get(b, ())

    def style_family_of(self, style: str | None) -> str:
        genre = self._style_genre.get(style or "", "")
        return style_family(genre, style or "")

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
                      pct_center=getattr(self.cfg, "bridge_pct", G.PCT_DEFAULT))

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


class Mood:
    """Настрой текущей сессии. Что значит каждый сигнал для выбора стиля:
    ранний пропуск и дизлайк — «хватит этого» (+ неусидчивость, отказ стилю); поздний пропуск —
    вполсилы; дослушал — успокоился (неусидчивость вдвое); возврат и «Похожие» — стиль в тему;
    лайк — стиль в тему и приглашение перейти в него, если играет другой."""

    def __init__(self):
        self.trace: list[tuple[int, str, float]] = []  # (трек, что сделал, вклад в неусидчивость)
        self.refused: Counter[str] = Counter()
        self.warm: set[str] = set()
        self.invite: str | None = None
        self.neg: np.ndarray | None = None
        self.skipped: set[int] = set()
        self.liked: set[int] = set()
        self.heard: Counter[int] = Counter()  # трек → сколько прослушиваний добавила эта сессия

    @classmethod
    def from_events(cls, lib: "Library", events, now_ms: int, pending_skips=()) -> "Mood":
        """events — события приборов слушателя, новые сверху (см. session_events). pending_skips —
        пропуски, которые плеер ещё не отправил: он ждёт причину до 12 секунд, а реагировать надо сейчас."""
        m = cls()
        session, later = [], now_ms
        for e in events:
            if later - e["ts"] > SESSION_GAP_MS or len(session) >= SESSION_EVENTS:
                break
            later = e["ts"]
            session.append(e)
        negs: list[int] = []
        seen = set()
        for e in reversed(session):
            tid = e["track_id"]
            if tid is None or tid not in lib.tracks:
                continue
            seen.add(tid)
            m._apply(lib, tid, e["event_type"], e["played_ratio_pct"] or 0, _field(e, "skip_reason"), negs)
        for tid in pending_skips:
            if tid in lib.tracks and tid not in seen:
                m._apply(lib, tid, "skip", 0, None, negs)
        if negs:
            m.neg = _norm(np.sum([lib.vec[lib.row[t]] for t in negs[-5:]], axis=0))
        return m

    def _apply(self, lib: "Library", tid: int, et: str, pct: int, reason: str | None, negs: list[int]) -> None:
        style, delta, refuse = lib.tracks[tid].style, 0.0, 0.0
        if et in ("listen", "replay"):
            self.heard[tid] += 1.0
        elif et == "similar" or (et == "skip" and pct >= LATE_SKIP_PCT):
            self.heard[tid] += 0.5
        if et == "listen":
            self.refused[style] -= 1
        elif et in ("replay", "similar", "like"):
            delta, refuse = -1.5 if et == "like" else -1.0, -1.0
            self.warm.add(style)
            if et == "like":
                self.liked.add(tid)
                self.invite = style
        elif et == "dislike":
            delta, refuse = 1.5, 1.5
            negs.append(tid)
        elif et == "skip":
            self.skipped.add(tid)
            if reason in ("genre", "not_now"):
                delta, refuse = 1.0, 1.0
            elif reason == "tired":
                delta = 0.5
            elif reason in ("rhythm", "vocal"):
                delta, refuse = 0.5, 0.5
            elif pct < EARLY_SKIP_PCT:
                delta, refuse = 1.0, 1.0
            elif pct < LATE_SKIP_PCT:
                delta, refuse = 0.5, 0.5
            if pct < EARLY_SKIP_PCT:
                negs.append(tid)
        else:
            return
        for s in self.refused:
            self.refused[s] *= REFUSE_FADE
        self.refused[style] += refuse
        self.trace.append((tid, et, delta))

    def restless(self, run: set[int]) -> float:
        """Неусидчивость в текущем стиле: только по трекам этого участка потока, иначе пропуски в
        прошлом стиле гнали бы дальше и дальше сразу после перехода."""
        r = 0.0
        for tid, et, delta in self.trace:
            if tid in run:
                r = r * 0.5 if et == "listen" else max(0.0, r + delta)
        return r

    def rejected(self, home: str | None = None) -> set[str]:
        return {s for s, c in self.refused.items() if c >= (REJECT_HOME if s == home else REJECT_STYLE)}


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
        self.fam_mass: Counter[int] = Counter()  # трек → тающая сумма прослушиваний (ступени знакомости)
        self.skip_mass: Counter[int] = Counter()  # трек → тающая сумма пропусков по ступеням
        # обратная связь шлюза (ФТ-8.1, ФТ-8.5): пара стилей → {условие: + перцентилей}
        self.pair_tighten: dict[frozenset, dict[str, float]] = {}
        self.pair_penalty: Counter[tuple[int, int]] = Counter()  # (предыдущий, пропущенный)
        self.tired: Counter[bytes] = Counter()  # recording_key → сколько раз «надоел»
        # стиль → тающие суммы: дослушивал и лайкал / пропускал и дизлайкал. Куда вести мост и подмес
        self.style_pos: Counter[str] = Counter()
        self.style_neg: Counter[str] = Counter()
        served: list[tuple[bool, int]] = []  # по порядку: был ли трек новым, когда прозвучал, и когда (для доли ползунка)
        running: dict[int, tuple[float, int]] = {}  # трек → знакомость и когда (для «был ли новым тогда»)
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
            elif et == "listen":
                self._bridge_accepted(lib, e, tid, last_by_dev.get(_field(e, "device_id")))
            if et in ("listen", "skip", "replay"):
                mass, at = running.get(tid, (0.0, e["ts"]))
                mass *= 0.5 ** ((e["ts"] - at) / 86_400_000 / FAM_HALF_LIFE_DAYS)
                served.append((mass <= NEW_AS_MASS and tid not in self.liked, e["ts"]))
                gain = 1.0 if et != "skip" else (0.5 if (e["played_ratio_pct"] or 0) >= LATE_SKIP_PCT else 0.0)
                running[tid] = (mass + gain, e["ts"])
            if et in ("listen", "skip", "replay", "similar"):
                age_days = max(0.0, (now_ms - e["ts"]) / 86_400_000)
                pct = e["played_ratio_pct"] or 0
                if et in ("listen", "replay"):
                    self.fam_mass[tid] += 0.5 ** (age_days / FAM_HALF_LIFE_DAYS)
                elif et == "similar" or pct >= LATE_SKIP_PCT:
                    self.fam_mass[tid] += 0.5 * 0.5 ** (age_days / FAM_HALF_LIFE_DAYS)
                # штраф самому треку — только за пропуск без причины: «не сейчас», «надоел» и
                # «не тот переход» учитываются своими путями; вернулся и дослушал — пропуск снят
                cancelled = tid in replayed_at and e["ts"] <= replayed_at[tid]
                if et == "skip" and _field(e, "skip_reason") is None and not cancelled:
                    self.skip_mass[tid] += -skip_step_weight(pct) * 0.5 ** (age_days / NEG_HALF_LIFE_DAYS)
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
                self.style_pos[lib.tracks[tid].style] += w * decay
            elif not not_now:  # «не сейчас» — только сессионный сигнал (ФТ-3.3)
                self.neg += -w * decay * v
                self.style_neg[lib.tracks[tid].style] += -w * decay
        for tid, w in recent[-SESSION_LEN:]:
            self.session += max(w, 0.2) * lib.vec[lib.row[tid]]
        self._scale_to_library(lib, events, now_ms)
        self._remember_share(served, now_ms)

    def _remember_share(self, served: list[tuple[bool, int]], now_ms: int) -> None:
        """Состояние модулятора: взвешенные S и N за недавнюю историю (см. SHARE_MEMORY_MIN)."""
        half = max(SHARE_MEMORY_MIN, self.plays_per_day)
        tail = served[-int(half * 8):]
        if not tail:
            self.share_count = self.share_new = 0.0
            return
        marks = np.array([x for x, _ in tail], float)
        ages = np.maximum(0.0, now_ms - np.array([t for _, t in tail], float)) / 86_400_000
        weights = 0.5 ** (np.arange(len(tail))[::-1] / half) * 0.5 ** (ages / SHARE_MEMORY_DAYS)
        self.share_count = float(weights.sum())
        self.share_new = float((weights * marks).sum())

    def _scale_to_library(self, lib: Library, events, now_ms: int) -> None:
        """Темп, размер знакомого и граница «нового» — по всей медиатеке, раз на сборку профиля."""
        since = now_ms - PACE_DAYS * 86_400_000
        heard = [e["ts"] for e in events if e["event_type"] in ("listen", "replay")]
        recent = [t for t in heard if t >= since]
        span = max(PACE_MIN_DAYS, min(PACE_DAYS, (now_ms - min(heard)) / 86_400_000)) if heard else float(PACE_DAYS)
        self.plays_per_day = len(recent) / span
        self._recording_of = {t: info.recording_key for t, info in lib.tracks.items()}
        self.new_mass = -1.0  # столько прослушиваний (с таянием) ещё считается «новым»; −1 — хватает неслушанного
        total = len(lib.tracks)
        need = min(total // 4, max(NEW_POOL_MIN, int(self.plays_per_day * NEW_POOL_DAYS)))
        # забытое — что не звучало дольше цикла; то, что прозвучало недавно, новым не станет ни при каком счёте
        self.forget_before = now_ms - 86_400_000 * self.plays_cycle_days(lib)
        masses = sorted(self.fam_mass.get(t, 0.0) for t in lib.tracks if t not in self.liked)
        fresh = sum(1 for t in lib.tracks if self.newness(t) >= 0.5)
        if fresh < need and masses:
            self.new_mass = masses[min(need, len(masses)) - 1]
        self.known_by_style: Counter[str] = Counter()
        for t in set(self.fam_mass) | self.liked:
            if t in lib.tracks and self.newness(t) < 0.5:
                self.known_by_style[lib.tracks[t].style] += 1
        self.known_count = sum(self.known_by_style.values())
        self.library_size = total
        rates = [1.0 / self.interval_factor(lib, t) for t in set(self.fam_mass) | self.liked
                 if t in lib.tracks and self.newness(t) < 0.5]
        self.mean_rate = float(np.mean(rates)) if rates else 1.0

    def plays_cycle_days(self, lib: Library) -> float:
        """За сколько дней при этом темпе можно прослушать всё уже слышанное."""
        return len(self.fam_mass) / max(self.plays_per_day, 1.0)

    def interval_factor(self, lib: Library, tid: int) -> float:
        """Во сколько раз интервал этого трека длиннее обычного, 1/aᵢ: лайк и любимое чаще, пропуски и
        «надоел» реже."""
        rk = lib.tracks[tid].recording_key
        f = LIKED_INTERVAL if tid in self.liked else 1.0
        f *= max(PLAYS_INTERVAL_MIN, 1.0 - 0.1 * math.log2(1 + self.plays.get(rk, 0)))
        f *= 1.0 + self.skip_mass.get(tid, 0.0)
        return f * TIRED_MULT ** min(self.tired.get(rk, 0), 6)

    def rest_days(self, known_share: float) -> float:
        """Цикл знакомого C = K / (P·s) в днях."""
        per_day = max(self.plays_per_day * max(known_share, 0.05), 1.0)
        return float(np.clip(self.known_count / per_day, REST_MIN_DAYS, REST_MAX_DAYS))

    def min_cycle(self) -> float:
        """Cₘᵢₙ(L): короче этого цикл не делаем — большая медиатека терпит меньше повторов, чем маленькая."""
        return 2.0 * (1.0 + math.log10(1.0 + self.library_size / 100.0))

    def feasible_known(self, share: float) -> float:
        """Доля знакомого, которую пул выдержит без заезженности: s ≤ K / (P·Cₘᵢₙ). Пул растёт — предел уходит."""
        if self.plays_per_day <= 0:
            return share
        return min(share, self.known_count / (self.plays_per_day * self.min_cycle()))

    def ripe_urgency(self, lib: Library, now_ms: int, cycle: float) -> float:
        """Порог зрелости: REST_FLOOR-квантиль просрочки всего знакомого пула."""
        known = [t for t in self.fam_mass if t in lib.tracks and self.newness(t) < 0.5]
        if not known:
            return 0.0
        return float(np.quantile([self.urgency(lib, t, now_ms, cycle) for t in known], REST_FLOOR))

    def urgency(self, lib: Library, tid: int, now_ms: int, cycle: float) -> float:
        """uᵢ = возраст / Tᵢ, где Tᵢ = C·ā·(1/aᵢ). 1 — точно в срок, больше — просрочен."""
        return _overdue(lib, self, tid, now_ms) / max(cycle * self.mean_rate, 1e-6)

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

    def _bridge_accepted(self, lib: "Library", e, tid: int, prev_seen: int | None) -> None:
        """Дослушал трек сразу после смены стиля — пара прижилась, прежнее ужесточение наполовину снято."""
        prev = _field(e, "prev_track_id") or prev_seen
        if prev is None or prev not in lib.tracks or lib.tracks[prev].style == lib.tracks[tid].style:
            return
        t = self.pair_tighten.get(frozenset((lib.tracks[prev].style, lib.tracks[tid].style)))
        if t:
            for c in t:
                t[c] = max(0.0, t[c] - GATE_STEP / 2)

    def style_accept(self, style: str) -> float:
        """Насколько стиль приживается у слушателя, 0…1; ничего не знаем — 0,5."""
        pos, neg = self.style_pos.get(style, 0.0), self.style_neg.get(style, 0.0)
        return (pos + 1.0) / (pos + neg + 2.0)

    def style_refused(self, style: str) -> bool:
        """Стиль устойчиво не заходит: мост и подмес ведут в него редко и только из близких стилей
        (см. UNLOVED_CHANCE); прямо выбрать его можно всегда."""
        return self.style_neg.get(style, 0.0) >= 3.0 and self.style_accept(style) < 0.25

    def familiarity(self, tid: int, minus: float = 0.0) -> float:
        """Знакомость трека 0…1 (ступени, см. FAM_HALF_LIFE_DAYS). minus — не считать столько прослушиваний."""
        if tid in self.liked:
            return 1.0
        return 1.0 - 0.5 ** max(0.0, self.fam_mass.get(tid, 0.0) - minus)

    def newness(self, tid: int, minus: float = 0.0) -> float:
        """Насколько трек новый для слушателя: 1 — новый, 0 — знакомый, между — наполовину."""
        absolute = float(np.clip((FAM_KNOWN - self.familiarity(tid, minus)) / (FAM_KNOWN - FAM_NEW), 0.0, 1.0))
        if absolute < 0.5 and tid not in self.liked and max(0.0, self.fam_mass.get(tid, 0.0) - minus) <= getattr(self, "new_mass", -1.0):
            rk = self._recording_of.get(tid)
            if self.last_played.get(rk, 0) <= self.forget_before:
                return 0.75  # неслушанного мало — самое забытое считается новым
        return absolute

    def forgotten(self, tid: int) -> bool:
        """Новый не потому, что не слышал, а потому, что давно и мало: у него есть прослушивания."""
        return self.fam_mass.get(tid, 0.0) > 0.0 and self.newness(tid) >= 0.5

    def skip_penalty(self, tid: int) -> float:
        """Штраф треку за пропуски 0…SKIP_TRACK_W: ранний свежий — почти весь, старый и поздний — малый."""
        return SKIP_TRACK_W * (1.0 - 0.5 ** self.skip_mass.get(tid, 0.0))

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


def skip_step_weight(pct: float) -> float:
    """Вес пропуска по ступеням SKIP_STEPS: чем меньше дослушал, тем сильнее «не моё»."""
    for limit, w in SKIP_STEPS:
        if pct < limit:
            return w
    return 0.0


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
    tired = prof.tired.get(rk, 0)
    if not tired:  # без «надоел» — только «не дважды за сессию»; давность решает очередь знакомого
        return True
    last = prof.last_played.get(rk)
    hours = TIRED_HOURS * TIRED_MULT ** min(tired, 6)  # «надоел» — реже (ФТ-8.5)
    return not last or now_ms - last > hours * 3_600_000


def _overdue(lib: Library, prof: Profile, tid: int, now_ms: int) -> float:
    """Насколько знакомый трек просрочен: дни с последнего звучания, делённые на его интервал."""
    last = prof.last_played.get(lib.tracks[tid].recording_key)
    age = (now_ms - last) / 86_400_000 if last else ROT_NEVER_DAYS
    return max(0.0, age) / prof.interval_factor(lib, tid)


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
    mood: Mood | None = None,
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
    mood = mood or Mood()
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
        pool = [t for t in prof.liked if t in lib.row] * 2 + \
            [t for t in prof.played if t in lib.row and prof.newness(t) < 0.5]
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
    never = [t for t in lib.tracks if prof.newness(t) >= 0.5]  # новое по ступеням: в т. ч. забытое и только пропущенное
    rng.shuffle(never)
    for tid in never[:200]:
        cands.setdefault(tid, "new")
    # и всё знакомое: новое добавляется пачкой случайных, а знакомое попадало в кандидаты, только если
    # оказалось среди ближайших по вкусу — при небольшой истории их горстка, и ползунок «знакомое»
    # проваливался в новое (профиль владельца 31.10.2026: «новое 0 %» давало 37 % нового)
    for tid in prof.played | prof.liked:
        if tid in lib.row and prof.newness(tid) < 0.5:
            cands.setdefault(tid, "known")
    if not cands:
        cands = {t: "new" for t in lib.tracks}

    # от чего считается переход: выбранный трек «Похожих» или последний сыгранный
    # «Свой поток» стартует от вкуса: трек, игравший до запуска, переход не задаёт — шлюз
    # включается, когда поток уже идёт (в сессии от двух треков)
    # start — первый запрос запущенного потока: что играло до него, переход не задаёт
    ref = anchor if anchor is not None else (hist[-1] if len(hist) >= 2 and not start else None)
    ref_style = lib.tracks[ref].style if ref is not None else None
    gated = ref is not None and not (f.styles or f.genres) and not relaxed
    # дом слушателя — стиль, который он больше всего слушает; истории нет — самый большой в медиатеке
    home = Counter(lib.tracks[t].style for t in prof.played if t in lib.tracks).most_common(1)
    home = home[0][0] if home else lib.main_style
    # участок потока в текущем стиле: по нему считается настрой (подмес участка не прерывает)
    run: set[int] = set()
    for i in range(len(hist) - 1, -1, -1):
        if lib.tracks[hist[i]].style == ref_style:
            run.add(hist[i])
        elif i == 0 or lib.tracks[hist[i - 1]].style != ref_style:
            break  # чужой трек между своими — подмес, участок продолжается
    restless = mood.restless(run) if ref_style is not None else 0.0
    rejected = mood.rejected(home)
    stay_longer = LIKE_STAY if ref_style in mood.warm and ref_style not in rejected else 0
    # Приглашение действует, пока лайк свежий: иначе после ухода из лайкнутого стиля поток рвался бы
    # обратно на каждом запросе
    invite = None
    if mood.invite not in (None, ref_style) and mood.invite not in rejected and any(
            h in mood.liked and lib.tracks[h].style == mood.invite for h in hist[-INVITE_FRESH:]):
        invite = mood.invite
    cross_now = (restless >= RESTLESS_LEAVE or invite is not None
                 or G.cross_allowed_now([lib.tracks[h].style for h in hist], home, stay_longer))
    # Чужой стиль без знакомых треков, а знакомого уже не хватает — домой раньше срока: иначе ползунок
    # проигрывает всю стоянку в новом
    wanted_new = min(1.0, max(0.0, 1.0 - f.familiar))
    # Состояние модулятора доли: история из профиля плюс то, чего в ней ещё нет (очередь, свежие события)
    in_events = {tid for tid, _, _ in mood.trace}
    unseen = [h for h in hist if h not in in_events]
    served_count = prof.share_count + len(unseen)
    served_new = prof.share_new + sum(1 for h in unseen if prof.newness(h) >= 0.5)
    was_new = served_new / served_count if served_count else wanted_new
    # Запас стиля: сколько в нём знакомого и сколько нового. Нужного разряда здесь не осталось — ползунок
    # проигрывает всю стоянку, поэтому уходим раньше срока. Правило одно на обе стороны: и знакомого нет в
    # чужом стиле, и новое кончается в небольшом, прослушанном целиком.
    def stock(style: str, new_side: bool) -> int:
        known_here = prof.known_by_style.get(style, 0)
        return max(0, lib.style_size.get(style, 0) - known_here) if new_side else known_here

    need_new = was_new <= wanted_new
    short_of_stock = (ref_style not in (None, home) and ref_style not in mood.warm and invite is None
                      and abs(was_new - wanted_new) > 0.02 and len(run) >= G.M_STICK
                      and stock(ref_style, need_new) < G.N_CROSS)
    short_of_known = short_of_stock
    cross_now = cross_now or short_of_stock
    bold = round(min(1.0, f.boldness + RESTLESS_BOLD * min(restless, 2.0)), 1)
    unloved: dict[str, bool] = {}

    def may_offer(style: str) -> bool:
        """Отвергнутый стиль: на этот запрос решается один раз — редко и только из близкого."""
        if not prof.style_refused(style):
            return True
        if style not in unloved:
            unloved[style] = (ref_style not in (None, home) and lib.styles_close(ref_style, style)
                              and rng.random() < UNLOVED_CHANCE)
        return unloved[style]
    # Мост ставится в поток сам, когда подошло время (ФТ-6.8): по оценке вкуса трек другого стиля
    # всегда проигрывает трекам основного стиля, а случайных кандидатов шлюз почти не пропускает —
    # поэтому поток владельца 12 дней не выходил из DnB. См. docs/decisions/0010.
    # Только «Мой поток»: выбрал жанр, стиль или фильтр нарочно — играет выбранное, без подмешивания
    # (владелец 04.11.2026).
    mixing = gated and mode == MODE_MY and not f.narrow()
    can_bridge = mixing and cross_now and not start
    if gated:
        ref_row = lib.row[ref]
        if can_bridge:
            for j in lib.gate.cross[ref_row]:
                if j >= 0:
                    cands.setdefault(int(lib.ids[j]), "bridge")
        # После моста поток держится нового стиля (ФТ-6.8), а кандидаты идут от вкуса — в небольшом
        # стиле их почти нет: без этого поток сразу падал бы обратно. Основному стилю это не нужно.
        # И в каждом стиле должны быть кандидаты обоих разрядов: новинки набираются случайной выборкой по всей
        # медиатеке, и в небольшом стиле их не оказывалось, хотя неслушанное там есть (за 20 дней модели
        # 69 раз ставился знакомый вместо нужного нового, почти все — вне основного стиля).
        same_style = np.flatnonzero(lib.gate.style_ids == lib.gate.style_ids[ref_row])
        in_style = [t for t in cands if t in lib.tracks and lib.tracks[t].style == ref_style]
        have_new = sum(1 for t in in_style if prof.newness(t) >= 0.5)
        # С запасом: правила об исполнителях и альбомах отсеивают большую часть, а регулятору нужен выбор
        if have_new < KNN or len(in_style) - have_new < KNN:
            near = lib.vec[same_style] @ lib.vec[ref_row]
            for j in same_style[np.argsort(-near)]:
                tid = int(lib.ids[j])
                if tid not in cands:
                    is_new = prof.newness(tid) >= 0.5
                    if (is_new and have_new < KNN) or (not is_new and len(in_style) - have_new < KNN):
                        cands[tid] = "near"
                        have_new += is_new
                        in_style.append(tid)
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
    if 0.0 < target_new < 1.0:  # пул ещё мал — знакомого столько, сколько он выдержит; крайние положения строги
        target_new = 1.0 - prof.feasible_known(1.0 - target_new)
    # Доля нового — из состояния модулятора, а не по текущей знакомости: дослушанный трек сразу становится
    # знакомым, и по ней доля выглядела бы нулевой (было: при 30–100 % в очереди 88–96 % нового)
    share_new = was_new
    new_deficit = max(0.0, target_new - share_new) / max(target_new, 0.05)
    # и обратно: нового уже больше цели — прибавку получают знакомые. Без этого при 4 % слушанной
    # медиатеки ползунок «50 %» давал 77 % нового (проверка на профиле владельца 30.10.2026)
    known_deficit = max(0.0, share_new - target_new) / max(1.0 - target_new, 0.05) if hist else 0.0
    # крайние положения ползунка — отбором, не прибавкой: только незнакомое / только знакомое,
    # если такие кандидаты вообще есть (кончились — играет что есть, см. strict_new ниже)
    if strict_new and (target_new >= NEW_STRICT or target_new <= 1 - NEW_STRICT):
        want_known = target_new <= 1 - NEW_STRICT
        keep = {t: src for t, src in cands.items() if (prof.newness(t) < 0.5) == want_known}
        if keep:
            cands = keep

    cycle = prof.rest_days(1.0 - target_new)
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
            # тот же альбом или исполнитель под другим стилем — разметка, а не смена музыки
            if (lib.tracks[tid].album_key == lib.tracks[ref].album_key
                    or lib.tracks[tid].artists & lib.tracks[ref].artists):
                continue
            pair = frozenset((ref_style, lib.tracks[tid].style))
            v = lib.gate.check(lib.row[ref], lib.row[tid], boldness=bold, tighten=prof.pair_tighten.get(pair))
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
        if mood.neg is not None:
            s -= SESSION_NEG_W * float(v @ mood.neg)
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
            s -= (REPEAT_W * 0.5 ** (days / max(REPEAT_HALF_LIFE_DAYS, cycle / 2)) * (1.0 - fit_of.get(tid, 0.0))
                  + PLAYS_W * float(np.log1p(prof.plays.get(rk, 0))))
        nw = prof.newness(tid)
        s += NEW_W * (new_deficit * nw + known_deficit * (1.0 - nw))
        s += (f.familiar - 0.5) * 0.4 * (1.0 - 2.0 * nw)
        s -= prof.skip_penalty(tid)  # пропускал — ниже в очереди, тем сильнее, чем раньше и свежее
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
    if can_bridge:
        def then(p: Pick) -> list[Pick]:
            rest = next_tracks(lib, prof, current=p.track_id, history=hist + [p.track_id], filters=filters, n=n - 1,
                               now_ms=now_ms, allowed=allowed, rng=rng, mode=mode, seed=seed,
                               strict_new=strict_new, mood=mood) if n > 1 else []
            return [p] + rest

        # Куда вести: сначала другое семейство и стиль, которого в сессии не было, иначе мост всё
        # время ведёт в ближайшего родственника (DnB ↔ Jungle — для слушателя тот же DnB).
        # Но и не уходить навсегда: после одного-двух чужих стилей мост ведёт домой (владелец 04.11.2026
        # хочет разнообразия, а не потерять DnB: на прогоне без этого DnB был 20–38 % при 69 % в медиатеке).
        heard = {lib.tracks[h].style for h in hist}
        here = lib.style_family_of(ref_style)
        stays = [lib.tracks[h].style for h in hist]
        last_home = max((i for i, st in enumerate(stays) if st == home), default=-1)
        away = sum(1 for i in range(last_home + 1, len(stays)) if i == last_home + 1 or stays[i] != stays[i - 1])
        go_home = (ref_style != home and home not in rejected and invite is None
                   and (short_of_known or away >= BRIDGE_AWAY_MAX or (away >= 1 and rng.random() < 0.5)))

        def level(style: str) -> int:
            if style == invite:
                return -2
            if prof.style_refused(style):  # редкий случай уже выпал (may_offer): иначе его перебьёт всё остальное
                return -1
            if style == home:
                return -1 if go_home else 4
            other_family = lib.style_family_of(style) != here
            return (0 if other_family else 2) + (0 if style not in heard else 1)

        def enough(style: str) -> bool:  # в стиль на пару треков заходить незачем: поток сразу выпадет обратно
            return (lib.style_size.get(style, 0) >= MAIN_STYLE_MIN and style not in rejected and may_offer(style)
                    # слушатель просит уйти или зовёт сам — это важнее доли, её модулятор доберёт потом
                    and (style == home or restless >= RESTLESS_LEAVE or style == invite
                         or stock(style, need_new) >= G.M_STICK))

        direct = [p for p in scored if p.why == "bridge" and enough(lib.tracks[p.track_id].style)
                  and _artist_ok(lib, p.track_id, hist, ARTIST_GAP)]
        def known_there(style: str) -> int:  # чтобы на всю стоянку там хватило того, чего не хватает
            return min(stock(style, need_new), G.N_CROSS)

        # Мост и подмес подчиняются тому же знаку ошибки: иначе с ростом пула они тянут долю к знакомому
        direct.sort(key=lambda p: (level(lib.tracks[p.track_id].style), (prof.newness(p.track_id) >= 0.5) != need_new,
                                   -known_there(lib.tracks[p.track_id].style),
                                   -round(prof.style_accept(lib.tracks[p.track_id].style), 1), -p.score))
        best_direct = level(lib.tracks[direct[0].track_id].style) if direct else 99
        if 2 <= best_direct <= 3 and restless < RESTLESS_LEAVE and rng.random() >= SAME_FAMILY_CHANCE:
            direct, best_direct = [], 99  # в родственника — не сейчас: поищем дальше на следующих запросах
        if best_direct > (-2 if invite else -1 if go_home else 0) and n > 1:
            # От этого трека лучшего моста нет — подойти к нему: следующий трек своего стиля, от
            # которого мост ведёт туда, куда нужно
            sources = lib.gate.bridge_sources(lib.gate.pct(bold))
            approach = []
            for p in scored:
                dest = sources.get(lib.row[p.track_id])
                if dest and lib.tracks[p.track_id].style == ref_style and _artist_ok(lib, p.track_id, hist, ARTIST_GAP):
                    known = [d for d in dest if enough(d)]
                    lv = min((level(d) for d in known), default=99)
                    if lv < best_direct:
                        accept = max((known_there(d), round(prof.style_accept(d), 1)) for d in known if level(d) == lv)
                        approach.append((lv, -accept[0], -accept[1], -p.score, p.track_id, p))
            if approach:
                return then(min(approach, key=lambda x: x[:5])[5])
        if direct:
            return then(direct[0])
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
    classes = None
    # Каждое место — разряду, которого сейчас не хватает (метод наибольшего остатка: отклонение от доли
    # не больше одного трека). Внутри знакомого — по просрочке uᵢ, с учётом уместности в сессии; не дозревшие
    # (uᵢ < REST_FLOOR) идут за ними и только если иначе не набрать долю; звучавшие сегодня — в самом конце.
    if 0.0 < target_new < 1.0 and anchor is None:
        rest = prof.rest_days(1.0 - target_new)
        fresh, known = [], []
        for p in scored:
            (fresh if prof.newness(p.track_id) >= 0.5 else known).append(p)
        urgent = {p.track_id: prof.urgency(lib, p.track_id, now_ms, rest) for p in known}

        def today(tid: int) -> bool:
            last = prof.last_played.get(lib.tracks[tid].recording_key)
            return last is not None and now_ms - last < 12 * 3_600_000

        known.sort(key=lambda p: -urgent[p.track_id] * (1.0 + ROT_FIT_W * fit_of.get(p.track_id, 0.0)))
        ripe = prof.ripe_urgency(lib, now_ms, rest)
        ready = [p for p in known if urgent[p.track_id] >= ripe and not today(p.track_id)]
        early = [p for p in known if urgent[p.track_id] < ripe and not today(p.track_id)]
        played_today = [p for p in known if today(p.track_id)]
        classes = (fresh, ready, early, played_today)
        scored = fresh + ready + early + played_today
    if anchor is None and not 0.0 < target_new < 1.0:
        # крайние положения ползунка: очередь знакомого среди ROT_POOL лучших — по просрочке
        known_pos = [i for i, p in enumerate(scored) if prof.newness(p.track_id) < 0.5][:ROT_POOL]
        pool = sorted((scored[i] for i in known_pos), key=lambda p: -_overdue(lib, prof, p.track_id, now_ms)
                      * (1.0 + ROT_FIT_W * fit_of.get(p.track_id, 0.0)))
        for i, p in zip(known_pos, pool):
            scored[i] = p
    for gap in (ARTIST_GAP, 1, 0):
        out: list[Pick] = []
        seq = list(hist)

        def fits(p: Pick) -> bool:  # запасные тоже не нарушают правил друг относительно друга
            return (_artist_ok(lib, p.track_id, seq, gap) and _album_ok(lib, p.track_id, seq)
                    and _recording_ok(lib, p.track_id, seq, prof, now_ms))

        if classes is None:
            for p in scored:
                if fits(p):
                    out.append(p)
                    seq.append(p.track_id)
                if len(out) >= n:
                    break
        else:
            # разряд выбирается на месте: правила об исполнителях отсеивают новое чаще, и пачка уходила бы от доли
            # Нужен знакомый, а созревшего здесь нет (чужой стиль) — лучше новый: долг модулятор отдаст позже, там,
            # где знакомого много, и пауза до повтора не страдает. Недозревшие — только когда больше нечего.
            at = [0, 0, 0, 0]
            count, new_count = served_count, served_new
            while len(out) < n:
                over = (new_count / count if count else target_new) > target_new
                picked = None
                for k in ((1, 0, 2, 3) if over else (0, 1, 2, 3)):
                    line = classes[k]
                    while picked is None and at[k] < len(line):
                        if fits(line[at[k]]):
                            picked = line[at[k]]
                        at[k] += 1
                    if picked is not None:
                        break
                if picked is None:
                    break
                out.append(picked)
                seq.append(picked.track_id)
                count += 1
                new_count += prof.newness(picked.track_id) >= 0.5
        if out:
            if mixing and not start and n >= 3 and len(out) >= 2:
                spice = _spice(lib, prof, out[0].track_id, hist, f, rng, mood, restless, allowed, now_ms, rejected, may_offer,
                               was_new <= wanted_new)
                if spice is not None:
                    out.insert(1, spice)  # не в конец: последний трек пачки — опора следующего запроса
                    out = out[:n]
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


def _spice(lib: Library, prof: Profile, after: int, hist: list[int], f: Filters, rng: random.Random, mood: Mood,
           restless: float, allowed: set[int] | None, now_ms: int, rejected: set[str], may_offer,
           need_new: bool = True) -> Pick | None:
    """Подмес: один трек другого стиля, близкий к after по динамике и подаче. Частота живая: чужой
    подмес, что пропустили, отодвигает следующий вдвое; лайкнутый — приближает; неусидчивость — тоже."""
    styles = [lib.tracks[h].style for h in hist]
    islands = [i for i in range(1, len(styles) - 1) if styles[i - 1] == styles[i + 1] != styles[i]]
    last = islands[-1] if islands else None
    changes = [i for i in range(1, len(styles)) if styles[i] != styles[i - 1] and i - 1 not in islands and i not in islands]
    begin = max(last + 1 if last is not None else 0, changes[-1] if changes else 0)
    gap = SPICE_GAP
    if last is not None and hist[last] in mood.liked:
        gap = SPICE_GAP // 2
    elif last is not None and hist[last] in mood.skipped:
        gap = SPICE_GAP * 2
    if len(hist) - begin < gap or f.boldness <= 0:
        return None
    chance = (SPICE_CHANCE + 0.2 * min(restless, 1.5)) * min(1.0, 2.0 * f.boldness)
    if rng.random() >= chance:
        return None
    a = lib.tracks[after]
    found = []
    for row, feel in lib.gate.feel_candidates(lib.row[after]):
        tid = int(lib.ids[row])
        t = lib.tracks[tid]
        if (tid in prof.disliked or tid in lib.dj_tools or (allowed is not None and tid not in allowed)
                or t.style in rejected or not may_offer(t.style) or t.artists & a.artists
                or t.album_key == a.album_key or not _filter_ok(lib, tid, f)
                or not _recording_ok(lib, tid, hist, prof, now_ms) or not _album_ok(lib, tid, hist)
                or not _artist_ok(lib, tid, hist, ARTIST_GAP)):
            continue
        # другое семейство интереснее ближайшего родственника: DnB ⇒ Jungle — почти не подмес
        unloved = prof.style_refused(t.style)  # редкий случай уже выпал (may_offer)
        other = unloved or lib.style_family_of(t.style) != lib.style_family_of(a.style)
        found.append((other, feel + unloved + 0.2 * prof.style_accept(t.style) - prof.skip_penalty(tid), tid))
    if not found:
        return None
    if any(x[0] for x in found):
        found = [x for x in found if x[0]]
    fits_share = [x for x in found if (prof.newness(x[2]) >= 0.5) == need_new]
    found = [(score, tid) for _, score, tid in (fits_share or found)]
    found.sort(reverse=True)
    score, tid = rng.choice(found[:SPICE_TOP])
    style = lib.tracks[tid].style
    return Pick(tid, f"{style} · подмес: близок по ритму и подаче к «{a.title or 'предыдущему'}»", score, "spice", a.title,
                ["rhythm", "vocal" if lib.gate.vocal[lib.row[after]] == 1 else "instr"])


def session_events(conn: sqlite3.Connection, device_ids: list[int]) -> list[sqlite3.Row]:
    """Свежие события для настроя сессии — новые сверху. Отдельно от кэша профиля: реакция нужна сразу."""
    q = ",".join("?" * len(device_ids))
    return conn.execute(f"SELECT * FROM events WHERE device_id IN ({q}) ORDER BY ts DESC LIMIT ?",
                        (*device_ids, SESSION_EVENTS)).fetchall()


def _why(tid: int, anchor: int | None, prof: "Profile") -> str:
    """Код причины — плеер сам переводит его на язык интерфейса."""
    if tid in prof.liked:
        return "liked"
    if anchor is not None:
        return "similar"
    if prof.forgotten(tid):
        return "forgotten"
    if prof.newness(tid) >= 0.5:
        return "new"
    return "taste"


def _reason(lib: Library, tid: int, anchor: int | None, src: str, prof: "Profile") -> str:
    """Подпись под треком: стиль и короткое человеческое «почему» (ФТ-8.2)."""
    t = lib.tracks[tid]
    if tid in prof.liked:
        why = "вы его лайкали"
    elif anchor is not None:
        why = f"похоже на «{lib.tracks[anchor].title or 'выбранный трек'}»"
    elif prof.forgotten(tid):
        why = "давно не слушали"
    elif prof.newness(tid) >= 0.5:
        why = "новый для вас трек"
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
