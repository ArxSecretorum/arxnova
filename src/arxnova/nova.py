"""Подбор следующего трека: профиль вкуса (М3), кандидаты и ранжирование (М4–М5) с жёсткими
правилами ФТ-4.4 и ФТ-4.9 и причинами выбора (ФТ-8.2).

Поток держится стиля текущего трека; в другой стиль — только через межжанровый шлюз М6
(gate.py), не чаще одного перехода на N треков (ФТ-6.8).
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass, field, replace

import numpy as np

from arxnova import drift as D
from arxnova import listener as LM
from arxnova import gate as G
from arxnova import keys, models
from arxnova.config import Config

# Стартовые параметры (НФТ-7: позже — в TOML)
# «similar» — нажали «Похожие» на треке: хочу такое — как возврат к треку, а не пропуск
EVENT_WEIGHTS = {"like": 3.0, "replay": 2.0, "similar": 2.0, "listen": 1.0, "skip_early": -1.0, "dislike": -3.0}
ALGO_VERSION = "drift-1"  # журнал: какой логикой выбрано; правку можно сравнивать только внутри одной версии
LISTENER_VERSION = "listener-2"  # «Свой поток» на модели слушателя (решение 0016) с маршрутом (0017): табло отделяет от 0.2.55
# Вес лог-шансов модели слушателя рядом с прочими частями оценки: их разброс среди кандидатов около единицы, у прежней
# близости ко вкусу — около 0,3, на неё настроены доля нового и память о звучавшем
LISTENER_W = 0.3
LISTENER_NEAR_K = 60  # кандидаты рядом с тремя последними принятыми треками сессии: в потоке модель держится рядом
# Как в сете: перешёл в новое место — держится его, пока не пора дальше (LM.DWELL_START), если треки не пропускают. Иначе по краю места
# поток рассыпался по одному треку разных стилей, а с удержанием в 6 треков вкус каждый раз возвращал в DnB (прогоны на снимке 06.11.2026)
LISTENER_HOLD = LM.DWELL_START
LISTENER_SEEN = 60  # что уже звучало в сессии за столько треков: туда толчок слабее, иначе пинг-понг DnB ↔ Jungle
# Инерция: следующий трек чуть вероятнее «впереди по ходу» — там, куда поток сдвигался последние треки. Без неё плавный шаг из DnB
# ведёт только в Jungle, а вкус возвращает домой раньше, чем поток пройдёт промежуточные треки дальше (прогоны на снимке 06.11.2026)
LISTENER_DRIFT_W = 0.15
LISTENER_DRIFT_BACK = 6  # откуда шли: середина стольких треков до прошлого
LISTENER_SPARE = 80  # вне шага считаются целиком только столько лучших: из них запасной ход «если пропустят»
LISTENER_INSIDE = 150  # внутри шага — столько лучших новых и столько знакомых
ROUTE_W = 2.0  # маршрут (решение 0017): вес приближения к цели — разница близостей кандидата и прошлого трека к центру стиля-цели
ROUTE_K = 60  # кандидаты маршрута: в пределах «соседнего места» от прошлого трека — столько ближайших к цели
FRESH_W = 0.3  # свежее (решение 0017, только с настройкой fresh_boost): надбавка неслушанному недавно добавленному, полная в день добавления
LISTENER_MOVE_W = 0.45  # толчок «пора дальше» к соседним стилям после LM.DWELL_START треков на месте: чем дольше, тем сильнее
JOURNAL_ALTERNATIVES = 8
JOURNAL_DAYS = 180
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
TIRED_MULT = 2  # «надоел»: интервал повтора записи удваивается за каждый такой пропуск (ФТ-8.5)
# Память о прослушанном (владелец, 30.10.2026: «час-два в день, а назавтра то же самое при 6 тыс.
# треков»): недавняя запись штрафуется — вчерашняя почти на REPEAT_W, через полмесяца почти ничем;
# часто игравшие — ещё и за частоту. Общего запрета по часам больше нет — его заменила очередь ниже.
# Очередь знакомого (владелец 31.10.2026: «то, что прослушано, встаёт в очередь; выдача — по стилю, но
# сначала старое»; решение 0006). Первым идёт самое просроченное: дни с последнего звучания / интервал трека. Интервал: лайк укорачивает вдвое, частые
# прослушивания — до PLAYS_INTERVAL_MIN (не сильнее: лайков мало, им нельзя занять поток), пропуски
# удлиняют, «надоел» удваивает. Подходящее к тому, что играет сейчас, — чуть вперёд (повтор «в тему»).
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
FADED_LIKE_SKIPS = 2
# Ступени пропуска: чем раньше пропустил, тем сильнее трек уходит вниз — штраф самому треку, не запрет;
# тает за NEG_HALF_LIFE_DAYS. Дизлайк — по-прежнему запрет: прямое «не предлагай» (решение 0003).
SKIP_STEPS = ((10, -1.0), (30, -0.6), (60, -0.25), (85, -0.1))  # (дослушал меньше %, вес)
NEG_HALF_LIFE_DAYS = 60.0
SKIP_TRACK_W = 0.5  # предел штрафа треку за пропуски
# Следующий трек — не строго лучший, а случайный среди PICK_POOL лучших с перевесом к близким
# (как у старта «Своего потока»): две сессии в одном вкусе не повторяют друг друга шаг в шаг.
PICK_POOL = 6
PICK_TEMP = 0.03
# Чередование таблицы «кто слушает вместе» (решение 0015). В доле LISTEN_TRIAL_SHARE запросов «Своего потока» к оценке трека
# прибавляется LISTEN_W × (близость его исполнителя к этой сессии + к вкусу) по таблице слушателей (listeners.py), в остальных
# выбор прежний. Журнал хранит рукав и то, что выбрал бы другой: сравнение — только там, где выборы разошлись.
LISTEN_TRIAL_SHARE = 0.5
LISTEN_W = 0.1
LISTEN_LIKE_W = 1.0  # лайк в центре вкуса весит как одно свежее прослушивание


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
    parts: dict[str, float] = field(default_factory=dict)  # из чего сложилась оценка (журнал выбора)
    slot: float | None = None  # проба: шанс, что проба стоит именно на этом месте пачки
    track_prob: float | None = None  # проба из близких: шанс этого трека внутри направления; None — выбран подбором внутри стиля, шанс неизвестен
    prob: float | None = None  # проба: шанс направления, усреднённый по выборке Томпсона; у остальных None, выбор детерминирован при кандидатах. Было: с какой вероятностью был выбран среди равных: без неё нельзя пересчитать другие веса
    phi: tuple[float, float] | None = None  # чередование (решение 0015): близость по таблице слушателей к сессии и к вкусу, в обоих рукавах
    q: float | None = None  # модель слушателя (решение 0016): вероятность пропуска по средним весам — для табло угадывания


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

# Настрой сессии (владелец 04.11.2026): пропуски и дизлайки — «хватит этого», лайк — «сюда можно».
# Сессия обрывается паузой; считается по свежей истории при каждом запросе, не по кэшу профиля.
SESSION_GAP_MS = 45 * 60_000
SESSION_EVENTS = 150
DRIFT_EVENTS = 600  # дрейф собирается по всей очереди: плеер шлёт до 400 треков, больше суток музыки подряд
RESTLESS_BOLD = 0.2  # и на столько смелее выбираем, куда: за единицу, не больше двух
REJECT_STYLE = 2.0  # столько отказов стилю за сессию — в эту сессию туда не идём
REJECT_HOME = 4.0  # от любимого стиля так легко не отказываются: пара случайных пропусков — не «надоел»
REFUSE_FADE = 0.99  # отказы тают с каждым новым действием: через несколько часов можно снова попробовать
# Стиль, что устойчиво не заходит, не запрещён (владелец 04.11.2026): предлагается крайне редко и когда
# уместен — поток уже ушёл в близкую к нему музыку. Не из любимого стиля: там отказ и сложился.
UNLOVED_CHANCE = 0.1
STYLE_NEAR = 3  # близкие стили: то же семейство или столько ближайших по звуку
SESSION_NEG_W = 0.5  # поток отходит от только что пропущенного и внутри стиля
# Подмес: одиночный трек другого стиля, близкий по динамике и подаче; это проба, частоту и направление задаёт дрейф (drift.py)
SPICE_TOP = 5  # случайно среди стольких лучших: не один и тот же трек каждый раз
PROBE_POOL = 4000  # среди стольких близких по ощущению выбираем: почти все, кто прошёл пороги
PROBE_MIN_TRACKS = 3  # в начале сессии проб нет: сначала понятно, что играет
HIST_WINDOW = 30
PROB_DRAWS = 64
HOME_PROBE_MAX = 0.5  # после дрейфа на обычный стиль идёт доля проб по его месту во вкусе, но не больше половины
STRONG_DIRECTION = 0.1  # направление с такой долей уже принято: трек выбираем в нём, а не по близости к текущему


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


MIX_MIN_S = 1800  # от получаса — непрерывный диджейский микс, а не трек: на медиатеке владельца все 56 такие
EDITION_LENGTH_RATIO = 1.4  # заметно короче — это отрывок из микса, а не издание той же песни


class Library:
    """Матрицы и метаданные в памяти. Пересобирается после индексации."""

    editions: dict[bytes, list[int]] = {}  # ключ записи → её издания, если их несколько
    best_edition: dict[int, int] = {}  # издание → лучшее издание той же записи (ФТ-4.8)
    # таблица «кто слушает вместе» по строкам медиатеки (решение 0015); пусто — таблицы нет, поток выбирает как раньше
    listen: np.ndarray = np.zeros((0, 0), np.float32)
    listen_ok: np.ndarray = np.zeros(0, bool)
    listen_rows: list = []  # строки таблицы у каждого трека: по ним «тот же исполнитель»

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
                "SELECT id, artist, albumartist, album, title, date, mb_album, compilation, recording_key, duration, genre_raw, bits, sample_rate, added_at FROM tracks"
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
            if (r["duration"] or 0) >= MIX_MIN_S:
                self.dj_tools.add(int(tid))  # час одним файлом в потоке не нужен (из 5 показов дослушан 1); сам включить можно
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
        self._merge_editions(rows)
        from arxnova import listeners

        table = listeners.Table.load()
        if table is not None:  # исполнитель, иначе исполнитель альбома; строки те же, что у vec
            self.listen, self.listen_ok, self.listen_rows = table.library(
                [(rows[int(t)]["artist"], rows[int(t)]["albumartist"]) for t in self.ids])
        self._style_genre = {t.style: t.genre for t in self.tracks.values()}
        self.style_size = Counter(t.style for t in self.tracks.values())
        self.main_style = self.style_size.most_common(1)[0][0] if self.style_size else ""
        self.clap = self._load_clap(conn)  # нужен шлюзу (вокал новых треков) и плиткам (голос, инструменты)
        self.gate = self._load_gate(d, {r[0] for r in conn.execute("SELECT track_id FROM track_features WHERE bpm_src = 'autocorr_only'")})
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
        # когда трек добавлен и где кончился первый большой разбор: «Недавно добавленное» и свежее в «Своём потоке» (решение 0017)
        self.added_at = {t: int(r["added_at"] or 0) for t, r in rows.items() if t in self.tracks}
        self.first_import_end = first_import_end(self.added_at.values())
        self._add_sound_tiles(conn, d, rows)

    # Без дат добавления (медиатеки тестов собираются мимо конструктора) свежего нет, маршрут работает
    added_at: dict[int, int] = {}
    first_import_end = 0
    _route_styles: dict[str, np.ndarray] | None = None
    _style_centers: dict[str, np.ndarray] | None = None

    def route_styles(self) -> dict[str, np.ndarray]:
        """Стиль → строки его треков, годных потоку (без голых дорожек и вторых изданий): из них выбирается цель маршрута."""
        if self._route_styles is None:
            by: dict[str, list[int]] = {}
            for t, info in self.tracks.items():
                if t in self.row and t not in self.dj_tools and getattr(self, "best_edition", {}).get(t, t) == t and info.style:
                    by.setdefault(info.style, []).append(self.row[t])
            self._route_styles = {s: np.array(r, dtype=int) for s, r in by.items()}
        return self._route_styles

    def route_capacity(self, style: str) -> int:
        """Сколько треков стиля можно поставить подряд за стоянку при правилах исполнителя (не чаще раза в ARTIST_GAP) и альбома
        (не чаще раза в ALBUM_GAP): у исполнителя — не больше DWELL_START / ARTIST_GAP и не больше его альбомов. В Black Metal владельца
        один исполнитель, в Celtic 64 трека из 66 у одного: туда маршрут приходил и не мог стоять (независимая проверка 07.11.2026)."""
        memo = self.__dict__.setdefault("_route_capacity", {})
        c = memo.get(style)
        if c is None:
            albums: dict[str, set] = {}
            for r in self.route_styles().get(style, ()):
                info = self.tracks[int(self.ids[r])]
                albums.setdefault(min(info.artists) if info.artists else f"#{info.album_key}", set()).add(info.album_key)
            per_artist = -(-LM.DWELL_START // ARTIST_GAP)
            c = memo[style] = sum(min(per_artist, len(a)) for a in albums.values())
        return c

    def style_closeness(self, style: str) -> np.ndarray:
        """Близость каждого трека медиатеки к центру стиля — раз на стиль: маршрут спрашивает её на каждом шаге пути."""
        memo = self.__dict__.setdefault("_style_closeness", {})
        c = memo.get(style)
        if c is None:
            c = memo[style] = self.vec @ self.style_center(style)
        return c

    def style_center(self, style: str) -> np.ndarray:
        """Единичный центр стиля на карте звука: к нему ведёт маршрут."""
        if self._style_centers is None:
            self._style_centers = {}
        c = self._style_centers.get(style)
        if c is None:
            rows = self.route_styles().get(style)
            v = self.vec[rows].mean(axis=0) if rows is not None and len(rows) else np.zeros(self.vec.shape[1], dtype=self.vec.dtype)
            c = (v / max(float(np.linalg.norm(v)), 1e-9)).astype(np.float32)
            self._style_centers[style] = c
        return c

    def _load_clap(self, conn: sqlite3.Connection, table: str = "clap_vectors") -> np.ndarray:
        """Векторы CLAP (микса или, table="clap_voc", выделенного голоса) в порядке ids; у трека без вектора строка NaN."""
        from arxnova import sound_tags

        m = np.full((len(self.ids), sound_tags.CLAP_DIM), np.nan, np.float32)
        for tid, blob in conn.execute(f"SELECT track_id, vector FROM {table}"):
            i = self.row.get(tid)
            if i is not None and len(blob) == sound_tags.CLAP_DIM * 4:
                m[i] = np.frombuffer(blob, np.float32)
        return m

    def _add_sound_tiles(self, conn: sqlite3.Connection, d, rows: dict) -> None:
        """Плитки из того, что уже посчитано: голос, инструменты, недавно добавленное. Нового разбора не требуют."""
        from arxnova import sound_tags

        have = ~np.isnan(self.clap).any(axis=1)
        # Голос (решение 0014): среднее трёх мнений — число по стему, слой на CLAP выделенного голоса и описания CLAP. Число по
        # стему — 8 отрывков, у треков, разобранных Demucs по целому треку, — их прежнее (того же качества); CLAP голоса у них
        # появляется, только если трек разобрал помощник новой версии. Без Demucs — запасной слой на CLAP микса
        stem = np.where(np.isnan(self._female_stem), self._female_stored, self._female_stem)
        female = sound_tags.female_from_voice(stem, self._load_clap(conn, "clap_voc"))
        need = np.isnan(female) & have
        if need.any():
            female[need] = sound_tags.female_probability_clap(self.clap[need])
        known = ~np.isnan(female)
        if known.any():
            voiced = (self.gate.vocal == 1) & known
            p = np.where(known, female, 0.5)
            self.scenarios["voice_female"] = {"title": "Женский вокал", "members": {int(t) for t in self.ids[voiced & (p >= sound_tags.FEMALE_FROM)]}}
            self.scenarios["voice_male"] = {"title": "Мужской вокал", "members": {int(t) for t in self.ids[voiced & (p < sound_tags.MALE_UNDER)]}}
        if have.any():
            ids = self.ids[have]
            for tag, members in sound_tags.tag_members(self.clap[have]).items():
                self.scenarios[tag] = {"title": sound_tags.TAGS[tag], "members": {int(t) for t in ids[members]}}
        # Недавно добавленное: всё, что позже первого большого разбора, за последние 90 дней (то же определение, что у свежего в потоке)
        if self.added_at:
            now_s = int(time.time())
            fresh = {t for t, a in self.added_at.items() if a > self.first_import_end and now_s - a <= 90 * 86_400}
            if fresh:
                self.scenarios["added_recent"] = {"title": "Недавно добавленное", "members": fresh}

    def _merge_editions(self, rows: dict) -> None:
        """Одна песня в разных изданиях — один ключ записи: не звучит дважды за сессию, знакомость и пропуски общие.
        Играет лучшее издание: больше разрядность, затем частота (ФТ-4.8)."""
        groups: dict[tuple[str, str], list[int]] = {}
        for tid, info in self.tracks.items():
            family = keys.song_family(info.artist, info.title)
            if family:
                groups.setdefault(family, []).append(tid)
        for family, tids in groups.items():
            if len(tids) < 2:
                continue
            tids.sort(key=lambda t: -self.tracks[t].duration)
            parts: list[list[int]] = []
            for t in tids:
                d = self.tracks[t].duration
                for part in parts:
                    if self.tracks[part[0]].duration <= EDITION_LENGTH_RATIO * d:
                        part.append(t)
                        break
                else:
                    parts.append([t])
            for i, part in enumerate(parts):
                if len(part) > 1 and len({self.tracks[t].recording_key for t in part}) > 1:
                    merged = hashlib.md5(f"song|{family[0]}|{family[1]}|{i}".encode()).digest()
                    for t in part:
                        self.tracks[t].recording_key = merged
        by_record: dict[bytes, list[int]] = {}
        for tid, info in self.tracks.items():
            by_record.setdefault(info.recording_key, []).append(tid)
        self.editions = {rk: tids for rk, tids in by_record.items() if len(tids) > 1}

        def quality(t: int) -> tuple:
            r = rows[t]
            bits = r["bits"] or 0
            return (24 if bits == 1 else bits, r["sample_rate"] or 0, -t)  # DSD: один бит, но это не хуже 24

        self.best_edition = {t: max(tids, key=quality) for tids in self.editions.values() for t in tids}

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

    def direction_prior(self, prof: "Profile", main: str) -> dict[str, float]:
        """φ⁰: куда пробовать из основного стиля: что приживается у слушателя и близко по звуку. Дому — доля по его месту во вкусе,
        всего не больше половины: чем больше слушатель живёт в одном стиле, тем вероятнее, что вернётся именно туда."""
        cache = prof.__dict__.setdefault("_direction_prior", {})  # у каждого профиля свой: id объекта после сборки мусора повторяется
        key = main
        if key in cache:
            return cache[key]
        centers = self.__dict__.get("_style_centers")
        if centers is None:
            ids = self.gate.style_ids
            centers = {self.gate.styles[int(np.argmax(ids == i))]: _norm(self.vec[ids == i].mean(axis=0)) for i in np.unique(ids)}
            self._style_centers = centers
        out: dict[str, float] = {}
        here = centers.get(main)
        for style, size in self.style_size.items():
            if style == main or size < MAIN_STYLE_MIN or style in G.NEVER_BRIDGE_TO or style not in centers:
                continue
            near = max(0.0, float(here @ centers[style])) if here is not None else 0.0
            kin = 0.25 if self.style_family_of(style) == self.style_family_of(main) else 1.0  # DnB → Jungle — почти не проба
            out[style] = prof.style_accept(style) * (near + 0.2) * kin
        total = sum(out.values())
        out = {s: x / total for s, x in out.items()} if total > 0 else {}
        back = {h: x for h, x in prof.homes.items() if h in out}
        if back:
            mass = min(HOME_PROBE_MAX, sum(back.values()))
            want = {h: max(out[h], mass * x / sum(back.values())) for h, x in back.items()}
            over = sum(want.values())
            if over > 1.0:  # мало стилей: дом и так занимает почти всё
                want = {h: x / over for h, x in want.items()}
                over = 1.0
            rest = sum(x for s, x in out.items() if s not in want)
            scale = (1.0 - over) / rest if rest > 0 else 0.0
            out = {s: want.get(s, x * scale) for s, x in out.items()}
        cache[key] = out
        return out

    def style_family_of(self, style: str | None) -> str:
        genre = self._style_genre.get(style or "", "")
        return style_family(genre, style or "")

    def _load_gate(self, d, weak_tempo: set[int] | None = None) -> G.Gate:
        """Шлюз по признакам звука (`arxnova facets`: вокал, энергия, темп); у трека без них межстилевые
        переходы закрыты — лучше меньше мостов, чем неуместные (КО-2)."""
        feats, cols = None, []
        if (d / "features.npy").exists() and (d / "features.json").exists():
            feats = np.load(d / "features.npy")
            cols = json.loads((d / "features.json").read_text())["columns"]
            if len(feats) != len(self.ids):
                feats, cols = None, []
        if feats is not None and "bpm" in cols and weak_tempo:
            # Запасной способ темпа неверен: DnB попадает в свой темп в 56 % против 91,5 %, а порог уверенности проходит.
            # Считаем темп неизвестным (замер 04.11.2026, 723 трека)
            feats = feats.copy()
            rows = [self.row[t] for t in weak_tempo if t in self.row]
            feats[rows, cols.index("bpm")] = np.nan
            if "bpm_conf" in cols:
                feats[rows, cols.index("bpm_conf")] = 0.0
        from arxnova import sound_tags

        n = len(self.ids)
        col = (lambda name: feats[:, cols.index(name)] if feats is not None and name in cols else np.full(n, np.nan))
        self._female_stored = col("female_p").astype(np.float64)  # Demucs по целому треку (до 0014), переведено в число
        self._female_stem = col("female_stem_p").astype(np.float64)  # стем 8 отрывков
        # Вокал: доля звучания по стему Demucs (целый трек или 8 отрывков — одного качества); без Demucs — запасной слой на CLAP микса
        vf = col("vocal_frac")
        pv = np.full(n, np.nan)
        have = ~np.isnan(self.clap).any(axis=1) & np.isnan(vf)
        if have.any():
            pv[have] = sound_tags.vocal_probability_clap(self.clap[have])
        vocal = np.where(~np.isnan(vf), (vf >= G.VOCAL_FRAC).astype(int), np.where(~np.isnan(pv), (pv >= 0.5).astype(int), -1))
        styles = [self.tracks[int(t)].style for t in self.ids]
        forbid = {frozenset(p) for p in getattr(self.cfg, "gate_forbid", []) if len(p) == 2}
        return G.Gate(self.vec, styles, self.style_scores, feats, cols, vocal=vocal, forbid=forbid,
                      pct_center=getattr(self.cfg, "bridge_pct", G.PCT_DEFAULT))

    def has_features(self) -> bool:
        return bool(self.gate.has_features.any())

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


def listening_day(ts_ms: int, utc_offset_s: int) -> int:
    """День слушателя начинается в 4 утра по часам сервера: вечер через полночь — один день, а не два голоса."""
    return (ts_ms // 1000 + utc_offset_s - 4 * 3600) // 86_400


def _without_undone(events: list) -> list:
    """Лайк и снятие лайка, «не предлагать» и «отменить» взаимно гасятся. events — новые сверху."""
    undone: set[int] = set()
    waiting: dict[tuple, list[int]] = {}
    for i, e in enumerate(reversed(events)):
        et, tid = e["event_type"], e["track_id"]
        if et in ("unlike", "undislike"):
            found = waiting.get((tid, et[2:]))
            if found:
                undone.add(found.pop())
            undone.add(i)
        elif et in ("like", "dislike"):
            waiting.setdefault((tid, et), []).append(i)
    n = len(events)
    return [e for k, e in enumerate(events) if n - 1 - k not in undone]


class Mood:
    """Настрой текущей сессии. Что значит каждый сигнал для выбора стиля:
    ранний пропуск и дизлайк — «хватит этого» (+ неусидчивость, отказ стилю); поздний пропуск —
    вполсилы; дослушал — успокоился (неусидчивость вдвое); возврат и «Похожие» — стиль в тему;
    лайк — стиль в тему и приглашение перейти в него, если играет другой."""

    def __init__(self):
        self.trace: list[tuple[int, str, float]] = []  # (трек, что сделал, вклад в неусидчивость)
        self.refused: Counter[str] = Counter()
        self.neg: np.ndarray | None = None
        self.skipped: set[int] = set()
        self.liked: set[int] = set()
        self.heard: Counter[int] = Counter()  # трек → сколько прослушиваний добавила эта сессия
        self.by_track: dict[int, list] = {}  # трек → действия за сессию (для дрейфа)

    @classmethod
    def from_events(cls, lib: "Library", events, now_ms: int, pending_skips=()) -> "Mood":
        """events — события приборов слушателя, новые сверху (см. session_events). pending_skips —
        пропуски, которые плеер ещё не отправил: он ждёт причину до 12 секунд, а реагировать надо сейчас."""
        m = cls()
        events = _without_undone(list(events)[:DRIFT_EVENTS])
        # Для дрейфа — все полученные события: очередь продолжается и после паузы, и сдвиг не должен забываться по сроку
        for e in reversed(events):
            tid = e["track_id"]
            if tid is None or tid not in lib.tracks:
                continue
            if e["event_type"] == "listen" and (_field(e, "seeks_back") or 0) > 0:
                m.by_track.setdefault(tid, []).append(("similar", 100, False, None, None, e["ts"]))  # возвращался внутри трека: сильный плюс
            m.by_track.setdefault(tid, []).append((e["event_type"], e["played_ratio_pct"] or 0, bool(_field(e, "is_organic")),
                                                   _field(e, "skip_reason"), _field(e, "left_by"), e["ts"]))
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
            if e["event_type"] == "skip" and (_field(e, "left_by") == "back" or _field(e, "is_organic")):
                continue  # «назад» и пролистывание своего альбома — не отказ стилю
            m._apply(lib, tid, e["event_type"], e["played_ratio_pct"] or 0, _field(e, "skip_reason"), negs)
        for tid in pending_skips:
            if tid in lib.tracks and tid not in seen:
                m.by_track.setdefault(tid, []).append(("skip", 0, False, None, None, now_ms))
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
            if et == "like":
                self.liked.add(tid)
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

    def rejected(self, homes=()) -> set[str]:
        return {s for s, c in self.refused.items() if c >= (REJECT_HOME if s in homes else REJECT_STYLE)}


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
        self.style_recent: list[tuple[int, str, float, float]] = []  # (когда, стиль, +принято, +отказ) за двое суток: счёт на прошлый момент
        by_day: dict[int, Counter[str]] = {}  # день → что и сколько слушал: для дома
        clock = time.localtime(now_ms / 1000).tm_gmtoff
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
                if et == "skip" and _field(e, "skip_reason") is None and not cancelled and _field(e, "left_by") != "back":
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
                if now_ms - e["ts"] < 2 * 86_400_000:
                    self.style_recent.append((e["ts"], lib.tracks[tid].style, w * decay, 0.0))
                by_day.setdefault(listening_day(e["ts"], clock), Counter())[lib.tracks[tid].style] += w
            elif not not_now:  # «не сейчас» — только сессионный сигнал (ФТ-3.3)
                self.neg += -w * decay * v
                self.style_neg[lib.tracks[tid].style] += -w * decay
                if now_ms - e["ts"] < 2 * 86_400_000:
                    self.style_recent.append((e["ts"], lib.tracks[tid].style, 0.0, -w * decay))
        for tid, w in recent[-SESSION_LEN:]:
            self.session += max(w, 0.2) * lib.vec[lib.row[tid]]
        self.homes = D.home_of(by_day[d] for d in sorted(by_day))
        self._share_between_editions(lib)
        self._forget_faded_likes(events, lib)
        self._scale_to_library(lib, events, now_ms)
        self._remember_share(served, now_ms)
        self._events = events
        self._listener: LM.Model | None = None
        self._new_memo: dict[int, float] = {}  # последним: до конца сборки поля профиля ещё меняются

    def listener_model(self, lib: Library) -> LM.Model:
        """Модель слушателя (решение 0016) по истории этого профиля: веса считаются один раз на профиль. Сервер зовёт это
        там же, где собирает профиль, в фоне, чтобы запрос не ждал."""
        if self._listener is None:
            self._listener = LM.fit_from_events(lib, self._events)
        return self._listener

    def _share_between_editions(self, lib: Library) -> None:
        """Слышал одно издание — знакомы все: иначе другое издание той же песни звучит как «новое для вас»."""
        for tids in getattr(lib, "editions", {}).values():
            for mass in (self.fam_mass, self.skip_mass):
                m = max((mass.get(t, 0.0) for t in tids), default=0.0)
                if m > 0.0:
                    for t in tids:
                        mass[t] = m

    def _forget_faded_likes(self, events, lib: Library) -> None:
        """Лайк не вечен для подбора: трек, который после лайка дважды пропустили и не дослушали, больше не «вы его лайкали»
        (дослушан 62 % против 88 %, замер 04.11.2026). В избранном на устройстве он остаётся."""
        count: Counter[int] = Counter()
        for e in events:
            tid = e["track_id"]
            if tid not in self.liked:
                continue
            et = e["event_type"]
            if et == "like":
                count[tid] = 0
            elif et == "skip" and not e["is_organic"] and (e["played_ratio_pct"] or 0) < LATE_SKIP_PCT:
                count[tid] += 1
            elif et in ("listen", "replay"):
                count[tid] = max(0, count[tid] - 1)
        self.liked -= {t for t, c in count.items() if c >= FADED_LIKE_SKIPS}

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
            # Ужесточается пара (ФТ-8.1). Порог у шлюза один — близость микса: фасетов больше нет (решение 0014), и «не тот ритм»
            # или «не тот вокал» тоже поднимают его, иначе причина пропуска ни на что бы не влияла
            if not (reason in ("rhythm", "vocal", "genre") or early):
                return
            t = self.pair_tighten.setdefault(frozenset((sa, sb)), {})
            t["mix"] = min(GATE_TIGHTEN_MAX, t.get("mix", 0.0) + GATE_STEP)
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

    def home_style(self, lib: "Library") -> str:
        """Обычный стиль слушателя (drift.home_of). Истории нет — самый большой стиль медиатеки."""
        return next(iter(self.homes), lib.main_style)

    def style_stats_before(self, ts: int) -> tuple[dict[str, float], dict[str, float]]:
        """Счёт стилей (принято, отказы) без событий с момента ts — каким он был перед ним (решение 0017: жребий цели маршрута)."""
        pos, neg = dict(self.style_pos), dict(self.style_neg)
        for at, style, dp, dn in self.style_recent:
            if at >= ts:
                pos[style] = pos.get(style, 0.0) - dp
                neg[style] = neg.get(style, 0.0) - dn
        return pos, neg

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
        """Насколько трек новый для слушателя: 1 — новый, 0 — знакомый, между — наполовину. Профиль после сборки не меняется,
        поэтому ответ запоминается: за запрос его спрашивают тысячи раз (профиль на Pi 06.11.2026)."""
        memo = getattr(self, "_new_memo", None) if minus == 0.0 else None
        if memo is not None and tid in memo:
            return memo[tid]
        value = min(1.0, max(0.0, (FAM_KNOWN - self.familiarity(tid, minus)) / (FAM_KNOWN - FAM_NEW)))
        if value < 0.5 and tid not in self.liked and max(0.0, self.fam_mass.get(tid, 0.0) - minus) <= getattr(self, "new_mass", -1.0):
            rk = self._recording_of.get(tid)
            if self.last_played.get(rk, 0) <= self.forget_before:
                value = 0.75  # неслушанного мало — самое забытое считается новым
        if memo is not None:
            memo[tid] = value
        return value

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
        elif et == "listen":  # перемотка назад — интерес, перемотка вперёд через весь трек — дослушиванием не считаем
            w = EVENT_WEIGHTS["listen"] * (1.5 if (_field(e, "seeks_back") or 0) > 0 else 0.25 if (_field(e, "seeks_forward") or 0) >= 2 else 1.0)
        elif et in ("replay", "similar"):
            w = EVENT_WEIGHTS[et]
        elif et == "skip":
            pct = e["played_ratio_pct"] or 0
            w = EVENT_WEIGHTS["skip_early"] if pct < EARLY_SKIP_PCT and _field(e, "left_by") != "back" else 0.0
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


def _listen_keys(lib: Library, r: int):
    """Чем исполнитель трека представлен в таблице: её строки. Медиатека без них (собрана вручную) — сам трек."""
    rows = lib.listen_rows
    return rows[r] if len(rows) == len(lib.ids) and rows[r] else (("track", r),)


def _listen_center(lib: Library, weights: dict[int, float]) -> tuple[np.ndarray | None, dict]:
    """Центр в таблице слушателей (решение 0015): сумма векторов треков с весами и вклад каждой строки таблицы отдельно —
    у кандидата его собственный исполнитель вычитается. Иначе близость к самому себе выдавалась бы за «слушают вместе».
    «Тот же исполнитель» — общая строка таблицы, как в проверке В4, а не похожее имя."""
    total = None
    by_key: dict = {}
    for t, w in weights.items():
        r = lib.row.get(t)
        if r is None or w <= 0 or t not in lib.tracks or not lib.listen_ok[r]:
            continue
        x = w * lib.listen[r]
        total = x.copy() if total is None else total + x
        for k in _listen_keys(lib, r):
            by_key.setdefault(k, {})[t] = x
    return total, by_key


def _listen_phi(lib: Library, tid: int, center) -> float:
    """Косинус вектора исполнителя трека с центром без треков того же исполнителя; нет данных — 0."""
    total, by_key = center
    r = lib.row[tid]
    if total is None or not lib.listen_ok[r]:
        return 0.0
    same: dict[int, np.ndarray] = {}
    for k in _listen_keys(lib, r):
        same.update(by_key.get(k, {}))
    c = total - np.sum(list(same.values()), axis=0) if same else total
    n = float(np.linalg.norm(c))
    return float(lib.listen[r] @ c) / n if n > 1e-6 else 0.0


def _listen_taste_weights(lib: Library, prof: "Profile") -> dict[int, float]:
    """Вес трека в центре вкуса: тающая масса прослушиваний и лайк, без дизлайков. Масса копируется на все издания записи
    (`_share_between_editions`), поэтому берётся только у лучшего издания: запись считается один раз (проверка 06.11.2026)."""
    return {t: (prof.fam_mass.get(t, 0.0) if lib.best_edition.get(t, t) == t else 0.0) + (LISTEN_LIKE_W if t in prof.liked else 0.0)
            for t in set(prof.fam_mass) | prof.liked if t not in prof.disliked}


def _run_since(history: list[int], run_from: int, mood: "Mood") -> int:
    """Граница запуска по событиям: последнее событие треков, сыгранных до него. Трек, повторившийся в этом запуске, границу не
    двигает: иначе его событие внутри запуска отрезало бы всё, что прозвучало в запуске до повтора (проверка 06.11.2026)."""
    inside = set(history[run_from:])
    return max((e[5] for t in history[:run_from] if t not in inside for e in mood.by_track.get(t, ()) if e[0] in ("listen", "skip")),
               default=0)


def _heard_in_run(whole: list[int], mood: "Mood", since: int) -> list[int]:
    """Треки этого запуска, которые слушатель не бросил в начале (как в проверке В4: не пропущен раньше 15 %).
    Ещё играющий трек событий не имеет и в центр сессии не входит."""
    out = []
    for t in dict.fromkeys(whole):
        evs = [e for e in mood.by_track.get(t, ()) if e[5] >= since]
        if any(e[0] == "skip" and e[1] < EARLY_SKIP_PCT and e[4] != "back" for e in evs):
            continue
        if any(e[0] in ("listen", "replay", "similar", "like") or (e[0] == "skip" and e[1] >= EARLY_SKIP_PCT) for e in evs):
            out.append(t)
    return out


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
    leave_style: str | None = None,
    run_from: int = 0,
    stepped: bool = False,
    force_leave: bool = False,
    trace: dict | None = None,
    listen: str | None = None,
    model: str = "drift",
    fresh: bool = False,
) -> list[Pick]:
    """Следующий трек и запасные (ФТ-4.6). history — треки сессии по порядку, current —
    последний; allowed — что доступно этому прибору (ФТ-4.5), None — всё.

    MODE_MY: опора — профиль вкуса (долгосрочный + сессионный вектор, лайки); что играло
    перед запуском, на выбор почти не влияет. MODE_TRACK: опора — seed, один и тот же на
    весь поток (не предыдущий трек — иначе поток цепочкой уплывает от выбранного).

    Смена стиля относительно опоры (MODE_TRACK) или последнего трека (MODE_MY) — только через
    шлюз М6; gate_log получает решения шлюза для журнала (ФТ-8.3). Если стили выбраны фильтром,
    выбор слушателя важнее шлюза. leave_style — слушатель просит уйти, а моста нет: шаг мимо шлюза из этого стиля.
    listen — чередование таблицы слушателей (решение 0015): "trial" — рукав по жребию, "A"/"B" — заданный (прогоны, тесты),
    None — без него (вложенные вызовы: рукав решается один раз на запрос). model — "listener": «Свой поток» на модели слушателя
    (решение 0016), "drift" — на правилах дрейфа (решение 0013). fresh — свежее (решение 0017, настройка fresh_boost): надбавка
    неслушанному недавно добавленному и стилям, где его много, при выборе цели маршрута; без неё маршрут выбирает цель без свежего."""
    f = filters or Filters()
    now_ms = now_ms or int(time.time() * 1000)
    rng = rng or random.Random()
    mood = mood or Mood()
    whole = [h for h in history[run_from:] if h in lib.tracks]  # этот запуск потока: по нему собирается дрейф
    hist = [h for h in history if h in lib.tracks][-HIST_WINDOW:]  # правила о повторах смотрят на недавнее, и до запуска тоже
    played = list(history)
    if current is not None and current in lib.tracks and (not hist or hist[-1] != current):
        hist.append(current)
        whole.append(current)
        played.append(current)  # дозагрузка шлёт последний трек очереди, которого нет в истории: остаток пачки не должен его повторить
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
    begin = None
    if start and anchor is None and taste is not None:
        pool = [t for t in prof.liked if t in lib.row] * 2 + \
            [t for t in prof.played if t in lib.row and prof.newness(t) < 0.5]
        # Начинаем с обычного; в пограничном состоянии — с одного из стилей дома по их долям. Стиль на пару десятков треков домом сессии быть не может
        at_home = {s: x for s, x in prof.homes.items() if lib.style_size.get(s, 0) >= MAIN_STYLE_MIN}
        if at_home and not f.narrow():
            begin = rng.choices(list(at_home), weights=list(at_home.values()))[0]
            pool = [t for t in pool if lib.tracks[t].style == begin] or pool
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
    if begin is not None:  # новое для начала сессии — в первую очередь из стиля, с которого начинаем
        never.sort(key=lambda t: lib.tracks[t].style != begin)
    # Модель слушателя берёт новое рядом (соседи принятых и вкуса): случайные новые со всей медиатеки почти все вне шага,
    # а признаки по ним считались зря (Pi 06.11.2026)
    lm_early = (model == "listener" and mode == MODE_MY and not (f.styles or f.genres) and not start and not relaxed
                and len(hist) >= 2)  # запасной путь (relaxed) идёт без модели: случайные новые ему нужны
    for tid in never[:0 if lm_early else 200]:
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
    # Только «Мой поток»: выбрал жанр, стиль или фильтр нарочно — играет выбранное, без подмешивания (владелец 04.11.2026)
    # Выбрал стиль или жанр — gated ложно и подмеса нет. Плитка звука охватывает разные стили: поток внутри неё живёт по тем же правилам,
    # иначе он заперт в стиле первого трека, сколько ни пропускай
    mixing = gated and mode == MODE_MY
    home = prof.home_style(lib)
    drift = None
    # Модель слушателя (решение 0016): одна оценка «зайдёт ли сейчас» вместо основного стиля, проб, ухода и мостов
    use_lm = mixing and model == "listener"
    lm = lm_state = beta = None
    lm_last_close = None  # близость всей медиатеки к прошлому треку, если он принят или играет: маршрут берёт её же
    if use_lm:
        lm = prof.listener_model(lib)
        lm_since = _run_since(history, run_from, mood)
        lm_state = LM.session_state(lib, whole, lambda t: [e for e in mood.by_track.get(t, []) if e[5] >= lm_since])
        beta = lm.sample(rng)  # выборка Томпсона: веса на этот запрос
        near_rows = [p.row for p in lm_state[-LM.WINDOW:] if p.outcome >= 0][-3:]  # в потоке модель держится рядом: соседи принятых
        if near_rows:
            close = lib.vec @ lib.vec[near_rows].T
            if lm_state and near_rows[-1] == lm_state[-1].row:
                lm_last_close = close[:, -1]
            top = min(LISTENER_NEAR_K, len(lib.ids))  # маленькая медиатека: соседей меньше, чем просим
            for k in range(close.shape[1]):
                near_idx = np.arange(len(lib.ids)) if top == len(lib.ids) else np.argpartition(-close[:, k], top - 1)[:top]
                for j in near_idx:
                    cands.setdefault(int(lib.ids[j]), "near")
    if mixing and not use_lm:  # основной стиль сессии, а не последнего трека: подмес его не сбивает, сдвигает только слушатель
        # Действия с прошлого запуска к этому не относятся: вчерашний лайк трека, что открыл сегодняшний поток, — не сдвиг. Граница по концу
        # прошлого трека, а не по времени запроса: часы устройства и сервера могут расходиться, и тогда пропали бы все действия сессии
        run_since = _run_since(history, run_from, mood)
        drift = D.follow(whole, lambda t: lib.tracks[t].style, lambda t: [e for e in mood.by_track.get(t, []) if e[5] >= run_since],
                         lambda style: lib.direction_prior(prof, style), home, f.boldness, prof.homes)
        if stepped and lib.tracks[whole[-1]].style != drift.main:
            drift._move_to(lib.tracks[whole[-1]].style)
        ref_style = drift.leaving or drift.main
        ref = next((h for h in reversed(hist) if lib.tracks[h].style == ref_style), ref)
    # участок потока в текущем стиле: по нему считается настрой (подмес участка не прерывает)
    run: set[int] = set()
    for i in range(len(hist) - 1, -1, -1):
        if lib.tracks[hist[i]].style == ref_style:
            run.add(hist[i])
        elif i == 0 or lib.tracks[hist[i - 1]].style != ref_style:
            break  # чужой трек между своими — подмес, участок продолжается
    restless = drift.restless if drift is not None else (mood.restless(run) if ref_style is not None else 0.0)
    homes = {home, *prof.homes}
    rejected = mood.rejected(homes)
    leave_now = drift is not None and drift.leaving is not None  # слушатель просит уйти
    cross_now = leave_now
    wanted_new = min(1.0, max(0.0, 1.0 - f.familiar))
    # Состояние модулятора доли: история из профиля плюс то, чего в ней ещё нет (очередь, свежие события)
    in_events = {tid for tid, _, _ in mood.trace}
    unseen = [h for h in hist if h not in in_events]
    served_count = prof.share_count + len(unseen)
    served_new = prof.share_new + sum(1 for h in unseen if prof.newness(h) >= 0.5)
    was_new = served_new / served_count if served_count else wanted_new
    # Запас стиля: сколько в нём знакомого и сколько нового: мост ведёт туда, где хватит нужного разряда
    def stock(style: str, new_side: bool) -> int:
        known_here = prof.known_by_style.get(style, 0)
        return max(0, lib.style_size.get(style, 0) - known_here) if new_side else known_here

    edge = wanted_new >= NEW_STRICT or wanted_new <= 1 - NEW_STRICT  # край ползунка: нужный разряд без исключений
    need_new = wanted_new >= NEW_STRICT if edge else was_new <= wanted_new
    # Край ползунка, а в основном стиле нужного разряда нет совсем. Ползунок важнее стиля: «выкрутил на 100 — знакомых быть не должно»
    forced_out = drift is not None and drift.leaving is None and (force_leave or (edge and stock(drift.main, need_new) == 0))
    if forced_out:
        drift.leaving = drift.main
        leave_now = cross_now = True
    bold = round(min(1.0, f.boldness + RESTLESS_BOLD * min(restless, 2.0)), 1)
    unloved: dict[str, bool] = {}

    def may_offer(style: str) -> bool:
        """Отвергнутый стиль: на этот запрос решается один раз — редко и только из близкого."""
        if not prof.style_refused(style):
            return True
        if style not in unloved:
            unloved[style] = (ref_style is not None and ref_style not in homes and lib.styles_close(ref_style, style)
                              and rng.random() < UNLOVED_CHANCE)
        return unloved[style]
    # Мост — только когда слушатель просит уйти. Сам по себе поток основной стиль не покидает (решение 0013).
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
            fit_of = {t: min(1.0, max(0.0, (c - lo) / (hi - lo + 1e-6))) for t, c in sims.items()}
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
    strict_known = None  # край ползунка в силе: True — только знакомое, False — только новое
    # крайние положения ползунка — отбором, не прибавкой: только незнакомое / только знакомое,
    # если такие кандидаты вообще есть (кончились — играет что есть, см. strict_new ниже)
    if strict_new and (target_new >= NEW_STRICT or target_new <= 1 - NEW_STRICT):
        want_known = target_new <= 1 - NEW_STRICT
        keep = {t: src for t, src in cands.items() if (prof.newness(t) < 0.5) == want_known}
        if keep:
            cands = keep
            strict_known = want_known

    # Чередование (решение 0015): рукав один на запрос. Мост по просьбе уйти — своя ветка, в сравнение не входит
    arm = None
    if listen and mode == MODE_MY and not can_bridge and lib.listen_ok.any():
        arm = listen if listen in ("A", "B") else ("B" if rng.random() < LISTEN_TRIAL_SHARE else "A")
        taste_c = _listen_center(lib, _listen_taste_weights(lib, prof))
        # сессия — только этот запуск; у старта своего запуска ещё нет, а плеер шлёт в истории прошлые (проверка 06.11.2026)
        sess_c = _listen_center(lib, {} if start else dict.fromkeys(_heard_in_run(whole, mood, _run_since(history, run_from, mood)), 1.0))
    cycle = prof.rest_days(1.0 - target_new)
    # шаг поиска: пока заходит — рядом с прошлым, после пропуска — в сторону; запасные пути (relaxed) шагом не ограничены
    lm_dwell = 0
    if use_lm and whole:
        last_style = lib.tracks[whole[-1]].style
        for t in reversed(whole):
            if lib.tracks[t].style != last_style:
                break
            lm_dwell += 1
    # дома стоянка вдвое длиннее (решение 0017): «пора дальше» и шаг считаются от половины. Удержание места — по настоящим трекам:
    # иначе дома оно включено с первого трека, и кто пропускает часть ДнБ, уходит из дома вдвое позже (независимая проверка 07.11.2026)
    lm_stand = lm_dwell // LM.HOME_DWELL if use_lm and whole and lib.tracks[whole[-1]].style == prof.home_style(lib) else lm_dwell
    lm_step = LM.Step.of(lib, lm_state, lm_stand) if use_lm and not start and not relaxed else None
    lm_flow = bool(lm_state) and lm_state[-1].outcome >= 0  # прошлый принят или ещё играет: не после пропуска
    lm_style = lib.tracks[whole[-1]].style if use_lm and whole else None
    # держаться нового места: стиль сменился в этой сессии меньше LISTENER_HOLD треков назад
    lm_hold = use_lm and lm_flow and not relaxed and lm_dwell < LISTENER_HOLD and len(whole) > lm_dwell
    lm_move = (min(1.0, max(0.0, (lm_stand - LM.DWELL_START + 1) / LM.DWELL_SPAN)) if use_lm and lm_flow and not relaxed else 0.0)
    lm_seen = Counter(lib.tracks[t].style for t in whole[-LISTENER_SEEN:]) if use_lm else Counter()
    # Маршрут (решение 0017): постояв на месте, поток идёт к цели — чаще к стилю без истории (с fresh — и туда, где много свежего) —
    # через переходные треки
    lm_route = None  # (стиль-цель, центр стиля, треков после жребия)
    lm_now_s = now_ms // 1000
    lm_fresh_scale = _fresh_scale(prof.style_pos) if use_lm and fresh else 0.0
    if use_lm and lm_flow and not relaxed and not start and lm_state:
        styles_ = [lib.tracks[int(lib.ids[p.row])].style for p in lm_state]
        home_ = prof.home_style(lib)
        origin = LM.route_origin(styles_, lm_state, home_)
        if origin is not None:
            s0, e0, seed_at = origin
            steps = len(lm_state) - 1 - seed_at  # от жребия: стоянка, которая не может уйти, тоже тратит шаги
            if steps <= LM.ROUTE_STEPS:
                # Счёт стилей — на момент жребия: события после него (трек цели посреди пачки, переходные, что зашли) цель не двигают
                later = {int(lib.ids[p.row]) for p in lm_state[seed_at:]}
                cut = min((e[5] for t in later for e in mood.by_track.get(t, []) if e[5] >= lm_since), default=None)
                route_pos, route_neg = prof.style_stats_before(cut) if cut is not None else (prof.style_pos, prof.style_neg)
                reachable = (_route_reachable(lib, prof, f, allowed, strict_known)
                             if allowed is not None or strict_known is not None or f.narrow() else None)
                # «уже звучал» — за LISTENER_SEEN треков до конца стоянки: окно привязано к стоянке, в пути не сдвигается
                # доля свежего неслушанного — тоже на момент жребия: слушанное после него и смена дня цель не двигают
                fresh_ = (_fresh_share(lib, prof, cut // 1000 if cut is not None else lm_now_s, played=prof.played - later)
                          if fresh else {})
                dest = _route_to(lib, prof, (int(lib.ids[lm_state[s0].row]), int(lib.ids[lm_state[seed_at].row])),
                                 set(styles_[max(0, e0 + 1 - LISTENER_SEEN):e0 + 1]), fresh_, styles_[e0],
                                 route_pos, route_neg, reachable)
                if dest is not None and styles_[-1] != dest:  # пришли — когда последний трек очереди в стиле цели
                    lm_route = (dest, lib.style_center(dest), steps, styles_[e0], int(lib.ids[lm_state[seed_at].row]))
    lm_toward: dict[int, float] = {}
    if lm_route is not None:
        lm_hold, lm_move = False, 0.0  # в пути место не держим и к соседям не толкаем: ведёт цель
        # шаг в пути: сначала посередине между «тем же местом» и «соседним» — переходные треки, а не прыжок; к середине пути — до
        # «соседнего», иначе из далёких мест (Nordic → ДнБ) поток топтался на месте и бросал маршрут (независимая проверка 07.11.2026)
        wide = LM.ROUTE_STEP_WIDE + (1 - LM.ROUTE_STEP_WIDE) * min(1.0, max(0, lm_route[2]) / (LM.ROUTE_STEPS / 2))
        lm_step = LM.Step.of(lib, lm_state, LM.DWELL_START - 1 + round(LM.DWELL_SPAN * wide))
        to_last = lm_last_close if lm_last_close is not None else lib.vec @ lib.vec[lm_state[-1].row]
        to_dest = lib.style_closeness(lm_route[0])
        lm_route_at = float(to_dest[lm_state[-1].row])
        ok_ = np.flatnonzero(lm_step.allows_many_rows(to_last))
        if strict_known is not None:  # край ползунка: в пути только его разряд
            ok_ = np.array([j for j in ok_ if (prof.newness(int(lib.ids[j])) < 0.5) == strict_known], dtype=int)
        path_ = ok_[np.argsort(-to_dest[ok_])[:ROUTE_K]]
        for j in path_:  # кандидаты пути: соседнее место, ближайшие к цели
            cands.setdefault(int(lib.ids[j]), "route")
        if trace is not None:  # в журнал: куда, сколько пройдено, откуда, сколько кандидатов пути отобрано, держится ли место
            trace["route"] = {"to": lm_route[0], "steps": lm_route[2], "from": lm_route[3], "seed": lm_route[4],
                              "cands": int(len(path_)), "hold": bool(lm_hold)}
    lm_ahead = None  # направление хода: от середины недавних треков к прошлому
    if use_lm and lm_flow and not relaxed and len(whole) > 2:
        came = np.mean([lib.vec[lib.row[t]] for t in whole[-1 - LISTENER_DRIFT_BACK:-1]], axis=0)
        way = lib.vec[lib.row[whole[-1]]] - came
        if float(np.linalg.norm(way)) > 1e-6:
            lm_ahead = (lib.vec[lib.row[whole[-1]]], way / float(np.linalg.norm(way)))
    lm_gain: dict[int, float] = {}  # лог-шансы «зайдёт» по весам этого запроса
    lm_q: dict[int, float] = {}  # вероятность пропуска по средним весам
    lm_sims = lm_ids = None
    lm_keep: set[int] = set()
    lm_inside: set[int] = set()
    lm_src: dict[int, str] = {}  # откуда кандидат: подпись собирается только выбранным
    lm_drift: dict[int, float] = {}
    if use_lm:
        def eligible(t: int) -> bool:  # те же проверки, что в цикле: отбор лучших по модели — только среди годных
            return (t not in prof.disliked and (allowed is None or t in allowed) and t not in lib.dj_tools
                    and (allowed is not None or lib.best_edition.get(t, t) == t) and _filter_ok(lib, t, f))
        ids = [t for t in cands if t in lib.row and eligible(t)]
        lm_ids = {t: i for i, t in enumerate(ids)}
        lm_rows = np.array([lib.row[t] for t in ids], dtype=int)
        lm_sims = LM.Sims(lib, lm_rows)
        feats = LM.features(lib, lm, lm_state, lm_rows, sims=lm_sims)
        gains = -lm.logit(feats, beta)
        lm_gain = dict(zip(ids, gains.tolist()))
        lm_q = dict(zip(ids, lm.p_skip(feats).tolist()))
        # Что разрешено сейчас (шаг, удержание места) — сразу для всех: полный расчёт только для них и для лучших вне шага,
        # которые нужны запасному ходу. Иначе через цикл шли все знакомые треки медиатеки (Pi 06.11.2026: 158 мс против 52 у дрейфа)
        inside = lm_step.allows_many(lm_sims) if lm_step is not None else np.ones(len(ids), dtype=bool)
        if lm_hold:
            inside &= np.array([lib.tracks[t].style == lm_style for t in ids])
        drift_v = None
        if lm_ahead is not None:  # инерция для всех разом: косинус шага к кандидату с направлением хода
            # векторы единичные: |v − x|² = 2 − 2·(v·x), а (v − x)·d = v·d − x·d — без матрицы разностей
            norm = np.sqrt(np.maximum(2.0 - 2.0 * lm_sims.col(lib.row[whole[-1]]), 0.0))
            ahead = lm_sims.V @ lm_ahead[1].astype(np.float32) - float(lm_ahead[0] @ lm_ahead[1])
            drift_v = np.where(norm > 1e-6, ahead / np.maximum(norm, 1e-6), 0.0)
            lm_drift = dict(zip(ids, drift_v.tolist()))
        rest = np.flatnonzero(~inside)
        # Запасной ход нужен на случай пропуска: и пул для него — лучшие в состоянии «играющий пропустили», разрешённые тем шагом.
        # По оценке «держаться рядом» в пул шли треки сразу за порогом у того самого трека (независимая проверка 06.11.2026)
        spare = rest[np.argsort(-gains[rest])[:LISTENER_SPARE]]
        if lm_state and len(rest):
            skip_state = lm_state[:-1] + [LM.Play(lm_state[-1].row, -1)]
            skip_gain = -lm.logit(LM.features(lib, lm, skip_state, lm_rows, sims=lm_sims), beta)
            skip_ok = LM.Step.of(lib, skip_state).allows_many(lm_sims)
            fit = rest[skip_ok[rest]]
            spare = fit[np.argsort(-skip_gain[fit])[:LISTENER_SPARE]]
            if len(spare) < LISTENER_SPARE:
                spare = np.concatenate([spare, rest[np.argsort(-skip_gain[rest])[:LISTENER_SPARE - len(spare)]]])
        # В цикл внутри шага идут лучшие каждого разряда (доле нового нужны и новые, и знакомые), а ещё лучшие других стилей,
        # когда пора дальше, и лучшие по ходу: иначе толчку и инерции не на кого действовать (прогон 06.11.2026: 95 % на месте)
        fresh = np.array([prof.newness(t) >= 0.5 for t in ids])
        keep_in = []
        groups = [(inside & fresh, gains), (inside & ~fresh, gains)]
        if lm_move > 0:
            groups.append((inside & np.array([lib.tracks[t].style != lm_style for t in ids]), gains))
        if drift_v is not None:
            groups.append((inside, drift_v))
        if lm_route is not None:  # к цели: приближение = близость кандидата к центру цели минус близость прошлого трека
            toward = lm_sims.V @ lm_route[1] - lm_route_at
            lm_toward = dict(zip(ids, toward.tolist()))
            groups.append((inside, toward))
        for mask, by in groups:
            at_ = np.flatnonzero(mask)
            keep_in.extend(at_[np.argsort(-by[at_])[:LISTENER_INSIDE]].tolist())
        lm_inside = {ids[i] for i in keep_in}
        lm_keep = lm_inside | {ids[i] for i in spare}
    scored: list[Pick] = []
    outside: list[Pick] = []
    for tid, src in cands.items():
        if use_lm and tid not in lm_keep:
            continue
        if tid in prof.disliked or (allowed is not None and tid not in allowed):
            continue
        if tid in lib.dj_tools:  # голая вокальная дорожка и часовой микс посреди потока не нужны
            continue
        if allowed is None and lib.best_edition.get(tid, tid) != tid:  # играет лучшее издание (ФТ-4.8)
            continue
        if not _filter_ok(lib, tid, f):
            continue
        if leave_style is not None and (lib.tracks[tid].style == leave_style or lib.tracks[tid].style in rejected
                                        or prof.style_refused(lib.tracks[tid].style)):
            continue
        bridge = None
        if gated and not use_lm and lib.tracks[tid].style != ref_style:
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
        parts: dict[str, float] = {}
        hub = HUB_WEIGHT * float(lib.hub[lib.row[tid]])  # поправка CSLS к близости (см. hub_scores)
        if use_lm:  # вкус, сессия, усталость, переходы и темп — внутри модели слушателя
            parts["listener"] = LISTENER_W * lm_gain.get(tid, 0.0)
            style = lib.tracks[tid].style
            if tid in lm_drift:  # инерция: продолжать в ту сторону, куда уже шли
                parts["drift"] = LISTENER_DRIFT_W * lm_drift[tid]
            if lm_move > 0 and style != lm_style and not prof.style_refused(style):
                # пора дальше: туда, где стиль приживается и чего в этой сессии ещё не было
                novelty = 1.0 - lm_seen.get(style, 0) / max(1, min(len(whole), LISTENER_SEEN))
                parts["move"] = LISTENER_MOVE_W * lm_move * prof.style_accept(style) * novelty
            if tid in lm_toward:
                parts["route"] = ROUTE_W * lm_toward[tid]
            if lm_fresh_scale > 0:  # свежее — только с fresh_boost
                fr = _freshness(lib, prof, tid, lm_now_s) * lm_fresh_scale
                if fr > 0:
                    parts["fresh"] = FRESH_W * fr
        elif anchor is not None:
            parts["taste"] = 1.2 * (float(v @ target) - hub)
            if taste is not None:
                parts["taste"] += 0.3 * (float(v @ taste) - hub)
        elif taste is not None:
            parts["taste"] = float(v @ taste) - hub
        else:
            parts["taste"] = rng.random() * 0.1  # нет ни опоры, ни истории — первые треки знакомят с медиатекой
        if neg is not None:
            parts["neg"] = -0.8 * float(v @ neg)
        if mood.neg is not None and not use_lm:
            parts["mood"] = -SESSION_NEG_W * float(v @ mood.neg)
        if ref is not None and (ref, tid) in prof.pair_penalty:
            parts["pair"] = -PAIR_PENALTY * prof.pair_penalty[(ref, tid)]
        if prev is not None and not relaxed:  # сведение с предыдущим треком (М7)
            ra, rb = lib.row[prev], lib.row[tid]
            if G.energy_jump(lib.gate, ra, rb):
                continue  # ФТ-7.2: резкий скачок энергии
            if lib.gate.tempo_ok(ra, rb) and not use_lm:
                parts["tempo"] = TEMPO_BONUS
        if recent_vecs and not use_lm:  # MMR: близко к стилю, но не копия последних треков
            parts["mmr"] = -MMR_WEIGHT * max(float(v @ r) for r in recent_vecs)
        rk = lib.tracks[tid].recording_key
        last = prof.last_played.get(rk)
        if last is not None:  # память о прослушанном: недавнее и частое — реже, кроме повтора «в тему»
            days = max(0.0, (now_ms - last) / 86_400_000)
            parts["repeat"] = -(REPEAT_W * 0.5 ** (days / max(REPEAT_HALF_LIFE_DAYS, cycle / 2)) * (1.0 - fit_of.get(tid, 0.0))
                                + PLAYS_W * float(np.log1p(prof.plays.get(rk, 0))))
        nw = prof.newness(tid)
        parts["share"] = NEW_W * (new_deficit * nw + known_deficit * (1.0 - nw)) + (f.familiar - 0.5) * 0.4 * (1.0 - 2.0 * nw)
        parts["skip"] = -prof.skip_penalty(tid)  # пропускал — ниже в очереди, тем сильнее, чем раньше и свежее
        if tid in prof.liked:
            parts["liked"] = 0.15
        phi = None
        if arm is not None:
            phi = (_listen_phi(lib, tid, sess_c), _listen_phi(lib, tid, taste_c))
            if arm == "B":
                parts["listen"] = LISTEN_W * (phi[0] + phi[1])
        s = sum(parts.values())
        seed_title = lib.tracks[anchor].title if anchor is not None else None
        if use_lm and not relaxed:  # подпись — только выбранным (_named): за запрос кандидатов сотни, а нужны три-четыре
            why, reason = "", ""
            lm_src[tid] = src
        else:
            why = _why(tid, anchor, prof)
            reason = _reason(lib, tid, anchor, src, prof)
        if relaxed and ref is not None and lib.tracks[tid].style != ref_style:
            why, reason = "restyle", f"{lib.tracks[tid].style} · в «{ref_style}» больше нечего предложить"
            if leave_style is not None:
                why, reason = "leave", f"{lib.tracks[tid].style} · вы пропускали «{leave_style}»"
        if bridge is not None:
            why = "bridge"
            seed_title = lib.tracks[ref].title
            reason = f"{lib.tracks[tid].style} · {G.bridge_reason(bridge)} с «{lib.tracks[ref].title or 'предыдущим'}»"
        pick = Pick(tid, reason, s, why, seed_title, parts=parts, phi=phi, q=lm_q.get(tid))
        if use_lm and tid not in lm_inside:
            outside.append(pick)  # вне шага сейчас, но годится для запасного хода «если пропустят»
        else:
            scored.append(pick)

    if gate_log is not None and ref is not None:
        gate_log.extend((ref, tid, v) for tid, v in verdicts.items())
    if trace is not None:
        trace.update(mode=mode, start=start, familiar=round(f.familiar, 3), boldness=round(f.boldness, 3), filtered=f.narrow(),
                     share_new=round(was_new, 3), slider_new=round(wanted_new, 3), target_new=round(target_new, 3),
                     candidates=len(scored), restless=round(restless, 2))
        if use_lm:
            trace.update(v=LISTENER_VERSION, homes={s: round(x, 3) for s, x in prof.homes.items()},
                         listener={"beta": dict(zip(("bias", *LM.FEATURES), (round(float(b), 3) for b in beta))), "rows": lm.rows,
                                   "skips": lm.skips, "last": lm_state[-1].outcome if lm_state else None})
        if drift is not None:
            trace.update(main=drift.main, leaving=drift.leaving, explore=round(drift.explore, 4), moves=drift.moves,
                         homes={s: round(x, 3) for s, x in prof.homes.items()},
                         directions={s: round(x, 3) for s, x in sorted(drift.directions.items(), key=lambda kv: -kv[1])[:5]})
        trace["alternatives"] = [{"t": p.track_id, "s": round(p.score, 4), "new": prof.newness(p.track_id) >= 0.5,
                                  "p": {k: round(x, 3) for k, x in p.parts.items()}}
                                 for p in sorted(scored, key=lambda p: -p.score)[:JOURNAL_ALTERNATIVES]]
    if can_bridge:
        def then(p: Pick) -> list[Pick]:
            rest = next_tracks(lib, prof, current=p.track_id, history=[*played, p.track_id], filters=filters, n=n - 1,
                               now_ms=now_ms, allowed=allowed, rng=rng, mode=mode, seed=seed,
                               strict_new=strict_new, mood=mood, run_from=run_from, stepped=True, model=model, fresh=fresh) if n > 1 else []
            return [p] + rest

        # Куда вести: сначала принятые пробы, потом дом. Без случайности: слушатель просит, а не пробует, и известное место есть.
        # Только если уходит из единственного дома, известного места нет: выборка Томпсона. Есть мост туда — по нему; нет — ближайший по ощущению трек
        # или лучший в этом стиле (решение 0013). Откуда только что ушли по отказам, туда не ведём.
        passes = {lib.tracks[t].style for t in lib.tracks if _filter_ok(lib, t, f)} if f.narrow() else None

        def enough(style: str) -> bool:
            return (lib.style_size.get(style, 0) >= MAIN_STYLE_MIN and style not in rejected and may_offer(style)
                    and not drift.refused_lately(style) and (passes is None or style in passes)
                    and (not edge or stock(style, need_new) > 0))

        direct = [p for p in scored if p.why == "bridge" and enough(lib.tracks[p.track_id].style)
                  and _artist_ok(lib, p.track_id, hist, ARTIST_GAP)]
        worth = (lambda s: _thompson(prof, rng, s)) if homes == {ref_style} else prof.style_accept
        options = {s: x * worth(s) for s, x in drift.directions.items() if enough(s) and s != ref_style}
        if options:
            target = max(options, key=options.get)
            there = [p for p in direct if lib.tracks[p.track_id].style == target]
            if there:
                there.sort(key=lambda p: ((prof.newness(p.track_id) >= 0.5) != need_new, -p.score))
                return then(there[0])
            step = _probe(lib, prof, ref, hist, f, rng, mood, drift, allowed, now_ms, rejected, may_offer, need_new, target=target, edge=edge)
            if step is not None:
                style = lib.tracks[step.track_id].style
                step.why, step.reason = (("restyle", f"{style} · в «{ref_style}» больше нечего предложить") if forced_out
                                         else ("leave", f"{style} · вы пропускали «{ref_style}»"))
                step.prob, step.track_prob, step.slot = None, None, None
                return then(step)
        if direct:
            direct.sort(key=lambda p: ((prof.newness(p.track_id) >= 0.5) != need_new, -p.score))
            return then(direct[0])
        way_out = next_tracks(lib, prof, current=current, history=history, filters=filters, n=1, now_ms=now_ms,
                              allowed=allowed, rng=rng, mode=mode, seed=seed, relaxed=True, strict_new=strict_new,
                              mood=mood, leave_style=ref_style, run_from=run_from, model=model, fresh=fresh)
        if way_out:
            return then(way_out[0])
    def assemble(scored: list[Pick], rng: random.Random) -> tuple[list[Pick], list[Pick]]:
        """Пачка без проб: порядок, разнесение исполнителей, доля нового. Второе — кандидаты после упорядочивания (пусто — нечего
        предложить, ниже запасные пути). Функция, а не кусок кода: чередование (решение 0015) собирает пачку и за другой рукав."""
        if begin is not None:
            scored = [p for p in scored if lib.tracks[p.track_id].style == begin] or scored
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
        if anchor is None:
            rest = prof.rest_days(1.0 - target_new)
            fresh, known = [], []
            for p in scored:
                (fresh if prof.newness(p.track_id) >= 0.5 else known).append(p)
            urgent = {p.track_id: prof.urgency(lib, p.track_id, now_ms, rest) for p in known}

            def today(tid: int) -> bool:
                last = prof.last_played.get(lib.tracks[tid].recording_key)
                return last is not None and now_ms - last < 12 * 3_600_000

            if use_lm:  # из тех, кому пора, — что сейчас зайдёт: иначе знакомые тянут туда, где их больше всего, мимо модели
                known.sort(key=lambda p: -p.score)
            else:
                known.sort(key=lambda p: -urgent[p.track_id] * (1.0 + ROT_FIT_W * fit_of.get(p.track_id, 0.0)))
            ripe = prof.ripe_urgency(lib, now_ms, rest)
            ready = [p for p in known if urgent[p.track_id] >= ripe and not today(p.track_id)]
            early = [p for p in known if urgent[p.track_id] < ripe and not today(p.track_id)]
            played_today = [p for p in known if today(p.track_id)]
            classes = (fresh, ready, early, played_today)
            scored = fresh + ready + early + played_today
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
                return out, scored
        return [], scored

    counter = None
    if arm is not None:  # что выбрал бы другой рукав: тот же жребий (копия генератора), другая оценка
        twin = random.Random()
        twin.setstate(rng.getstate())
        if arm == "B":
            other = [replace(p, score=p.score - p.parts.get("listen", 0.0)) for p in scored]
        else:
            other = [replace(p, score=p.score + (LISTEN_W * (p.phi[0] + p.phi[1]) if p.phi else 0.0)) for p in scored]
        counter, _ = assemble(other, twin)
    out, scored = assemble(scored, rng)
    if out and lm_route is not None:
        # в пути ближайший к цели — последним: плеер берёт пачку целиком и спрашивает дальше от последнего трека очереди
        c_ = lib.style_closeness(lm_route[0])
        out.sort(key=lambda p: float(c_[lib.row[p.track_id]]))
    if out and use_lm and trace is not None:
        trace["on_skip"] = _on_skip(lib, prof, lm, lm_state, beta, scored + outside, out, hist, now_ms, lm_sims, lm_ids)
    if use_lm and lm_src:
        for p in [*out, *((trace or {}).get("on_skip") or {}).values()]:
            if not p.reason:
                p.why, p.reason = _why(p.track_id, anchor, prof), _reason(lib, p.track_id, anchor, lm_src.get(p.track_id, "taste"), prof)
    if out:
        main = [p.track_id for p in out]
        if drift is not None and not start:
            out = _mix_in_probes(lib, prof, out, hist, f, rng, mood, drift, allowed, now_ms, rejected, may_offer, need_new, edge, n)
        if trace is not None and arm is not None:
            # отдельным блоком: в trace уже есть свой main — основной стиль сессии (стенд 06.11.2026 поймал затирание)
            trace["listen"] = {"arm": arm, "w": LISTEN_W, "main": main, "counter": [p.track_id for p in counter or []]}
        return out
    if trace is not None:
        trace.pop("route", None)  # ответ придёт из запасного вызова: маршрута этого запроса в нём нет
    if strict_new and not scored and (target_new >= NEW_STRICT or target_new <= 1 - NEW_STRICT):
        # Нужный разряд в этом стиле кончился: все такие треки только что звучали. Край ползунка важнее стиля, поэтому туда, где он есть:
        # сначала принятые пробы, потом дом, как при просьбе уйти. Нет — любой другой стиль. Нигде нет — что есть, не тишина.
        if drift is not None and not force_leave and not stepped:
            out = next_tracks(lib, prof, current=current, history=history, filters=filters, n=n, now_ms=now_ms, allowed=allowed,
                              rng=rng, mode=mode, seed=seed, start=start, mood=mood, run_from=run_from, force_leave=True, model=model, fresh=fresh)
            if out:
                return out
        if not relaxed and mode == MODE_MY and (gated or prev is not None):
            out = next_tracks(lib, prof, current=current, history=history, filters=filters, n=n, now_ms=now_ms, allowed=allowed,
                              rng=rng, mode=mode, seed=seed, relaxed=True, start=start, mood=mood, run_from=run_from, model=model, fresh=fresh)
            if out:
                return out
        return next_tracks(lib, prof, current=current, history=history, filters=filters, n=n, now_ms=now_ms,
                           allowed=allowed, rng=rng, mode=mode, seed=seed, gate_log=gate_log, relaxed=relaxed,
                           start=start, strict_new=False, mood=mood, run_from=run_from, model=model, fresh=fresh)
    if not relaxed and mode == MODE_MY and (gated or prev is not None):
        # шлюз и сведение отсекли всё (маленький стиль исчерпан): лучше честная смена стиля, чем тишина, причина говорит об этом прямо.
        # Сначала туда же, куда при просьбе уйти: иначе каждый трек выбирается без шлюза и проб, и поток скачет по стилям
        if drift is not None and not force_leave and not stepped:
            out = next_tracks(lib, prof, current=current, history=history, filters=filters, n=n, now_ms=now_ms, allowed=allowed,
                              rng=rng, mode=mode, seed=seed, start=start, mood=mood, run_from=run_from, force_leave=True, model=model, fresh=fresh)
            if out:
                return out
        return next_tracks(lib, prof, current=current, history=history, filters=filters, n=n, now_ms=now_ms,
                           allowed=allowed, rng=rng, mode=mode, seed=seed, gate_log=None, relaxed=True, start=start,
                           mood=mood, run_from=run_from, model=model, fresh=fresh)
    return []


def _route_to(lib: Library, prof: Profile, origin: tuple[int, int], heard: set[str], fresh_share: dict[str, float],
              stand_style: str, pos: dict[str, float] | None = None, neg: dict[str, float] | None = None,
              reachable: set[str] | None = None) -> str | None:
    """Цель маршрута (решение 0017). Со стоянки не дома — домой с вероятностью, равной доле дома среди принятого (от 0,3 до 0,8):
    дом держится, а поток ходит. Иначе — выборка Томпсона по стилям: доля «зашло» из истории, бета-распределение по дослушанным
    и пропущенным; у стиля без истории оно широкое. Надбавка за долю свежего — с настройкой fresh_boost, без неё fresh_share пустой.
    Стили, звучавшие за LISTENER_SEEN треков до конца
    стоянки, устойчиво незаходящие и недостижимые под ограничениями запроса (reachable) — не цель. Жребий привязан к стоянке (origin —
    id её первого трека и трека жребия) и у каждого стиля свой; счёт стилей pos/neg — на момент жребия: пока поток идёт к цели, все запросы
    видят ту же цель."""
    pos = prof.style_pos if pos is None else pos
    neg = prof.style_neg if neg is None else neg
    def coin(tag: str) -> random.Random:
        return random.Random(int.from_bytes(hashlib.blake2b(f"{origin[0]}:{origin[1]}:{tag}".encode(), digest_size=8).digest(), "big"))
    home = prof.home_style(lib)
    accepted = sum(max(0.0, x) for x in pos.values())
    back = min(LM.ROUTE_HOME_MAX, max(LM.ROUTE_HOME_MIN, max(0.0, pos.get(home, 0.0)) / accepted)) if accepted > 0 else LM.ROUTE_HOME_MIN
    if (stand_style != home and len(lib.route_styles().get(home, ())) >= LM.ROUTE_MIN and not prof.style_refused(home)
            and (reachable is None or home in reachable) and coin("домой").random() < back):
        return home
    best, best_w = None, 0.0
    for style, rows in sorted(lib.route_styles().items()):
        if (len(rows) < LM.ROUTE_MIN or style in heard or prof.style_refused(style) or (reachable is not None and style not in reachable)
                or lib.route_capacity(style) < LM.DWELL_START):
            continue
        theta = coin(style).betavariate(max(0.0, pos.get(style, 0.0)) + 1.0, max(0.0, neg.get(style, 0.0)) + 1.0)
        w = theta * (1.0 + LM.ROUTE_FRESH * _fresh_scale(pos) * fresh_share.get(style, 0.0))
        if w > best_w:
            best, best_w = style, w
    return best


def first_import_end(added) -> int:
    """Когда кончился первый большой разбор медиатеки: от первого трека, пока между добавлениями нет перерыва дольше
    LM.FIRST_IMPORT_GAP_H часов. «Самый частый день» ломался, если разбор пересёк полночь или новый диск больше первого
    (независимая проверка 07.11.2026). Пусто — 0."""
    times = sorted(int(a) for a in added if a)
    if not times:
        return 0
    end = times[0]
    for a in times[1:]:
        if a - end > LM.FIRST_IMPORT_GAP_H * 3600:
            break
        end = a
    return end


def _route_reachable(lib: Library, prof: Profile, f: Filters, allowed: set[int] | None, strict_known: bool | None) -> set[str]:
    """Стили, где хватает треков (LM.ROUTE_MIN), проходящих ограничения запроса: «только треки устройства», плитку, фильтры, край ползунка.
    Иначе цель недостижима, и поток качается у границы (независимая проверка 07.11.2026)."""
    out = set()
    ids = lib.ids
    for style, rows in lib.route_styles().items():
        n = 0
        for r in rows:
            t = int(ids[r])
            if ((allowed is None or t in allowed) and t not in prof.disliked and _filter_ok(lib, t, f)
                    and (strict_known is None or (prof.newness(t) < 0.5) == strict_known)):
                n += 1
                if n >= LM.ROUTE_MIN:
                    out.add(style)
                    break
    return out


def _fresh_scale(pos: dict[str, float]) -> float:
    """Сколько надбавки за свежее у этого слушателя: у нового неслушанное всё, и надбавка за недавно добавленное становилась главной силой
    выбора — у гостя без истории 95 % потока шло из новых покупок владельца (07.11.2026). Растёт с принятым до LM.FRESH_FULL_AT."""
    return min(1.0, max(0.0, sum(max(0.0, x) for x in pos.values())) / LM.FRESH_FULL_AT)


def _freshness(lib: Library, prof: Profile, tid: int, now_s: int) -> float:
    """Свежесть 0…1 (решение 0017): неслушанный трек, добавленный после первого большого разбора, первые LM.FRESH_DAYS дней."""
    at = lib.added_at.get(tid, 0)
    if at <= lib.first_import_end or prof.newness(tid) < 0.5:
        return 0.0
    return min(1.0, max(0.0, 1.0 - (now_s - at) / 86_400 / LM.FRESH_DAYS))


def _fresh_share(lib: Library, prof: Profile, at_s: int, played: set[int] | None = None) -> dict[str, float]:
    """Доля свежего неслушанного в каждом стиле — для выбора цели маршрута. Свежие треки стиля — по датам добавления, раз на медиатеку и
    день (перебор всей медиатеки на каждый профиль стоил на Pi десятки миллисекунд, решение 0017, M6); неслушанные — по played, на момент
    жребия: стиль, где всё свежее уже прослушано, надбавки не тянет (независимая проверка 07.11.2026)."""
    day = at_s // 86_400
    memo = lib.__dict__.setdefault("_fresh_ids", {})
    fresh = memo.get(day)
    if fresh is None:
        added, first, ids = lib.added_at, lib.first_import_end, lib.ids
        fresh = {}
        for style, rows in lib.route_styles().items():
            tids = [int(ids[r]) for r in rows]
            fresh[style] = (len(tids), [t for t in tids if added.get(t, 0) > first and at_s - added.get(t, 0) < LM.FRESH_DAYS * 86_400])
        if len(memo) >= 3:
            memo.clear()
        memo[day] = fresh
    played = prof.played if played is None else played
    return {style: (sum(1 for t in tids if t not in played) / n if n else 0.0) for style, (n, tids) in fresh.items()}


def _on_skip(lib: Library, prof: Profile, lm: LM.Model, state: list[LM.Play], beta: np.ndarray, scored: list[Pick],
             out: list[Pick], hist: list[int], now_ms: int, sims: "LM.Sims | None" = None,
             index: dict[int, int] | None = None) -> dict[int, Pick]:
    """Запасной ход (решение 0016): для играющего трека и каждого трека пачки — что ставить следующим, если его пропустят.
    Тот же выбор в состоянии, где этот трек пропущен: прочие части оценки те же, меняются только лог-шансы модели слушателя.
    Плеер ставит запасной трек сразу после пропуска, без нового запроса и без ожидания второго пропуска."""
    chosen = {p.track_id for p in out}
    pool = [p for p in scored if p.track_id not in chosen and p.track_id in lib.row and (index is None or p.track_id in index)]
    if not pool:
        return {}
    rows = np.array([lib.row[p.track_id] for p in pool], dtype=int)
    at = np.array([index[p.track_id] for p in pool], dtype=int) if index is not None and sims is not None else None
    sub = sims.subset(at) if at is not None else None  # близости только пула, уже посчитанные — не пересчитываются
    # без частей, посчитанных для состояния «пока заходит»: инерции и толчка «пора дальше» после пропуска нет и в основном выборе
    base = np.array([p.score - p.parts.get("listener", 0.0) - p.parts.get("drift", 0.0) - p.parts.get("move", 0.0)
                     - p.parts.get("route", 0.0) for p in pool])  # маршрут после пропуска снят (решение 0017)
    cur = hist[-1] if hist else None
    before = list(state)
    cur_play = LM.Play(lib.row[cur], 0) if cur in lib.row else None
    if cur is not None and before and before[-1].row == lib.row.get(cur):
        cur_play = before[-1]  # настоящий исход играющего: после «не предлагать» это отказ, а не «принят» (независимая проверка 06.11.2026)
        before = before[:-1]
    steps = ([(cur, [])] if cur is not None else []) + [(p.track_id, [q.track_id for q in out[:i]]) for i, p in enumerate(out)]
    result: dict[int, Pick] = {}
    for x, ahead in steps:
        if x not in lib.row:
            continue
        now = before + ([cur_play] if cur_play is not None and x != cur else [])
        now = now + [LM.Play(lib.row[t], 0) for t in ahead] + [LM.Play(lib.row[x], -1)]
        feats = LM.features(lib, lm, now, rows, sims=sub)
        gain = -lm.logit(feats, beta)
        score = base + LISTENER_W * gain
        q = lm.p_skip(feats)
        seq = list(hist) + ([] if x == cur else [*ahead, x])
        step = LM.Step.of(lib, now)
        order = np.argsort(-score)
        for strict in (True, False):  # в шаге никого — лучший без шага: отталкивание от пропущенного есть и в самой оценке
            for k in order:
                t = pool[k].track_id
                if t == x or t in seq or (strict and step is not None and not step.allows(lib.vec[rows[k]])):
                    continue
                if _artist_ok(lib, t, seq, ARTIST_GAP) and _album_ok(lib, t, seq) and _recording_ok(lib, t, seq, prof, now_ms):
                    result[x] = replace(pool[k], score=float(score[k]), q=float(q[k]),
                                        parts={**pool[k].parts, "listener": float(LISTENER_W * gain[k])})
                    break
            if x in result:
                break
    return result


def _mix_in_probes(lib: Library, prof: Profile, picks: list[Pick], hist: list[int], f: Filters, rng: random.Random, mood: Mood,
                   drift: D.Drift, allowed: set[int] | None, now_ms: int, rejected: set[str], may_offer, need_new: bool,
                   edge: bool, n: int) -> list[Pick]:
    """Места отдаются пробам с частотой π: каждое место — основному, если очередь пробы ещё не подошла."""
    since = 0
    for h in reversed(hist):
        if lib.tracks[h].style != drift.main:
            break
        since += 1
    out: list[Pick] = []
    seq, queue = list(hist), list(picks)
    gap, first_slot = drift.gap(rng), True
    while queue and len(out) < n:
        since += 1
        if since >= gap and len(seq) >= PROBE_MIN_TRACKS:
            p = _probe(lib, prof, seq[-1], seq, f, rng, mood, drift, allowed, now_ms, rejected, may_offer, need_new, edge=edge)
            if p is not None:
                p.slot = drift.gap_chance(since, first_slot)
                out.append(p)
                seq.append(p.track_id)
                since, gap, first_slot = 0, drift.gap(rng), True
                continue
        first_slot = False
        p = queue.pop(0)
        out.append(p)
        seq.append(p.track_id)
    return out


def _direction_chance(prof: Profile, base: dict[str, float], style: str, rng: random.Random) -> float:
    """Настоящий шанс направления: выбор случаен и сам опирается на случайную выборку, поэтому усредняем по ней, а не берём один раз."""
    names = list(base)
    a = np.array([1.0 + prof.style_pos.get(s, 0.0) for s in names])
    b = np.array([1.0 + prof.style_neg.get(s, 0.0) for s in names])
    draws = np.random.default_rng(rng.getrandbits(32)).beta(a, b, size=(PROB_DRAWS, len(names))) * np.array([base[s] for s in names])
    return float((draws[:, names.index(style)] / draws.sum(axis=1)).mean())


def _thompson(prof: Profile, rng: random.Random, style: str) -> float:
    """Случайная оценка приживаемости стиля из Beta(дослушано + 1, отказано + 1): незнакомые стили получают шанс, устойчиво
    не зашедшие почти не звучат. Выборка, а не среднее: так разведка сама затухает там, где всё уже ясно."""
    return rng.betavariate(1.0 + prof.style_pos.get(style, 0.0), 1.0 + prof.style_neg.get(style, 0.0))


def _probe(lib: Library, prof: Profile, after: int, hist: list[int], f: Filters, rng: random.Random, mood: Mood, drift: D.Drift,
           allowed: set[int] | None, now_ms: int, rejected: set[str], may_offer, need_new: bool = True,
           target: str | None = None, edge: bool = False) -> Pick | None:
    """Проба: один трек другого стиля. Сначала направление: случайно с вероятностью φ × выборка Томпсона. Затем трек в нём:
    близкий к after по динамике и подаче; в уже принятом направлении — лучший для слушателя в этом стиле.
    target — уход по просьбе слушателя: направление уже выбрано, и близкий трек не обязателен.
    edge — ползунок на краю: направление без трека нужного разряда не пробуем. Иначе «только знакомое» давало 4 % нового."""
    a = lib.tracks[after]
    direction_prob = 1.0
    by_style: dict[str, list[tuple[float, int]]] = {}
    for row, feel in lib.gate.feel_candidates(lib.row[after], k=PROBE_POOL):
        tid = int(lib.ids[row])
        t = lib.tracks[tid]
        if (t.style in rejected or tid in prof.disliked or tid in lib.dj_tools or (allowed is not None and tid not in allowed)
                or (allowed is None and lib.best_edition.get(tid, tid) != tid)
                or t.artists & a.artists or t.album_key == a.album_key or not _filter_ok(lib, tid, f)
                or not _recording_ok(lib, tid, hist, prof, now_ms) or not _album_ok(lib, tid, hist)
                or not _artist_ok(lib, tid, hist, ARTIST_GAP)):
            continue
        if edge and (prof.newness(tid) >= 0.5) != need_new:
            continue
        by_style.setdefault(t.style, []).append((feel - prof.skip_penalty(tid), tid))
    if target is not None:
        style = target
    else:
        tried = drift.tried()  # принятое направление это не касается
        base = {s: x / (1.0 + (0 if s in drift.accepted else tried[s])) for s, x in drift.directions.items() if x > 0.0
                and s not in rejected and not drift.refused_lately(s) and (s in by_style or drift.share(s) >= STRONG_DIRECTION)
                and may_offer(s)}
        if not base:
            return None
        options = {s: x * _thompson(prof, rng, s) for s, x in base.items()}
        style = rng.choices(list(options), weights=list(options.values()))[0]
        direction_prob = _direction_chance(prof, base, style, rng)
    if style in by_style and (target is not None or drift.share(style) < STRONG_DIRECTION):
        found = by_style[style]
        fits_share = [x for x in found if (prof.newness(x[1]) >= 0.5) == need_new]
        found = sorted(fits_share or found, reverse=True)
        score, tid = rng.choice(found[:SPICE_TOP])
        return Pick(tid, f"{style} · подмес: близок по звучанию к «{a.title or 'предыдущему'}»", score, "spice", a.title,
                    parts={"feel": round(score, 3)}, prob=direction_prob, track_prob=1.0 / min(SPICE_TOP, len(found)))
    chosen = next_tracks(lib, prof, current=None, history=hist, filters=replace(f, styles=[style], boldness=0.0),
                         n=1, now_ms=now_ms, allowed=allowed, rng=rng, mode=MODE_MY, mood=mood)
    if not chosen or (edge and (prof.newness(chosen[0].track_id) >= 0.5) != need_new):
        return None
    p = chosen[0]
    p.why, p.seed_title = "spice", a.title
    p.reason = f"{style} · вы отзывались на этот стиль"
    p.prob = direction_prob
    return p


def session_events(conn: sqlite3.Connection, device_ids: list[int]) -> list[sqlite3.Row]:
    """Свежие события для настроя сессии — новые сверху. Отдельно от кэша профиля: реакция нужна сразу."""
    q = ",".join("?" * len(device_ids))
    return conn.execute(f"SELECT * FROM events WHERE device_id IN ({q}) ORDER BY ts DESC LIMIT ?",
                        (*device_ids, DRIFT_EVENTS)).fetchall()


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


def _sig(x: float | None) -> float | None:
    """Три значащие цифры: round(x, 3) превращал редкий выбор в ноль."""
    return None if x is None else float(f"{x:.3g}")


def stream_start(conn: sqlite3.Connection, device_id: int, history: list[int]) -> int:
    """Где в истории начался этот запуск потока. Плеер ставит поток после сыгранного и шлёт весь хвост очереди, а запуск — новая сессия:
    вчерашний Ambient не должен становиться её основным стилем. Запрос с прогревом мог не пригодиться, поэтому смотрим несколько.
    Признак — первый трек пачки: он звучит первым, остальные могли звучать и раньше, в прошлых запусках."""
    where = {t: i for i, t in enumerate(history)}
    for (result,) in conn.execute("SELECT result FROM nova_requests WHERE device_id = ? AND is_start = 1 ORDER BY id DESC LIMIT 8", (device_id,)):
        picks = json.loads(result or "[]")
        if picks and picks[0]["track"] in where:
            return where[picks[0]["track"]]
    return 0


def journal(lib: Library, prof: Profile, picks: list[Pick], trace: dict) -> dict:
    """Запись журнала выбора: что выбрано, из чего сложилась оценка, с какой вероятностью, какие были варианты рядом.
    q — вероятность пропуска по модели слушателя (решение 0016): по ней табло считает, как сервер угадывает пропуски."""
    rest = {k: v for k, v in trace.items() if k != "on_skip"}
    out = {"v": ALGO_VERSION, **rest,
           "picks": [{"t": p.track_id, "why": p.why, "style": lib.tracks[p.track_id].style, "new": prof.newness(p.track_id) >= 0.5,
                      "s": round(p.score, 4), "prob": _sig(p.prob), "track": _sig(p.track_prob), "slot": _sig(p.slot),
                      "p": {k: round(x, 3) for k, x in p.parts.items()},
                      **({"q": round(p.q, 4)} if p.q is not None else {}),
                      **({"phi": [round(p.phi[0], 3), round(p.phi[1], 3)]} if p.phi else {})} for p in picks]}
    if trace.get("on_skip"):
        out["on_skip"] = {str(x): {"t": p.track_id, "style": lib.tracks[p.track_id].style, "q": _sig(p.q)}
                          for x, p in trace["on_skip"].items()}
    return out


def log_request(conn: sqlite3.Connection, device_id: int, seed: int | None, picks: list[Pick],
                gate_log: list | None = None, detail: dict | None = None) -> int:
    """Запрос и решения шлюза: построчно — принятые и отклонённые после близости микса,
    остальные отказы (микс, пара стилей) — счётчиками, иначе ~400 строк на шаг (ФТ-8.3)."""
    now = int(time.time() * 1000)
    counts: Counter[str] = Counter()
    rows = []
    for a, b, v in gate_log or []:
        counts["accepted" if v.ok else v.failed] += 1
        if v.ok or v.failed not in (G.COND_MIX, G.COND_PAIR):
            why = json.dumps({"detail": v.detail}) if v.detail else None
            rows.append((now, a, b, int(v.ok), v.failed, why, round(v.mix, 4)))
    with conn:
        cur = conn.execute(
            "INSERT INTO nova_requests (device_id, ts, seed_track, result, gate_counts, detail, is_start) VALUES (?,?,?,?,?,?,?)",
            (device_id, now, seed,
             json.dumps([{"track": p.track_id, "reason": p.reason, "score": round(p.score, 4), "why": p.why} for p in picks], ensure_ascii=False),
             json.dumps(counts) if counts else None, json.dumps(detail, ensure_ascii=False, separators=(",", ":")) if detail else None,
             int(bool(detail and detail.get("start")))),
        )
        conn.execute("UPDATE nova_requests SET detail = NULL WHERE ts < ? AND detail IS NOT NULL", (now - JOURNAL_DAYS * 86_400_000,))
        if rows:
            conn.executemany(
                "INSERT INTO gate_log (request_id, ts, from_track, to_track, accepted, failed, detail, mix) VALUES (?,?,?,?,?,?,?,?)",
                [(cur.lastrowid, *r) for r in rows])
            conn.execute("DELETE FROM gate_log WHERE ts < ?", (now - GATE_LOG_DAYS * 86_400_000,))
    return cur.lastrowid
