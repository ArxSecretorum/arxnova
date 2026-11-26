"""Стиль трека по совокупности свидетельств (решение 0023).

Модель стилей (MAEST) слышит три отрывка одного трека и ошибается там, где человек не ошибся бы: он знает, что это за
альбом и кто исполнитель. Здесь к звуку трека добавляется то, что есть в любой медиатеке без интернета:

- звук остальных треков того же альбома (сборник — не альбом);
- звук остальных треков исполнителя; у ремикса — треков автора ремикса, а не исполнителя оригинала;
- звук тех треков исполнителя, что ближе всего по вектору микса (у исполнителя бывает две стороны: Calibre — DnB и эмбиент);
- стиль из тега файла;
- жанры MusicBrainz, если сервер их спрашивал.

Оценка стиля — взвешенная сумма логарифмов; весов пять (шесть с MusicBrainz), они общие для всех стилей, поэтому
переносятся на чужую медиатеку. Подобраны по эталону Discogs на медиатеке владельца 25.11.2026 (5330 треков с изданиями на
Discogs, исполнители проверки в подборе не участвовали): точное совпадение стиля 86,5 → 90,3 % у синглов и 74,0 → 86,3 % у
альбомов, принято хотя бы одной базой 96,7 и 94,5 %; с MusicBrainz — 91,1 и 87,4 %. Сам Discogs согласен с собой на 80 %.
Темп, сравнение звука с названиями стилей (CLAP), примета «транс против хауса» и своя модель поверх эмбеддингов на том
же эталоне ничего не добавили. Замер и скрипты — bench/genre_truth/, docs/decisions/0023-stil-po-svidetelstvam.md."""

from __future__ import annotations

import base64
import collections
import re
import unicodedata

import numpy as np

from arxnova import style_cues_data as _cues

EPS = 1e-4
WEIGHTS = {"sound": 2.654, "album": 0.746, "artist": 1.101, "similar": 0.813, "tag": 1.64, "tag_genre": 0.981}
WEIGHTS_MB = {"sound": 2.318, "album": 0.586, "artist": 0.956, "similar": 0.82, "tag": 1.236, "tag_genre": 0.925, "mb": 1.911}
ALBUM_MIN_OTHERS = 1  # у сингла из двух треков второй — уже свидетель
ARTIST_MIN_OTHERS = 2
SIMILAR_K = 10  # столько ближайших по звучанию треков исполнителя
SIMILAR_MIN_TRACKS = 4
COMPILATION_MIN_ARTISTS = 4

SYNONYMS = {"drum and bass": "Drum n Bass", "drum & bass": "Drum n Bass", "dnb": "Drum n Bass",
            "drum'n'bass": "Drum n Bass", "drum 'n' bass": "Drum n Bass", "d&b": "Drum n Bass",
            "psytrance": "Psy-Trance", "psychedelic trance": "Psy-Trance", "goa": "Goa Trance",
            "neoclassical": "Neo-Classical", "post-classical": "Modern Classical", "trip-hop": "Trip Hop",
            "trip hop": "Trip Hop", "hip hop": "Boom Bap", "hip-hop": "Boom Bap", "neo soul": "Neo Soul",
            "dubstep": "Dubstep", "uk garage": "UK Garage", "2-step": "UK Garage",
            "nu jazz": "Future Jazz", "dark wave": "Darkwave", "ethereal wave": "Ethereal",
            "contemporary classical": "Contemporary", "nordic folk": "Nordic", "film score": "Score"}

# Слово жанра в теге файла → жанр Discogs: поддержка всем стилям этого жанра
GENRE_WORDS = {"Jazz": ["jazz"], "Electronic": ["electronic", "electronica", "エレクトロニック", "electro", "edm", "dance"],
               "Classical": ["classical", "classique", "klassik", "classic"], "Rock": ["rock", "metal", "punk"],
               "Folk, World, & Country": ["folk", "world", "country", "celtic", "ethnic"], "Pop": ["pop"],
               "Hip Hop": ["hip hop", "hip-hop", "hiphop", "rap"], "Reggae": ["reggae", "dancehall"], "Blues": ["blues"],
               "Latin": ["latin"], "Stage & Screen": ["soundtrack", "sound track", "score", "ost", "film", "bandes originales"],
               "Funk / Soul": ["funk", "soul", "r&b", "rnb"]}
_GENRE_RE = {g: re.compile("|".join(r"(?<![a-z])" + re.escape(w) + r"(?![a-z])" for w in ws)) for g, ws in GENRE_WORDS.items()}

_NOBODY = {"various artists", "various", "va", "v a", "unknown artist", "unknown", "разные исполнители", "сборник"}  # не исполнитель
_SPLIT = re.compile(r"\s*(?:,|&|\bfeat\.?|\bft\.?|\bfeaturing\b|\bvs\.?|\bx\b|\+|/|;)\s*", re.I)
_REMIX = re.compile(r"[\(\[]\s*([^\(\)\[\]]+?)\s+(?:remix|rmx|rework|refix|bootleg|re-?edit|remake|flip|vip|mix|edit|dub|version)\s*[\)\]]", re.I)
_REMIX_WORD = re.compile(r"\b(?:remix|rmx|rework|refix|bootleg|flip|remake)\b", re.I)
_GENERIC = {"original", "radio", "extended", "club", "instrumental", "vocal", "album", "single", "live", "acoustic", "continuous",
            "dj", "full", "short", "long", "main", "alternate", "alternative", "demo", "new", "old", "final", "special", "bonus",
            "continuous dj", "original 12", "12", "7", "clean", "explicit", "mono", "stereo", "digital", "vinyl", "video", "tv",
            "piano", "orchestral", "unplugged", "stripped", "ambient", "dub", "vip", "remastered", "remaster", "us", "uk"}


def norm(s: str | None) -> str:
    s = unicodedata.normalize("NFKD", s or "").casefold()
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", "".join(c for c in s if not unicodedata.combining(c)))).strip()


def norm_style(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower().replace("&", "n").replace(" and ", "n"))


def tag_genres(tag: str | None) -> set[str]:
    """Жанры Discogs, названные в теге файла: «ECM Style Jazz» → {Jazz}."""
    t = (tag or "").casefold()
    return {g for g, rx in _GENRE_RE.items() if rx.search(t)} if t else set()


def remixers(title: str | None) -> set[str]:
    """Кто сделал ремикс, по названию трека: «Ready for Your Love (Etherwood Remix)» → {etherwood}."""
    out: set[str] = set()
    for m in _REMIX.finditer(title or ""):
        name = re.sub(r"\b\d{4}\b", "", m.group(1)).strip()
        if not name or norm(name) in _GENERIC:
            continue
        for p in _SPLIT.split(name):
            p = norm(re.sub(r"'s$", "", p.strip()))
            if p and p not in _GENERIC:
                out.add(p)
    return out


def _loo_mean(m: np.ndarray, groups: list, min_others: int) -> np.ndarray:
    """Среднее активаций по группе без самого трека; у малых групп — NaN."""
    out = np.full_like(m, np.nan)
    idx: dict = collections.defaultdict(list)
    for i, g in enumerate(groups):
        if g is not None:
            idx[g].append(i)
    for ii in idx.values():
        if len(ii) - 1 < min_others:
            continue
        s = m[ii].sum(axis=0)
        out[ii] = (s[None, :] - m[ii]) / (len(ii) - 1)
    return out


def _relative(mean: np.ndarray, own: np.ndarray) -> np.ndarray:
    """log(среднее группы) − log(свой звук); строка без группы (NaN) — ноль. На месте: большая матрица не дублируется."""
    have = ~np.isnan(mean[:, 0]) if mean.size else np.zeros(len(mean), bool)
    mean[~have] = 0
    mean[have] = np.log(mean[have] + EPS) - own[have]
    return mean


def _named(names, index: dict[str, list[int]], lower: bool) -> list[int]:
    return [j for name in names if name.strip()
            for j in index.get(norm_style(SYNONYMS.get(name.strip().lower() if lower else name.strip(), name.strip())), [])]


def _hits(ids, labels: list[str], meta, mb: dict | None) -> dict[str, list[tuple[int, list[int]]]]:
    """Что названо словами: стиль в теге, жанр в теге, жанры MusicBrainz — (строка трека, номера стилей)."""
    index: dict[str, list[int]] = collections.defaultdict(list)
    for j, lab in enumerate(labels):
        index[norm_style(lab.split("---")[-1])].append(j)
    genre_of = np.array([lab.split("---")[0] for lab in labels])
    out: dict[str, list] = {"tag": [], "tag_genre": [], "mb": []}
    for i, tid in enumerate(ids):
        tag = meta[int(tid)]["genre_raw"]
        if tag:
            if js := _named(re.split(r"[;,/|]", tag), index, lower=True):
                out["tag"].append((i, js))
            if gs := tag_genres(tag):
                out["tag_genre"].append((i, np.flatnonzero(np.isin(genre_of, list(gs))).tolist()))
        if mb and (js := _named(mb.get(int(tid), {}), index, lower=False)):
            out["mb"].append((i, js))
    return out


def _terms(ids, styles: np.ndarray, meta, mix: np.ndarray, hits: dict):
    """Свидетельства по одному — (имя, матрица треки × стили): все сразу на большой медиатеке не поместятся в память Pi."""
    m = np.asarray(styles, np.float32)
    n, k = m.shape
    a = np.log(m + EPS)
    yield "sound", a
    prim: list = []
    album: list = []
    album_artists: dict = collections.defaultdict(set)
    for tid in ids:
        r = meta[int(tid)]
        p = norm(_SPLIT.split(r["artist"])[0]) if r["artist"] else None
        prim.append(None if p in _NOBODY else p or None)
        key = f"{norm(r['albumartist'] or r['artist'])}|{norm(r['album'])}" if r["album"] else None
        album.append(key)
        if key:
            album_artists[key].add(p)
    album = [None if key is None or meta[int(t)]["compilation"] or len(album_artists[key]) >= COMPILATION_MIN_ARTISTS else key
             for key, t in zip(album, ids)]
    # ремикс: контекст — треки автора ремикса (как исполнителя или автора других ремиксов), а не исполнителя оригинала.
    # «(Gnomes Mix)», «(4x4 mix)» — названия версий: ремикс — при слове remix или известном в медиатеке имени
    rmx = [remixers(meta[int(t)]["title"]) for t in ids]
    by_name: dict = collections.defaultdict(list)
    for i, p in enumerate(prim):
        if p and not rmx[i]:
            by_name[p].append(i)
    for i, r in enumerate(rmx):
        for x in r:
            by_name[x].append(i)
    known = {p for p in prim if p}
    is_rmx = np.array([bool(r) and (bool(_REMIX_WORD.search(meta[int(t)]["title"] or "")) or bool(r & known))
                       for r, t in zip(rmx, ids)], bool) if n else np.zeros(0, bool)
    # автора ремикса в медиатеке нет (меньше двух его треков) — свидетели те же, что у оригинала: альбом и исполнитель.
    # У 125 таких ремиксов с эталоном совпадение без контекста 82 %, с контекстом оригинала — 94 % (25.11.2026)
    others = {int(i): sorted({j for r in rmx[i] for j in by_name.get(r, []) if j != i}) for i in np.flatnonzero(is_rmx)}
    by_remixer = np.zeros(n, bool)
    by_remixer[[i for i, ii in others.items() if len(ii) >= 2]] = True
    x = _loo_mean(m, album, ALBUM_MIN_OTHERS)
    x[by_remixer] = np.nan
    yield "album", _relative(x, a)
    x = _loo_mean(m, prim, ARTIST_MIN_OTHERS)
    for i in np.flatnonzero(by_remixer):
        x[i] = m[others[int(i)]].mean(axis=0)
    yield "artist", _relative(x, a)
    x = np.zeros_like(a)
    for ii in by_name.values():
        ii = np.array(sorted(set(ii)))
        if len(ii) < SIMILAR_MIN_TRACKS:
            continue
        v = np.asarray(mix[ii], np.float32)  # по группе: копия векторов всей медиатеки — десятки мегабайт
        v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
        sim = v @ v.T
        np.fill_diagonal(sim, -2)
        kk = min(SIMILAR_K, len(ii) - 1)
        nn = np.argpartition(-sim, kk - 1, axis=1)[:, :kk]
        x[ii] = np.log(m[ii][nn].mean(axis=1) + EPS) - a[ii]
    yield "similar", x
    for name in ("tag", "tag_genre", "mb"):
        x = np.zeros((n, k), np.float32)
        for i, js in hits[name]:
            x[i, js] = 1
        yield name, x


def evidence(ids, styles: np.ndarray, labels: list[str], meta, mix: np.ndarray, mb: dict | None = None) -> dict[str, np.ndarray]:
    """Свидетельства по каждому стилю каждого трека. meta: id → запись с artist, albumartist, album, title, genre_raw,
    compilation. mix — векторы микса в порядке ids. mb — id → жанры MusicBrainz (mbgenres.track_genres)."""
    return {name: x.copy() for name, x in _terms(ids, styles, meta, mix, _hits(ids, labels, meta, mb))}


# Трек выбивается из альбома: звук уверенно указывает на другое семейство, и темп с ним согласен. Границы темпа — известные
# границы жанров (BPM; у DnB и басовой музыки детектор долей отдаёт и половину). У семейств без границ (фолк, джаз, рок…)
# темп ничего не говорит, и спор остаётся за контекстом.
TEMPO = {"dnb": ((160, 180), (80, 90)), "bass": ((124, 150), (62, 75)), "house": ((114, 150),), "chill": ((40, 115),)}
OWN_MIN = 0.3    # звук уверен: лучшая оценка своего семейства не ниже
OWN_RATIO = 2.0  # и вдвое выше лучшей оценки семейства, которое выбрал контекст
# Стиль внутри своего семейства: веса подобраны на цель «самый слышимый в треке стиль издания» (эталон Discogs, 25.11.2026)
INSIDE = {"sound": 2.65, "album": 0.245, "artist": 0.831, "similar": 0.569, "tag": 0.673, "tag_genre": 0.278}
INSIDE_MB = {"sound": 2.485, "album": 0.133, "artist": 0.764, "similar": 0.576, "tag": 0.546, "tag_genre": 0.408, "mb": 0.902}
# Примета «звук транса против звука хауса» (CLAP) внутри танцевального семейства: центр и разброс — по медиатеке владельца.
# Вес 0,4 сверку с базой не меняет (группа хаус/техно/транс 87,6 % на 169 треках) и согласуется со слухом владельца
DANCE = "house"
SIGN_CENTER, SIGN_SCALE, SIGN_W = 0.02359, 0.11501, 0.4


TRANCE_TEXTS = _cues.TRANCE_TEXTS


def cue_vectors() -> np.ndarray:
    """Текстовые векторы CLAP описаний: сначала приметы транса, затем хауса."""
    return np.frombuffer(base64.b64decode(_cues.VECTORS), np.float32).reshape(-1, 512)


def trance_sign(clap: np.ndarray) -> np.ndarray:
    """Насколько в треке слышны приметы транса (кислотная линия 303, скрип фильтра, психоделический лид, арпеджио,
    суперсо) против примет хауса (соул-вокал, диско-сэмпл, пиано) — в долях разброса по медиатеке; у строки с NaN — NaN."""
    c = np.asarray(clap, np.float32) @ cue_vectors().T
    k = len(TRANCE_TEXTS)
    return (c[:, :k].max(axis=1) - c[:, k:].max(axis=1) - SIGN_CENTER) / SIGN_SCALE


def tempo_fits(family: str, bpm: float | None) -> bool | None:
    """Подходит ли темп семейству; None — у семейства нет границ или темп неизвестен."""
    if family not in TEMPO or not bpm or bpm != bpm:
        return None
    return any(lo <= bpm <= hi for lo, hi in TEMPO[family])


def _stands_out(ids, m: np.ndarray, labels: list[str], top: np.ndarray, inside: np.ndarray, family, bpm, clap) -> np.ndarray:
    """Кто выбивается из альбома — тому семейство по своему звуку, стиль — лучший внутри него."""
    fam = np.asarray(family)
    own = m.argmax(axis=1)
    best = m.max(axis=1)
    out = top.copy()
    cand = np.flatnonzero((fam[own] != fam[top]) & (best >= OWN_MIN))
    if not len(cand):
        return out
    short = [lab.split("---")[-1] for lab in labels]
    side = np.array([1.0 if "Trance" in s else -1.0 if ("House" in s or "Disco" in s) else 0.0 for s in short], np.float32)
    side *= fam == DANCE
    sign = trance_sign(clap) if clap is not None else None
    for i in cand:
        f1, f2 = fam[own[i]], fam[top[i]]
        b = bpm.get(int(ids[i]))
        if tempo_fits(f1, b) is not True or tempo_fits(f2, b) is not False:
            continue
        if best[i] < OWN_RATIO * m[i, fam == f2].max():
            continue
        z = inside[i].astype(np.float32).copy()
        if sign is not None and f1 == DANCE and sign[i] == sign[i]:
            z += SIGN_W * sign[i] * side
        z[fam != f1] = -np.inf
        out[i] = int(z.argmax())
    return out


def decide(ids, styles: np.ndarray, labels: list[str], meta, mix: np.ndarray, mb: dict | None = None,
           family=None, bpm: dict | None = None, clap: np.ndarray | None = None) -> np.ndarray:
    """Номер стиля каждого трека. family — семейство каждого стиля из labels, bpm — id → темп, clap — векторы CLAP в
    порядке ids: с ними трек, который выбивается из альбома, получает стиль по своему звуку (_stands_out)."""
    if not len(ids):
        return np.zeros(0, np.int64)
    hits = _hits(ids, labels, meta, mb)
    w, w_in = (WEIGHTS_MB, INSIDE_MB) if hits["mb"] else (WEIGHTS, INSIDE)
    two = family is not None and bpm is not None
    score = inside = None
    for name, x in _terms(ids, styles, meta, mix, hits):
        if name not in w:
            continue
        if score is None:
            score = w[name] * x  # «звук» нужен дальше — не трогаем
            inside = w_in[name] * x if two else None
        else:
            if two:
                inside += w_in[name] * x
            x *= w[name]
            score += x
    top = score.argmax(axis=1)
    if two:
        top = _stands_out(ids, np.asarray(styles, np.float32), labels, top, inside, family, bpm, clap)
    return top
