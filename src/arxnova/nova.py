"""Подбор следующего трека: профиль вкуса (М3), кандидаты и ранжирование (М4–М5) с жёсткими
правилами ФТ-4.4 и ФТ-4.9 и причинами выбора (ФТ-8.2).

Этап 1: поток держится внутри жанра текущего трека. Межжанровый шлюз М6 — этап 2.
"""

from __future__ import annotations

import json
import random
import re
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

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
FACET_MIN_TRACKS = 50  # фильтр/сценарий показываем, если под него набирается столько записей


def split_artists(s: str | None) -> frozenset[str]:
    s = keys.normalize(s)
    parts = re.split(r"\s*(?:&|,|;|\bfeat\.?|\bft\.?|\bx\b|\bvs\.?|\band\b)\s*", s)
    return frozenset(p for p in parts if p)


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
    why: str = ""  # код причины для перевода в плеере: liked | similar | new | taste
    seed_title: str | None = None


@dataclass
class Filters:
    genres: list[str] = field(default_factory=list)  # жанры Discogs
    styles: list[str] = field(default_factory=list)  # стили Discogs
    familiar: float = 0.5  # 0 — только новое, 1 — только знакомое (ФТ-4.3)


class Library:
    """Матрицы и метаданные в памяти. Пересобирается после индексации."""

    def __init__(self, cfg: Config, conn: sqlite3.Connection):
        self.cfg = cfg
        d = cfg.index_dir / models.EFFNET.id
        self.ids = np.load(d / "ids.npy")
        v = np.load(d / "mix.npy").astype(np.float32)
        v -= v.mean(axis=0)
        v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
        self.vec = v
        self.row = {int(t): i for i, t in enumerate(self.ids)}
        labels = models.style_labels(cfg.models_dir)
        styles = np.load(d / "styles.npy")
        top = styles.argmax(axis=1)
        self.style_scores = styles
        self.style_labels = labels
        rows = {
            r["id"]: r
            for r in conn.execute(
                "SELECT id, artist, albumartist, album, title, date, mb_album, compilation, recording_key, duration FROM tracks"
            )
        }
        album_artists: dict[str, set[frozenset[str]]] = {}
        album_size: Counter[str] = Counter()
        tmp = {}
        for i, tid in enumerate(self.ids):
            r = rows[int(tid)]
            if r["mb_album"]:
                ak = "mb:" + r["mb_album"]
            else:
                ak = f"{keys.normalize(r['albumartist'] or r['artist'])}|{keys.normalize(r['album'])}|{(r['date'] or '')[:4]}"
            genre, _, style = labels[top[i]].partition("---")
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

    def facets(self, min_tracks: int = FACET_MIN_TRACKS) -> dict:
        g = Counter(t.genre for t in self.tracks.values())
        s = Counter((t.genre, t.style) for t in self.tracks.values())
        return {
            "genres": [{"name": k, "tracks": n} for k, n in g.most_common() if n >= min_tracks],
            "styles": [{"genre": k[0], "name": k[1], "tracks": n} for k, n in s.most_common() if n >= min_tracks],
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
        self.played: set[int] = set()
        recent = []
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
            if et in ("listen", "skip", "replay", "similar"):
                self.played.add(tid)
                rk = lib.tracks[tid].recording_key
                self.last_played[rk] = max(self.last_played.get(rk, 0), e["ts"])
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


def _norm(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


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
    return not last or now_ms - last > REPEAT_HOURS * 3_600_000


def _filter_ok(lib: Library, tid: int, f: Filters) -> bool:
    t = lib.tracks[tid]
    if f.styles and t.style not in f.styles:
        return False
    if f.genres and t.genre not in f.genres:
        return False
    return True


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
) -> list[Pick]:
    """Следующий трек и запасные (ФТ-4.6). history — треки сессии по порядку, current —
    последний; allowed — что доступно этому прибору (ФТ-4.5), None — всё.

    MODE_MY: опора — профиль вкуса (долгосрочный + сессионный вектор, лайки); что играло
    перед запуском, на выбор почти не влияет. MODE_TRACK: опора — seed, один и тот же на
    весь поток (не предыдущий трек — иначе поток цепочкой уплывает от выбранного)."""
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

    # кандидаты (ФТ-4.1)
    cands: dict[int, str] = {}
    if target is not None:
        for j in np.argsort(-(lib.vec @ target))[: KNN * 2]:
            tid = int(lib.ids[j])
            if tid != anchor:
                cands.setdefault(tid, "near" if anchor is not None else "taste")
    for tid in prof.liked:
        cands.setdefault(tid, "liked")
    if f.styles or f.genres:  # выбран стиль — в кандидатах весь стиль, а не случайная выборка
        for tid in lib.tracks:
            if _filter_ok(lib, tid, f):
                cands.setdefault(tid, "style")
    never = [t for t in lib.tracks if t not in prof.played]
    rng.shuffle(never)
    for tid in never[:200]:
        cands.setdefault(tid, "new")
    if not cands:
        cands = {t: "new" for t in lib.tracks}

    anchor_genre = lib.tracks[anchor].genre if anchor is not None else None
    recent_vecs = [lib.vec[lib.row[h]] for h in hist[-5:]]

    scored: list[Pick] = []
    for tid, src in cands.items():
        if tid in prof.disliked or (allowed is not None and tid not in allowed):
            continue
        if not _filter_ok(lib, tid, f):
            continue
        # «Похожие на трек» без фильтра держатся жанра выбранного трека
        if not f.genres and not f.styles and anchor_genre and lib.tracks[tid].genre != anchor_genre:
            continue
        if not (_recording_ok(lib, tid, hist, prof, now_ms) and _album_ok(lib, tid, hist)):
            continue
        v = lib.vec[lib.row[tid]]
        s = 0.0
        if anchor is not None:
            s += 1.2 * float(v @ target)
            if taste is not None:
                s += 0.3 * float(v @ taste)
        elif taste is not None:
            s += 1.0 * float(v @ taste)
        else:
            s += rng.random() * 0.1  # нет ни опоры, ни истории — первые треки знакомят с медиатекой
        if neg is not None:
            s -= 0.8 * float(v @ neg)
        if recent_vecs:  # MMR: близко к стилю, но не копия последних треков
            s -= MMR_WEIGHT * max(float(v @ r) for r in recent_vecs)
        known = tid in prof.played or tid in prof.liked
        s += (f.familiar - 0.5) * (0.4 if known else -0.4)
        if tid in prof.liked:
            s += 0.15
        why = _why(tid, anchor, prof)
        seed_title = lib.tracks[anchor].title if anchor is not None else None
        scored.append(Pick(tid, _reason(lib, tid, anchor, src, prof), s, why, seed_title))

    scored.sort(key=lambda p: -p.score)
    # Разнесение исполнителей ослабляется, только если иначе тишина (узкий стиль одного
    # исполнителя): сначала «не чаще 1 на K», потом «не подряд», потом без ограничения.
    # Правила альбома и повтора записи не ослабляются никогда.
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


def log_request(conn: sqlite3.Connection, device_id: int, seed: int | None, picks: list[Pick]) -> int:
    with conn:
        cur = conn.execute(
            "INSERT INTO nova_requests (device_id, ts, seed_track, result) VALUES (?,?,?,?)",
            (device_id, int(time.time() * 1000), seed,
             json.dumps([{"track": p.track_id, "reason": p.reason, "score": round(p.score, 4)} for p in picks], ensure_ascii=False)),
        )
    return cur.lastrowid
