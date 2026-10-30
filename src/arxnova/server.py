"""HTTP API сервера (ТЗ, §15): сопряжение, события, подбор, просмотр медиатеки, поток.

Контракт — /v1/...; описание OpenAPI отдаёт сам сервер по /openapi.json (API-1).
Всё, кроме /v1/health и /v1/pair, требует токен устройства: `Authorization: Bearer <токен>`.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
import time
from pathlib import Path

import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from arxnova import (
    __version__,
    access,
    auth,
    bridges,
    db,
    flac,
    guests,
    keys,
    mixes,
    models,
    nova,
    presence,
)
from arxnova import folders as server_folders
from arxnova import updates  # «folders» — уже эндпоинт медиатеки
from arxnova.config import Config

VARIOUS_ARTISTS = "@various"  # служебная метка: наружу уходит как "various": true, подпись даёт плеер
EVENT_TYPES = {"listen", "skip", "like", "unlike", "dislike", "undislike", "replay", "similar"}
SKIP_REASONS = {"rhythm", "vocal", "genre", "not_now", "tired"}


class PairIn(BaseModel):
    code: str
    name: str = "устройство"


class ClaimIn(BaseModel):
    name: str = "устройство"


class AccessAnswerIn(BaseModel):
    allow: bool
    role: str = "guest"  # кем пустить; по умолчанию — гостем: «своим» — только явно


class LibraryIn(BaseModel):
    paths: list[str] = Field(min_length=1, max_length=8)


class InviteIn(BaseModel):
    name: str = Field(min_length=1, max_length=60)  # кому приглашение (поимённо, ОД-10)
    # как гости увидят владельца («кто пригласил»); None — не менять, "" — не показывать
    owner_name: str | None = Field(default=None, max_length=40)


class JoinIn(BaseModel):
    code: str = Field(min_length=4, max_length=32)
    name: str = "устройство"


class EventIn(BaseModel):
    event_id: str = Field(min_length=8, max_length=64)
    ts: int
    md5: str | None = None  # hex; можно не слать, если известен track_id сервера
    cue_ms: int = 0
    track_id: int | None = None
    event_type: str
    played_ratio_pct: int | None = None
    track_length_s: int | None = None
    is_organic: bool = True
    session_id: int | None = None
    prev_track_id: int | None = None
    transition_type: str | None = None
    nova_request_id: int | None = None
    skip_reason: str | None = None


class EventsIn(BaseModel):
    events: list[EventIn]


class FiltersIn(BaseModel):
    genres: list[str] = []
    styles: list[str] = []
    familiar: float = 0.5
    vocal: str | None = None  # yes | no
    bpm: list[float] | None = None  # [от, до]
    bpm_exact: bool = False  # темп как есть, без half/double-time (плитки плеера)
    scenario: str | None = None  # id сценария (ФТ-4.10)
    energy: list[float] | None = None  # [от, до], 0–1

    def to_filters(self, boldness: float):
        def rng(x):
            return (float(x[0]), float(x[1])) if x and len(x) == 2 else None

        return nova.Filters(self.genres, self.styles, self.familiar, boldness,
                            self.vocal if self.vocal in ("yes", "no") else None, rng(self.bpm), self.bpm_exact, rng(self.energy), self.scenario or None)


class PairHelloIn(BaseModel):
    key: str  # hex sha1 ключа «tcp» из кода
    nonce: str  # hex, 16–64 байта


class LabelIn(BaseModel):
    a: int
    b: int
    label: str


class PctIn(BaseModel):
    pct: float = Field(ge=75, le=99)


class GateSettingsIn(BaseModel):
    boldness: float | None = Field(None, ge=0, le=1)
    forbid: list[list[str]] | None = None


class ToggleIn(BaseModel):
    on: bool


class NextIn(BaseModel):
    mode: str = "my"  # my — «Свой поток»; track — «Похожие на трек» (seed_track)
    seed_track: int | None = None
    current_track: int | None = None
    history: list[int] = []
    filters: FiltersIn = FiltersIn()
    n: int = Field(4, ge=1, le=20)
    only_on_device: bool = False
    start: bool = False  # первый запрос запущенного потока (плитка, кнопка «Мой поток»)


class State:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.conn = db.connect(cfg.db_path, threaded=True)
        self.conn.execute("PRAGMA busy_timeout = 5000")
        self.lock = threading.RLock()
        self._lib: nova.Library | None = None
        self._lib_stamp = None
        self._profiles: dict[tuple, tuple] = {}  # приборы → (метка событий, профиль вкуса)
        self._building: set[tuple] = set()
        self._catalog = None
        self._catalog_stamp = None
        self.server_name = cfg.server_name

    def _index_stamp(self):
        d = self.cfg.index_dir / models.EFFNET.id
        p = d / "ids.npy"
        if not p.exists():
            return None
        # фасеты дописываются отдельным проходом (`arxnova facets`) — шлюз перечитывается и тогда
        f = d / "features.npy"
        return (p.stat().st_mtime_ns, f.stat().st_mtime_ns if f.exists() else 0)

    def profile(self, lib: "nova.Library", device_ids: list[int], now: int) -> "nova.Profile":
        """Профиль вкуса (М3) из кэша. Пересчёт по всей истории — O(событий): на 50 000 событий
        ~1 с на Mac и в разы дольше на Pi, на каждый «следующий трек». Поэтому: новые события
        пришли — запрос отвечает по прежнему профилю (отстаёт на один трек), а свежий собирается
        в фоне; первого профиля нет — собирается сразу."""
        key = tuple(sorted(device_ids))
        q = ",".join("?" * len(key))
        with self.lock:
            stamp = self.conn.execute(f"SELECT count(*), max(rowid) FROM events WHERE device_id IN ({q})", key).fetchone()
        stamp = (id(lib), tuple(stamp))
        hit = self._profiles.get(key)
        if hit and hit[0] == stamp:
            return hit[1]

        def build():
            with self.lock:
                events = nova.load_events(self.conn, list(key))
            p = nova.Profile(lib, events, now)
            self._profiles[key] = (stamp, p)
            return p

        marks = False
        if hit:  # «нравится» / «не предлагать» должны действовать сразу — тогда без отставания
            with self.lock:
                marks = self.conn.execute(
                    f"SELECT 1 FROM events WHERE device_id IN ({q}) AND rowid > ? AND event_type IN "
                    "('like','unlike','dislike','undislike') LIMIT 1", (*key, hit[0][1][1] or 0)).fetchone() is not None
        if hit and hit[0][0] == id(lib) and not marks:
            if key not in self._building:
                self._building.add(key)

                def bg():
                    try:
                        build()
                    finally:
                        self._building.discard(key)
                threading.Thread(target=bg, name="profile", daemon=True).start()
            return hit[1]
        return build()

    def indexed_count(self) -> int:
        """Сколько треков в индексе — без построения модели подбора."""
        if self._lib is not None and self._lib_stamp == self._index_stamp():
            return len(self._lib.ids)
        p = self.cfg.index_dir / models.EFFNET.id / "ids.npy"
        if not p.exists() or not models.is_installed(self.cfg.models_dir, models.EFFNET):
            return 0
        try:
            return int(np.load(p, mmap_mode="r").shape[0])
        except Exception:
            return 0

    def library(self) -> nova.Library | None:
        stamp = self._index_stamp()
        if stamp is None or not models.is_installed(self.cfg.models_dir, models.EFFNET):
            return None
        if self._lib is None or stamp != self._lib_stamp:
            with self.lock:
                self._lib = nova.Library(self.cfg, self.conn)
                self._lib_stamp = stamp
        return self._lib

    def catalog(self) -> "Catalog":
        with self.lock:  # одно соединение на все потоки обработчиков
            stamp = tuple(self.conn.execute("SELECT count(*), max(id) FROM tracks").fetchone())
            if self._catalog is None or stamp != self._catalog_stamp:
                self._catalog = Catalog(self.conn, self.cfg.libraries)
                self._catalog_stamp = stamp
            return self._catalog


class Catalog:
    """Медиатека для просмотра (П-13): альбомы, исполнители, папки — из базы, без модели."""

    def __init__(self, conn: sqlite3.Connection, roots: list[Path]):
        rows = conn.execute(
            """SELECT t.*, min(p.path) AS path FROM tracks t JOIN track_paths p ON p.track_id = t.id
               GROUP BY t.id"""
        ).fetchall()
        self.rows = {r["id"]: r for r in rows}
        self.roots = [str(r) for r in roots]
        self.albums: dict[str, dict] = {}
        self.artists: dict[str, dict] = {}
        for r in rows:
            akey = self.album_key(r)
            a = self.albums.setdefault(akey, {
                "id": hashlib.md5(akey.encode()).hexdigest()[:16], "title": r["album"] or "",  # пусто — плеер подпишет «без альбома» на своём языке
                "artist": r["albumartist"] or r["artist"], "year": (r["date"] or "")[:4],
                "tracks": [], "artists": set(),
            })
            a["tracks"].append(r["id"])
            a["artists"].add(r["artist"])
            for name in sorted(nova.split_artists(r["artist"])) or [""]:  # пусто — «неизвестный исполнитель» в плеере
                ar = self.artists.setdefault(name, {"id": hashlib.md5(name.encode()).hexdigest()[:16], "name": name, "tracks": []})
                ar["tracks"].append(r["id"])
        for a in self.albums.values():
            if len(a["artists"]) > 1:
                a["artist"] = r_various(a["artists"])
            a["tracks"].sort(key=lambda t: (self.rows[t]["path"], self.rows[t]["cue_ms"],
                                            int(self.rows[t]["tracknumber"]) if str(self.rows[t]["tracknumber"] or "").isdigit() else 0))
        self.album_by_id = {a["id"]: a for a in self.albums.values()}
        self.artist_by_id = {a["id"]: a for a in self.artists.values()}
        # исходное имя исполнителя для показа — самое частое написание
        for ar in self.artists.values():
            names = [self.rows[t]["artist"] for t in ar["tracks"]]
            ar["display"] = min(names, key=lambda n: (len(nova.split_artists(n)), -names.count(n))) if names else ar["name"]

    @staticmethod
    def album_key(r) -> str:
        if r["mb_album"]:
            return "mb:" + r["mb_album"]
        return f"{keys.normalize(r['albumartist'] or '')}|{keys.normalize(r['album'])}|{(r['date'] or '')[:4]}|{album_folder(r['path'])}"

    def rel(self, path: str) -> tuple[int, str] | None:
        for i, root in enumerate(self.roots):
            if path.startswith(root.rstrip("/") + "/"):
                return i, path[len(root.rstrip("/")) + 1 :]
        return None


# Папка диска многодискового издания: «CD1», «CD 2», «Disc 1», «Disk2», «CD1 - Mixed», «Диск 1».
_DISC_FOLDER = re.compile(r"^(cd|disc|disk|диск)\s*[-_.]?\s*\d+\b", re.IGNORECASE)


def album_folder(path: str) -> Path:
    """Папка альбома: для файла в папке диска — папка выше, иначе CD1 и CD2 одного издания
    становились двумя альбомами (замечание владельца 27.10.2026)."""
    d = Path(path).parent
    return d.parent if _DISC_FOLDER.match(d.name) else d


def r_various(artists: set) -> str:
    # VARIOUS — плеер подпишет «Разные исполнители» на своём языке
    return VARIOUS_ARTISTS if len(artists) > 2 else " & ".join(sorted(a for a in artists if a))


def track_json(r, lib: nova.Library | None = None, reason: str | None = None, pick: "nova.Pick | None" = None) -> dict:
    d = {
        "id": r["id"], "md5": bytes(r["md5"]).hex(), "cue_ms": r["cue_ms"], "cue_end_ms": r["cue_end_ms"],
        "title": r["title"], "artist": r["artist"], "album": r["album"], "year": (r["date"] or "")[:4],
        "duration": r["duration"], "sample_rate": r["sample_rate"], "bits": r["bits"], "channels": r["channels"],
    }
    try:  # размер файла — плееру при открытии трека не нужно спрашивать его по сети
        d["size"] = Path(r["path"]).stat().st_size
    except (OSError, KeyError, IndexError):
        pass
    if lib and r["id"] in lib.tracks:
        t = lib.tracks[r["id"]]
        d["genre"], d["style"] = t.genre, t.style
        bpm = getattr(lib, "bpm_of", {}).get(r["id"])
        if bpm:
            d["bpm"] = round(float(bpm), 1)
    if reason:
        d["reason"] = reason
    if pick is not None:
        d["why"] = pick.why
        if pick.facets:
            d["bridge"] = pick.facets
        if pick.seed_title:
            d["seed_title"] = pick.seed_title
    return d


IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}


def find_cover(d: Path) -> Path | None:
    """Обложка рядом с файлом: сначала привычные имена, потом любая картинка папки (с «cover»,
    «front», «folder» в имени — первыми), потом то же в папке выше (альбом в подпапке «год —
    название», картинка — уровнем выше). Так находится обложка и у CUE-образов без встроенной."""
    def pick(folder: Path) -> Path | None:
        try:
            imgs = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXT]
        except OSError:
            return None
        if not imgs:
            return None
        def rank(p: Path) -> tuple:
            n = p.stem.lower()
            return (0 if n in ("cover", "folder", "front") else 1 if any(k in n for k in ("cover", "front", "folder")) else 2, n)
        return min(imgs, key=rank)
    return pick(d) or pick(d.parent)


def _range_len(header: str | None, size: int | None) -> int:
    """Сколько байт уйдёт по запросу (для учёта объёма гостя)."""
    size = size or 0
    if not header or not header.startswith("bytes="):
        return size
    try:
        a, b = header[6:].split(",")[0].split("-")
        start = int(a) if a else max(0, size - int(b))
        end = int(b) if a and b else size - 1
        return max(0, min(end, size - 1) - start + 1)
    except ValueError:
        return size


def thumbnail(raw: bytes, size: int) -> bytes:
    """Картинка → JPEG не больше size по большей стороне (качество 85, прогрессивный)."""
    import io

    from PIL import Image

    im = Image.open(io.BytesIO(raw))
    im.draft("RGB", (size, size))  # JPEG декодируется сразу уменьшенным — быстро и без лишней памяти
    im = im.convert("RGB")
    im.thumbnail((size, size), Image.LANCZOS)
    out = io.BytesIO()
    im.save(out, "JPEG", quality=85, optimize=True, progressive=True)
    return out.getvalue()


WARM_BYTES = 1 << 20


def _warm_starts(paths: list[str]) -> None:
    """Начала треков, которые плеер вот-вот запросит, — в кэш ОС заранее. Медиатека часто на
    USB-диске с головками: холодное чтение начала файла — 50–300 мс, уснувший диск — секунды, и
    всё это ложилось на первый звук (замер 26.10.2026: первый кусок 1 с, следующие 0,17 с)."""
    def run():
        for p in paths:
            try:
                with open(p, "rb", buffering=0) as f:
                    f.read(WARM_BYTES)
            except OSError:
                pass
    threading.Thread(target=run, name="warm-starts", daemon=True).start()


def _quiet(fn) -> None:
    try:
        fn()
    except Exception as e:  # прогрев не должен ронять запуск
        print(f"прогрев: {e!r}", flush=True)


def create_app(cfg: Config, reach=None, fingerprint: str | None = None) -> FastAPI:
    """reach — состояние доступа из интернета (reach.Reach), fingerprint — отпечаток сертификата
    порта для интернета. Оба None — сервер только для домашней сети."""
    app = FastAPI(title="Arx Nova", version=__version__)
    st = State(cfg)
    app.state.nova = st
    # модель подбора — в фоне сразу при запуске: первый «следующий трек» не ждёт её построения
    # и вкус владельца: первый «Мой поток» после перезапуска сервера ждал его сборки ~1,5 с
    # (замер 26.10.2026); собираем сразу за моделью
    def warm():
        lib = st.library()
        if lib is None:
            return
        with st.lock:
            owners = [r[0] for r in st.conn.execute("SELECT id FROM devices WHERE role = 'owner'")]  # как listener_devices
        if owners:
            st.profile(lib, owners, int(time.time() * 1000))
    threading.Thread(target=lambda: _quiet(warm), name="warm-library", daemon=True).start()

    def from_internet(request: Request) -> bool:
        """Запрос пришёл на порт для интернета (ОД-3а) — там действуют более строгие правила."""
        server = request.scope.get("server")
        return bool(cfg.remote_port) and bool(server) and server[1] == cfg.remote_port

    bad_tokens: dict[str, list[float]] = {}
    BAD_TOKEN_LIMIT, BAD_TOKEN_WINDOW_S = 20, 600

    @app.middleware("http")
    async def internet_guard(request: Request, call_next):
        """Порт для интернета (ОД-3а): только API /v1/; служебные страницы (описание API) — только
        дома. С адреса, который раз за разом шлёт чужие токены, запросы без действующего токена
        получают отказ на 10 минут. Действующий токен проходит всегда: за одним адресом сидят
        многие (роутер, мобильный оператор), и отозванный гость не должен отрезать владельца."""
        if not from_internet(request):
            return await call_next(request)
        if not request.url.path.startswith("/v1/"):
            return Response(status_code=404)
        ip = request.client.host if request.client else "?"
        now = time.time()
        recent = [t for t in bad_tokens.get(ip, []) if now - t < BAD_TOKEN_WINDOW_S]
        bad_tokens[ip] = recent
        if len(recent) >= BAD_TOKEN_LIMIT and not valid_token(request.headers.get("authorization")):
            return Response(status_code=429)
        resp = await call_next(request)
        if resp.status_code == 401:
            bad_tokens[ip] = recent + [now]
        return resp

    def valid_token(authorization: str | None) -> bool:
        if not authorization or " " not in authorization:
            return False
        with st.lock:
            return auth.device_for_token(st.conn, authorization.split(None, 1)[1].strip()) is not None

    def net_json() -> dict:
        out = {"tls": None, "external": None}
        if fingerprint and cfg.remote and cfg.remote_port:
            out["tls"] = {"port": cfg.remote_port, "fingerprint": fingerprint}
            # плеер вне дома может пробить NAT до этого сервера (UDP на том же порту)
            out["tunnel"] = getattr(app.state, "tunnel", None) is not None
            if reach is not None:
                out.update(reach.status.as_json())
        return out

    def device(authorization: str | None = Header(default=None)) -> sqlite3.Row:
        if not authorization or not authorization.lower().startswith("bearer "):
            raise HTTPException(401, "нужен токен устройства")
        # одно соединение SQLite на весь сервер: запись last_seen — под общей блокировкой
        with st.lock:
            row = auth.device_for_token(st.conn, authorization.split(None, 1)[1].strip())
        if not row:
            raise HTTPException(401, "токен неизвестен или отозван")
        return row

    app.state.throttle = guests.Throttle()
    app.state.streams = guests.Streams()

    def owner(dev=Depends(device)) -> sqlite3.Row:
        """Управлять доступом может только владелец (ОД-5)."""
        if dev["role"] != "owner":
            raise HTTPException(403, "только для владельца медиатеки")
        return dev

    def lan_url() -> str | None:
        ip = reach.status.local_ip if reach is not None else None
        return f"http://{ip}:{cfg.port}" if ip else None

    def listener_devices(dev) -> list[int]:
        """Чьи отметки считать своими: у владельца — все его устройства, у гостя — только своё."""
        if dev["role"] == "owner":
            with st.lock:
                return [r[0] for r in st.conn.execute("SELECT id FROM devices WHERE role = 'owner'")]
        return [dev["id"]]

    @app.get("/v1/health")
    def health(request: Request, authorization: str | None = Header(default=None)):
        if from_internet(request):  # из интернета без токена — только «я здесь»
            token = authorization.split(None, 1)[1].strip() if authorization and " " in authorization else ""
            with st.lock:
                known = bool(token) and auth.device_for_token(st.conn, token) is not None
            if not known:
                return {"name": "arxnova", "version": __version__}
        # «жив ли сервер» — отвечает сразу: модель подбора не строим (на большой медиатеке это
        # секунды, окно приложения ждёт ответа 5 с), берём готовую или число строк индекса
        with st.lock:  # одно соединение на все потоки: без замка параллельные запросы рвут курсор
            n = st.conn.execute("SELECT count(*) FROM tracks").fetchone()[0]
        indexed = st.indexed_count()
        with st.lock:
            unclaimed = not auth.has_owner(st.conn)
        out = {"name": "arxnova", "version": __version__, "server": st.server_name, "tracks": n,
               "model": models.EFFNET.id if indexed else None, "indexed": indexed,
               "libraries": len(cfg.libraries)}
        if unclaimed and from_home(request):
            out["setup"] = True  # новый сервер: плеер в этой сети может забрать его без кода
        # что не так на сервере — плеер покажет словами (у сервера без экрана больше негде)
        problems = []
        missing = [str(p) for p in cfg.libraries if not Path(p).is_dir()]
        if missing:
            problems.append({"code": "library_missing", "detail": ", ".join(missing)})
        if not models.is_installed(cfg.models_dir, models.EFFNET):
            problems.append({"code": "no_model"})
        latest = updates.latest()
        if latest and updates.newer(latest, __version__):
            problems.append({"code": "update", "detail": latest})
        if problems:
            out["problems"] = problems
        return out

    def from_home(request: Request) -> bool:
        """Запрос из домашней сети напрямую: не порт для интернета, не канал пробивки, адрес
        частный (или этот же компьютер)."""
        import ipaddress

        if from_internet(request) or request.client is None:
            return False
        try:
            ip = ipaddress.ip_address(request.client.host)
        except ValueError:
            return False
        return ip.is_private or ip.is_loopback

    # ---- первая настройка с телефона: забрать сервер, выбрать папку с музыкой ----

    @app.post("/v1/setup/claim")
    def setup_claim(body: ClaimIn, request: Request):
        if not from_home(request):
            raise HTTPException(403, "забрать новый сервер можно только из домашней сети")
        try:
            with st.lock:
                dev_id, token = auth.claim(st.conn, body.name)
        except auth.PairError as e:
            raise HTTPException(409, str(e))
        print(f"Сервер забрал «{body.name}» — теперь это его владелец", flush=True)
        return {"device_id": dev_id, "token": token, "server": st.server_name, "net": net_json(),
                "libraries": [str(p) for p in cfg.libraries]}

    @app.get("/v1/setup/folders")
    def setup_folders(path: str | None = None, dev=Depends(owner)):
        """Обзор папок сервера: без path — корни (домашняя папка, диски) и догадки, где музыка."""
        if path is None:
            return {"roots": server_folders.roots(), "suggested": server_folders.suggest(),
                    "libraries": [str(p) for p in cfg.libraries]}
        try:
            return server_folders.listing(path)
        except (FileNotFoundError, PermissionError, OSError):
            raise HTTPException(404, "папка недоступна")

    # ---- модели разбора: согласие и установка из плеера (у сервера без экрана — больше негде) ----
    app.state.model_jobs = {}  # id → "installing" | "error: …"

    @app.get("/v1/setup/models")
    def setup_models(dev=Depends(owner)):
        return {"models": [
            {"id": m.id, "title": m.title, "license": m.license, "license_url": m.license_url,
             "attribution": m.attribution, "consent": m.consent, "size_mb": m.size_mb,
             "installed": models.is_installed(cfg.models_dir, m), "state": app.state.model_jobs.get(m.id)}
            for m in (models.EFFNET,)]}  # MAEST — по желанию, из командной строки (350 МБ)

    @app.post("/v1/setup/models/{model_id}")
    def setup_model_install(model_id: str, dev=Depends(owner)):
        """Владелец согласился с лицензией в плеере — скачать модель и начать разбор."""
        m = models.CATALOG.get(model_id)
        if m is None or m is not models.EFFNET:
            raise HTTPException(404, "такой модели нет")
        if models.is_installed(cfg.models_dir, m) or app.state.model_jobs.get(m.id) == "installing":
            return {"state": "installed" if models.is_installed(cfg.models_dir, m) else "installing"}
        app.state.model_jobs[m.id] = "installing"
        print(f"Модель {m.id}: согласие получено с «{dev['name']}» — скачиваю", flush=True)

        def run():
            try:
                models.install(cfg.models_dir, m, progress=lambda s: print(s, flush=True))
                app.state.model_jobs.pop(m.id, None)
                hook = getattr(app.state, "on_libraries", None)
                if hook:
                    hook()  # сразу к разбору медиатеки
            except Exception as e:  # noqa: BLE001
                app.state.model_jobs[m.id] = f"error: {e}"
        threading.Thread(target=run, name=f"model-{m.id}", daemon=True).start()
        return {"state": "installing"}

    @app.put("/v1/setup/library")
    def setup_library(body: LibraryIn, dev=Depends(owner)):
        paths = [Path(p).expanduser() for p in body.paths]
        bad = [str(p) for p in paths if not p.is_dir()]
        if bad:
            raise HTTPException(400, f"нет такой папки: {', '.join(bad)}")
        cfg.libraries[:] = paths  # тот же список читают обход и наблюдатель за папками
        cfg.save()
        hook = getattr(app.state, "on_libraries", None)
        if hook:
            hook()
        print(f"Медиатека: {', '.join(map(str, paths))} (выбрано с телефона)", flush=True)
        return {"libraries": [str(p) for p in paths]}

    # ---- своё устройство без кода: запрос → «Разрешить» на устройстве владельца ----
    app.state.access = access.AccessRequests()

    def approve_on_server(rid: str, role: str) -> None:
        try:
            with st.lock:
                app.state.access.answer(st.conn, rid, True, role, cfg.max_guests)
        except access.AccessError:
            pass  # уже ответили с плеера владельца или запрос истёк

    # Подтверждение на самом сервере (окно / кнопка Pi); запускает cli при старте сервера.
    app.state.server_confirm = presence.ServerConfirm(app.state.access.pending, approve_on_server, st.server_name)

    @app.post("/v1/access/request")
    def access_request(body: ClaimIn, request: Request):
        if not from_home(request):
            raise HTTPException(403, "попросить доступ можно из домашней сети, иначе — по коду")
        with st.lock:
            if not auth.has_owner(st.conn):
                raise HTTPException(409, "у сервера ещё нет владельца — его можно забрать без запроса")
        try:
            r = app.state.access.ask(body.name)
        except access.AccessError as e:
            raise HTTPException(e.status, str(e))
        confirm = presence.method()
        if confirm == "home":
            with st.lock:
                gone = presence.owners_gone(st.conn)
            if gone:
                # Дом без хозяина: Linux без монитора и кнопки, устройств владельца не осталось —
                # устройство из домашней сети становится своим само.
                approve_on_server(r.id, "owner")
                print(f"«{r.name}»: своё устройство — у сервера не осталось устройств владельца", flush=True)
            else:
                confirm = "device"
        else:
            where = "на экране сервера" if confirm == "window" else "кнопкой на корпусе Raspberry Pi"
            print(f"«{r.name}» просит доступ — подтвердите {where} или на своём плеере (цифры {r.check})", flush=True)
        return {"id": r.id, "secret": r.secret, "check": r.check, "server": st.server_name, "ttl": access.TTL_S,
                "confirm": confirm}

    @app.get("/v1/access/request/{rid}")
    def access_status(rid: str, secret: str):
        try:
            r = app.state.access.poll(rid, secret)
        except access.AccessError as e:
            raise HTTPException(e.status, str(e))
        out = {"status": r.status, "check": r.check}
        if r.status == "approved":
            out.update(device_id=r.device_id, token=r.token, role=r.role, server=st.server_name, net=net_json())
            if r.role == "guest":
                out["owner"] = cfg.inviter
        return out

    @app.get("/v1/access/requests")
    def access_pending(dev=Depends(owner)):
        now = time.time()
        return {"requests": [{"id": r.id, "name": r.name, "check": r.check, "age_s": int(now - r.created)}
                             for r in app.state.access.pending()]}

    @app.post("/v1/access/requests/{rid}")
    def access_answer(rid: str, body: AccessAnswerIn, dev=Depends(owner)):
        try:
            with st.lock:
                r = app.state.access.answer(st.conn, rid, body.allow, body.role, cfg.max_guests)
        except access.AccessError as e:
            raise HTTPException(e.status, str(e))
        what = ("своё устройство" if r.role == "owner" else "гость") if body.allow else "отказано"
        print(f"«{r.name}»: {what} — решение с «{auth.tidy_name(dev['name'])}»", flush=True)
        return {"status": r.status}

    @app.get("/v1/devices")
    def my_devices(dev=Depends(owner)):
        """Устройства владельца — чтобы увидеть, кому дан полный доступ, и отключить лишнее
        (гости — отдельно, /v1/guests)."""
        with st.lock:
            rows = st.conn.execute(
                "SELECT id, name, created_at, last_seen FROM devices WHERE role = 'owner' AND revoked = 0 ORDER BY created_at").fetchall()
        return {"devices": [{"id": r["id"], "name": auth.tidy_name(r["name"]), "since": r["created_at"], "last_seen": r["last_seen"],
                             "me": r["id"] == dev["id"]} for r in rows]}

    @app.delete("/v1/devices/{device_id}")
    def revoke_device(device_id: int, dev=Depends(owner)):
        if device_id == dev["id"]:
            raise HTTPException(400, "это устройство — отключите его на другом своём устройстве")
        with st.lock:
            n = st.conn.execute("UPDATE devices SET revoked = 1 WHERE id = ? AND role = 'owner' AND revoked = 0",
                                (device_id,)).rowcount
            st.conn.commit()
        if not n:
            raise HTTPException(404, "такого устройства нет")
        print(f"Устройство #{device_id} отключено с «{dev['name']}»", flush=True)
        return {"revoked": device_id}

    @app.post("/v1/pair/code")
    def pair_code(dev=Depends(owner)):
        """Код для своего устройства — из плеера владельца, без окна сервера и терминала."""
        with st.lock:
            code = auth.new_pair_code(st.conn)
        return {"code": code, "ttl": auth.CODE_TTL_S}

    @app.post("/v1/pair/hello")
    def pair_hello(body: PairHelloIn):
        """Шаг 1 сопряжения из любой точки: плеер по ключу из кода спрашивает, тот ли это сервер.
        Сервер отвечает отпечатком своего сертификата и доказательством, что знает код; сам код
        плеер отдаст только после проверки (см. auth)."""
        try:
            key, nonce = bytes.fromhex(body.key), bytes.fromhex(body.nonce)
        except ValueError:
            raise HTTPException(400)
        if len(key) != 20 or not 16 <= len(nonce) <= 64 or not fingerprint:
            raise HTTPException(400)
        with st.lock:
            secret = auth.secret_for_key(st.conn, key)
        if secret is None:
            raise HTTPException(404, "код неверный или истёк")
        return {"fingerprint": fingerprint, "proof": auth.pair_proof(secret, nonce, fingerprint), "server": st.server_name}

    @app.post("/v1/pair")
    def pair(body: PairIn, request: Request):
        # Сопряжение — и дома, и из интернета (решение владельца 22.10.2026); из интернета
        # попыток меньше (auth.MAX_FAILED_REMOTE). Гости входят по приглашениям (ОД-1, ОД-2).
        try:
            with st.lock:
                dev_id, token = auth.redeem(st.conn, body.code, body.name, remote=from_internet(request))
        except auth.PairError as e:
            raise HTTPException(403, str(e))
        return {"device_id": dev_id, "token": token, "server": st.server_name, "net": net_json()}

    # ---- общий доступ (§15.1) ----

    @app.post("/v1/invites")
    def create_invite(body: InviteIn, dev=Depends(owner)):
        if body.owner_name is not None and body.owner_name.strip() != cfg.owner_name:
            cfg.owner_name = body.owner_name.strip()
            cfg.save()
        try:
            with st.lock:
                inv = guests.create_invite(st.conn, body.name, dev["id"], cfg.max_guests)
        except guests.GuestError as e:
            raise HTTPException(e.status, str(e))
        inviter = cfg.inviter
        inv["link"] = guests.link(inv["code"], st.server_name, inviter, net_json(), lan_url())
        inv["server"] = st.server_name
        return inv

    @app.get("/v1/guests")
    def list_guests(dev=Depends(owner)):
        with st.lock:
            out = guests.list_guests(st.conn)
        out["limit"] = guests.limit(cfg.max_guests)
        out["owner_name"] = cfg.inviter
        return out

    @app.delete("/v1/guests/{guest_id}")
    def revoke_guest(guest_id: int, dev=Depends(owner)):
        with st.lock:
            if not guests.revoke_guest(st.conn, guest_id):
                raise HTTPException(404, "такого гостя нет")
        return {"revoked": guest_id}

    @app.delete("/v1/invites/{invite_id}")
    def revoke_invite(invite_id: int, dev=Depends(owner)):
        with st.lock:
            ok = guests.revoke_invite(st.conn, invite_id)
        if not ok:
            raise HTTPException(404, "такого приглашения нет")
        return {"revoked": invite_id}

    @app.post("/v1/join")
    def join(body: JoinIn, request: Request):
        """Вход гостя по коду (ОД-2) — в том числе из интернета: гости живут не дома."""
        ip = request.client.host if request.client else "?"
        try:
            with st.lock:
                dev_id, token, inv = guests.join(st.conn, body.code, body.name, ip, app.state.throttle)
        except guests.GuestError as e:
            raise HTTPException(e.status, str(e))
        return {"device_id": dev_id, "token": token, "server": st.server_name,
                "owner": cfg.inviter, "guest_name": inv["guest_name"],
                "role": "guest", "net": net_json()}

    @app.get("/v1/devices/me")
    def me(dev=Depends(device)):
        return {"id": dev["id"], "name": dev["name"], "role": dev["role"], "server": st.server_name,
                "guest_name": dev["guest_name"], "owner": cfg.inviter}

    # ---- окно приложения на этом же компьютере (без токена, только с 127.0.0.1) ----

    def local_only(request: Request) -> None:
        host = request.client.host if request.client else ""
        if from_internet(request) or host not in ("127.0.0.1", "::1"):
            raise HTTPException(403, "только с этого компьютера")

    @app.get("/v1/local/net")
    def local_net(request: Request):
        local_only(request)
        return {"remote": bool(reach is not None), **(reach.details() if reach is not None else {}), **tunnel_json()}

    @app.post("/v1/local/net/check")
    def local_net_check(request: Request):
        """Кнопка «Проверить и настроить сеть»: проброс, внешний адрес, проверка снаружи, DHT — сейчас."""
        local_only(request)
        if reach is None:
            return {"remote": False}
        reach.refresh(force=True)
        return {"remote": True, **reach.details(), **tunnel_json()}

    def tunnel_json() -> dict:
        t = getattr(app.state, "tunnel", None)
        if t is None:
            return {"tunnel": None}
        return {"tunnel": {"dht_nodes": t.dht_nodes, "sessions": len(t.sessions), "knocking": t.peers_seen}}

    @app.get("/v1/net")
    def net(dev=Depends(device)):
        """Как достучаться до сервера вне дома: порт с TLS, отпечаток ключа, внешний адрес."""
        return net_json()

    @app.post("/v1/events")
    def events(body: EventsIn, dev=Depends(device)):
        added = dup = bad = 0
        with st.lock, st.conn:
            for e in body.events:
                if e.event_type not in EVENT_TYPES or (e.skip_reason and e.skip_reason not in SKIP_REASONS):
                    bad += 1
                    continue
                tid, md5, rk = e.track_id, None, None
                if e.md5:
                    try:
                        md5 = bytes.fromhex(e.md5)
                    except ValueError:
                        bad += 1
                        continue
                    row = st.conn.execute("SELECT id, recording_key FROM tracks WHERE md5 = ? AND cue_ms = ?", (md5, e.cue_ms)).fetchone()
                    if row:
                        tid, rk = row["id"], row["recording_key"]
                elif tid is not None:
                    row = st.conn.execute("SELECT md5, cue_ms, recording_key FROM tracks WHERE id = ?", (tid,)).fetchone()
                    if not row:
                        bad += 1
                        continue
                    md5, rk = row["md5"], row["recording_key"]
                    e.cue_ms = row["cue_ms"]
                else:
                    bad += 1
                    continue
                cur = st.conn.execute(
                    """INSERT OR IGNORE INTO events (event_id, device_id, ts, md5, cue_ms, track_id, event_type,
                           played_ratio_pct, track_length_s, is_organic, session_id, prev_track_id,
                           transition_type, nova_request_id, recording_key, skip_reason)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (e.event_id, dev["id"], e.ts, md5, e.cue_ms, tid, e.event_type, e.played_ratio_pct,
                     e.track_length_s, int(e.is_organic), e.session_id, e.prev_track_id, e.transition_type,
                     e.nova_request_id, rk, e.skip_reason if e.event_type == "skip" else None),
                )
                if cur.rowcount:
                    added += 1
                else:
                    dup += 1
        return {"accepted": added, "duplicates": dup, "rejected": bad}

    @app.get("/v1/nova/facets")
    def facets(min_tracks: int = Query(nova.FACET_MIN_TRACKS, ge=1), styles: list[str] = Query([]),
               dev=Depends(device)):
        lib = st.library()
        if not lib:
            raise HTTPException(503, "медиатека ещё не проиндексирована или модель не установлена")
        return lib.facets(min_tracks, styles or None)

    @app.post("/v1/nova/next")
    def nova_next(body: NextIn, dev=Depends(device)):
        lib = st.library()
        if not lib:
            raise HTTPException(503, "медиатека ещё не проиндексирована или модель не установлена")
        now = int(time.time() * 1000)
        prof = st.profile(lib, listener_devices(dev), now)  # вкус владельца — со всех его устройств
        with st.lock:
            allowed = None
            if body.only_on_device:
                allowed = {r[0] for r in st.conn.execute(
                    "SELECT t.id FROM tracks t JOIN device_catalog c ON c.md5 = t.md5 AND c.device_id = ?", (dev["id"],))}
            gate_log: list = []
            filters = body.filters.to_filters(st.cfg.boldness)
            picks = nova.next_tracks(
                lib, prof, current=body.current_track, history=body.history,
                filters=filters,
                n=body.n, now_ms=now, allowed=allowed,
                mode=nova.MODE_TRACK if body.mode == "track" else nova.MODE_MY, seed=body.seed_track,
                gate_log=gate_log, start=body.start,
            )
            req_id = nova.log_request(st.conn, dev["id"], body.current_track, picks, gate_log)
        cat = st.catalog()
        _warm_starts([cat.rows[p.track_id]["path"] for p in picks if p.track_id in cat.rows])
        out = {"request_id": req_id, "tracks": [track_json(cat.rows[p.track_id], lib, p.reason, p) for p in picks if p.track_id in cat.rows]}
        if filters.narrow():  # «мало треков» — плеер предложит ослабить фильтр (ФТ-4.10)
            out["filter_tracks"] = nova.filter_count(lib, filters)
            out["min_tracks"] = nova.FACET_MIN_TRACKS
        return out

    def collect_marks(dev) -> tuple[dict[int, int], set[int]]:
        """Лайки (track_id → время лайка) и «не предлагать» — последняя отметка по треку побеждает."""
        ids = listener_devices(dev)
        liked: dict[int, int] = {}
        disliked: set[int] = set()
        q = ",".join("?" * len(ids))
        with st.lock:
            rows = st.conn.execute(
                f"""SELECT track_id, event_type, ts FROM events WHERE device_id IN ({q}) AND track_id IS NOT NULL
                   AND event_type IN ('like','unlike','dislike','undislike') ORDER BY ts""", ids).fetchall()
        for r in rows:
            t, et = r["track_id"], r["event_type"]
            if et == "like":
                liked[t] = r["ts"]; disliked.discard(t)
            elif et == "unlike":
                liked.pop(t, None)
            elif et == "dislike":
                disliked.add(t); liked.pop(t, None)
            else:
                disliked.discard(t)
        return liked, disliked

    @app.get("/v1/devices/me/marks")
    def marks(dev=Depends(device)):
        liked, disliked = collect_marks(dev)
        return {"liked": sorted(liked), "disliked": sorted(disliked)}

    @app.get("/v1/library/favorites")
    def favorites(dev=Depends(device)):
        """«Избранное»: лайкнутые треки, свежие сверху. У владельца — со всех его устройств,
        у гостя — только его собственные."""
        liked, _ = collect_marks(dev)
        cat = st.catalog()
        lib = st.library()
        order = sorted(liked, key=lambda t: -liked[t])
        return {"tracks": [track_json(cat.rows[t], lib) for t in order if t in cat.rows]}

    @app.delete("/v1/devices/me/history")
    def delete_history(dev=Depends(device)):
        """Удалить историю этого устройства (ФТ-1.5, П-7): события и решения подбора."""
        with st.lock, st.conn:
            n = st.conn.execute("DELETE FROM events WHERE device_id = ?", (dev["id"],)).rowcount
            st.conn.execute("DELETE FROM nova_requests WHERE device_id = ?", (dev["id"],))
        return {"deleted": n}

    @app.get("/v1/devices/me/mixes")
    def device_mixes(dev=Depends(device)):
        """Заготовленные подборки для игры без сервера (ФТ-4.7, П-5): только из треков, что лежат
        на этом приборе; у каждого трека — MD5 звука и смещение CUE, чтобы плеер нашёл его у себя."""
        lib = st.library()
        if not lib:
            raise HTTPException(503, "медиатека ещё не проиндексирована или модель не установлена")
        with st.lock:
            data, built = mixes.get(st.conn, lib, dev["id"], listener_devices(dev))
        cat = st.catalog()
        out = []
        for m in data:
            tracks = [track_json(cat.rows[t["id"]], lib, t["reason"]) for t in m["tracks"] if t["id"] in cat.rows]
            if tracks:
                out.append({"seed": m["seed"], "title": tracks[0]["title"], "style": tracks[0].get("style"), "tracks": tracks})
        return {"built_at": built, "mixes": out}

    @app.put("/v1/devices/me/catalog")
    def put_catalog(md5s: list[str], dev=Depends(device)):
        with st.lock, st.conn:
            st.conn.execute("DELETE FROM device_catalog WHERE device_id = ?", (dev["id"],))
            st.conn.executemany("INSERT OR IGNORE INTO device_catalog VALUES (?,?)",
                                [(dev["id"], bytes.fromhex(m)) for m in md5s if len(m) == 32])
            matched = st.conn.execute(
                "SELECT count(*) FROM tracks t JOIN device_catalog c ON c.md5 = t.md5 AND c.device_id = ?",
                (dev["id"],)).fetchone()[0]
        return {"tracks": len(md5s), "matched": matched}

    # ---- просмотр медиатеки (П-13) ----

    def page(items: list, offset: int, limit: int) -> dict:
        return {"total": len(items), "offset": offset, "items": items[offset : offset + limit]}

    @app.get("/v1/library/albums")
    def albums(offset: int = 0, limit: int = Query(100, le=500), dev=Depends(device)):
        cat = st.catalog()
        items = sorted(cat.albums.values(), key=lambda a: (keys.normalize(a["artist"]), a["year"], keys.normalize(a["title"])))
        items = [{"id": a["id"], "title": a["title"], "artist": "" if a["artist"] == VARIOUS_ARTISTS else a["artist"],
                  "various": a["artist"] == VARIOUS_ARTISTS, "year": a["year"],
                  "tracks": len(a["tracks"]), "cover_track": a["tracks"][0]} for a in items]
        return page(items, offset, limit)

    @app.get("/v1/library/albums/{album_id}")
    def album(album_id: str, dev=Depends(device)):
        cat = st.catalog()
        a = cat.album_by_id.get(album_id)
        if not a:
            raise HTTPException(404)
        lib = st.library()
        return {"id": a["id"], "title": a["title"], "artist": "" if a["artist"] == VARIOUS_ARTISTS else a["artist"],
                "various": a["artist"] == VARIOUS_ARTISTS, "year": a["year"],
                "tracks": [track_json(cat.rows[t], lib) for t in a["tracks"]]}

    @app.get("/v1/library/artists")
    def artists(offset: int = 0, limit: int = Query(200, le=1000), dev=Depends(device)):
        cat = st.catalog()
        items = sorted(cat.artists.values(), key=lambda a: a["name"])
        return page([{"id": a["id"], "name": a["display"], "tracks": len(a["tracks"])} for a in items], offset, limit)

    @app.get("/v1/library/artists/{artist_id}")
    def artist(artist_id: str, dev=Depends(device)):
        cat = st.catalog()
        a = cat.artist_by_id.get(artist_id)
        if not a:
            raise HTTPException(404)
        lib = st.library()
        tracks = sorted(a["tracks"], key=lambda t: (cat.rows[t]["album"] or "", cat.rows[t]["path"]))
        return {"id": a["id"], "name": a["display"], "tracks": [track_json(cat.rows[t], lib) for t in tracks]}

    @app.get("/v1/library/folders")
    def folders(path: str = "", dev=Depends(device)):
        """Папки медиатеки как в проводнике: path — относительный путь от корня (корни —
        «0/», «1/» … если их несколько)."""
        cat = st.catalog()
        lib = st.library()
        prefix = path.strip("/")
        dirs: dict[str, int] = {}
        files = []
        multi = len(cat.roots) > 1
        for r in cat.rows.values():
            rel = cat.rel(r["path"])
            if not rel:
                continue
            i, p = rel
            full = f"{i}/{p}" if multi else p
            if prefix and not full.startswith(prefix + "/"):
                continue
            rest = full[len(prefix) + 1 :] if prefix else full
            head, sep, _ = rest.partition("/")
            if sep:
                dirs[head] = dirs.get(head, 0) + 1
            else:
                files.append(r)
        files.sort(key=lambda r: r["path"])
        return {"path": prefix, "folders": [{"name": k, "tracks": v} for k, v in sorted(dirs.items(), key=lambda kv: kv[0].lower())],
                "tracks": [track_json(r, lib) for r in files]}

    @app.get("/v1/tracks/{track_id}/folder")
    def track_folder(track_id: int, dev=Depends(device)):
        """Папка трека в проводнике медиатеки — тем же путём, что /v1/library/folders (нажатие на
        название трека из потока открывает его папку, замечание владельца 26.10.2026)."""
        cat = st.catalog()
        r = cat.rows.get(track_id)
        rel = cat.rel(r["path"]) if r else None
        if not rel:
            raise HTTPException(404)
        i, p = rel
        full = f"{i}/{p}" if len(cat.roots) > 1 else p
        return {"path": full.rpartition("/")[0]}

    @app.get("/v1/library/folders/tracks")
    def folder_tracks(path: str = "", dev=Depends(device)):
        """«Играть папку»: все треки папки вместе с подпапками, в порядке проводника
        (по пути, внутри CUE-образа — по началу трека)."""
        cat = st.catalog()
        lib = st.library()
        prefix = path.strip("/")
        multi = len(cat.roots) > 1
        picked = []
        for r in cat.rows.values():
            rel = cat.rel(r["path"])
            if not rel:
                continue
            i, p = rel
            full = f"{i}/{p}" if multi else p
            if not prefix or full.startswith(prefix + "/"):
                picked.append((full.lower(), r["cue_ms"], r))
        picked.sort(key=lambda x: (x[0], x[1]))
        return {"path": prefix, "tracks": [track_json(r, lib) for _, _, r in picked[:5000]]}

    @app.get("/v1/library/genres")
    def genres(dev=Depends(device)):
        lib = st.library()
        if not lib:
            return {"items": []}
        return lib.facets(1)

    @app.get("/v1/library/styles/{style}")
    def style_tracks(style: str, dev=Depends(device)):
        lib = st.library()
        if not lib:
            raise HTTPException(503)
        cat = st.catalog()
        ids = [t.id for t in lib.tracks.values() if t.style == style or t.id in lib.shelves.get(style, ())]
        return {"style": style, "tracks": [track_json(cat.rows[t], lib) for t in ids if t in cat.rows]}

    @app.get("/v1/library/search")
    def search(q: str = Query(min_length=1), limit: int = Query(100, le=500), dev=Depends(device)):
        cat = st.catalog()
        lib = st.library()
        nq = keys.normalize(q)
        hits = [r for r in cat.rows.values()
                if nq in keys.normalize(f"{r['artist']} {r['title']} {r['album']}")]
        hits.sort(key=lambda r: (keys.normalize(r["artist"]), keys.normalize(r["title"])))
        return {"q": q, "tracks": [track_json(r, lib) for r in hits[:limit]]}

    # ---- разметка мостов и калибровка шлюза (КО-1, КО-2, ФТ-8.4) — только с этого компьютера ----

    def bridge_track(tid: int) -> dict:
        with st.lock:
            r = st.conn.execute("SELECT artist, title, duration, cue_ms FROM tracks WHERE id = ?", (tid,)).fetchone()
        lib = st.library()
        t = lib.tracks.get(tid) if lib else None
        return {"id": tid, "artist": r["artist"], "title": r["title"], "duration": r["duration"],
                "offset": (r["cue_ms"] or 0) / 1000.0, "style": t.style if t else ""}

    @app.get("/ui/bridges")
    def bridges_page(request: Request):
        local_only(request)
        return Response(bridges.PAGE, media_type="text/html; charset=utf-8")

    @app.get("/v1/local/bridges/next")
    def bridges_next(request: Request):
        local_only(request)
        lib = st.library()
        if not lib:
            raise HTTPException(503, "медиатека ещё не проиндексирована")
        with st.lock:
            bridges.ensure_candidates(st.conn, lib)
            row = bridges.next_unlabeled(st.conn)
            prog = bridges.progress(st.conn)
        if not row:
            return {"done": True, **prog}
        return {"done": False, "a": bridge_track(row[0]), "b": bridge_track(row[1]), "mix": round(row[2], 3), **prog}

    @app.post("/v1/local/bridges/label")
    def bridges_label(body: LabelIn, request: Request):
        local_only(request)
        try:
            with st.lock:
                bridges.label(st.conn, body.a, body.b, body.label)
        except ValueError:
            raise HTTPException(400, "label: good | bad | skip")
        return {"ok": True}

    @app.get("/v1/local/bridges/calibration")
    def bridges_calibration(request: Request):
        local_only(request)
        lib = st.library()
        if not lib:
            raise HTTPException(503)
        with st.lock:
            cal = bridges.calibrate(st.conn, lib)
        return {**cal, "current_pct": cfg.gate_pct, "facets": lib.has_facets()}

    def _carry_settings():
        """Настройки потока — сразу и в медиатеку (см. sidecar): диск на другом сервере — тот же поток."""
        from arxnova import sidecar

        threading.Thread(target=sidecar.sync_settings, args=(cfg, cfg.libraries), daemon=True).start()

    @app.post("/v1/local/bridges/apply")
    def bridges_apply(body: PctIn, request: Request):
        local_only(request)
        cfg.gate_pct = body.pct
        cfg.save()
        _carry_settings()
        st._lib = None  # шлюз пересобирается с новым перцентилем
        return {"ok": True, "pct": cfg.gate_pct}

    # ---- анализ медиатеки: состояние, кнопка «сейчас», компоненты (окно приложения) ----

    @app.get("/v1/local/analysis")
    def analysis_state(request: Request):
        local_only(request)
        from arxnova import analysis

        with st.lock:
            counts = analysis.pending(st.conn)
        return {**counts, "installed": analysis.installed(cfg), "size_gb": analysis.SIZE_HINT_GB, **analysis.state}

    @app.post("/v1/local/settings/musicbrainz")
    def settings_musicbrainz(body: ToggleIn, request: Request):
        """Жанры из MusicBrainz — переключатель в окне приложения (по согласию владельца)."""
        local_only(request)
        from arxnova import analysis

        cfg.musicbrainz = body.on
        cfg.save()
        if body.on:
            analysis.wake.set()  # спросить сразу, не дожидаясь изменений в папке
        return {"ok": True, "musicbrainz": cfg.musicbrainz}

    @app.post("/v1/local/analysis/run")
    def analysis_run(request: Request):
        local_only(request)
        from arxnova import analysis

        analysis.wake.set()
        return {"ok": True}

    @app.post("/v1/local/analysis/install")
    def analysis_install(request: Request):
        local_only(request)
        from arxnova import analysis

        def go():
            analysis.install(cfg)
            analysis.wake.set()  # сразу разобрать медиатеку

        threading.Thread(target=go, name="analysis-install", daemon=True).start()
        return {"ok": True}

    @app.get("/ui/gate")
    def gate_page(request: Request):
        local_only(request)
        return Response(bridges.GATE_PAGE, media_type="text/html; charset=utf-8")

    @app.get("/v1/local/gate/state")
    def gate_state(request: Request, days: int = Query(7, ge=1, le=90)):
        """Панель шлюза (ФТ-8.4): настройки, счётчики отказов по условиям, последние мосты."""
        local_only(request)
        lib = st.library()
        if not lib:
            raise HTTPException(503)
        since = int((time.time() - days * 86400) * 1000)
        counts: dict[str, int] = {}
        with st.lock:
            for (gc,) in st.conn.execute("SELECT gate_counts FROM nova_requests WHERE ts >= ? AND gate_counts IS NOT NULL", (since,)):
                for k, n in json.loads(gc).items():
                    counts[k] = counts.get(k, 0) + n
            recent = st.conn.execute(
                "SELECT ts, from_track, to_track, accepted, failed, detail FROM gate_log WHERE ts >= ? ORDER BY ts DESC LIMIT 40",
                (since,)).fetchall()

        def name(tid):
            t = lib.tracks.get(tid)
            return {"id": tid, "title": t.title if t else "?", "artist": t.artist if t else "", "style": t.style if t else ""}

        styles = sorted({t.style for t in lib.tracks.values()})
        return {
            "boldness": cfg.boldness, "gate_pct": cfg.gate_pct, "forbid": cfg.gate_forbid, "styles": styles,
            "facets": int(lib.gate.has_facets.sum()), "tracks": len(lib.tracks), "counts": counts,
            "recent": [{"ts": r[0], "from": name(r[1]), "to": name(r[2]), "accepted": bool(r[3]), "failed": r[4],
                        "detail": json.loads(r[5]) if r[5] else {}} for r in recent],
        }

    @app.post("/v1/local/gate/settings")
    def gate_settings(body: GateSettingsIn, request: Request):
        local_only(request)
        if body.boldness is not None:
            cfg.boldness = body.boldness
        if body.forbid is not None:
            cfg.gate_forbid = [list(p) for p in body.forbid if len(p) == 2 and p[0] != p[1]]
        cfg.save()
        _carry_settings()
        st._lib = None  # шлюз пересобирается
        return {"ok": True}

    @app.get("/v1/local/map")
    def library_map(request: Request):
        """Карта медиатеки (КО-5): точки UMAP со стилем; считается в `arxnova facets`."""
        local_only(request)
        lib = st.library()
        p = cfg.index_dir / models.EFFNET.id / "map2d.npy"
        if not lib or not p.exists():
            return {"points": []}
        xy = np.load(p)
        if len(xy) != len(lib.ids):
            return {"points": []}
        pts = []
        for (x, y), tid in zip(xy, lib.ids):
            t = lib.tracks[int(tid)]
            pts.append([round(float(x), 4), round(float(y), 4), t.style, t.title or "", t.artist or "", int(tid)])
        return {"points": pts}

    @app.get("/v1/local/audio/{track_id}")
    def local_audio(track_id: int, request: Request):
        local_only(request)
        r = st.catalog().rows.get(track_id)
        if not r:
            raise HTTPException(404)
        return FileResponse(r["path"], media_type="audio/flac")

    # ---- звук и обложки ----

    @app.get("/v1/tracks/{track_id}/audio")
    def audio(track_id: int, request: Request, dev=Depends(device)):
        r = st.catalog().rows.get(track_id)
        if not r:
            raise HTTPException(404)
        if dev["role"] == "owner":
            app.state.streams.owner_listening()
        if dev["role"] == "guest":  # ОД-7: не больше двух гостей сразу и суточный объём
            try:
                app.state.streams.admit(dev["id"])
                size = Path(r["path"]).stat().st_size
                with st.lock:
                    guests.charge(st.conn, dev["id"], _range_len(request.headers.get("range"), size), cfg.guest_daily_mb)
            except guests.GuestError as e:
                raise HTTPException(e.status, str(e))
        # FileResponse отвечает на Range (П-8.2: плеер берёт трек частями)
        return FileResponse(r["path"], media_type="audio/flac", headers={"Cache-Control": "no-store"})

    # ---- скачивание на устройство владельца (решение владельца 26.10.2026) ----
    # Лайк или кнопка «Скачать» кладут файл на устройство — только устройствам владельца: гостю
    # сервер целый файл не отдаёт (у него только поток кусками, без файлов). Путь на устройстве —
    # как на сервере от корня медиатеки: DnB/Liquid/Исполнитель/Альбом/трек.flac.

    COVER_FILES = {"cover.jpg", "cover.jpeg", "cover.png", "folder.jpg", "folder.png", "front.jpg", "front.png"}

    _known_cache: dict = {}

    def _known_paths(cat) -> frozenset[str]:
        """Пути файлов каталога (раскрытые) — один раз на каталог, а не на каждый кусок: раньше
        каждый запрос куска раскрывал все 6000 путей на USB-диске — ~1 с процессора Pi, и канал
        через интернет всё это время стоял (профиль 26.10.2026: 92 % времени сервера)."""
        got = _known_cache.get("cat")
        if got is None or got[0] is not cat:
            got = (cat, frozenset(str(Path(r["path"]).resolve()) for r in cat.rows.values()))
            _known_cache["cat"] = got
        return got[1]

    def _rel(path: Path) -> tuple[Path, str] | None:
        p = path.resolve()
        for root in cfg.libraries:
            r = Path(root).resolve()
            if p == r or r in p.parents:
                return r, p.relative_to(r).as_posix()
        return None

    @app.get("/v1/downloads/plan")
    def download_plan(ids: str = Query(..., max_length=20000), dev=Depends(owner)):
        """Какие файлы скачать для этих треков и куда их положить (путь от корня медиатеки).
        Трек из CUE-образа — образ и .cue к нему (один раз на альбом); обложка папки — рядом."""
        from arxnova import cue as cuemod

        cat = st.catalog()
        out, seen = [], set()

        def add(path: Path, track_id: int | None):
            got = _rel(path)
            if not got or got[1] in seen or not path.is_file():
                return
            seen.add(got[1])
            out.append({"rel": got[1], "size": path.stat().st_size, "track_id": track_id})

        for tid in {int(x) for x in ids.split(",") if x.strip().isdigit()}:
            r = cat.rows.get(tid)
            if not r:
                continue
            path = Path(r["path"])
            add(path, tid)
            found = cuemod.find_for(path)
            if found and found[2] != "embedded":
                add(path.parent / found[2].rsplit(":", 2)[0], None)
            try:
                for f in path.parent.iterdir():
                    if f.name.lower() in COVER_FILES:
                        add(f, None)
            except OSError:
                pass
        return {"files": out}

    @app.get("/v1/downloads/file")
    def download_file(rel: str = Query(..., max_length=4096), dev=Depends(owner)):
        """Файл медиатеки по пути от её корня, с докачкой (Range). Только файлы медиатеки: трек
        каталога или .cue / обложка в папке такого трека; выход за корень — 404."""
        known = _known_paths(st.catalog())
        for root in cfg.libraries:
            p = (Path(root) / rel).resolve()
            r = Path(root).resolve()
            if r not in p.parents or not p.is_file():
                continue
            sibling = p.suffix.lower() == ".cue" or p.name.lower() in COVER_FILES
            if str(p) in known or (sibling and any(str(q.resolve()) in known for q in p.parent.iterdir() if q.is_file())):
                media = "audio/flac" if p.suffix.lower() == ".flac" else None
                return FileResponse(p, media_type=media, filename=p.name, headers={"Cache-Control": "no-store"})
        raise HTTPException(404)

    COVER_SIZES = (128, 256, 512, 1024)

    @app.get("/v1/tracks/{track_id}/cover")
    def cover(track_id: int, size: int | None = Query(None, ge=32, le=2048), dev=Depends(device)):
        """Обложка трека. size — наибольшая сторона: плитки списка берут 256, экран плеера — 1024.
        Без уменьшения одна обложка бывала 7 МБ, а список из 81 альбома — 20 МБ: по мобильной сети
        полминуты на экран. Уменьшенные копии кэшируются на диске (до правки файла)."""
        r = st.catalog().rows.get(track_id)
        if not r:
            raise HTTPException(404)
        src = Path(r["path"])
        if size is None:
            pic = flac.read_picture(r["path"])
            if pic:
                return Response(pic[1], media_type=pic[0], headers={"Cache-Control": "max-age=86400"})
            img = find_cover(src.parent)
            if img:
                return FileResponse(img, headers={"Cache-Control": "max-age=86400"})
            raise HTTPException(404, "обложки нет")
        size = next((s for s in COVER_SIZES if s >= size), COVER_SIZES[-1])
        try:
            mtime = src.stat().st_mtime_ns
        except OSError:
            raise HTTPException(404)
        cache = cfg.tmp_dir / "covers" / f"{track_id}_{size}_{mtime}.jpg"
        if not cache.exists():
            pic = flac.read_picture(r["path"])
            raw = pic[1] if pic else None
            if raw is None:
                img = find_cover(src.parent)
                raw = img.read_bytes() if img else None
            if raw is None:
                raise HTTPException(404, "обложки нет")
            data = thumbnail(raw, size)
            cache.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache.with_suffix(".part")
            tmp.write_bytes(data)
            tmp.replace(cache)
        return FileResponse(cache, media_type="image/jpeg", headers={"Cache-Control": "max-age=604800"})

    return app
