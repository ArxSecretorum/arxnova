"""Разбор медиатеки на другом компьютере по локальной сети (решение 0009).

Сервер на Pi сам считает только общий звук: Demucs занимает у него 19–37 минут на трек. Стили, ритм, вокал,
темп и сценарии считает другой компьютер с Arx Nova: в его окне включают «Анализ медиатеки на другом компьютере»
и выбирают компьютер с медиатекой.

Сервер с медиатекой (функции ниже) отдаёт список недоразобранного и сами файлы, принимает числа. Токен роли
«assist» подходит только к /v1/assist/* и только из домашней сети: чужому компьютеру в той же сети нечего
читать, а этому — больше ничего не положено.

Компьютер-помощник (Worker) по кругу забирает пачку файлов, считает её отдельным процессом в окружении
компонентов анализа, отправляет числа и удаляет файлы. Ключ трека — (MD5, cue_ms) с сервера, а не свой: у
дорожек с потерями MD5 считается по звуку и зависит от версии декодера.

Файлу верим только по sha256: при обрыве tar однажды довёл файл до полного размера мусором, и числа по нему
ушли на Pi как настоящие (Loreena McKennitt, 03.11.2026)."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import re
import shutil
import sqlite3
import sys
import threading
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

from arxnova import analysis, auth, config, db, facets, models, platforms, scenarios
from arxnova.config import Config

BATCH_TRACKS = 50  # пачка на один процесс разбора: на Mac 6–7 минут, потом память GPU начинается заново
BATCH_BYTES = 3 << 30  # а DSD и образы альбомов считаются по объёму, чтобы не копировать десятки ГБ разом
CHUNK_ITEMS = 40  # треков в одной отправке: около мегабайта
FAILED_TTL_S = 24 * 3600  # что не разобралось — снова через сутки, а не по кругу
OFFLINE_S, IDLE_S, RETRY_S = 60, 120, 600
# Wi-Fi под нагрузкой пропадает на минуты (Mac ↔ Pi, 03–04.11.2026). Файл перекачать недолго, а числа — это минуты GPU:
# их ждём дольше и не выбрасываем при первом обрыве
FILE_RETRIES, SEND_RETRIES, BLIP_S = 6, 60, 30
MAEST_CLASSES, MAEST_DIM, EFFNET_DIM, CLAP_DIM = 519, 768, 1280, 512
FACETS = ("rhythm", "instr", "vocal")
NAME_MARK = " · "  # «Mac · разбор медиатеки» в запросе и списке устройств: видно, зачем этот компьютер просит доступ
FEAT_COLS = ("version", "bpm", "bpm_conf", "bpm_src", "musical_key", "key_conf", "loudness_db",
             "onset_rate", "vocal_frac", "vocal_ratio")


# ---------- сервер с медиатекой ----------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(1 << 20):
            h.update(b)
    return h.hexdigest()


def pending_rows(conn: sqlite3.Connection) -> list[tuple[bytes, int, str]]:
    """(MD5, cue_ms, путь файла) треков, которым чего-то не хватает."""
    return [(bytes(r[0]), r[1], r[2]) for r in conn.execute(
        """SELECT t.md5, t.cue_ms, min(p.path) FROM tracks t JOIN track_paths p ON p.track_id = t.id
           WHERE t.id NOT IN (SELECT track_id FROM maest) OR t.id NOT IN (SELECT track_id FROM clap_vectors)
              OR t.id NOT IN (SELECT track_id FROM track_features WHERE version = ?)
           GROUP BY t.id ORDER BY t.id""", (facets.VERSION,))]


def listing(rows: list[tuple[bytes, int, str]], skip: set[tuple[str, int]], limit: int = BATCH_TRACKS,
            max_bytes: int = BATCH_BYTES) -> dict:
    """Пачка: целые файлы, пока не наберётся limit треков или max_bytes. Образ с листом CUE — один файл на
    много треков, поэтому лист кладётся рядом: без него на помощнике образ станет одним длинным треком."""
    by_file: dict[str, list[dict]] = {}
    total = 0
    for md5, cue, path in rows:
        key = md5.hex()
        if (key, cue) in skip:
            continue
        total += 1
        by_file.setdefault(path, []).append({"md5": key, "cue_ms": cue})
    files, count, size, dirs = [], 0, 0, {}
    for path, tracks in by_file.items():
        try:
            n = os.path.getsize(path)
        except OSError:
            total -= len(tracks)
            continue
        if files and (count + len(tracks) > limit or size + n > max_bytes):
            break
        d = os.path.dirname(path)
        dirs[d] = hashlib.sha1(d.encode("utf-8", "surrogateescape")).hexdigest()[:12]
        files.append({"path": path, "name": os.path.basename(path), "size": n, "dir": dirs[d], "tracks": tracks})
        count += len(tracks)
        size += n
    cues = []
    for d, dir_id in dirs.items():
        try:
            names = os.listdir(d)
        except OSError:
            continue
        for name in names:
            if name.lower().endswith(".cue") and not name.startswith("._"):
                p = os.path.join(d, name)
                cues.append({"path": p, "name": name, "size": os.path.getsize(p), "dir": dir_id})
    return {"total": total, "files": files, "cues": cues}


def allowed_file(conn: sqlite3.Connection, cfg: Config, path: str) -> Path | None:
    """Только файлы медиатеки и листы CUE внутри её папок: токен помощника не читает диск сервера целиком."""
    p = Path(path)
    if conn.execute("SELECT 1 FROM files WHERE path = ?", (path,)).fetchone():
        return p if p.is_file() else None
    if p.suffix.lower() == ".cue" and p.is_file():
        real = p.resolve()
        if any(Path(r).resolve() in real.parents for r in cfg.libraries):
            return real
    return None


def _floats(b64, dim: int) -> bytes:
    raw = base64.b64decode(b64, validate=True)
    if len(raw) != dim * 4 or not np.isfinite(np.frombuffer(raw, np.float32)).all():
        raise ValueError("вектор не той длины или с NaN")
    return raw


def _feature_values(feat: dict) -> tuple:
    if feat.get("version") != facets.VERSION:
        raise ValueError("другая версия признаков")
    out = []
    for k in FEAT_COLS:
        v = feat.get(k)
        if k == "bpm_src":
            v = None if v is None else str(v)[:16]
        elif v is not None:
            v = float(v)
            if not math.isfinite(v):
                raise ValueError(f"{k}: не число")
        out.append(v)
    return tuple(out)


def save_results(conn: sqlite3.Connection, items: list[dict]) -> dict:
    """Числа с помощника. Только дополняет: посчитанное раньше не перезаписывается. Плохой трек не мешает
    остальным в той же отправке."""
    saved = rejected = unknown = 0
    with conn:
        for it in items:
            try:
                md5, cue = bytes.fromhex(it["md5"]), int(it["cue_ms"])
                row = conn.execute("SELECT id FROM tracks WHERE md5 = ? AND cue_ms = ?", (md5, cue)).fetchone()
                if row is None:
                    unknown += 1
                    continue
                tid = row[0]
                # проверка всего трека до записи: половина разбора от плохого трека в базу не попадает
                ins = []
                if m := it.get("maest"):
                    ins.append(("INSERT OR IGNORE INTO maest (track_id, activations, embedding, windows) VALUES (?,?,?,?)",
                                (tid, _floats(m["a"], MAEST_CLASSES), _floats(m["e"], MAEST_DIM), int(m["w"]))))
                if f := it.get("facets"):
                    if set(f) != set(FACETS):
                        raise ValueError("не все фасеты")
                    for name in FACETS:
                        ins.append(("INSERT OR IGNORE INTO facet_vectors (track_id, facet, model, vector) VALUES (?,?,?,?)",
                                    (tid, name, models.EFFNET.id, _floats(f[name], EFFNET_DIM))))
                if feat := it.get("feat"):
                    ins.append((f"INSERT OR IGNORE INTO track_features (track_id, {', '.join(FEAT_COLS)}) "
                                f"VALUES (?,{','.join('?' * len(FEAT_COLS))})", (tid, *_feature_values(feat))))
                if c := it.get("clap"):
                    ins.append(("INSERT OR IGNORE INTO clap_vectors (track_id, vector) VALUES (?,?)",
                                (tid, _floats(c, CLAP_DIM))))
            except (KeyError, ValueError, TypeError) as e:
                print(f"Разбор с другого компьютера: трек отклонён: {e}", flush=True)
                rejected += 1
                continue
            for sql, args in ins:
                conn.execute(sql, args)
            saved += 1
    return {"saved": saved, "rejected": rejected, "unknown": unknown}


def _scenario_text_path(cfg: Config) -> Path:
    return cfg.index_dir / "laion-clap-music" / "scenario_text.npz"


def needs(cfg: Config) -> dict:
    """Чего сервер не может получить сам: названия стилей MAEST и векторы описаний сценариев считает модель,
    которой на нём нет. Без них посчитанные числа лежали бы без дела."""
    text = _scenario_text_path(cfg)
    want_text = not text.exists()
    if not want_text:
        try:
            with np.load(text) as z:
                want_text = list(z["prompts"]) != scenarios._texts(scenarios.load_defs(cfg))[0]
        except Exception:
            want_text = True
    return {"labels": models.maest_labels_for(cfg.models_dir, cfg.index_dir) is None, "scenario_text": want_text}


def save_extras(cfg: Config, labels: list[str] | None, scenario_text: str | None) -> None:
    if labels is not None and len(labels) == MAEST_CLASSES and all(isinstance(x, str) for x in labels):
        p = models.maest_labels_path(cfg.index_dir)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(labels, ensure_ascii=False))
    if scenario_text is not None:
        raw = base64.b64decode(scenario_text, validate=True)
        p = _scenario_text_path(cfg)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".part")
        tmp.write_bytes(raw)
        try:
            with np.load(tmp, allow_pickle=False) as z:  # файл закрыт до замены: на Windows открытый не заменить
                if z["text"].shape[1] != CLAP_DIM or len(z["ids"]) != len(z["text"]):
                    raise ValueError("векторы описаний не той формы")
        except Exception:  # не архив векторов — старый файл остаётся на месте
            tmp.unlink(missing_ok=True)
            raise
        os.replace(tmp, p)


# ---------- связь с сервером ----------

class LinkDown(Exception):
    """Сервер не отвечает: другая сеть, выключен."""


class LinkError(Exception):
    def __init__(self, status: int, msg: str = ""):
        super().__init__(msg or str(status))
        self.status = status


class BadFile(Exception):
    """Файл пришёл не таким, каким его видит сервер."""


class Link:
    def __init__(self, addr: str, port: int, token: str | None = None):
        self.base, self.token = f"http://{addr}:{port}", token

    def _req(self, path: str, params: dict | None = None, body: dict | None = None, method: str | None = None):
        url = self.base + path + ("?" + urllib.parse.urlencode(params) if params else "")
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        data = None
        if body is not None:
            data = json.dumps(body, ensure_ascii=False).encode()
            headers["Content-Type"] = "application/json"
        return urllib.request.Request(url, data=data, headers=headers, method=method)

    def _open(self, req, timeout: float):
        try:
            return urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as e:
            raise LinkError(e.code, e.reason) from None
        except OSError as e:  # URLError, сброс, таймаут
            raise LinkDown(str(e)) from None

    def call(self, path: str, params: dict | None = None, body: dict | None = None, method: str | None = None,
             timeout: float = 60) -> dict:
        with self._open(self._req(path, params, body, method), timeout) as r:
            return json.loads(r.read() or b"{}")

    def download(self, path: str, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".part")
        h = hashlib.sha256()
        try:
            with self._open(self._req("/v1/assist/file", {"path": path}), 120) as r, open(tmp, "wb") as out:
                want = r.headers.get("X-Arx-Sha256")
                while b := r.read(1 << 20):
                    h.update(b)
                    out.write(b)
        except OSError as e:  # соединение оборвалось посреди файла
            tmp.unlink(missing_ok=True)
            raise LinkDown(str(e)) from None
        if not want or h.hexdigest() != want:
            tmp.unlink(missing_ok=True)
            raise BadFile(path)
        os.replace(tmp, dest)


def _safe_name(name: str) -> str:
    """На Windows в именах нельзя то, что можно на Linux. Лист и образ меняются одинаково и остаются парой."""
    return re.sub(r'[<>:"/\\|?*]', "_", name) if sys.platform == "win32" else name.replace("/", "_")


class Finder:
    """Компьютеры с Arx Nova в этой сети (mDNS, тем же именем службы, что слышит плеер)."""

    def __init__(self, own_name: str):
        self.own_name = own_name
        self._found: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._zc = None

    def start(self) -> None:
        if self._zc is not None:
            return
        try:
            from zeroconf import IPVersion, ServiceBrowser, ServiceStateChange, Zeroconf

            from arxnova.discovery import SERVICE

            def changed(zeroconf, service_type, name, state_change):
                if state_change is ServiceStateChange.Removed:
                    with self._lock:
                        self._found.pop(name, None)
                    return
                info = zeroconf.get_service_info(service_type, name, timeout=3000)
                addrs = info.parsed_addresses(IPVersion.V4Only) if info else []
                if not addrs:
                    return
                props = {k.decode(): (v or b"").decode(errors="replace") for k, v in info.properties.items()}
                shown = props.get("name") or name.split(".")[0]
                if shown == self.own_name:
                    return
                with self._lock:
                    self._found[name] = {"name": shown, "addr": addrs[0], "port": info.port,
                                         "version": props.get("version", "")}

            self._zc = Zeroconf()
            self._browser = ServiceBrowser(self._zc, SERVICE, handlers=[changed])
        except Exception as e:  # noqa: BLE001  без поиска адрес вводится вручную
            print(f"Поиск компьютеров в сети недоступен: {e}", flush=True)
            self._zc = False

    def found(self) -> list[dict]:
        with self._lock:
            return sorted(self._found.values(), key=lambda x: x["name"].lower())

    def addr_of(self, name: str) -> dict | None:
        return next((x for x in self.found() if x["name"] == name), None)


# ---------- компьютер-помощник ----------

def _same_path(p) -> str:
    return unicodedata.normalize("NFC", os.path.realpath(p))


def collect(work: Path, batch: dict, local: dict[str, Path]) -> tuple[list[dict], list[dict]]:
    """Числа из базы рабочей папки — по ключам сервера. Трек без полного разбора уходит в failed."""
    conn = db.connect(work / "db" / "arxnova.sqlite3")
    conn.row_factory = sqlite3.Row
    try:
        ids = {(_same_path(r[1]), r[2]): r[0] for r in conn.execute(
            "SELECT t.id, min(p.path), t.cue_ms FROM tracks t JOIN track_paths p ON p.track_id = t.id GROUP BY t.id")}
        b64 = lambda x: base64.b64encode(bytes(x)).decode()  # noqa: E731
        items, failed = [], []
        for f in batch["files"]:
            lp = local.get(f["path"])
            for t in f["tracks"]:
                tid = ids.get((_same_path(lp), t["cue_ms"])) if lp else None
                m = c = feat = None
                vec = {}
                if tid is not None:
                    m = conn.execute("SELECT activations, embedding, windows FROM maest WHERE track_id = ?", (tid,)).fetchone()
                    c = conn.execute("SELECT vector FROM clap_vectors WHERE track_id = ?", (tid,)).fetchone()
                    feat = conn.execute(f"SELECT {', '.join(FEAT_COLS)} FROM track_features WHERE track_id = ?", (tid,)).fetchone()
                    vec = {r[0]: r[1] for r in conn.execute("SELECT facet, vector FROM facet_vectors WHERE track_id = ?", (tid,))}
                item = {"md5": t["md5"], "cue_ms": t["cue_ms"]}
                if m:
                    item["maest"] = {"a": b64(m[0]), "e": b64(m[1]), "w": m[2]}
                if c:
                    item["clap"] = b64(c[0])
                if feat and set(vec) >= set(FACETS):
                    item["feat"] = {k: feat[k] for k in FEAT_COLS}
                    item["facets"] = {k: b64(vec[k]) for k in FACETS}
                if len(item) > 2:
                    items.append(item)
                if not (m and c and "feat" in item):
                    failed.append({"md5": t["md5"], "cue_ms": t["cue_ms"]})
        return items, failed
    finally:
        conn.close()


def _run_batch(cfg: Config, work: Path, on_line) -> None:
    """Отдельным процессом в окружении компонентов: torch и Demucs в сервере не живут, память GPU — заново."""
    env = analysis.deep_env(cfg)
    env["ARXNOVA_MODELS_DIR"] = str(cfg.models_dir)  # те же модели, а не вторая копия на гигабайты
    analysis._run(analysis._cmd(cfg) + ["--data-dir", str(work), "assist-batch"], env, on_line=on_line)


class Worker:
    """Включается в окне приложения, живёт в службе сервера: окно можно закрыть."""

    def __init__(self, cfg: Config, finder: Finder | None = None, link=Link, batch=_run_batch,
                 on_battery=platforms.on_battery):
        self.cfg, self.finder, self.link, self.batch, self.on_battery = cfg, finder, link, batch, on_battery
        self.path = cfg.data_dir / "assist.json"
        self.work = cfg.data_dir / "assist-work"
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self.settings: dict = {"on": False, "name": "", "addr": "", "port": config.DEFAULT_PORT, "token": None}
        try:
            self.settings.update(json.loads(self.path.read_text()))
        except (OSError, ValueError):
            pass
        self.state: dict = {"stage": "off"}
        self.done = 0
        self._ask_req: dict | None = None
        self._blocked = False  # отказали или запрос истёк: снова просит только сам человек, не по кругу
        self.blip_s = BLIP_S

    def _save(self) -> None:
        self.cfg.data_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.settings, ensure_ascii=False))
        os.chmod(tmp, 0o600)  # в файле токен
        os.replace(tmp, self.path)

    def _set(self, stage: str, **kw) -> None:
        with self._lock:
            self.state = {"stage": stage, "at": time.time(), **kw}

    def snapshot(self) -> dict:
        if self.finder:
            self.finder.start()  # окно открыто: список компьютеров нужен, даже пока выключено
        with self._lock:
            s = dict(self.settings)
            state = dict(self.state)
        return {"on": s["on"], "name": s["name"], "addr": s["addr"], "port": s["port"], "paired": bool(s["token"]),
                "found": self.finder.found() if self.finder else [], "state": state, "done": self.done}

    def configure(self, on: bool, name: str | None = None, addr: str | None = None, port: int | None = None,
                  retry: bool = False) -> None:
        with self._lock:
            s = self.settings
            s["on"] = on
            if addr is not None and (addr != s["addr"] or (port or s["port"]) != s["port"]) and name != s["name"]:
                s["token"] = None  # другой компьютер — другое разрешение
                self._ask_req = None
                self._blocked = False
            if addr is not None:
                s["addr"], s["port"] = addr, port or config.DEFAULT_PORT
            if name is not None:
                s["name"] = name
            if retry:
                self._ask_req = None
                self._blocked = False
            self._save()
        self._wake.set()

    def start(self) -> None:
        threading.Thread(target=self._loop, name="assist", daemon=True).start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                wait = self.tick()
            except LinkDown:
                self._set("offline", name=self.settings["name"])
                wait = OFFLINE_S
            except Exception as e:  # noqa: BLE001  разбор не должен ронять сервер
                print(f"Разбор на другом компьютере: {e}", flush=True)
                self._set("error", name=self.settings["name"], detail=str(e)[:200])
                wait = RETRY_S
            self._wake.wait(wait)
            self._wake.clear()

    def _reach(self, s: dict):
        """Сохранённый адрес, а если не отвечает — тот же компьютер по имени: адрес мог смениться."""
        try:
            link = self.link(s["addr"], s["port"], s["token"])
            return link, link.call("/v1/health", timeout=5)
        except LinkDown:
            hit = self.finder.addr_of(s["name"]) if self.finder and s["name"] else None
            if not hit or (hit["addr"] == s["addr"] and hit["port"] == s["port"]):
                raise
            with self._lock:
                s["addr"], s["port"] = hit["addr"], hit["port"]
                self._save()
            link = self.link(s["addr"], s["port"], s["token"])
            return link, link.call("/v1/health", timeout=5)

    def _ask(self, link) -> float:
        name = self.settings["name"]
        if self._blocked:
            self._set("denied", name=name)
            return 3600
        if self._ask_req is None:
            r = link.call("/v1/access/request", body={"name": f"{self.cfg.server_name}{NAME_MARK}разбор медиатеки",
                                                      "kind": auth.ROLE_ASSIST})
            self._ask_req = r
        r = self._ask_req
        try:
            st = link.call(f"/v1/access/request/{r['id']}", {"secret": r["secret"]})
        except LinkError as e:
            if e.status != 404:
                raise
            st = {"status": "denied"}  # истёк
        if st["status"] == "approved":
            with self._lock:
                self.settings["token"] = st["token"]
                self._save()
            self._ask_req = None
            return 0
        if st["status"] == "denied":
            self._ask_req = None
            self._blocked = True
            self._set("denied", name=name)
            return 3600
        self._set("asking", name=name, check=r.get("check"), confirm=r.get("confirm"), me=self.cfg.server_name)
        return 3

    def tick(self) -> float:
        """Один шаг; возвращает, сколько секунд можно спать."""
        s = self.settings
        if not s["on"]:
            self._set("off")
            return 3600
        if self.finder:
            self.finder.start()
        if not s["addr"]:
            self._set("choose")
            return 3600
        link, health = self._reach(s)
        name = health.get("server") or s["name"] or s["addr"]
        if name != s["name"]:
            with self._lock:
                s["name"] = name
                self._save()
        if not health.get("assist"):  # старый сервер не знает роли: дал бы гостя или полный доступ
            self._set("old_server", name=name)
            return RETRY_S
        if not s["token"]:
            return self._ask(link)
        try:
            batch = link.call("/v1/assist/pending", {"tracks": BATCH_TRACKS})
        except LinkError as e:
            if e.status == 401:  # отключили на том компьютере или в плеере
                with self._lock:
                    s["token"] = None
                    self._save()
                self._blocked = True
                self._set("denied", name=name)
                return 3600
            if e.status == 403:
                self._set("not_lan", name=name)
                return RETRY_S
            raise
        if not batch["files"]:
            self._set("idle", name=name, done=self.done)
            return IDLE_S
        left = batch["total"]
        if not analysis.installed(self.cfg):
            self._set("components", name=name, left=left)
            return OFFLINE_S
        if self.on_battery():
            self._set("battery", name=name, left=left)
            return OFFLINE_S
        if not analysis._lock.acquire(blocking=False):  # свой разбор или установка: GPU один
            self._set("busy", name=name, left=left)
            return OFFLINE_S
        try:
            return self._do_batch(link, batch, name)
        finally:
            analysis._lock.release()
            shutil.rmtree(self.work, ignore_errors=True)

    def _fetch(self, link, batch: dict, name: str) -> tuple[dict[str, Path], list[dict]]:
        """Сбой sha256 — повтор; так и не сошлось — треки этого файла в failed. Обрыв связи — наверх, заход отменён."""
        lib = self.work / "lib"
        all_files = batch["files"] + batch["cues"]
        local: dict[str, Path] = {}
        failed: list[dict] = []
        for i, f in enumerate(all_files):
            self._set("fetch", name=name, i=i + 1, n=len(all_files), left=batch["total"])
            dest = lib / f["dir"] / _safe_name(f["name"])
            bad, down = 0, 0
            while bad < 3:
                try:
                    link.download(f["path"], dest)
                    local[f["path"]] = dest
                    break
                except BadFile:
                    bad += 1
                    print(f"Файл не сошёлся по sha256 ({bad}/3): {f['path']}", flush=True)
                except LinkDown:
                    down += 1
                    if down >= FILE_RETRIES:
                        raise
                    self._stop.wait(self.blip_s)
                except LinkError as e:
                    print(f"Файл не отдан ({e.status}): {f['path']}", flush=True)
                    break
            if f["path"] not in local:
                failed += [{"md5": t["md5"], "cue_ms": t["cue_ms"]} for t in f.get("tracks", [])]
        return local, failed

    def _do_batch(self, link, batch: dict, name: str) -> float:
        shutil.rmtree(self.work, ignore_errors=True)
        lib = self.work / "lib"
        lib.mkdir(parents=True)
        local, failed = self._fetch(link, batch, name)
        n = sum(len(f["tracks"]) for f in batch["files"])
        left = batch["total"]
        config.Config(data_dir=self.work, libraries=[lib], server_name="assist", remote=False, keep_awake=False).save()
        if batch.get("scenarios_toml"):
            (self.work / "scenarios.toml").write_text(batch["scenarios_toml"], encoding="utf-8")
        self._set("work", name=name, n=n, left=left, detail="")

        def on_line(line: str) -> None:
            if line.startswith("  …"):
                self._set("work", name=name, n=n, left=left, detail=line.strip(" …"))

        error = None
        if local:
            try:
                self.batch(self.cfg, self.work, on_line)
            except Exception as e:  # noqa: BLE001  посчитанное до сбоя всё равно отправляем
                error = str(e)[:200]
        items, not_done = collect(self.work, batch, local) if (self.work / "db").exists() else ([], [])
        if error is None:
            failed += not_done  # сбой всего процесса — не вина треков: их не откладываем на сутки
        self._set("send", name=name, n=n, left=left)
        extras = {}
        if batch.get("need", {}).get("labels"):
            try:
                extras["labels"] = models.maest_labels(self.cfg.models_dir)
            except OSError:
                pass
        text = self.work / "index" / "laion-clap-music" / "scenario_text.npz"
        if batch.get("need", {}).get("scenario_text") and text.exists():
            extras["scenario_text"] = base64.b64encode(text.read_bytes()).decode()
        last = left <= n
        chunks = [items[i:i + CHUNK_ITEMS] for i in range(0, len(items), CHUNK_ITEMS)] or [[]]
        for i, chunk in enumerate(chunks):
            final = i == len(chunks) - 1
            body = {"items": chunk, "failed": failed if final else [], "last": last and final, **(extras if final else {})}
            for attempt in range(SEND_RETRIES):
                try:
                    r = link.call("/v1/assist/results", body=body, timeout=120)
                    break
                except LinkDown:  # посчитанное дороже ожидания: повтор безопасен, запись только дополняет
                    if attempt == SEND_RETRIES - 1:
                        raise
                    self._set("offline", name=name)
                    self._stop.wait(self.blip_s)
            self.done += r.get("saved", 0)
        if failed:
            print(f"Не разобрано на этом заходе треков: {len(failed)}", flush=True)
        if error:
            self._set("error", name=name, detail=error)
            return RETRY_S
        return 0
