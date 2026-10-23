"""Глубокий разбор медиатеки из приложения сервера (ФТ-2.3, 2.4, 4.10): стили, ритм, вокал,
темп, тональность и сценарии.

Для него нужны torch, Demucs и CLAP — около 3 ГБ; в установщик сервера они не входят (ОП-5).
Компоненты ставятся один раз, с согласия, в папку данных сервера: uv (одиночный файл от Astral,
со сверкой sha256) создаёт окружение `arxnova[stems]` той же версии, что и сервер; веса CLAP
(LAION, CC0) — со сверкой sha256. Разбор идёт отдельными процессами — сервер не тяжелеет и не
падает, если разбор упал. Новые треки разбираются сами (см. cli.rescan_loop)."""

from __future__ import annotations

import hashlib
import io
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import urllib.request
import zipfile
from pathlib import Path

from arxnova import __version__
from arxnova.config import Config

SIZE_HINT_GB = 3
RETRY_S = 600  # упавший разбор — снова через 10 минут
FACETS_CHUNK = 250  # треков на один процесс разбора
RELEASES = "https://github.com/ArxSecretorum/arxnova/releases/download"
UV_RELEASES = "https://github.com/astral-sh/uv/releases/latest/download"

# что сейчас происходит — для окна приложения (/v1/local/analysis)
state: dict = {"busy": False, "stage": "", "detail": "", "error": "", "at": 0.0}
_lock = threading.Lock()
wake = threading.Event()  # кнопка «Проанализировать сейчас» — обход медиатеки без ожидания


def _set(**kw) -> None:
    state.update(kw, at=time.time())


def root(cfg: Config) -> Path:
    return cfg.data_dir / "analysis"


def _env_exe(cfg: Config) -> Path:
    env = root(cfg) / "env"
    return env / ("Scripts/arxnova.exe" if sys.platform == "win32" else "bin/arxnova")


def _uv(cfg: Config) -> str | None:
    own = root(cfg) / "bin" / ("uv.exe" if sys.platform == "win32" else "uv")
    if own.exists():
        return str(own)
    return shutil.which("uv")


def _dev_mode() -> bool:
    """Запуск из исходников, где [stems] уже стоит: компоненты не нужны."""
    if getattr(sys, "frozen", False):
        return False
    try:
        import demucs  # noqa: F401

        return True
    except ImportError:
        return False


def installed(cfg: Config) -> bool:
    if _dev_mode():
        return _uv(cfg) is not None
    marker = root(cfg) / "installed.json"
    if not (marker.exists() and _env_exe(cfg).exists() and _uv(cfg)):
        return False
    return json.loads(marker.read_text()).get("version") == __version__  # обновили сервер — обновить и разбор


def _cmd(cfg: Config) -> list[str]:
    if _dev_mode():
        return [sys.executable, "-c", "import arxnova; arxnova.main()"]
    return [str(_env_exe(cfg))]


def _uv_asset() -> str:
    m = platform.machine().lower()
    arch = "aarch64" if m in ("arm64", "aarch64") else "x86_64"
    if sys.platform == "darwin":
        return f"uv-{arch}-apple-darwin.tar.gz"
    if sys.platform == "win32":
        return f"uv-{arch}-pc-windows-msvc.zip"
    return f"uv-{arch}-unknown-linux-gnu.tar.gz"


def _download(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": f"arxnova/{__version__}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def _install_uv(cfg: Config) -> str:
    asset = _uv_asset()
    _set(stage="Скачиваю установщик пакетов uv")
    blob = _download(f"{UV_RELEASES}/{asset}")
    want = _download(f"{UV_RELEASES}/{asset}.sha256").split()[0].decode()
    if hashlib.sha256(blob).hexdigest() != want:
        raise RuntimeError("контрольная сумма uv не совпала")
    dest = root(cfg) / "bin"
    dest.mkdir(parents=True, exist_ok=True)
    name = "uv.exe" if sys.platform == "win32" else "uv"
    if asset.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            member = next(n for n in z.namelist() if n.endswith("/" + name) or n == name)
            (dest / name).write_bytes(z.read(member))
    else:
        with tarfile.open(fileobj=io.BytesIO(blob)) as t:
            member = next(m for m in t.getmembers() if m.name.endswith("/" + name))
            (dest / name).write_bytes(t.extractfile(member).read())
        os.chmod(dest / name, 0o755)
    return str(dest / name)


def _run(cmd: list[str], env: dict | None = None, on_line=None) -> None:
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    tail = []
    for line in p.stdout:
        line = line.rstrip()
        if line:
            tail = (tail + [line])[-5:]
            if on_line:
                on_line(line)
    if p.wait() != 0:
        raise RuntimeError(tail[-1] if tail else f"код {p.returncode}")


def install(cfg: Config) -> None:
    """Компоненты анализа (~3 ГБ): uv, окружение arxnova[stems] этой версии, веса CLAP."""
    if not _lock.acquire(blocking=False):
        return
    try:
        _set(busy=True, error="", stage="Готовлю компоненты анализа", detail="")
        uv = _uv(cfg) or _install_uv(cfg)
        if not _dev_mode():
            env = root(cfg) / "env"
            _set(stage="Создаю окружение анализа")
            _run([uv, "venv", "--python", "3.12", "--allow-existing", str(env)])
            spec = os.environ.get("ARXNOVA_ANALYSIS_SPEC") or \
                f"arxnova[stems] @ {RELEASES}/v{__version__}/arxnova-{__version__}-py3-none-any.whl"
            _set(stage="Скачиваю torch, Demucs и модели (это надолго)")
            _run([uv, "pip", "install", "--python", str(env / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")),
                  spec], on_line=lambda l: _set(detail=l[-120:]))
        from arxnova import scenarios

        _set(stage="Скачиваю веса CLAP для сценариев (2,3 ГБ)", detail="")
        scenarios.ensure_ckpt(cfg, progress=lambda m: _set(detail=m[-120:]))
        root(cfg).mkdir(parents=True, exist_ok=True)
        (root(cfg) / "installed.json").write_text(json.dumps({"version": __version__, "at": int(time.time())}))
        _set(stage="Компоненты анализа установлены", detail="")
    except Exception as e:
        _set(error=f"Не удалось установить компоненты анализа: {e}")
    finally:
        _set(busy=False)
        _lock.release()


def pending(conn) -> dict:
    """Сколько треков ещё не разобрано по каждому этапу."""
    from arxnova import facets

    q = lambda sql, *a: conn.execute(sql, a).fetchone()[0]  # noqa: E731
    total = q("SELECT count(*) FROM tracks")
    return {
        "tracks": total,
        "mix": q("SELECT count(DISTINCT track_id) FROM embeddings"),
        "facets": q("SELECT count(*) FROM track_features WHERE version = ?", facets.VERSION),
        "clap": q("SELECT count(*) FROM clap_vectors"),
        "tempo": q("SELECT count(*) FROM track_features WHERE bpm_src = 'autocorr'"),
        "maest": q("SELECT count(*) FROM maest"),
    }


def consented(cfg: Config) -> bool:
    """Владелец уже соглашался на компоненты (стоят от прошлой версии сервера)."""
    return (root(cfg) / "installed.json").exists()


def run_deep(cfg: Config, conn) -> bool:
    """Полный разбор того, что ещё не разобрано: фасеты, затем CLAP и сценарии. True — что-то
    сделано. Без компонентов — ничего (окно предложит их скачать)."""
    if not installed(cfg):
        if not consented(cfg):
            return False
        install(cfg)  # обновили сервер — компоненты под новую версию, без повторного вопроса
        if not installed(cfg):
            return False
    p = pending(conn)
    from arxnova import facets

    tempo = p["tempo"] > 0 and facets.beats_worth_trying(cfg.models_dir)
    if p["facets"] >= p["tracks"] and p["clap"] >= p["tracks"] and p["maest"] >= p["tracks"] and not tempo:
        return False
    if not _lock.acquire(blocking=False):
        return False
    from arxnova import sidecar

    try:
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"  # счётчик «… фасеты N/M» доходит до окна сразу, а не пачкой в конце
        uv = _uv(cfg)
        if uv:  # процесс CLAP запускается через uv — он должен быть в PATH
            env["PATH"] = str(Path(uv).parent) + os.pathsep + env.get("PATH", "")
        base = _cmd(cfg) + ["--data-dir", str(cfg.data_dir)]
        _set(busy=True, error="")
        if p["facets"] < p["tracks"] or tempo:
            _set(stage="Разбираю звук: стили, ритм, вокал, темп", detail="")
            # кусками по FACETS_CHUNK треков, каждый — новым процессом: что бы ни копилось в памяти
            # torch/Metal за проход, оно не дорастает до десятков гигабайт (39 ГБ за 3000 треков)
            while True:
                before = pending(conn)["facets"]
                _run(base + ["facets", "--limit", str(FACETS_CHUNK)], env,
                     on_line=lambda l: l.startswith("  …") and _set(detail=l.strip(" …")))
                sidecar.export(conn, cfg.libraries, sidecar.index_files(cfg))  # разбор — сразу и рядом с медиатекой (см. sidecar)
                now = pending(conn)
                if now["facets"] >= now["tracks"] or now["facets"] <= before:
                    break
        # стили MAEST-519 — после фасетов, на GPU (~1 с на трек; на процессоре ~4 с)
        if pending(conn)["maest"] < p["tracks"]:
            _set(stage="Определяю стили (MAEST, 519 стилей)", detail="")
            _run(base + ["maest"], env, on_line=lambda l: l.startswith("  …") and _set(detail=l.strip(" …")))
            sidecar.export(conn, cfg.libraries, sidecar.index_files(cfg))
        if p["clap"] < p["tracks"]:
            _set(stage="Сценарии: сравниваю звук с описаниями", detail="")
        # сценарии пересчитываются и после новых фасетов: от них зависят фильтры сценариев
        _run(base + ["scenarios"], env, on_line=lambda l: l.startswith("  …") and _set(detail=l.strip(" …")))
        sidecar.export(conn, cfg.libraries, sidecar.index_files(cfg))
        _set(stage="Медиатека разобрана", detail="")
        return True
    except Exception as e:
        _set(error=f"Разбор не удался: {e}")
        # готовое сохранено по трекам; без повтора разбор стоял бы до следующего изменения в папке
        t = threading.Timer(RETRY_S, wake.set)
        t.daemon = True
        t.start()
        return False
    finally:
        _set(busy=False)
        _lock.release()
