"""Выбор папки с музыкой с телефона (решение владельца 23.10.2026): у сервера без экрана (Pi,
NAS) нет проводника — его заменяет плеер. Сервер отдаёт только имена папок и сколько в них
музыки; файлы и их содержимое наружу не уходят, и только владельцу.

Корни — то, где обычно лежит музыка: домашняя папка и подключённые диски (macOS — /Volumes,
Linux — /media, /mnt, /run/media, Windows — буквы дисков)."""

from __future__ import annotations

import os
import string
import sys
import time
from pathlib import Path

AUDIO = {".flac", ".mp3", ".m4a", ".aac", ".ogg", ".opus", ".wav", ".aiff", ".aif", ".alac", ".wv", ".ape", ".dsf", ".dff"}
# системное и служебное — в списке не показываем
SKIP = {n.lower() for n in (
    "proc", "sys", "dev", "run", "boot", "etc", "bin", "sbin", "lib", "lib64", "usr", "var", "tmp", "snap",
    "lost+found", "System", "Library", "Applications", "Windows", "Program Files", "Program Files (x86)",
    "ProgramData", "System Volume Information", "AppData", "node_modules", "found.000")}


def _hidden(name: str) -> bool:
    # .Trashes, .arxnova — скрытые; $RECYCLE.BIN, $Extend — служебные Windows (и на дисках с Mac/Linux)
    return name.startswith((".", "$")) or name.lower() in SKIP


def roots() -> list[dict]:
    """Откуда начинать обзор: домашняя папка и подключённые диски."""
    out: list[tuple[str, Path]] = [("home", Path.home())]  # название переводит плеер (kind = home)
    if sys.platform == "win32":
        for letter in string.ascii_uppercase:
            p = Path(f"{letter}:\\")
            if p.exists():
                out.append((f"{letter}:", p))
    else:
        bases = [Path("/Volumes")] if sys.platform == "darwin" else \
            [Path("/media"), Path("/mnt"), Path("/run/media"), Path("/srv")]
        for base in bases:
            # /Volumes/Macintosh HD — ссылка на системный диск, а не отдельный диск
            for p in (q for q in _subdirs(base) if not q.is_symlink()):
                # /media/<пользователь>/<диск> и /run/media/<пользователь>/<диск>
                inner = _subdirs(p) if base.name in ("media", "run") and p.name == Path.home().name else []
                out += [(q.name, q) for q in inner] or [(p.name, p)]
    seen, res = set(), []
    for title, p in out:
        if str(p) not in seen and p.is_dir():
            seen.add(str(p))
            home = title == "home" and p == Path.home()
            res.append({"title": p.name if home else title, "path": str(p), "kind": "home" if home else "disk"})
    return res


def _subdirs(p: Path) -> list[Path]:
    try:
        return sorted((q for q in p.iterdir() if q.is_dir() and not _hidden(q.name)),
                      key=lambda q: q.name.lower())
    except OSError:
        return []


def _audio_here(p: Path) -> int:
    try:
        return sum(1 for e in os.scandir(p) if e.is_file() and os.path.splitext(e.name)[1].lower() in AUDIO)
    except OSError:
        return 0


def listing(path: str) -> dict:
    """Папки внутри path: имя, путь, есть ли внутри папки, сколько музыки прямо в ней."""
    p = Path(path).expanduser()
    if not p.is_dir():
        raise FileNotFoundError(path)
    dirs = []
    for q in _subdirs(p):
        dirs.append({"name": q.name, "path": str(q), "audio": _audio_here(q), "has_dirs": bool(_subdirs(q))})
    parent = str(p.parent) if p.parent != p else None
    return {"path": str(p), "parent": parent, "audio": _audio_here(p), "dirs": dirs}


def suggest(budget_s: float = 4.0, max_depth: int = 5) -> list[dict]:
    """Где, похоже, лежит медиатека: папки с наибольшим числом музыкальных файлов. Берём
    «собирающую» папку (Music, а не одну из папок исполнителей в ней): ни одна вложенная не
    держит 90 % её музыки. Обход ограничен по времени — большой диск не держит ответ."""
    deadline = time.monotonic() + budget_s
    counts: dict[str, int] = {}
    children: dict[str, list[str]] = {}

    def walk(p: Path, depth: int) -> int:
        if time.monotonic() > deadline:
            return 0
        n = _audio_here(p)
        kids = []
        if depth < max_depth:
            for q in _subdirs(p):
                k = walk(q, depth + 1)
                if k:
                    kids.append(str(q))
                    counts[str(q)] = k
                n += k
        children[str(p)] = kids
        return n

    for r in roots():
        counts[r["path"]] = walk(Path(r["path"]), 0)
    root_paths = {r["path"] for r in roots()}
    picks = []
    for path, n in counts.items():
        if n < 20:
            continue
        if any(counts.get(k, 0) >= 0.9 * n for k in children.get(path, [])):
            continue  # почти вся музыка — в одной вложенной папке: предлагаем её
        if path == str(Path.home()) and path in root_paths:
            continue  # вся домашняя папка — не медиатека
        picks.append({"path": path, "name": Path(path).name or path, "audio": n})
    picks.sort(key=lambda x: -x["audio"])
    # вложенные друг в друга — оставляем верхнюю
    out: list[dict] = []
    for x in picks:
        if not any(x["path"].startswith(y["path"].rstrip(os.sep) + os.sep) for y in out):
            out.append(x)
    return out[:5]
