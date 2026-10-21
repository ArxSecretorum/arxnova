"""CUE-листы: альбом одним FLAC-файлом («образ») + разметка треков (ТЗ, ФТ-2.1: ключ трека —
MD5 звука + смещение CUE). Лист — рядом с файлом или внутри FLAC (тег CUESHEET).
Кодировки: UTF-8 (с BOM и без), иначе Windows-1251 — так их пишут русские рипы."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

FRAMES_PER_SECOND = 75


@dataclass
class CueTrack:
    number: int
    start_ms: int
    title: str | None = None
    performer: str | None = None


@dataclass
class CueSheet:
    title: str | None = None
    performer: str | None = None
    date: str | None = None
    genre: str | None = None
    files: dict[str, list[CueTrack]] = field(default_factory=dict)  # FILE → треки


def decode(raw: bytes) -> str:
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw[3:].decode("utf-8", errors="replace")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("cp1251", errors="replace")


_QUOTED = re.compile(r'^\s*(\w+)\s+(?:"(.*)"|(\S.*?))\s*$')
_INDEX = re.compile(r"^\s*INDEX\s+(\d+)\s+(\d+):(\d+):(\d+)\s*$", re.IGNORECASE)
_TRACK = re.compile(r"^\s*TRACK\s+(\d+)\s+(\w+)", re.IGNORECASE)
_FILE = re.compile(r'^\s*FILE\s+(?:"(.+)"|(\S+))\s+\w+\s*$', re.IGNORECASE)
_REM = re.compile(r'^\s*REM\s+(\w+)\s+(?:"(.*)"|(.*?))\s*$', re.IGNORECASE)


def parse(text: str) -> CueSheet:
    sheet = CueSheet()
    cur_file: str | None = None
    cur: CueTrack | None = None
    for line in text.splitlines():
        if m := _FILE.match(line):
            cur_file = (m.group(1) or m.group(2)).strip()
            sheet.files.setdefault(cur_file, [])
            cur = None
            continue
        if m := _TRACK.match(line):
            if cur_file is None:
                continue
            cur = CueTrack(number=int(m.group(1)), start_ms=-1)
            sheet.files[cur_file].append(cur)
            continue
        if m := _INDEX.match(line):
            if cur is not None and int(m.group(1)) == 1:
                mm, ss, ff = int(m.group(2)), int(m.group(3)), int(m.group(4))
                cur.start_ms = (mm * 60 + ss) * 1000 + ff * 1000 // FRAMES_PER_SECOND
            continue
        if m := _REM.match(line):
            key, val = m.group(1).upper(), (m.group(2) if m.group(2) is not None else m.group(3))
            if key == "DATE":
                sheet.date = val
            elif key == "GENRE":
                sheet.genre = val
            continue
        if m := _QUOTED.match(line):
            key, val = m.group(1).upper(), (m.group(2) if m.group(2) is not None else m.group(3))
            if key in ("TITLE", "PERFORMER"):
                target = cur if cur is not None else sheet
                setattr(target, key.lower(), val)
    for tracks in sheet.files.values():
        tracks[:] = [t for t in tracks if t.start_ms >= 0]
        tracks.sort(key=lambda t: t.start_ms)
    return sheet


def tracks_for(sheet: CueSheet, flac_name: str) -> list[CueTrack]:
    """Треки листа, относящиеся к этому файлу. Имя в FILE часто расходится с реальным
    (другое расширение — .wav у исходного рипа, регистр): сравниваем по имени без расширения,
    а если файл в листе один — берём его."""
    if not sheet.files:
        return []
    stem = Path(flac_name).stem.casefold()
    for name, tracks in sheet.files.items():
        if Path(name).stem.casefold() == stem or name.casefold() == flac_name.casefold():
            return tracks
    if len(sheet.files) == 1:
        return next(iter(sheet.files.values()))
    return []


def find_for(flac_path: Path, embedded: str | None = None) -> tuple[CueSheet, list[CueTrack], str] | None:
    """Лист для файла: сначала рядом лежащий .cue, потом встроенный тег CUESHEET.
    Возвращает (лист, треки этого файла, подпись источника для повторного обхода)."""
    candidates = [flac_path.with_suffix(".cue"), flac_path.with_suffix(flac_path.suffix + ".cue")]
    try:
        candidates += sorted(p for p in flac_path.parent.glob("*.cue") if p not in candidates)
    except OSError:
        pass
    for c in candidates:
        if not c.is_file():
            continue
        try:
            sheet = parse(decode(c.read_bytes()))
        except OSError:
            continue
        tracks = tracks_for(sheet, flac_path.name)
        if len(tracks) > 1:
            st = c.stat()
            return sheet, tracks, f"{c.name}:{st.st_size}:{st.st_mtime_ns}"
    if embedded:
        sheet = parse(embedded)
        tracks = tracks_for(sheet, flac_path.name) or (next(iter(sheet.files.values())) if sheet.files else [])
        if len(tracks) > 1:
            return sheet, tracks, "embedded"
    return None
