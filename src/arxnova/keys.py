"""Ключи: файла (MD5 звука) и записи (одна песня в разных изданиях — один ключ, ФТ-2.10)."""

from __future__ import annotations

import hashlib
import re
import unicodedata

# Издания одной записи (CD, хайрез, ремастер) расходятся по длительности на доли секунды.
DURATION_TOLERANCE_S = 2.0

_SPACES = re.compile(r"\s+")
# «(Remastered 2011)», «[24-bit]», «- Remaster» и т. п. не делают запись другой
_EDITION = re.compile(
    r"\s*[\(\[][^\)\]]*(remaster|remastered|hi-?res|24[- ]?bit|deluxe|bonus)[^\)\]]*[\)\]]"
    r"|\s+-\s+.*remaster.*$",
    re.IGNORECASE,
)


def normalize(s: str | None) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s).casefold()
    s = _EDITION.sub("", s)
    return _SPACES.sub(" ", s).strip()


def recording_basis(*, mb_recording: str | None, artist: str | None, title: str | None) -> str:
    """По чему узнаётся запись: MusicBrainz recording ID, а без него — «исполнитель | название»."""
    if mb_recording:
        return "mb:" + mb_recording.strip().lower()
    return f"tag:{normalize(artist)}|{normalize(title)}"


def new_recording_key(basis: str, duration: float) -> bytes:
    """Ключ новой записи. Для тегов длительность входит в ключ, чтобы одноимённые песни
    разной длины (концертная версия) не склеились. Совпадение изданий с допуском
    DURATION_TOLERANCE_S решает match_recording — не округление: граница округления
    разрезала бы 301,4 с и 300,6 с одной записи на две."""
    if basis.startswith("mb:"):
        return hashlib.md5(basis.encode("utf-8")).digest()
    return hashlib.md5(f"{basis}|{duration:.1f}".encode("utf-8")).digest()


def match_recording(basis: str, duration: float, candidates) -> bytes | None:
    """Из уже известных (ключ, длительность) с той же основой — ключ ближайшего по
    длительности в пределах допуска."""
    if basis.startswith("mb:"):
        return new_recording_key(basis, duration)
    best = None
    for key, dur in candidates:
        diff = abs(dur - duration)
        if diff <= DURATION_TOLERANCE_S and (best is None or diff < best[0]):
            best = (diff, key)
    return best[1] if best else None


# Одна песня в разных изданиях (владелец: «Rewind» дважды за час с подписью «новое для вас»): у изданий разные теги и длина,
# поэтому recording_key их не склеивает: 625 песен, 1594 трека на медиатеке владельца (04.11.2026).
_EDITION_NOTE = re.compile(
    r"\s*[\(\[][^\)\]]*(remaster|deluxe|bonus|explicit|album version|album mix|original mix|original version|radio edit"
    r"|single version|single edit|\b(?:19|20)\d\d\b)[^\)\]]*[\)\]]", re.IGNORECASE)
_OTHER_TAKE = re.compile(r"remix|\bdub\b|\bvip\b|instrumental|acoustic|\blive\b|cappella|accapella|rework|bootleg", re.IGNORECASE)
_GENERIC_TITLE = re.compile(r"(?:i{1,3}|iv|v|vi{1,3}|ix|x|\d+|intro|outro|interlude|untitled|track \d+)", re.IGNORECASE)
_FIRST_CREDIT = re.compile(r"\s*(?:&|,|;|/|\bfeat\.?|\bft\.?|\bfeaturing\b|\bx\b|\bvs\.?)\s*", re.IGNORECASE)


def song_family(artist: str | None, title: str | None) -> tuple[str, str] | None:
    """Первый исполнитель и название без пометок издания; ремикс, концерт и инструментал — другая песня.
    None — название слишком общее, чтобы склеивать («Intro», «II», «90»)."""
    if not artist or not title:
        return None

    def strip(m: re.Match) -> str:
        return m.group(0) if _OTHER_TAKE.search(m.group(0)) else ""

    name = _EDITION_NOTE.sub(strip, unicodedata.normalize("NFKC", title).replace("’", "'"))
    name = _SPACES.sub(" ", re.sub(r"[^\w ]+", " ", name.casefold())).strip()
    if len(name) < 4 or _GENERIC_TITLE.fullmatch(name):
        return None
    first = _FIRST_CREDIT.split(unicodedata.normalize("NFKC", artist).casefold(), maxsplit=1)[0]
    first = _SPACES.sub(" ", re.sub(r"[^\w ]+", "", first)).strip()
    return (first, name) if first else None
