"""Таблица «кто слушает вместе» (решение 0015): вектор исполнителя по дампу ListenBrainz (CC0, ноябрь 2026). Работает без сети.

Близкие векторы — у исполнителей, которых слушают одни и те же люди: Calibre рядом с Commix и Lenzman, Wardruna — с Heilung.
Это свойство самих исполнителей, а не слушателей сервера: таблица одинаково годится любой медиатеке.

Строка таблицы ищется по хэшу имени после той же свёртки, что при сборке (bench/eval_goal/lb_norm.py, тест сверяет их).
Имя «A, B» или «A & B» ищется целиком, а если такого нет — по участникам: при сборке редкие сочетания раскладывались на них.
Таблица собирается bench/eval_goal/lb_table.py; векторы единичные, в 8 битах (× 127)."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent / "data" / "listeners"

_SPECIAL = str.maketrans({"ø": "o", "Ø": "o", "æ": "ae", "Æ": "ae", "œ": "oe", "Œ": "oe", "đ": "d", "Đ": "d",
                          "ł": "l", "Ł": "l", "ð": "d", "Ð": "d", "þ": "th", "Þ": "th", "ı": "i",
                          "ё": "е", "Ё": "е", "й": "и", "Й": "и"})
_NOT_WORD = re.compile(r"[\W_]+", re.U)
_FEAT = re.compile(r"\s*[\(\[]\s*(?:feat\.?|ft\.?|featuring)\b.*$|\s+(?:feat\.?|ft\.?|featuring)\s+.*$", re.I)
_SPLIT = re.compile(r"\s*(?:,|;|/|\s&\s|\s\+\s|\sand\s|\sx\s|\s×\s|\svs\.?\s|\sversus\s|\smeets\s)\s*", re.I)
# Служебные имена — не исполнители. В сводке дампа они просачивались участником строки «X, Various Artists» и получили
# бессмысленный вектор сборников (06.11.2026); lb_table.py их выбрасывает, здесь они не ищутся ни целиком, ни частью.
JUNK = frozenset({"", "variousartists", "various", "va", "unknown", "unknownartist", "noartist", "nonartist", "artist", "none",
                  "null", "undefined", "untitled", "track"})


def fold(s: str) -> str:
    s = unicodedata.normalize("NFKD", s.translate(_SPECIAL))
    return "".join(c for c in s if not unicodedata.combining(c)).casefold()


def key(s: str | None) -> str:
    """Буквы и цифры подряд: «S.P.Y» = «spy», «Nu:Tone» = «Nu Tone», «The Prodigy» = «Prodigy»."""
    s = fold(s or "").strip()
    if s.startswith("the "):
        s = s[4:]
    s = s.replace("&", " and ")
    k = _NOT_WORD.sub("", s)
    return k or s.strip()


def artist_whole(raw: str | None) -> str:
    return _FEAT.sub("", raw or "").strip()


def artist_key(raw: str | None) -> str:
    return key(artist_whole(raw))


def artist_parts(raw: str | None) -> list[str]:
    """Все участники: «A, B & C feat. D» → a, b, c. Часть «feat.» отбрасывается: это гость, а не автор."""
    return [k for k in (key(p) for p in _SPLIT.split(artist_whole(raw))) if k]


def h64(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode("utf8", "surrogatepass"), digest_size=8).digest(), "little")


class Table:
    def __init__(self, keys: np.ndarray, vecs: np.ndarray, scale: float = 127.0):
        self.keys, self.vecs, self.scale = keys, vecs, float(scale)

    @classmethod
    def load(cls, folder: Path = DATA) -> "Table | None":
        """None — таблицы нет (сборка без неё): поток работает как раньше."""
        try:
            keys = np.load(folder / "keys.npy", mmap_mode="r")
            vecs = np.load(folder / "vecs.npy", mmap_mode="r")
            scale = json.loads((folder / "meta.json").read_text()).get("scale", 127)
        except (OSError, ValueError):
            return None
        return cls(keys, vecs, scale) if len(keys) == len(vecs) and len(keys) else None

    def _row(self, h: int) -> int | None:
        i = int(np.searchsorted(self.keys, np.uint64(h)))
        return i if i < len(self.keys) and int(self.keys[i]) == h else None

    def lookup(self, raw: str | None) -> tuple[np.ndarray | None, frozenset[int]]:
        """Единичный вектор исполнителя и строки таблицы, из которых он собран; (None, ∅), если исполнителя нет."""
        if not raw or not raw.strip():
            return None, frozenset()
        k = artist_key(raw)
        whole = self._row(h64(k)) if k not in JUNK else None
        rows = [whole] if whole is not None else sorted({r for r in (self._row(h64(p)) for p in artist_parts(raw) if p not in JUNK)
                                                         if r is not None})
        if not rows:
            return None, frozenset()
        v = np.asarray(self.vecs[rows], np.float32).mean(axis=0) / self.scale
        n = float(np.linalg.norm(v))
        return (v / n, frozenset(rows)) if n > 1e-6 else (None, frozenset())

    def vector(self, raw: str | None) -> np.ndarray | None:
        """Единичный вектор исполнителя или None, если его нет в таблице."""
        return self.lookup(raw)[0]

    def library(self, names: list[tuple[str | None, ...]]) -> tuple[np.ndarray, np.ndarray, list[frozenset[int]]]:
        """Векторы строк медиатеки: для каждой строки — первое найденное из имён (исполнитель, затем исполнитель альбома).
        Третье — строки таблицы каждого трека: «тот же исполнитель» — общая строка, а не похожее написание имени
        (Esbjörn и Esbjorn Svensson Trio — одна строка; у «X feat. Y» вектор только от X)."""
        out = np.zeros((len(names), self.vecs.shape[1]), np.float32)
        ok = np.zeros(len(names), bool)
        rows: list[frozenset[int]] = [frozenset()] * len(names)
        cache: dict[str, tuple[np.ndarray | None, frozenset[int]]] = {}
        for i, options in enumerate(names):
            for raw in options:
                if raw is None:
                    continue
                if raw not in cache:
                    cache[raw] = self.lookup(raw)
                v, r = cache[raw]
                if v is not None:
                    out[i], ok[i], rows[i] = v, True, r
                    break
        return out, ok, rows
