"""Полки, которых нет среди 400 стилей Discogs-EffNet, — выводятся правилом из того, что модель уже
посчитала (решение владельца 23.10.2026: «хочу слушать ликвид»).

Liquid Funk у модели целиком в «Drum n Bass». Полка — треки DnB, которые:
  1) на бите DnB: темп 160–185 по долям (Izzamuzzic, даунтемпо и прочее «мягкое» отсекаются);
  2) без «жёсткости»: модель почти не видит в них Hardstyle / Gabber / Hard Techno / Hardcore —
     самый сильный признак (у тяжёлого DnB эта примесь заметна);
  3) с джазово-соуловой окраской (Soul-Jazz, Future Jazz, Broken Beat, Fusion, Downtempo…).
Место на полке — по «окраска × 0,5 − жёсткость», на полку — верхние 40 % DnB.

Проверено на медиатеке владельца по исполнителям: тяжёлое почти не попадает (Mistabishi 0/35,
Voltage 2/26, Camo & Krooked 2/120), классический ликвид — больше половины (Calibre ~2/3).
Ясность тональности и доля вокала ликвид не отличают (проверено — 50/50). Слабое место — яркий
мелодичный ликвид с громким сведением (Hybrid Minds, T & Sugah): для модели он звучит как
танцпол. Внутри полки подбор всё равно идёт по вкусу слушателя (дослушал / пропустил / лайк)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Shelf:
    name: str
    genre: str
    inside: tuple[str, ...]          # стили модели, внутри которых полка
    bpm: tuple[float, float]
    soft: tuple[str, ...]            # окраска «за»
    hard: tuple[str, ...]            # окраска «против»
    soft_weight: float
    share: float                     # доля стиля на полке


LIQUID = Shelf(
    name="Liquid Funk", genre="Electronic", inside=("Drum n Bass",), bpm=(160, 185),
    soft=("Soul-Jazz", "Downtempo", "Broken Beat", "Fusion", "Future Jazz", "Cool Jazz", "UK Street Soul",
          "Acid Jazz", "Trip Hop", "Jazzdance"),
    hard=("Jumpstyle", "Hardstyle", "Gabber", "Hard Techno", "Hardcore", "Schranz", "Hard Trance", "Speedcore",
          "Happy Hardcore", "Hands Up"),
    soft_weight=0.5, share=0.4,
)
SHELVES = (LIQUID,)


def members(shelf: Shelf, ids: np.ndarray, scores: np.ndarray, labels: list[str],
            style_of: dict[int, str], bpm_of: dict[int, float | None]) -> set[int]:
    """Треки полки. scores — активации стилей (строки — ids), labels — «Жанр---Стиль». Трек, у
    которого темп ещё не посчитан, проходит по стилю (полка не пустеет, пока идёт разбор)."""
    short = [lab.split("---")[-1] for lab in labels]
    soft = [short.index(s) for s in shelf.soft if s in short]
    hard = [short.index(s) for s in shelf.hard if s in short]
    if not soft or not hard:
        return set()
    logs = np.log(scores + 1e-4)
    z = lambda v: (v - v.mean()) / (v.std() + 1e-9)  # noqa: E731
    score = shelf.soft_weight * z(logs[:, soft].mean(1)) - z(logs[:, hard].mean(1))
    lo, hi = shelf.bpm
    cand = np.array([
        style_of.get(int(t)) in shelf.inside
        and (bpm_of.get(int(t)) is None or lo <= bpm_of[int(t)] <= hi)
        for t in ids])
    if not cand.any():
        return set()
    cut = np.quantile(score[cand], 1 - shelf.share)
    return {int(t) for t, ok, s in zip(ids, cand, score) if ok and s >= cut}
