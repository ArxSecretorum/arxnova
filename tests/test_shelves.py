"""Полка Liquid Funk правилом: бит DnB + нет жёсткости + джазово-соуловая окраска."""

import numpy as np

from arxnova import shelves


def test_liquid_shelf_rule():
    labels = [f"Electronic---{s}" for s in ("Drum n Bass", "Hardcore", "Hardstyle", "Soul-Jazz", "Future Jazz", "Downtempo")]
    sh = shelves.Shelf("Liquid Funk", "Electronic", ("Drum n Bass",), (160, 185),
                       soft=("Soul-Jazz", "Future Jazz", "Downtempo"), hard=("Hardcore", "Hardstyle"),
                       soft_weight=0.5, share=0.5)
    #             DnB   Hardc  Hards  SoulJ  FutJ   Downt
    rows = {1: [0.6, 0.001, 0.001, 0.05, 0.05, 0.05],   # джазовый, не жёсткий — на полку
            2: [0.6, 0.002, 0.002, 0.03, 0.03, 0.03],   # мягкий DnB — на полку
            3: [0.5, 0.10, 0.10, 0.001, 0.001, 0.001],  # жёсткий — нет
            4: [0.5, 0.08, 0.05, 0.002, 0.002, 0.002],  # жёсткий — нет
            5: [0.1, 0.001, 0.001, 0.2, 0.2, 0.2],      # мягкий, но не DnB (даунтемпо) — нет
            6: [0.6, 0.001, 0.001, 0.06, 0.06, 0.06]}   # мягкий DnB, но темп 120 — нет
    ids = np.array(list(rows))
    scores = np.array([rows[i] for i in rows], np.float32)
    style = {1: "Drum n Bass", 2: "Drum n Bass", 3: "Drum n Bass", 4: "Drum n Bass", 5: "Downtempo", 6: "Drum n Bass"}
    bpm = {1: 174.0, 2: None, 3: 175.0, 4: 172.0, 5: 85.0, 6: 120.0}
    assert shelves.members(sh, ids, scores, labels, style, bpm) == {1, 2}


def test_real_labels_present():
    """Стили правила Liquid Funk — настоящие имена классов Discogs-EffNet (опечатка — пустая полка)."""
    from pathlib import Path

    p = Path(__file__).parent / "data" / "discogs_styles.txt"
    names = {line.strip().split("---")[-1] for line in p.read_text().splitlines()}
    assert set(shelves.LIQUID.soft + shelves.LIQUID.hard) <= names
