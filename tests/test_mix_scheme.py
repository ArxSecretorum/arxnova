"""Эмбеддинг микса по отрывкам и объединённый вектор похожести (30.10.2026)."""

import numpy as np

from arxnova import features as F
from arxnova import indexer, models, nova


def test_excerpt_starts_inside_track():
    for dur in (3.0, 20.0, 61.0, 240.0, 1800.0):
        st = F.excerpt_starts(dur)
        assert len(st) == (1 if dur <= F.EXCERPTS[1] else F.EXCERPTS[0])
        assert all(0 <= s and s + F.EXCERPTS[1] <= max(dur, F.EXCERPTS[1]) + 1e-9 for s in st)
        assert st == sorted(st)
    # длинный трек: отрывки внутри 10–90 % и не перекрываются
    st = F.excerpt_starts(600.0)
    assert st[0] >= 60.0 and st[-1] + F.EXCERPTS[1] <= 540.0
    assert all(b - a >= F.EXCERPTS[1] for a, b in zip(st, st[1:]))


def test_mix_id_carries_scheme():
    assert indexer.mix_id().startswith(models.EFFNET.id + "@x")
    assert indexer.mix_id() != models.EFFNET.id


def test_export_keeps_old_vectors_until_excerpts_catch_up(tmp_path):
    import sqlite3

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE embeddings (track_id INTEGER, model TEXT, vector BLOB, styles BLOB, windows INTEGER)")
    v = np.zeros(4, np.float32).tobytes()
    for t in range(5):
        conn.execute("INSERT INTO embeddings VALUES (?,?,?,?,1)", (t, models.EFFNET.id, v, v))
    for t in range(3):  # отрывки досчитаны не для всех — индекс остаётся на старых
        conn.execute("INSERT INTO embeddings VALUES (?,?,?,?,1)", (t, indexer.mix_id(), v, v))
    assert indexer.mix_rows_model(conn) == models.EFFNET.id
    for t in range(3, 5):
        conn.execute("INSERT INTO embeddings VALUES (?,?,?,?,1)", (t, indexer.mix_id(), v, v))
    assert indexer.mix_rows_model(conn) == indexer.mix_id()


def test_similarity_vectors_mix_sound_and_style():
    rng = np.random.default_rng(1)
    mix = rng.normal(size=(6, 8)).astype(np.float32)
    styles = np.zeros((6, 3), np.float32)
    styles[:3, 0] = 1.0
    styles[3:, 1] = 1.0
    v = nova.similarity_vectors(mix, styles, 0.5)
    assert v.shape == (6, 11)
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-5)
    # без стилей — прежний вектор звука
    v0 = nova.similarity_vectors(mix, styles, 0.0)
    assert v0.shape == (6, 8)
    # одинаковый звук, разный стиль — дальше, чем одинаковый звук и стиль
    same = np.vstack([mix[0], mix[0]])
    st = np.array([[1, 0, 0], [0, 1, 0]], np.float32)
    a = nova.similarity_vectors(np.vstack([mix, same]), np.vstack([styles, st]), 0.5)
    assert a[6] @ a[7] < 1.0 - 1e-3


def test_hub_scores_penalise_track_close_to_everyone():
    rng = np.random.default_rng(2)
    v = rng.normal(size=(40, 16)).astype(np.float32)
    v[0] = v[1:].mean(axis=0) * 3  # «хаб»: направлен в центр облака
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    h = nova.hub_scores(v, k=5, block=7)  # блоки не делят длину — проверяем хвост
    assert h.shape == (40,)
    assert h[0] > np.median(h[1:])
    assert np.allclose(h, nova.hub_scores(v, k=5, block=256))  # размер блока не меняет результат
