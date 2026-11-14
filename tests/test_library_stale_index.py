"""Индекс старше базы (разбор 07.11.2026, docs/bugs/2026-11-07-library-keyerror-stale-index.md): трек удалён, а индекс ещё с ним —
каталог собирается без него, позиционные матрицы (признаки, карта) выравниваются; индекс, в котором нет ни одного трека из базы
(Mac без своей медиатеки), — каталог «пуст», сервер отвечает как без индекса, а экспорт такой индекс стирает."""

import json

import numpy as np
import pytest

from arxnova import config, db, indexer, models, nova
from arxnova import server as S
from tests.test_server_nova import fake_model

N_STYLES = 400


def _setup(tmp_path, n_tracks=4):
    cfg = config.Config(data_dir=tmp_path / "data", libraries=[tmp_path / "lib"])
    cfg.ensure_dirs()
    fake_model(cfg)
    conn = db.connect(cfg.db_path, threaded=True)
    rng = np.random.default_rng(1)
    for t in range(1, n_tracks + 1):
        conn.execute(
            "INSERT INTO tracks (id, md5, md5_source, recording_key, recording_basis, sample_rate, bits, channels, duration, "
            "artist, album, title) VALUES (?, ?, 'streaminfo', ?, 'x', 44100, 16, 2, 200, ?, ?, ?)",
            (t, bytes([t]), bytes([t]), f"Artist{t}", f"Album{t}", f"T{t}"),
        )
        sty = np.zeros(N_STYLES, np.float32)
        sty[t % 3] = 1.0
        conn.execute("INSERT INTO embeddings VALUES (?,?,?,?,?)",
                     (t, models.EFFNET.id, rng.normal(size=64).astype(np.float32).tobytes(), sty.tobytes(), 1))
    conn.commit()
    assert indexer.export_matrices(cfg, conn) is not None
    return cfg, conn


def _features(cfg, n):
    """Признаки звука в порядке индекса: темп и доля вокала у всех — шлюз считает такие треки разобранными."""
    d = cfg.index_dir / models.EFFNET.id
    feats = np.full((n, 2), np.nan, np.float32)
    feats[:, 0] = 120.0
    feats[:, 1] = 0.0
    np.save(d / "features.npy", feats)
    (d / "features.json").write_text(json.dumps({"columns": ["bpm", "vocal_frac"], "version": 2}))


def test_track_deleted_after_index_is_skipped(tmp_path, capsys):
    cfg, conn = _setup(tmp_path, 4)
    _features(cfg, 4)
    with conn:
        conn.execute("DELETE FROM tracks WHERE id = 3")  # индекс остался на четырёх
    lib = nova.Library(cfg, conn)
    assert set(lib.tracks) == {1, 2, 4}
    assert [int(t) for t in lib.ids] == [1, 2, 4]
    assert lib.row == {1: 0, 2: 1, 4: 2}
    assert lib.index_size == 4 and list(lib.pos) == [0, 1, 3]
    assert lib.vec.shape[0] == 3 and lib.style_scores.shape[0] == 3
    # признаки звука шли в порядке старого индекса — выровнены, а не отброшены
    assert len(lib.gate.has_features) == 3 and lib.gate.has_features.all()
    # карта и прочие матрицы в порядке старого индекса выравниваются той же маской
    xy = np.arange(8, dtype=np.float32).reshape(4, 2)
    assert lib.aligned(xy).tolist() == [[0, 1], [2, 3], [6, 7]]
    assert "которых нет в базе" in capsys.readouterr().out


def test_index_without_any_track_is_empty(tmp_path, monkeypatch):
    cfg, conn = _setup(tmp_path, 2)
    with conn:
        conn.execute("DELETE FROM tracks")
    with pytest.raises(nova.LibraryEmpty):
        nova.Library(cfg, conn)
    monkeypatch.setattr(models, "is_installed", lambda *a, **k: True)
    builds = []
    real = nova.Library
    monkeypatch.setattr(nova, "Library", lambda *a, **k: (builds.append(1), real(*a, **k))[1])
    st = S.State(cfg)
    assert st.library() is None  # как без индекса, без ошибки запроса
    assert st.library() is None
    assert len(builds) == 1  # пока индекс тот же, каталог заново не собирается
    st.conn.close()


def test_export_drops_index_when_library_is_gone(tmp_path):
    cfg, conn = _setup(tmp_path, 2)
    d = cfg.index_dir / models.EFFNET.id
    with conn:
        conn.execute("DELETE FROM tracks")  # векторы ушли вместе с треками (ON DELETE CASCADE)
    assert indexer.export_matrices(cfg, conn) is None
    assert not (d / "ids.npy").exists() and not (d / "mix.npy").exists()


def test_export_keeps_index_while_vectors_are_recounted(tmp_path):
    """Треки есть, векторов временно нет (пересчёт): индекс остаётся на старых векторах, как и было."""
    cfg, conn = _setup(tmp_path, 2)
    d = cfg.index_dir / models.EFFNET.id
    with conn:
        conn.execute("DELETE FROM embeddings")
    assert indexer.export_matrices(cfg, conn) is None
    assert (d / "ids.npy").exists()
