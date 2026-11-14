"""Медиатека исчезла (диск переехал на другой компьютер), а матрицы индекса остались: модель подбора собиралась по номерам
треков, которых в базе уже нет, и падала на каждом запуске (Mac 06.11.2026: «прогрев: KeyError(1)»)."""

import numpy as np

from arxnova import cli, config, db, indexer, models, nova, scan
from arxnova.server import State
from tests.test_flac_keys import make_flac, needs_ffmpeg
from tests.test_server_nova import fake_model


def _library(tmp_path):
    lib = tmp_path / "lib"
    cfg = config.Config(data_dir=tmp_path / "data", libraries=[lib])
    cfg.ensure_dirs()
    fake_model(cfg)
    lib.mkdir()
    for i in range(4):
        make_flac(lib / f"{i:02d}.flac", seconds=0.3, freq=200 + 50 * i, ARTIST=f"A{i}", ALBUM=f"Al{i}", TITLE=f"T{i}")
    conn = db.connect(cfg.db_path)
    scan.scan(conn, [lib])
    rng = np.random.default_rng(0)
    for (t,) in conn.execute("SELECT id FROM tracks").fetchall():
        sty = np.zeros(400, np.float32)
        sty[0] = 1.0
        conn.execute("INSERT INTO embeddings VALUES (?,?,?,?,?)",
                     (t, models.EFFNET.id, rng.normal(size=64).astype(np.float32).tobytes(), sty.tobytes(), 1))
    conn.commit()
    indexer.export_all(cfg, conn)
    assert len(nova.Library(cfg, conn).ids) == 4
    return lib, cfg, conn


def _gone(lib, conn):
    for f in lib.iterdir():
        f.unlink()
    rep = scan.scan(conn, [lib])
    assert rep.removed == 4 and conn.execute("SELECT count(*) FROM tracks").fetchone()[0] == 0


@needs_ffmpeg
def test_matrices_of_a_library_that_is_gone_are_dropped(tmp_path):
    """Обход убрал все треки — выгрузка убирает и матрицы: подбора нет, но сервер не падает."""
    lib, cfg, conn = _library(tmp_path)
    _gone(lib, conn)
    indexer.export_all(cfg, conn)  # так делает обход медиатеки сервера, когда треки удалены
    assert not (cfg.index_dir / models.EFFNET.id / "ids.npy").exists()
    assert State(cfg).library() is None
    assert State(cfg).indexed_count() == 0


@needs_ffmpeg
def test_server_start_drops_matrices_left_by_an_older_version(tmp_path):
    """Сервер прежней версии матрицы оставил: запуск их убирает сам, без нового обхода."""
    lib, cfg, conn = _library(tmp_path)
    _gone(lib, conn)  # треки ушли, а матрицы остались — как на Mac
    conn.close()
    assert (cfg.index_dir / models.EFFNET.id / "ids.npy").exists()
    cli.prepare_storage(cfg)
    assert not any((cfg.index_dir / models.EFFNET.id).glob("*.npy"))
    assert State(cfg).library() is None


@needs_ffmpeg
def test_start_keeps_the_index_of_a_library_that_is_there(tmp_path):
    lib, cfg, conn = _library(tmp_path)
    conn.close()
    cli.prepare_storage(cfg)
    assert len(State(cfg).library().ids) == 4
