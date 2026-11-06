"""Решение 0014: Demucs только по 8 отрывкам. Голос — среднее числа по стему, слоя на CLAP выделенного голоса и описаний CLAP;
у разобранных Demucs по целому треку остаются прежние числа (доля звучания вокала, вероятность женского голоса). Без Demucs —
запасные слои на CLAP микса."""

import sqlite3
import sys
import types

import numpy as np
import pytest

from arxnova import config, db, indexer, models, nova, scan, sound_tags
from tests.test_flac_keys import make_flac, needs_ffmpeg
from tests.test_server_nova import fake_model


@needs_ffmpeg
def test_new_tracks_take_vocal_and_voice_from_clap(tmp_path, monkeypatch):
    lib = tmp_path / "lib"
    cfg = config.Config(data_dir=tmp_path / "data", libraries=[lib])
    cfg.ensure_dirs()
    fake_model(cfg)
    lib.mkdir()
    for i in range(6):
        make_flac(lib / f"{i:02d}.flac", seconds=0.3, freq=200 + 50 * i, ARTIST=f"A{i}", ALBUM=f"Al{i}", TITLE=f"T{i}")
    conn = db.connect(cfg.db_path)
    scan.scan(conn, [lib])
    path_of = {r[0]: r[1] for r in conn.execute("SELECT track_id, path FROM files")}
    row = {int(path_of[t].split("/")[-1][:2]): t for t in path_of}  # номер файла → трек
    rng = np.random.default_rng(0)
    for t in row.values():
        sty = np.zeros(400, np.float32)
        sty[0] = 1.0
        conn.execute("INSERT INTO embeddings VALUES (?,?,?,?,?)",
                     (t, models.EFFNET.id, rng.normal(size=64).astype(np.float32).tobytes(), sty.tobytes(), 1))
    # 0 и 1 разобраны с Demucs: доля вокала по стему и число голоса; остальные — без признаков вокала
    conn.execute("INSERT INTO track_features (track_id, version, bpm, vocal_frac, female_p) VALUES (?,1,120,0.6,0.9)", (row[0],))
    conn.execute("INSERT INTO track_features (track_id, version, bpm, vocal_frac) VALUES (?,1,120,0.0)", (row[1],))
    conn.execute("INSERT INTO track_features (track_id, version, bpm) VALUES (?,2,120)", (row[2],))
    # CLAP: первая координата — «вокал есть», вторая — «женский голос» (слои подменены ниже); у трека 4 CLAP нет
    want = {0: (0.0, 0.0), 1: (0.9, 0.9), 2: (0.9, 0.1), 3: (0.1, 0.9), 5: (0.8, 0.95)}
    for i, (pv, pf) in want.items():
        c = np.zeros(sound_tags.CLAP_DIM, np.float32)
        c[0], c[1] = pv, pf
        conn.execute("INSERT INTO clap_vectors VALUES (?,?)", (row[i], c.tobytes()))
    # у трека 5 есть CLAP выделенного голоса (решение 0014): среднее по голосу главнее CLAP микса
    conn.execute("INSERT INTO clap_voc VALUES (?,?)", (row[5], np.full(sound_tags.CLAP_DIM, 0.01, np.float32).tobytes()))
    conn.commit()
    monkeypatch.setattr(sound_tags, "vocal_probability_clap", lambda c: c[:, 0].astype(float))
    monkeypatch.setattr(sound_tags, "female_probability_clap", lambda c: c[:, 1].astype(float))
    seen = {}

    def fake_voice(stem, voc):
        seen["stem"] = stem.copy()
        return np.where(~np.isnan(voc).any(axis=1), 0.05, stem)

    monkeypatch.setattr(sound_tags, "female_from_voice", fake_voice)
    indexer.export_matrices(cfg, conn)
    indexer.export_features(cfg, conn)
    lb = nova.Library(cfg, conn)
    vocal = {i: int(lb.gate.vocal[lb.row[t]]) for i, t in row.items()}
    # прежняя доля по стему главнее CLAP (у 0 CLAP «нет вокала», у 1 — «есть»); у остальных — по CLAP; без CLAP — неизвестно
    assert vocal == {0: 1, 1: 0, 2: 1, 3: 0, 4: -1, 5: 1}
    female = lb.scenarios["voice_female"]["members"]
    male = lb.scenarios["voice_male"]["members"]
    # голос: у 0 — сохранённое число 0,9 (а не CLAP микса 0,0); у 5 — по выделенному голосу 0,05 (а не CLAP микса 0,95);
    # у 2 — по CLAP микса; у 1 и 3 вокала нет — ни в одной плитке
    assert female == {row[0]} and male == {row[2], row[5]}
    assert abs(seen["stem"][lb.row[row[0]]] - 0.9) < 1e-6  # у разобранного по целому треку его число идёт в среднее как мнение стема


@needs_ffmpeg
def test_features_pass_needs_no_demucs(tmp_path, monkeypatch):
    """Проход признаков — по миксу: темп, громкость, плотность ударов. Ни Demucs, ни векторов фасетов, ни модели EffNet."""
    import sys

    from arxnova import facets

    lib = tmp_path / "lib"
    cfg = config.Config(data_dir=tmp_path / "data", libraries=[lib])
    cfg.ensure_dirs()
    lib.mkdir()
    make_flac(lib / "a.flac", seconds=6.0, freq=440, ARTIST="A", TITLE="T")
    conn = db.connect(cfg.db_path)
    scan.scan(conn, [lib])
    monkeypatch.setattr(facets, "ensure_beats", lambda models_dir: None)  # без сети: темп автокорреляцией
    monkeypatch.setitem(sys.modules, "demucs", None)  # импорт Demucs упал бы
    assert indexer.facets_pass(cfg, conn) == 1
    r = conn.execute("SELECT version, bpm_src, loudness_db, onset_rate, vocal_frac FROM track_features").fetchone()
    assert r[0] == facets.VERSION == 2 and r[1] == "autocorr" and r[2] < 0 and r[3] >= 0 and r[4] is None
    assert conn.execute("SELECT count(*) FROM facet_vectors").fetchone()[0] == 0
    assert indexer.facets_pass(cfg, conn) == 0  # повтор ничего не считает


def test_old_index_features_are_refreshed(tmp_path):
    """Сервер после обновления: в признаках индекса нет числа голоса, рядом матрицы фасетов — выгружаются заново при запуске."""
    import json

    cfg = config.Config(data_dir=tmp_path / "data", libraries=[])
    cfg.ensure_dirs()
    conn = db.connect(cfg.db_path)
    d = cfg.index_dir / models.EFFNET.id
    d.mkdir(parents=True)
    np.save(d / "ids.npy", np.array([1, 2]))
    np.save(d / "facet_vocal.npy", np.zeros((2, 4), np.float32))
    (d / "features.json").write_text(json.dumps({"columns": ["bpm", "vocal_frac"], "facets": ["vocal"], "version": 1}))
    assert indexer.refresh_features(cfg, conn)
    assert "female_p" in json.loads((d / "features.json").read_text())["columns"]
    assert not list(d.glob("facet_*.npy"))
    assert not indexer.refresh_features(cfg, conn)  # уже свежий


def test_progress_counts_features_of_any_version(tmp_path):
    """Медиатека, разобранная с Demucs (признаки версии 1), — разобрана: окно не показывает «0 из N», а разбор не крутится
    при каждом обходе."""
    from arxnova import analysis

    cfg = config.Config(data_dir=tmp_path / "data", libraries=[])
    cfg.ensure_dirs()
    conn = db.connect(cfg.db_path)
    for t in (1, 2):
        conn.execute("INSERT INTO tracks (id, md5, md5_source, recording_key, recording_basis, sample_rate, bits, channels, duration) "
                     "VALUES (?, ?, 'streaminfo', ?, 'x', 44100, 16, 2, 100)", (t, bytes([t]), bytes([t])))
    conn.execute("INSERT INTO track_features (track_id, version, bpm) VALUES (1, 1, 120)")
    conn.execute("INSERT INTO track_features (track_id, version, bpm) VALUES (2, 2, 120)")
    assert analysis.pending(conn)["facets"] == 2


@needs_ffmpeg
def test_broken_legacy_voice_vector_does_not_stop_import(tmp_path):
    """Битая строка старого вектора в файле медиатеки (длина не кратна 4) не обрывает импорт: остальное приходит."""
    from arxnova import sidecar

    lib = tmp_path / "disk"
    lib.mkdir()
    make_flac(lib / "a.flac", freq=300.0, ARTIST="A", TITLE="One")
    cfg = config.Config(data_dir=tmp_path / "data", libraries=[lib])
    cfg.ensure_dirs()
    conn = db.connect(cfg.db_path)
    scan.scan(conn, [lib])
    md5 = bytes(conn.execute("SELECT md5 FROM tracks").fetchone()[0])
    sc = sidecar._open(lib, create=True)
    with sc:
        sc.execute("INSERT INTO facet VALUES (?,?,?,?,?)", (md5, 0, "vocal", "effnet-discogs-1", b"x" * 7))
        sc.execute("INSERT INTO feat (md5, cue_ms, version, bpm) VALUES (?,0,1,120.0)", (md5,))
    sc.close()
    assert sidecar.import_missing(conn, [lib]) == 1
    assert conn.execute("SELECT bpm, female_p FROM track_features").fetchone()[:] == (120.0, None)


class _FakeDemucs:
    """Подмена htdemucs: голос — доля входа (вокальный трек) или ноль (инструментал)."""
    segment, samplerate, sources = 7.8, 44100, ["drums", "bass", "other", "vocals"]

    def __init__(self, voice_share):
        self.voice_share = voice_share

    def to(self, device):
        return self

    def __call__(self, x):
        import torch

        out = torch.zeros(x.shape[0], 4, *x.shape[1:])
        out[:, 3] = x * self.voice_share
        out[:, 0] = x * (1 - self.voice_share)
        return out


class _FakeEffnet:
    def run_windows(self, patches):
        return np.tile(np.linspace(-1, 1, 1280, dtype=np.float32), (len(patches), 1))


def _analyzer(monkeypatch, voice_share):
    import sys

    import torch

    from arxnova import facets

    monkeypatch.setitem(sys.modules, "demucs", None)
    an = facets.Analyzer(_FakeEffnet())
    an.demucs, an.torch, an.device = _FakeDemucs(voice_share), torch, "cpu"
    return an


def test_voice_from_eight_excerpts(monkeypatch):
    """Голос по 8 отрывкам: доля звучания и энергии, вероятность голоса по стему в окнах с голосом, сам голос для CLAP."""
    from arxnova import facets

    monkeypatch.setattr(sound_tags, "female_probability_stem", lambda v: np.full(len(v), 0.8))
    t = np.arange(facets.SR * 90) / facets.SR
    x = np.stack([np.sin(2 * np.pi * 220 * t), np.sin(2 * np.pi * 220 * t)]).astype(np.float32) * 0.3
    f = _analyzer(monkeypatch, 0.9).analyze(x)
    assert f.vocal_frac == 1.0 and f.vocal_ratio > 0.7 and f.female_stem_p == 0.8
    assert len(f.voice) == round(facets.EXCERPTS * 7.8 * facets.VOICE_SR)
    g = _analyzer(monkeypatch, 0.0).analyze(x)  # инструментал: голоса нет
    assert g.vocal_frac == 0.0 and g.vocal_ratio < 1e-6


@needs_ffmpeg
def test_voice_goes_to_clap_and_file_is_removed(tmp_path, monkeypatch):
    """Проход признаков пишет выделенный голос во временный файл; проход сценариев кладёт CLAP голоса в базу и удаляет файл."""
    from arxnova import facets, scenarios

    lib = tmp_path / "lib"
    cfg = config.Config(data_dir=tmp_path / "data", libraries=[lib])
    cfg.ensure_dirs()
    lib.mkdir()
    make_flac(lib / "a.flac", seconds=8.0, freq=330, ARTIST="A", TITLE="T")
    conn = db.connect(cfg.db_path)
    scan.scan(conn, [lib])
    tid = conn.execute("SELECT id FROM tracks").fetchone()[0]
    voice = np.zeros(facets.VOICE_SR * 3, np.float32)

    class Stub:
        beats, demucs, device = None, object(), "cpu"

        def __init__(self, *a, **k):
            pass

        def analyze(self, stereo):
            return facets.TrackFacets(120.0, 0.9, "beat_this", -12.0, 3.0, 0.6, 0.2, 0.3, voice)

    monkeypatch.setattr(facets, "Analyzer", Stub)
    monkeypatch.setattr(facets, "ensure_beats", lambda d: None)
    assert indexer.facets_pass(cfg, conn) == 1
    vf = indexer.voice_dir(cfg) / f"{tid}.flac"
    assert vf.exists()
    row = conn.execute("SELECT version, vocal_frac, female_stem_p FROM track_features").fetchone()
    assert tuple(row) == (facets.VERSION, 0.6, 0.3)
    seen = {}

    def fake_worker(cfg_, tracks, texts, progress=print):
        seen["tracks"] = tracks
        ids = np.array([t[0] for t in tracks])
        return {"ids": ids, "audio": np.ones((len(ids), 512), np.float32) * 0.01, "text": np.zeros((len(texts), 512), np.float32)}

    monkeypatch.setattr(scenarios, "ensure_ckpt", lambda cfg_, progress=print: None)
    monkeypatch.setattr(scenarios, "run_worker", fake_worker)
    scenarios.index_pass(cfg, conn, progress=lambda m: None)
    assert any(t[0] == -tid for t in seen["tracks"])  # голос — отдельной строкой со знаком минус
    assert conn.execute("SELECT count(*) FROM clap_voc WHERE track_id = ?", (tid,)).fetchone()[0] == 1
    assert conn.execute("SELECT count(*) FROM clap_vectors WHERE track_id = ?", (tid,)).fetchone()[0] == 1
    assert not vf.exists()


def test_schema_18_of_the_published_branch_gets_voice_columns(tmp_path, monkeypatch):
    """Миграция 18 в опубликованной ветке добавляла одну female_p: база, прошедшая её, получает остальное миграцией 19."""
    path = tmp_path / "a.sqlite3"
    published = db.MIGRATIONS[:17] + ["ALTER TABLE track_features ADD COLUMN female_p REAL;"]  # 18 в коммите d2af16c
    monkeypatch.setattr(db, "MIGRATIONS", published)
    db.connect(path).close()
    monkeypatch.undo()
    conn = db.connect(path)
    assert {"female_p", "female_stem_p"} <= {r[1] for r in conn.execute("PRAGMA table_info(track_features)")}
    assert conn.execute("SELECT count(*) FROM clap_voc").fetchone()[0] == 0


def test_demucs_that_fails_to_load_stops_the_pass(monkeypatch):
    """Demucs установлен, но не загрузился (веса не скачались, устройство): проход падает и повторится позже, а не пишет треки
    без голоса навсегда. Не установлен — разбор без голоса."""
    pytest.importorskip("torch")
    from arxnova import facets

    def broken(name):
        raise RuntimeError("нет сети")

    mod = types.ModuleType("demucs.pretrained")
    mod.get_model = broken
    monkeypatch.setitem(sys.modules, "demucs", types.ModuleType("demucs"))
    monkeypatch.setitem(sys.modules, "demucs.pretrained", mod)
    with pytest.raises(RuntimeError):
        facets.Analyzer(None)
    monkeypatch.setitem(sys.modules, "demucs", None)
    monkeypatch.delitem(sys.modules, "demucs.pretrained")
    assert facets.Analyzer(None).demucs is None


def test_voice_only_clap_takes_just_the_voice(tmp_path, monkeypatch):
    """CLAP выделенного голоса после куска разбора: только файлы голоса, без очереди CLAP микса и без описаний сценариев;
    нет файлов — процесс CLAP не запускается."""
    from arxnova import scenarios

    cfg = config.Config(data_dir=tmp_path / "data", libraries=[])
    cfg.ensure_dirs()
    conn = db.connect(cfg.db_path)
    for i in (1, 2):
        conn.execute("INSERT INTO tracks (id, md5, md5_source, recording_key, recording_basis, sample_rate, bits, channels, duration) "
                     "VALUES (?,?,'pcm',?,'md5',44100,16,2,200.0)", (i, bytes([i]) * 16, bytes([i]) * 16))
    conn.commit()
    calls = []

    def fake_worker(cfg_, tracks, texts, progress=print):
        calls.append(tracks)
        ids = np.array([t[0] for t in tracks])
        return {"ids": ids, "audio": np.ones((len(ids), 512), np.float32), "text": np.zeros((len(texts), 512), np.float32)}

    monkeypatch.setattr(scenarios, "run_worker", fake_worker)
    monkeypatch.setattr(scenarios, "ensure_ckpt", lambda cfg_, progress=print: calls.append("ckpt"))
    assert scenarios.index_pass(cfg, conn, progress=lambda m: None, voice_only=True) == 0
    assert calls == []
    vdir = indexer.voice_dir(cfg)
    vdir.mkdir(parents=True)
    (vdir / "2.flac").write_bytes(b"x")
    assert scenarios.index_pass(cfg, conn, progress=lambda m: None, voice_only=True) == 1
    assert [t[0] for t in calls[-1]] == [-2]  # трек 1 без CLAP микса в очередь не попал
    assert [r[0] for r in conn.execute("SELECT track_id FROM clap_voc")] == [2]
    assert conn.execute("SELECT count(*) FROM clap_vectors").fetchone()[0] == 0
    assert not (vdir / "2.flac").exists() and not (cfg.index_dir / "laion-clap-music" / "scenario_text.npz").exists()


def test_deep_analysis_sends_voice_to_clap_after_each_chunk(tmp_path, monkeypatch):
    """Свой разбор: после каждого куска признаков — CLAP выделенного голоса, временные файлы не копятся за всю медиатеку."""
    from arxnova import analysis, sidecar

    cfg = config.Config(data_dir=tmp_path / "data", libraries=[])
    cfg.ensure_dirs()
    state = {"facets": 0, "files": 0}
    log = []

    def fake_pending(conn):
        return {"tracks": 600, "need_ffmpeg": 0, "mix": 600, "facets": state["facets"], "clap": 600, "tempo": 0, "maest": 600}

    def fake_run(cmd, env, on_line=None):
        args = cmd[cmd.index("--data-dir") + 2:]
        log.append((" ".join(args), state["files"]))
        vdir = indexer.voice_dir(cfg)
        if args[0] == "facets":
            state["facets"] = min(600, state["facets"] + analysis.FACETS_CHUNK)
            vdir.mkdir(parents=True, exist_ok=True)
            for i in range(3):
                (vdir / f"{state['facets'] + i}.flac").write_bytes(b"x")
                state["files"] += 1
        elif args == ["scenarios", "--voice"]:
            for f in vdir.glob("*.flac"):
                f.unlink()
            state["files"] = 0

    monkeypatch.setattr(analysis, "helper_active", lambda: False)
    monkeypatch.setattr(analysis, "installed", lambda cfg_: True)
    monkeypatch.setattr(analysis, "pending", fake_pending)
    monkeypatch.setattr(analysis, "deep_env", lambda cfg_: {})
    monkeypatch.setattr(analysis, "_cmd", lambda cfg_: ["arxnova"])
    monkeypatch.setattr(analysis, "_run", fake_run)
    monkeypatch.setattr(sidecar, "export", lambda *a, **k: None)
    monkeypatch.setattr(sidecar, "index_files", lambda cfg_: [])
    assert analysis.run_deep(cfg, None) is True
    cmds = [c for c, _ in log]
    assert cmds == ["facets --limit 250", "scenarios --voice"] * 3 + ["scenarios"]
    assert max(n for _, n in log) <= 3  # не больше одного куска голоса на диске
