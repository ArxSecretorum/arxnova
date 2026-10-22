"""Индексация звука (М2, ФТ-2.2): эмбеддинги микса по трекам, возобновляемо. После прохода
собирает рабочие матрицы index/<model>/{ids,mix,styles}.npy + meta.json (§5.2)."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import numpy as np

from arxnova import features, models
from arxnova.config import Config


def pending(conn: sqlite3.Connection, model_id: str) -> list[tuple[int, str, int, float, int]]:
    """(трек, путь, начало участка мс, длительность с, конец участка мс или -1) — у трека из
    CUE-образа анализируется только его участок файла."""
    return conn.execute(
        """SELECT t.id, min(p.path), t.cue_ms, t.duration, t.cue_end_ms FROM tracks t
           JOIN track_paths p ON p.track_id = t.id
           WHERE t.id NOT IN (SELECT track_id FROM embeddings WHERE model = ?)
           GROUP BY t.id ORDER BY t.id""",
        (model_id,),
    ).fetchall()


def embed_pending(cfg: Config, conn: sqlite3.Connection, progress=None, should_pause=None) -> int:
    """Считает эмбеддинги для треков без них. should_pause() → True ставит на паузу между
    треками (на Pi — пока генерирует LLM бота, ОП-8); сделанное не теряется."""
    from arxnova.embed import EffnetDiscogs

    model = models.EFFNET
    if not models.is_installed(cfg.models_dir, model):
        return 0
    todo = pending(conn, model.id)
    if not todo:
        return 0
    net = EffnetDiscogs(models.onnx_path(cfg.models_dir, model), threads=cfg.index_threads)
    done = 0
    for tid, path, cue_ms, duration, cue_end_ms in todo:
        while should_pause and should_pause():
            time.sleep(2)
        try:
            whole = cue_ms == 0 and cue_end_ms < 0
            e = net.embed_file(path) if whole else net.embed_file(path, cue_ms / 1000.0, duration)
        except Exception as exc:  # битый файл не останавливает индексацию
            if progress:
                progress(f"  пропуск {path}: {exc}")
            continue
        with conn:
            conn.execute(
                "INSERT OR REPLACE INTO embeddings (track_id, model, vector, styles, windows) VALUES (?,?,?,?,?)",
                (tid, model.id, e.vector.astype(np.float32).tobytes(), e.styles.astype(np.float32).tobytes(), e.windows),
            )
        done += 1
        if progress and done % 25 == 0:
            progress(f"  … эмбеддинги {done}/{len(todo)}")
    return done


def export_matrices(cfg: Config, conn: sqlite3.Connection, model_id: str = models.EFFNET.id) -> Path | None:
    rows = conn.execute(
        "SELECT track_id, vector, styles FROM embeddings WHERE model = ? ORDER BY track_id", (model_id,)
    ).fetchall()
    if not rows:
        return None
    out = cfg.index_dir / model_id
    out.mkdir(parents=True, exist_ok=True)
    ids = np.array([r[0] for r in rows], dtype=np.int64)
    mix = np.stack([np.frombuffer(r[1], dtype=np.float32) for r in rows])
    np.save(out / "ids.npy", ids)
    np.save(out / "mix.npy", mix)
    if rows[0][2] is not None:
        np.save(out / "styles.npy", np.stack([np.frombuffer(r[2], dtype=np.float32) for r in rows]))
    (out / "meta.json").write_text(
        json.dumps(
            {
                "model": model_id,
                "tracks": len(ids),
                "dim": int(mix.shape[1]),
                "preprocessing": {
                    "sample_rate": features.SR_EFFNET, "frame": features.FRAME, "hop": features.HOP,
                    "mels": features.N_MELS, "patch": features.PATCH, "patch_hop": features.PATCH_HOP,
                    "resampler": "soxr HQ", "aggregate": "mean",
                },
                "sha256": {f.name: f.sha256 for f in models.CATALOG[model_id].files},
                "license": models.CATALOG[model_id].license,
                "built_at": int(time.time()),
            },
            ensure_ascii=False, indent=1,
        )
    )
    return out


def facets_pending(conn: sqlite3.Connection) -> list[tuple[int, str, int, float, int]]:
    from arxnova import facets

    return conn.execute(
        """SELECT t.id, min(p.path), t.cue_ms, t.duration, t.cue_end_ms FROM tracks t
           JOIN track_paths p ON p.track_id = t.id
           WHERE t.id NOT IN (SELECT track_id FROM track_features WHERE version = ?)
           GROUP BY t.id ORDER BY t.id""",
        (facets.VERSION,),
    ).fetchall()


def facets_pass(cfg: Config, conn: sqlite3.Connection, progress=None, limit: int | None = None) -> int:
    """Фасеты по трекам без них (ФТ-2.3, ФТ-2.4). Возобновляемо: единица работы — трек."""
    from arxnova import facets
    from arxnova.embed import EffnetDiscogs

    model = models.EFFNET
    if not models.is_installed(cfg.models_dir, model):
        return 0
    todo = facets_pending(conn)[:limit]
    if not todo:
        return tempo_pass(cfg, conn, progress)
    import os

    # проход идёт на настольной машине (ОП-5): бюджет Pi в один поток здесь ни к чему
    threads = max(cfg.index_threads, (os.cpu_count() or 2) // 2)
    an = facets.Analyzer(EffnetDiscogs(models.onnx_path(cfg.models_dir, model), threads=threads),
                         beats_ckpt=facets.ensure_beats(cfg.models_dir))
    if progress:
        progress(f"фасеты: {len(todo)} треков, темп — {'Beat This!' if an.beats else 'по ударным'}, "
                 f"устройство {an.device}")
    from concurrent.futures import ThreadPoolExecutor

    def load(item):
        tid, path, cue_ms, duration, cue_end_ms = item
        whole = cue_ms == 0 and cue_end_ms < 0
        return facets.load_stereo(path) if whole else facets.load_stereo(path, cue_ms / 1000.0, duration)

    def save(tid, f):
        with conn:
            for name, v in f.vectors.items():
                conn.execute("INSERT OR REPLACE INTO facet_vectors (track_id, facet, model, vector) VALUES (?,?,?,?)",
                             (tid, name, model.id, v.tobytes()))
            conn.execute(
                """INSERT OR REPLACE INTO track_features (track_id, version, bpm, bpm_conf, bpm_src, musical_key,
                   key_conf, loudness_db, onset_rate, vocal_frac, vocal_ratio) VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (tid, facets.VERSION, f.bpm, f.bpm_conf, f.bpm_src, f.key, f.key_conf, f.loudness_db,
                 f.onset_rate, f.vocal_frac, f.vocal_ratio),
            )

    def skip(path, exc):  # битый файл не останавливает проход
        if progress:
            progress(f"  пропуск {path}: {exc}")

    # конвейер: GPU разделяет трек N, пока процессор декодирует N+1 и считает EffNet/темп по N−1
    # (стемы — 85 % времени трека; остальное прячется за ними)
    done, t0 = 0, time.time()
    pool = ThreadPoolExecutor(max_workers=2)
    nxt = pool.submit(load, todo[0])
    tail = None  # (tid, path, future describe) предыдущего трека
    try:
        for i, item in enumerate(todo):
            tid, path = item[0], item[1]
            try:
                x = nxt.result()
            except Exception as exc:
                x = None
                skip(path, exc)
            nxt = pool.submit(load, todo[i + 1]) if i + 1 < len(todo) else None
            cur = None
            if x is not None:
                try:
                    st = an.separate(x)
                    cur = (tid, path, pool.submit(an.describe, x, st))
                except Exception as exc:
                    skip(path, exc)
            if tail:
                done += _finish(tail, save, skip)
            tail = cur
            if progress and done and done % 10 == 0:
                left = (time.time() - t0) / done * (len(todo) - done)
                progress(f"  … фасеты {done}/{len(todo)}, осталось ~{int(left // 60)} мин")
        if tail:
            done += _finish(tail, save, skip)
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    return done + tempo_pass(cfg, conn, progress, an.beats)


def tempo_pending(conn: sqlite3.Connection) -> list[tuple[int, str, int, float, int]]:
    return conn.execute(
        """SELECT t.id, min(p.path), t.cue_ms, t.duration, t.cue_end_ms FROM tracks t
           JOIN track_paths p ON p.track_id = t.id JOIN track_features f ON f.track_id = t.id
           WHERE f.bpm_src = 'autocorr' GROUP BY t.id ORDER BY t.id""").fetchall()


def tempo_pass(cfg: Config, conn: sqlite3.Connection, progress=None, beats=None) -> int:
    """Темп по долям (Beat This!) для треков, где он был посчитан по ударным, пока весов не было.
    Нужен только микс — без стемов, ~1 с на трек. Не нашёл долей — помечаем, чтобы не повторять."""
    from arxnova import facets

    todo = tempo_pending(conn)
    if not todo:
        return 0
    if beats is None:
        ckpt = facets.ensure_beats(cfg.models_dir)
        if not ckpt:
            return 0
        import torch
        from beat_this.inference import Audio2Beats

        beats = Audio2Beats(checkpoint_path=ckpt, device="mps" if torch.backends.mps.is_available() else "cpu", dbn=False)
    if progress:
        progress(f"темп по долям: {len(todo)} треков")
    done = 0
    for tid, path, cue_ms, duration, cue_end_ms in todo:
        try:
            whole = cue_ms == 0 and cue_end_ms < 0
            x = facets.load_stereo(path) if whole else facets.load_stereo(path, cue_ms / 1000.0, duration)
            got = facets.tempo_from_beats(beats, x)
        except Exception:
            got = None
        with conn:
            if got:
                conn.execute("UPDATE track_features SET bpm = ?, bpm_conf = ?, bpm_src = 'beat_this' WHERE track_id = ?",
                             (got[0], got[1], tid))
                done += 1
            else:
                conn.execute("UPDATE track_features SET bpm_src = 'autocorr_only' WHERE track_id = ?", (tid,))
        if progress and done and done % 50 == 0:
            progress(f"  … темп {done}/{len(todo)}")
    return done


def _finish(tail, save, skip) -> int:
    tid, path, fut = tail
    try:
        save(tid, fut.result())
        return 1
    except Exception as exc:
        skip(path, exc)
        return 0


def export_facets(cfg: Config, conn: sqlite3.Connection, model_id: str = models.EFFNET.id) -> Path | None:
    """Матрицы фасетов рядом с матрицей микса, строки — в порядке ids.npy; у трека без фасетов
    строка нулевая и has_facets = 0 (шлюз такой трек межжанрово не пускает)."""
    d = cfg.index_dir / model_id
    if not (d / "ids.npy").exists():
        return None
    ids = np.load(d / "ids.npy")
    row = {int(t): i for i, t in enumerate(ids)}
    from arxnova import facets

    mats: dict[str, np.ndarray] = {}
    for tid, facet, blob in conn.execute("SELECT track_id, facet, vector FROM facet_vectors WHERE model = ?", (model_id,)):
        if tid not in row:
            continue
        v = np.frombuffer(blob, dtype=np.float32)
        m = mats.setdefault(facet, np.zeros((len(ids), len(v)), np.float32))
        m[row[tid]] = v
    for name in facets.FACETS:
        if name in mats:
            np.save(d / f"facet_{name}.npy", mats[name])
    cols = ["bpm", "bpm_conf", "musical_key", "key_conf", "loudness_db", "onset_rate", "vocal_frac", "vocal_ratio"]
    feat = np.full((len(ids), len(cols)), np.nan, np.float32)
    for r in conn.execute(f"SELECT track_id, {', '.join(cols)} FROM track_features"):
        if r[0] in row:
            feat[row[r[0]]] = [np.nan if x is None else x for x in r[1:]]
    np.save(d / "features.npy", feat)
    (d / "features.json").write_text(json.dumps({"columns": cols, "facets": sorted(mats), "version": facets.VERSION}))
    return d


def export_all(cfg: Config, conn: sqlite3.Connection) -> None:
    """Все матрицы разом: строки фасетов, признаков, карты и сценариев идут в порядке ids.npy —
    пересобрали микс (трек добавили или убрали), пересобрать и их. Иначе размеры расходятся, и
    сервер молча выключает шлюз и фильтры (так было после удаления одного трека)."""
    export_matrices(cfg, conn)
    export_facets(cfg, conn)
    export_map(cfg)
    try:
        from arxnova import nova, scenarios

        scenarios.export(cfg, conn, nova.Library(cfg, conn))
    except Exception:
        pass  # без векторов CLAP сценариев просто нет


def export_map(cfg: Config, model_id: str = models.EFFNET.id) -> Path | None:
    """Карта медиатеки на плоскости (КО-5): UMAP по векторам микса — для панели шлюза; мосты
    видны как треки на стыках облаков. Только на настольной машине (umap-learn — в [stems])."""
    d = cfg.index_dir / model_id
    if not (d / "mix.npy").exists():
        return None
    try:
        import umap
    except ImportError:
        return None
    v = np.load(d / "mix.npy").astype(np.float32)
    v -= v.mean(axis=0)
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
    xy = umap.UMAP(n_neighbors=min(15, len(v) - 1), min_dist=0.15, metric="cosine", random_state=7).fit_transform(v)
    xy = (xy - xy.min(axis=0)) / (np.ptp(xy, axis=0) + 1e-9)
    np.save(d / "map2d.npy", xy.astype(np.float32))
    return d / "map2d.npy"
