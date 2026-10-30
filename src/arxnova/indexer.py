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


def mix_id() -> str:
    """Идентификатор векторов микса в таблице embeddings: модель + схема отрывков."""
    from arxnova.embed import MIX_SCHEME

    return f"{models.EFFNET.id}@{MIX_SCHEME}"


# Порядок разбора — не по id (альбом за альбомом), а вразброс по всей медиатеке: после 10 %
# работы в потоке уже есть треки всех стилей, а не первые альбомы по алфавиту. Перестановка
# детерминированная (хеш id), поэтому возобновление после паузы ничего не теряет.
SPREAD_ORDER = "((t.id * 2654435761) % 1000003), t.id"


def pending(conn: sqlite3.Connection, model_id: str) -> list[tuple[int, str, int, float, int]]:
    """(трек, путь, начало участка мс, длительность с, конец участка мс или -1) — у трека из
    CUE-образа анализируется только его участок файла."""
    return conn.execute(
        f"""SELECT t.id, min(p.path), t.cue_ms, t.duration, t.cue_end_ms FROM tracks t
           JOIN track_paths p ON p.track_id = t.id
           WHERE t.id NOT IN (SELECT track_id FROM embeddings WHERE model = ?)
           GROUP BY t.id ORDER BY {SPREAD_ORDER}""",
        (model_id,),
    ).fetchall()


def copy_same_recording(conn: sqlite3.Connection, model_id: str) -> int:
    """Одна запись в нескольких изданиях (ремастер, хайрез; ФТ-2.10) — вектор считается один раз
    и переписывается остальным. У владельца это 6 % треков."""
    with conn:
        cur = conn.execute(
            """INSERT OR IGNORE INTO embeddings (track_id, model, vector, styles, windows)
               SELECT t2.id, e.model, e.vector, e.styles, e.windows
               FROM embeddings e JOIN tracks t1 ON t1.id = e.track_id
               JOIN tracks t2 ON t2.recording_key = t1.recording_key AND t2.id <> t1.id
               WHERE e.model = ? AND t2.id NOT IN (SELECT track_id FROM embeddings WHERE model = ?)""",
            (model_id, model_id),
        )
    return cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0


def embed_pending(cfg: Config, conn: sqlite3.Connection, progress=None, should_pause=None) -> int:
    """Считает эмбеддинги для треков без них. should_pause() → True ставит на паузу между
    треками (на Pi — пока генерирует LLM бота, ОП-8); сделанное не теряется."""
    from arxnova.embed import EffnetDiscogs

    model = models.EFFNET
    if not models.is_installed(cfg.models_dir, model):
        return 0
    done = copy_same_recording(conn, mix_id())
    todo = pending(conn, mix_id())
    if not todo:
        return done
    net = EffnetDiscogs(models.onnx_path(cfg.models_dir, model), threads=cfg.index_threads)
    from concurrent.futures import ThreadPoolExecutor

    # конвейер: пока сеть считает трек N, соседний поток декодирует и готовит окна трека N+1
    # (декодирование — половина цены трека на процессоре; soundfile и soxr отпускают GIL)
    pool = ThreadPoolExecutor(max_workers=1)
    done_here = 0
    nxt = pool.submit(net.excerpt_patches, todo[0][1], todo[0][3], todo[0][2] / 1000.0)
    try:
        for i, (tid, path, cue_ms, duration, cue_end_ms) in enumerate(todo):
            while should_pause and should_pause():
                time.sleep(2)
            try:
                pats = nxt.result()
            except Exception as exc:  # битый файл не останавливает индексацию
                pats = None
                if progress:
                    progress(f"  пропуск {path}: {exc}")
            if i + 1 < len(todo):
                n_path, n_cue, n_dur = todo[i + 1][1], todo[i + 1][2], todo[i + 1][3]
                nxt = pool.submit(net.excerpt_patches, n_path, n_dur, n_cue / 1000.0)
            if pats is None:
                continue
            e = net.run(pats)
            with conn:
                conn.execute(
                    "INSERT OR REPLACE INTO embeddings (track_id, model, vector, styles, windows) VALUES (?,?,?,?,?)",
                    (tid, mix_id(), e.vector.astype(np.float32).tobytes(), e.styles.astype(np.float32).tobytes(), e.windows),
                )
            done_here += 1
            if progress and done_here % 25 == 0:
                progress(f"  … эмбеддинги {done_here}/{len(todo)}")
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    return done + done_here + copy_same_recording(conn, mix_id())


def maest_pending(conn: sqlite3.Connection) -> list[tuple[int, str, int, float, int]]:
    return conn.execute(
        f"""SELECT t.id, min(p.path), t.cue_ms, t.duration, t.cue_end_ms FROM tracks t
           JOIN track_paths p ON p.track_id = t.id
           WHERE t.id NOT IN (SELECT track_id FROM maest) GROUP BY t.id ORDER BY {SPREAD_ORDER}""").fetchall()


def maest_pass(cfg: Config, conn: sqlite3.Connection, progress=None, limit: int | None = None) -> int:
    """Стили MAEST-519 для треков без них (см. maest.py). Возобновляемо: единица — трек."""
    from arxnova import maest

    if not models.is_installed(cfg.models_dir, models.MAEST):
        return 0
    todo = maest_pending(conn)[:limit]
    if not todo:
        return 0
    net = maest.Maest(models.onnx_path(cfg.models_dir, models.MAEST), threads=max(2, cfg.index_threads // 2))
    if progress:
        progress(f"стили MAEST: {len(todo)} треков, " + (f"GPU ({net.device})" if net.torch_net is not None else "процессор"))
    done, t0 = 0, time.time()
    for tid, path, cue_ms, duration, cue_end_ms in todo:
        try:
            r = net.run_file(path, cue_ms / 1000.0, duration)
        except Exception as exc:  # битый файл не останавливает проход
            if progress:
                progress(f"  пропуск {path}: {exc}")
            continue
        with conn:
            conn.execute("INSERT OR REPLACE INTO maest (track_id, activations, embedding, windows) VALUES (?,?,?,?)",
                         (tid, r.activations.astype(np.float32).tobytes(), r.embedding.astype(np.float32).tobytes(),
                          r.windows))
        done += 1
        if progress and done % 25 == 0:
            left = (time.time() - t0) / done * (len(todo) - done)
            progress(f"  … стили {done}/{len(todo)}, осталось ~{int(left // 60)} мин")
    return done


def export_maest(cfg: Config, conn: sqlite3.Connection, model_id: str = models.EFFNET.id) -> Path | None:
    """Активации MAEST в порядке ids.npy (у трека без них — строка нулей, maest_has = 0)."""
    d = cfg.index_dir / model_id
    if not (d / "ids.npy").exists():
        return None
    ids = np.load(d / "ids.npy")
    row = {int(t): i for i, t in enumerate(ids)}
    acts = None
    has = np.zeros(len(ids), np.uint8)
    for tid, blob in conn.execute("SELECT track_id, activations FROM maest"):
        i = row.get(tid)
        if i is None:
            continue
        v = np.frombuffer(blob, np.float32)
        if acts is None:
            acts = np.zeros((len(ids), len(v)), np.float32)
        acts[i] = v
        has[i] = 1
    if acts is None:
        return None
    np.save(d / "maest.npy", acts)
    np.save(d / "maest_has.npy", has)
    return d


def mix_rows_model(conn: sqlite3.Connection) -> str:
    """Чьи векторы идут в индекс: схемы отрывков или прежние по целому треку. После обновления
    сервер досчитывает отрывки в фоне, а индекс до конца пересчёта остаётся на старых векторах —
    иначе медиатека в потоке временно «усохла» бы до пересчитанной части."""
    counts = {m: n for m, n in conn.execute("SELECT model, count(*) FROM embeddings GROUP BY model")}
    new, old = counts.get(mix_id(), 0), counts.get(models.EFFNET.id, 0)
    return mix_id() if new >= old else models.EFFNET.id


def export_matrices(cfg: Config, conn: sqlite3.Connection, model_id: str = models.EFFNET.id) -> Path | None:
    """Матрицы в index/<model_id>/: векторы — схемы отрывков, если она досчитана (mix_rows_model)."""
    rows_model = mix_rows_model(conn) if model_id == models.EFFNET.id else model_id
    rows = conn.execute(
        "SELECT track_id, vector, styles FROM embeddings WHERE model = ? ORDER BY track_id", (rows_model,)
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
                "rows": rows_model,
                "tracks": len(ids),
                "dim": int(mix.shape[1]),
                "preprocessing": {
                    "sample_rate": features.SR_EFFNET, "frame": features.FRAME, "hop": features.HOP,
                    "mels": features.N_MELS, "patch": features.PATCH, "patch_hop": features.PATCH_HOP,
                    "resampler": "soxr HQ", "aggregate": "mean",
                    "excerpts": list(features.EXCERPTS) if rows_model != model_id else None,
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
    export_maest(cfg, conn)
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
