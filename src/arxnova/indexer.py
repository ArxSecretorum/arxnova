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
