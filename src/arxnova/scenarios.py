"""Сценарии подбора (ФТ-4.10): готовые сочетания фильтров и направления по звуку.

Трек относится к сценарию, если его звук близок к текстовому описанию (CLAP «текст → звук»,
оригинальные веса LAION, CC0) — входит в верхнюю четверть медиатеки по этой близости — и
проходит фильтры сценария (темп с half/double-time, энергия, вокал). Принадлежность считается при
индексации (`arxnova scenarios`), а не на каждый запрос (бюджет §4). Плеер показывает только
сценарии, под которые набирается не меньше FACET_MIN_TRACKS записей.

Свои сценарии владелец пишет в <data>/scenarios.toml:

    [[scenario]]
    id = "sunday"
    title = "Воскресное утро"
    prompt = "soft warm acoustic morning music"
    energy = [0.0, 0.5]
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
import sys
import tempfile
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from arxnova.config import Config

CKPT_NAME = "music_audioset_epoch_15_esc_90.14.pt"
CKPT_URL = "https://huggingface.co/lukewys/laion_clap/resolve/main/" + CKPT_NAME
CKPT_SHA256 = "fae3e9c087f2909c28a09dc31c8dfcdacbc42ba44c70e972b58c1bd1caf6dedd"
MODEL_DIR = "laion-clap-music"
TOP_SHARE = 0.25  # к сценарию — верхняя четверть медиатеки по близости к описанию
# окружение процесса CLAP (laion_clap требует numpy < 2) — версии, проверенные на этапе 0
WORKER_DEPS = ["laion-clap==1.1.7", "numpy<2", "torch==2.14.1", "transformers==4.46.3",
               "soundfile==0.14.0", "soxr==1.1.0", "torchvision==0.29.1"]


@dataclass
class Scenario:
    id: str
    title: str
    prompt: str
    bpm: tuple[float, float] | None = None
    energy: tuple[float, float] | None = None
    vocal: str | None = None
    # только треки этих стилей Discogs-EffNet (полка внутри жанра), и место — по разнице «близость к
    # prompt минус близость к against»: так выделяется то, чего нет среди стилей модели
    styles: tuple[str, ...] | None = None
    against: str | None = None
    members: set[int] = field(default_factory=set)


DEFAULTS = [
    Scenario("run", "Для бега", "fast energetic running music with a driving beat", bpm=(150, 185), energy=(0.5, 1.0)),
    Scenario("focus", "Для работы", "calm instrumental background music for concentration", energy=(0.0, 0.6), vocal="no"),
    Scenario("mystic", "Мистическая", "dark mysterious atmospheric music"),
    Scenario("chill", "Расслабиться", "relaxed chillout lounge music", energy=(0.0, 0.65)),
    Scenario("night", "Ночная дорога", "night drive music with deep bass and synthesizers"),
    Scenario("uplift", "Поднять настроение", "uplifting happy energetic music", energy=(0.4, 1.0)),
]
STYLE_SHARE = 0.4  # полка внутри стиля — верхние 40 % этого стиля по разнице


def load_defs(cfg: Config) -> list[Scenario]:
    defs = [Scenario(s.id, s.title, s.prompt, s.bpm, s.energy, s.vocal, s.styles, s.against) for s in DEFAULTS]
    own = cfg.data_dir / "scenarios.toml"
    if own.exists():
        for r in tomllib.loads(own.read_text(encoding="utf-8")).get("scenario", []):
            if not r.get("id") or not r.get("prompt"):
                continue
            rng = lambda k: tuple(map(float, r[k])) if isinstance(r.get(k), list) and len(r[k]) == 2 else None  # noqa: E731
            defs = [d for d in defs if d.id != r["id"]] + [
                Scenario(str(r["id"]), str(r.get("title", r["id"])), str(r["prompt"]), rng("bpm"), rng("energy"),
                         r.get("vocal") if r.get("vocal") in ("yes", "no") else None,
                         tuple(map(str, r["styles"])) if isinstance(r.get("styles"), list) else None,
                         str(r["against"]) if r.get("against") else None)]
    return defs


AGAINST = "~against"


def _texts(defs: list[Scenario]) -> tuple[list[str], list[str]]:
    """Тексты для CLAP и их ключи: описание сценария и, если есть, «против чего»."""
    texts, ids = [], []
    for d in defs:
        texts.append(d.prompt); ids.append(d.id)
        if d.against:
            texts.append(d.against); ids.append(d.id + AGAINST)
    return texts, ids


def ckpt_path(cfg: Config) -> Path:
    return cfg.models_dir / MODEL_DIR / CKPT_NAME


def ensure_ckpt(cfg: Config, progress=print) -> Path:
    """Веса LAION (CC0, 2,3 ГБ) — с репозитория автора, со сверкой sha256."""
    p = ckpt_path(cfg)
    if p.exists():
        return p
    import urllib.request

    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".part")
    progress(f"скачиваю веса CLAP (LAION, CC0, 2,3 ГБ): {CKPT_URL}")
    urllib.request.urlretrieve(CKPT_URL, tmp)
    h = hashlib.sha256()
    with open(tmp, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    if h.hexdigest() != CKPT_SHA256:
        tmp.unlink()
        raise RuntimeError("контрольная сумма весов CLAP не совпала")
    tmp.rename(p)
    return p


def run_worker(cfg: Config, tracks: list[tuple], texts: list[str], progress=print) -> dict:
    """CLAP в своём окружении (uv) — numpy < 2 не смешивается с сервером."""
    worker = Path(__file__).with_name("clap_worker.py")
    with tempfile.TemporaryDirectory() as td:
        job = Path(td) / "job.json"
        out = Path(td) / "out.npz"
        job.write_text(json.dumps({"ckpt": str(ckpt_path(cfg)), "tracks": tracks, "texts": texts, "out": str(out)}))
        cmd = ["uv", "run", "--no-project", "--python", f"{sys.version_info.major}.{sys.version_info.minor}"]
        for d in WORKER_DEPS:
            cmd += ["--with", d]
        cmd += ["python", str(worker), str(job)]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            if line.startswith(("  …", "пропуск")):
                progress(line.rstrip())
        if proc.wait() != 0 or not out.exists():
            raise RuntimeError("процесс CLAP завершился с ошибкой")
        z = np.load(out)
        return {"ids": z["ids"], "audio": z["audio"], "text": z["text"]}


def index_pass(cfg: Config, conn: sqlite3.Connection, progress=print) -> int:
    """Векторы CLAP для треков без них и векторы описаний сценариев; затем принадлежность."""
    ensure_ckpt(cfg, progress)
    todo = conn.execute(
        """SELECT t.id, min(p.path), t.cue_ms, t.duration, t.cue_end_ms FROM tracks t
           JOIN track_paths p ON p.track_id = t.id
           WHERE t.id NOT IN (SELECT track_id FROM clap_vectors) GROUP BY t.id ORDER BY t.id""").fetchall()
    defs = load_defs(cfg)
    tracks = [(tid, path, cue / 1000.0, dur if (cue or end >= 0) else None) for tid, path, cue, dur, end in todo]
    if tracks:
        progress(f"CLAP: {len(tracks)} треков")
    texts, text_ids = _texts(defs)
    res = run_worker(cfg, tracks, texts, progress)
    with conn:
        for tid, v in zip(res["ids"], res["audio"]):
            conn.execute("INSERT OR REPLACE INTO clap_vectors (track_id, vector) VALUES (?,?)",
                         (int(tid), v.astype(np.float32).tobytes()))
    d = cfg.index_dir / "laion-clap-music"
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / "scenario_text.npz", ids=np.array(text_ids), text=res["text"], prompts=np.array(texts))
    return len(res["ids"])


def export(cfg: Config, conn: sqlite3.Connection, lib) -> Path | None:
    """Принадлежность треков сценариям → index/<модель>/scenarios.json (читает сервер)."""
    from arxnova import nova

    tf = cfg.index_dir / "laion-clap-music" / "scenario_text.npz"
    if not tf.exists():
        return None
    z = np.load(tf)
    text = {str(i): v for i, v in zip(z["ids"], z["text"])}
    rows = conn.execute("SELECT track_id, vector FROM clap_vectors").fetchall()
    ids = [r[0] for r in rows if r[0] in lib.tracks]
    if not ids:
        return None
    audio = np.stack([np.frombuffer(r[1], np.float32) for r in rows if r[0] in lib.tracks])
    out = {}
    for s in load_defs(cfg):
        if s.id not in text:
            continue  # описание добавили после прохода — посчитается в следующий
        if s.against and s.id + AGAINST not in text:
            continue
        sims = audio @ text[s.id] - (audio @ text[s.id + AGAINST] if s.against else 0.0)
        f = nova.Filters(vocal=s.vocal, bpm=s.bpm, energy=s.energy)
        if s.styles:
            inside = np.array([lib.tracks[int(t)].style in s.styles for t in ids])
            if not inside.any():
                continue
            cut = np.quantile(sims[inside], 1 - STYLE_SHARE)
        else:
            inside = np.ones(len(ids), bool)
            cut = np.quantile(sims, 1 - TOP_SHARE)
        members = [int(t) for t, x, ok in zip(ids, sims, inside) if ok and x >= cut and nova._filter_ok(lib, int(t), f)]
        out[s.id] = {"title": s.title, "prompt": s.prompt, "tracks": members}
    from arxnova import models

    p = cfg.index_dir / models.EFFNET.id / "scenarios.json"
    p.write_text(json.dumps(out, ensure_ascii=False))
    return p
