"""Отдельный процесс CLAP «текст → звук» (ФТ-4.10, сценарии) — оригинальные веса LAION
`music_audioset_epoch_15_esc_90.14.pt` (CC0). Сборка HuggingFace сломана (этап 0: 8 из 30
против 23 из 30 у оригинала), а пакет `laion_clap` требует numpy < 2 — поэтому он живёт в своём
окружении (uv, см. scenarios.run_worker) и ничего не импортирует из arxnova.

Вход — JSON: {"ckpt": путь, "tracks": [[id, путь, начало_с, длительность_с], …], "texts": [...],
"out": путь.npz}. Выход — npz: ids, audio (N×512), text (T×512), всё нормировано."""

from __future__ import annotations

import json
import sys

import numpy as np
import soundfile as sf
import soxr
import torch

SR = 48000
WIN_S = 10.0  # CLAP смотрит на 10 с; берём три окна (25 %, 50 %, 75 % трека) и усредняем
POINTS = (0.25, 0.5, 0.75)


def windows(path: str, start: float, duration: float | None) -> np.ndarray:
    with sf.SoundFile(path) as f:
        sr = f.samplerate
        total = (duration if duration and duration > 0 else (f.frames / sr - start))
        out = []
        for p in POINTS:
            t0 = start + max(0.0, total * p - WIN_S / 2)
            f.seek(int(t0 * sr))
            x = f.read(int(WIN_S * sr), dtype="float32", always_2d=True).mean(axis=1)
            x = soxr.resample(x, sr, SR) if sr != SR else x
            n = int(WIN_S * SR)
            out.append(np.pad(x[:n], (0, max(0, n - len(x)))))
    return np.stack(out).astype(np.float32)


def main() -> None:
    job = json.loads(open(sys.argv[1]).read())
    import laion_clap

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base", device=device)
    model.load_ckpt(job["ckpt"], verbose=False)
    ids, embs = [], []
    for i, (tid, path, start, dur) in enumerate(job["tracks"]):
        try:
            w = windows(path, float(start or 0), dur)
            with torch.no_grad():
                e = model.get_audio_embedding_from_data(x=w, use_tensor=False)
        except Exception as exc:  # битый файл не останавливает проход
            print(f"пропуск {path}: {exc}", flush=True)
            continue
        v = e.mean(axis=0)
        embs.append(v / (np.linalg.norm(v) + 1e-9))
        ids.append(tid)
        if (i + 1) % 25 == 0:
            print(f"  … CLAP {i + 1}/{len(job['tracks'])}", flush=True)
    text = np.zeros((0, 512), np.float32)
    if job.get("texts"):
        with torch.no_grad():
            text = model.get_text_embedding(job["texts"], use_tensor=False)
        text = text / (np.linalg.norm(text, axis=1, keepdims=True) + 1e-9)
    np.savez(job["out"], ids=np.array(ids, np.int64),
             audio=np.array(embs, np.float32).reshape(len(embs), -1), text=np.asarray(text, np.float32))


if __name__ == "__main__":
    main()
