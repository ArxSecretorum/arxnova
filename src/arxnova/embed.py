"""Эмбеддинг полного микса через ONNX Runtime (ТЗ, ФТ-2.2)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

import arxnova  # noqa: F401  — телеметрия ONNX выключается до импорта onnxruntime
from arxnova import features as F

# На Pi пачка больше 16 окон не ускоряет, но почти удваивает память (ТЗ, §4.1).
MAX_BATCH = 16
# Схема предобработки микса — часть идентификатора векторов в базе (правило совместимости §3:
# одна модель, одна предобработка). Целый трек и отрывки в одном индексе не смешиваются.
MIX_SCHEME = f"x{F.EXCERPTS[0]}x{F.EXCERPTS[1]:g}"


@dataclass
class Embedding:
    vector: np.ndarray  # среднее по окнам, 1280 измерений
    styles: np.ndarray  # средние активации 400 стилей Discogs
    windows: int


class EffnetDiscogs:
    def __init__(self, model_path: Path, threads: int = 1):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(model_path), opts, providers=["CPUExecutionProvider"]
        )
        self.input = self.session.get_inputs()[0].name
        outs = [o.name for o in self.session.get_outputs()]
        # выходы по схеме модели: активации стилей и эмбеддинг
        self.out_styles, self.out_embed = outs[0], outs[1]

    def run(self, patches: np.ndarray) -> Embedding:
        emb, sty = [], []
        for i in range(0, len(patches), MAX_BATCH):
            batch = np.ascontiguousarray(patches[i : i + MAX_BATCH], dtype=np.float32)  # окна — вид, копия только пачки
            s, e = self.session.run([self.out_styles, self.out_embed], {self.input: batch})
            sty.append(s)
            emb.append(e)
        emb_all = np.concatenate(emb)
        return Embedding(emb_all.mean(axis=0), np.concatenate(sty).mean(axis=0), len(emb_all))

    def embed_file(self, path, start_s: float = 0.0, duration_s: float | None = None) -> Embedding:
        """Целый трек (или его участок start_s/duration_s — трек CUE-образа)."""
        x = F.load_mono(path, F.SR_EFFNET, start_s, duration_s)
        return self.run(F.patches(F.melspec(x)))

    @staticmethod
    def excerpt_patches(path, duration_s: float, start_s: float = 0.0) -> np.ndarray:
        """Окна модели по отрывкам (features.EXCERPTS): декодируются только они — ни целого
        трека в памяти, ни его мел-спектра. duration_s — длина трека (участка), start_s — его
        начало в файле. Без модели — индексатор готовит их в соседнем потоке, пока сеть занята."""
        return np.concatenate([F.patches(F.melspec(F.load_mono(path, F.SR_EFFNET, start_s + at, F.EXCERPTS[1])))
                               for at in F.excerpt_starts(duration_s)])

    def embed_excerpts(self, path, duration_s: float, start_s: float = 0.0) -> Embedding:
        return self.run(self.excerpt_patches(path, duration_s, start_s))
