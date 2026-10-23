"""Эмбеддинг полного микса через ONNX Runtime (ТЗ, ФТ-2.2)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

import arxnova  # noqa: F401  — телеметрия ONNX выключается до импорта onnxruntime
from arxnova import features as F

# На Pi пачка больше 16 окон не ускоряет, но почти удваивает память (ТЗ, §4.1).
MAX_BATCH = 16


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
        x = F.load_mono(path, F.SR_EFFNET, start_s, duration_s)
        return self.run(F.patches(F.melspec(x)))
