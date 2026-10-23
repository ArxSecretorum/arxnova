"""MAEST-519 (MTG, ISMIR 2023): стили трека по 519 классам Discogs — точнее Discogs-EffNet (на
медиатеке владельца: GoGo Penguin — Contemporary Jazz / Fusion вместо Ambient / House у EffNet,
Max Richter — Choral / Contemporary). Вход — тот же лог-мел, что у EffNet (96 полос, 16 кГц),
но нормированный, как в TensorflowPredictMAEST; окно — 30 с (1876 кадров). На трек — до трёх окон
(25 / 50 / 75 % длины), активации и эмбеддинг усредняются. На процессоре ~1 с на окно (Mac M-серии,
CoreML медленнее)."""

from __future__ import annotations

import arxnova  # noqa: F401  — телеметрия выключается до импорта onnxruntime

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from arxnova import features as F

FRAMES = 1876  # 30 с при шаге 256 на 16 кГц
MEAN, STD = 2.06755686098554, 1.268292820667291  # нормировка входа MAEST (Essentia)
WINDOWS = (0.25, 0.5, 0.75)
EMB_LAYER = "layer_06_embeddings"  # 7-й слой — «embeddings» в карточке модели


@dataclass
class MaestResult:
    activations: np.ndarray  # (519,)
    embedding: np.ndarray    # (768,)
    windows: int


class Maest:
    """На Mac с GPU — та же модель, переведённая из ONNX в PyTorch (onnx2torch, расхождение ~1e-6),
    на MPS: 0,16 с на окно против 1,2 с на процессоре (CoreML медленнее процессора). Без torch или
    GPU — onnxruntime на процессоре."""

    def __init__(self, onnx_path: Path, threads: int = 4, device: str | None = None):
        self.torch_net = None
        try:
            import onnx
            import torch
            from onnx2torch import convert

            dev = device or ("mps" if torch.backends.mps.is_available() else
                             "cuda" if torch.cuda.is_available() else None)
            if dev:
                self.torch, self.device = torch, dev
                self.torch_net = convert(onnx.load(str(onnx_path))).eval().to(dev)
                names = [o.name for o in onnx.load(str(onnx_path), load_external_data=False).graph.output]
                self._ia, self._ie = names.index("activations"), names.index(EMB_LAYER)
        except Exception:
            self.torch_net = None
        if self.torch_net is None:
            import onnxruntime as ort

            so = ort.SessionOptions()
            so.intra_op_num_threads = threads
            self.session = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])

    def _infer(self, inp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.torch_net is not None:
            with self.torch.no_grad():
                out = self.torch_net(self.torch.from_numpy(inp).to(self.device))
            return out[self._ia].cpu().numpy(), out[self._ie].cpu().numpy()
        a, e = self.session.run(["activations", EMB_LAYER], {"melspectrogram": inp})
        return a, e

    def run_mel(self, mel: np.ndarray) -> MaestResult:
        """mel — лог-мел трека (кадры × 96), как features.melspec."""
        n = len(mel)
        if n <= FRAMES:
            starts = [0]
        else:
            starts = sorted({min(n - FRAMES, max(0, int(n * w) - FRAMES // 2)) for w in WINDOWS})
        acts, embs = [], []
        for s in starts:
            m = mel[s:s + FRAMES]
            if len(m) < FRAMES:
                m = np.pad(m, ((0, FRAMES - len(m)), (0, 0)), mode="wrap") if len(m) else np.zeros((FRAMES, 96))
            inp = ((m - MEAN) / (STD * 2)).astype(np.float32)[None]
            a, e = self._infer(inp)
            acts.append(a[0])
            embs.append(e[0].mean(axis=0))
        return MaestResult(np.mean(acts, axis=0), np.mean(embs, axis=0), len(starts))

    def run_file(self, path, start_s: float, duration_s: float) -> MaestResult:
        """Декодируются только окна (длинный микс целиком не читается). start_s / duration_s —
        границы трека в файле (лист CUE) или 0 / длина файла."""
        win = FRAMES * F.HOP / F.SR_EFFNET + 0.1
        if duration_s <= win:
            return self.run_mel(F.melspec(F.load_mono(path, F.SR_EFFNET, start_s, duration_s)))
        acts, embs = [], []
        for w in WINDOWS:
            at = start_s + min(max(0.0, duration_s * w - win / 2), duration_s - win)
            r = self.run_mel(F.melspec(F.load_mono(path, F.SR_EFFNET, at, win)))
            acts.append(r.activations)
            embs.append(r.embedding)
        return MaestResult(np.mean(acts, axis=0), np.mean(embs, axis=0), len(WINDOWS))
