"""Паритет мел-спектра с Essentia (эталон снят TensorflowInputMusiCNN + FrameGenerator,
startFromZero=True; Essentia — AGPL, в пакет не входит, эталон лежит в tests/data)."""

from pathlib import Path

import numpy as np

from arxnova import features as F

DATA = Path(__file__).parent / "data"


def _ref():
    return np.load(DATA / "essentia_musicnn_input.npy"), np.load(DATA / "essentia_musicnn_mel.npy")


def test_melspec_matches_essentia():
    x, ref = _ref()
    mel = F.melspec(x)
    assert mel.shape == ref.shape  # и число кадров совпадает
    # красное: нормировка «сумма весов = 1» вместо unit_tri — расхождение заметно
    fb_wrong = F._mel_matrix() / F._mel_matrix().sum(axis=1, keepdims=True)
    fr = F.frames(x).astype(np.float64) * F._window()
    wrong = np.log10(1 + 10000 * (np.abs(np.fft.rfft(fr, axis=1)) ** 2 @ fb_wrong.T))
    assert np.abs(wrong - ref).max() > 0.5
    # зелёное
    assert np.abs(mel - ref).max() < 1e-3


def test_patches_shape():
    mel = np.zeros((1000, 96), dtype=np.float32)
    p = F.patches(mel)
    assert p.shape == (len(range(0, 1000 - 128 + 1, 62)), 128, 96)
    assert F.patches(np.zeros((50, 96), dtype=np.float32)).shape == (1, 128, 96)
