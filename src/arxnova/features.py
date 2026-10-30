"""Предобработка звука для моделей. Один и тот же код на всех узлах — на этом держится
правило совместимости эмбеддингов (ТЗ, §3, ФТ-2.8).

Мел-спектр повторяет TensorflowInputMusiCNN из Essentia (вход Discogs-EffNet), но без
самой Essentia: она под AGPL и ставится не везде. Совпадение с эталоном проверяется тестом
на машине разработчика (tests/test_features_parity.py).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

SR_EFFNET = 16000
FRAME = 512
HOP = 256
N_MELS = 96
PATCH = 128  # кадров в окне модели
PATCH_HOP = 62  # сдвиг окна, как в TensorflowPredictEffnetDiscogs

# Эмбеддинг микса — по отрывкам, а не по целому треку: 10 кусков по 6 с, равномерно внутри
# 10–90 % длины. На 1500 треках медиатеки владельца (30.10.2026) соседи по стилю, исполнителю,
# альбому и темпу те же, что у целого трека (0,531 / 0,218 / 0,187 / 0,684 против
# 0,532 / 0,224 / 0,195 / 0,685), а на процессоре это 0,17 с вместо 1,35 с на трек (в 8 раз).
EXCERPTS = (10, 6.0)  # (сколько, секунд каждый)
EXCERPT_MARGIN = 0.1  # не берём первые и последние 10 % трека (тишина, вступление, затухание)


def excerpt_starts(duration_s: float, count: int = EXCERPTS[0], length_s: float = EXCERPTS[1]) -> list[float]:
    """Начала отрывков (с от начала трека). Трек короче отрывка — один отрывок с нуля; короткий
    трек — отрывки перекрываются, но все лежат внутри него."""
    if duration_s <= length_s:
        return [0.0]
    a = duration_s * EXCERPT_MARGIN
    b = max(a, duration_s * (1 - EXCERPT_MARGIN) - length_s)
    if count <= 1:
        return [(a + b) / 2]
    return [a + (b - a) * (i + 1) / (count + 1) for i in range(count)]


def load_mono(path, sr: int, start_s: float = 0.0, duration_s: float | None = None) -> np.ndarray:
    """FLAC → моно float32 с частотой sr (soxr, качество HQ). start_s/duration_s — участок
    файла (трек CUE-образа): читается только он."""
    import soundfile as sf
    import soxr

    # Кусками по ~2 с: трек целиком в исходном качестве (24/192, 8 минут) — это сотни МБ,
    # а пересчёт частоты разом раздувал пик до гигабайт (бюджет Pi — 768 МБ, ОП-2).
    # Потоковый soxr даёт тот же результат, что и пересчёт целиком.
    out: np.ndarray | None = None
    filled = 0

    def put(chunk: np.ndarray) -> None:
        nonlocal out, filled
        if out is None or filled + len(chunk) > len(out):
            grow = np.empty(max(len(chunk) + filled, int((len(out) if out is not None else 0) * 1.25) + sr * 10), np.float32)
            if out is not None:
                grow[:filled] = out[:filled]
            out = grow
        out[filled:filled + len(chunk)] = chunk
        filled += len(chunk)

    with sf.SoundFile(str(path)) as f:
        file_sr = f.samplerate
        start = int(start_s * file_sr)
        left = None if duration_s is None else int(duration_s * file_sr)
        if start:
            f.seek(start)
        total = (f.frames - start) if left is None else min(left, f.frames - start)
        # буфер сразу нужного размера (+запас на хвост фильтра): склейка кусков удваивала пик
        out = np.empty(int(total * sr / file_sr) + 4096, np.float32)
        rs = soxr.ResampleStream(file_sr, sr, 1, dtype="float32", quality="HQ") if file_sr != sr else None
        block = max(4096, 2 * file_sr)
        while left is None or left > 0:
            n = block if left is None else min(block, left)
            data = f.read(n, dtype="float32", always_2d=True)
            last = len(data) < n or (left is not None and left - len(data) <= 0)
            if len(data):
                mono = data.mean(axis=1, dtype=np.float32)
                put(rs.resample_chunk(mono, last=last) if rs else mono)
            elif rs:
                put(rs.resample_chunk(np.zeros(0, np.float32), last=True))
            if left is not None:
                left -= len(data)
            if last:
                break
    return out[:filled] if out is not None else np.zeros(0, np.float32)


def _hz_to_mel_slaney(f):
    f = np.asarray(f, dtype=np.float64)
    f_sp = 200.0 / 3
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    return np.where(f < min_log_hz, f / f_sp, min_log_mel + np.log(np.maximum(f, 1e-10) / min_log_hz) / logstep)


def _mel_to_hz_slaney(m):
    m = np.asarray(m, dtype=np.float64)
    f_sp = 200.0 / 3
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    return np.where(m < min_log_mel, m * f_sp, min_log_hz * np.exp(logstep * (m - min_log_mel)))


@lru_cache(maxsize=4)
def _mel_matrix(sr: int = SR_EFFNET, n_fft: int = FRAME, n_mels: int = N_MELS) -> np.ndarray:
    """Треугольные фильтры по шкале Слейни до sr/2. Нормировка unit_tri как в Essentia:
    единичная площадь в герцах, то есть вершина 2/(hi−lo)."""
    n_bins = n_fft // 2 + 1
    freqs = np.arange(n_bins) * sr / n_fft
    mel_edges = np.linspace(_hz_to_mel_slaney(0.0), _hz_to_mel_slaney(sr / 2), n_mels + 2)
    hz = _mel_to_hz_slaney(mel_edges)
    fb = np.zeros((n_mels, n_bins), dtype=np.float64)
    for i in range(n_mels):
        lo, c, hi = hz[i], hz[i + 1], hz[i + 2]
        tri = np.maximum(0.0, np.minimum((freqs - lo) / (c - lo), (hi - freqs) / (hi - c)))
        fb[i] = tri * 2.0 / (hi - lo)
    return fb


@lru_cache(maxsize=2)
def _window(n: int = FRAME) -> np.ndarray:
    """Симметричное окно Ханна без нормировки (Windowing(type='hann', normalized=False))."""
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / (n - 1))


def frames(x: np.ndarray, frame: int = FRAME, hop: int = HOP) -> np.ndarray:
    """Нарезка как FrameGenerator(startFromZero=True) в Essentia: кадр начинается с 0 и
    берётся, пока его середина внутри сигнала; хвост дополняется нулями."""
    n = len(x)
    if n == 0:
        return np.zeros((0, frame), dtype=np.float32)
    count = max(1, int(np.ceil((n - frame // 2) / hop)))
    padded = np.zeros((count - 1) * hop + frame, dtype=np.float32)
    padded[: min(n, len(padded))] = x[: len(padded)]
    # окна без копии (вид на тот же буфер): таблица индексов и копия кадров весили гигабайт
    return np.lib.stride_tricks.sliding_window_view(padded, frame)[::hop][:count]


MEL_BLOCK = 4096  # кадров за раз (~16 МБ в float64)


def melspec(x: np.ndarray) -> np.ndarray:
    """(кадры × 96) лог-мел для Discogs-EffNet: |FFT|², мел-фильтры, log10(1 + 10000·x).
    Считается в float64 и отдаётся во float32 — так совпадение с эталоном не зависит от
    порядка суммирования на разных процессорах."""
    n = len(x)
    count = 0 if n == 0 else max(1, int(np.ceil((n - FRAME // 2) / HOP)))
    win, mm = _window(), _mel_matrix().T
    out = np.empty((count, mm.shape[1]), dtype=np.float32)
    # кадры независимы — считаем блоками прямо по сигналу (хвост блока дополняется нулями,
    # как в frames()); ни копии сигнала, ни всех кадров сразу в двойной точности
    for i in range(0, count, MEL_BLOCK):
        k = min(MEL_BLOCK, count - i)
        seg = x[i * HOP : i * HOP + (k - 1) * HOP + FRAME]
        if len(seg) < (k - 1) * HOP + FRAME:
            seg = np.concatenate([seg, np.zeros((k - 1) * HOP + FRAME - len(seg), np.float32)])
        fr = np.lib.stride_tricks.sliding_window_view(seg, FRAME)[::HOP][:k].astype(np.float64) * win
        power = np.abs(np.fft.rfft(fr, axis=1)) ** 2
        out[i : i + k] = np.log10(1.0 + 10000.0 * (power @ mm))
    return out


def patches(mel: np.ndarray, size: int = PATCH, hop: int = PATCH_HOP) -> np.ndarray:
    """Окна (n × 128 × 96); хвост короче окна отбрасывается, кроме случая, когда трек
    короче одного окна — тогда он дополняется нулями."""
    if len(mel) < size:
        out = np.zeros((1, size, mel.shape[1]), dtype=np.float32)
        out[0, : len(mel)] = mel
        return out
    # вид без копии: окна берутся пачками при прогоне сети (embed.run)
    return np.lib.stride_tricks.sliding_window_view(mel, size, axis=0)[::hop].transpose(0, 2, 1)
