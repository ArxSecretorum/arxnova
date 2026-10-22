"""Фасеты звука (ФТ-2.3, ФТ-2.4) — только на настольной машине (ОП-5, дополнение `[stems]`).

Трек раскладывается Demucs на стемы, и по ним считаются:
- эмбеддинги фасетов той же моделью, что и микс (EffNet): «ритм» — ударные, «инструментовка» —
  бас и прочее, «вокал» — вокал. Шлюз М6 сравнивает каждый фасет отдельно (ФТ-6.2);
- явные признаки: темп (Beat This!, а если его веса недоступны — автокорреляция по ударным),
  тональность (хрома по гармоническим стемам + профили Крумхансла), громкость и плотность
  ударов (энергия), доля вокала (по энергии стема «вокал», отдельная модель не нужна).

Аудио стемов нигде не сохраняется: только векторы и числа в базе (§5.6)."""

from __future__ import annotations

import arxnova  # noqa: F401  — телеметрия выключается до импорта onnxruntime

from dataclasses import dataclass

import numpy as np

from arxnova import features as F

FACETS = ("rhythm", "instr", "vocal")
SR = 44100  # частота Demucs
VERSION = 1  # меняется — признаки пересчитываются

# Профили тональностей Крумхансла — Кесслер (мажор, минор), от тоники
_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
# Camelot: номер колеса для мажора (B) и минора (A) по тонике C, C#, …, B
_CAMELOT_MAJOR = [8, 3, 10, 5, 12, 7, 2, 9, 4, 11, 6, 1]
_CAMELOT_MINOR = [5, 12, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10]
NOTE = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


@dataclass
class TrackFacets:
    vectors: dict[str, np.ndarray]  # фасет → эмбеддинг
    bpm: float
    bpm_conf: float
    bpm_src: str  # beat_this | autocorr
    key: int  # 0–11 мажор от C, 12–23 минор от C
    key_conf: float
    loudness_db: float
    onset_rate: float  # ударов в секунду по стему ударных
    vocal_frac: float  # доля звучащего времени, где слышен вокал
    vocal_ratio: float  # доля энергии вокала в миксе


def camelot(key: int) -> str:
    return f"{_CAMELOT_MINOR[key - 12]}A" if key >= 12 else f"{_CAMELOT_MAJOR[key]}B"


def key_name(key: int) -> str:
    return NOTE[key % 12] + ("m" if key >= 12 else "")


def load_stereo(path, start_s: float = 0.0, duration_s: float | None = None) -> np.ndarray:
    """FLAC → стерео float32 44,1 кГц, форма (2, n). Кусками, как load_mono: хайрез целиком в
    памяти — это сотни мегабайт."""
    import soundfile as sf
    import soxr

    with sf.SoundFile(str(path)) as f:
        file_sr = f.samplerate
        start = int(start_s * file_sr)
        left = f.frames - start if duration_s is None else min(int(duration_s * file_sr), f.frames - start)
        if start:
            f.seek(start)
        out = np.empty((int(left * SR / file_sr) + 4096, 2), np.float32)
        filled = 0
        rs = soxr.ResampleStream(file_sr, SR, 2, dtype="float32", quality="HQ") if file_sr != SR else None
        block = max(4096, 2 * file_sr)
        while left > 0:
            data = f.read(min(block, left), dtype="float32", always_2d=True)
            if not len(data):
                break
            left -= len(data)
            st = np.repeat(data[:, :1], 2, axis=1) if data.shape[1] == 1 else data[:, :2]
            chunk = rs.resample_chunk(np.ascontiguousarray(st), last=left <= 0) if rs else st
            out[filled:filled + len(chunk)] = chunk
            filled += len(chunk)
    return np.ascontiguousarray(out[:filled].T)


def _mono16k(stereo: np.ndarray) -> np.ndarray:
    import soxr

    return soxr.resample(stereo.mean(axis=0), SR, F.SR_EFFNET, quality="HQ").astype(np.float32)


def _rms_frames(x: np.ndarray, sr: int, sec: float = 1.0) -> np.ndarray:
    n = int(sr * sec)
    k = len(x) // n
    if k == 0:
        return np.array([np.sqrt(np.mean(x ** 2) + 1e-12)])
    return np.sqrt(np.mean(x[: k * n].reshape(k, n) ** 2, axis=1) + 1e-12)


def vocal_stats(vocals: np.ndarray, mix: np.ndarray) -> tuple[float, float]:
    """(доля звучащих секунд, где вокал не тише микса на 12 дБ; доля энергии вокала)."""
    v = _rms_frames(vocals.mean(axis=0), SR)
    m = _rms_frames(mix.mean(axis=0), SR)
    loud = m > 10 ** (-45 / 20)  # тишину между треками и в паузах не считаем
    if not loud.any():
        return 0.0, 0.0
    frac = float(np.mean((v[loud] / m[loud]) > 10 ** (-12 / 20)))
    ratio = float(np.sum(vocals ** 2) / (np.sum(mix ** 2) + 1e-12))
    return frac, ratio


def _onset_envelope(x: np.ndarray, sr: int, hop: int = 512) -> tuple[np.ndarray, float]:
    """Спектральный поток (положительные приращения лог-спектра) — огибающая ударов."""
    n_fft = 2048
    k = 1 + max(0, (len(x) - n_fft) // hop)
    if k < 4:
        return np.zeros(1), sr / hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(k)[:, None]
    spec = np.abs(np.fft.rfft(x[idx] * np.hanning(n_fft), axis=1))
    logs = np.log1p(100 * spec)
    flux = np.maximum(0.0, np.diff(logs, axis=0)).sum(axis=1)
    return flux - flux.mean(), sr / hop


def tempo_autocorr(drums_mono: np.ndarray, sr: int) -> tuple[float, float]:
    """Темп по автокорреляции огибающей ударов в диапазоне 70–180 BPM: (BPM, уверенность 0–1)."""
    env, fps = _onset_envelope(drums_mono, sr)
    if len(env) < 64:
        return 0.0, 0.0
    ac = np.correlate(env, env, mode="full")[len(env) - 1:]
    ac /= ac[0] + 1e-12
    lags = np.arange(len(ac))
    lo, hi = int(fps * 60 / 180), int(fps * 60 / 70)
    if hi >= len(ac):
        return 0.0, 0.0
    # сумма гармоник лага, чтобы не путать темп с его половиной
    score = ac[lo:hi + 1] + 0.5 * np.interp(2 * lags[lo:hi + 1], lags, ac)
    best = lo + int(np.argmax(score))
    return float(60 * fps / best), float(max(0.0, min(1.0, ac[best])))


def onset_rate(drums_mono: np.ndarray, sr: int) -> float:
    env, fps = _onset_envelope(drums_mono, sr)
    if len(env) < 3:
        return 0.0
    thr = env.std() * 1.0
    peaks = (env[1:-1] > env[:-2]) & (env[1:-1] >= env[2:]) & (env[1:-1] > thr)
    return float(peaks.sum() / (len(env) / fps))


def key_estimate(harmonic_mono16k: np.ndarray) -> tuple[int, float]:
    """Тональность по средней хроме: (0–11 мажор / 12–23 минор, уверенность — отрыв лучшей
    корреляции от второй)."""
    sr, n_fft, hop = F.SR_EFFNET, 8192, 4096
    x = harmonic_mono16k
    k = 1 + max(0, (len(x) - n_fft) // hop)
    if k < 2:
        return 0, 0.0
    chroma = np.zeros(12)
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    band = (freqs >= 55) & (freqs <= 2000)
    pc = (np.round(12 * np.log2(freqs[band] / 440.0)) + 9).astype(int) % 12  # 0 = C
    win = np.hanning(n_fft)
    for s in range(0, k, 64):  # пачками окон — без лишней памяти
        idx = np.arange(n_fft)[None, :] + hop * np.arange(s, min(k, s + 64))[:, None]
        mag = np.abs(np.fft.rfft(x[idx] * win, axis=1))[:, band]
        mag = np.log1p(1000 * mag / (mag.max(axis=1, keepdims=True) + 1e-9))
        np.add.at(chroma, pc, mag.sum(axis=0))
    chroma = (chroma - chroma.mean()) / (chroma.std() + 1e-9)
    corr = []
    for prof in (_MAJOR, _MINOR):
        p = (prof - prof.mean()) / prof.std()
        corr += [float(np.dot(chroma, np.roll(p, t)) / 12) for t in range(12)]
    order = np.argsort(corr)[::-1]
    return int(order[0]), float(corr[order[0]] - corr[order[1]])


class Analyzer:
    """Demucs + EffNet + темп. Держит модели загруженными на весь проход."""

    def __init__(self, effnet, device: str | None = None):
        import torch
        from demucs.pretrained import get_model

        self.torch = torch
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.demucs = get_model("htdemucs")
        self.demucs.eval()
        self.effnet = effnet
        self.beats = None
        try:
            from beat_this.inference import Audio2Beats

            self.beats = Audio2Beats(checkpoint_path="final0", device=self.device, dbn=False)
        except Exception:  # веса Beat This! недоступны — темп по ударным (tempo_autocorr)
            self.beats = None

    def separate(self, stereo: np.ndarray) -> dict[str, np.ndarray]:
        from demucs.apply import apply_model

        torch = self.torch
        wav = torch.from_numpy(stereo)
        ref = wav.mean(0)
        mean, std = ref.mean(), ref.std() + 1e-8
        with torch.no_grad():
            out = apply_model(self.demucs, ((wav - mean) / std)[None], device=self.device,
                              shifts=0, split=True, overlap=0.25, progress=False)[0]
        out = (out * std + mean).cpu().numpy()
        return {name: out[i] for i, name in enumerate(self.demucs.sources)}

    def tempo(self, stereo: np.ndarray, drums: np.ndarray) -> tuple[float, float, str]:
        if self.beats is not None:
            try:
                beats, _ = self.beats(stereo.T, SR)
                if len(beats) > 8:
                    ibi = np.diff(beats)
                    med = float(np.median(ibi))
                    conf = float(np.mean(np.abs(ibi - med) < 0.08 * med))  # ровность долей
                    return 60.0 / med, conf, "beat_this"
            except Exception:
                pass
        bpm, conf = tempo_autocorr(drums.mean(axis=0), SR)
        return bpm, conf, "autocorr"

    def analyze(self, stereo: np.ndarray) -> TrackFacets:
        st = self.separate(stereo)
        mix = stereo
        parts = {
            "rhythm": st["drums"],
            "instr": st["bass"] + st["other"],
            "vocal": st["vocals"],
        }
        vectors = {}
        for name, audio in parts.items():
            m = _mono16k(audio)
            vectors[name] = self.effnet.run(F.patches(F.melspec(m))).vector.astype(np.float32)
        bpm, bpm_conf, src = self.tempo(stereo, st["drums"])
        key, key_conf = key_estimate(_mono16k(st["bass"] + st["other"] + st["vocals"]))
        rms = float(np.sqrt(np.mean(mix ** 2)) + 1e-12)
        frac, ratio = vocal_stats(st["vocals"], mix)
        return TrackFacets(
            vectors, float(bpm), bpm_conf, src, key, key_conf, float(20 * np.log10(rms)),
            onset_rate(st["drums"].mean(axis=0), SR), frac, ratio,
        )
