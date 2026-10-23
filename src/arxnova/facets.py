"""Фасеты звука (ФТ-2.3, ФТ-2.4) — только на настольной машине (ОП-5, дополнение `[stems]`).

Трек раскладывается Demucs на стемы, и по ним считаются:
- эмбеддинги фасетов той же моделью, что и микс (EffNet): «ритм» — ударные, «инструментовка» —
  бас и прочее, «вокал» — вокал. Шлюз М6 сравнивает каждый фасет отдельно (ФТ-6.2);
- явные признаки: темп (Beat This!, а если его веса недоступны — автокорреляция по ударным),
  тональность (хрома по гармоническим стемам + профили Крумхансла), громкость и плотность
  ударов (энергия), доля вокала (по энергии стема «вокал», отдельная модель не нужна).

Аудио стемов нигде не сохраняется: только векторы и числа в базе (§5.6)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import arxnova  # noqa: F401  — телеметрия выключается до импорта onnxruntime
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
    bpm_src: str  # beat_this | autocorr (весов не было) | autocorr_only (долей не нашлось)
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


# Перекрытие окон Demucs. 0,10 вместо стандартных 0,25 — на 9 % быстрее, а фасеты те же: на
# 8 треках косинус 1,000 и 10 из 10 ближайших соседей совпадают по ритму, инструментам и вокалу
# (23.10.2026). Лёгких замен Demucs с тем же качеством нет: Open-Unmix не быстрее, а соседи
# совпадают лишь на 0–2 из 10; половинная точность на MPS скорости не даёт.
OVERLAP = 0.10
BATCH = 8  # кусков за проход на GPU; 16 уже не быстрее, 32 — вдвое медленнее (M-серия)


# Веса Beat This! (CPJKU, MIT). Сервер авторов (cloud.cp.jku.at) отдаёт 503 с осени 2026 —
# берём ту же final0 с зеркала на Hugging Face в safetensors (без исполняемого кода) и сверяем
# sha256; проверено: темп совпадает с независимым по ударным в 14 из 15 треков.
BEATS_URL = "https://huggingface.co/safe-models/beat-this-final0/resolve/main/model.safetensors"
BEATS_SHA256 = "f758c77ba5d862cad3718a95705964e9af55298b1fb3544c62c9d79184ea0f2e"
BEATS_HPARAMS = {"spect_dim": 128, "transformer_dim": 512, "ff_mult": 4, "n_layers": 6, "head_dim": 32,
                 "stem_dim": 32, "dropout": {"frontend": 0.1, "transformer": 0.2}, "sum_head": True,
                 "partial_transformers": True}


def ensure_beats(models_dir) -> str | None:
    """Чекпойнт Beat This! в формате библиотеки; None — не скачался (темп тогда по ударным)."""
    import hashlib
    import urllib.request
    from pathlib import Path

    ckpt = Path(models_dir) / "beat_this-final0.ckpt"
    if ckpt.exists():
        return str(ckpt)
    failed = Path(models_dir) / "beat_this.failed"
    try:
        import torch
        from beat_this.model.beat_tracker import BeatThis
        from safetensors.torch import load

        with urllib.request.urlopen(BEATS_URL, timeout=120) as r:
            blob = r.read()
        if hashlib.sha256(blob).hexdigest() != BEATS_SHA256:
            raise ValueError("sha256")
        sd = load(blob)
        full = BeatThis(**BEATS_HPARAMS).state_dict()  # частоты rotary — не веса, safetensors их не хранит
        full.update(sd)
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        tmp = ckpt.with_suffix(".part")
        torch.save({"hyper_parameters": BEATS_HPARAMS, "state_dict": {"model." + k: v for k, v in full.items()}}, tmp)
        tmp.replace(ckpt)
        failed.unlink(missing_ok=True)
        return str(ckpt)
    except Exception:
        failed.parent.mkdir(parents=True, exist_ok=True)
        failed.write_text(str(int(__import__("time").time())))  # не дёргать сеть на каждом проходе
        return None


def beats_worth_trying(models_dir) -> bool:
    """Темп по долям доступен или его стоит попробовать снова (раз в неделю после неудачи)."""
    import time
    from pathlib import Path

    d = Path(models_dir)
    if (d / "beat_this-final0.ckpt").exists():
        return True
    f = d / "beat_this.failed"
    return not f.exists() or time.time() - f.stat().st_mtime > 7 * 86400


BEATS_MIN_CONF = 0.5
BPM_LOW = 70  # диапазон темпа после свёртки: [70, 200)


def tempo_from_beats(beats, stereo: np.ndarray) -> tuple[float, float] | None:
    """(темп, ровность долей) по Beat This!; None — долей не нашлось (эмбиент, речь)."""
    try:
        b, _ = beats(stereo.T, SR)
    except Exception:
        return None
    if len(b) <= 8:
        return None
    ibi = np.diff(b)
    med = float(np.median(ibi))
    even = np.abs(ibi - med) < 0.08 * med
    conf = float(np.mean(even))
    if conf < BEATS_MIN_CONF:  # доли гуляют — такому темпу верить нельзя, остаётся темп по ударным
        return None
    # среднее по ровным долям, не медиана: доли стоят на кадрах по 20 мс, медиана округляет
    # темп ступеньками (129 → 130,4)
    bpm = 60.0 / float(np.mean(ibi[even]))
    while bpm < BPM_LOW:  # без DBN модель иногда ловит каждую вторую долю (60 вместо 120) или
        bpm *= 2  # дробит пополам (217 вместо 108); сворачиваем в привычный диапазон
    while bpm >= BPM_LOW * 2 + 60:
        bpm /= 2
    return bpm, conf


class Analyzer:
    """Demucs + EffNet + темп. Держит модели загруженными на весь проход."""

    def __init__(self, effnet, device: str | None = None, beats_ckpt: str | None = None):
        import torch
        from demucs.pretrained import get_model

        self.torch = torch
        # на процессоре torch — STFT Demucs и Beat This!; по умолчанию он берёт все ядра, а рядом
        # EffNet со своими потоками: Mac часами грелся на 450 % (23.10.2026). 4 потока — почти та же
        # скорость: основное время — на GPU
        torch.set_num_threads(min(4, torch.get_num_threads()))
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.demucs = get_model("htdemucs")
        self.demucs.eval()
        self.effnet = effnet
        # GPU (MPS) — по одному: темп по долям считается в потоке описания, пока Demucs разделяет
        # следующий трек; два потока в одном Metal-буфере команд роняют процесс («A command encoder
        # is already encoding to this command buffer», 22.10.2026)
        import threading

        self.gpu = threading.Lock()
        self.beats = None
        try:
            from beat_this.inference import Audio2Beats

            if beats_ckpt:
                self.beats = Audio2Beats(checkpoint_path=beats_ckpt, device=self.device, dbn=False)
        except Exception:  # веса Beat This! недоступны — темп по ударным (tempo_autocorr)
            self.beats = None

    def separate(self, stereo: np.ndarray) -> dict[str, np.ndarray]:
        """Стемы. Куски по 7,8 с идут на GPU пачками по BATCH, а не по одному, как в apply_model:
        на M-серии это 7,5 с вместо 10,6 с на 5-минутный трек при том же результате (те же окна,
        то же перекрытие и треугольное сшивание)."""
        torch = self.torch
        wav = torch.from_numpy(stereo)
        ref = wav.mean(0)
        mean, std = ref.mean(), ref.std() + 1e-8
        wav = (wav - mean) / std
        bag = getattr(self.demucs, "models", None)
        with self.gpu:
            if bag is not None and len(bag) != 1:  # ансамбль — штатным путём
                from demucs.apply import apply_model

                with torch.no_grad():
                    out = apply_model(self.demucs, wav[None], device=self.device, shifts=0, split=True,
                                      overlap=OVERLAP, progress=False)[0]
            else:
                out = self._batched(bag[0] if bag else self.demucs, wav)
            out = (out * std + mean).cpu().numpy()
            if self.device == "mps":
                torch.mps.empty_cache()
        return {name: out[i] for i, name in enumerate(self.demucs.sources)}

    def _batched(self, model, wav):
        torch = self.torch
        model.to(self.device)
        seg = int(float(model.segment) * model.samplerate)
        stride = int(seg * (1 - OVERLAP))
        n = wav.shape[1]
        starts = list(range(0, max(n - seg, 0) + stride, stride))
        half = seg // 2
        w = torch.cat([torch.arange(1, half + 1), torch.arange(seg - half, 0, -1)]).float()
        w = w / w.max()
        out = torch.zeros(len(model.sources), wav.shape[0], n)
        norm = torch.zeros(n)
        with torch.no_grad():
            for i in range(0, len(starts), BATCH):
                group = starts[i:i + BATCH]
                # пачка всегда полная (хвост — нулями): форма входа одна на весь проход, и GPU
                # переиспользует свои буферы. С хвостом своего размера у каждого трека буферы
                # копились — 20 ГБ видеопамяти за 3000 треков (23.10.2026)
                x = torch.zeros(BATCH, wav.shape[0], seg)
                for j, s in enumerate(group):
                    piece = wav[:, s:s + seg]
                    x[j, :, :piece.shape[1]] = piece
                y = model(x.to(self.device)).cpu()
                for j, s in enumerate(group):
                    k = min(seg, n - s)
                    out[..., s:s + k] += y[j, ..., :k] * w[:k]
                    norm[s:s + k] += w[:k]
        return out / norm.clamp_min(1e-8)

    def tempo(self, stereo: np.ndarray, drums: np.ndarray) -> tuple[float, float, str]:
        got = None
        if self.beats is not None:
            with self.gpu:
                got = tempo_from_beats(self.beats, stereo)
        if got:
            return got[0], got[1], "beat_this"
        bpm, conf = tempo_autocorr(drums.mean(axis=0), SR)
        return bpm, conf, "autocorr" if self.beats is None else "autocorr_only"

    def analyze(self, stereo: np.ndarray) -> TrackFacets:
        return self.describe(stereo, self.separate(stereo))

    def describe(self, stereo: np.ndarray, st: dict[str, np.ndarray]) -> TrackFacets:
        """Всё, что после разделения, — на процессоре; проход гонит это параллельно со стемами
        следующего трека на GPU."""
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
