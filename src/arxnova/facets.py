"""Признаки звука для шлюза, фильтров и подбора (ФТ-2.3, ФТ-2.4) — на настольной машине (ОП-5, дополнение `[stems]`).

По миксу:
- темп — Beat This! (если его весов нет или ровных долей не нашлось — автокорреляция огибающей ударов);
- громкость и плотность ударов (из них энергия, gate._energy). Плотность по миксу повторяет прежнюю по стему ударных
  (Спирмен 0,93 на 484 треках копии медиатеки).

По выделенному голосу (решение 0014): Demucs разделяет не целый трек, а 8 отрывков ровно по длине своего куска (7,8 с) одной
пачкой на GPU — в 4,7 раза дешевле, а наличие вокала то же, что по целому треку (AUC 0,926 против 0,927 на 1297 треках
с известной правдой).
- доля звучания вокала и доля его энергии (наличие вокала);
- вероятность женского голоса по вектору стема в окнах, где голос звучит; вместе с CLAP выделенного голоса (проход сценариев)
  ошибок в плитках голоса меньше, чем у прежнего слоя на векторе стема целого трека (3,4 % против 4,9 % при равном охвате);
- сам выделенный голос (моно, 32 кГц) — временно, для CLAP: файл удаляется сразу после.

Векторов фасетов (ритм, инструменты, вокал) больше нет: на переходах чужих слушателей между стилями формула по ним угадывала
хуже микса. Аудио в базе не хранится: только числа (§5.6)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import arxnova  # noqa: F401  — телеметрия выключается до импорта onnxruntime
from arxnova import features as F

SR = 44100  # частота, на которой читается звук
VERSION = 2  # 1 — Demucs по целому треку и векторы фасетов; 2 — Demucs по 8 отрывкам, голос, удары по миксу (решение 0014)
EXCERPTS = 8  # отрывков по длине куска Demucs — ровно одна пачка на GPU
VOICE_DB = -12.0  # окно, где голос не тише микса на 12 дБ (как доля звучания вокала)
VOICE_SR = 32000  # выделенный голос для CLAP

# Веса Beat This! (CPJKU, MIT). Сервер авторов (cloud.cp.jku.at) отдаёт 503 с осени 2026, берём ту же final0 с зеркала на Hugging Face
# в safetensors (без исполняемого кода), sha256 проверяем; темп совпадает с независимым по ударным в 14 из 15 случаев.
BEATS_URL = "https://huggingface.co/safe-models/beat-this-final0/resolve/main/model.safetensors"
BEATS_SHA256 = "f758c77ba5d862cad3718a95705964e9af55298b1fb3544c62c9d79184ea0f2e"
BEATS_HPARAMS = {"spect_dim": 128, "transformer_dim": 512, "ff_mult": 4, "n_layers": 6, "head_dim": 32,
                 "stem_dim": 32, "dropout": {"frontend": 0.1, "transformer": 0.2}, "sum_head": True,
                 "partial_transformers": True}


@dataclass
class TrackFacets:
    bpm: float
    bpm_conf: float
    bpm_src: str  # beat_this | autocorr (весов не было) | autocorr_only (долей не нашлось)
    loudness_db: float
    onset_rate: float  # ударов в секунду по миксу
    vocal_frac: float | None = None  # доля звучащих секунд, где голос не тише микса на 12 дБ (без Demucs — неизвестно)
    vocal_ratio: float | None = None  # доля энергии голоса в миксе
    female_stem_p: float | None = None  # вероятность женского голоса по вектору стема (окна с голосом)
    voice: np.ndarray | None = None  # выделенный голос по отрывкам подряд, моно VOICE_SR — для CLAP; в базе не хранится


def excerpt_starts(dur_s: float, length_s: float, k: int = EXCERPTS) -> list[float]:
    """Начала k отрывков, ровно распределённых между 5 % и 95 % трека (вступление и конец почти всегда без голоса)."""
    a, b = dur_s * 0.05, dur_s * 0.95 - length_s
    if b <= a:
        a, b = 0.0, max(0.0, dur_s - length_s)
    return [a + (b - a) * (i + 0.5) / k for i in range(k)]


def vocal_stats(vocals: np.ndarray, mix: np.ndarray) -> tuple[float, float]:
    """(доля звучащих секунд, где вокал не тише микса на 12 дБ; доля энергии вокала). Формы (2, n), SR."""
    k = vocals.shape[1] // SR
    if k == 0:
        return 0.0, 0.0
    v = np.sqrt(np.mean(vocals.mean(axis=0)[: k * SR].reshape(k, SR) ** 2, axis=1) + 1e-12)
    m = np.sqrt(np.mean(mix.mean(axis=0)[: k * SR].reshape(k, SR) ** 2, axis=1) + 1e-12)
    loud = m > 10 ** (-45 / 20)  # тишину между треками и в паузах не считаем
    if not loud.any():
        return 0.0, 0.0
    frac = float(np.mean((v[loud] / m[loud]) > 10 ** (VOICE_DB / 20)))
    ratio = float(np.sum(vocals ** 2) / (np.sum(mix ** 2) + 1e-12))
    return frac, ratio


def patch_windows(n_samples: int) -> list[tuple[int, int]]:
    """Окна EffNet в отсчётах SR_EFFNET: PATCH кадров мел-спектра с шагом PATCH_HOP."""
    n_frames = 1 + max(0, (n_samples - F.FRAME) // F.HOP)
    k = 1 + max(0, (n_frames - F.PATCH) // F.PATCH_HOP)
    return [(i * F.PATCH_HOP * F.HOP, (i * F.PATCH_HOP + F.PATCH) * F.HOP + F.FRAME) for i in range(k)]


def load_stereo(path, start_s: float = 0.0, duration_s: float | None = None) -> np.ndarray:
    """Файл → стерео float32 44,1 кГц, форма (2, n). Кусками, как load_mono: хайрез целиком в
    памяти — это сотни мегабайт."""
    import soundfile as sf
    import soxr

    from arxnova import decode

    if decode.needs_ffmpeg(path):
        return _load_stereo_ffmpeg(path, start_s, duration_s)

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


def _load_stereo_ffmpeg(path, start_s: float, duration_s: float | None) -> np.ndarray:
    """То же для форматов, которые читает только ffmpeg (decode.py)."""
    import soxr

    from arxnova import decode

    rate, _, blocks = decode.read_blocks(path, start_s, duration_s)
    rs = soxr.ResampleStream(rate, SR, 2, dtype="float32", quality="HQ") if rate != SR else None
    parts = []
    for data in blocks:
        st = np.repeat(data[:, :1], 2, axis=1) if data.shape[1] == 1 else data[:, :2]
        parts.append(rs.resample_chunk(np.ascontiguousarray(st), last=False) if rs else st)
    if rs:
        parts.append(rs.resample_chunk(np.zeros((0, 2), np.float32), last=True))
    return np.ascontiguousarray(np.concatenate(parts).T) if parts else np.zeros((2, 0), np.float32)


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


def tempo_autocorr(mono: np.ndarray, sr: int) -> tuple[float, float]:
    """Темп по автокорреляции огибающей ударов в диапазоне 70–180 BPM: (BPM, уверенность 0–1)."""
    env, fps = _onset_envelope(mono, sr)
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


def onset_rate(mono: np.ndarray, sr: int) -> float:
    """Ударов в секунду: пики огибающей выше её разброса."""
    env, fps = _onset_envelope(mono, sr)
    if len(env) < 3:
        return 0.0
    thr = env.std() * 1.0
    peaks = (env[1:-1] > env[:-2]) & (env[1:-1] >= env[2:]) & (env[1:-1] > thr)
    return float(peaks.sum() / (len(env) / fps))


def ensure_beats(models_dir) -> str | None:
    """Чекпойнт Beat This! в формате библиотеки; None — не скачался (темп тогда автокорреляцией)."""
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
    if conf < BEATS_MIN_CONF:  # доли гуляют — такому темпу верить нельзя, остаётся автокорреляция
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
    """Темп, громкость, удары и голос. Держит модели загруженными на весь проход. Без torch — темп автокорреляцией и без голоса;
    без весов Beat This! — темп автокорреляцией; без Demucs — без голоса (наличие вокала тогда по CLAP микса, sound_tags)."""

    def __init__(self, effnet=None, device: str | None = None, beats_ckpt: str | None = None):
        self.effnet = effnet
        self.beats = None
        self.demucs = None
        self.device = device or "cpu"
        try:
            import torch
        except Exception:
            return
        self.torch = torch
        # по умолчанию torch берёт все ядра, а рядом декодируется следующий трек: 4 потока — та же скорость
        torch.set_num_threads(min(4, torch.get_num_threads()))
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        if beats_ckpt:
            try:
                from beat_this.inference import Audio2Beats

                self.beats = Audio2Beats(checkpoint_path=beats_ckpt, device=self.device, dbn=False)
            except Exception:  # веса Beat This! недоступны — темп автокорреляцией
                self.beats = None
        try:
            from demucs.pretrained import get_model
        except ImportError:  # Demucs не установлен — без голоса
            return
        # Веса качаются при первом запуске. Сбой здесь (сеть, устройство) роняет проход, и он повторится позже: иначе треки
        # записались бы без голоса и больше не пересчитывались бы
        bag = get_model("htdemucs")
        bag.eval()
        self.demucs = bag.models[0] if getattr(bag, "models", None) else bag
        self.demucs.to(self.device)

    def tempo(self, stereo: np.ndarray) -> tuple[float, float, str]:
        got = tempo_from_beats(self.beats, stereo) if self.beats is not None else None
        if got:
            return got[0], got[1], "beat_this"
        bpm, conf = tempo_autocorr(stereo.mean(axis=0), SR)
        return bpm, conf, "autocorr" if self.beats is None else "autocorr_only"

    def separate_excerpts(self, stereo: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Стем вокала и микс по EXCERPTS отрывкам: формы (k, 2, кусок). Нормировка общая по отрывкам (как по треку раньше)."""
        torch, model = self.torch, self.demucs
        seg = int(float(model.segment) * model.samplerate)
        x = np.zeros((EXCERPTS, 2, seg), np.float32)
        for i, s in enumerate(excerpt_starts(stereo.shape[1] / SR, seg / SR)):
            p = stereo[:, int(s * SR):int(s * SR) + seg]
            x[i, :, :p.shape[1]] = p
        wav = torch.from_numpy(x)
        ref = wav.mean(1)
        mean, std = ref.mean(), ref.std() + 1e-8
        with torch.no_grad():
            y = model(((wav - mean) / std).to(self.device)).cpu()
        if self.device == "mps":
            torch.mps.empty_cache()
        voc = (y[:, list(model.sources).index("vocals")] * std + mean).numpy()
        return voc, x

    def voice(self, stereo: np.ndarray) -> tuple[float, float, float | None, np.ndarray]:
        """(доля звучания вокала, доля энергии, вероятность женского голоса по стему, выделенный голос для CLAP)."""
        import soxr

        from arxnova import sound_tags

        voc, mix = self.separate_excerpts(stereo)
        frac, ratio = vocal_stats(np.concatenate(list(voc), axis=1), np.concatenate(list(mix), axis=1))
        female = None
        if self.effnet is not None:
            emb, act = [], []
            for k in range(len(mix)):
                m16 = soxr.resample(mix[k].mean(0), SR, F.SR_EFFNET).astype(np.float32)
                v16 = soxr.resample(voc[k].mean(0), SR, F.SR_EFFNET).astype(np.float32)
                e = self.effnet.run_windows(F.patches(F.melspec(v16)))
                wins = patch_windows(len(v16))[: len(e)]
                for a0, a1 in wins:
                    act.append(20 * np.log10(np.sqrt(np.mean(v16[a0:a1] ** 2) + 1e-12) / np.sqrt(np.mean(m16[a0:a1] ** 2) + 1e-12)))
                emb.append(e[: len(wins)])
            emb = np.concatenate(emb) if emb else np.zeros((0, 1280), np.float32)
            w = np.array(act) > VOICE_DB
            if len(emb):
                vec = emb[w].mean(0) if w.sum() >= 3 else emb.mean(0)
                female = float(sound_tags.female_probability_stem(vec[None])[0])
        audio = soxr.resample(voc.mean(axis=1).reshape(-1), SR, VOICE_SR).astype(np.float32)
        return frac, ratio, female, audio

    def analyze(self, stereo: np.ndarray) -> TrackFacets:
        mono = stereo.mean(axis=0)
        bpm, bpm_conf, src = self.tempo(stereo)
        rms = float(np.sqrt(np.mean(stereo ** 2)) + 1e-12)
        f = TrackFacets(float(bpm), float(bpm_conf), src, float(20 * np.log10(rms)), onset_rate(mono, SR))
        if self.demucs is not None:
            f.vocal_frac, f.vocal_ratio, f.female_stem_p, f.voice = self.voice(stereo)
        return f
