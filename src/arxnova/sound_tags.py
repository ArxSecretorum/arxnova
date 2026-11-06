"""Голос и инструменты из уже посчитанных векторов: без нового разбора и без тяжёлых моделей на сервере (решения 0013, 0014).

Женский или мужской голос по выделенному голосу 8 отрывков (решение 0014): среднее трёх — слой на векторе стема в окнах, где голос
звучит, слой на CLAP выделенного голоса и описания CLAP «женщина − мужчина». На певцах вне обучения ошибок в плитках меньше, чем
у прежнего слоя на векторе стема целого трека: 3,4 % против 4,9 % при равном охвате (тот слой остался для перевода в число треков,
разобранных до 0014; их число идёт в среднее как мнение стема). Запасной путь без
Demucs — слои на CLAP микса (есть ли вокал и голос). Слои — bench/eval_goal/train_voice.py; разметка голоса по известным певцам,
на слух не проверено. Инструменты: близость вектора CLAP к готовому вектору текстового описания.
Это фильтры, а не слагаемые оценки: на дослушивание они не влияют (заметка о модели, раздел 8)."""

from __future__ import annotations

import base64

import numpy as np

from arxnova import sound_tags_data as D

FEMALE_FROM, MALE_UNDER = 0.7, 0.3
VOICE_DIM, CLAP_DIM = 1280, 512  # вектор стема вокала (EffNet) и вектор CLAP
TAG_SHARE = 0.05  # к инструменту — верхние 5 % медиатеки
TAGS = {"inst_piano": "Фортепиано", "inst_sax": "Саксофон", "inst_strings": "Струнные", "inst_guitar": "Акустическая гитара",
        "inst_choir": "Хор"}


def _floats(s: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(s), np.float32)


def _logit(vectors: np.ndarray, coef: str, mean: str, std: str, intercept: float) -> np.ndarray:
    z = (np.nan_to_num(vectors) - _floats(mean)) / _floats(std) @ _floats(coef) + intercept
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def female_probability(vectors: np.ndarray) -> np.ndarray:
    """По вектору стема вокала (треки, разобранные с Demucs; нужен один раз — перевести их в число, db.facets_to_numbers)."""
    return _logit(vectors, D.VOICE_COEF, D.VOICE_MEAN, D.VOICE_STD, D.VOICE_INTERCEPT)


def female_probability_stem(vectors: np.ndarray) -> np.ndarray:
    """По вектору стема вокала 8 отрывков в окнах, где голос звучит (facets.Analyzer.voice)."""
    return _logit(vectors, D.STEM_VOICE_COEF, D.STEM_VOICE_MEAN, D.STEM_VOICE_STD, D.STEM_VOICE_INTERCEPT)


def female_from_voice(stem_p: np.ndarray, voice_clap: np.ndarray) -> np.ndarray:
    """Среднее трёх по выделенному голосу: слой стема, слой CLAP голоса, описания CLAP. Строки voice_clap с NaN — без CLAP;
    stem_p NaN — без слоя стема. Где нет ничего — NaN."""
    stem_p = np.asarray(stem_p, np.float64)
    have = ~np.isnan(voice_clap).any(axis=1)
    layer = np.full(len(stem_p), np.nan)
    zero_shot = np.full(len(stem_p), np.nan)
    if have.any():
        v = voice_clap[have]
        layer[have] = _logit(v, D.VOICE_CLAP_COEF, D.VOICE_CLAP_MEAN, D.VOICE_CLAP_STD, D.VOICE_CLAP_INTERCEPT)
        zero_shot[have] = 1.0 / (1.0 + np.exp(-np.clip(D.FEM_ZS_W * (v @ _floats(D.FEM_TEXT)) + D.FEM_ZS_B, -30, 30)))
    parts = np.vstack([stem_p, layer, zero_shot])
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # у трека без голоса ничего нет — NaN, это не ошибка
        return np.nanmean(parts, axis=0)


def female_probability_clap(clap: np.ndarray) -> np.ndarray:
    return _logit(clap, D.FEMALE_CLAP_COEF, D.FEMALE_CLAP_MEAN, D.FEMALE_CLAP_STD, D.FEMALE_CLAP_INTERCEPT)


def vocal_probability_clap(clap: np.ndarray) -> np.ndarray:
    return _logit(clap, D.VOCAL_CLAP_COEF, D.VOCAL_CLAP_MEAN, D.VOCAL_CLAP_STD, D.VOCAL_CLAP_INTERCEPT)


def tag_members(audio: np.ndarray) -> dict[str, np.ndarray]:
    """Строки audio (единичные векторы CLAP), ближайшие к описанию инструмента."""
    k = max(1, int(TAG_SHARE * len(audio)))
    return {tag: np.argsort(-(audio @ _floats(D.TAG_TEXT[tag])))[:k] for tag in TAGS if tag in D.TAG_TEXT}
