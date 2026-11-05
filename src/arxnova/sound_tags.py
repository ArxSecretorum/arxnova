"""Голос и инструменты из уже посчитанных векторов: без нового разбора и без тяжёлых моделей на сервере (решение 0013).

Женский или мужской голос: логистический слой на векторе вокальной дорожки, 92,7 % на певцах вне обучения
(bench/eval_goal/train_voice.py; разметка по известным певцам, на слух не проверено). Инструменты: близость вектора CLAP к готовому
вектору текстового описания. Это фильтры, а не слагаемые оценки: на дослушивание они не влияют (заметка о модели, раздел 8)."""

from __future__ import annotations

import base64

import numpy as np

from arxnova import sound_tags_data as D

FEMALE_FROM, MALE_UNDER = 0.7, 0.3
TAG_SHARE = 0.05  # к инструменту — верхние 5 % медиатеки
TAGS = {"inst_piano": "Фортепиано", "inst_sax": "Саксофон", "inst_strings": "Струнные", "inst_guitar": "Акустическая гитара",
        "inst_choir": "Хор"}


def _floats(s: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(s), np.float32)


def female_probability(vectors: np.ndarray) -> np.ndarray:
    coef, mean, std = _floats(D.VOICE_COEF), _floats(D.VOICE_MEAN), _floats(D.VOICE_STD)
    z = (np.nan_to_num(vectors) - mean) / std @ coef + D.VOICE_INTERCEPT
    return 1.0 / (1.0 + np.exp(-z))


def tag_members(audio: np.ndarray) -> dict[str, np.ndarray]:
    """Строки audio (единичные векторы CLAP), ближайшие к описанию инструмента."""
    k = max(1, int(TAG_SHARE * len(audio)))
    return {tag: np.argsort(-(audio @ _floats(D.TAG_TEXT[tag])))[:k] for tag in TAGS if tag in D.TAG_TEXT}
