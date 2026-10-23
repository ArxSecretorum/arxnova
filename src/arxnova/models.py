"""Модели: что можно скачать, откуда, с какой лицензией и контрольной суммой (ТЗ, §6.4,
УСТ-6). Модели не входят в пакет — их скачивает сам сервер, с согласия пользователя."""

from __future__ import annotations

import hashlib
import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelFile:
    name: str
    url: str
    sha256: str


@dataclass(frozen=True)
class Model:
    id: str
    title: str
    files: tuple[ModelFile, ...]
    license: str
    license_url: str
    attribution: str
    commercial_ok: bool  # можно ли раздавать в составе продукта
    consent: str  # что показываем пользователю перед скачиванием
    size_mb: int


EFFNET = Model(
    id="effnet-discogs-1",
    title="Discogs-EffNet (MTG, Universitat Pompeu Fabra)",
    files=(
        ModelFile(
            "discogs-effnet-bsdynamic-1.onnx",
            "https://essentia.upf.edu/models/feature-extractors/discogs-effnet/discogs-effnet-bsdynamic-1.onnx",
            "a280825b334797cf677939db8cd5762c0392aedd0ca6415dbc1cd083f045e43c",
        ),
        ModelFile(
            "discogs-effnet-bsdynamic-1.json",
            "https://essentia.upf.edu/models/feature-extractors/discogs-effnet/discogs-effnet-bsdynamic-1.json",
            "a2e85b2e7372d5f8e0f35bdd6aeae1139f101087d183d0b2fb60b0ea0f01a0ff",
        ),
    ),
    license="CC BY-NC-SA 4.0",
    license_url="https://creativecommons.org/licenses/by-nc-sa/4.0/",
    attribution="Music Technology Group, Universitat Pompeu Fabra — https://essentia.upf.edu/models.html",
    commercial_ok=False,
    consent=(
        "Arx Nova лучше всего подбирает музыку с моделью Discogs-EffNet.\n"
        "Её создала Music Technology Group (Universitat Pompeu Fabra, Барселона),\n"
        "и она распространяется под некоммерческой лицензией CC BY-NC-SA 4.0:\n"
        "её можно использовать только для личного прослушивания, не для заработка.\n"
        "Модель скачается с сайта авторов (18 МБ) и останется только на этом компьютере."
    ),
    size_mb=18,
)

MAEST = Model(
    id="maest-519",
    title="MAEST, Discogs 519 стилей (MTG, Universitat Pompeu Fabra)",
    files=(
        ModelFile(
            "discogs-maest-30s-pw-519l-2.onnx",
            "https://essentia.upf.edu/models/feature-extractors/maest/discogs-maest-30s-pw-519l-2.onnx",
            "c90a51a752cdd94f37de886787d5e3a5b2071c6d0ef49ea788058f65f11883b1",
        ),
        ModelFile(
            "discogs-maest-30s-pw-519l-2.json",
            "https://essentia.upf.edu/models/feature-extractors/maest/discogs-maest-30s-pw-519l-2.json",
            "83240aa553ffb491b0ec5a24565eb612553e5f38da5207403c25b890c5b34acd",
        ),
    ),
    license="CC BY-NC-SA 4.0",
    license_url="https://creativecommons.org/licenses/by-nc-sa/4.0/",
    attribution="Music Technology Group, Universitat Pompeu Fabra — https://essentia.upf.edu/models.html",
    commercial_ok=False,
    consent=(
        "Стили треков точнее определяет модель MAEST (519 стилей Discogs, в том числе 22 стиля\n"
        "метала). Та же лицензия, что у Discogs-EffNet: CC BY-NC-SA 4.0, только личное\n"
        "использование. Скачается с сайта авторов (350 МБ) вместе с компонентами анализа."
    ),
    size_mb=350,
)

CATALOG = {m.id: m for m in (EFFNET, MAEST)}


def model_dir(models_root: Path, model: Model) -> Path:
    return models_root / model.id


def is_installed(models_root: Path, model: Model) -> bool:
    d = model_dir(models_root, model)
    return all((d / f.name).exists() for f in model.files) and (d / "license.json").exists()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class ChecksumError(RuntimeError):
    pass


def install(models_root: Path, model: Model, *, source_dir: Path | None = None, progress=print) -> Path:
    """Скачивает файлы модели и сверяет sha256. Файл с неверной суммой удаляется — сервер
    не запустит модель, которую не узнал. source_dir — взять файлы с диска (без сети)."""
    d = model_dir(models_root, model)
    d.mkdir(parents=True, exist_ok=True)
    for f in model.files:
        dst = d / f.name
        if dst.exists() and _sha256(dst) == f.sha256:
            continue
        tmp = dst.with_suffix(dst.suffix + ".part")
        if source_dir and (source_dir / f.name).exists():
            tmp.write_bytes((source_dir / f.name).read_bytes())
        else:
            progress(f"  скачиваю {f.name} …")
            req = urllib.request.Request(f.url, headers={"User-Agent": "arxnova"})
            with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as out:
                while chunk := r.read(1 << 20):
                    out.write(chunk)
        got = _sha256(tmp)
        if got != f.sha256:
            tmp.unlink(missing_ok=True)
            raise ChecksumError(f"{f.name}: sha256 {got} не совпадает с ожидаемой {f.sha256}")
        tmp.replace(dst)
    (d / "license.json").write_text(
        json.dumps(
            {"model": model.id, "title": model.title, "license": model.license,
             "license_url": model.license_url, "attribution": model.attribution,
             "commercial_ok": model.commercial_ok},
            ensure_ascii=False, indent=1,
        ),
        encoding="utf-8",
    )
    return d


def onnx_path(models_root: Path, model: Model) -> Path:
    return model_dir(models_root, model) / next(f.name for f in model.files if f.name.endswith(".onnx"))


def maest_labels(models_root: Path) -> list[str]:
    meta = json.loads((model_dir(models_root, MAEST) / "discogs-maest-30s-pw-519l-2.json").read_text())
    return meta["classes"]


def style_labels(models_root: Path) -> list[str]:
    meta = json.loads((model_dir(models_root, EFFNET) / "discogs-effnet-bsdynamic-1.json").read_text())
    return meta["classes"]
