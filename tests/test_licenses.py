"""Профиль «раздача» (ТЗ Nova, приёмка п. 11): в сервере нет зависимостей под AGPL/GPL (они
обязали бы открыть код всего, что с ними поставляется), а модели под некоммерческой лицензией
(CC BY-NC) в пакет не кладутся — они скачиваются по согласию владельца (см. models.py)."""

import re
from importlib import metadata
from pathlib import Path

import pytest

COPYLEFT = re.compile(r"\bA?GPL\b|GNU (Affero )?General Public|AGPL", re.I)
LESSER = re.compile(r"LGPL|Lesser General Public|Library General Public", re.I)


def _closure(root: str) -> dict[str, metadata.Distribution]:
    seen: dict[str, metadata.Distribution] = {}
    todo = [root]
    while todo:
        name = re.split(r"[\s;\[<>=!~(]", todo.pop(), maxsplit=1)[0].lower().replace("_", "-")
        if not name or name in seen:
            continue
        try:
            d = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            continue  # необязательная зависимость не установлена
        seen[name] = d
        for req in d.requires or []:
            if "extra ==" not in req:  # только обязательные
                todo.append(req)
    return seen


def _license_text(d: metadata.Distribution) -> str:
    m = d.metadata
    parts = [m.get("License-Expression") or "", m.get("License") or ""]
    parts += [c for c in m.get_all("Classifier") or [] if c.startswith("License ::")]
    return " | ".join(p for p in parts if p)


def test_no_copyleft_dependencies():
    bad = []
    for name, d in _closure("arxnova").items():
        if name == "arxnova":
            continue
        text = _license_text(d)
        if COPYLEFT.search(text) and not LESSER.search(text):
            bad.append(f"{name}: {text[:120]}")
    assert not bad, "зависимости под GPL/AGPL:\n" + "\n".join(bad)


def test_no_model_files_in_package():
    """Веса моделей (ONNX, TensorFlow) в исходниках пакета не лежат — только скачиваются."""
    src = Path(__file__).resolve().parents[1] / "src" / "arxnova"
    heavy = [p for p in src.rglob("*") if p.suffix.lower() in {".onnx", ".pb", ".tflite", ".pt", ".ckpt", ".safetensors"}]
    assert not heavy, f"веса моделей в пакете: {heavy}"


@pytest.mark.parametrize("attr", ["EFFNET"])
def test_nc_models_require_consent(attr):
    from arxnova import models

    m = getattr(models, attr)
    if "NC" in m.license:
        assert m.license_url and m.attribution, "у модели NC — ссылка на лицензию и атрибуция для окна согласия"
