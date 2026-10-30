"""Лицензия сервера — PolyForm Strict 1.0.0 (решение владельца 30.10.2026): текст лежит в LICENSE и
указан в описании пакета; установщик несёт тексты лицензий всех сторонних библиотек, а исходники
библиотек под LGPL прикладываются к выпуску (packaging/lgpl_sources.py)."""

import importlib.util
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "packaging" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_license_file_is_polyform_strict():
    text = (ROOT / "LICENSE").read_text()
    assert text.startswith("Required Notice: Copyright 2026 Arx Secretorum")
    assert "# PolyForm Strict License 1.0.0" in text
    assert "<https://polyformproject.org/licenses/strict/1.0.0>" in text
    assert "other than distributing the software or making changes or new works based on the software" in text
    assert "Additional Permission for Bundled LGPL Libraries" in text


def test_package_metadata_declares_license():
    proj = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert proj["license"] == "LicenseRef-PolyForm-Strict-1.0.0"
    assert proj["license-files"] == ["LICENSE"]


def test_third_party_notices_cover_every_dependency():
    tp = _load("third_party")
    deps = {n for n in tp.closure() if n != "arxnova"}
    assert len(deps) > 10
    text = tp.build()
    for name, d in tp.closure().items():
        if name != "arxnova":
            assert f"{d.metadata['Name']} {d.version} — " in text, name
    assert "текст лицензии в пакете не найден" not in text


def test_lgpl_sources_list_matches_dependencies():
    tp, src = _load("third_party"), _load("lgpl_sources")
    lgpl = {n for n, d in tp.closure().items() if "LGPL" in tp.license_expr(d) or "Lesser" in tp.license_expr(d)}
    assert lgpl == set(src.LGPL), f"исходники LGPL к выпуску: {sorted(lgpl)}"
