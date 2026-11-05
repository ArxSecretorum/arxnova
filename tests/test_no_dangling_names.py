"""Имя, которого нет ни в модуле, ни среди встроенных. Тесты его не видят, пока строка не выполнится: при чистке разбора удалили
OVERLAP и веса Beat This!, и каждый новый трек падал в Demucs с NameError."""

from __future__ import annotations

import ast
import builtins
import importlib
import pathlib
import symtable

import arxnova


def _dangling(path: pathlib.Path) -> list[str]:
    top = symtable.symtable(path.read_text(encoding="utf-8"), str(path), "exec")
    known = {s.get_name() for s in top.get_symbols()} | set(dir(builtins)) | {"__file__", "__name__", "__doc__", "__version__"}
    found: list[str] = []

    def walk(table) -> None:
        for s in table.get_symbols():
            if table is not top and s.is_global() and s.is_referenced() and s.get_name() not in known:
                found.append(f"{path.name}:{table.get_lineno()} {table.get_name()}: {s.get_name()}")
        for child in table.get_children():
            walk(child)

    walk(top)
    return found


def test_every_name_in_the_code_exists():
    root = pathlib.Path(arxnova.__file__).parent
    found = [x for p in sorted(root.rglob("*.py")) for x in _dangling(p)]
    assert not found, found


def test_every_attribute_of_our_modules_exists():
    """То же через точку: G.harmonic, facets.key_name. После удаления функции такое обращение падает только когда до него дойдёт очередь."""
    root = pathlib.Path(arxnova.__file__).parent
    missing: list[str] = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        ours = {a.asname or a.name: f"arxnova.{a.name}" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module == "arxnova"
                for a in n.names}
        for n in ast.walk(tree):
            if isinstance(n, ast.ImportFrom) and n.module and n.module.startswith("arxnova."):
                module = importlib.import_module(n.module)
                missing += [f"{path.name}:{n.lineno} {n.module}.{a.name}" for a in n.names if a.name != "*" and not hasattr(module, a.name)]
            elif isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id in ours:
                if not hasattr(importlib.import_module(ours[n.value.id]), n.attr):
                    missing.append(f"{path.name}:{n.lineno} {n.value.id}.{n.attr}")
    assert not missing, missing
