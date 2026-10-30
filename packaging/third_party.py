"""Тексты лицензий всего, что едет в установщике (macOS, Windows, Linux): библиотеки сервера,
сам Python и загрузчик PyInstaller. MIT, BSD, Apache и LGPL требуют прикладывать их при раздаче.
Запуск при сборке: uv run --project .. --with pyinstaller python third_party.py [файл]"""

from __future__ import annotations

import re
import sys
import sysconfig
from importlib import metadata
from pathlib import Path

LICENSE_NAME = re.compile(r"(^|/)(LICEN[CS]E|COPYING|NOTICE|AUTHORS|ThirdPartyNotices)[^/]*$", re.I)
LGPL_SOURCES = "https://github.com/ArxSecretorum/arxnova/releases"


def closure(root: str = "arxnova") -> dict[str, metadata.Distribution]:
    """Обязательные зависимости пакета вместе с зависимостями зависимостей."""
    seen: dict[str, metadata.Distribution] = {}
    todo = [root]
    while todo:
        name = re.split(r"[\s;\[<>=!~(]", todo.pop(), maxsplit=1)[0].lower().replace("_", "-")
        if not name or name in seen:
            continue
        try:
            d = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            continue
        seen[name] = d
        todo += [r for r in d.requires or [] if "extra ==" not in r]
    return seen


def license_expr(d: metadata.Distribution) -> str:
    m = d.metadata
    expr = m.get("License-Expression") or ""
    if not expr:
        raw = (m.get("License") or "").strip().splitlines()
        expr = raw[0] if raw and len(raw[0]) < 80 else ""
    if not expr:
        cls = [c.split("::")[-1].strip() for c in m.get_all("Classifier") or [] if c.startswith("License ::")]
        expr = "; ".join(cls)
    return expr or "см. текст ниже"


def license_texts(d: metadata.Distribution) -> list[tuple[str, str]]:
    out = []
    for f in d.files or []:
        s = str(f)
        # в dist-info или прямо в папке пакета (onnxruntime/LICENSE, onnxruntime/ThirdPartyNotices.txt)
        if LICENSE_NAME.search(s) and (".dist-info/" in s or s.count("/") == 1):
            try:
                out.append((s.split(".dist-info/", 1)[-1], Path(d.locate_file(f)).read_text("utf-8", "replace")))
            except OSError:
                pass
    if not out:  # старые пакеты кладут лицензию только в поле License
        raw = (d.metadata.get("License") or "").strip()
        if len(raw) > 80:
            out.append(("License", raw))
    return out


def section(title: str, body: str) -> str:
    return f"{'=' * 78}\n{title}\n{'=' * 78}\n\n{body.strip()}\n\n"


def build() -> str:
    deps = sorted(closure().items())
    # пакет без текста лицензии (flatbuffers): стандартный текст Apache 2.0 из соседнего пакета
    apache = next((t for _, d in deps for _, t in license_texts(d)
                   if "Apache License" in t and "Version 2.0, January 2004" in t), "")
    parts = ["Arx Nova — лицензии сторонних компонентов / third-party licenses\n\n"
             "Сам Arx Nova — по лицензии PolyForm Strict 1.0.0 (файл LICENSE). Компоненты ниже входят в\n"
             "установщик и остаются под своими лицензиями. Arx Nova itself is licensed under PolyForm\n"
             "Strict 1.0.0 (see LICENSE); the components below ship with the installer under their own licenses.\n\n"]
    for name, d in deps:
        if name == "arxnova":
            continue
        expr = license_expr(d)
        head = f"{d.metadata['Name']} {d.version} — {expr}"
        url = d.metadata.get("Home-page") or next(
            (u.split(",", 1)[1].strip() for u in d.metadata.get_all("Project-URL") or [] if "," in u), "")
        body = f"{url}\n\n" if url else ""
        if "LGPL" in expr or "Lesser" in expr:
            body += (f"Исходный код этой версии приложен к каждому выпуску / source code of this version is "
                     f"attached to every release: {LGPL_SOURCES}\n\n")
        texts = license_texts(d)
        if not texts and "Apache" in expr and apache:
            texts = [("Apache License 2.0 (стандартный текст)", apache)]
        body += "\n\n".join(f"--- {fn} ---\n{t.strip()}" for fn, t in texts) if texts else "(текст лицензии в пакете не найден)"
        parts.append(section(head, body))
    py = Path(sysconfig.get_paths()["stdlib"]) / "LICENSE.txt"
    if py.exists():
        parts.append(section(f"Python {sys.version.split()[0]} — PSF-2.0", py.read_text("utf-8", "replace")))
    try:
        pi = metadata.distribution("pyinstaller")
        texts = license_texts(pi)
        if texts:
            parts.append(section(f"PyInstaller {pi.version} bootloader — GPL-2.0 with bootloader exception",
                                 "\n\n".join(t for _, t in texts)))
    except metadata.PackageNotFoundError:
        pass
    return "".join(parts)


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).with_name("THIRD_PARTY_LICENSES.txt"))
    out.write_text(build(), "utf-8")
    print(f"{out}: {out.stat().st_size // 1024} КБ")
