"""Исходники библиотек под LGPL (soxr, zeroconf) — к каждому выпуску: LGPL требует давать их там же,
где раздаётся программа. Версии — из uv.lock, архивы — sdist с PyPI.
Запуск: python packaging/lgpl_sources.py <папка> [--dry-run]"""

import json
import sys
import tomllib
import urllib.request
from pathlib import Path

LGPL = ("soxr", "zeroconf")


def main(out: Path, dry: bool) -> None:
    lock = tomllib.loads((Path(__file__).resolve().parents[1] / "uv.lock").read_text())
    vers = {p["name"]: p["version"] for p in lock["package"] if p["name"] in LGPL}
    out.mkdir(parents=True, exist_ok=True)
    for name in LGPL:
        ver = vers[name]
        info = json.load(urllib.request.urlopen(f"https://pypi.org/pypi/{name}/{ver}/json", timeout=60))
        sdist = next(u for u in info["urls"] if u["packagetype"] == "sdist")
        dest = out / f"source-LGPL-{sdist['filename']}"
        print(name, ver, sdist["url"], "->", dest.name)
        if not dry:
            urllib.request.urlretrieve(sdist["url"], dest)


if __name__ == "__main__":
    main(Path(sys.argv[1]), "--dry-run" in sys.argv)
