"""Точка входа собранного приложения: без аргументов — окно, с «serve» — фоновый сервер."""
import os
import sys

# В оконной сборке stdout/stderr может не быть (Windows) — печать в никуда, а не падение.
for name in ("stdout", "stderr"):
    if getattr(sys, name) is None:
        setattr(sys, name, open(os.devnull, "w", encoding="utf-8"))

from arxnova import main  # noqa: E402

main()
