"""Arx Nova — локальная рекомендательная система для домашней FLAC-медиатеки.

Телеметрия ONNX Runtime выключается здесь, при импорте пакета, — раньше, чем любой
модуль успеет импортировать onnxruntime (ТЗ, НФТ-4). Вызова disable_telemetry_events()
недостаточно: к нему начальное событие уже может уйти.
"""

import os
import sys

os.environ["ORT_DISABLE_TELEMETRY"] = "1"
# FastAPI встроил OpenTelemetry: без настройки он ничего не отправляет, но экспорт можно включить
# переменными окружения. Закрываем это явно — сервер не шлёт наружу ничего о запросах.
os.environ["FASTAPI_OTEL_AUTO_CONFIGURE"] = "false"
os.environ["OTEL_SDK_DISABLED"] = "true"
for _k in [k for k in os.environ if k.startswith("OTEL_") and k.endswith("_EXPORTER")]:
    os.environ[_k] = "none"

__version__ = "0.2.54"


def _fewer_malloc_arenas() -> None:
    """glibc резервирует на каждую арену памяти по 64 МБ адресов: у сервера это был 1 ГБ из 1,9. Реальной
    памяти резерв не занимает, но на Pi с LimitAS=3G адреса кончались, и сборка каталога падала с
    MemoryError при 1,5 ГБ занятой памяти (03.11.2026). Двух арен хватает: почти всё выделение в Python
    и так идёт под GIL. До импорта всего остального: потоки уже не успеют завести свои арены."""
    if not sys.platform.startswith("linux"):
        return
    try:
        import ctypes

        ctypes.CDLL(None).mallopt(-8, 2)  # M_ARENA_MAX
    except (OSError, AttributeError):
        pass  # не glibc


def main() -> None:
    _fewer_malloc_arenas()
    from arxnova.cli import main as cli_main

    raise SystemExit(cli_main())
