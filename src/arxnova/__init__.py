"""Arx Nova — локальная рекомендательная система для домашней FLAC-медиатеки.

Телеметрия ONNX Runtime выключается здесь, при импорте пакета, — раньше, чем любой
модуль успеет импортировать onnxruntime (ТЗ, НФТ-4). Вызова disable_telemetry_events()
недостаточно: к нему начальное событие уже может уйти.
"""

import os

os.environ["ORT_DISABLE_TELEMETRY"] = "1"
# FastAPI встроил OpenTelemetry: без настройки он ничего не отправляет, но экспорт можно включить
# переменными окружения. Закрываем это явно — сервер не шлёт наружу ничего о запросах.
os.environ["FASTAPI_OTEL_AUTO_CONFIGURE"] = "false"
os.environ["OTEL_SDK_DISABLED"] = "true"
for _k in [k for k in os.environ if k.startswith("OTEL_") and k.endswith("_EXPORTER")]:
    os.environ[_k] = "none"

__version__ = "0.2.21"


def main() -> None:
    from arxnova.cli import main as cli_main

    raise SystemExit(cli_main())
