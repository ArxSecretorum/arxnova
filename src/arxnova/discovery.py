"""Поиск сервера в домашней сети (API-3): объявление _arxnova._tcp по mDNS.
zeroconf — LGPL-2.1, подключается как отдельный пакет (динамически), в белом списке УСТ-8."""

from __future__ import annotations

import socket

from arxnova import __version__

SERVICE = "_arxnova._tcp.local."


def _lan_addresses() -> list[bytes]:
    addrs = set()
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("192.0.2.1", 9))  # пакет не уходит: только выбор исходящего интерфейса
            addrs.add(s.getsockname()[0])
    except OSError:
        pass
    return [socket.inet_aton(a) for a in addrs if not a.startswith("127.")]


def advertise(name: str, port: int):
    """Возвращает объект, который нужно закрыть (.close()) при остановке сервера."""
    from zeroconf import ServiceInfo, Zeroconf

    zc = Zeroconf()
    info = ServiceInfo(
        SERVICE,
        f"{name}.{SERVICE}",
        addresses=_lan_addresses(),
        port=port,
        properties={"version": __version__, "name": name},
        server=f"{socket.gethostname().split('.')[0]}.local.",
    )
    zc.register_service(info)

    class Handle:
        def close(self):
            zc.unregister_service(info)
            zc.close()

    return Handle()
