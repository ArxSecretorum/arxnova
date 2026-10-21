"""`arxnova doctor` — сводка: платформа, папки, медиатека, порт, телеметрия (ТЗ, §6.1)."""

from __future__ import annotations

import os
import platform
import socket
import sys

from arxnova import __version__
from arxnova.config import Config

OK, WARN, FAIL = "ok", "внимание", "ошибка"


def _port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("0.0.0.0", port))
            return True
        except OSError:
            return False


def checks(cfg: Config) -> list[tuple[str, str, str]]:
    out: list[tuple[str, str, str]] = []
    out.append((OK, "версия", f"arxnova {__version__}, Python {platform.python_version()}"))
    out.append((OK, "платформа", f"{sys.platform} {platform.machine()}"))
    out.append(
        (OK, "настройки", str(cfg.path)) if cfg.path.exists()
        else (WARN, "настройки", f"{cfg.path} нет — запустите `arxnova setup`")
    )
    if not cfg.libraries:
        out.append((WARN, "медиатека", "не задана"))
    for lib in cfg.libraries:
        out.append((OK, "медиатека", str(lib)) if lib.is_dir() else (FAIL, "медиатека", f"{lib} недоступна"))
    out.append(
        (OK, "порт", f"{cfg.port} свободен") if _port_free(cfg.port)
        else (WARN, "порт", f"{cfg.port} занят (возможно, уже работает `arxnova serve`)")
    )
    tele = os.environ.get("ORT_DISABLE_TELEMETRY") == "1"
    out.append(
        (OK, "телеметрия ONNX", "выключена") if tele else (FAIL, "телеметрия ONNX", "ВКЛЮЧЕНА")
    )
    otel = os.environ.get("FASTAPI_OTEL_AUTO_CONFIGURE") == "false" and os.environ.get("OTEL_SDK_DISABLED") == "true"
    out.append((OK, "телеметрия FastAPI", "выключена") if otel else (FAIL, "телеметрия FastAPI", "может быть включена окружением"))
    ort_cache = os.path.expanduser("~/.cache/Microsoft/DeveloperTools/.onnxruntime")
    if os.path.exists(ort_cache):
        out.append((FAIL, "телеметрия ONNX", f"найден {ort_cache} — ONNX Runtime запускали с телеметрией"))
    from arxnova import models

    for m in models.CATALOG.values():
        if models.is_installed(cfg.models_dir, m):
            out.append((OK, "модель", f"{m.title} — {m.license}"))
        else:
            out.append((WARN, "модель", f"{m.title} не установлена — `arxnova models install {m.id}`"))
    from arxnova import service

    ok, text = service.status()
    out.append((OK if ok else WARN, "автозапуск", text))
    out.extend(net_checks(cfg))
    if cfg.db_path.exists():
        from arxnova import db

        conn = db.connect(cfg.db_path)
        n = conn.execute("SELECT count(*) FROM tracks").fetchone()[0]
        e = conn.execute("SELECT count(*) FROM embeddings").fetchone()[0]
        out.append((OK, "каталог", f"{n} треков, проанализировано {e}, база {cfg.db_path}"))
    return out


def net_checks(cfg: Config) -> list[tuple[str, str, str]]:
    """Связь вне дома (ОД-3а). Только чтение: проброс не создаётся, чтобы не мешать серверу."""
    from arxnova import platforms, portmap, stun
    from arxnova.reach import DIAGNOSIS, Reach, Status

    out: list[tuple[str, str, str]] = []
    p = platforms.current()
    out.append((OK, "профиль платформы", p.name))
    if not cfg.remote:
        out.append((WARN, "вне дома", "выключено (remote = false в настройках)"))
        return out
    out.append((OK, "порт для интернета", f"{cfg.remote_port} (TCP, только TLS)") if cfg.remote_port
               else (WARN, "порт для интернета", "ещё не выбран — `arxnova setup`"))
    tls = cfg.data_dir / "tls" / "server.crt"
    if tls.exists():
        from arxnova import tlsid

        out.append((OK, "ключ сервера", f"отпечаток {tlsid.fingerprint_of(tls)[:16]}…"))
    else:
        out.append((WARN, "ключ сервера", "ещё не создан — появится при `arxnova setup` или `serve`"))
    fw = p.firewall(cfg.remote_port)
    if fw:
        out.append((OK if fw[0] == "ok" else FAIL, "брандмауэр", fw[1]))
    local = p.local_ip()
    gw = p.gateway(local)
    out.append((OK, "домашняя сеть", f"компьютер {local}, роутер {gw}") if local
               else (FAIL, "домашняя сеть", "нет адреса в сети"))
    methods = []
    router_ip = None
    if gw:
        try:
            router_ip = portmap.natpmp_external(gw)
            methods.append("NAT-PMP")
        except (portmap.MapError, OSError):
            pass
        try:
            portmap.upnp_find(local)
            methods.append("UPnP")
        except (portmap.MapError, OSError):
            pass
    out.append((OK, "проброс порта", "роутер умеет: " + ", ".join(methods)) if methods
               else (WARN, "проброс порта", f"роутер не открывает порты сам — пробросьте вручную: "
                                            f"TCP {cfg.remote_port} → {local}:{cfg.remote_port}; "
                                            f"закрепите за компьютером адрес {local} в роутере"))
    st = Status(profile=p.name, local_ip=local, gateway=gw, ipv6=p.global_ipv6(),
                method="probe" if methods else None, router_ip=router_ip,
                public_ip=stun.external_ip(local), port=cfg.remote_port)
    d = Reach._diagnose(st)
    out.append((OK if d in ("public", "manual") else FAIL, "диагноз сети",
                ("белый адрес, роутер сам откроет порт — сервер будет доступен из интернета" if d == "public"
                 else DIAGNOSIS[d]) + f"; внешний адрес {st.public_ip or '—'}"))
    from arxnova.reach import address_changes

    n, since = address_changes(cfg.data_dir / "address_history.log")
    if since:
        out.append((OK if n == 0 else WARN, "адрес дома",
                    f"с {since} не менялся" if n == 0 else f"с {since} менялся {n} раз — плеер найдёт новый через DHT"))
    if st.ipv6:
        out.append((OK, "IPv6", f"есть глобальный адрес {st.ipv6} (пока не используется)"))
    if p.key == "macos":
        out.append((OK if cfg.keep_awake else WARN, "сон",
                    "Mac не засыпает, пока работает сервер и подключено питание" if cfg.keep_awake
                    else "Mac может уснуть — вне дома сервер будет недоступен (keep_awake = false)"))
    return out
