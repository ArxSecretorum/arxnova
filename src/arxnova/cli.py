"""Команда `arxnova` (ТЗ, §6.1). Пока: setup, doctor, index (обход медиатеки)."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from arxnova import config, db, doctor, indexer, models, scan

RESCAN_MIN = 10


def _cmd_setup(args) -> int:
    cfg = config.load(args.data_dir)
    if args.library:
        cfg.libraries = [Path(p).expanduser().resolve() for p in args.library]
    elif not cfg.libraries and sys.stdin.isatty():
        # установка одной командой: спросить папку с музыкой, а не требовать флаг
        guess = Path.home() / "Music"
        try:
            ans = input(f"Папка с музыкой [{guess}]: ").strip()
        except EOFError:
            ans = ""
        p = Path(ans or guess).expanduser().resolve()
        if p.is_dir():
            cfg.libraries = [p]
        else:
            print(f"Папки {p} нет — задайте позже: `arxnova setup --library <папка>`")
    if args.port:
        cfg.port = args.port
    cfg.pick_remote_port()
    cfg.ensure_dirs()
    cfg.save()
    if cfg.remote:
        from arxnova import tlsid

        ident = tlsid.ensure(cfg.data_dir, cfg.server_name)
        print(f"Ключ сервера для связи вне дома готов, порт {cfg.remote_port}, отпечаток {ident.fingerprint[:16]}…")
    print(f"Настройки записаны: {cfg.path}")
    _offer_models(cfg, assume_yes=args.yes, source_dir=args.models_from)
    return 0


def _offer_models(cfg, *, assume_yes: bool | None, source_dir: Path | None) -> None:
    """Сразу при установке предлагаем докачать модель — понятно и одним ответом."""
    m = models.EFFNET
    if models.is_installed(cfg.models_dir, m):
        print(f"Модель {m.title} уже установлена.")
        return
    print()
    print(m.consent)
    print(f"Лицензия: {m.license} — {m.license_url}")
    print(f"Авторы: {m.attribution}")
    if assume_yes is None:
        try:
            answer = input("Скачать модель? [Д/н] ").strip().lower()
        except EOFError:
            answer = "н"
        ok = answer in ("", "д", "да", "y", "yes")
    else:
        ok = assume_yes
    if not ok:
        print("Хорошо. Скачать позже: `arxnova models install effnet-discogs-1`.")
        return
    models.install(cfg.models_dir, m, source_dir=source_dir)
    print("Модель установлена.")


def _cmd_models(args) -> int:
    cfg = config.load(args.data_dir)
    if args.action == "list":
        for m in models.CATALOG.values():
            state = "установлена" if models.is_installed(cfg.models_dir, m) else "не установлена"
            print(f"{m.id}: {m.title} — {m.license} — {state}")
        return 0
    m = models.CATALOG.get(args.model or "")
    if not m:
        print(f"Нет такой модели. Есть: {', '.join(models.CATALOG)}", file=sys.stderr)
        return 2
    _offer_models(cfg, assume_yes=args.yes, source_dir=args.models_from)
    return 0


def _cmd_doctor(args) -> int:
    cfg = config.load(args.data_dir)
    worst = 0
    for status, name, detail in doctor.checks(cfg):
        print(f"[{status:^8}] {name}: {detail}")
        worst = max(worst, {doctor.OK: 0, doctor.WARN: 0, doctor.FAIL: 1}[status])
    return worst


def _cmd_index(args) -> int:
    cfg = config.load(args.data_dir)
    if not cfg.libraries:
        print("Медиатека не задана: `arxnova setup --library <папка>`", file=sys.stderr)
        return 2
    cfg.ensure_dirs()
    conn = db.connect(cfg.db_path)
    t0 = time.monotonic()

    def progress(rep):
        if rep.seen % 50 == 0:
            print(f"  … {rep.seen} файлов", flush=True)

    rep = scan.scan(conn, cfg.libraries, progress)
    n_emb = indexer.embed_pending(cfg, conn, progress=print)
    if n_emb or not (cfg.index_dir / models.EFFNET.id / "ids.npy").exists():
        indexer.export_matrices(cfg, conn)
    if not models.is_installed(cfg.models_dir, models.EFFNET):
        print("Модель звука не установлена — подбор по звуку недоступен: `arxnova models install effnet-discogs-1`.")
    dt = time.monotonic() - t0
    print(
        f"Файлов: {rep.seen}; новых треков: {rep.added}; без изменений: {rep.unchanged}; "
        f"тот же звук по другому пути: {rep.duplicates}; MD5 по звуку: {rep.pcm_md5}; "
        f"удалено: {rep.removed}; ошибок: {len(rep.errors)}; эмбеддингов: {n_emb}; {dt:.1f} с"
    )
    for p, e in rep.errors[:20]:
        print(f"  ошибка: {p}: {e}", file=sys.stderr)
    return 1 if rep.errors else 0


def _cmd_serve(args) -> int:
    import uvicorn

    from arxnova.server import create_app

    cfg = config.load(args.data_dir)
    cfg.ensure_dirs()
    if cfg.pick_remote_port():
        cfg.save()
    host = args.host
    print(f"Arx Nova слушает {host}:{cfg.port}. Сопряжение: `arxnova pair`.")
    ident = reach = awake = None
    if cfg.remote:
        from arxnova import platforms, tlsid
        from arxnova.reach import DIAGNOSIS, Reach

        ident = tlsid.ensure(cfg.data_dir, cfg.server_name)
        profile = platforms.current()
        reach = Reach(cfg.remote_port, profile, history=cfg.data_dir / "address_history.log",
                      fingerprint=ident.fingerprint)
        reach.start()
        if cfg.keep_awake:
            awake = profile.keep_awake()
        print(f"Связь вне дома: порт {cfg.remote_port} (только TLS), профиль {profile.name}.")
    handle = None
    try:
        from arxnova import discovery

        handle = discovery.advertise(cfg.server_name, cfg.port)
        print("Сервер виден в домашней сети — Arx Player найдёт его сам.")
    except Exception as e:  # без mDNS сервер работает, адрес вводится вручную
        print(f"Поиск в сети недоступен ({e}); в плеере введите адрес вручную.")
    import threading

    stop = threading.Event()

    def rescan_loop():
        """Новые файлы медиатеки подхватываются сами: сразу после запуска (первая установка —
        медиатека ещё не проиндексирована) и дальше раз в RESCAN_MIN минут (§5.4)."""
        first = True
        while first or not stop.wait(RESCAN_MIN * 60):
            first = False
            try:
                c = db.connect(cfg.db_path)
                rep = scan.scan(c, cfg.libraries)
                n = indexer.embed_pending(cfg, c)
                if n or rep.removed:
                    indexer.export_matrices(cfg, c)
                if rep.added or rep.removed or n:
                    print(f"Медиатека обновлена: новых {rep.added}, удалено {rep.removed}, проанализировано {n}", flush=True)
                c.close()
            except Exception as e:  # обход не должен ронять сервер
                print(f"Обход медиатеки не удался: {e}", flush=True)

    def report_reach():
        """Один раз, когда первая проверка сети закончилась: что получилось и что делать."""
        while reach and not stop.wait(2):
            st = reach.status
            if st.diagnosis != "starting":
                how = {"pcp": "PCP", "natpmp": "NAT-PMP", "upnp": "UPnP"}.get(st.method or "", "—")
                print(f"Вне дома: {DIAGNOSIS.get(st.diagnosis, st.diagnosis)}; проброс: {how}; "
                      f"адрес: {st.endpoint() or 'нет'}", flush=True)
                return

    threading.Thread(target=report_reach, name="reach-report", daemon=True).start()
    app = create_app(cfg, reach, ident.fingerprint if ident else None)
    # обход — после того как сервер открыл и обновил базу: не толкаться на первом запуске
    threading.Thread(target=rescan_loop, name="rescan", daemon=True).start()
    servers = [uvicorn.Server(uvicorn.Config(app, host=host, port=cfg.port, log_level="warning"))]
    if ident:
        servers.append(uvicorn.Server(uvicorn.Config(
            app, host="0.0.0.0", port=cfg.remote_port, log_level="warning",
            ssl_certfile=str(ident.cert), ssl_keyfile=str(ident.key))))

    async def run_all():
        # Два слушателя в одном процессе: остановка любого (Ctrl-C) останавливает оба.
        import asyncio

        tasks = [asyncio.create_task(s.serve()) for s in servers]
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for s in servers:
            s.should_exit = True
        await asyncio.gather(*tasks, return_exceptions=True)

    try:
        import asyncio

        asyncio.run(run_all())
    finally:
        stop.set()
        if reach:
            reach.close()  # снять проброс с роутера
        if awake:
            awake.terminate()
        if handle:
            handle.close()
    return 0


def _cmd_pair(args) -> int:
    from arxnova import auth

    cfg = config.load(args.data_dir)
    conn = db.connect(cfg.db_path)
    code = auth.new_pair_code(conn)
    print(f"Код сопряжения: {code[:3]} {code[3:]}  (действует {auth.CODE_TTL_S // 60} минут)")
    print("Введите его в Arx Player: Настройки → Arx Nova → Моя медиатека.")
    return 0


def _cmd_invite(args) -> int:
    """Приглашение гостю (ОД-1): код на 24 ч и ссылка для мессенджера."""
    from arxnova import guests, platforms, stun, tlsid

    cfg = config.load(args.data_dir)
    conn = db.connect(cfg.db_path)
    try:
        inv = guests.create_invite(conn, args.name, None, cfg.max_guests)
    except guests.GuestError as e:
        print(str(e), file=sys.stderr)
        return 1
    net = None
    if cfg.remote and cfg.remote_port:
        ident = tlsid.ensure(cfg.data_dir, cfg.server_name)
        local = platforms.current().local_ip()
        public = stun.external_ip(local)
        net = {"tls": {"port": cfg.remote_port, "fingerprint": ident.fingerprint},
               "external": f"{public}:{cfg.remote_port}" if public else None}
    local = platforms.current().local_ip()
    url = guests.link(inv["code"], cfg.server_name, cfg.inviter, net,
                      f"http://{local}:{cfg.port}" if local else None)
    c = inv["code"]
    print(f"Приглашение для «{inv['guest_name']}»: код {c[:4]}-{c[4:]} (действует 24 часа, один раз).")
    print("Отправьте эту ссылку — Arx Player откроет вход сам:")
    print(url)
    return 0


def _cmd_guests(args) -> int:
    from arxnova import guests

    cfg = config.load(args.data_dir)
    conn = db.connect(cfg.db_path)
    if args.revoke:
        ok = guests.revoke_guest(conn, args.revoke)
        print("Доступ отозван, история и «Избранное» гостя удалены." if ok else "Такого гостя нет.")
        return 0 if ok else 1
    out = guests.list_guests(conn)
    print(f"Гостей: {len(out['guests'])} из {guests.limit(cfg.max_guests)}")
    for g in out["guests"]:
        seen = time.strftime("%d.%m %H:%M", time.localtime(g["last_seen"])) if g["last_seen"] else "ещё не был"
        print(f"  {g['id']}: {g['name']} ({g['device']}), был в сети {seen}, скачано {g['bytes'] // (1 << 20)} МБ")
    for i in out["invites"]:
        print(f"  приглашение для «{i['name']}» ждёт входа до {time.strftime('%d.%m %H:%M', time.localtime(i['expires_at']))}")
    return 0


def _cmd_service(args) -> int:
    from arxnova import service

    cfg = config.load(args.data_dir)
    try:
        if args.action == "install":
            print(service.install(cfg.data_dir))
        elif args.action == "remove":
            print(service.remove())
        else:
            ok, text = service.status()
            print(text)
            return 0 if ok else 1
    except (RuntimeError, OSError) as e:
        print(f"Не получилось: {e}", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="arxnova", description="Arx Nova — сервер рекомендаций")
    ap.add_argument("--data-dir", type=Path, default=None, help="каталог данных")
    sub = ap.add_subparsers(dest="cmd")

    p = sub.add_parser("setup", help="первый запуск: медиатека, порт")
    p.add_argument("--library", action="append", help="папка медиатеки (можно несколько)")
    p.add_argument("--port", type=int)
    p.add_argument("--yes", action="store_true", default=None, help="согласиться скачать модели без вопроса")
    p.add_argument("--no-models", dest="yes", action="store_false", help="не скачивать модели")
    p.add_argument("--models-from", type=Path, help="взять файлы моделей из папки (без сети)")
    p.set_defaults(func=_cmd_setup)

    p = sub.add_parser("models", help="модели: list | install <id>")
    p.add_argument("action", choices=["list", "install"])
    p.add_argument("model", nargs="?")
    p.add_argument("--yes", action="store_true", default=None)
    p.add_argument("--models-from", type=Path)
    p.set_defaults(func=_cmd_models)

    p = sub.add_parser("doctor", help="проверить установку")
    p.set_defaults(func=_cmd_doctor)

    p = sub.add_parser("index", help="обойти медиатеку и обновить каталог")
    p.set_defaults(func=_cmd_index)

    p = sub.add_parser("serve", help="запустить сервер")
    p.add_argument("--host", default="0.0.0.0", help="адрес (по умолчанию — все сети компьютера)")
    p.set_defaults(func=_cmd_serve)

    p = sub.add_parser("pair", help="выдать код сопряжения устройства")
    p.set_defaults(func=_cmd_pair)

    p = sub.add_parser("invite", help="пригласить гостя (код на 24 ч и ссылка)")
    p.add_argument("name", help="кому приглашение")
    p.set_defaults(func=_cmd_invite)

    p = sub.add_parser("service", help="автозапуск сервера: install | remove | status")
    p.add_argument("action", choices=["install", "remove", "status"])
    p.set_defaults(func=_cmd_service)

    p = sub.add_parser("guests", help="гости: список; --revoke <id> — отозвать доступ")
    p.add_argument("--revoke", type=int)
    p.set_defaults(func=_cmd_guests)

    p = sub.add_parser("app", help="окно приложения (по умолчанию без команды)")
    p.set_defaults(func=lambda a: __import__("arxnova.app", fromlist=["main"]).main(a.data_dir))

    argv = sys.argv[1:] if argv is None else argv
    # без команды (двойной щелчок по приложению) — окно; служба автозапуска зовёт с «serve»
    if not any(a in sub.choices for a in argv) and "-h" not in argv and "--help" not in argv:
        argv = [*argv, "app"]
    args = ap.parse_args(argv)
    return args.func(args)
