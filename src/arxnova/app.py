"""Приложение сервера Arx Nova — одно окно, почти без настроек (решение владельца 21.10.2026).

Первый запуск: выбрать папку с музыкой (обычный проводник), согласиться на модель звука —
«Запустить». Дальше приложение само: скачивает модель, включает автозапуск сервера, сервер
разбирает медиатеку. Потом в окне — состояние и две кнопки: «Подключить плеер» (код из
6 цифр) и «Пригласить». Сервер работает фоновой службой (service.py): окно можно закрыть.
"""

from __future__ import annotations

import json
import locale
import threading
import time
import tkinter as tk
import urllib.request
from pathlib import Path
from tkinter import filedialog, ttk

from arxnova import __version__, config, db, models

BG, SURFACE, LINE = "#0e0e0f", "#1a1a1c", "#2a2a2d"
TEXT, MUTED, BRASS, ON_BRASS = "#ecebe7", "#8d8b86", "#c9a96b", "#141210"

RU = {
    "title": "Arx Nova",
    "subtitle": "Ваша медиатека в Arx Player — из любой точки",
    "step_folder": "1. Папка с музыкой",
    "pick": "Выбрать папку…",
    "no_folder": "не выбрана",
    "step_model": "2. Модель звука",
    "consent": "Скачать модель Discogs-EffNet (MTG, лицензия CC BY-NC-SA 4.0, ~18 МБ) — "
               "по ней Nova понимает звучание треков. Без неё медиатека слушается, но поток «по вкусу» не работает.",
    "agree": "Согласен скачать",
    "start": "Запустить",
    "working": "Работает",
    "stopped": "Сервер не отвечает — перезапускаю…",
    "stats": "{tracks} треков · разобрано {pct}%",
    "pair": "Подключить плеер",
    "pair_hint": "На телефоне: Arx Player → Настройки → Arx Nova → поле «Код из окна Arx Nova» — введите эти 6 цифр. Телефон может быть где угодно: дома, на работе, на мобильном интернете. Код действует 5 минут.",
    "invite": "Пригласить гостя",
    "invite_name": "Кому (имя):",
    "invite_make": "Создать приглашение",
    "invite_done": "Приглашение для «{name}» — код {code}. Ссылка скопирована: отправьте её гостю.",
    "folder_change": "Сменить папку",
    "setting_up": "Настраиваю: {what}…",
    "dl_model": "скачиваю модель звука",
    "enable_service": "включаю автозапуск",
    "err": "Не получилось: {e}",
    "close_hint": "Окно можно закрыть — сервер продолжит работать и запустится сам после перезагрузки.",
    "srv_restart": "Перезапустить",
    "srv_stop": "Остановить",
    "srv_start": "Запустить",
    "srv_stopped": "Остановлен — плееры не подключатся, пока не запустите",
    "srv_starting": "Запускаю…",
    "lib_title": "Медиатека",
    "lib_counts": "Общий звук: {mix} из {n} · стили и ритм: {fac} из {n} · сценарии: {clap} из {n}",
    "lib_auto": "Новые треки в папке разбираются сами.",
    "lib_run": "Проанализировать сейчас",
    "lib_install": "Скачать компоненты анализа (~{gb} ГБ)",
    "lib_need": "Чтобы Nova понимала стили, ритм, вокал и сценарии («для бега», «мистическая»), нужны компоненты анализа: Demucs, torch и CLAP — около {gb} ГБ. Скачиваются один раз и остаются на этом компьютере.",
    "lib_consent": "Скачать компоненты анализа (~{gb} ГБ)?\n\nDemucs (разделение на ударные, бас, вокал — MIT), torch, Beat This! (темп — MIT) и CLAP LAION (описания «текст → звук» — CC0). Всё остаётся на этом компьютере; музыка никуда не отправляется.",
    "lib_idle": "Всё разобрано.",
    "net_btn": "Проверить и настроить сеть",
    "stream_btn": "Настроить поток: мосты между стилями",
    "net_running": "Проверяю сеть: роутер, порт, адрес, торрент-сеть… (до минуты)",
    "net_lan_ok": "В сети: компьютер {ip}, роутер {gw}",
    "net_lan_bad": "Нет сети — подключите компьютер к роутеру",
    "net_port_ok": "Роутер открыл порт {port} сам ({method})",
    "net_port_bad": "Роутер не открывает порт сам — включите в его настройках UPnP или пробросьте TCP {port} → {ip}:{port}",
    "net_ip_public": "Адрес в интернете {ip} — белый, сервер доступен извне",
    "net_ip_cgnat": "Провайдер выдаёт «серый» адрес — из интернета до дома не достучаться. Попросите у провайдера белый IP",
    "net_ip_double": "Перед вашим роутером есть ещё одна подсеть: его внешний адрес {router} — внутренний, а в интернете дом виден как {ip}. Порт открыт, но снаружи до него не дойти. Если дома есть модем провайдера — переведите его в режим моста (или пробросьте порт {port} и на нём). Если модема нет — это NAT провайдера: попросите белый IP-адрес",
    "net_ip_offline": "Нет выхода в интернет",
    "net_loop_ok": "Сервер отвечает по внешнему адресу",
    "net_loop_na": "Проверить снаружи изнутри сети нельзя (роутер не умеет «петлю») — это не значит, что не работает",
    "net_dht_ok": "Отмечен в торрент-сети ({n} узлов): если адрес сменится, плеер найдёт сервер сам",
    "net_dht_bad": "Торрент-сеть недоступна — при смене адреса плеер найдёт сервер только дома",
    "net_fw": "Брандмауэр: {text}",
    "net_auto_ok": "Автозапуск включён",
    "net_auto_fixed": "Автозапуск был выключен — включил",
    "net_sleep_ok": "Компьютер не уснёт, пока работает сервер и есть питание",
    "net_remote_off": "Доступ из интернета выключен в настройках (remote = false)",
    "net_fail": "Сервер не ответил: {e}",
    "net_tunnel_ok": "Пробивка NAT готова ({n} узлов торрент-сети знают сервер): плеер вне дома достучится напрямую, даже если порт снаружи закрыт",
    "net_tunnel_wait": "Пробивка NAT: сервер ещё отмечается в торрент-сети — проверьте через минуту",
    "net_lan_perm": "Роутер не ответил приложению. На Mac: Системные настройки → Конфиденциальность и безопасность → Локальная сеть → включите «Arx Nova», затем нажмите кнопку ещё раз",
}
EN = {
    "title": "Arx Nova",
    "subtitle": "Your library in Arx Player — from anywhere",
    "step_folder": "1. Music folder",
    "pick": "Choose folder…",
    "no_folder": "not chosen",
    "step_model": "2. Sound model",
    "consent": "Download the Discogs-EffNet model (MTG, CC BY-NC-SA 4.0 license, ~18 MB) — "
               "Nova uses it to understand how tracks sound. Without it the library plays, but “my stream” doesn’t.",
    "agree": "I agree to download",
    "start": "Start",
    "working": "Running",
    "stopped": "Server not responding — restarting…",
    "stats": "{tracks} tracks · analysed {pct}%",
    "pair": "Connect a player",
    "pair_hint": "On your phone: Arx Player → Settings → Arx Nova → “Code from the Arx Nova window” — type these 6 digits. The phone can be anywhere: at home, at work, on mobile data. Valid for 5 minutes.",
    "invite": "Invite a guest",
    "invite_name": "Who (name):",
    "invite_make": "Create invite",
    "invite_done": "Invite for “{name}” — code {code}. Link copied: send it to your guest.",
    "folder_change": "Change folder",
    "setting_up": "Setting up: {what}…",
    "dl_model": "downloading the sound model",
    "enable_service": "enabling autostart",
    "err": "Failed: {e}",
    "close_hint": "You can close this window — the server keeps running and starts by itself after a reboot.",
    "srv_restart": "Restart",
    "srv_stop": "Stop",
    "srv_start": "Start",
    "srv_stopped": "Stopped — players can’t connect until you start it",
    "srv_starting": "Starting…",
    "lib_title": "Library",
    "lib_counts": "Overall sound: {mix} of {n} · styles and rhythm: {fac} of {n} · scenarios: {clap} of {n}",
    "lib_auto": "New tracks in the folder are analysed automatically.",
    "lib_run": "Analyse now",
    "lib_install": "Download analysis components (~{gb} GB)",
    "lib_need": "For Nova to understand styles, rhythm, vocals and scenarios (“for running”, “mysterious”) it needs analysis components: Demucs, torch and CLAP — about {gb} GB. Downloaded once, they stay on this computer.",
    "lib_consent": "Download analysis components (~{gb} GB)?\n\nDemucs (splits drums, bass, vocals — MIT), torch, Beat This! (tempo — MIT) and LAION CLAP (“text → sound” descriptions — CC0). Everything stays on this computer; your music is not sent anywhere.",
    "lib_idle": "Everything is analysed.",
    "net_btn": "Check and set up network",
    "stream_btn": "Tune the stream: bridges between styles",
    "net_running": "Checking the network: router, port, address, torrent network… (up to a minute)",
    "net_lan_ok": "Online: computer {ip}, router {gw}",
    "net_lan_bad": "No network — connect the computer to the router",
    "net_port_ok": "The router opened port {port} by itself ({method})",
    "net_port_bad": "The router doesn’t open ports by itself — enable UPnP in its settings or forward TCP {port} → {ip}:{port}",
    "net_ip_public": "Internet address {ip} is public — the server is reachable from outside",
    "net_ip_cgnat": "Your provider gives a shared (CGNAT) address — home can’t be reached from the internet. Ask for a public IP",
    "net_ip_double": "There is another network in front of your router: its outside address {router} is internal, while the internet sees your home as {ip}. The port is open but unreachable from outside. If you have a provider modem, switch it to bridge mode (or forward port {port} on it too). If not, it’s your provider’s NAT: ask for a public IP address",
    "net_ip_offline": "No internet connection",
    "net_loop_ok": "The server answers on its internet address",
    "net_loop_na": "Can’t test from inside the network (router has no NAT loopback) — doesn’t mean it fails outside",
    "net_dht_ok": "Announced in the torrent network ({n} nodes): if the address changes, players find the server",
    "net_dht_bad": "Torrent network unreachable — after an address change players find the server only at home",
    "net_fw": "Firewall: {text}",
    "net_auto_ok": "Autostart is on",
    "net_auto_fixed": "Autostart was off — turned it on",
    "net_sleep_ok": "The computer won’t sleep while the server runs on power",
    "net_remote_off": "Internet access is off in settings (remote = false)",
    "net_fail": "The server didn’t answer: {e}",
    "net_tunnel_ok": "NAT punching ready ({n} torrent-network nodes know the server): players away from home connect directly even if the port is closed",
    "net_tunnel_wait": "NAT punching: the server is still announcing itself — check again in a minute",
    "net_lan_perm": "The router didn’t answer the app. On a Mac: System Settings → Privacy & Security → Local Network → turn on “Arx Nova”, then press the button again",
}


def _lang() -> dict:
    try:
        code = (locale.getlocale()[0] or locale.getdefaultlocale()[0] or "").lower()
    except Exception:
        code = ""
    return RU if code.startswith("ru") else EN


class App:
    def __init__(self, data_dir: Path | None = None):
        self.cfg = config.load(data_dir)
        self.t = _lang()
        self.root = tk.Tk()
        self.root.title(self.t["title"])
        self.root.configure(bg=BG)
        self.root.minsize(460, 420)
        self._style()
        self.body = tk.Frame(self.root, bg=BG, padx=22, pady=18)
        self.body.pack(fill="both", expand=True)
        self.status = tk.StringVar()
        # фоновые потоки окно не трогают: кладут действие в очередь, а главный поток её разбирает
        # (вызов Tk из другого потока на macOS не гарантирован — статус застревал на «…»)
        import queue

        self._uiq: queue.Queue = queue.Queue()
        self._pump()
        if self.cfg.libraries and self.cfg.path.exists():
            self._main_screen()
        else:
            self._first_run()

    def _ui(self, fn) -> None:
        """Выполнить fn в главном потоке окна — из любого потока."""
        self._uiq.put(fn)

    def _pump(self) -> None:
        while not self._uiq.empty():
            try:
                self._uiq.get_nowait()()
            except Exception:
                pass
        self.root.after(100, self._pump)

    # ---------- оформление ----------
    def _style(self):
        s = ttk.Style(self.root)
        s.theme_use("clam")
        s.configure("Brass.TButton", background=BRASS, foreground=ON_BRASS, borderwidth=0,
                    font=("Helvetica", 14, "bold"), padding=(14, 9))
        s.map("Brass.TButton", background=[("disabled", LINE), ("active", "#d8bb81")],
              foreground=[("disabled", MUTED)])
        s.configure("Flat.TButton", background=SURFACE, foreground=TEXT, borderwidth=0,
                    font=("Helvetica", 13), padding=(12, 7))
        s.map("Flat.TButton", background=[("active", LINE)])
        s.configure("TCheckbutton", background=BG, foreground=TEXT, font=("Helvetica", 13))
        s.map("TCheckbutton", background=[("active", BG)])

    def _label(self, parent, text, size=13, color=TEXT, bold=False, wrap=420):
        return tk.Label(parent, text=text, bg=BG, fg=color, justify="left", wraplength=wrap,
                        font=("Helvetica", size, "bold" if bold else "normal"))

    def _clear(self):
        for w in self.body.winfo_children():
            w.destroy()

    def _header(self):
        self._label(self.body, self.t["title"], 26, BRASS, bold=True).pack(anchor="w")
        self._label(self.body, self.t["subtitle"], 12, MUTED).pack(anchor="w", pady=(0, 14))

    # ---------- первый запуск ----------
    def _first_run(self):
        self._clear()
        self._header()
        self.folder = tk.StringVar(value=str(self.cfg.libraries[0]) if self.cfg.libraries else "")
        self.agree = tk.BooleanVar(value=models.is_installed(self.cfg.models_dir, models.EFFNET))

        self._label(self.body, self.t["step_folder"], 14, TEXT, bold=True).pack(anchor="w")
        row = tk.Frame(self.body, bg=BG)
        row.pack(fill="x", pady=(4, 14))
        ttk.Button(row, text=self.t["pick"], style="Flat.TButton", command=self._pick).pack(side="left")
        self.folder_lbl = self._label(row, self.folder.get() or self.t["no_folder"], 12, MUTED, wrap=280)
        self.folder_lbl.pack(side="left", padx=10)

        self._label(self.body, self.t["step_model"], 14, TEXT, bold=True).pack(anchor="w")
        self._label(self.body, self.t["consent"], 12, MUTED).pack(anchor="w", pady=(4, 4))
        ttk.Checkbutton(self.body, text=self.t["agree"], variable=self.agree).pack(anchor="w", pady=(0, 16))

        self.start_btn = ttk.Button(self.body, text=self.t["start"], style="Brass.TButton", command=self._start)
        self.start_btn.pack(fill="x")
        self._label(self.body, "", 12, BRASS, wrap=420).pack(anchor="w", pady=(10, 0))
        self.msg = self.body.winfo_children()[-1]
        self._refresh_start()

    def _pick(self):
        d = filedialog.askdirectory(title=self.t["step_folder"], mustexist=True,
                                    initialdir=self.folder.get() or str(Path.home()))
        if d:
            self.folder.set(d)
            self.folder_lbl.configure(text=d, fg=TEXT)
            self._refresh_start()

    def _refresh_start(self):
        self.start_btn.state(["!disabled"] if self.folder.get() else ["disabled"])

    def _start(self):
        self.start_btn.state(["disabled"])
        folder, agree = self.folder.get(), self.agree.get()
        threading.Thread(target=self._setup, args=(folder, agree), daemon=True).start()

    def _say(self, text):
        self._ui(lambda: self.msg.configure(text=text))

    def _setup(self, folder: str, agree: bool):
        from arxnova import service

        try:
            self.cfg.libraries = [Path(folder)]
            self.cfg.pick_remote_port()
            self.cfg.ensure_dirs()
            self.cfg.save()
            if agree and not models.is_installed(self.cfg.models_dir, models.EFFNET):
                self._say(self.t["setting_up"].format(what=self.t["dl_model"]))
                models.install(self.cfg.models_dir, models.EFFNET, progress=lambda *_: None)
            self._say(self.t["setting_up"].format(what=self.t["enable_service"]))
            if service.status()[0]:
                service.restart()
            else:
                service.install(self.cfg.data_dir)
            self._ui(self._main_screen)
        except Exception as e:  # покажем по-человечески, кнопка снова доступна
            self._say(self.t["err"].format(e=e))
            self._ui(lambda: self.start_btn.state(["!disabled"]))

    # ---------- рабочий экран ----------
    def _main_screen(self):
        self._clear()
        self._header()
        top = tk.Frame(self.body, bg=BG)
        top.pack(fill="x")
        self.dot = tk.Label(top, text="●", bg=BG, fg=MUTED, font=("Helvetica", 16))
        self.dot.pack(side="left")
        self.state_lbl = self._label(top, "…", 15, TEXT, bold=True)
        self.state_lbl.pack(side="left", padx=6)
        # управление сервером: перезапуск и остановка/запуск (служба в фоне, окно — только пульт)
        self.toggle_btn = ttk.Button(top, text=self.t["srv_stop"], style="Flat.TButton", command=self._toggle_server)
        self.toggle_btn.pack(side="right")
        ttk.Button(top, text=self.t["srv_restart"], style="Flat.TButton", command=self._restart_server).pack(side="right", padx=(0, 6))
        self.stats_lbl = self._label(self.body, "", 12, MUTED)
        self.stats_lbl.pack(anchor="w", pady=(2, 2))
        self._label(self.body, ", ".join(str(p) for p in self.cfg.libraries), 11, MUTED).pack(anchor="w", pady=(0, 14))

        ttk.Button(self.body, text=self.t["pair"], style="Brass.TButton", command=self._pair).pack(fill="x")
        self.code_lbl = tk.Label(self.body, text="", bg=BG, fg=BRASS, font=("Menlo", 34, "bold"))
        self.code_lbl.pack(pady=(8, 0))
        self.code_hint = self._label(self.body, "", 11, MUTED)
        self.code_hint.pack(anchor="w")

        row = tk.Frame(self.body, bg=BG)
        row.pack(fill="x", pady=(12, 0))
        ttk.Button(row, text=self.t["invite"], style="Flat.TButton", command=self._invite).pack(side="left")
        # медиатека: что разобрано, что идёт сейчас, кнопки анализа
        lib = tk.Frame(self.body, bg=SURFACE, padx=12, pady=10)
        lib.pack(fill="x", pady=(14, 0))
        self._label(lib, self.t["lib_title"], 13, TEXT, bold=True).pack(anchor="w")
        self.lib_counts = self._label(lib, "", 11, MUTED, wrap=460)
        self.lib_counts.pack(anchor="w", pady=(2, 0))
        self.lib_stage = self._label(lib, "", 12, BRASS, wrap=460)
        self.lib_stage.pack(anchor="w", pady=(4, 0))
        self.lib_btns = tk.Frame(lib, bg=SURFACE)
        self.lib_btns.pack(fill="x", pady=(8, 0))
        self.run_btn = ttk.Button(self.lib_btns, text=self.t["lib_run"], style="Flat.TButton", command=self._analyse_now)
        self.install_btn = ttk.Button(self.lib_btns, text=self.t["lib_install"].format(gb=3), style="Brass.TButton",
                                      command=self._install_analysis)
        self.run_btn.pack(side="left")
        self.net_btn = ttk.Button(self.body, text=self.t["net_btn"], style="Flat.TButton", command=self._net_check)
        self.net_btn.pack(fill="x", pady=(10, 0))
        self.net_box = tk.Frame(self.body, bg=BG)
        self.net_box.pack(fill="x", pady=(6, 0))
        # панель шлюза и разметка мостов (ФТ-8.4, КО-1) — страницы сервера, только с этого компьютера
        ttk.Button(self.body, text=self.t["stream_btn"], style="Flat.TButton",
                   command=lambda: __import__("webbrowser").open(f"http://127.0.0.1:{self.cfg.port}/ui/gate")).pack(fill="x", pady=(6, 0))
        ttk.Button(row, text=self.t["folder_change"], style="Flat.TButton", command=self._first_run).pack(side="right")
        self.msg = self._label(self.body, "", 12, BRASS)
        self.msg.pack(anchor="w", pady=(10, 0))
        self._label(self.body, self.t["close_hint"], 11, MUTED).pack(anchor="w", side="bottom", pady=(14, 0))
        self._restarted_at = 0.0
        self._poll()
        # сразу при открытии: macOS спросит про локальную сеть, пока человек смотрит на окно
        threading.Thread(target=self._touch_lan, daemon=True).start()

    def _poll(self):
        threading.Thread(target=self._poll_bg, daemon=True).start()
        self.root.after(3000, self._poll)

    def _poll_bg(self):
        ok = False
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{self.cfg.port}/v1/health", timeout=2) as r:
                h = json.loads(r.read())
                ok = h.get("name") == "arxnova"
        except Exception:
            h = {}
        tracks = analysed = 0
        if self.cfg.db_path.exists():
            try:
                c = db.connect(self.cfg.db_path)
                tracks = c.execute("SELECT count(*) FROM tracks").fetchone()[0]
                analysed = c.execute("SELECT count(*) FROM embeddings").fetchone()[0]
                c.close()
            except Exception:
                pass
        an = None
        if ok:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.cfg.port}/v1/local/analysis", timeout=2) as r:
                    an = json.loads(r.read())
            except Exception:
                an = None
        if not ok and not self._user_stopped and time.monotonic() - self._restarted_at > 60:
            # служба легла или ещё не поднялась — один толчок в минуту
            from arxnova import service

            self._restarted_at = time.monotonic()
            try:
                service.restart() if service.status()[0] else service.install(self.cfg.data_dir)
            except Exception:
                pass
        pct = int(analysed * 100 / tracks) if tracks else 0

        def show():
            self.dot.configure(fg="#5fbf6f" if ok else (MUTED if self._user_stopped else BRASS))
            self.state_lbl.configure(text=self.t["working"] if ok else
                                     (self.t["srv_stopped"] if self._user_stopped else self.t["stopped"]))
            self.toggle_btn.configure(text=self.t["srv_start"] if self._user_stopped else self.t["srv_stop"])
            self.stats_lbl.configure(text=self.t["stats"].format(tracks=tracks, pct=pct))
            self._show_analysis(an)
        self._ui(show)

    # ---------- управление сервером ----------
    _user_stopped = False

    def _restart_server(self):
        from arxnova import service

        self._user_stopped = False
        self.state_lbl.configure(text=self.t["srv_starting"])
        threading.Thread(target=lambda: service.restart() if service.status()[0] else service.install(self.cfg.data_dir),
                         daemon=True).start()

    def _toggle_server(self):
        from arxnova import service

        if self._user_stopped:
            self._user_stopped = False
            self.state_lbl.configure(text=self.t["srv_starting"])
            threading.Thread(target=lambda: service.start(self.cfg.data_dir), daemon=True).start()
        else:
            self._user_stopped = True
            threading.Thread(target=service.stop, daemon=True).start()

    # ---------- анализ медиатеки ----------
    def _show_analysis(self, an):
        if not an:
            self.lib_counts.configure(text="")
            self.lib_stage.configure(text="")
            return
        n = an.get("tracks", 0)
        self.lib_counts.configure(text=self.t["lib_counts"].format(n=n, mix=an.get("mix", 0), fac=an.get("facets", 0),
                                                                   clap=an.get("clap", 0)) + "\n" + self.t["lib_auto"])
        if an.get("error"):
            line = an["error"]
        elif an.get("busy") or an.get("stage"):
            line = " — ".join(x for x in (an.get("stage"), an.get("detail")) if x)
        elif not an.get("installed"):
            line = self.t["lib_need"].format(gb=an.get("size_gb", 3))
        elif min(an.get("facets", 0), an.get("clap", 0)) >= n:
            line = self.t["lib_idle"]
        else:
            line = ""
        self.lib_stage.configure(text=line)
        want_install = not an.get("installed") and not an.get("busy")
        if want_install and not self.install_btn.winfo_ismapped():
            self.install_btn.configure(text=self.t["lib_install"].format(gb=an.get("size_gb", 3)))
            self.install_btn.pack(side="right")
        elif not want_install and self.install_btn.winfo_ismapped():
            self.install_btn.pack_forget()
        self.run_btn.state(["disabled"] if an.get("busy") else ["!disabled"])

    def _post_local(self, path: str):
        req = urllib.request.Request(f"http://127.0.0.1:{self.cfg.port}{path}", data=b"", method="POST")
        threading.Thread(target=lambda: urllib.request.urlopen(req, timeout=5), daemon=True).start()

    def _analyse_now(self):
        self._post_local("/v1/local/analysis/run")

    def _install_analysis(self):
        from tkinter import messagebox

        if messagebox.askyesno(self.t["title"], self.t["lib_consent"].format(gb=3)):
            self.install_btn.pack_forget()
            self._post_local("/v1/local/analysis/install")

    # ---------- «Проверить и настроить сеть» ----------
    def _touch_lan(self) -> bool:
        """Обратиться к роутеру из окна. На macOS это вызывает запрос «Разрешить находить
        устройства в локальной сети?»: фоновая служба спросить не может, а разрешение даётся
        приложению целиком — и ей тоже. True — роутер ответил (разрешение есть)."""
        from arxnova import platforms, portmap

        p = platforms.current()
        local = p.local_ip()
        gw = p.gateway(local)
        if not (local and gw):
            return False
        try:
            portmap.natpmp_external(gw)
            return True
        except (portmap.MapError, OSError):
            pass
        try:
            portmap.upnp_find(local, timeout=1.5)
            return True
        except (portmap.MapError, OSError):
            return False

    def _net_check(self):
        self.net_btn.state(["disabled"])
        for w in self.net_box.winfo_children():
            w.destroy()
        self._label(self.net_box, self.t["net_running"], 12, MUTED).pack(anchor="w")
        threading.Thread(target=self._net_check_bg, daemon=True).start()

    def _net_check_bg(self):
        from arxnova import platforms, service

        t = self.t
        lines: list[tuple[str, str]] = []  # (знак, текст): ✓ хорошо, ✗ плохо, • к сведению
        # окно спрашивает у macOS доступ к локальной сети; если служба его раньше не имела —
        # перезапуск, чтобы она открыла порт уже с разрешением
        if self._touch_lan():
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.cfg.port}/v1/local/net", timeout=3) as r:
                    if not json.loads(r.read()).get("method"):
                        service.restart()
                        time.sleep(5)
            except Exception:
                pass
        else:
            lines.append(("•", t["net_lan_perm"]))
        # автозапуск — настраиваем сами, если выключен
        try:
            if service.status()[0]:
                lines.append(("✓", t["net_auto_ok"]))
            else:
                service.install(self.cfg.data_dir)
                lines.append(("✓", t["net_auto_fixed"]))
                time.sleep(4)  # служба поднимается
        except Exception as e:
            lines.append(("✗", t["err"].format(e=e)))
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{self.cfg.port}/v1/local/net/check", method="POST", data=b"")
            with urllib.request.urlopen(req, timeout=90) as r:
                n = json.loads(r.read())
        except Exception as e:
            n = None
            lines.append(("✗", t["net_fail"].format(e=e)))
        if n is not None and not n.get("remote"):
            lines.append(("•", t["net_remote_off"]))
        elif n is not None:
            ip, gw, port = n.get("local_ip"), n.get("gateway"), n.get("port")
            lines.append(("✓", t["net_lan_ok"].format(ip=ip, gw=gw)) if ip else ("✗", t["net_lan_bad"]))
            method = {"pcp": "PCP", "natpmp": "NAT-PMP", "upnp": "UPnP"}.get(n.get("method") or "")
            lines.append(("✓", t["net_port_ok"].format(port=port, method=method)) if method
                         else ("✗", t["net_port_bad"].format(port=port, ip=ip)))
            d = n.get("diagnosis")
            lines.append({
                "public": ("✓", t["net_ip_public"].format(ip=n.get("public_ip"))),
                "manual": ("•", t["net_ip_public"].format(ip=n.get("public_ip"))),
                "cgnat": ("✗", t["net_ip_cgnat"]),
                "double_nat": ("✗", t["net_ip_double"].format(router=n.get("router_ip"), ip=n.get("public_ip"), port=port)),
            }.get(d, ("✗", t["net_ip_offline"])))
            if d in ("public", "manual"):
                lines.append(("✓", t["net_loop_ok"]) if n.get("loopback") else ("•", t["net_loop_na"]))
            lines.append(("✓", t["net_dht_ok"].format(n=n.get("dht_nodes"))) if n.get("dht_nodes") else ("✗", t["net_dht_bad"]))
            tun = n.get("tunnel")
            if tun is not None:
                if tun.get("dht_nodes"):
                    lines.append(("✓", t["net_tunnel_ok"].format(n=tun["dht_nodes"])))
                else:
                    lines.append(("•", t["net_tunnel_wait"]))
        fw = platforms.current().firewall(self.cfg.remote_port)
        if fw:
            lines.append(("✓" if fw[0] == "ok" else "✗", t["net_fw"].format(text=fw[1])))
        if self.cfg.keep_awake and platforms.current().key == "macos":
            lines.append(("✓", t["net_sleep_ok"]))

        def show():
            for w in self.net_box.winfo_children():
                w.destroy()
            for mark, text in lines:
                color = {"✓": "#5fbf6f", "✗": "#e07a6c"}.get(mark, MUTED)
                row = tk.Frame(self.net_box, bg=BG)
                row.pack(fill="x", anchor="w", pady=1)
                tk.Label(row, text=mark, bg=BG, fg=color, font=("Helvetica", 13, "bold"), width=2, anchor="n").pack(side="left", anchor="n")
                self._label(row, text, 12, TEXT, wrap=390).pack(side="left", anchor="w")
            self.net_btn.state(["!disabled"])
        self._ui(show)

    def _pair(self):
        from arxnova import auth

        c = db.connect(self.cfg.db_path)
        code = auth.new_pair_code(c)
        c.close()
        self.code_lbl.configure(text=f"{code[:3]} {code[3:]}")
        self.code_hint.configure(text=self.t["pair_hint"].format(server=self.cfg.server_name))
        self.root.after(auth.CODE_TTL_S * 1000, lambda: (self.code_lbl.configure(text=""), self.code_hint.configure(text="")))

    def _invite(self):
        win = tk.Toplevel(self.root, bg=BG, padx=18, pady=14)
        win.title(self.t["invite"])
        self._label(win, self.t["invite_name"], 13).pack(anchor="w")
        name = tk.Entry(win, bg=SURFACE, fg=TEXT, insertbackground=TEXT, relief="flat", font=("Helvetica", 14))
        name.pack(fill="x", pady=6)
        name.focus_set()

        def make():
            from arxnova import guests, platforms, stun, tlsid

            n = name.get().strip()
            if not n:
                return
            try:
                c = db.connect(self.cfg.db_path)
                inv = guests.create_invite(c, n, None, self.cfg.max_guests)
                c.close()
                ident = tlsid.ensure(self.cfg.data_dir, self.cfg.server_name)
                local = platforms.current().local_ip()
                public = stun.external_ip(local)
                net = {"tls": {"port": self.cfg.remote_port, "fingerprint": ident.fingerprint},
                       "external": f"{public}:{self.cfg.remote_port}" if public else None}
                link = guests.link(inv["code"], self.cfg.server_name, self.cfg.inviter, net,
                                   f"http://{local}:{self.cfg.port}" if local else None)
                self.root.clipboard_clear()
                self.root.clipboard_append(link)
                code = inv["code"][:4] + "-" + inv["code"][4:]
                self.msg.configure(text=self.t["invite_done"].format(name=n, code=code))
                win.destroy()
            except Exception as e:
                self.msg.configure(text=self.t["err"].format(e=e))
                win.destroy()

        ttk.Button(win, text=self.t["invite_make"], style="Brass.TButton", command=make).pack(fill="x", pady=(6, 0))

    def run(self):
        self.root.mainloop()


def main(data_dir: Path | None = None) -> int:
    App(data_dir).run()
    return 0
