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
    "pair_hint": "Введите в Arx Player: Настройки → Arx Nova. Код действует 5 минут.",
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
    "pair_hint": "Enter it in Arx Player: Settings → Arx Nova. Valid for 5 minutes.",
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
        if self.cfg.libraries and self.cfg.path.exists():
            self._main_screen()
        else:
            self._first_run()

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
        self.root.after(0, lambda: self.msg.configure(text=text))

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
            self.root.after(0, self._main_screen)
        except Exception as e:  # покажем по-человечески, кнопка снова доступна
            self._say(self.t["err"].format(e=e))
            self.root.after(0, lambda: self.start_btn.state(["!disabled"]))

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
        ttk.Button(row, text=self.t["folder_change"], style="Flat.TButton", command=self._first_run).pack(side="right")
        self.msg = self._label(self.body, "", 12, BRASS)
        self.msg.pack(anchor="w", pady=(10, 0))
        self._label(self.body, self.t["close_hint"], 11, MUTED).pack(anchor="w", side="bottom", pady=(14, 0))
        self._restarted_at = 0.0
        self._poll()

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
        if not ok and time.monotonic() - self._restarted_at > 60:
            # служба легла или ещё не поднялась — один толчок в минуту
            from arxnova import service

            self._restarted_at = time.monotonic()
            try:
                service.restart() if service.status()[0] else service.install(self.cfg.data_dir)
            except Exception:
                pass
        pct = int(analysed * 100 / tracks) if tracks else 0

        def show():
            self.dot.configure(fg="#5fbf6f" if ok else BRASS)
            self.state_lbl.configure(text=self.t["working"] if ok else self.t["stopped"])
            self.stats_lbl.configure(text=self.t["stats"].format(tracks=tracks, pct=pct))
        self.root.after(0, show)

    def _pair(self):
        from arxnova import auth

        c = db.connect(self.cfg.db_path)
        code = auth.new_pair_code(c)
        c.close()
        self.code_lbl.configure(text=f"{code[:3]} {code[3:]}")
        self.code_hint.configure(text=self.t["pair_hint"])
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
