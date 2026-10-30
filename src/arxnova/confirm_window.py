"""Окно на экране сервера: «Пустить новое устройство?» Печатает ответ (owner | guest | deny) и
закрывается. Запускается отдельным процессом из presence.ServerConfirm: служба сервера сама окон
не держит, а Tk в фоновом потоке на macOS не работает."""

from __future__ import annotations

import sys
import tkinter as tk


def main() -> int:
    server, name, check = (sys.argv[1:4] + ["", "", ""])[:3]
    answer = {"v": "deny"}
    root = tk.Tk()
    root.title("Arx Nova — новое устройство")
    root.configure(bg="#0b0c0e", padx=28, pady=24)
    root.attributes("-topmost", True)
    fg, brass = "#f2efe8", "#c8a96a"
    tk.Label(root, text=f"«{name}» просит доступ к медиатеке «{server}»", fg=fg, bg="#0b0c0e",
             font=("Helvetica", 16), wraplength=520, justify="left").pack(anchor="w")
    tk.Label(root, text=f"Цифры для сверки: {check}", fg=brass, bg="#0b0c0e",
             font=("Helvetica", 28, "bold")).pack(anchor="w", pady=(14, 4))
    tk.Label(root, text="Они должны совпадать с экраном нового устройства.", fg=fg, bg="#0b0c0e",
             font=("Helvetica", 12)).pack(anchor="w", pady=(0, 18))

    def pick(v: str) -> None:
        answer["v"] = v
        root.destroy()

    row = tk.Frame(root, bg="#0b0c0e")
    row.pack(anchor="w")
    for text, v in (("Это моё устройство", "owner"), ("Пустить гостем", "guest"), ("Отказать", "deny")):
        tk.Button(row, text=text, command=lambda v=v: pick(v), font=("Helvetica", 13), padx=12, pady=6).pack(
            side="left", padx=(0, 10))
    root.protocol("WM_DELETE_WINDOW", lambda: pick("deny"))
    root.mainloop()
    print(answer["v"], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
