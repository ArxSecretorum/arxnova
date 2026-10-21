"""Настройки: всё, что выбрано, лежит в одном TOML, чтобы установку можно было повторить
(ТЗ, УСТ-3, НФТ-7). Пути — из настроек, а не зашиты под одну ОС (ТЗ, ОП-7)."""

from __future__ import annotations

import os
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_PORT = 8765


def default_data_dir() -> Path:
    env = os.environ.get("ARXNOVA_DATA_DIR")
    if env:
        return Path(env).expanduser()
    home = Path.home()
    if sys.platform == "darwin":
        return home / "Library" / "Application Support" / "arxnova"
    if sys.platform == "win32":
        return Path(os.environ.get("LOCALAPPDATA", home / "AppData" / "Local")) / "arxnova"
    return Path(os.environ.get("XDG_DATA_HOME", home / ".local" / "share")) / "arxnova"


@dataclass
class Config:
    data_dir: Path
    libraries: list[Path] = field(default_factory=list)
    port: int = DEFAULT_PORT
    # Потоки ONNX Runtime при индексации. На Pi — 1 (ТЗ, §4.2); на Mac можно больше.
    index_threads: int = 1
    # Имя медиатеки, которое видят плеер и гости.
    server_name: str = ""
    # Доступ из интернета (ОД-3, ОД-3а): отдельный порт только с TLS. 0 — выбрать при установке.
    remote: bool = True
    remote_port: int = 0
    # Не давать компьютеру засыпать, пока работает сервер и подключено питание (профиль macOS).
    keep_awake: bool = True
    # Общий доступ (ОД-7): гостей не больше 5 при любом значении; суточный объём гостя, МБ (0 — без предела).
    max_guests: int = 5
    guest_daily_mb: int = 4096
    # Как владельца видят гости в приглашении («кто пригласил»). Пусто — имя медиатеки.
    owner_name: str = ""

    def pick_remote_port(self) -> bool:
        """Порт для интернета выбирается один раз случайно из 20 000–60 000 (ОД-3а)."""
        if self.remote_port:
            return False
        import secrets

        while True:
            p = 20000 + secrets.randbelow(40000)
            if p != self.port:
                self.remote_port = p
                return True

    def __post_init__(self):
        if not self.server_name:
            import socket

            self.server_name = socket.gethostname().split(".")[0]

    @property
    def inviter(self) -> str:
        """«Кто пригласил» — только то имя, которое владелец сам указал (owner_name). Из ОС
        ничего не берём: настоящее имя не должно без спроса уходить в ссылки-приглашения."""
        return self.owner_name.strip()

    @property
    def path(self) -> Path:
        return self.data_dir / "arxnova.toml"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "db" / "arxnova.sqlite3"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"

    @property
    def tmp_dir(self) -> Path:
        return self.data_dir / "tmp"

    def ensure_dirs(self) -> None:
        for d in (self.db_path.parent, self.models_dir, self.index_dir, self.tmp_dir):
            d.mkdir(parents=True, exist_ok=True)

    def save(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        libs = ", ".join(_toml_str(str(p)) for p in self.libraries)
        self.path.write_text(
            "# Arx Nova — настройки. Файл можно править руками.\n"
            f"libraries = [{libs}]\n"
            f"port = {self.port}\n"
            f"index_threads = {self.index_threads}\n"
            f"server_name = {_toml_str(self.server_name)}\n"
            "# Доступ из интернета: порт только с шифрованием, роутер открывает его сам\n"
            f"remote = {'true' if self.remote else 'false'}\n"
            f"remote_port = {self.remote_port}\n"
            f"keep_awake = {'true' if self.keep_awake else 'false'}\n"
            "# Гости: не больше 5 (больше нельзя), суточный объём на гостя в МБ\n"
            f"max_guests = {self.max_guests}\n"
            f"guest_daily_mb = {self.guest_daily_mb}\n"
            f"owner_name = {_toml_str(self.owner_name)}\n",
            encoding="utf-8",
        )


def _toml_str(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def load(data_dir: Path | None = None) -> Config:
    data_dir = data_dir or default_data_dir()
    cfg = Config(data_dir=data_dir)
    if cfg.path.exists():
        raw = tomllib.loads(cfg.path.read_text(encoding="utf-8"))
        cfg.libraries = [Path(p).expanduser() for p in raw.get("libraries", [])]
        cfg.port = int(raw.get("port", DEFAULT_PORT))
        cfg.index_threads = int(raw.get("index_threads", 1))
        cfg.remote = bool(raw.get("remote", True))
        cfg.remote_port = int(raw.get("remote_port", 0))
        cfg.keep_awake = bool(raw.get("keep_awake", True))
        cfg.max_guests = int(raw.get("max_guests", 5))
        cfg.guest_daily_mb = int(raw.get("guest_daily_mb", 4096))
        cfg.owner_name = str(raw.get("owner_name", ""))
        if raw.get("server_name"):
            cfg.server_name = str(raw["server_name"])
    return cfg
