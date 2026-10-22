"""База SQLite (WAL): каталог треков. События, устройства и решения шлюза добавятся
следующими миграциями (ТЗ, §5.2, §7)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

# Каждая миграция переводит базу из версии i в i+1 (PRAGMA user_version).
MIGRATIONS = [
    """
    CREATE TABLE tracks (
        id            INTEGER PRIMARY KEY,
        md5           BLOB    NOT NULL,      -- ключ файла: MD5 из STREAMINFO (или PCM)
        cue_ms        INTEGER NOT NULL DEFAULT 0,
        md5_source    TEXT    NOT NULL,      -- 'streaminfo' | 'pcm'
        recording_key BLOB    NOT NULL,      -- ключ записи (ФТ-2.10)
        recording_basis TEXT  NOT NULL,      -- по чему узнаётся запись (keys.recording_basis)
        sample_rate   INTEGER NOT NULL,
        bits          INTEGER NOT NULL,
        channels      INTEGER NOT NULL,
        duration      REAL    NOT NULL,
        artist        TEXT,
        albumartist   TEXT,
        album         TEXT,
        title         TEXT,
        date          TEXT,
        genre_raw     TEXT,
        tracknumber   TEXT,
        mb_recording  TEXT,
        mb_album      TEXT,
        compilation   INTEGER NOT NULL DEFAULT 0,
        added_at      INTEGER NOT NULL DEFAULT (unixepoch()),
        UNIQUE (md5, cue_ms)
    );
    -- Один звук по двум путям — один трек (ТЗ, §5.4): пути отдельно.
    CREATE TABLE files (
        path     TEXT    PRIMARY KEY,
        track_id INTEGER NOT NULL REFERENCES tracks(id) ON DELETE CASCADE,
        size     INTEGER NOT NULL,
        mtime_ns INTEGER NOT NULL
    );
    CREATE INDEX files_track ON files(track_id);
    CREATE INDEX tracks_recording ON tracks(recording_key);
    CREATE INDEX tracks_recording_basis ON tracks(recording_basis);
    """,
    """
    -- Эмбеддинги: по строке на трек и модель. Единица работы индексации — трек (ФТ-2.6).
    CREATE TABLE embeddings (
        track_id INTEGER NOT NULL REFERENCES tracks(id) ON DELETE CASCADE,
        model    TEXT    NOT NULL,
        vector   BLOB    NOT NULL,   -- float32
        styles   BLOB,               -- float32, активации стилей (если модель их даёт)
        windows  INTEGER NOT NULL,
        PRIMARY KEY (track_id, model)
    );
    -- Устройства: владелец и (этап 1.5) гости. Токен хранится только как хеш (ОД-9).
    CREATE TABLE devices (
        id         INTEGER PRIMARY KEY,
        name       TEXT    NOT NULL,
        role       TEXT    NOT NULL DEFAULT 'owner',
        token_hash BLOB    NOT NULL UNIQUE,
        created_at INTEGER NOT NULL DEFAULT (unixepoch()),
        last_seen  INTEGER,
        revoked    INTEGER NOT NULL DEFAULT 0
    );
    CREATE TABLE pair_codes (
        code_hash  BLOB    PRIMARY KEY,
        role       TEXT    NOT NULL DEFAULT 'owner',
        expires_at INTEGER NOT NULL,
        used       INTEGER NOT NULL DEFAULT 0
    );
    -- Журнал событий М1, схема совместима с Yambda (ТЗ, §7).
    CREATE TABLE events (
        event_id         TEXT    PRIMARY KEY,   -- UUID с устройства: повтор не создаёт дубль
        device_id        INTEGER NOT NULL REFERENCES devices(id),
        ts               INTEGER NOT NULL,      -- UTC, мс
        md5              BLOB    NOT NULL,
        cue_ms           INTEGER NOT NULL DEFAULT 0,
        track_id         INTEGER,               -- null: трека пока нет в медиатеке
        event_type       TEXT    NOT NULL,      -- listen, skip, like, unlike, dislike, undislike
        played_ratio_pct INTEGER,
        track_length_s   INTEGER,
        is_organic       INTEGER NOT NULL DEFAULT 1,
        session_id       INTEGER,
        prev_track_id    INTEGER,
        transition_type  TEXT,
        nova_request_id  INTEGER,
        recording_key    BLOB,
        skip_reason      TEXT
    );
    CREATE INDEX events_device_ts ON events(device_id, ts);
    CREATE INDEX events_track ON events(track_id);
    -- Решения подбора: что подано и почему (М8, объяснимость).
    CREATE TABLE nova_requests (
        id         INTEGER PRIMARY KEY,
        device_id  INTEGER NOT NULL,
        ts         INTEGER NOT NULL,
        seed_track INTEGER,
        result     TEXT    NOT NULL   -- JSON: треки и причины
    );
    """,
    """
    -- Каталог прибора (П-2): какие треки лежат на устройстве (по MD5 звука).
    CREATE TABLE device_catalog (
        device_id INTEGER NOT NULL REFERENCES devices(id) ON DELETE CASCADE,
        md5       BLOB    NOT NULL,
        PRIMARY KEY (device_id, md5)
    );
    """,
    """
    -- CUE: у файла-образа несколько треков (общий MD5, разные cue_ms). files.track_id указывает
    -- на трек с cue_ms = 0, остальные треки образа находятся по MD5.
    ALTER TABLE tracks ADD COLUMN cue_end_ms INTEGER NOT NULL DEFAULT -1;
    ALTER TABLE files ADD COLUMN cue_sig TEXT;
    CREATE VIEW track_paths AS
        SELECT t.id AS track_id, f.path AS path
        FROM tracks t
        JOIN tracks r ON r.md5 = t.md5 AND r.cue_ms = 0
        JOIN files f ON f.track_id = r.id;
    """,
    """
    -- Общий доступ (ТЗ, §15.1): поимённые одноразовые приглашения, гости, их расход трафика.
    CREATE TABLE invites (
        id         INTEGER PRIMARY KEY,
        code_hash  BLOB    NOT NULL UNIQUE,
        guest_name TEXT    NOT NULL,             -- кому выдано (ОД-10, правило 1)
        created_by INTEGER,
        created_at INTEGER NOT NULL DEFAULT (unixepoch()),
        expires_at INTEGER NOT NULL,
        used_at    INTEGER,
        device_id  INTEGER,                      -- гость, который вошёл по коду
        revoked    INTEGER NOT NULL DEFAULT 0
    );
    ALTER TABLE devices ADD COLUMN guest_name TEXT;
    CREATE TABLE guest_usage (
        device_id INTEGER NOT NULL,
        day       TEXT    NOT NULL,              -- YYYY-MM-DD по времени сервера
        bytes     INTEGER NOT NULL DEFAULT 0,
        PRIMARY KEY (device_id, day)
    );
    """,
    """
    -- Заготовленные подборки прибора (ФТ-4.7): кэш на сутки, пересчёт при смене каталога.
    CREATE TABLE device_mixes (
        device_id INTEGER PRIMARY KEY,
        built_at  INTEGER NOT NULL,
        stamp     TEXT    NOT NULL,
        data      TEXT    NOT NULL
    );
    """,
    """
    -- Фасеты (ФТ-2.3, ФТ-2.4): эмбеддинги стемов и явные признаки. Считаются на настольной
    -- машине (дополнение [stems]); аудио стемов не хранится.
    CREATE TABLE facet_vectors (
        track_id INTEGER NOT NULL REFERENCES tracks(id) ON DELETE CASCADE,
        facet    TEXT    NOT NULL,               -- rhythm | instr | vocal
        model    TEXT    NOT NULL,
        vector   BLOB    NOT NULL,
        PRIMARY KEY (track_id, facet)
    );
    CREATE TABLE track_features (
        track_id    INTEGER PRIMARY KEY REFERENCES tracks(id) ON DELETE CASCADE,
        version     INTEGER NOT NULL,
        bpm         REAL,
        bpm_conf    REAL,
        bpm_src     TEXT,
        musical_key INTEGER,                     -- 0–11 мажор от C, 12–23 минор
        key_conf    REAL,
        loudness_db REAL,
        onset_rate  REAL,
        vocal_frac  REAL,
        vocal_ratio REAL,
        computed_at INTEGER NOT NULL DEFAULT (unixepoch())
    );
    """,
    """
    -- Журнал решений шлюза (ФТ-8.3): построчно — принятые и отклонённые после T_mix;
    -- остальные отказы — счётчиками на запрос. Построчное хранится 90 дней.
    CREATE TABLE gate_log (
        id         INTEGER PRIMARY KEY,
        request_id INTEGER,
        ts         INTEGER NOT NULL,
        from_track INTEGER NOT NULL,
        to_track   INTEGER NOT NULL,
        accepted   INTEGER NOT NULL,
        failed     TEXT,                         -- условие отказа (gate.COND_*)
        detail     TEXT,                         -- какой фасет и близости — JSON
        mix        REAL
    );
    CREATE INDEX gate_log_ts ON gate_log(ts);
    ALTER TABLE nova_requests ADD COLUMN gate_counts TEXT;  -- {"mix": 37, "pair": 2, …}
    """,
    """
    -- Разметка мостов (КО-1): пары-кандидаты и оценки владельца «уместно / неуместно».
    CREATE TABLE bridge_candidates (
        a   INTEGER NOT NULL,                    -- играет первым
        b   INTEGER NOT NULL,
        mix REAL    NOT NULL,
        ord INTEGER NOT NULL,
        PRIMARY KEY (a, b)
    );
    CREATE TABLE bridge_labels (
        a     INTEGER NOT NULL,
        b     INTEGER NOT NULL,
        label TEXT    NOT NULL,                  -- good | bad | skip
        ts    INTEGER NOT NULL,
        PRIMARY KEY (a, b)
    );
    """,
    """
    -- CLAP «текст → звук» для сценариев (ФТ-4.10): вектор трека, оригинальные веса LAION.
    CREATE TABLE clap_vectors (
        track_id INTEGER PRIMARY KEY REFERENCES tracks(id) ON DELETE CASCADE,
        vector   BLOB    NOT NULL
    );
    """,
    """
    -- Сопряжение через интернет: секрет, выведенный из кода (ключи поиска в DHT и доказательство).
    ALTER TABLE pair_codes ADD COLUMN secret BLOB;
    """,
]


def connect(path: Path, *, threaded: bool = False) -> sqlite3.Connection:
    """threaded=True — одно соединение на весь сервер; SQLite собран в режиме serialized,
    запись дополнительно сериализуется блокировкой сервера."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=not threaded)
    conn.row_factory = sqlite3.Row
    # обход медиатеки и сервер пишут из разных соединений: подождать, а не падать с «locked»
    conn.execute("PRAGMA busy_timeout = 15000")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    migrate(conn)
    return conn


def _statements(script: str):
    """Скрипт миграции по одной команде (executescript сам делает COMMIT — внутри своей
    транзакции им пользоваться нельзя)."""
    buf = ""
    for line in script.splitlines(keepends=True):
        buf += line
        if sqlite3.complete_statement(buf):
            if buf.strip():
                yield buf
            buf = ""
    if buf.strip() and not buf.strip().startswith("--"):
        yield buf


def migrate(conn: sqlite3.Connection) -> None:
    """Обновление схемы. Базу могут открыть сразу двое (сервер и обход медиатеки, окно
    приложения): каждый шаг — под захваченной на запись транзакцией с повторной проверкой
    версии, иначе второй получал «database is locked» или «table already exists»."""
    if conn.execute("PRAGMA user_version").fetchone()[0] >= len(MIGRATIONS):
        return
    for i in range(len(MIGRATIONS)):
        conn.execute("BEGIN IMMEDIATE")  # ждёт по busy_timeout, пока другой закончит
        try:
            if conn.execute("PRAGMA user_version").fetchone()[0] > i:
                conn.execute("COMMIT")
                continue
            for stmt in _statements(MIGRATIONS[i]):
                conn.execute(stmt)
            conn.execute(f"PRAGMA user_version = {i + 1}")
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
