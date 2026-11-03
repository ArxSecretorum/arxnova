"""Звук форматов, которые не читает libsndfile (ALAC, AAC, WMA, DSD, APE, WavPack, TTA): через
отдельную программу ffmpeg, по трубе. FLAC, WAV, AIFF, OGG, Opus и MP3 идут прежним путём (soundfile).

Почему не библиотека. Готовые колёса PyAV объявлены как BSD, но их libavcodec связана с libx264 и
libx265 под GPL (проверено на av 19.0.2, 03.11.2026): поставлять такое с сервером нельзя, решение
0008. ffmpeg — самостоятельная программа пользователя: мы её не поставляем и с ней не связываемся,
а только передаём ей путь и читаем отсчёты из трубы. Заодно её память не считается в бюджете
сервера. Без ffmpeg новые форматы в медиатеке есть и играют, не разбирается только их звук."""

from __future__ import annotations

import os
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
from collections.abc import Iterator
from pathlib import Path

import numpy as np

# расширения, которые читаются только через ffmpeg (остальное в audiokey.EXTENSIONS — soundfile)
NEEDS_FFMPEG = {".m4a", ".alac", ".aac", ".caf", ".wma", ".ape", ".wv", ".tta", ".dsf", ".dff"}
# окно приложения на Mac не видит PATH оболочки: туда ffmpeg ставит Homebrew
_SEARCH = ("/opt/homebrew/bin", "/usr/local/bin", "/usr/bin")
BLOCK_S = 2.0  # кусками, как features.load_mono: хайрез и DSD целиком — сотни мегабайт
TIMEOUT_S = 1800  # зависший на битом файле ffmpeg не должен останавливать разбор навсегда


class DecoderMissing(RuntimeError):
    """ffmpeg не найден: разбор этого файла отложен до его установки."""


class DecodeError(RuntimeError):
    pass


def ffmpeg_path() -> str | None:
    """Без кэша: пользователь мог поставить ffmpeg, пока сервер работает."""
    explicit = os.environ.get("ARXNOVA_FFMPEG")
    if explicit:
        return explicit if os.path.isfile(explicit) else None
    found = shutil.which("ffmpeg")
    if found:
        return found
    for d in _SEARCH:
        p = os.path.join(d, "ffmpeg")
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    return None


def needs_ffmpeg(path) -> bool:
    return Path(path).suffix.lower() in NEEDS_FFMPEG


def can_decode(path) -> bool:
    return not needs_ffmpeg(path) or ffmpeg_path() is not None


def decodable(rows: list, path_index: int = 1) -> list:
    """Строки очереди разбора без файлов, которые сейчас нечем прочитать: иначе каждый проход
    заново спотыкался бы о них, а счётчик «разобрано» никогда не сходился."""
    if ffmpeg_path() is not None:
        return rows
    return [r for r in rows if not needs_ffmpeg(r[path_index])]


def waiting_count(conn) -> int:
    """Сколько треков ждут ffmpeg: их файл читается только через него, а его нет."""
    if ffmpeg_path() is not None:
        return 0
    rows = conn.execute("SELECT t.id, min(p.path) FROM tracks t JOIN track_paths p ON p.track_id = t.id GROUP BY t.id")
    return sum(1 for _, path in rows if needs_ffmpeg(path))


def _wav_header(pipe) -> tuple[int, int]:
    """Частота и каналы из заголовка WAV в начале трубы; труба остаётся на первом отсчёте.
    ffmpeg сам сообщает, что вышло из декодера: у DSD частота на выходе своя, а не из файла."""
    head = pipe.read(12)
    if len(head) < 12 or head[:4] != b"RIFF" or head[8:12] != b"WAVE":
        raise DecodeError("ffmpeg не отдал звук")
    rate = channels = 0
    while True:
        hdr = pipe.read(8)
        if len(hdr) < 8:
            raise DecodeError("ffmpeg: заголовок без данных")
        cid, clen = hdr[:4], struct.unpack("<I", hdr[4:8])[0]
        if cid == b"data":
            if not rate or not channels:
                raise DecodeError("ffmpeg: нет описания звука")
            return rate, channels
        body = pipe.read(clen + (clen & 1))
        if cid == b"fmt " and len(body) >= 8:
            channels, rate = struct.unpack("<HI", body[2:8])


def read_blocks(path, start_s: float = 0.0, duration_s: float | None = None) -> tuple[int, int, Iterator[np.ndarray]]:
    """(частота, каналы, куски float32 формы (кадры, каналы)). start_s/duration_s — участок файла:
    ffmpeg читает только его. Весь поток ffmpeg живёт, пока читаются куски."""
    exe = ffmpeg_path()
    if not exe:
        raise DecoderMissing("нужен ffmpeg")
    cmd = [exe, "-nostdin", "-v", "error"]
    if start_s > 0:
        cmd += ["-ss", f"{start_s:.6f}"]  # до -i: быстрый переход, точность добирает сам ffmpeg
    cmd += ["-i", str(path)]
    if duration_s is not None:
        cmd += ["-t", f"{max(0.0, duration_s):.6f}"]
    cmd += ["-map", "0:a:0", "-c:a", "pcm_f32le", "-f", "wav", "-bitexact", "pipe:1"]
    err = tempfile.TemporaryFile()  # не труба: сообщений об ошибках может быть больше её буфера
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err, stdin=subprocess.DEVNULL, creationflags=flags)
    watchdog = threading.Timer(TIMEOUT_S, proc.kill)
    watchdog.daemon = True
    watchdog.start()

    def stop() -> int:
        watchdog.cancel()
        proc.stdout.close()
        try:
            return proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            return proc.wait()

    def fail_text() -> str:
        err.seek(0)
        text = err.read().decode("utf-8", "replace").strip().splitlines()
        err.close()
        return (text[-1] if text else "без сообщения")[:200]

    try:
        rate, channels = _wav_header(proc.stdout)
    except Exception as e:
        stop()
        raise DecodeError(f"{e}: {fail_text()}") from e

    def blocks() -> Iterator[np.ndarray]:
        size = int(rate * BLOCK_S) * channels * 4
        finished = False
        try:
            while True:
                raw = proc.stdout.read(size)
                if not raw:
                    finished = True
                    break
                n = len(raw) // (channels * 4)
                if n:
                    yield np.frombuffer(raw, "<f4", n * channels).reshape(n, channels)
        finally:
            code = stop()
            if not finished:  # читатель остановился раньше конца: это не ошибка файла
                err.close()
            elif code != 0:
                raise DecodeError(f"ffmpeg: {fail_text()}")
            else:
                err.close()

    return rate, channels, blocks()
