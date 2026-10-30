"""Файлы не-FLAC в медиатеке: WAV, AIFF, OGG (Vorbis/Opus), MP3 — ключ, сведения о звуке, теги.

Ключ трека по ТЗ (ФТ-2.1) — MD5 звука. У FLAC он записан в заголовке; у остальных форматов его
нет, а декодировать каждый файл целиком — часы на тысячи треков. Поэтому ключ считается по байтам
самого аудиопотока, без тегов, и выборкой: первые и последние SAMPLE байт потока плюс его длина.
Цена — одно чтение ~1 МБ на файл, как у заголовка FLAC на жёстком диске; перезапись тегов (ID3,
Vorbis comment, LIST INFO) ключ не меняет, потому что теги в поток не входят. Один и тот же звук,
скопированный по другому пути, даёт тот же ключ; другой кодек или другое издание — другой.

Сведения о звуке (частота, каналы, разрядность, длительность) и теги (ID3 у MP3/WAV/AIFF, Vorbis
comment у OGG/Opus, LIST INFO у WAV) — tinytag (MIT, чистый Python, читает только заголовки;
mutagen не подходит — GPL).
Владелец 30.10.2026: «нужна скорость, сопоставимая с индексацией FLAC, или быстрее»."""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, field
from pathlib import Path

SAMPLE = 512 * 1024
# расширения, которые сканер берёт в медиатеку (FLAC — своим путём, flac.py)
EXTENSIONS = {".flac", ".wav", ".wave", ".rf64", ".aif", ".aiff", ".aifc", ".ogg", ".oga", ".opus", ".mp3"}
LOSSY = {".ogg", ".oga", ".opus", ".mp3"}


class AudioError(RuntimeError):
    pass


@dataclass
class AudioInfo:
    """Та же поверхность, что у flac.FlacInfo: сканер не различает форматы."""

    sample_rate: int
    channels: int
    bits: int  # 0 — у сжатых с потерями исходной разрядности нет
    duration: float
    md5: bytes
    md5_source: str = "bytes"
    tags: dict[str, list[str]] = field(default_factory=dict)

    @property
    def md5_is_zero(self) -> bool:
        return False

    def tag(self, name: str) -> str | None:
        vals = self.tags.get(name.upper())
        return vals[0] if vals else None


def is_audio(path: Path) -> bool:
    return path.suffix.lower() in EXTENSIONS and not path.name.startswith("._")


# ---------------------------------------------------------------- границы аудиопотока

def _wav_stream(f) -> tuple[int, int]:
    """(начало, длина) чанка data. RF64 — длина из ds64."""
    head = f.read(12)
    if len(head) < 12 or head[:4] not in (b"RIFF", b"RF64") or head[8:12] != b"WAVE":
        raise AudioError("не WAV")
    rf64_data = None
    pos = 12
    data = None
    f.seek(0, 2)
    size = f.tell()
    while pos + 8 <= size:
        f.seek(pos)
        hdr = f.read(8)
        if len(hdr) < 8:
            break
        cid, clen = hdr[:4], struct.unpack("<I", hdr[4:8])[0]
        if cid == b"ds64" and head[:4] == b"RF64":
            body = f.read(min(clen, 28))
            if len(body) >= 16:
                rf64_data = struct.unpack("<Q", body[8:16])[0]
        elif cid == b"data":
            length = rf64_data if (clen == 0xFFFFFFFF and rf64_data is not None) else clen
            data = (pos + 8, min(length, size - pos - 8))
        if clen == 0xFFFFFFFF:
            break
        pos += 8 + clen + (clen & 1)
    if data is None:
        raise AudioError("в WAV нет чанка data")
    return data


def _aiff_stream(f) -> tuple[int, int]:
    head = f.read(12)
    if len(head) < 12 or head[:4] != b"FORM" or head[8:12] not in (b"AIFF", b"AIFC"):
        raise AudioError("не AIFF")
    f.seek(0, 2)
    size = f.tell()
    pos = 12
    while pos + 8 <= size:
        f.seek(pos)
        hdr = f.read(8)
        if len(hdr) < 8:
            break
        cid, clen = hdr[:4], struct.unpack(">I", hdr[4:8])[0]
        if cid == b"SSND":
            off = struct.unpack(">I", f.read(4))[0]
            start = pos + 8 + 8 + off
            return start, max(0, min(clen - 8 - off, size - start))
        pos += 8 + clen + (clen & 1)
    raise AudioError("в AIFF нет чанка SSND")


def _mp3_stream(f) -> tuple[int, int]:
    """Поток между тегами: пропустить ID3v2 в начале (может быть несколько подряд), ID3v1 и APEv2
    в конце. Кадры Xing/LAME — часть потока, они у одного кодирования одинаковы."""
    f.seek(0, 2)
    size = f.tell()
    start = 0
    while True:
        f.seek(start)
        head = f.read(10)
        if len(head) == 10 and head[:3] == b"ID3":
            n = 0
            for b in head[6:10]:
                n = (n << 7) | (b & 0x7F)
            start += 10 + n + (10 if head[5] & 0x10 else 0)
        else:
            break
    end = size
    if end - start >= 128:
        f.seek(end - 128)
        if f.read(3) == b"TAG":
            end -= 128
    if end - start >= 32:
        f.seek(end - 32)
        foot = f.read(32)
        if foot[:8] == b"APETAGEX":
            n = struct.unpack("<I", foot[12:16])[0]
            flags = struct.unpack("<I", foot[20:24])[0]
            end -= n + (32 if flags & 0x80000000 else 0)
    return start, max(0, end - start)


def _ogg_pages(f, pos: int, limit: int):
    """Страницы Ogg от pos (pos должен указывать на «OggS»): (granule, payload) пока не кончатся
    или не набралось limit байт полезной нагрузки."""
    got = 0
    while got < limit:
        f.seek(pos)
        hdr = f.read(27)
        if len(hdr) < 27 or hdr[:4] != b"OggS":
            return
        granule = struct.unpack("<q", hdr[6:14])[0]
        nseg = hdr[26]
        segs = f.read(nseg)
        body = sum(segs)
        yield granule, f.read(body)
        got += body
        pos += 27 + nseg + body


def _ogg_key(f) -> tuple[bytes, int]:
    """MD5(первые SAMPLE байт полезной нагрузки аудиостраниц, длина аудиостраниц, последняя
    гранула) и последняя гранула (число отсчётов — для длительности). Заголовочные страницы
    (гранула 0) в ключ не входят: перезапись тегов меняет только их."""
    f.seek(0, 2)
    size = f.tell()
    f.seek(0)
    if f.read(4) != b"OggS":
        raise AudioError("не Ogg")
    h = hashlib.md5()
    audio_start = None
    got = 0
    pos = 0
    while got < SAMPLE:
        f.seek(pos)
        hdr = f.read(27)
        if len(hdr) < 27 or hdr[:4] != b"OggS":
            break
        granule = struct.unpack("<q", hdr[6:14])[0]
        nseg = hdr[26]
        segs = f.read(nseg)
        body = sum(segs)
        if granule > 0 or audio_start is not None:
            if audio_start is None:
                audio_start = pos
            data = f.read(body)
            h.update(data)
            got += body
        pos += 27 + nseg + body
    if audio_start is None:
        raise AudioError("в Ogg нет аудиостраниц")
    # последняя страница — последняя гранула (длительность)
    tail = max(0, size - 64 * 1024)
    f.seek(tail)
    chunk = f.read()
    last = chunk.rfind(b"OggS")
    granule = 0
    if last >= 0 and last + 14 <= len(chunk):
        granule = struct.unpack("<q", chunk[last + 6:last + 14])[0]
    h.update(struct.pack("<qq", size - audio_start, granule))
    return h.digest(), granule


def _sampled_md5(f, start: int, length: int) -> bytes:
    h = hashlib.md5()
    f.seek(start)
    h.update(f.read(min(SAMPLE, length)))
    if length > SAMPLE:
        f.seek(start + max(SAMPLE, length - SAMPLE))
        h.update(f.read(min(SAMPLE, length - SAMPLE)))
    h.update(struct.pack("<q", length))
    return h.digest()


# ---------------------------------------------------------------- теги и звук (tinytag)

_TAG_KEYS = {"artist": "ARTIST", "album": "ALBUM", "title": "TITLE", "year": "DATE", "genre": "GENRE",
             "track": "TRACKNUMBER", "albumartist": "ALBUMARTIST", "compilation": "COMPILATION",
             "tcmp": "COMPILATION", "musicbrainz_trackid": "MUSICBRAINZ_TRACKID",
             "musicbrainz_albumid": "MUSICBRAINZ_ALBUMID", "musicbrainz_releasetrackid": "MUSICBRAINZ_RELEASETRACKID"}


def _tinytag(path: Path, image: bool = False):
    from tinytag import TinyTag

    return TinyTag.get(str(path), image=image)


def _tags_of(t) -> dict[str, list[str]]:
    """Теги в ключах Vorbis comment (как у FLAC): сканер не различает форматы."""
    out: dict[str, list[str]] = {}
    for key, vals in t.as_dict().items():
        name = _TAG_KEYS.get(str(key).lower())
        if not name or vals in (None, "", []):
            continue
        for v in (vals if isinstance(vals, list) else [vals]):
            if v not in (None, ""):
                out.setdefault(name, []).append(str(v))
    return out


def read_picture(path: Path) -> tuple[str, bytes] | None:
    """Вложенная обложка (ID3 APIC, Vorbis METADATA_BLOCK_PICTURE) → (тип, байты)."""
    try:
        img = _tinytag(path, image=True).images.any
    except Exception:  # noqa: BLE001
        return None
    if img is None or not img.data:
        return None
    return img.mime_type or "image/jpeg", bytes(img.data)


# ---------------------------------------------------------------- главное

def read(path: str | Path) -> AudioInfo:
    """Ключ, звук и теги файла не-FLAC. Ошибка формата — AudioError (сканер пишет её в отчёт)."""
    p = Path(path)
    ext = p.suffix.lower()
    granule = 0
    with open(p, "rb") as f:
        if ext in (".wav", ".wave", ".rf64"):
            start, length = _wav_stream(f)
            md5 = _sampled_md5(f, start, length)
        elif ext in (".aif", ".aiff", ".aifc"):
            start, length = _aiff_stream(f)
            md5 = _sampled_md5(f, start, length)
        elif ext == ".mp3":
            start, length = _mp3_stream(f)
            md5 = _sampled_md5(f, start, length)
        elif ext in (".ogg", ".oga", ".opus"):
            md5, granule = _ogg_key(f)
        else:
            raise AudioError(f"формат не поддерживается: {ext}")
    try:
        t = _tinytag(p)
    except Exception as e:  # noqa: BLE001
        raise AudioError(f"tinytag: {e}") from e
    if not t.samplerate or not t.channels:
        raise AudioError("tinytag: нет частоты или каналов")
    bits = 0 if ext in LOSSY else int(t.bitdepth or 0)
    duration = float(t.duration or 0.0)
    if ext == ".opus" and granule > 0:
        duration = granule / 48000.0  # у Opus гранулы всегда в 48 кГц
    return AudioInfo(int(t.samplerate), int(t.channels), bits, duration, md5, "bytes", _tags_of(t))
