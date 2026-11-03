"""Файлы не-FLAC в медиатеке: WAV, AIFF, OGG (Vorbis/Opus), MP3, ALAC и AAC (m4a), WMA, DSD, APE,
WavPack, TTA — ключ, сведения о звуке, теги.

Ключ трека по ТЗ (ФТ-2.1) — MD5 звука. У FLAC он записан в заголовке; у остальных форматов его
нет, а декодировать каждый файл целиком — часы на тысячи треков. Поэтому ключ считается по байтам
самого аудиопотока, без тегов, и выборкой: первые и последние SAMPLE байт потока плюс его длина.
Цена — одно чтение ~1 МБ на файл, как у заголовка FLAC на жёстком диске; перезапись тегов (ID3,
Vorbis comment, LIST INFO) ключ не меняет, потому что теги в поток не входят. Один и тот же звук,
скопированный по другому пути, даёт тот же ключ; другой кодек или другое издание — другой.

Сведения о звуке (частота, каналы, разрядность, длительность) и теги (ID3 у MP3/WAV/AIFF, Vorbis
comment у OGG/Opus, LIST INFO у WAV, атомы M4A, ASF у WMA) — tinytag (MIT, чистый Python, читает
только заголовки; mutagen не подходит — GPL). APE, WavPack, TTA, DSF, DFF, CAF и голый AAC tinytag
не знает: их заголовки и теги разбираются здесь же. Индексации ffmpeg не нужен (decode.py — только
для разбора звука), и цена та же: одно чтение ~1 МБ на файл.
Владелец 30.10.2026: «нужна скорость, сопоставимая с индексацией FLAC, или быстрее»."""

from __future__ import annotations

import hashlib
import io
import struct
from dataclasses import dataclass, field
from pathlib import Path

SAMPLE = 512 * 1024
# расширения, которые сканер берёт в медиатеку (FLAC — своим путём, flac.py)
EXTENSIONS = {".flac", ".wav", ".wave", ".rf64", ".aif", ".aiff", ".aifc", ".ogg", ".oga", ".opus", ".mp3",
              ".m4a", ".alac", ".aac", ".caf", ".wma", ".ape", ".wv", ".tta", ".dsf", ".dff"}
LOSSY = {".ogg", ".oga", ".opus", ".mp3", ".aac"}  # у m4a, caf и wma зависит от кодека внутри


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


# ---------------------------------------------------------------- MP4 (ALAC, AAC)

def _boxes(f, start: int, end: int):
    """Ящики ISO BMFF в [start, end): (тип, начало тела, конец)."""
    pos = start
    while pos + 8 <= end:
        f.seek(pos)
        hdr = f.read(8)
        if len(hdr) < 8:
            return
        size, kind = int.from_bytes(hdr[:4], "big"), hdr[4:8]
        body = pos + 8
        if size == 1:  # длина не влезла в 32 бита
            size = int.from_bytes(f.read(8), "big")
            body = pos + 16
        elif size == 0:  # до конца файла
            size = end - pos
        if size < body - pos:
            return
        yield kind, body, min(pos + size, end)
        pos += size


def _mp4_stream(f) -> tuple[int, int]:
    """Самый большой ящик mdat: там лежат кадры. Теги и обложка — в moov и при перезаписи не трогают mdat."""
    f.seek(0, 2)
    size = f.tell()
    f.seek(4)
    if f.read(4) != b"ftyp":
        raise AudioError("не MP4")
    best = None
    for kind, body, end in _boxes(f, 0, size):
        if kind == b"mdat" and (best is None or end - body > best[1]):
            best = (body, end - body)
    if best is None:
        raise AudioError("в MP4 нет mdat")
    return best


def _child(f, parent: tuple[int, int], name: bytes) -> tuple[int, int] | None:
    for kind, body, end in _boxes(f, *parent):
        if kind == name:
            return body, end
    return None


def _mp4_codec(f) -> str:
    """Кодек первой звуковой дорожки: alac, mp4a, ec-3… Плеер играет не всё, что бывает в m4a."""
    f.seek(0, 2)
    moov = _child(f, (0, f.tell()), b"moov")
    if moov is None:
        raise AudioError("в MP4 нет moov")
    for kind, body, end in _boxes(f, *moov):
        if kind != b"trak":
            continue
        mdia = _child(f, (body, end), b"mdia")
        hdlr = mdia and _child(f, mdia, b"hdlr")
        if hdlr:
            f.seek(hdlr[0] + 8)
            if f.read(4) != b"soun":
                continue
        minf = mdia and _child(f, mdia, b"minf")
        stbl = minf and _child(f, minf, b"stbl")
        stsd = stbl and _child(f, stbl, b"stsd")
        if stsd:
            f.seek(stsd[0] + 12)
            return f.read(4).decode("latin-1").strip()
    raise AudioError("в MP4 нет звуковой дорожки")


# ---------------------------------------------------------------- WMA (ASF)

_ASF_HEADER = bytes.fromhex("3026B2758E66CF11A6D900AA0062CE6C")
_ASF_DATA = bytes.fromhex("3626B2758E66CF11A6D900AA0062CE6C")
_ASF_STREAM = bytes.fromhex("9107DCB7B7A9CF118EE600C00C205365")
_ASF_AUDIO = bytes.fromhex("409E69F84D5BCF11A8FD00805F5C442B")
WMA_LOSSLESS = 0x0163


def _asf_stream(f) -> tuple[int, int, int]:
    """(начало, длина) пакетов Data Object и кодек звука. Теги — в Header Object, перед данными."""
    f.seek(0, 2)
    size = f.tell()
    f.seek(0)
    hdr = f.read(30)
    if len(hdr) < 30 or hdr[:16] != _ASF_HEADER:
        raise AudioError("не ASF")
    header_end = struct.unpack("<Q", hdr[16:24])[0]
    codec = 0
    pos = 30
    while pos + 24 <= min(header_end, size):
        f.seek(pos)
        obj = f.read(24)
        n = struct.unpack("<Q", obj[16:24])[0]
        if n < 24:
            break
        if obj[:16] == _ASF_STREAM:
            body = f.read(80)
            if body[:16] == _ASF_AUDIO and len(body) >= 56:  # тип, ошибки, смещение, длины, флаги, резерв
                codec = struct.unpack("<H", body[54:56])[0]
        pos += n
    f.seek(header_end)
    data = f.read(24)
    if len(data) < 24 or data[:16] != _ASF_DATA:
        raise AudioError("в ASF нет Data Object")
    n = struct.unpack("<Q", data[16:24])[0]
    return header_end + 50, max(0, min(n - 50, size - header_end - 50)), codec


# ---------------------------------------------------------------- теги APEv2 (APE, WavPack, TTA)

_APE_NAMES = {"YEAR": "DATE", "TRACK": "TRACKNUMBER", "ALBUM ARTIST": "ALBUMARTIST", "DISC": "DISCNUMBER"}


def _ape_items(f) -> list[tuple[str, int, bytes]]:
    """Элементы APEv2 в конце файла: (ключ в верхнем регистре, тип, значение); тип 1 — двоичное."""
    f.seek(0, 2)
    end = f.tell()
    if end >= 128:
        f.seek(end - 128)
        if f.read(3) == b"TAG":
            end -= 128
    if end < 32:
        return []
    f.seek(end - 32)
    foot = f.read(32)
    if foot[:8] != b"APETAGEX":
        return []
    size = struct.unpack("<I", foot[12:16])[0]
    if size < 32 or size > end:
        return []
    f.seek(end - size)
    body = f.read(size - 32)
    out = []
    pos = 0
    while pos + 8 < len(body):
        vlen, flags = struct.unpack("<II", body[pos:pos + 8])
        zero = body.find(b"\0", pos + 8)
        if zero < 0 or zero + 1 + vlen > len(body):
            break
        key = body[pos + 8:zero].decode("ascii", "replace").upper()
        out.append((key, (flags >> 1) & 3, body[zero + 1:zero + 1 + vlen]))
        pos = zero + 1 + vlen
    return out


def _ape_tags(f) -> dict[str, list[str]]:
    """Теги в ключах Vorbis comment. Несколько значений разделены нулём."""
    tags: dict[str, list[str]] = {}
    for key, kind, value in _ape_items(f):
        if kind != 0:
            continue
        name = _APE_NAMES.get(key, key)
        for v in value.decode("utf-8", "replace").split("\0"):
            v = v.split("/")[0] if name == "TRACKNUMBER" else v  # «3/12»
            if v:
                tags.setdefault(name, []).append(v)
    return tags


def _ape_picture(f) -> tuple[str, bytes] | None:
    for key, kind, value in _ape_items(f):
        if kind == 1 and key.startswith("COVER ART"):
            data = value[value.find(b"\0") + 1:]  # «имя файла\0байты»
            if data:
                return ("image/png" if data[:4] == b"\x89PNG" else "image/jpeg"), data
    return None


# ---------------------------------------------------------------- теги ID3 в чужих файлах (DSF, DFF, AAC)

def _id3_of(block: bytes, image: bool = False):
    from tinytag import TinyTag

    return TinyTag.get(file_obj=io.BytesIO(block), filename="t.mp3", duration=False, image=image)


def _id3_tags(block: bytes | None) -> dict[str, list[str]]:
    if not block:
        return {}
    try:
        return _tags_of(_id3_of(block))
    except Exception:  # noqa: BLE001 — битые теги не повод терять звук
        return {}


# ---------------------------------------------------------------- заголовки без tinytag
# Каждый читатель отдаёт (частота, каналы, разрядность, длительность, начало потока, длина, теги).

_Head = tuple[int, int, int, float, int, int, dict[str, list[str]]]


def _ape_head(f) -> _Head:
    """Monkey's Audio: с версии 3.98 дескриптор + заголовок, раньше — один короткий заголовок."""
    start, length = _mp3_stream(f)  # ID3v2 в начале, APEv2 и ID3v1 в конце
    f.seek(start)
    h = f.read(76)
    if len(h) < 32 or h[:4] != b"MAC ":
        raise AudioError("не APE")
    version = struct.unpack("<H", h[4:6])[0]
    if version >= 3980:
        desc = struct.unpack("<I", h[8:12])[0]
        f.seek(start + desc)
        m = f.read(24)
        if len(m) < 24:
            raise AudioError("обрезанный заголовок APE")
        blocks_per_frame, final, frames = struct.unpack("<III", m[4:16])
        bits, channels, rate = struct.unpack("<HHI", m[16:24])
    else:
        level, flags, channels, rate = struct.unpack("<HHHI", h[6:16].ljust(10, b"\0")[:10])
        frames, final = struct.unpack("<II", h[24:32])
        if version >= 3950:
            blocks_per_frame = 73728 * 4
        elif version >= 3900 or (version >= 3800 and level >= 4000):
            blocks_per_frame = 73728
        else:
            blocks_per_frame = 9216
        bits = 8 if flags & 1 else 24 if flags & 8 else 16
    if not rate or not channels:
        raise AudioError("APE: нет частоты или каналов")
    samples = (frames - 1) * blocks_per_frame + final if frames else 0
    return rate, channels, bits, samples / rate, start, length, _ape_tags(f)


_WV_RATES = (6000, 8000, 9600, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000, 64000, 88200, 96000, 192000)


def _wv_head(f) -> _Head:
    """WavPack: всё нужное в заголовке первого блока; частоты не из таблицы (176,4 кГц) — в его метаданных."""
    start, length = _mp3_stream(f)
    f.seek(start)
    h = f.read(32)
    if len(h) < 32 or h[:4] != b"wvpk":
        raise AudioError("не WavPack")
    size = struct.unpack("<I", h[4:8])[0]
    total = struct.unpack("<I", h[12:16])[0] | (h[11] << 32)
    flags = struct.unpack("<I", h[24:28])[0]
    if flags & 0x80000000:
        raise AudioError("WavPack с DSD внутри не поддерживается")
    rate_index = (flags >> 23) & 0xF
    rate = _WV_RATES[rate_index] if rate_index < len(_WV_RATES) else 0
    if not rate:
        meta = f.read(max(0, size + 8 - 32))
        pos = 0
        while pos + 2 <= len(meta):
            mid = meta[pos]
            big = mid & 0x80
            words = int.from_bytes(meta[pos + 1:pos + (4 if big else 2)], "little")
            body = pos + (4 if big else 2)
            if (mid & 0x3F) == 0x27:  # ID_SAMPLE_RATE
                rate = int.from_bytes(meta[body:body + 3], "little")
                break
            pos = body + words * 2
    if not rate:
        raise AudioError("WavPack: нет частоты")
    # многоканальный звук — несколько блоков на отсчёт, последний помечен FINAL_BLOCK
    channels = 0
    pos = start
    while True:
        f.seek(pos)
        b = f.read(32)
        if len(b) < 32 or b[:4] != b"wvpk":
            break
        fl = struct.unpack("<I", b[24:28])[0]
        channels += 1 if fl & 4 else 2
        if fl & 0x1000 or channels >= 64:
            break
        pos += struct.unpack("<I", b[4:8])[0] + 8
    bits = 32 if flags & 0x80 else ((flags & 3) + 1) * 8 - ((flags >> 13) & 0x1F)
    duration = 0.0 if h[12:16] == b"\xff\xff\xff\xff" else total / rate
    return rate, channels or 2, bits, duration, start, length, _ape_tags(f)


def _tta_head(f) -> _Head:
    start, length = _mp3_stream(f)
    f.seek(start)
    h = f.read(22)
    if len(h) < 18 or h[:4] != b"TTA1":
        raise AudioError("не TTA")
    channels, bits, rate, samples = struct.unpack("<HHII", h[6:18])
    if not rate or not channels:
        raise AudioError("TTA: нет частоты или каналов")
    return rate, channels, bits, samples / rate, start, length, _ape_tags(f)


def _dsf_id3(f) -> bytes | None:
    f.seek(20)
    meta = struct.unpack("<Q", f.read(8))[0]
    if not meta:
        return None
    f.seek(meta)
    return f.read(16 << 20)  # блок ID3 лежит в самом конце; обложка больше не бывает


def _dsf_head(f) -> _Head:
    """DSF: чанк «fmt » и чанк «data»; теги — блок ID3v2 по смещению из заголовка."""
    h = f.read(92)
    if len(h) < 92 or h[:4] != b"DSD " or h[28:32] != b"fmt ":
        raise AudioError("не DSF")
    channels, rate, bits = struct.unpack("<III", h[52:64])
    samples = struct.unpack("<Q", h[64:72])[0]
    fmt_len = struct.unpack("<Q", h[32:40])[0]
    data = 28 + fmt_len
    f.seek(data)
    d = f.read(12)
    if len(d) < 12 or d[:4] != b"data":
        raise AudioError("в DSF нет чанка data")
    length = struct.unpack("<Q", d[4:12])[0] - 12
    if not rate or not channels:
        raise AudioError("DSF: нет частоты или каналов")
    # у DSF bits — порядок бит в байте (1 или 8), звук всегда однобитный
    return rate, channels, 1, samples / rate, data + 12, length, _id3_tags(_dsf_id3(f))


def _dff_chunks(f, pos: int, end: int):
    """Чанки DSDIFF: (тип, начало тела, длина). Длины восьмибайтные, старшим байтом вперёд, тело чётной длины."""
    while pos + 12 <= end:
        f.seek(pos)
        h = f.read(12)
        if len(h) < 12:
            return
        n = struct.unpack(">Q", h[4:12])[0]
        yield h[:4], pos + 12, n
        pos += 12 + n + (n & 1)


def _dff_head(f) -> _Head:
    """DFF: свойства в PROP/SND, звук в чанке «DSD » или «DST » (сжатый), теги в «ID3 » и DIIN."""
    h = f.read(16)
    if len(h) < 16 or h[:4] != b"FRM8" or h[12:16] != b"DSD ":
        raise AudioError("не DFF")
    f.seek(0, 2)
    size = f.tell()
    rate = channels = 0
    stream = None
    dst_frames = 0.0
    id3 = None
    tags: dict[str, list[str]] = {}

    def text(start: int, n: int) -> str:
        f.seek(start)
        return f.read(n).decode("latin-1", "replace").rstrip("\0 ")

    for cid, body, n in _dff_chunks(f, 16, size):
        if cid == b"PROP":
            for sub, sb, sn in _dff_chunks(f, body + 4, body + n):
                f.seek(sb)
                if sub == b"FS  ":
                    rate = struct.unpack(">I", f.read(4))[0]
                elif sub == b"CHNL":
                    channels = struct.unpack(">H", f.read(2))[0]
        elif cid in (b"DSD ", b"DST "):
            stream = (body, min(n, size - body))
            if cid == b"DST ":
                for sub, sb, sn in _dff_chunks(f, body, body + n):
                    if sub == b"FRTE":
                        f.seek(sb)
                        count, per_second = struct.unpack(">IH", f.read(6))
                        dst_frames = count / per_second if per_second else 0.0
                        break
        elif cid == b"ID3 ":
            f.seek(body)
            id3 = f.read(min(n, 16 << 20))
        elif cid == b"DIIN":
            for sub, sb, sn in _dff_chunks(f, body, body + n):
                if sub in (b"DIAR", b"DITI"):
                    f.seek(sb)
                    count = struct.unpack(">I", f.read(4))[0]
                    tags["ARTIST" if sub == b"DIAR" else "TITLE"] = [text(sb + 4, min(count, sn - 4))]
    if stream is None:
        raise AudioError("в DFF нет звука")
    if not rate or not channels:
        raise AudioError("DFF: нет частоты или каналов")
    tags.update(_id3_tags(id3))
    duration = dst_frames or stream[1] * 8 / channels / rate
    return rate, channels, 1, duration, stream[0], stream[1], tags


def _dff_id3(f) -> bytes | None:
    f.seek(0, 2)
    size = f.tell()
    for cid, body, n in _dff_chunks(f, 16, size):
        if cid == b"ID3 ":
            f.seek(body)
            return f.read(min(n, 16 << 20))
    return None


_CAF_TAGS = {"title": "TITLE", "artist": "ARTIST", "album": "ALBUM", "year": "DATE", "genre": "GENRE",
             "track number": "TRACKNUMBER"}


def _caf_head(f) -> _Head:
    """CAF: «desc» — формат, «pakt» — точное число кадров, «kuki» у ALAC — разрядность, «info» — теги."""
    h = f.read(8)
    if len(h) < 8 or h[:4] != b"caff":
        raise AudioError("не CAF")
    f.seek(0, 2)
    size = f.tell()
    pos = 8
    rate = 0.0
    fmt = b""
    channels = bits = 0
    frames_per_packet = bytes_per_packet = 0
    valid_frames = 0
    stream = None
    tags: dict[str, list[str]] = {}
    while pos + 12 <= size:
        f.seek(pos)
        c = f.read(12)
        cid, n = c[:4], struct.unpack(">q", c[4:12])[0]
        body = pos + 12
        if n < 0:  # «data» неизвестной длины — до конца файла
            n = size - body
        if cid == b"desc":
            d = f.read(32)
            rate = struct.unpack(">d", d[:8])[0]
            fmt = d[8:12]
            bytes_per_packet, frames_per_packet, channels, bits = struct.unpack(">IIII", d[16:32])
        elif cid == b"data":
            stream = (body + 4, n - 4)  # editCount
        elif cid == b"pakt":
            valid_frames = struct.unpack(">q", f.read(16)[8:16])[0]
        elif cid == b"kuki" and fmt == b"alac":
            cookie = f.read(min(n, 64))
            at = cookie.rfind(b"alac")  # настройки — сразу или в ящике alac (бывает ещё обёртка frma)
            at = at + 8 if at >= 0 else 0
            if len(cookie) >= at + 6:
                bits = cookie[at + 5]
        elif cid == b"info":
            parts = f.read(min(n, 1 << 16))[4:].split(b"\0")
            for key, val in zip(parts[::2], parts[1::2]):
                name = _CAF_TAGS.get(key.decode("utf-8", "replace").lower())
                if name and val:
                    tags[name] = [val.decode("utf-8", "replace")]
        pos = body + n
    if stream is None or not rate or not channels:
        raise AudioError("CAF: нет звука или описания")
    if fmt not in (b"alac", b"aac ", b"lpcm"):
        raise AudioError(f"кодек {fmt.decode('latin-1').strip()} плеер не играет")
    if not valid_frames and fmt == b"lpcm" and bytes_per_packet:
        valid_frames = stream[1] // bytes_per_packet * max(frames_per_packet, 1)
    return int(rate), channels, 0 if fmt == b"aac " else bits, valid_frames / rate, stream[0], stream[1], tags


_AAC_RATES = (96000, 88200, 64000, 48000, 44100, 32000, 24000, 22050, 16000, 12000, 11025, 8000, 7350)


def _aac_head(f) -> _Head:
    """Голый AAC (ADTS): частота и каналы в заголовке кадра; длительность — по среднему размеру
    кадров в первых 2 МБ, идти по всему файлу ради неё не стоит."""
    start, length = _mp3_stream(f)
    f.seek(start)
    buf = f.read(min(2 << 20, length))
    if len(buf) < 7 or buf[0] != 0xFF or (buf[1] & 0xF6) != 0xF0:
        raise AudioError("не AAC ADTS")
    index = (buf[2] >> 2) & 0xF
    rate = _AAC_RATES[index] if index < len(_AAC_RATES) else 0
    channels = ((buf[2] & 1) << 2) | (buf[3] >> 6) or 2
    if not rate:
        raise AudioError("AAC: нет частоты")
    pos = frames = 0
    while pos + 7 <= len(buf) and buf[pos] == 0xFF and (buf[pos + 1] & 0xF6) == 0xF0:
        n = ((buf[pos + 3] & 3) << 11) | (buf[pos + 4] << 3) | (buf[pos + 5] >> 5)
        if n < 7:
            break
        pos += n
        frames += 1
    duration = length / (pos / frames) * 1024 / rate if frames else 0.0
    f.seek(0)
    return rate, channels, 0, duration, start, length, _id3_tags(f.read(start) if start else None)


_HEADS = {".ape": _ape_head, ".wv": _wv_head, ".tta": _tta_head, ".dsf": _dsf_head, ".dff": _dff_head,
          ".caf": _caf_head, ".aac": _aac_head}
_MP4 = {".m4a", ".alac"}

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
    """Вложенная обложка (ID3 APIC, Vorbis METADATA_BLOCK_PICTURE, атом covr, APEv2) → (тип, байты)."""
    ext = Path(path).suffix.lower()
    try:
        if ext in (".ape", ".wv", ".tta"):
            with open(path, "rb") as f:
                return _ape_picture(f)
        if ext in (".dsf", ".dff"):
            with open(path, "rb") as f:
                block = _dsf_id3(f) if ext == ".dsf" else _dff_id3(f)
            img = _id3_of(block, image=True).images.any if block else None
        elif ext == ".aac":
            with open(path, "rb") as f:
                start = _mp3_stream(f)[0]
                f.seek(0)
                block = f.read(start)
            img = _id3_of(block, image=True).images.any if block else None
        else:
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
    lossy = ext in LOSSY
    try:
        with open(p, "rb") as f:
            if ext in _HEADS:
                rate, channels, bits, duration, start, length, tags = _HEADS[ext](f)
                md5 = _sampled_md5(f, start, length)
                return AudioInfo(rate, channels, bits, duration, md5, "bytes", tags)
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
            elif ext in _MP4:
                start, length = _mp4_stream(f)
                codec = _mp4_codec(f)
                if codec not in ("alac", "mp4a"):
                    raise AudioError(f"кодек {codec} плеер не играет")
                lossy = codec == "mp4a"
                md5 = _sampled_md5(f, start, length)
            elif ext == ".wma":
                start, length, tag = _asf_stream(f)
                lossy = tag != WMA_LOSSLESS
                md5 = _sampled_md5(f, start, length)
            else:
                raise AudioError(f"формат не поддерживается: {ext}")
    except (struct.error, IndexError, ValueError, OverflowError, ZeroDivisionError) as e:
        raise AudioError(f"битый заголовок: {e}") from e  # файл плохой, а сканер идёт дальше
    try:
        t = _tinytag(p)
    except Exception as e:  # noqa: BLE001
        raise AudioError(f"tinytag: {e}") from e
    if not t.samplerate or not t.channels:
        raise AudioError("tinytag: нет частоты или каналов")
    bits = 0 if lossy else int(t.bitdepth or 0)
    duration = float(t.duration or 0.0)
    if ext == ".opus" and granule > 0:
        duration = granule / 48000.0  # у Opus гранулы всегда в 48 кГц
    return AudioInfo(int(t.samplerate), int(t.channels), bits, duration, md5, "bytes", _tags_of(t))
