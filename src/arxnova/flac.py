"""Чтение заголовка FLAC без декодера: STREAMINFO (с MD5) и Vorbis-комментарии.

Ключ трека — MD5 из STREAMINFO (ТЗ, ФТ-2.1, П-1): он одинаков на сервере и на
приборе, не зависит от пути и тегов. Если кодер оставил MD5 нулевым, его считаем по
декодированному PCM (pcm_md5) — тем же способом, что и сам FLAC.

Своя разборка, а не mutagen: mutagen под GPL, а в пакете допустимы только
разрешительные лицензии (ТЗ, УСТ-8).
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, field
from pathlib import Path

ZERO_MD5 = bytes(16)


class FlacError(ValueError):
    pass


@dataclass
class FlacInfo:
    sample_rate: int
    channels: int
    bits: int
    total_samples: int
    md5: bytes
    tags: dict[str, list[str]] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.total_samples / self.sample_rate if self.sample_rate else 0.0

    @property
    def md5_is_zero(self) -> bool:
        return self.md5 == ZERO_MD5

    def tag(self, name: str) -> str | None:
        values = self.tags.get(name.upper())
        return values[0] if values else None


def _skip_id3v2(f) -> None:
    """ID3v2 в начале файла бывает у FLAC из некоторых магазинов — пропускаем его."""
    head = f.read(10)
    if len(head) == 10 and head[:3] == b"ID3":
        flags = head[5]
        size = 0
        for b in head[6:10]:  # synchsafe: по 7 бит в байте
            size = (size << 7) | (b & 0x7F)
        if flags & 0x10:  # есть footer
            size += 10
        f.seek(10 + size)
    else:
        f.seek(0)


def _parse_streaminfo(data: bytes) -> tuple[int, int, int, int, bytes]:
    if len(data) != 34:
        raise FlacError(f"STREAMINFO длиной {len(data)} вместо 34")
    # байты 10..17: 20 бит частоты, 3 бита (каналы-1), 5 бит (разрядность-1), 36 бит сэмплов
    packed = int.from_bytes(data[10:18], "big")
    sample_rate = packed >> 44
    channels = ((packed >> 41) & 0x7) + 1
    bits = ((packed >> 36) & 0x1F) + 1
    total_samples = packed & 0xFFFFFFFFF
    return sample_rate, channels, bits, total_samples, data[18:34]


def _parse_vorbis_comment(data: bytes) -> dict[str, list[str]]:
    tags: dict[str, list[str]] = {}
    try:
        (vendor_len,) = struct.unpack_from("<I", data, 0)
        pos = 4 + vendor_len
        (count,) = struct.unpack_from("<I", data, pos)
        pos += 4
        for _ in range(count):
            (n,) = struct.unpack_from("<I", data, pos)
            pos += 4
            entry = data[pos : pos + n].decode("utf-8", errors="replace")
            pos += n
            if "=" in entry:
                k, v = entry.split("=", 1)
                tags.setdefault(k.strip().upper(), []).append(v)
    except struct.error as e:
        raise FlacError(f"битый VORBIS_COMMENT: {e}") from e
    return tags


def read_flac(path: str | Path) -> FlacInfo:
    with open(path, "rb") as f:
        _skip_id3v2(f)
        if f.read(4) != b"fLaC":
            raise FlacError("нет сигнатуры fLaC")
        info: FlacInfo | None = None
        tags: dict[str, list[str]] = {}
        last = False
        first = True
        while not last:
            header = f.read(4)
            if len(header) < 4:
                raise FlacError("файл оборвался в метаданных")
            last = bool(header[0] & 0x80)
            block_type = header[0] & 0x7F
            length = int.from_bytes(header[1:4], "big")
            if first and block_type != 0:
                raise FlacError("первый блок метаданных не STREAMINFO")
            first = False
            if block_type in (0, 4):
                data = f.read(length)
                if len(data) < length:
                    raise FlacError("файл оборвался в метаданных")
                if block_type == 0:
                    sr, ch, bits, total, md5 = _parse_streaminfo(data)
                    info = FlacInfo(sr, ch, bits, total, md5)
                else:
                    tags = _parse_vorbis_comment(data)
            else:
                f.seek(length, 1)
        assert info is not None
        info.tags = tags
        return info


def pcm_md5(path: str | Path, bits: int) -> bytes:
    """MD5 несжатого сигнала так, как его считает FLAC: сэмплы вперемешку по каналам,
    little-endian, (bits+7)//8 байт на сэмпл, со знаком."""
    import numpy as np
    import soundfile as sf

    width = (bits + 7) // 8
    h = hashlib.md5()
    with sf.SoundFile(str(path)) as snd:
        for block in snd.blocks(blocksize=65536, dtype="int32", always_2d=True):
            # libsndfile выравнивает сэмплы по старшим битам int32 — сдвигаем обратно
            samples = (block >> (32 - bits)).astype("<i4", copy=False)
            raw = samples.reshape(-1).view(np.uint8).reshape(-1, 4)[:, :width]
            h.update(raw.tobytes())
    return h.digest()


def read_picture(path: str | Path) -> tuple[str, bytes] | None:
    """Обложка из блока PICTURE (тип 6): сначала «передняя обложка» (3), иначе первая."""
    best = None
    with open(path, "rb") as f:
        _skip_id3v2(f)
        if f.read(4) != b"fLaC":
            return None
        last = False
        while not last:
            header = f.read(4)
            if len(header) < 4:
                break
            last = bool(header[0] & 0x80)
            block_type = header[0] & 0x7F
            length = int.from_bytes(header[1:4], "big")
            if block_type != 6:
                f.seek(length, 1)
                continue
            data = f.read(length)
            try:
                pos = 0
                pic_type = int.from_bytes(data[pos : pos + 4], "big"); pos += 4
                mlen = int.from_bytes(data[pos : pos + 4], "big"); pos += 4
                mime = data[pos : pos + mlen].decode("ascii", "replace"); pos += mlen
                dlen = int.from_bytes(data[pos : pos + 4], "big"); pos += 4 + dlen
                pos += 16  # ширина, высота, глубина, число цветов
                plen = int.from_bytes(data[pos : pos + 4], "big"); pos += 4
                pic = data[pos : pos + plen]
            except (ValueError, IndexError):
                continue
            if pic_type == 3:
                return mime or "image/jpeg", pic
            best = best or (mime or "image/jpeg", pic)
    return best
