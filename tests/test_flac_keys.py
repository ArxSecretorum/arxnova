"""Ключи треков и обход медиатеки.

Правило проекта (ТЗ, §21): каждая проверка сначала показывает, что ловит сломанный
вход, и только потом проходит на правильном."""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from arxnova import db, flac, keys, scan

FFMPEG = shutil.which("ffmpeg")
needs_ffmpeg = pytest.mark.skipif(not FFMPEG, reason="нужен ffmpeg для эталонных FLAC")


def make_flac(path: Path, *, bits=16, rate=44100, seconds=1.5, freq=440.0, tweak=False, **tags):
    """FLAC через ffmpeg: он, как эталонный кодер, пишет MD5 в STREAMINFO."""
    t = np.arange(int(rate * seconds)) / rate
    x = 0.4 * np.sin(2 * np.pi * freq * t)
    stereo = np.stack([x, 0.5 * x], axis=1)
    if tweak:
        stereo[1000, 0] += 1e-3  # один сэмпл другой — звук «почти тот же»
    wav = path.with_suffix(".wav")
    sf.write(wav, stereo, rate, subtype="PCM_16" if bits == 16 else "PCM_24")
    meta = []
    for k, v in tags.items():
        meta += ["-metadata", f"{k}={v}"]
    fmt = ["-sample_fmt", "s16"] if bits == 16 else ["-sample_fmt", "s32", "-bits_per_raw_sample", "24"]
    subprocess.run(
        [FFMPEG, "-v", "error", "-y", "-i", str(wav), *fmt, *meta, "-c:a", "flac", str(path)],
        check=True,
    )
    wav.unlink()
    return path


def zero_md5(src: Path, dst: Path) -> Path:
    """Копия с обнулённым MD5 в STREAMINFO — как у кодеров, которые его не пишут."""
    data = bytearray(src.read_bytes())
    assert data[:4] == b"fLaC"
    # 4 байта сигнатуры + 4 байта заголовка блока + 18 байт до MD5
    data[26:42] = bytes(16)
    dst.write_bytes(bytes(data))
    return dst


@needs_ffmpeg
@pytest.mark.parametrize("bits,rate", [(16, 44100), (24, 48000), (24, 96000)])
def test_streaminfo_fields(tmp_path, bits, rate):
    f = make_flac(tmp_path / "a.flac", bits=bits, rate=rate, seconds=2.0, ARTIST="A", TITLE="T")
    info = flac.read_flac(f)
    assert (info.sample_rate, info.bits, info.channels) == (rate, bits, 2)
    assert info.total_samples == rate * 2
    assert info.tag("artist") == "A" and info.tag("TITLE") == "T"
    assert not info.md5_is_zero


@needs_ffmpeg
@pytest.mark.parametrize("bits", [16, 24])
def test_pcm_md5_matches_encoder(tmp_path, bits):
    """Наш MD5 по звуку обязан совпасть с MD5, который записал кодер."""
    a = make_flac(tmp_path / "a.flac", bits=bits)
    b = make_flac(tmp_path / "b.flac", bits=bits, tweak=True)
    md5_a = flac.read_flac(a).md5
    # красное: другой на один сэмпл звук даёт другой ключ — сравнение не «всегда да»
    assert flac.pcm_md5(b, bits) != md5_a
    # зелёное
    assert flac.pcm_md5(a, bits) == md5_a


@needs_ffmpeg
def test_id3_prefix_is_skipped(tmp_path, monkeypatch):
    a = make_flac(tmp_path / "a.flac")
    body = b"\x00" * 100
    size = len(body)
    synchsafe = bytes([(size >> 21) & 0x7F, (size >> 14) & 0x7F, (size >> 7) & 0x7F, size & 0x7F])
    tagged = tmp_path / "id3.flac"
    tagged.write_bytes(b"ID3\x04\x00\x00" + synchsafe + body + a.read_bytes())
    # красное: без пропуска ID3 файл не читается
    monkeypatch.setattr(flac, "_skip_id3v2", lambda f: f.seek(0))
    with pytest.raises(flac.FlacError):
        flac.read_flac(tagged)
    monkeypatch.undo()
    # зелёное
    assert flac.read_flac(tagged).md5 == flac.read_flac(a).md5


def test_not_flac(tmp_path):
    p = tmp_path / "x.flac"
    p.write_bytes(b"RIFF" + bytes(100))
    with pytest.raises(flac.FlacError):
        flac.read_flac(p)


@needs_ffmpeg
def test_scan_dedup_and_zero_md5(tmp_path):
    lib = tmp_path / "lib"
    (lib / "x").mkdir(parents=True)
    a = make_flac(lib / "a.flac", ARTIST="Artist", TITLE="Song")
    shutil.copy(a, lib / "x" / "copy.flac")  # тот же звук по другому пути
    zero_md5(a, lib / "x" / "zero.flac")  # тот же звук, MD5 не записан
    make_flac(lib / "other.flac", freq=880.0, ARTIST="Artist", TITLE="Other")

    conn = db.connect(tmp_path / "db.sqlite3")
    rep = scan.scan(conn, [lib])
    assert rep.errors == []
    assert rep.seen == 4
    assert rep.pcm_md5 == 1
    # три файла одного звука — один трек; другой звук — второй трек
    assert conn.execute("SELECT count(*) FROM tracks").fetchone()[0] == 2
    assert conn.execute("SELECT count(*) FROM files").fetchone()[0] == 4

    # повторный обход ничего не пересчитывает
    rep2 = scan.scan(conn, [lib])
    assert (rep2.unchanged, rep2.added) == (4, 0)

    # удалили все копии одного звука — трек ушёл из каталога
    for p in (lib / "a.flac", lib / "x" / "copy.flac", lib / "x" / "zero.flac"):
        p.unlink()
    rep3 = scan.scan(conn, [lib])
    assert rep3.removed == 3
    assert conn.execute("SELECT count(*) FROM tracks").fetchone()[0] == 1


def test_recording_key_tolerance():
    b_cd = keys.recording_basis(mb_recording=None, artist="GoGo Penguin", title="Hopopono")
    b_hr = keys.recording_basis(mb_recording=None, artist="GoGo Penguin ", title="Hopopono (Remastered 2020)")
    assert b_cd == b_hr
    cd_key = keys.new_recording_key(b_cd, 301.4)
    known = [(cd_key, 301.4)]
    # красное: округление до 2 с режет одну запись на две по границе 301 с
    rounded = lambda d: int(round(d / 2.0)) * 2
    assert rounded(301.4) != rounded(300.6)
    # зелёное: допуск ±2 с склеивает издания
    assert keys.match_recording(b_hr, 300.6, known) == cd_key
    # концертная версия той же песни — другая запись
    assert keys.match_recording(b_cd, 420.0, known) is None
    # другая песня — другая основа
    assert keys.recording_basis(mb_recording=None, artist="GoGo Penguin", title="Murmuration") != b_cd
    # MusicBrainz ID важнее тегов и не зависит от регистра
    m1 = keys.recording_basis(mb_recording="ABC", artist="x", title="y")
    m2 = keys.recording_basis(mb_recording="abc", artist="z", title="w")
    assert keys.new_recording_key(m1, 1) == keys.new_recording_key(m2, 99)


@needs_ffmpeg
def test_scan_editions_share_recording(tmp_path):
    lib = tmp_path / "lib"
    lib.mkdir()
    make_flac(lib / "cd.flac", bits=16, rate=44100, seconds=3.0, ARTIST="A", TITLE="Song")
    make_flac(lib / "hr.flac", bits=24, rate=96000, seconds=3.2, ARTIST="A", TITLE="Song (Remastered)")
    make_flac(lib / "live.flac", bits=16, rate=44100, seconds=9.0, freq=300.0, ARTIST="A", TITLE="Song")
    conn = db.connect(tmp_path / "db.sqlite3")
    scan.scan(conn, [lib])
    rows = conn.execute("SELECT count(*), count(DISTINCT recording_key) FROM tracks").fetchone()
    assert tuple(rows) == (3, 2)  # три файла, две записи: издания склеены, концертная отдельно


def test_onnx_telemetry_disabled_on_import():
    import os

    import arxnova  # noqa: F401

    assert os.environ["ORT_DISABLE_TELEMETRY"] == "1"


CUE_CP1251 = """PERFORMER "Катя Чехова"
TITLE "Я - Робот"
REM DATE 2005
FILE "image.wav" WAVE
  TRACK 01 AUDIO
    TITLE "Крылья"
    INDEX 01 00:00:00
  TRACK 02 AUDIO
    TITLE "Я - Робот"
    INDEX 00 00:02:50
    INDEX 01 00:03:00
  TRACK 03 AUDIO
    TITLE "Таю"
    PERFORMER "Гость"
    INDEX 01 00:06:37
"""


def test_cue_parse_cp1251():
    from arxnova import cue

    sheet = cue.parse(cue.decode(CUE_CP1251.encode("cp1251")))
    assert sheet.performer == "Катя Чехова" and sheet.title == "Я - Робот" and sheet.date == "2005"
    tr = cue.tracks_for(sheet, "image.flac")  # в листе .wav — имя сравнивается без расширения
    assert [t.start_ms for t in tr] == [0, 3000, 6493]  # INDEX 01, кадры по 1/75 с; INDEX 00 не берётся
    assert tr[2].performer == "Гость" and tr[1].title == "Я - Робот"


@needs_ffmpeg
def test_scan_cue_image_split_and_unsplit(tmp_path):
    from arxnova import cue

    lib = tmp_path / "lib"
    lib.mkdir()
    make_flac(lib / "image.flac", seconds=9.0, ARTIST="Катя Чехова", ALBUM="Я - Робот")
    conn = db.connect(tmp_path / "db.sqlite3")
    # красное: без листа образ — один трек на 9 с
    scan.scan(conn, [lib])
    assert conn.execute("SELECT count(*) FROM tracks").fetchone()[0] == 1
    # лист появился рядом — повторный обход режет образ, хотя сам FLAC не менялся
    (lib / "image.cue").write_bytes(CUE_CP1251.encode("cp1251"))
    rep = scan.scan(conn, [lib])
    assert rep.unchanged == 0
    rows = conn.execute("SELECT cue_ms, cue_end_ms, round(duration, 2), title, artist, album FROM tracks ORDER BY cue_ms").fetchall()
    assert [tuple(r) for r in rows] == [
        (0, 3000, 3.0, "Крылья", "Катя Чехова", "Я - Робот"),
        (3000, 6493, 3.49, "Я - Робот", "Катя Чехова", "Я - Робот"),
        (6493, -1, 2.51, "Таю", "Гость", "Я - Робот"),
    ]
    # путь есть у каждого трека образа
    assert conn.execute("SELECT count(DISTINCT track_id) FROM track_paths").fetchone()[0] == 3
    # повторный обход без изменений ничего не трогает
    assert scan.scan(conn, [lib]).unchanged == 1
    # лист убрали — снова один трек на весь файл
    (lib / "image.cue").unlink()
    scan.scan(conn, [lib])
    assert [tuple(r) for r in conn.execute("SELECT cue_ms, cue_end_ms FROM tracks")] == [(0, -1)]


@needs_ffmpeg
def test_scan_move_keeps_analysis(tmp_path):
    """Медиатеку перенесли на другой диск: тот же звук по новому пути сохраняет разбор (эмбеддинги),
    а изменённый на месте файл — разбирается заново."""
    old, new = tmp_path / "old", tmp_path / "new"
    old.mkdir()
    make_flac(old / "a.flac", ARTIST="A", TITLE="Song")
    conn = db.connect(tmp_path / "db.sqlite3")
    scan.scan(conn, [old])
    tid = conn.execute("SELECT id FROM tracks").fetchone()[0]
    conn.execute("INSERT INTO embeddings VALUES (?,?,?,?,?)", (tid, "m", b"\0" * 4, b"\0" * 4, 1))
    conn.commit()
    shutil.move(str(old), str(new))
    scan.scan(conn, [new])
    assert conn.execute("SELECT id FROM tracks").fetchone()[0] == tid
    assert conn.execute("SELECT count(*) FROM embeddings").fetchone()[0] == 1
    # файл переписали на месте (другие теги — другой размер) — разбор заново
    make_flac(new / "a.flac", ARTIST="A", TITLE="Song (edit)")
    scan.scan(conn, [new])
    assert conn.execute("SELECT count(*) FROM embeddings").fetchone()[0] == 0
