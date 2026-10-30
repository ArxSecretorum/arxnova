"""Медиатека не только из FLAC (30.10.2026): WAV, AIFF, OGG/Opus, MP3 попадают в базу с тегами и
ключом по байтам потока; перезапись тегов и копия по другому пути ключ не меняют; анализ читает их."""

import shutil
import subprocess

import numpy as np
import pytest

from arxnova import audiokey, config, db, scan
from arxnova import features as F

needs_ffmpeg = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="нужен ffmpeg")


def make_audio(path, fmt: str, seconds: float = 2.0, freq: int = 440, **tags):
    """Синус через ffmpeg с тегами: wav (LIST INFO), aiff/mp3 (ID3), ogg/opus (Vorbis comment)."""
    codec = {"wav": ["-c:a", "pcm_s16le"], "aiff": ["-c:a", "pcm_s16be", "-write_id3v2", "1"],
             "mp3": ["-c:a", "libmp3lame", "-b:a", "128k"], "ogg": ["-c:a", "vorbis", "-strict", "experimental"],
             "opus": ["-c:a", "libopus"]}[fmt]
    meta = [x for k, v in tags.items() for x in ("-metadata", f"{k}={v}")]
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"sine=frequency={freq}:duration={seconds}",
                    "-ac", "2", *codec, *meta, str(path)], check=True)
    return path


@needs_ffmpeg
@pytest.mark.parametrize("fmt", ["wav", "aiff", "mp3", "ogg", "opus"])
def test_format_is_indexed_with_tags(tmp_path, fmt):
    lib = tmp_path / "lib"
    lib.mkdir()
    make_audio(lib / f"t.{fmt}", fmt, artist="Artist A", album="Album B", title="Song C", date="2021", genre="Jazz")
    cfg = config.Config(data_dir=tmp_path / "data", libraries=[lib])
    cfg.ensure_dirs()
    conn = db.connect(cfg.db_path)
    rep = scan.scan(conn, [lib])
    assert rep.errors == [] and rep.added == 1, rep
    r = conn.execute("SELECT artist, album, title, date, genre_raw, sample_rate, channels, bits, duration, md5_source "
                     "FROM tracks").fetchone()
    assert (r["artist"], r["album"], r["title"]) == ("Artist A", "Album B", "Song C"), dict(r)
    assert r["date"].startswith("2021") and r["genre_raw"] == "Jazz"
    assert r["sample_rate"] in (44100, 48000) and r["channels"] == 2 and r["md5_source"] == "bytes"
    assert abs(r["duration"] - 2.0) < 0.2, r["duration"]
    assert r["bits"] == (16 if fmt in ("wav", "aiff") else 0)


@needs_ffmpeg
@pytest.mark.parametrize("fmt", ["wav", "mp3", "ogg"])
def test_key_survives_retag_and_copy(tmp_path, fmt):
    a = make_audio(tmp_path / f"a.{fmt}", fmt, title="One", artist="X")
    k1 = audiokey.read(a).md5
    # красное: другой звук — другой ключ
    b = make_audio(tmp_path / f"b.{fmt}", fmt, freq=660, title="One", artist="X")
    assert audiokey.read(b).md5 != k1
    # зелёное: перезаписали теги (длиннее прежних) без перекодирования — ключ тот же
    long_title = "Совсем другое и очень длинное название трека " * 3
    retagged = tmp_path / f"re.{fmt}"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(a), "-c", "copy", "-map_metadata", "0",
                    "-metadata", f"title={long_title}", "-metadata:s:a:0", f"title={long_title}",  # OGG хранит теги в потоке
                    str(retagged)], check=True)
    retagged.replace(a)
    info = audiokey.read(a)
    assert info.md5 == k1
    assert "Совсем другое" in (info.tag("TITLE") or "")
    # копия по другому пути — тот же трек
    c = tmp_path / "copy" / f"a.{fmt}"
    c.parent.mkdir()
    shutil.copy(a, c)
    assert audiokey.read(c).md5 == k1


@needs_ffmpeg
def test_rescan_is_idempotent_and_duplicate_is_one_track(tmp_path):
    lib = tmp_path / "lib"
    (lib / "x").mkdir(parents=True)
    make_audio(lib / "t.mp3", "mp3", title="T")
    shutil.copy(lib / "t.mp3", lib / "x" / "t2.mp3")
    cfg = config.Config(data_dir=tmp_path / "data", libraries=[lib])
    cfg.ensure_dirs()
    conn = db.connect(cfg.db_path)
    rep = scan.scan(conn, [lib])
    assert rep.added == 1 and rep.duplicates == 1, rep
    assert conn.execute("SELECT count(*) FROM tracks").fetchone()[0] == 1
    assert conn.execute("SELECT count(*) FROM files").fetchone()[0] == 2
    rep2 = scan.scan(conn, [lib])
    assert rep2.added == 0 and rep2.unchanged == 2, rep2


@needs_ffmpeg
@pytest.mark.parametrize("fmt", ["wav", "mp3", "ogg"])
def test_analysis_reads_new_formats(tmp_path, fmt):
    p = make_audio(tmp_path / f"t.{fmt}", fmt, seconds=8.0)
    x = F.load_mono(p, F.SR_EFFNET)
    assert 7.5 * F.SR_EFFNET < len(x) < 8.5 * F.SR_EFFNET
    assert np.abs(x).max() > 0.05  # не тишина (синус lavfi — около 0,09)
    pats = F.patches(F.melspec(F.load_mono(p, F.SR_EFFNET, 1.0, 6.0)))  # отрывок, как в embed.excerpt_patches
    assert pats.shape[1:] == (F.PATCH, F.N_MELS) and len(pats) >= 3


def test_media_type_by_extension():
    from arxnova.server import media_type

    assert media_type("a/b.MP3") == "audio/mpeg" and media_type("x.flac") == "audio/flac"
    assert media_type("x.opus") == "audio/ogg" and media_type("x.wav") == "audio/wav"
    assert media_type("x.txt") == "application/octet-stream"


@needs_ffmpeg
def test_system_folders_skipped_and_format_in_json(tmp_path):
    """Корзина Windows и служебные папки диска в медиатеку не попадают; у трека в JSON — формат."""
    from arxnova.server import track_json

    lib = tmp_path / "lib"
    (lib / "$RECYCLE.BIN" / "x").mkdir(parents=True)
    (lib / "System Volume Information").mkdir()
    (lib / "Album").mkdir()
    make_audio(lib / "$RECYCLE.BIN" / "x" / "junk.mp3", "mp3", title="Junk")
    make_audio(lib / "System Volume Information" / "junk.wav", "wav", title="Junk")
    make_audio(lib / "Album" / "song.mp3", "mp3", title="Song")
    cfg = config.Config(data_dir=tmp_path / "data", libraries=[lib])
    cfg.ensure_dirs()
    conn = db.connect(cfg.db_path)
    rep = scan.scan(conn, [lib])
    assert rep.added == 1 and rep.errors == [], rep
    r = conn.execute("SELECT t.*, f.path AS path FROM tracks t JOIN files f ON f.track_id = t.id").fetchone()
    j = track_json(r)
    assert j["format"] == "mp3" and j["bits"] == 0 and j["size"] > 0 and j["title"] == "Song"
