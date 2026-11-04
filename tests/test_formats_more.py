"""ALAC и AAC (m4a), WMA, DSD, APE, WavPack, TTA, CAF (03.11.2026, решение 0008): попадают в медиатеку
без ffmpeg, разбор звука — через ffmpeg, а без него честно ждёт и не считается несделанным."""

import errno
import random
import shutil
import struct
import subprocess

import numpy as np
import pytest

from arxnova import analysis, audiokey, config, cue, db, decode, indexer, scan
from arxnova import features as F

needs_ffmpeg = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="нужен ffmpeg")
needs_mac = pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("mac")), reason="нужен Monkey's Audio")

SECONDS = 8
TAGS = {"artist": "Artist A", "album": "Album B", "title": "Song C"}
# первая половина файла — один тон, вторая — другой: так видно, что отрывок взят из нужного места
TONE = "0.4*sin(2*PI*(if(lt(t,4),{low},{high}))*t)"
CODECS = {
    "alac.m4a": ["-c:a", "alac", "-sample_fmt", "s16p"],
    "aac.m4a": ["-c:a", "aac", "-b:a", "128k"],
    "t.aac": ["-c:a", "aac", "-b:a", "128k", "-f", "adts"],
    "t.wma": ["-c:a", "wmav2", "-b:a", "128k"],
    "t.wv": ["-c:a", "wavpack", "-sample_fmt", "s16p"],
    "t.tta": ["-c:a", "tta", "-sample_fmt", "s16"],
    "alac.caf": ["-c:a", "alac", "-sample_fmt", "s16p", "-f", "caf"],
}


def tone(path, args, low=440, high=880, rate=44100, tags=None):
    expr = TONE.format(low=low, high=high)
    meta = [x for k, v in (TAGS if tags is None else tags).items() for x in ("-metadata", f"{k}={v}")]
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"aevalsrc='{expr}':s={rate}:d={SECONDS}",
                    "-ac", "2", *args, *meta, str(path)], check=True)
    return path


def ape_tail(pairs: dict[str, str]) -> bytes:
    """Блок APEv2 с подвалом, как его пишут редакторы тегов."""
    items = b"".join(struct.pack("<II", len(v.encode()), 0) + k.encode() + b"\0" + v.encode() for k, v in pairs.items())
    size = len(items) + 32
    return items + b"APETAGEX" + struct.pack("<IIII", 2000, size, len(pairs), 0) + b"\0" * 8


def make_ape(path, low=440, high=880, tail=True):
    wav = path.with_suffix(".src.wav")
    tone(wav, ["-c:a", "pcm_s16le"], low, high, tags={})
    subprocess.run(["mac", str(wav), str(path), "-c2000"], check=True, capture_output=True)
    wav.unlink()
    if tail:
        with open(path, "ab") as f:
            f.write(ape_tail({"Artist": "Artist A", "Album": "Album B", "Title": "Song C", "Track": "3/12"}))
    return path


def id3(title="Song C") -> bytes:
    def frame(name, text):
        body = b"\x01" + text.encode("utf-16")
        return name + struct.pack(">I", len(body)) + b"\0\0" + body

    frames = frame(b"TIT2", title) + frame(b"TPE1", "Artist A") + frame(b"TALB", "Album B")
    n = len(frames)
    size = bytes([(n >> 21) & 0x7F, (n >> 14) & 0x7F, (n >> 7) & 0x7F, n & 0x7F])
    return b"ID3\x03\0\0" + size + frames


def pdm(rate: int, low=440, high=880) -> np.ndarray:
    """Однобитный поток модулятором первого порядка: как DSD, только шумнее."""
    n = rate * SECONDS
    t = np.arange(n, dtype=np.float64) / rate
    x = 0.4 * np.sin(2 * np.pi * np.where(t < 4, low, high) * t)
    return np.diff(np.floor(np.cumsum((x + 1) / 2)), prepend=0).astype(np.uint8)


def make_dsf(path, rate=2822400, tag=None, low=440, high=880):
    """DSF: блоки по 4096 байт на канал, младший бит первым, теги — ID3 в конце."""
    bits = pdm(rate, low, high)
    samples = len(bits)
    ch = np.packbits(bits, bitorder="little")
    blocks = len(ch) // 4096 + 1
    pad = np.zeros(blocks * 4096, np.uint8)
    pad[:len(ch)] = ch
    body = np.column_stack([pad.reshape(-1, 4096)] * 2).tobytes()
    data = b"data" + struct.pack("<Q", 12 + len(body)) + body
    fmt = b"fmt " + struct.pack("<QIIIIIIQII", 52, 1, 0, 2, 2, rate, 1, samples, 4096, 0)
    meta = id3(tag) if tag else b""
    total = 28 + len(fmt) + len(data) + len(meta)
    head = b"DSD " + struct.pack("<QQQ", 28, total, 28 + len(fmt) + len(data) if meta else 0)
    path.write_bytes(head + fmt + data + meta)
    return path


def make_dff(path, rate=2822400, tag=None, low=440, high=880):
    """DFF: старший бит первым, байты каналов через один, теги — чанк ID3 после звука."""
    def chunk(cid, body):
        return cid + struct.pack(">Q", len(body)) + body + (b"\0" if len(body) & 1 else b"")

    ch = np.packbits(pdm(rate, low, high), bitorder="big")
    audio = np.column_stack([ch, ch]).tobytes()
    prop = b"SND " + chunk(b"FS  ", struct.pack(">I", rate)) + chunk(b"CHNL", struct.pack(">H", 2) + b"SLFTSRGT") \
        + chunk(b"CMPR", b"DSD " + b"\x0enot compressed\0")
    body = b"DSD " + chunk(b"FVER", bytes([1, 5, 0, 0])) + chunk(b"PROP", prop) + chunk(b"DSD ", audio)
    if tag:
        body += chunk(b"ID3 ", id3(tag))
    path.write_bytes(b"FRM8" + struct.pack(">Q", len(body)) + body)
    return path


def scanned(tmp_path, *files):
    lib = tmp_path / "lib"
    lib.mkdir(exist_ok=True)
    for f in files:
        if f.parent != lib:
            shutil.move(str(f), lib / f.name)
    cfg = config.Config(data_dir=tmp_path / "data", libraries=[lib])
    cfg.ensure_dirs()
    conn = db.connect(cfg.db_path)
    return conn, scan.scan(conn, [lib]), lib


def dominant_hz(x: np.ndarray) -> float:
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    return float(np.argmax(spec)) * F.SR_EFFNET / len(x)


@needs_ffmpeg
@pytest.mark.parametrize(("name", "bits"), [("alac.m4a", 16), ("aac.m4a", 0), ("t.wma", 0), ("t.wv", 16),
                                            ("t.tta", 16), ("alac.caf", 16), ("t.aac", 0)])
def test_new_format_is_indexed(tmp_path, name, bits):
    src = tone(tmp_path / name, CODECS[name])
    conn, rep, _ = scanned(tmp_path, src)
    assert rep.errors == [] and rep.added == 1, rep
    r = conn.execute("SELECT artist, album, title, sample_rate, channels, bits, duration, md5_source FROM tracks").fetchone()
    assert (r["sample_rate"], r["channels"], r["bits"], r["md5_source"]) == (44100, 2, bits, "bytes"), dict(r)
    assert abs(r["duration"] - SECONDS) < 0.5, r["duration"]
    if name not in ("t.aac", "t.tta", "alac.caf"):  # голый AAC тегов не несёт; у этих ffmpeg их не пишет
        assert (r["artist"], r["album"], r["title"]) == ("Artist A", "Album B", "Song C"), dict(r)


@needs_mac
def test_ape_is_indexed_with_tags(tmp_path):
    conn, rep, _ = scanned(tmp_path, make_ape(tmp_path / "t.ape"))
    assert rep.errors == [] and rep.added == 1, rep
    r = conn.execute("SELECT artist, album, title, tracknumber, sample_rate, channels, bits, duration FROM tracks").fetchone()
    assert (r["artist"], r["album"], r["title"], r["sample_rate"], r["channels"], r["bits"]) == \
        ("Artist A", "Album B", "Song C", 44100, 2, 16), dict(r)
    assert abs(r["duration"] - SECONDS) < 0.01, r["duration"]


@pytest.mark.parametrize("maker", [make_dsf, make_dff])
def test_dsd_is_indexed_with_tags(tmp_path, maker):
    p = maker(tmp_path / ("t.dsf" if maker is make_dsf else "t.dff"), tag="Song C")
    conn, rep, _ = scanned(tmp_path, p)
    assert rep.errors == [] and rep.added == 1, rep
    r = conn.execute("SELECT artist, title, sample_rate, channels, bits, duration FROM tracks").fetchone()
    assert (r["artist"], r["title"], r["sample_rate"], r["channels"], r["bits"]) == ("Artist A", "Song C", 2822400, 2, 1), dict(r)
    assert abs(r["duration"] - SECONDS) < 0.01, r["duration"]


@needs_ffmpeg
@pytest.mark.parametrize(("args", "rate", "bits"), [
    (["-c:a", "alac", "-sample_fmt", "s32p"], 96000, 24),
    (["-c:a", "alac", "-sample_fmt", "s32p"], 192000, 24),
    (["-c:a", "wavpack", "-sample_fmt", "s32p"], 176400, 32),
    (["-c:a", "wavpack", "-sample_fmt", "s16p"], 96000, 16),
])
def test_hi_res_rate_and_depth(tmp_path, args, rate, bits):
    ext = "m4a" if "alac" in args else "wv"
    info = audiokey.read(tone(tmp_path / f"t.{ext}", args, rate=rate))
    assert (info.sample_rate, info.bits) == (rate, bits), info
    assert abs(info.duration - SECONDS) < 0.1, info.duration


@needs_ffmpeg
def test_wavpack_counts_all_channels(tmp_path):
    p = tmp_path / "t.wv"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=2:sample_rate=48000",
                    "-af", "pan=5.1|FL=c0|FR=c0|FC=c0|LFE=c0|BL=c0|BR=c0", "-c:a", "wavpack", str(p)], check=True)
    assert audiokey.read(p).channels == 6


@needs_ffmpeg
def test_alac_extension_is_mp4(tmp_path):
    p = tone(tmp_path / "t.m4a", CODECS["alac.m4a"])
    renamed = tmp_path / "t.alac"
    shutil.copy(p, renamed)
    info = audiokey.read(renamed)
    assert (info.sample_rate, info.bits, info.tag("TITLE")) == (44100, 16, "Song C"), info
    assert info.md5 == audiokey.read(p).md5


@needs_ffmpeg
def test_codec_player_cannot_play_is_reported_not_indexed(tmp_path):
    """E-AC-3 в m4a: декодера в плеере нет, поэтому в медиатеку не берём, и это видно в отчёте."""
    p = tone(tmp_path / "t.m4a", ["-c:a", "eac3", "-b:a", "192k", "-f", "mp4"])
    conn, rep, _ = scanned(tmp_path, p)
    assert rep.added == 0 and len(rep.errors) == 1 and "ec-3" in rep.errors[0][1], rep
    assert conn.execute("SELECT count(*) FROM tracks").fetchone()[0] == 0


def _append_ape(p, out):
    """Редактор тегов заменяет блок в конце файла, а звук не трогает."""
    with open(p, "rb") as f:
        start, length = audiokey._mp3_stream(f)
    out.write_bytes(p.read_bytes()[:start + length] + ape_tail({"Title": "Совсем другое название " * 4}))


def _insert_asf_padding(p, out):
    """Редактор тегов меняет только заголовок: данные сдвигаются, но не меняются."""
    raw = p.read_bytes()
    size = struct.unpack("<Q", raw[16:24])[0]
    pad = bytes.fromhex("74D40618DFCA0945A4BA9AABCB96AAE8") + struct.pack("<Q", 24 + 4096) + b"\0" * 4096
    count = struct.unpack("<I", raw[24:28])[0]
    head = raw[:16] + struct.pack("<Q", size + len(pad)) + struct.pack("<I", count + 1) + raw[28:size] + pad
    out.write_bytes(head + raw[size:])


def _retag_ffmpeg(p, out):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(p), "-c", "copy", "-map_metadata", "0",
                    "-metadata", "title=" + "Совсем другое название " * 4, str(out)], check=True)


def _prepend_id3(p, out):
    out.write_bytes(id3("Совсем другое название") + p.read_bytes())


def _insert_caf_free(p, out):
    raw = p.read_bytes()
    out.write_bytes(raw[:8] + b"free" + struct.pack(">q", 512) + b"\0" * 512 + raw[8:])


@needs_ffmpeg
@pytest.mark.parametrize(("name", "retag"), [("alac.m4a", _retag_ffmpeg), ("aac.m4a", _retag_ffmpeg),
                                             ("t.wma", _insert_asf_padding), ("t.wv", _append_ape),
                                             ("t.tta", _append_ape), ("t.aac", _prepend_id3),
                                             ("alac.caf", _insert_caf_free)])
def test_key_survives_retag_and_differs_for_other_sound(tmp_path, name, retag):
    a = tone(tmp_path / name, CODECS[name])
    k = audiokey.read(a).md5
    # красное: другой звук — другой ключ
    b = tone(tmp_path / f"b{name}", CODECS[name], low=660, high=990)
    assert audiokey.read(b).md5 != k
    # зелёное: другие теги, тот же звук; копия по другому пути тоже
    out = tmp_path / f"re{name}"
    retag(a, out)
    assert audiokey.read(out).md5 == k
    assert a.read_bytes() != out.read_bytes()


@needs_mac
def test_ape_key_survives_retag(tmp_path):
    a = make_ape(tmp_path / "a.ape", tail=False)
    k = audiokey.read(a).md5
    assert audiokey.read(make_ape(tmp_path / "b.ape", low=660, high=990, tail=False)).md5 != k
    out = tmp_path / "re.ape"
    _append_ape(a, out)
    assert audiokey.read(out).md5 == k and audiokey.read(out).tag("TITLE").startswith("Совсем")


@pytest.mark.parametrize("maker", [make_dsf, make_dff])
def test_dsd_key_survives_retag(tmp_path, maker):
    ext = "dsf" if maker is make_dsf else "dff"
    k = audiokey.read(maker(tmp_path / f"a.{ext}", tag="Один")).md5
    assert audiokey.read(maker(tmp_path / f"b.{ext}", tag="Один", low=660, high=990)).md5 != k
    assert audiokey.read(maker(tmp_path / f"c.{ext}", tag="Совсем другое и длинное название")).md5 == k


@needs_ffmpeg
@pytest.mark.parametrize("name", sorted(CODECS))
def test_excerpt_comes_from_the_right_place(tmp_path, name):
    p = tone(tmp_path / name, CODECS[name])
    assert abs(dominant_hz(F.load_mono(p, F.SR_EFFNET, 1.0, 2.0)) - 440) < 5
    assert abs(dominant_hz(F.load_mono(p, F.SR_EFFNET, 5.0, 2.0)) - 880) < 5
    whole = F.load_mono(p, F.SR_EFFNET)
    assert abs(len(whole) / F.SR_EFFNET - SECONDS) < 0.2
    pats = F.patches(F.melspec(F.load_mono(p, F.SR_EFFNET, 1.0, 6.0)))  # отрывок, как в embed.excerpt_patches
    assert pats.shape[1:] == (F.PATCH, F.N_MELS) and len(pats) >= 3


@needs_ffmpeg
def test_stereo_loader_for_facets(tmp_path):
    from arxnova import facets

    p = tone(tmp_path / "alac.m4a", CODECS["alac.m4a"])
    x = facets.load_stereo(p, 5.0, 2.0)
    assert x.shape[0] == 2 and abs(x.shape[1] - 2 * facets.SR) < 2000
    assert abs(dominant_hz(facets._mono16k(x)) - 880) < 5
    mono = tmp_path / "mono.wv"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=2:sample_rate=44100",
                    "-c:a", "wavpack", str(mono)], check=True)
    assert facets.load_stereo(mono).shape[0] == 2  # моно разводится на два канала, как у soundfile


@needs_ffmpeg
def test_clap_worker_reads_through_ffmpeg(tmp_path, monkeypatch):
    """Процесс CLAP живёт в своём окружении и не импортирует arxnova: у него своя короткая копия чтения."""
    import importlib.util
    import sys
    import types

    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace())
    spec = importlib.util.spec_from_file_location("clap_worker_test", F.__file__.replace("features.py", "clap_worker.py"))
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    p = tone(tmp_path / "t.wv", CODECS["t.wv"])
    w = worker.windows_ffmpeg(shutil.which("ffmpeg"), str(p), 0.0, float(SECONDS))
    assert w.shape == (3, int(worker.WIN_S * worker.SR))
    assert np.abs(w[0]).max() > 0.1
    with pytest.raises(ValueError):
        worker.windows_ffmpeg(shutil.which("ffmpeg"), str(p), 0.0, None)


@needs_mac
def test_excerpt_from_ape(tmp_path):
    p = make_ape(tmp_path / "t.ape")
    assert abs(dominant_hz(F.load_mono(p, F.SR_EFFNET, 1.0, 2.0)) - 440) < 5
    assert abs(dominant_hz(F.load_mono(p, F.SR_EFFNET, 5.0, 2.0)) - 880) < 5


@needs_ffmpeg
@pytest.mark.parametrize("maker", [make_dsf, make_dff])
def test_dsd_excerpt_is_the_tone_not_noise(tmp_path, maker):
    """Порядок бит и чередование каналов верны, только если из однобитного потока вышел тон."""
    p = maker(tmp_path / ("t.dsf" if maker is make_dsf else "t.dff"))
    assert abs(dominant_hz(F.load_mono(p, F.SR_EFFNET, 1.0, 2.0)) - 440) < 5
    assert abs(dominant_hz(F.load_mono(p, F.SR_EFFNET, 5.0, 2.0)) - 880) < 5


@needs_ffmpeg
def test_cue_image_in_new_format(tmp_path):
    lib = tmp_path / "lib"
    lib.mkdir()
    tone(lib / "image.wv", CODECS["t.wv"])
    (lib / "image.cue").write_text('PERFORMER "Artist A"\nTITLE "Album B"\nFILE "image.wv" WAVE\n'
                                   '  TRACK 01 AUDIO\n    TITLE "One"\n    INDEX 01 00:00:00\n'
                                   '  TRACK 02 AUDIO\n    TITLE "Two"\n    INDEX 01 00:04:00\n')
    conn, rep, _ = scanned(tmp_path)
    assert rep.errors == [], rep
    rows = conn.execute("SELECT title, cue_ms, cue_end_ms FROM tracks ORDER BY cue_ms").fetchall()
    assert [tuple(r) for r in rows] == [("One", 0, 4000), ("Two", 4000, -1)]
    assert abs(dominant_hz(F.load_mono(lib / "image.wv", F.SR_EFFNET, 4.5, 2.0)) - 880) < 5


@needs_ffmpeg
def test_cue_of_image_does_not_split_neighbours(tmp_path):
    """Лист на один файл не делит соседние файлы папки. А один файл с чужим именем в листе — делит."""
    lib = tmp_path / "lib"
    lib.mkdir()
    sheet = 'FILE "image.wav" WAVE\n  TRACK 01 AUDIO\n    INDEX 01 00:00:00\n  TRACK 02 AUDIO\n    INDEX 01 00:04:00\n'
    tone(lib / "image.wv", CODECS["t.wv"])
    tone(lib / "bonus.wv", CODECS["t.wv"], low=660, high=990)
    (lib / "image.cue").write_text(sheet)
    conn, rep, _ = scanned(tmp_path)
    assert rep.errors == [], rep
    assert sorted(r[0] for r in conn.execute("SELECT cue_ms FROM tracks")) == [0, 0, 4000]

    solo = tmp_path / "solo"
    solo.mkdir()
    tone(solo / "renamed.wv", CODECS["t.wv"])
    (solo / "other.cue").write_text(sheet)
    found = cue.find_for(solo / "renamed.wv")
    assert found and [t.start_ms for t in found[1]] == [0, 4000]


@needs_ffmpeg
def test_without_ffmpeg_tracks_wait_and_do_not_count(tmp_path, monkeypatch):
    """Новый формат в медиатеке есть, разбор ждёт ffmpeg. Красное: без него читать нечем."""
    flac_like = tone(tmp_path / "plain.wav", ["-c:a", "pcm_s16le"])
    alac = tone(tmp_path / "alac.m4a", CODECS["alac.m4a"])
    conn, rep, lib = scanned(tmp_path, flac_like, alac)
    assert rep.added == 2, rep
    assert len(indexer.pending(conn, "m")) == 2 and analysis.pending(conn)["tracks"] == 2
    assert analysis.pending(conn)["need_ffmpeg"] == 0

    monkeypatch.setattr(decode, "ffmpeg_path", lambda: None)
    with pytest.raises(decode.DecoderMissing):
        F.load_mono(lib / "alac.m4a", F.SR_EFFNET)
    assert [r[1].endswith("plain.wav") for r in indexer.pending(conn, "m")] == [True]
    assert [r[1].endswith("plain.wav") for r in indexer.maest_pending(conn)] == [True]
    counts = analysis.pending(conn)
    assert (counts["tracks"], counts["need_ffmpeg"]) == (1, 1)
    assert len(F.load_mono(lib / "plain.wav", F.SR_EFFNET)) > 0  # soundfile работает как работал


@needs_ffmpeg
def test_broken_file_is_a_decode_error_not_a_hang(tmp_path):
    p = tmp_path / "t.m4a"
    p.write_bytes(b"\0" * 4096)
    with pytest.raises(decode.DecodeError):
        F.load_mono(p, F.SR_EFFNET)


@needs_ffmpeg
def test_covers_of_new_formats(tmp_path):
    from PIL import Image

    img = tmp_path / "c.png"
    Image.new("RGB", (8, 8), (200, 10, 10)).save(img)
    png = img.read_bytes()
    src = tone(tmp_path / "s.wav", ["-c:a", "pcm_s16le"], tags={})
    m4a = tmp_path / "c.m4a"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-i", str(img), "-map", "0:a", "-map", "1:v", "-c:a", "alac",
                    "-c:v", "copy", "-disposition:v", "attached_pic", str(m4a)], check=True)
    mime, data = audiokey.read_picture(m4a)
    assert data[:4] == b"\x89PNG"

    wv = tone(tmp_path / "c.wv", CODECS["t.wv"], tags={})
    cover = b"cover.png\0" + png
    item = struct.pack("<II", len(cover), 1 << 1) + b"Cover Art (Front)\0" + cover
    with open(wv, "ab") as f:
        f.write(item + b"APETAGEX" + struct.pack("<IIII", 2000, len(item) + 32, 1, 0) + b"\0" * 8)
    mime, data = audiokey.read_picture(wv)
    assert mime == "image/png" and data == png

    def apic(block):
        body = b"\0image/png\0\x03\0" + png
        frame = b"APIC" + struct.pack(">I", len(body)) + b"\0\0" + body
        n = len(frame)
        return b"ID3\x03\0\0" + bytes([(n >> 21) & 0x7F, (n >> 14) & 0x7F, (n >> 7) & 0x7F, n & 0x7F]) + frame

    d = make_dsf(tmp_path / "c.dsf", tag="x")
    raw = d.read_bytes()
    meta = struct.unpack("<Q", raw[20:28])[0]
    d.write_bytes(raw[:meta] + apic(None))
    mime, data = audiokey.read_picture(d)
    assert data == png


@needs_ffmpeg
@pytest.mark.parametrize("name", sorted(CODECS))
def test_damaged_header_never_breaks_the_scan(tmp_path, name):
    """Сканер ловит только AudioError: всё остальное из разбора уронило бы весь обход медиатеки."""
    raw = tone(tmp_path / name, CODECS[name]).read_bytes()
    _assert_only_audio_error(tmp_path, name, raw)


@pytest.mark.parametrize("maker", [make_dsf, make_dff])
def test_damaged_dsd_never_breaks_the_scan(tmp_path, maker):
    name = "t.dsf" if maker is make_dsf else "t.dff"
    _assert_only_audio_error(tmp_path, name, maker(tmp_path / name, tag="x").read_bytes())


@needs_mac
def test_damaged_ape_never_breaks_the_scan(tmp_path):
    _assert_only_audio_error(tmp_path, "t.ape", make_ape(tmp_path / "t.ape").read_bytes())


def test_header_offset_beyond_filesystem_is_a_bad_file(tmp_path, monkeypatch):
    """На ext4 seek на такое смещение отвечает EINVAL, на APFS и tmpfs молчит: поэтому проверка без диска."""
    p = tmp_path / "t.dsf"
    p.write_bytes(b"\0" * 128)

    def fail(code):
        def head(f):
            raise OSError(code, "тест")
        return head

    monkeypatch.setitem(audiokey._HEADS, ".dsf", fail(errno.EINVAL))
    with pytest.raises(audiokey.AudioError):
        audiokey.read(p)
    monkeypatch.setitem(audiokey._HEADS, ".dsf", fail(errno.EIO))  # ошибка диска — не вина файла
    with pytest.raises(OSError):
        audiokey.read(p)


def _assert_only_audio_error(tmp_path, name, raw):
    rnd = random.Random(1)
    probe = tmp_path / f"probe{name}"
    head = min(len(raw), 512)
    variants = [raw[:n] for n in (0, 1, 3, 4, 8, 11, 12, 16, 24, 31, 32, 40, 52, 64, 76, 92, 100, 200)]
    for _ in range(150):
        b = bytearray(raw[:head])
        for _ in range(rnd.randint(1, 12)):
            b[rnd.randrange(head)] = rnd.randrange(256)
        variants.append(bytes(b) + raw[head:head + 4096])
    for v in variants:
        probe.write_bytes(v)
        try:
            audiokey.read(probe)
        except audiokey.AudioError:
            pass


def test_untagged_file_is_titled_by_its_name():
    from arxnova import server

    row = {"id": 1, "md5": b"\0", "cue_ms": 0, "cue_end_ms": -1, "title": None, "artist": None, "album": None,
           "date": None, "duration": 1.0, "sample_rate": 44100, "bits": 0, "channels": 2, "size": 1,
           "path": "/m/a/03 Без тегов.aac"}
    assert server.track_json(row)["title"] == "03 Без тегов"
    assert server.track_json({**row, "title": "Из тегов"})["title"] == "Из тегов"


def test_malloc_arena_limit_is_harmless():
    """На Linux с glibc он сокращает резерв адресов (замер на Pi: 996 МБ из 1,9 ГБ), в остальных местах ничего не делает."""
    import arxnova

    arxnova._fewer_malloc_arenas()


def test_every_format_is_watched_and_served():
    from arxnova import cli, folders, server

    missing = [e for e in audiokey.EXTENSIONS if server.media_type(f"x{e}") == "application/octet-stream"]
    assert not missing, missing
    assert set(audiokey.EXTENSIONS) <= set(cli.AUDIO_EXT) and folders.AUDIO == audiokey.EXTENSIONS
    assert decode.NEEDS_FFMPEG <= audiokey.EXTENSIONS
    assert server.media_type("a/b.M4A") == "audio/mp4" and server.media_type("x.dsf") == "audio/x-dsf"
