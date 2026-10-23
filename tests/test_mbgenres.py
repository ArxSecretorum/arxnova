"""Жанры MusicBrainz: поиск, кэш, повтор ненайденного через 30 дней, выбор альбом → исполнитель."""


from arxnova import db, mbgenres


def fake(responses, calls):
    def fetch(url):
        calls.append(url)
        for key, val in responses.items():
            if key in url:
                return val
        return {}
    return fetch


def test_refresh_and_track_genres(tmp_path, monkeypatch):
    monkeypatch.setattr(mbgenres, "PAUSE_S", 0)
    conn = db.connect(tmp_path / "d.sqlite3")
    with conn:
        conn.execute("INSERT INTO tracks (id, md5, md5_source, recording_key, recording_basis, sample_rate, bits, channels, "
                     "duration, artist, albumartist, album, title) VALUES (1, x'01', 'streaminfo', x'01', 'x', 44100, 16, 2, "
                     "100, 'Calibre', 'Calibre', 'Shelflife', 'Mr Majestic')")
        conn.execute("INSERT INTO tracks (id, md5, md5_source, recording_key, recording_basis, sample_rate, bits, channels, "
                     "duration, artist, albumartist, album, title) VALUES (2, x'02', 'streaminfo', x'02', 'x', 44100, 16, 2, "
                     "100, 'Nobody', 'Nobody', 'Unknown LP', 'X')")
    calls = []
    resp = {
        "release-group/?": {"release-groups": [{"id": "rg1", "score": 100}]},
        "release-group/rg1": {"genres": [{"name": "Liquid Funk", "count": 3}], "tags": [{"name": "drum and bass", "count": 5}]},
        "artist/?": {"artists": [{"id": "a1", "score": 100}]},
        "artist/a1": {"genres": [{"name": "drum and bass", "count": 9}]},
    }
    client = mbgenres.Client(fetch=fake(resp, calls))
    # альбомы нашлись — исполнителей не спрашиваем (только для ненайденных альбомов)
    assert mbgenres.refresh(conn, client) == 2
    assert all(u.startswith(mbgenres.API) and "fmt=json" in u for u in calls)
    g = mbgenres.track_genres(conn)
    assert g[1] == {"liquid funk": 3, "drum and bass": 5}  # жанры альбома (теги — тоже)
    # повтор — ничего не спрашивает: всё в кэше (и ненайденное — до 30 дней)
    calls.clear()
    assert mbgenres.refresh(conn, client) == 0 and calls == []
    with conn:
        conn.execute("UPDATE mb_cache SET at = at - ? WHERE mbid IS NULL", (mbgenres.RETRY_MISS_S + 1,))
    rows = conn.execute("SELECT count(*) FROM mb_cache WHERE mbid IS NULL").fetchone()[0]
    assert mbgenres.refresh(conn, client) == rows
    assert mbgenres._primary("Aperio, Kasger") == "Aperio" and mbgenres._primary("Calibre feat. DRS") == "Calibre"


def test_disabled_by_default(tmp_path):
    from arxnova import config

    cfg = config.Config(data_dir=tmp_path)
    assert cfg.musicbrainz is False
    cfg.musicbrainz = True
    cfg.save()
    assert config.load(tmp_path).musicbrainz is True
