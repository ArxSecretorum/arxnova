"""Фокус медиатеки сервера (решение 0022): группы полей альбома для фильтра в плеере и тег жанра в треках."""

import shutil
import subprocess

import pytest
from fastapi.testclient import TestClient

from arxnova import auth, config, db, scan
from arxnova.server import create_app
from tests.test_flac_keys import make_flac
from tests.test_formats_more import make_dsf


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    if not shutil.which("ffmpeg"):
        pytest.skip("нужен ffmpeg")
    root = tmp_path_factory.mktemp("focus")
    lib = root / "lib"
    for d in ("cd", "hires", "mixed", "dsd", "mp3"):
        (lib / d).mkdir(parents=True)
    for i in range(2):  # CD, 1975, составной жанр
        make_flac(lib / "cd" / f"{i}.flac", seconds=0.3, freq=300 + 20 * i, ARTIST="A", ALBUM="Seventies",
                  TITLE=f"S{i}", DATE="1975-03-01", GENRE="Jazz; Fusion")
    make_flac(lib / "hires" / "0.flac", bits=24, rate=96000, seconds=0.3, freq=500, ARTIST="B", ALBUM="Hires",
              TITLE="H0", DATE="2019", GENRE="Electronic")
    make_flac(lib / "mixed" / "0.flac", seconds=0.3, freq=600, ARTIST="C", ALBUM="Mixed", TITLE="M0", DATE="1999")
    make_flac(lib / "mixed" / "1.flac", bits=24, rate=96000, seconds=0.3, freq=650, ARTIST="C", ALBUM="Mixed",
              TITLE="M1", DATE="1999")
    make_dsf(lib / "dsd" / "0.dsf", tag="Dsd Song")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=f=700:d=0.5", "-c:a", "libmp3lame",
                    "-metadata", "ARTIST=D", "-metadata", "ALBUM=Lossy", "-metadata", "TITLE=L0",
                    "-metadata", "date=1983", str(lib / "mp3" / "0.mp3")], check=True)
    cfg = config.Config(data_dir=root / "data", libraries=[lib])
    cfg.ensure_dirs()
    conn = db.connect(cfg.db_path, threaded=True)
    scan.scan(conn, [lib])
    client = TestClient(create_app(cfg))
    code = auth.new_pair_code(conn)
    tok = client.post("/v1/pair", json={"code": code, "name": "test"}).json()["token"]
    return client, {"Authorization": f"Bearer {tok}"}


def _albums(client, h, **params) -> dict:
    return {a["title"]: a for a in client.get("/v1/library/albums", headers=h, params=params).json()["items"]}


def test_without_facets_answer_is_unchanged(env):
    client, h = env
    items = _albums(client, h)
    assert set(items) >= {"Seventies", "Hires", "Mixed", "Lossy"}
    assert all("facets" not in a for a in items.values())  # старый плеер не платит за новые байты


def test_album_facets_group_tracks_by_raw_fields(env):
    client, h = env
    items = _albums(client, h, facets=1)
    assert items["Seventies"]["facets"] == [["flac", 44100, 16, "1975", "Jazz; Fusion", 2]]
    assert items["Hires"]["facets"] == [["flac", 96000, 24, "2019", "Electronic", 1]]
    assert sorted(items["Mixed"]["facets"]) == [["flac", 44100, 16, "1999", "", 1], ["flac", 96000, 24, "1999", "", 1]]
    lossy = items["Lossy"]["facets"]
    assert len(lossy) == 1 and lossy[0][0] == "mp3" and lossy[0][2] == 0 and lossy[0][3] == "1983"
    dsd = next(a for a in items.values() if a["facets"] and a["facets"][0][0] == "dsf")
    assert dsd["facets"][0][1] == 2822400 and dsd["facets"][0][5] == 1


def test_tracks_carry_the_genre_tag(env):
    client, h = env
    found = client.get("/v1/library/search", headers=h, params={"q": "Seventies"}).json()["tracks"]
    assert found and all(t["genre_tag"] == "Jazz; Fusion" for t in found)
    album_id = _albums(client, h)["Mixed"]["id"]
    tracks = client.get(f"/v1/library/albums/{album_id}", headers=h).json()["tracks"]
    assert all("genre_tag" not in t for t in tracks)  # тега нет — поля нет
