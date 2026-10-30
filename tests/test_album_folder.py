from pathlib import Path

from arxnova.server import album_folder


def test_disc_folders_belong_to_one_album():
    # CD1 и CD2 одного издания — один альбом (замечание владельца 27.10.2026)
    base = "/m/LTJ Bukem/1996 (3) - Logical Progression"
    assert album_folder(f"{base}/CD1/01.flac") == Path(base)
    assert album_folder(f"{base}/CD2/01.flac") == Path(base)
    assert album_folder(f"{base}/CD1 - LTJ Bukem feat. MC Conrad/01.flac") == Path(base)
    for disc in ("Disc 1", "disk2", "CD 3", "cd_4", "Диск 1"):
        assert album_folder(f"{base}/{disc}/01.flac") == Path(base)


def test_ordinary_folders_stay_as_they_are():
    assert album_folder("/m/Artist/Album/01.flac") == Path("/m/Artist/Album")
    # слово в начале названия — не папка диска
    assert album_folder("/m/Artist/CDs of the year/01.flac") == Path("/m/Artist/CDs of the year")
    assert album_folder("/m/Artist/Discovery/01.flac") == Path("/m/Artist/Discovery")
