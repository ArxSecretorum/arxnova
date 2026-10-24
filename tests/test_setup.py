"""Первая настройка с телефона: новый сервер забирает плеер из домашней сети без кода, владелец
выбирает папку с музыкой обзором папок сервера (решение владельца 23.10.2026)."""

from fastapi.testclient import TestClient

from arxnova import config, folders, tlsid
from arxnova.reach import Reach
from arxnova.server import create_app


def _app(tmp_path):
    cfg = config.Config(data_dir=tmp_path / "data", remote_port=23056)
    cfg.ensure_dirs()
    ident = tlsid.ensure(cfg.data_dir, "дом")
    app = create_app(cfg, Reach(23056), ident.fingerprint)
    return cfg, app


def test_claim_from_home_only_once(tmp_path):
    cfg, app = _app(tmp_path)
    home = TestClient(app, base_url="http://testserver:8765", client=("192.168.0.199", 40000))
    away = TestClient(app, base_url="https://testserver:23056", client=("192.168.0.199", 40000))
    stranger = TestClient(app, base_url="http://testserver:8765", client=("8.8.8.8", 40000))

    assert home.get("/v1/health").json()["setup"] is True
    assert "setup" not in stranger.get("/v1/health").json()
    # из интернета и с публичного адреса — нельзя, даже пока владельца нет
    assert away.post("/v1/setup/claim", json={"name": "M9"}).status_code == 403
    assert stranger.post("/v1/setup/claim", json={"name": "M9"}).status_code == 403

    r = home.post("/v1/setup/claim", json={"name": "M9"})
    assert r.status_code == 200
    token = r.json()["token"]
    assert r.json()["net"]["tls"]["port"] == 23056
    # второй раз — уже нет: дальше только по коду сопряжения
    assert home.post("/v1/setup/claim", json={"name": "чужой"}).status_code == 409
    assert "setup" not in home.get("/v1/health").json()
    me = home.get("/v1/devices/me", headers={"Authorization": f"Bearer {token}"}).json()
    assert me["role"] == "owner"


def test_pick_library_from_phone(tmp_path):
    cfg, app = _app(tmp_path)
    music = tmp_path / "disk" / "Music"
    for i in range(25):
        album = music / f"Artist {i % 3}" / f"Album {i}"
        album.mkdir(parents=True, exist_ok=True)
        (album / "01.flac").write_bytes(b"")
    (tmp_path / "disk" / "Photos").mkdir()
    (tmp_path / "disk" / "$RECYCLE.BIN").mkdir()  # служебные папки не показываем
    (tmp_path / "disk" / "System Volume Information").mkdir()
    called = []
    app.state.on_libraries = lambda: called.append(True)
    home = TestClient(app, base_url="http://testserver:8765", client=("192.168.0.199", 40000))
    assert home.get("/v1/setup/folders").status_code == 401  # только владельцу
    h = {"Authorization": "Bearer " + home.post("/v1/setup/claim", json={"name": "M9"}).json()["token"]}

    top = home.get("/v1/setup/folders", headers=h).json()
    assert top["roots"] and top["libraries"] == []
    assert top["roots"][-1]["kind"] == "home" and top["roots"][-1]["size"] > 0  # съёмные — первыми, домашняя — в конце
    ls = home.get("/v1/setup/folders", params={"path": str(tmp_path / "disk")}, headers=h).json()
    assert [d["name"] for d in ls["dirs"]] == ["Music", "Photos"] and ls["dirs"][0]["has_dirs"]
    assert home.get("/v1/setup/folders", params={"path": str(tmp_path / "нет")}, headers=h).status_code == 404

    assert home.put("/v1/setup/library", json={"paths": [str(tmp_path / "нет")]}, headers=h).status_code == 400
    r = home.put("/v1/setup/library", json={"paths": [str(music)]}, headers=h)
    assert r.status_code == 200 and called == [True]
    assert cfg.libraries == [music]
    assert config.load(cfg.data_dir).libraries == [music]  # сохранено
    assert home.get("/v1/health").json()["libraries"] == 1


def test_suggest_picks_collecting_folder(tmp_path, monkeypatch):
    disk = tmp_path / "disk"
    for i in range(30):
        a = disk / "Music" / f"Artist {i % 4}" / f"Album {i}"
        a.mkdir(parents=True, exist_ok=True)
        (a / "t.flac").write_bytes(b"")
    (disk / "Docs").mkdir()
    monkeypatch.setattr(folders, "roots", lambda: [{"title": "Диск", "path": str(disk)}])
    got = folders.suggest()
    # не весь диск и не папка одного исполнителя — папка Music, где собрана музыка
    assert got[0]["path"] == str(disk / "Music") and got[0]["audio"] == 30


def test_linux_mounts_find_usb_and_nas_volumes():
    """Диски Linux — из таблицы монтирования: USB-диск в /media, тома NAS (Synology /volume1),
    сетевой диск; служебное (корень, /boot, снапы, tmpfs, gvfs) — мимо."""
    mounts = "\n".join([
        "/dev/mmcblk0p2 / ext4 rw 0 0",
        "/dev/mmcblk0p1 /boot/firmware vfat rw 0 0",
        "tmpfs /run tmpfs rw 0 0",
        "gvfsd-fuse /run/user/1000/gvfs fuse.gvfsd-fuse rw 0 0",
        "/dev/loop3 /snap/core/1 squashfs ro 0 0",
        r"/dev/sda2 /media/roman/Backup\040Plus exfat rw 0 0",
        "/dev/md2 /volume1 btrfs rw 0 0",
        "/dev/sdb1 /srv/music ext4 rw 0 0",
        "//nas/music /mnt/nas cifs rw 0 0",
    ])
    got = folders.linux_mounts(mounts, is_usb=lambda dev: dev == "/dev/sdb1")
    assert [(t, p.as_posix(), k) for t, p, k in got] == [  # as_posix: тест идёт и на Windows
        ("Backup Plus", "/media/roman/Backup Plus", "external"),
        ("volume1", "/volume1", "disk"),
        ("music", "/srv/music", "external"),  # USB-диск, смонтированный не в /media
        ("nas", "/mnt/nas", "external"),
    ]
