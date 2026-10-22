"""Медиатека разбирается по событиям ОС, а не по таймеру: новый FLAC будит обход, картинка — нет."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

from arxnova import analysis, cli


def _wait(ev, s):
    end = time.monotonic() + s
    while time.monotonic() < end:
        if ev.is_set():
            return True
        time.sleep(0.1)
    return False


def test_new_flac_wakes_scan_but_cover_does_not(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "QUIET_S", 0.2)
    stop = threading.Event()
    analysis.wake.clear()
    obs = cli.watch_library(SimpleNamespace(libraries=[tmp_path]), stop)
    try:
        time.sleep(0.5)
        (tmp_path / "cover.jpg").write_bytes(b"x")  # красное: картинка не повод разбирать медиатеку
        assert not _wait(analysis.wake, 7)
        (tmp_path / "track.flac").write_bytes(b"fLaC")
        assert _wait(analysis.wake, 10)
    finally:
        stop.set()
        obs.stop()
        analysis.wake.clear()
