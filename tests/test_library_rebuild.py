"""Новые треки не останавливают сервер (04.11.2026). Сборка каталога на Pi идёт 21 с, плеер ждёт ответа 15 с.
Сборка шла под общим замком, и каждый дождавшийся запрос собирал каталог ещё раз: сервер молчал минутами."""

import os
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from arxnova import config, models, nova
from arxnova import server as S


class Builds:
    """Сборка, которая идёт, пока тест её не отпустит."""

    def __init__(self):
        self.count = 0
        self.fail = None
        self.started = threading.Event()
        self.gate = threading.Event()
        self.gate.set()

    def __call__(self, *args):
        self.count += 1
        n = self.count
        self.started.set()
        assert self.gate.wait(30)
        if self.fail:
            e, self.fail = self.fail, None
            raise e
        return SimpleNamespace(n=n)


@pytest.fixture
def env(tmp_path, monkeypatch):
    cfg = config.Config(data_dir=tmp_path / "data", libraries=[tmp_path / "lib"])
    cfg.ensure_dirs()
    monkeypatch.setattr(models, "is_installed", lambda *a, **k: True)
    builds = Builds()
    monkeypatch.setattr(nova, "Library", builds)
    d = cfg.index_dir / models.EFFNET.id
    d.mkdir(parents=True)
    np.save(d / "ids.npy", np.arange(3))
    index_written(d, ago=200)
    st = S.State(cfg)
    st.index_settle_s = 0
    yield st, d, builds
    builds.gate.set()
    st.conn.close()


def index_written(d, ago=100.0):
    t = time.time_ns() - int(ago * 1e9)
    os.utime(d / "ids.npy", ns=(t, t))


def wait_until(cond, timeout=10.0):
    deadline = time.monotonic() + timeout
    while not cond():
        assert time.monotonic() < deadline, "не дождались"
        time.sleep(0.01)


def lock_is_free(st) -> bool:
    got = st.lock.acquire(timeout=1)
    if got:
        st.lock.release()
    return got


def in_threads(fn, n=5):
    got = []
    threads = [threading.Thread(target=lambda: got.append(fn()), daemon=True) for _ in range(n)]
    for t in threads:
        t.start()
    return threads, got


def test_new_index_does_not_stop_requests_and_is_built_once(env):
    st, d, builds = env
    old = st.library()
    index_written(d)
    builds.started.clear()
    builds.gate.clear()
    threads, got = in_threads(st.library)
    assert builds.started.wait(5)
    for t in threads:
        t.join(3)
    assert not any(t.is_alive() for t in threads), "запросы ждут сборку"
    assert got == [old] * 5, "пока идёт сборка, отвечает прежний каталог"
    assert lock_is_free(st), "сборка держит общий замок"
    builds.gate.set()
    wait_until(lambda: st.library() is not old)
    assert builds.count == 2


def test_first_build_is_one_for_all_waiters(env):
    st, _, builds = env
    builds.gate.clear()
    threads, got = in_threads(st.library)
    assert builds.started.wait(5)
    assert lock_is_free(st), "сборка держит общий замок"
    builds.gate.set()
    for t in threads:
        t.join(5)
    assert builds.count == 1
    assert len(got) == 5 and all(x is got[0] for x in got)


def test_failed_build_keeps_old_catalog_and_does_not_spin(env):
    st, d, builds = env
    old = st.library()
    index_written(d)
    builds.fail = ValueError("битый файл индекса")
    assert st.library() is old
    wait_until(lambda: builds.count == 2 and not builds.fail and st._lib_lock.acquire(blocking=False) and (st._lib_lock.release() or True))
    assert st.library() is old
    time.sleep(0.1)
    assert builds.count == 2, "упавшая сборка не повторяется на каждый запрос"
    st._lib_retry_at = 0.0
    assert st.library() is old
    wait_until(lambda: st.library() is not old)
    assert builds.count == 3


def test_no_room_for_two_catalogs_frees_the_old_one(env):
    st, d, builds = env
    old = st.library()
    index_written(d)
    builds.fail = MemoryError()
    assert st.library() is old
    wait_until(lambda: st._lib is not None and st._lib is not old)
    assert builds.count == 3


def test_index_still_being_written_is_not_read(env):
    st, d, builds = env
    old = st.library()
    st.index_settle_s = 30
    index_written(d, ago=0)
    assert st.library() is old
    time.sleep(0.1)
    assert builds.count == 1, "файлы индекса ещё пишутся"
    index_written(d, ago=31)
    assert st.library() is old
    wait_until(lambda: st.library() is not old)
    assert builds.count == 2


def test_settings_changed_during_build_are_not_lost(env):
    st, d, builds = env
    st.library()
    index_written(d)
    builds.started.clear()
    builds.gate.clear()
    st.library()
    assert builds.started.wait(5)
    st.forget_library()  # новый порог шлюза, а идущая сборка читала прежний
    threads, got = in_threads(st.library, n=1)
    time.sleep(0.05)
    assert lock_is_free(st)
    builds.gate.set()
    threads[0].join(5)
    assert got[0].n == 3, "каталог собран заново, уже с новыми настройками"
    assert builds.count == 3


def test_any_index_file_counts(env):
    st, d, _ = env
    before = st._index_stamp()
    p = d / "scenarios.json"  # сценарии выходят позже матриц: на Pi через 47 с
    p.write_text("{}")
    t = time.time_ns()
    os.utime(p, ns=(t, t))
    assert st._index_stamp() != before


def test_catalog_is_built_once_and_not_under_the_common_lock(env, monkeypatch):
    st, _, builds = env
    monkeypatch.setattr(S, "Catalog", builds)
    builds.gate.clear()
    threads, got = in_threads(st.catalog)
    assert builds.started.wait(5)
    assert lock_is_free(st), "сборка держит общий замок"
    builds.gate.set()
    for t in threads:
        t.join(5)
    assert builds.count == 1
    assert len(got) == 5 and all(x is got[0] for x in got)
