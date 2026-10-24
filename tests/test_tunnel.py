"""Канал поверх UDP (пробивка NAT, ОД-3 п. 4): рукопожатие с закреплённым ключом, большие
ответы, потери пакетов, подмена сервера."""

import asyncio
import os

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import Response

from arxnova import tlsid, tunnel

BIG = os.urandom(700_000)


def stub_app():
    app = FastAPI()

    @app.get("/v1/health")
    def health(request: Request):
        return {"name": "arxnova", "port": request.scope["server"][1], "client": request.client.host}

    @app.get("/big")
    def big(request: Request):
        r = request.headers.get("range")
        if r:
            a, b = (int(x) for x in r[6:].split("-"))
            return Response(BIG[a:b + 1], status_code=206, headers={"Content-Range": f"bytes {a}-{b}/{len(BIG)}"})
        return Response(BIG)

    @app.post("/echo")
    async def echo(request: Request):
        return Response(await request.body())

    return app


async def _pair(tmp_path, drop=0.0, fp_override=None):
    ident = tlsid.ensure(tmp_path, "test")
    server = await tunnel.start(stub_app(), 0, ident.cert, ident.key, ident.fingerprint)
    port = server.transport.get_extra_info("sockname")[1]
    loop = asyncio.get_running_loop()
    cl = tunnel.TunnelClient(fp_override or ident.fingerprint, drop=drop)
    await loop.create_datagram_endpoint(lambda: cl, local_addr=("127.0.0.1", 0))
    return server, cl, ("127.0.0.1", port)


def test_roundtrip_big_and_lossy(tmp_path):
    async def go():
        server, cl, addr = await _pair(tmp_path)
        await cl.connect(addr)
        st, h, body = await cl.request("GET", "/v1/health")
        assert st == 200 and b'"name":"arxnova"' in body
        st, h, body = await cl.request("GET", "/big", {"Range": "bytes=0-524287"})
        assert st == 206 and body == BIG[:524288] and h["content-range"].startswith("bytes 0-")
        payload = os.urandom(5000)
        st, _, body = await cl.request("POST", "/echo", {"content-type": "application/octet-stream"}, payload)
        assert body == payload
        # потери 15% в обе стороны — всё равно доходит целиком
        lossy = tunnel.TunnelClient(cl.fp, drop=0.15)
        await asyncio.get_running_loop().create_datagram_endpoint(lambda: lossy, local_addr=("127.0.0.1", 0))
        await lossy.connect(addr)
        st, _, body = await lossy.request("GET", "/big", {"Range": "bytes=100000-399999"}, timeout=60)
        assert body == BIG[100000:400000]
        server.transport.close()
    asyncio.run(go())


def test_wrong_key_rejected(tmp_path):
    async def go():
        server, cl, addr = await _pair(tmp_path, fp_override="ab" * 32)
        with pytest.raises((PermissionError, TimeoutError)):
            await cl.connect(addr, timeout=3)
        server.transport.close()
    asyncio.run(go())


def test_low_rtt_throughput(tmp_path):
    """Круг почти нулевой (локальная сеть, петля): темп не застревает на ~100 пакетах/с.
    Было 3,3 с на 512 КБ (0,15 МБ/с), стало ~0,08 с."""
    import time

    async def go():
        server, cl, addr = await _pair(tmp_path)
        await cl.connect(addr)
        await cl.request("GET", "/big", {"Range": "bytes=0-524287"})  # разгон
        t = time.monotonic()
        for _ in range(4):
            st, _, body = await cl.request("GET", "/big", {"Range": "bytes=0-524287"})
            assert len(body) == 524288
        server.transport.close()
        return (time.monotonic() - t) / 4
    assert asyncio.run(go()) < 1.0


class _NarrowLink(tunnel.TunnelClient):
    """Узкое место к плееру (скорость rate пакетов/с, очередь без предела) и телефон в экономии:
    подтверждения уходят пачкой раз в 150 мс."""

    def __init__(self, fp, rate):
        super().__init__(fp)
        self.rate, self.q, self.worst = rate, [], 0.0

    def datagram_received(self, data, addr):
        if self.session is None:
            return super().datagram_received(data, addr)
        import time
        self.q.append((time.monotonic(), data, addr))

    def send_raw(self, data, addr):
        import time
        if self.session is None:
            return super().send_raw(data, addr)
        now = time.monotonic()
        asyncio.get_running_loop().call_later((int(now / 0.15) + 1) * 0.15 - now, super().send_raw, data, addr)

    async def link(self):
        import time
        credit, at = 0.0, time.monotonic()
        while True:
            now = time.monotonic()
            credit, at = min(credit + (now - at) * self.rate, 2.0), now
            while self.q and credit >= 1:
                t, d, a = self.q.pop(0)
                credit -= 1
                self.worst = max(self.worst, now - t)
                tunnel.TunnelClient.datagram_received(self, d, a)
            await asyncio.sleep(0.001)


def test_no_bloat_after_rate_drop(tmp_path):
    """Плеер набрал запас и молчал, а сеть тем временем сузилась в 15 раз (с Wi-Fi на сотовую):
    кусок не ложится в очередь узкого места целиком. Было: очередь до 4 с (окно росло вслед за
    сглаженным кругом, после паузы — залп по старой скорости), на сотовой первый кусок 8,7 с."""
    async def go():
        ident = tlsid.ensure(tmp_path, "test")
        server = await tunnel.start(stub_app(), 0, ident.cert, ident.key, ident.fingerprint)
        cl = _NarrowLink(ident.fingerprint, 3000)
        await asyncio.get_running_loop().create_datagram_endpoint(lambda: cl, local_addr=("127.0.0.1", 0))
        pump = asyncio.get_running_loop().create_task(cl.link())
        await cl.connect(("127.0.0.1", server.transport.get_extra_info("sockname")[1]))
        for _ in range(2):
            await cl.request("GET", "/big", {"Range": "bytes=0-524287"})
        await asyncio.sleep(3)
        cl.rate, cl.worst = 200, 0.0
        st, _, body = await cl.request("GET", "/big", {"Range": "bytes=0-524287"}, timeout=60)
        assert body == BIG[:524288]
        pump.cancel()
        server.transport.close()
        return cl.worst
    assert asyncio.run(go()) < 1.2  # весь кусок в очереди — 2,4 с


def test_unknown_session_is_reset(tmp_path):
    """Сервер перезапустили — сессии плеера он не знает: отвечает сбросом сразу, и плеер
    переподключается, а не ждёт таймаута каждого запроса (на M9 — больше минуты)."""
    import time

    async def go():
        server, cl, addr = await _pair(tmp_path)
        await cl.connect(addr)
        assert (await cl.request("GET", "/v1/health"))[0] == 200
        server.sessions.clear()  # как после перезапуска
        t = time.monotonic()
        with pytest.raises(ConnectionResetError):
            await cl.request("GET", "/v1/health", timeout=10)
        took = time.monotonic() - t
        assert cl.reset
        fresh = tunnel.TunnelClient(cl.fp)
        await asyncio.get_running_loop().create_datagram_endpoint(lambda: fresh, local_addr=("127.0.0.1", 0))
        await fresh.connect(addr)
        assert (await fresh.request("GET", "/v1/health"))[0] == 200
        server.transport.close()
        return took
    assert asyncio.run(go()) < 1.0


def test_server_answers_ping(tmp_path):
    """Плеер держит NAT пингом раз в 15 с; сервер отвечает — иначе плеер через 2 минуты без
    запросов считал канал мёртвым и рвал его (прогретый канал потока умирал к «Play»)."""
    async def go():
        server, cl, addr = await _pair(tmp_path)
        await cl.connect(addr)
        s = cl.session
        before = s.last_rx
        await asyncio.sleep(0.05)
        s.send_raw(s.seal(tunnel.frame(tunnel.F_PING, b"")), addr)
        await asyncio.sleep(0.3)
        server.transport.close()
        return s.last_rx > before
    assert asyncio.run(go())
