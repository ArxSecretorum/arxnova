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
