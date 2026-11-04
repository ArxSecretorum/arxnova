"""Ответ на отменённый запрос не должен забивать канал (04.11.2026). Плеер отменяет запрос через 10 с без движения и
куски отменённого не подтверждает. Отмена, пришедшая раньше ответа, терялась: сервер слал такой ответ без конца, окно
было занято, и новые ответы не уходили, пока сессия не закроется. На M9 звук так стоял 4 минуты."""

import asyncio
import os
import struct

from fastapi import FastAPI
from fastapi.responses import Response

from arxnova import tlsid
from arxnova import tunnel as T


def session(sent):
    return T.Session(b"12345678", b"r" * 32, b"t" * 32, ("192.0.2.1", 1), lambda d, a: sent.append(d),
                     lambda stream, msg: None, 1)


def cancel(s, stream):
    s.on_packet(T.frame(T.F_CANCEL, struct.pack("!I", stream)))


BIG = b"x" * 400_000  # больше стартового окна


def test_cancel_before_answer_is_remembered_and_new_answers_go_out():
    async def run():
        sent = []
        s = session(sent)
        cancel(s, 5)  # ответ ещё готовится
        done = s.send_message(5, BIG)
        assert done.done() and done.result() is False
        assert 5 not in s.out
        s.send_message(7, BIG)
        assert s.out[7].next_new > 0, "новый ответ не встал в очередь: окно занято отменённым"
    asyncio.run(run())


def test_cancelled_answer_is_not_sent_from_the_pacing_queue():
    async def run():
        sent = []
        s = session(sent)
        s.send_message(5, BIG)
        assert s.paced
        cancel(s, 5)
        s.pace_credit, s.btlbw = 0.0, 100000.0
        s.pace_at -= 1.0
        sent.clear()
        s.pace()
        assert not sent, "отменённое ушло в сеть"
    asyncio.run(run())


def test_answer_nobody_acknowledges_is_dropped_while_session_is_alive():
    """Обе отмены потерялись в сети: ответ всё равно не должен жить вечно."""
    async def run():
        s = session([])
        done = s.send_message(5, BIG)
        s.pump()
        assert 5 in s.out
        s.out[5].moved_at -= T.STREAM_STALL_S + 1
        s.pump()
        assert 5 not in s.out and done.result() is False
        s.send_message(7, BIG)
        assert s.out[7].next_new > 0
    asyncio.run(run())


def test_slow_answer_that_moves_is_kept():
    async def run():
        s = session([])
        s.send_message(5, BIG)
        o = s.out[5]
        o.moved_at -= T.STREAM_STALL_S + 1
        count = len(o.frags)
        s.on_packet(T.frame(T.F_ACK, struct.pack("!II", 5, count) + bytes([1]) + bytes((count + 7) // 8 - 1)))
        s.pump()
        assert 5 in s.out, "подтверждения идут: ответ живой"
    asyncio.run(run())


def test_real_udp_channel_survives_cancel_before_answer(tmp_path):
    """Настоящие пакеты: сервер занят, плеер отменяет запрос и молчит о его кусках, следующий запрос должен пройти."""
    body = os.urandom(700_000)
    app = FastAPI()

    @app.get("/slow")
    async def slow():
        await asyncio.sleep(0.5)
        return Response(body)

    @app.get("/next")
    async def next_chunk():
        return Response(body[:524288])

    async def go():
        ident = tlsid.ensure(tmp_path, "test")
        server = await T.start(app, 0, ident.cert, ident.key, ident.fingerprint)
        addr = ("127.0.0.1", server.transport.get_extra_info("sockname")[1])
        cl = T.TunnelClient(ident.fingerprint)
        await asyncio.get_running_loop().create_datagram_endpoint(lambda: cl, local_addr=("127.0.0.1", 0))
        await cl.connect(addr)
        s = cl.session
        stream = cl.next_stream
        real = s.on_packet

        def like_the_player(plain):  # NovaTunnel.onData: «брошенный — молчим»
            real(b"".join(T.frame(k, p) for k, p in T.frames(plain)
                          if not (k == T.F_FRAG and struct.unpack("!I", p[:4])[0] == stream)))

        s.on_packet = like_the_player
        waiting = asyncio.create_task(cl.request("GET", "/slow"))
        await asyncio.sleep(0.2)  # запрос дошёл, ответ готовится
        s.send_raw(s.seal(T.frame(T.F_CANCEL, struct.pack("!I", stream))), s.addr)
        await asyncio.sleep(0.8)  # ответ готов
        status, _, got = await cl.request("GET", "/next", timeout=5)
        assert status == 200 and got == body[:524288]
        waiting.cancel()
        server.transport.close()

    asyncio.run(go())
