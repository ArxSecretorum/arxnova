"""Порт занят другой программой только на 127.0.0.1: система такое разрешает, а окно попадает не в свой сервер."""

import http.server
import json
import socket
import threading

import pytest

from arxnova import cli, config


def _serve(status, body):
    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.parametrize("status,body", [(404, {"error": "нет такой страницы"}), (200, {"name": "другая"})])
def test_foreign_program_moves_server_to_next_port(tmp_path, status, body):
    srv = _serve(status, body)
    try:
        taken = srv.server_address[1]
        cfg = config.Config(data_dir=tmp_path, port=taken, remote_port=taken + 1)
        cli._avoid_foreign_port(cfg)
        assert cfg.port > taken and cfg.port != cfg.remote_port
        assert config.load(tmp_path).port == cfg.port  # окно прочитает новый порт из настроек
    finally:
        srv.shutdown()


def test_own_server_keeps_its_port(tmp_path):
    """Перезапуск: старый экземпляр ещё отвечает — это не повод менять порт."""
    srv = _serve(200, {"name": "arxnova"})
    try:
        cfg = config.Config(data_dir=tmp_path, port=srv.server_address[1])
        cli._avoid_foreign_port(cfg)
        assert cfg.port == srv.server_address[1] and not cfg.path.exists()
    finally:
        srv.shutdown()


def test_free_port_stays(tmp_path):
    port = _free_port()
    cfg = config.Config(data_dir=tmp_path, port=port)
    cli._avoid_foreign_port(cfg)
    assert cfg.port == port and not cfg.path.exists()
