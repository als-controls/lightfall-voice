"""Shared fixtures: a fake voice daemon capturing JSON-line requests."""

from __future__ import annotations

import json
import socket
import threading

import pytest


class FakeDaemon:
    def __init__(self):
        self.requests: list[dict] = []
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(5)
        self.port = self._sock.getsockname()[1]
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self):
        self._sock.settimeout(0.1)
        while not self._stop.is_set():
            try:
                client, _ = self._sock.accept()
            except socket.timeout:
                continue
            with client:
                data = b""
                client.settimeout(1.0)
                try:
                    while not data.endswith(b"\n"):
                        chunk = client.recv(4096)
                        if not chunk:
                            break
                        data += chunk
                    if data.strip():
                        self.requests.append(json.loads(data.decode("utf-8")))
                    client.sendall(b'{"status": "ok", "action": "queued"}\n')
                except OSError:
                    pass

    def close(self):
        self._stop.set()
        self._thread.join(timeout=2)
        self._sock.close()


@pytest.fixture
def fake_daemon():
    daemon = FakeDaemon()
    yield daemon
    daemon.close()
