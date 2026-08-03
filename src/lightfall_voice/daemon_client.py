"""Fire-and-forget TCP client for the voice daemon's JSON line protocol.

One background worker drains a queue; every send opens a short-lived
connection. Any failure drops the message -- a missed notification is
always better than a blocked or crashing GUI.
"""

from __future__ import annotations

import json
import queue
import socket
import threading

from lightfall.utils.logging import logger

_SENTINEL = object()


class DaemonClient:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 52849,
        connect_timeout: float = 2.0,
    ) -> None:
        self._host = host
        self._port = port
        self._timeout = connect_timeout
        self._queue: queue.Queue = queue.Queue()
        self._thread = threading.Thread(
            target=self._run, name="lightfall-voice-daemon-client", daemon=True
        )
        self._thread.start()

    def configure(self, host: str, port: int) -> None:
        self._host = host
        self._port = int(port)

    def speak(self, message: str, voice: str | None = None) -> None:
        self._queue.put({"action": "speak", "args": {"message": message, "voice": voice}})

    def stop(self) -> None:
        self._queue.put(_SENTINEL)
        self._thread.join(timeout=3)

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is _SENTINEL:
                return
            try:
                self._send(item)
            except Exception as e:
                logger.debug("voice daemon unreachable, dropping message: {}", e)

    def _send(self, request: dict) -> None:
        with socket.create_connection((self._host, self._port), timeout=self._timeout) as sock:
            sock.sendall((json.dumps(request) + "\n").encode("utf-8"))
            sock.settimeout(self._timeout)
            try:
                sock.recv(4096)  # read-and-ignore the response line
            except OSError:
                pass
