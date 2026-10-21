"""End to end: the real VoiceInputController and QtServiceClient against the in-process
SttHost on a real local broker, with a fake transcriber and a synthetic recorder."""

from __future__ import annotations

import asyncio
import threading
import uuid

import numpy as np
import pytest
from PySide6.QtCore import QObject, QTimer, Signal

from lightfall_utils.ipc.services.core import NatsCore
from lightfall_utils.ipc.services.descriptor import ServiceDescriptor
from lightfall_utils.ipc.services.qt import QtServiceClient
from lightfall_voice.audio import CHUNK_SAMPLES, RATE, float32_to_pcm16
from lightfall_voice.stt.controller import VoiceInputController, VoiceInputSettings
from lightfall_voice.stt_service.service import fake_factory, make_host

TEXT = "move the sample stage"


def _speech_then_silence() -> list[bytes]:
    t = np.arange(int(1.0 * RATE)) / RATE
    tone = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    audio = np.concatenate([tone, np.zeros(int(3.0 * RATE), dtype=np.float32)])
    return [
        float32_to_pcm16(audio[i : i + CHUNK_SAMPLES])
        for i in range(0, len(audio), CHUNK_SAMPLES)
    ]


class SyntheticRecorder(QObject):
    """Recorder stand-in: emits 1 s of tone then silence, a chunk every 10 ms."""

    chunk = Signal(bytes)
    level = Signal(float)
    failed = Signal(str)

    def __init__(self, device_id: str = "") -> None:
        super().__init__()
        self._chunks = _speech_then_silence()
        self._timer = QTimer(self)
        self._timer.setInterval(10)
        self._timer.timeout.connect(self._tick)
        self.stopped = False

    def start(self) -> None:
        self._timer.start()

    def stop(self) -> None:
        self.stopped = True
        self._timer.stop()

    def _tick(self) -> None:
        if not self._chunks:
            self._timer.stop()
            return
        self.chunk.emit(self._chunks.pop(0))


class OriginWidget(QObject):
    def __init__(self) -> None:
        super().__init__()
        self.submitted: list[str] = []
        self.flashes: list[str] = []

    def window(self):
        return None

    def add_input_accessory(self, w) -> None:
        pass

    def submit_user_prompt(self, text, *, source="voice"):
        self.submitted.append(text)
        return "sent"

    def placeholder_flash(self, text, ms=1500) -> None:
        self.flashes.append(text)


class Voice:
    def __init__(self) -> None:
        self.held = 0

    def hold(self) -> None:
        self.held += 1

    def release(self) -> None:
        self.held -= 1


class Manager(QObject):
    client_created = Signal(str)
    services_changed = Signal()

    def __init__(self, client) -> None:
        super().__init__()
        self._client = client

    def client(self, name):
        return self._client if name == "stt" else None

    def ensure_started(self, name) -> None:
        pass

    def statuses(self):
        return []


class HostThread:
    """The SttHost on its own asyncio loop thread."""

    def __init__(self, broker_url: str, scope: str) -> None:
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()
        self.host = make_host(
            scope,
            transcriber_factory=fake_factory(TEXT),
            initial_load_delay_s=0.0,
            heartbeat_s=0.1,
            single_instance=False,
        )
        self.core = NatsCore(broker_url)
        self._run(self._start())

    def _run(self, coro, timeout=15):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout)

    async def _start(self) -> None:
        await self.core.connect()
        await self.host.start(self.core)

    async def _stop(self) -> None:
        await self.host.stop()
        await self.core.close()

    def stop(self) -> None:
        try:
            self._run(self._stop())
        finally:
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.thread.join(5)


@pytest.fixture
def stack(qtbot, broker_url):
    scope = f"lfsttint{uuid.uuid4().hex[:8]}"
    host = HostThread(broker_url, scope)
    desc = ServiceDescriptor(name="stt", scope=scope, contract="1.0", launch="external")
    client = QtServiceClient(
        desc, broker_url, scope=scope, app_name="itest", app_version="1"
    )
    events: list[tuple[str, dict]] = []
    client.event.connect(lambda name, payload: events.append((name, payload)))
    voice = Voice()
    recorders: list[SyntheticRecorder] = []

    def make_recorder(device_id=""):
        r = SyntheticRecorder(device_id)
        recorders.append(r)
        return r

    widget = OriginWidget()
    ctl = VoiceInputController(
        VoiceInputSettings(enabled=True),
        voice,
        manager=Manager(client),
        recorder_factory=make_recorder,
        panel_lookup=lambda: None,
        widget_scanner=lambda: [],
        window_lookup=lambda: None,
    )
    client.start()
    ctl.start()
    try:
        yield ctl, client, widget, voice, events, recorders
    finally:
        ctl.stop()
        client.stop()
        host.stop()


def test_silence_auto_stop_submits_transcript_to_origin_widget(qtbot, stack):
    ctl, client, widget, voice, events, recorders = stack
    qtbot.waitUntil(lambda: ctl.state == "idle", timeout=15000)  # service up, model ready

    ctl.toggle(widget)
    qtbot.waitUntil(lambda: ctl.state == "listening", timeout=5000)
    assert voice.held == 1

    qtbot.waitUntil(lambda: bool(widget.submitted), timeout=15000)
    assert widget.submitted == [TEXT]
    ended = [p for n, p in events if n == "utterance.ended"]
    assert [p["reason"] for p in ended] == ["silence"]  # the service stopped it, not us
    assert recorders[-1].stopped
    assert ctl.state == "idle"
    assert voice.held == 0
