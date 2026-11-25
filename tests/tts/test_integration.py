"""End to end: the real TtsClient, QtServiceClient and Player against the in-process
TtsHost on a real local broker, with FakeSynth and a fake audio sink."""

from __future__ import annotations

import asyncio
from pathlib import Path
import threading
import uuid

import pytest
from PySide6.QtCore import QBuffer, QIODevice, QObject, QTimer, Signal
from PySide6.QtMultimedia import QAudio, QAudioFormat

from lightfall_utils.ipc.services.core import NatsCore
from lightfall_utils.ipc.services.descriptor import ServiceDescriptor
from lightfall_utils.ipc.services.qt import QtServiceClient
from lightfall_voice.tts.client import OutputSettings, TtsClient
from lightfall_voice.tts.player import TICK_MS, Player
from lightfall_voice.tts_service.service import make_host
from lightfall_voice.tts_service.synth import FakeSynth

FIRST_TEXT = " ".join(f"First stream sentence {i} carries five words." for i in range(8))
SECOND_TEXT = " ".join(f"Second stream sentence {i} carries five words." for i in range(8))
HIGH_TEXT = " ".join(f"Urgent alert sentence {i} carries five words." for i in range(8))


def _fake_resolve(quality: str, model_dir: str = "", **_kw) -> tuple[Path, Path]:
    return Path(f"fake-{quality}.onnx"), Path("fake-voices.bin")


class FakeSink(QObject):
    stateChanged = Signal(object)

    def __init__(self, fmt: QAudioFormat, bytes_free: int = 9600) -> None:
        super().__init__()
        self._fmt = fmt
        self.buffer = QBuffer()
        self.buffer.open(QIODevice.OpenModeFlag.WriteOnly)
        self.bytes_free = bytes_free
        self._error = QAudio.Error.NoError
        self.stopped = False
        self._timer = QTimer(self)
        self._timer.setInterval(TICK_MS)
        self._timer.timeout.connect(self._tick)

    def format(self) -> QAudioFormat:
        return self._fmt

    def start(self) -> QBuffer:
        self._timer.start()
        return self.buffer

    def stop(self) -> None:
        self.stopped = True
        self._timer.stop()

    def bytesFree(self) -> int:
        return self.bytes_free

    def error(self):
        return self._error

    def written(self) -> bytes:
        return bytes(self.buffer.data())

    def _tick(self) -> None:
        if not self.stopped:
            self.stateChanged.emit(QAudio.State.IdleState)


class RecordingPlayer(Player):
    def __init__(self, device_id: str = "", *, sink_factory=None) -> None:
        super().__init__(device_id, sink_factory=sink_factory)
        self.begun: list[str] = []
        self.fed: list[tuple[str, bytes, bool]] = []

    def begin(self, stream_id: str, rate: int = 24000, channels: int = 1) -> None:
        self.begun.append(stream_id)
        super().begin(stream_id, rate, channels)

    def feed(self, stream_id: str, data: bytes, final: bool = False) -> None:
        self.fed.append((stream_id, bytes(data), bool(final)))
        super().feed(stream_id, data, final)


class Manager(QObject):
    client_created = Signal(str)
    services_changed = Signal()

    def __init__(self, client) -> None:
        super().__init__()
        self._client = client

    def client(self, name: str):
        return self._client if name == "tts" else None

    def ensure_started(self, name: str) -> None:
        pass

    def statuses(self) -> list[dict]:
        return []


class HostThread:
    """The TtsHost on its own asyncio loop thread."""

    def __init__(self, broker_url: str, scope: str) -> None:
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()
        self.host = make_host(
            scope,
            synth_factory=FakeSynth,
            resolve=_fake_resolve,
            initial_load_delay_s=0.0,
            heartbeat_s=0.1,
            single_instance=False,
        )
        self.core = NatsCore(broker_url)
        self._run(self._start())

    def _run(self, coro, timeout: float = 15):
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
    scope = f"lfttsint{uuid.uuid4().hex[:8]}"
    host = HostThread(broker_url, scope)
    desc = ServiceDescriptor(name="tts", scope=scope, contract="1.0", launch="external")
    client = QtServiceClient(
        desc, broker_url, scope=scope, app_name="itest", app_version="1"
    )
    chunks: dict[str, list[tuple[int, bytes, bool]]] = {}
    client.stream_chunk.connect(
        lambda sid, seq, data, final: chunks.setdefault(sid, []).append(
            (int(seq), bytes(data), bool(final))
        )
    )
    sinks: list[FakeSink] = []
    players: list[RecordingPlayer] = []
    finished: list[str] = []

    def make_sink(fmt: QAudioFormat, _dev) -> FakeSink:
        sink = FakeSink(fmt)
        sinks.append(sink)
        return sink

    def make_player(device_id: str = "") -> RecordingPlayer:
        player = RecordingPlayer(device_id, sink_factory=make_sink)
        player.finished.connect(finished.append)
        players.append(player)
        return player

    tts = TtsClient(
        OutputSettings(enabled=True),
        manager=Manager(client),
        player_factory=make_player,
    )
    client.start()
    tts.start()
    try:
        yield tts, client, players[-1], sinks, finished, chunks
    finally:
        tts.stop()
        client.stop()
        host.stop()


def test_one_message_plays_to_finished(qtbot, stack):
    tts, _client, player, sinks, finished, chunks = stack
    qtbot.waitUntil(lambda: tts.state == "ready", timeout=15000)

    tts.speak(FIRST_TEXT)
    qtbot.waitUntil(lambda: len(finished) == 1, timeout=15000)

    assert len(player.begun) == 1
    sid = player.begun[0]
    assert finished == [sid]
    assert not player.is_playing
    assert any(final for _seq, _data, final in chunks[sid])
    expected_pcm = b"".join(data for _sid, data, final in player.fed if _sid == sid and not final)
    assert len(expected_pcm) > 0
    assert sinks[-1].written() == expected_pcm


def test_second_normal_message_cancels_first(qtbot, stack):
    tts, _client, player, _sinks, finished, chunks = stack
    qtbot.waitUntil(lambda: tts.state == "ready", timeout=15000)

    tts.speak(FIRST_TEXT)
    qtbot.waitUntil(lambda: player.is_playing and len(player.fed) >= 1, timeout=5000)
    sid1 = player.begun[0]
    assert finished == []

    fed_before = len(player.fed)
    tts.speak(SECOND_TEXT)

    qtbot.waitUntil(
        lambda: (
            len(player.begun) == 2
            and player.begun[1] in finished
            and any(final for _seq, _data, final in chunks.get(sid1, []))
        ),
        timeout=15000,
    )
    sid2 = player.begun[1]
    assert sid1 != sid2
    assert chunks[sid1][-1][2] is True
    fed_after = player.fed[fed_before:]
    assert fed_after
    assert {sid for sid, _data, _final in fed_after} == {sid2}
    assert finished == [sid2]


def test_high_message_not_interrupted_by_normal(qtbot, stack):
    tts, _client, player, sinks, finished, _chunks = stack
    qtbot.waitUntil(lambda: tts.state == "ready", timeout=15000)

    tts.speak(HIGH_TEXT, priority="high")
    qtbot.waitUntil(lambda: player.is_playing and len(player.fed) >= 1, timeout=5000)
    sid_high = player.begun[0]
    assert finished == []

    tts.speak(SECOND_TEXT, priority="normal")
    assert player.begun == [sid_high]
    assert player.is_playing

    qtbot.waitUntil(lambda: sid_high in finished, timeout=15000)
    high_pcm = b"".join(data for sid, data, final in player.fed if sid == sid_high and not final)
    assert sinks[0].written() == high_pcm

    qtbot.waitUntil(lambda: len(finished) == 2, timeout=15000)
    sid_normal = player.begun[1]
    assert finished == [sid_high, sid_normal]
