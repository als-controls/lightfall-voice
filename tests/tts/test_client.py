"""TtsClient against fakes: no NATS, no audio output device."""

from __future__ import annotations

import concurrent.futures

import pytest
from PySide6.QtCore import QObject, Signal

from lightfall_voice.tts.client import OutputSettings, TtsClient
from lightfall_voice.tts.scheduler import SpeechItem

READY = {"state": "ready", "summary": {"status": "ready", "model": "fp16", "error": ""}}


def _done(value) -> concurrent.futures.Future:
    f: concurrent.futures.Future = concurrent.futures.Future()
    f.set_result(value)
    return f


class FakeClient(QObject):
    state_changed = Signal(str)
    restarted = Signal()
    session_reset = Signal(list)
    info_changed = Signal(dict)
    event = Signal(str, dict)
    stream_chunk = Signal(str, int, object, bool)

    def __init__(self) -> None:
        super().__init__()
        self.health = "up"
        self.last_error = ""
        self.info = dict(READY)
        self.calls: list[tuple[str, dict]] = []
        self.watched: list[str] = []
        self._next_sid = 0
        self.speak_futures: list[concurrent.futures.Future] = []

    def actions(self) -> list[str]:
        return [a for a, _ in self.calls]

    def call(self, action: str, payload: dict, timeout: float = 10.0) -> concurrent.futures.Future:
        self.calls.append((action, dict(payload)))
        if action == "speak":
            if self.speak_futures:
                return self.speak_futures.pop(0)
            self._next_sid += 1
            sid = f"s{self._next_sid}"
            return _done(
                {
                    "status": "ok",
                    "stream_id": sid,
                    "rate": 24000,
                    "format": "s16le",
                    "channels": 1,
                }
            )
        return _done({"status": "ok"})

    def watch_event(self, name: str) -> None:
        self.watched.append(name)

    def set_health(self, health: str, error: str = "") -> None:
        self.health, self.last_error = health, error
        self.state_changed.emit(health)

    def set_summary(self, **summary) -> None:
        self.info = {"state": summary.get("status", "starting"), "summary": summary}
        self.info_changed.emit(self.info)


class FakePlayer(QObject):
    finished = Signal(str)
    failed = Signal(str)

    def __init__(self, device_id: str = "") -> None:
        super().__init__()
        self.device_id = device_id
        self.begun: list[tuple[str, int, int]] = []
        self.fed: list[tuple[str, bytes, bool]] = []
        self.stops = 0
        self.is_playing = False
        self.stream_id: str | None = None

    def begin(self, stream_id: str, rate: int = 24000, channels: int = 1) -> None:
        self.begun.append((stream_id, rate, channels))
        self.stream_id = stream_id
        self.is_playing = True

    def feed(self, stream_id: str, data: bytes, final: bool = False) -> None:
        if self.is_playing and stream_id == self.stream_id:
            self.fed.append((stream_id, bytes(data), final))

    def stop(self) -> None:
        self.stops += 1
        self.is_playing = False
        self.stream_id = None


class FakeManager(QObject):
    client_created = Signal(str)
    services_changed = Signal()

    def __init__(self, client: FakeClient | None = None) -> None:
        super().__init__()
        self._client = client
        self.started: list[str] = []
        self.stopped: list[str] = []

    def client(self, name: str) -> FakeClient | None:
        return self._client if name == "tts" else None

    def ensure_started(self, name: str) -> None:
        self.started.append(name)

    def stop_service(self, name: str) -> None:
        self.stopped.append(name)
        self._client = None
        self.services_changed.emit()

    def statuses(self) -> list[dict]:
        return []

    def create(self, client: FakeClient) -> None:
        self._client = client
        self.client_created.emit("tts")


class Rig:
    def __init__(self, *, enabled: bool = True, client: bool = True) -> None:
        self.now = 100.0
        self.client = FakeClient() if client else None
        self.manager = FakeManager(self.client)
        self.players: list[FakePlayer] = []
        self.settings = OutputSettings(
            enabled=enabled,
            speed=1.25,
            quality="fp16",
            model_dir="",
            output_device="dev0",
            default_voice="af_bella",
        )
        self.tts = TtsClient(
            self.settings,
            manager=self.manager,
            player_factory=self._make_player,
            clock=lambda: self.now,
        )
        self.tts.start()

    def _make_player(self, device_id: str = "") -> FakePlayer:
        p = FakePlayer(device_id)
        self.players.append(p)
        return p

    @property
    def player(self) -> FakePlayer:
        return self.players[-1]


@pytest.fixture
def rig(qtbot):
    r = Rig()
    yield r
    r.tts.stop()


def test_speak_calls_service_and_feeds_player(rig):
    assert rig.tts.state == "ready"
    assert len(rig.players) == 1
    assert rig.player.device_id == "dev0"
    rig.client.calls.clear()

    rig.tts.speak("Hello world")
    assert rig.client.calls == [
        ("speak", {"text": "Hello world", "voice": "af_bella", "speed": 1.25})
    ]
    assert rig.player.begun == [("s1", 24000, 1)]

    rig.client.stream_chunk.emit("other", 0, b"\xff\xff", False)
    rig.client.stream_chunk.emit("s1", 0, b"\x01\x02", False)
    rig.client.stream_chunk.emit("s1", 1, b"\x03\x04", True)
    assert rig.player.fed == [
        ("s1", b"\x01\x02", False),
        ("s1", b"\x03\x04", True),
    ]

    rig.player.finished.emit("s1")
    # Default voice fallback to af_heart when default_voice is empty and voice is None
    rig.tts.apply_settings(
        OutputSettings(
            enabled=True,
            speed=1.0,
            quality="fp16",
            model_dir="",
            output_device="dev0",
            default_voice="",
        )
    )
    rig.client.calls.clear()
    rig.tts.speak("Fallback voice")
    assert rig.client.calls == [
        ("speak", {"text": "Fallback voice", "voice": "af_heart", "speed": 1.0})
    ]


def test_newer_normal_cancels_older(rig):
    rig.client.calls.clear()
    rig.tts.speak("First normal")
    assert rig.player.begun == [("s1", 24000, 1)]

    rig.tts.speak("Second normal", voice="am_adam")
    assert rig.client.calls == [
        ("speak", {"text": "First normal", "voice": "af_bella", "speed": 1.25}),
        ("cancel", {"stream_id": "s1"}),
        ("speak", {"text": "Second normal", "voice": "am_adam", "speed": 1.25}),
    ]
    assert rig.player.stops >= 1
    assert rig.player.begun == [("s1", 24000, 1), ("s2", 24000, 1)]


def test_high_protected_then_pending_plays(rig):
    rig.client.calls.clear()
    rig.tts.speak("Urgent alert", priority="high")
    assert rig.player.begun == [("s1", 24000, 1)]

    # Normal items arriving during high do not interrupt; only the newest normal is kept.
    rig.tts.speak("Old normal", priority="normal", ttl=10.0)
    rig.tts.speak("Newest normal", priority="normal", ttl=10.0)
    assert rig.client.actions() == ["speak"]

    rig.player.finished.emit("s1")
    assert rig.client.calls[-1] == (
        "speak",
        {"text": "Newest normal", "voice": "af_bella", "speed": 1.25},
    )
    assert rig.player.begun == [("s1", 24000, 1), ("s2", 24000, 1)]


def test_stop_all_stops_player_and_cancels_stream(rig):
    rig.client.calls.clear()
    rig.tts.speak("High item", priority="high", ttl=30.0)
    rig.tts.speak("Queued normal", priority="normal", ttl=10.0)
    assert rig.player.is_playing

    interrupted = rig.tts.stop_all()
    assert isinstance(interrupted, SpeechItem)
    assert interrupted.text == "High item"
    assert interrupted.priority == "high"
    assert interrupted.deadline == 130.0
    assert rig.client.calls[-1] == ("cancel", {"stream_id": "s1"})
    assert not rig.player.is_playing

    # Requeue submits with original deadline
    rig.now = 120.0
    rig.tts.requeue(interrupted)
    assert rig.player.begun[-1] == ("s2", 24000, 1)


def test_late_reply_after_cancel_is_cancelled_not_played(rig):
    rig.client.calls.clear()
    pending: concurrent.futures.Future = concurrent.futures.Future()
    rig.client.speak_futures.append(pending)

    rig.tts.speak("Slow start")
    assert rig.client.actions() == ["speak"]
    assert rig.player.begun == []

    # Cancel before the speak reply arrives
    rig.tts.stop_all()
    assert rig.player.begun == []

    # Now the speak reply arrives late
    pending.set_result(
        {"status": "ok", "stream_id": "s_late", "rate": 24000, "channels": 1}
    )
    assert rig.player.begun == []
    assert rig.client.calls[-1] == ("cancel", {"stream_id": "s_late"})


def test_not_ready_drops(rig):
    rig.client.calls.clear()
    rig.client.set_summary(status="loading", model="fp16", error="")
    assert rig.tts.state == "loading"

    rig.tts.speak("Dropped while loading", priority="high")
    assert rig.client.calls == []
    assert rig.player.begun == []

    # Becoming ready later does not play the dropped item
    rig.client.set_summary(status="ready", model="fp16", error="")
    assert rig.tts.state == "ready"
    assert rig.client.calls == []
    assert rig.player.begun == []


def test_enable_disable_starts_stops_service(qtbot):
    r = Rig(enabled=False, client=False)
    try:
        assert r.manager.started == []
        assert r.tts.state == "unavailable"

        r.tts.apply_settings(
            OutputSettings(
                enabled=True,
                speed=1.0,
                quality="int8",
                model_dir="D:/models",
                output_device="dev0",
                default_voice="af_heart",
            )
        )
        assert r.manager.started == ["tts"]
        assert r.tts.state == "loading"

        c = FakeClient()
        r.manager.create(c)
        assert r.tts.state == "ready"
        assert c.calls[0] == (
            "config.set",
            {"model_dir": "D:/models", "quality": "int8", "lang": "en-us"},
        )

        r.tts.apply_settings(
            OutputSettings(
                enabled=False,
                speed=1.0,
                quality="int8",
                model_dir="D:/models",
                output_device="dev0",
                default_voice="af_heart",
            )
        )
        assert r.manager.stopped == ["tts"]
        assert r.tts.state == "unavailable"
    finally:
        r.tts.stop()


def test_config_pushed_on_connect_and_change(qtbot):
    r = Rig(enabled=True, client=False)
    try:
        c = FakeClient()
        r.manager.create(c)
        assert c.calls == [
            ("config.set", {"model_dir": "", "quality": "fp16", "lang": "en-us"})
        ]
        assert "speech.error" in c.watched

        r.tts.apply_settings(
            OutputSettings(
                enabled=True,
                speed=1.0,
                quality="int8",
                model_dir="/tmp/kokoro",
                output_device="dev1",
                default_voice="af_heart",
            )
        )
        assert c.calls[-1] == (
            "config.set",
            {"model_dir": "/tmp/kokoro", "quality": "int8", "lang": "en-us"},
        )
        # output_device changed -> player recreated
        assert len(r.players) == 2
        assert r.player.device_id == "dev1"
    finally:
        r.tts.stop()


def test_restart_clears_playback(rig):
    states: list[str] = []
    rig.tts.state_changed.connect(states.append)

    rig.tts.speak("Playing high", priority="high")
    rig.tts.speak("Queued high", priority="high")
    rig.tts.speak("Pending normal", priority="normal")
    assert rig.player.is_playing

    stops_before = rig.player.stops
    rig.client.restarted.emit()
    assert rig.player.stops > stops_before
    assert not rig.player.is_playing
    assert rig.tts.state == "loading"
    assert states[-1] == "loading"

    # Late finished signal from before restart does not start queued items
    calls_after_restart = len(rig.client.calls)
    rig.player.finished.emit("s1")
    assert len(rig.client.calls) == calls_after_restart

    # Once ready again, queue was cleared so nothing auto-starts
    rig.client.set_summary(status="ready", model="fp16", error="")
    assert rig.tts.state == "ready"
    assert len(rig.client.calls) == calls_after_restart

    # Also check session_reset
    rig.tts.speak("After recovery", priority="high")
    assert rig.player.is_playing
    rig.client.session_reset.emit(["s2"])
    assert not rig.player.is_playing
    assert rig.tts.state == "loading"


def test_start_retries_until_service_registered(qtbot):
    class LateManager(FakeManager):
        registered = False

        def ensure_started(self, name: str) -> None:
            if not self.registered:
                raise KeyError(name)
            super().ensure_started(name)

    manager = LateManager()
    tts = TtsClient(
        OutputSettings(enabled=True),
        manager=manager,
        player_factory=FakePlayer,
    )
    tts.start()
    try:
        assert manager.started == []
        manager.registered = True
        manager.services_changed.emit()
        assert manager.started == ["tts"]
        manager.services_changed.emit()
        assert manager.started == ["tts"]
    finally:
        tts.stop()


def test_speak_rpc_error_and_player_failed_advance_queue(rig):
    rig.client.calls.clear()
    rig.client.speak_futures.append(
        _done({"status": "error", "code": "bad_request", "message": "unknown voice"})
    )

    rig.tts.speak("Bad voice item", priority="high")
    rig.tts.speak("Second high item", priority="high")
    rig.tts.speak("Third high item", priority="high")

    # First item errored on speak RPC -> second item started immediately
    assert [c[1].get("text") for c in rig.client.calls if c[0] == "speak"] == [
        "Bad voice item",
        "Second high item",
    ]
    assert rig.player.begun == [("s1", 24000, 1)]

    # Player fails during second item -> treated as finished, starts third item
    rig.player.failed.emit("Audio output error: IOError")
    assert [c[1].get("text") for c in rig.client.calls if c[0] == "speak"] == [
        "Bad voice item",
        "Second high item",
        "Third high item",
    ]
    assert rig.player.begun == [("s1", 24000, 1), ("s2", 24000, 1)]
