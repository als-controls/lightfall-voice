"""VoiceInputController against fakes: no NATS, no microphone."""

from __future__ import annotations

import concurrent.futures

import pytest
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QMainWindow

from lightfall_voice.stt.controller import VoiceInputController, VoiceInputSettings
from lightfall_voice.stt.mic_button import MicButton

READY = {"state": "ready", "summary": {"status": "ready", "model": "small", "device": "cpu"}}


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

    def __init__(self, sid="s1"):
        super().__init__()
        self.health = "up"
        self.last_error = ""
        self.info = dict(READY)
        self.calls: list[tuple[str, dict]] = []
        self.adopted: list[tuple[str, str]] = []
        self.chunks: list[tuple[str, int, bytes]] = []
        self.watched: list[str] = []
        self.replies: dict[str, dict] = {"utterance.start": {"status": "ok", "stream_id": sid}}

    def actions(self) -> list[str]:
        return [a for a, _ in self.calls]

    def call(self, action, payload, timeout=10.0):
        self.calls.append((action, payload))
        return _done(self.replies.get(action, {"status": "ok"}))

    def adopt_stream(self, sid, kind=None):
        self.adopted.append((sid, kind))
        return _done(None)

    def send_chunk(self, sid, seq, data, final=False):
        self.chunks.append((sid, seq, data))
        return _done(None)

    def watch_event(self, name):
        self.watched.append(name)

    # helpers
    def set_health(self, health, error=""):
        self.health, self.last_error = health, error
        self.state_changed.emit(health)

    def set_summary(self, **summary):
        self.info = {"state": "starting", "summary": summary}
        self.info_changed.emit(self.info)


class FakeManager(QObject):
    client_created = Signal(str)
    services_changed = Signal()

    def __init__(self, client=None):
        super().__init__()
        self._client = client
        self.started: list[str] = []
        self.stopped: list[str] = []

    def client(self, name):
        return self._client if name == "stt" else None

    def ensure_started(self, name):
        self.started.append(name)

    def stop_service(self, name):
        self.stopped.append(name)
        self._client = None
        self.services_changed.emit()

    def statuses(self):
        return []

    def create(self, client):
        self._client = client
        self.client_created.emit("stt")


class FakeRecorder(QObject):
    chunk = Signal(bytes)
    level = Signal(float)
    failed = Signal(str)

    def __init__(self, device_id=""):
        super().__init__()
        self.device_id = device_id
        self.recording = False
        self.starts = 0

    def start(self):
        self.recording = True
        self.starts += 1

    def stop(self):
        self.recording = False


class FakeWidget(QObject):
    def __init__(self):
        super().__init__()
        self.accessories: list = []
        self.submitted: list[str] = []
        self.flashes: list[tuple[str, int]] = []

    def add_input_accessory(self, w):
        self.accessories.append(w)

    def submit_user_prompt(self, text, *, source="voice"):
        self.submitted.append(text)
        return "sent"

    def placeholder_flash(self, text, ms=1500):
        self.flashes.append((text, ms))

    @property
    def button(self) -> MicButton:
        return self.accessories[0]

    def flashed(self) -> list[str]:
        return [t for t, _ in self.flashes]


class FakePanel:
    def __init__(self):
        self.current = None

    def current_assistant_widget(self):
        return self.current


class FakeVoice:
    def __init__(self):
        self.holds = 0
        self.releases = 0

    @property
    def held(self) -> int:
        return self.holds - self.releases

    def hold(self):
        self.holds += 1

    def release(self):
        self.releases += 1


class Rig:
    def __init__(self, qtbot, *, widgets=2, enabled=True, client=True):
        self.window = QMainWindow()
        qtbot.addWidget(self.window)
        self.widgets = [FakeWidget() for _ in range(widgets)]
        self.panel = FakePanel()
        self.panel.current = self.widgets[0] if self.widgets else None
        self.voice = FakeVoice()
        self.client = FakeClient() if client else None
        self.manager = FakeManager(self.client)
        self.recorders: list[FakeRecorder] = []
        self.ctl = VoiceInputController(
            VoiceInputSettings(enabled=enabled),
            self.voice,
            manager=self.manager,
            recorder_factory=self._make_recorder,
            panel_lookup=lambda: self.panel,
            widget_scanner=lambda: list(self.widgets),
            window_lookup=lambda: self.window,
        )
        self.ctl.start()

    def _make_recorder(self, device_id=""):
        r = FakeRecorder(device_id)
        self.recorders.append(r)
        return r

    @property
    def rec(self) -> FakeRecorder:
        return self.recorders[-1]

    def emit(self, name, **payload):
        self.client.event.emit(name, payload)


@pytest.fixture
def rig(qtbot):
    r = Rig(qtbot)
    yield r
    r.ctl.stop()


def test_button_toggle_starts_and_stops(rig):
    w = rig.widgets[0]
    assert len(w.accessories) == 1 and isinstance(w.button, MicButton)
    assert rig.ctl.state == "idle"
    assert w.button.isEnabled() and w.button.toolTip() == "Voice input (F9)"
    rig.client.calls.clear()

    w.button.click()
    assert rig.client.calls == [
        ("utterance.start", {"rate": 16000, "format": "s16le", "channels": 1})
    ]
    assert rig.client.adopted == [("s1", "audio")]
    assert rig.rec.recording
    assert rig.ctl.state == "listening"
    assert w.button.isChecked()
    assert w.flashed()[-1] == "Listening…"

    w.button.click()
    assert rig.client.calls[-1] == ("utterance.stop", {"stream_id": "s1"})
    assert not rig.rec.recording
    assert rig.ctl.state == "transcribing"
    assert w.flashed()[-1] == "Transcribing…"


def test_auto_stop_on_silence_event_then_transcript_submits_to_origin_widget(rig):
    a, b = rig.widgets
    a.button.click()
    rig.panel.current = b  # the user switched tabs while speaking
    rig.emit("utterance.listening", stream_id="s1")
    rig.emit("utterance.ended", stream_id="s1", reason="silence")
    assert not rig.rec.recording
    assert rig.ctl.state == "transcribing"
    assert "utterance.stop" not in rig.client.actions()

    rig.emit("utterance.transcript", stream_id="other", text="ignored")
    assert rig.ctl.state == "transcribing"
    rig.emit("utterance.transcript", stream_id="s1", text="move the stage")
    assert a.submitted == ["move the stage"]
    assert b.submitted == []
    assert rig.ctl.state == "idle"


def test_esc_cancels_nothing_sent(rig):
    w = rig.widgets[0]
    w.button.click()
    assert rig.ctl._esc_shortcut.isEnabled()
    rig.ctl._esc_shortcut.activated.emit()
    assert rig.client.calls[-1] == ("utterance.cancel", {"stream_id": "s1"})
    assert not rig.rec.recording
    assert rig.ctl.state == "idle"
    assert not rig.ctl._esc_shortcut.isEnabled()
    # a late transcript for the cancelled stream is dropped
    rig.emit("utterance.transcript", stream_id="s1", text="hello")
    assert w.submitted == []
    assert rig.voice.held == 0


def test_empty_transcript_flashes(rig):
    w = rig.widgets[0]
    w.button.click()
    rig.emit("utterance.ended", stream_id="s1", reason="stop")
    rig.emit("utterance.transcript", stream_id="s1", text="")
    assert w.submitted == []
    assert w.flashed()[-1] == "Didn't catch that"
    assert rig.ctl.state == "idle"


def test_no_speech_flashes_and_returns_idle(rig):
    w = rig.widgets[0]
    w.button.click()
    rig.emit("utterance.ended", stream_id="s1", reason="no_speech")
    assert not rig.rec.recording
    assert w.flashed()[-1] == "Didn't catch that"
    assert rig.ctl.state == "idle"
    assert rig.voice.held == 0


def test_error_event_flashes_message(rig):
    w = rig.widgets[0]
    w.button.click()
    rig.emit("utterance.ended", stream_id="s1", reason="stop")
    rig.emit("utterance.error", stream_id="s1", code="unknown", message="CUDA out of memory")
    assert w.flashed()[-1] == "CUDA out of memory"
    assert rig.ctl.state == "idle"
    assert rig.voice.held == 0


def test_busy_start_releases_hold(rig):
    w = rig.widgets[0]
    rig.client.replies["utterance.start"] = {
        "status": "error", "code": "busy", "message": "an utterance is already active",
    }
    w.button.click()
    assert rig.voice.holds == 1 and rig.voice.releases == 1
    assert rig.ctl.state == "idle"
    assert rig.client.adopted == []
    assert not rig.recorders or not rig.rec.recording
    assert w.flashed()[-1] == "an utterance is already active"


def test_start_future_exception_releases_hold(rig):
    w = rig.widgets[0]
    f: concurrent.futures.Future = concurrent.futures.Future()
    f.set_exception(RuntimeError("QtServiceClient is not running"))
    rig.client.call = lambda action, payload, timeout=10.0: f
    w.button.click()
    assert rig.voice.held == 0
    assert rig.ctl.state == "idle"
    assert w.flashed()[-1] == "QtServiceClient is not running"


def test_notifications_held_during_recording(rig):
    w = rig.widgets[0]
    w.button.click()
    assert rig.voice.held == 1
    w.button.click()
    assert rig.voice.held == 1  # still held while transcribing
    rig.emit("utterance.ended", stream_id="s1", reason="stop")
    rig.emit("utterance.transcript", stream_id="s1", text="hi")
    assert rig.voice.held == 0
    assert rig.voice.holds == 1


def test_loading_and_unavailable_states_from_info_and_health(rig):
    w = rig.widgets[0]
    states: list[str] = []
    rig.ctl.state_changed.connect(states.append)

    rig.client.set_summary(status="downloading", model="small", device="", error="")
    assert rig.ctl.state == "loading"
    assert not w.button.isEnabled()
    assert w.button.toolTip() == "Loading speech model…"

    rig.client.set_summary(status="error", model="small", device="", error="no CUDA device")
    assert rig.ctl.state == "unavailable"
    assert w.button.toolTip() == "no CUDA device"

    rig.client.set_summary(status="ready", model="small", device="cuda", error="")
    assert rig.ctl.state == "idle" and w.button.isEnabled()

    rig.client.set_health("starting")
    assert rig.ctl.state == "loading"
    rig.client.set_health("down", "Install lightfall-voice[stt-service]")
    assert rig.ctl.state == "unavailable"
    assert w.button.toolTip() == "Install lightfall-voice[stt-service]"
    assert not w.button.isEnabled()
    w.button.click()  # disabled: nothing happens
    rig.ctl.toggle(w)  # nor via the API
    assert "utterance.start" not in rig.client.actions()

    rig.client.set_health("up")
    assert rig.ctl.state == "idle"
    assert states == ["loading", "unavailable", "idle", "loading", "unavailable", "idle"]


def test_client_created_later_is_adopted(qtbot):
    r = Rig(qtbot, client=False)
    try:
        assert r.ctl.state == "loading"
        c = FakeClient()
        r.manager.create(c)
        assert set(c.watched) == {
            "utterance.listening", "utterance.ended", "utterance.transcript", "utterance.error",
        }
        assert c.calls[0][0] == "config.set"
        assert r.ctl.state == "idle"
        # a replacement client: the old one is disconnected
        c2 = FakeClient(sid="s2")
        r.manager.create(c2)
        c.set_health("down", "old")
        assert r.ctl.state == "idle"
        r.widgets[0].button.click()
        assert c2.adopted == [("s2", "audio")]
    finally:
        r.ctl.stop()


def test_service_reset_while_listening_returns_idle(rig):
    w = rig.widgets[0]
    w.button.click()
    n_calls = len(rig.client.calls)
    rig.client.session_reset.emit(["s1"])
    assert not rig.rec.recording
    assert rig.ctl.state == "idle"
    assert w.flashed()[-1] == "Speech service restarted"
    assert rig.voice.held == 0
    # nothing sent for the lost stream; only the config is re-pushed
    assert [a for a, _ in rig.client.calls[n_calls:]] == ["config.set"]
    rig.rec.chunk.emit(b"\0" * 3200)
    assert rig.client.chunks == []

    w.button.click()
    rig.emit("utterance.ended", stream_id="s1", reason="stop")
    rig.client.restarted.emit()
    assert rig.ctl.state == "idle"
    assert rig.voice.held == 0


def test_recorder_failure_returns_idle(rig):
    w = rig.widgets[0]
    w.button.click()
    rig.rec.failed.emit("No microphone found")
    assert rig.ctl.state == "idle"
    assert w.flashed()[-1] == "No microphone found"
    assert rig.client.calls[-1] == ("utterance.cancel", {"stream_id": "s1"})
    assert rig.voice.held == 0
    assert w.submitted == []


def test_hotkey_without_agent_tab_is_noop_and_double_press_stops(rig):
    sc = rig.ctl._hotkey_shortcut
    assert sc.key().toString() == "F9"
    rig.panel.current = None
    rig.client.calls.clear()
    sc.activated.emit()
    assert rig.client.calls == []
    assert rig.ctl.state == "idle"

    rig.panel.current = rig.widgets[1]
    sc.activated.emit()
    assert rig.ctl.state == "listening"
    sc.activated.emit()
    assert rig.ctl.state == "transcribing"
    assert rig.client.actions() == ["utterance.start", "utterance.stop"]
    rig.emit("utterance.transcript", stream_id="s1", text="scan it")
    assert rig.widgets[1].submitted == ["scan it"]


def test_only_one_recording_across_widgets(rig):
    a, b = rig.widgets
    a.button.click()
    assert not b.button.isEnabled()
    b.button.click()
    assert rig.ctl.state == "listening"
    assert rig.client.actions().count("utterance.start") == 1
    assert len(rig.recorders) == 1
    # a hotkey press on another tab stops the one recording
    rig.panel.current = b
    rig.ctl.toggle()
    assert rig.ctl.state == "transcribing"
    assert rig.client.actions().count("utterance.start") == 1


def test_widgets_attached_once(rig):
    rig.ctl.rescan()
    rig.ctl.attach_widget(rig.widgets[0])
    assert [len(w.accessories) for w in rig.widgets] == [1, 1]
    late = FakeWidget()
    rig.widgets.append(late)
    rig.ctl.rescan()
    assert len(late.accessories) == 1


def test_enable_toggle_starts_and_stops_service(qtbot):
    r = Rig(qtbot, enabled=False, client=False)
    try:
        w = r.widgets[0]
        assert r.manager.started == []
        assert r.ctl.state == "unavailable"
        assert w.button.toolTip() == "Voice input is disabled"

        r.ctl.apply_settings(VoiceInputSettings(enabled=True, model="medium"))
        assert r.manager.started == ["stt"]
        assert r.ctl.state == "loading"
        c = FakeClient()
        r.manager.create(c)
        assert c.calls[0] == (
            "config.set",
            {"model": "medium", "model_dir": "", "device": "auto", "language": "en",
             "silence_s": 1.0},
        )
        r.ctl.apply_settings(VoiceInputSettings(enabled=True, model="large-v3", hotkey="F8"))
        assert c.calls[-1][0] == "config.set" and c.calls[-1][1]["model"] == "large-v3"
        assert r.ctl._hotkey_shortcut.key().toString() == "F8"
        assert w.button.toolTip() == "Voice input (F8)"

        r.ctl.apply_settings(VoiceInputSettings(enabled=False))
        assert r.manager.stopped == ["stt"]
        assert r.ctl.state == "unavailable"
        assert w.button.toolTip() == "Voice input is disabled"
        assert not w.button.isEnabled()
    finally:
        r.ctl.stop()


def test_enabled_start_ensures_service(qtbot):
    r = Rig(qtbot, client=False)
    try:
        assert r.manager.started == ["stt"]
    finally:
        r.ctl.stop()


def test_chunks_forwarded_with_increasing_seq(rig):
    w = rig.widgets[0]
    w.button.click()
    for i in range(3):
        rig.rec.chunk.emit(bytes([i]) * 3200)
    assert [(s, q) for s, q, _ in rig.client.chunks] == [("s1", 0), ("s1", 1), ("s1", 2)]
    assert rig.client.chunks[2][2] == b"\x02" * 3200
    w.button.click()
    rig.emit("utterance.transcript", stream_id="s1", text="x")
    rig.client.replies["utterance.start"] = {"status": "ok", "stream_id": "s2"}
    w.button.click()
    rig.rec.chunk.emit(b"\0" * 3200)
    assert rig.client.chunks[-1][:2] == ("s2", 0)


def test_settings_roundtrip():
    class Prefs(dict):
        def set(self, k, v):
            self[k] = v

    prefs = Prefs()
    s = VoiceInputSettings.from_preferences(prefs)
    assert s == VoiceInputSettings()
    assert (s.enabled, s.hotkey, s.model, s.device, s.silence_s, s.language) == (
        False, "F9", "small", "auto", 1.0, "en",
    )
    s2 = VoiceInputSettings(enabled=True, model_dir="D:/m", hotkey="Ctrl+M", model="base",
                            device="cpu", input_device="dev1", silence_s=1.5, language="",
                            python="C:/py/python.exe")
    s2.save(prefs)
    assert prefs["voice.input.model_dir"] == "D:/m"
    assert VoiceInputSettings.from_preferences(prefs) == s2


def test_stop_removes_buttons(qtbot):
    r = Rig(qtbot)
    r.widgets[0].button.click()
    r.ctl.stop()
    assert r.client.calls[-1][0] == "utterance.cancel"
    assert r.voice.held == 0
    assert r.ctl._hotkey_shortcut is None


def test_start_reply_from_worker_thread_is_handled_on_main_thread(rig, qtbot):
    import threading

    pending: concurrent.futures.Future = concurrent.futures.Future()
    rig.client.call = lambda action, payload, timeout=10.0: pending
    seen: list[bool] = []
    factory = rig.ctl._recorder_factory

    def make(device_id=""):
        seen.append(threading.current_thread() is threading.main_thread())
        return factory(device_id)

    rig.ctl._recorder_factory = make
    rig.widgets[0].button.click()
    assert rig.ctl.state == "idle"  # waiting for the reply
    t = threading.Thread(target=pending.set_result, args=({"status": "ok", "stream_id": "s9"},))
    t.start()
    t.join()
    qtbot.waitUntil(lambda: rig.ctl.state == "listening", timeout=2000)
    assert seen == [True]
    assert rig.client.adopted == [("s9", "audio")]
