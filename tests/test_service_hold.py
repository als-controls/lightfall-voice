"""hold()/release() queue spoken notifications while the user is dictating."""

from test_service import StubWidget, make_service

from lightfall_voice.tts.scheduler import SpeechItem


def _setup(qtbot):
    w = StubWidget()
    service, client, clock = make_service(qtbot, [w])
    return w, service, client, clock


def test_held_events_spoken_on_release(qtbot):
    w, service, client, clock = _setup(qtbot)
    service.hold()
    w.agent.permission_requested.emit("r1", "Bash", {})
    w.agent.error_occurred.emit("boom")
    assert client.spoken == []
    service.release()
    assert len(client.spoken) == 2
    texts = [c["text"] for c in client.spoken]
    assert texts[0].startswith("Permission needed")


def test_hold_stops_playback(qtbot):
    w, service, client, clock = _setup(qtbot)
    lock_states = []
    client.on_stop_all = lambda: lock_states.append(service._hold_lock.locked())
    service.hold()
    assert client.stop_all_calls == 1
    assert lock_states == [False]
    service.hold()
    assert client.stop_all_calls == 1
    service.release()
    service.release()
    service.hold()
    assert client.stop_all_calls == 2
    service.release()


def test_hold_requeues_interrupted_high_on_release(qtbot):
    w, service, client, clock = _setup(qtbot)
    interrupted = SpeechItem(
        text="Permission needed to run a command",
        voice="af_heart",
        speed=1.0,
        priority="high",
        deadline=None,
    )
    client.interrupted = interrupted
    service.hold()
    assert client.stop_all_calls == 1
    w.agent.error_occurred.emit("boom")
    assert client.requeued == []
    assert client.spoken == []
    service.release()
    assert client.requeued == [interrupted]
    assert [c["text"] for c in client.spoken] == ["Agent error. Attention needed"]
    assert [kind for kind, _ in client.events] == ["stop_all", "requeue", "speak"]


def test_stale_held_events_dropped(qtbot):
    w, service, client, clock = _setup(qtbot)
    service.hold()
    w.agent.tool_called.emit("Read", {"file_path": "C:/x/loader.py"})
    clock["t"] += 11.0
    service.release()
    assert client.spoken == []


def test_nested_hold(qtbot):
    w, service, client, clock = _setup(qtbot)
    service.hold()
    service.hold()
    w.agent.permission_requested.emit("r1", "Bash", {})
    service.release()
    assert client.spoken == []
    service.release()
    assert len(client.spoken) == 1


def test_high_priority_no_ttl_kept(qtbot):
    w, service, client, clock = _setup(qtbot)
    service.hold()
    w.agent.permission_requested.emit("r1", "Bash", {})
    clock["t"] += 600.0
    service.release()
    assert client.spoken == [
        {
            "text": "Permission needed to run a command",
            "voice": "af_heart",
            "ttl": None,
            "priority": "high",
        }
    ]


def test_release_without_hold_is_noop(qtbot):
    w, service, client, clock = _setup(qtbot)
    service.release()
    w.agent.permission_requested.emit("r1", "Bash", {})
    assert len(client.spoken) == 1


def test_event_arriving_during_drain_is_spoken_last(qtbot):
    w, service, client, clock = _setup(qtbot)
    service.hold()
    w.agent.permission_requested.emit("r1", "Bash", {})
    w.agent.error_occurred.emit("boom")
    spoken = []

    def speak(text, voice=None, ttl=None, priority="normal"):
        spoken.append(text)
        if len(spoken) == 1:
            w.agent.permission_requested.emit("r2", "Edit", {})

    client.on_speak = speak
    service.release()
    assert len(spoken) == 3
    assert spoken[2].startswith("Permission needed")
    assert not spoken[1].startswith("Permission needed")
