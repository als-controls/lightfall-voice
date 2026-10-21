"""hold()/release() queue spoken notifications while the user is dictating."""

from test_service import StubWidget, make_service


def _setup(qtbot):
    w = StubWidget()
    service, client, clock = make_service(qtbot, [w])
    return w, service, client, clock


def test_held_events_spoken_on_release(qtbot):
    w, service, client, clock = _setup(qtbot)
    service.hold()
    w.agent.permission_requested.emit("r1", "Bash", {})
    w.agent.error_occurred.emit("boom")
    client.speak.assert_not_called()
    service.release()
    assert client.speak.call_count == 2
    texts = [c.args[0] for c in client.speak.call_args_list]
    assert texts[0].startswith("Permission needed")


def test_stale_held_events_dropped(qtbot):
    w, service, client, clock = _setup(qtbot)
    service.hold()
    w.agent.tool_called.emit("Read", {"file_path": "C:/x/loader.py"})
    clock["t"] += 11.0
    service.release()
    client.speak.assert_not_called()


def test_nested_hold(qtbot):
    w, service, client, clock = _setup(qtbot)
    service.hold()
    service.hold()
    w.agent.permission_requested.emit("r1", "Bash", {})
    service.release()
    client.speak.assert_not_called()
    service.release()
    client.speak.assert_called_once()


def test_high_priority_no_ttl_kept(qtbot):
    w, service, client, clock = _setup(qtbot)
    service.hold()
    w.agent.permission_requested.emit("r1", "Bash", {})
    clock["t"] += 600.0
    service.release()
    client.speak.assert_called_once_with(
        "Permission needed to run a command", voice=None, ttl=None
    )


def test_release_without_hold_is_noop(qtbot):
    w, service, client, clock = _setup(qtbot)
    service.release()
    w.agent.permission_requested.emit("r1", "Bash", {})
    client.speak.assert_called_once()


def test_event_arriving_during_drain_is_spoken_last(qtbot):
    w, service, client, clock = _setup(qtbot)
    service.hold()
    w.agent.permission_requested.emit("r1", "Bash", {})
    w.agent.error_occurred.emit("boom")
    spoken = []

    def speak(text, voice=None, ttl=None):
        spoken.append(text)
        if len(spoken) == 1:
            w.agent.permission_requested.emit("r2", "Edit", {})

    client.speak.side_effect = speak
    service.release()
    assert len(spoken) == 3
    assert spoken[2].startswith("Permission needed")
    assert not spoken[1].startswith("Permission needed")
