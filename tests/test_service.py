"""Service tests drive stub QObjects with QtClaudeAgent-shaped signals."""

from types import SimpleNamespace

from PySide6.QtCore import QObject, Signal

from lightfall_voice.policy import AgentOverride, VoiceSettings
from lightfall_voice.service import VoiceService
from lightfall_voice.tts.client import OutputSettings
from lightfall_voice.tts.scheduler import SpeechItem


class FakeTtsClient:
    def __init__(self, interrupted: SpeechItem | None = None) -> None:
        self.spoken: list[dict] = []
        self.requeued: list[SpeechItem] = []
        self.events: list[tuple[str, object]] = []
        self.stop_all_calls = 0
        self.interrupted = interrupted
        self.applied: list[OutputSettings] = []
        self.started = 0
        self.stopped = 0
        self.on_speak = None
        self.on_stop_all = None

    def start(self) -> None:
        self.started += 1

    def stop(self) -> None:
        self.stopped += 1

    def apply_settings(self, settings: OutputSettings) -> None:
        self.applied.append(settings)

    def speak(
        self,
        message: str,
        voice: str | None = None,
        ttl: float | None = None,
        priority: str = "normal",
    ) -> None:
        if self.on_speak is not None:
            self.on_speak(message, voice=voice, ttl=ttl, priority=priority)
        entry = {"text": message, "voice": voice, "ttl": ttl, "priority": priority}
        self.spoken.append(entry)
        self.events.append(("speak", entry))

    def stop_all(self) -> SpeechItem | None:
        if self.on_stop_all is not None:
            self.on_stop_all()
        self.stop_all_calls += 1
        self.events.append(("stop_all", None))
        item, self.interrupted = self.interrupted, None
        return item

    def requeue(self, item: SpeechItem) -> None:
        self.requeued.append(item)
        self.events.append(("requeue", item))


class StubAgent(QObject):
    tool_called = Signal(str, dict)
    message_received = Signal(str)
    result_received = Signal(dict)
    error_occurred = Signal(str)
    permission_requested = Signal(str, str, dict)
    task_finished = Signal(str, str, str, str, dict)


class StubWidget(QObject):
    def __init__(self, agent_name="lightfall"):
        super().__init__()
        self.agent = StubAgent()
        self.spec = SimpleNamespace(name=agent_name)


class BusEndpointStubWidget(QObject):
    """Widget shaped like the real ClaudeAssistantWidget: no .spec, only a
    bus_endpoint whose .name comes from the agent spec."""

    def __init__(self, agent_name="lightfall"):
        super().__init__()
        self.agent = StubAgent()
        self.bus_endpoint = SimpleNamespace(name=agent_name)


def make_service(qtbot, widgets, settings=None, t0=1000.0):
    clock = {"t": t0}
    client = FakeTtsClient()
    service = VoiceService(
        settings=settings or VoiceSettings(),
        client=client,
        widget_scanner=lambda: widgets,
        clock=lambda: clock["t"],
    )
    service.rescan()
    return service, client, clock


def test_tool_call_speaks_phrase(qtbot):
    w = StubWidget()
    service, client, _ = make_service(qtbot, [w])
    w.agent.tool_called.emit("Read", {"file_path": "C:/x/loader.py"})
    assert client.spoken == [
        {"text": "Reading loader", "voice": "af_heart", "ttl": 10.0, "priority": "normal"}
    ]


def test_permission_request_speaks_high_priority_phrase(qtbot):
    w = StubWidget()
    service, client, _ = make_service(qtbot, [w])
    w.agent.permission_requested.emit("req1", "Bash", {})
    assert client.spoken == [
        {
            "text": "Permission needed to run a command",
            "voice": "af_heart",
            "ttl": None,
            "priority": "high",
        }
    ]


def test_priority_passed_through(qtbot):
    w = StubWidget()
    service, client, _ = make_service(qtbot, [w])
    w.agent.tool_called.emit("Read", {"file_path": "C:/x/loader.py"})
    w.agent.result_received.emit({"text": "Finished."})
    w.agent.permission_requested.emit("req1", "Bash", {})
    w.agent.error_occurred.emit("boom")
    assert [(c["text"], c["priority"], c["ttl"]) for c in client.spoken] == [
        ("Reading loader", "normal", 10.0),
        ("Finished.", "normal", 10.0),
        ("Permission needed to run a command", "high", None),
        ("Agent error. Attention needed", "high", None),
    ]


def test_result_speaks_buffered_message(qtbot):
    w = StubWidget()
    service, client, _ = make_service(qtbot, [w])
    w.agent.message_received.emit("Fixed the flaky test.")
    w.agent.result_received.emit({"total_cost_usd": 0.01, "session_id": "s1"})
    assert client.spoken == [
        {
            "text": "Fixed the flaky test.",
            "voice": "af_heart",
            "ttl": 10.0,
            "priority": "normal",
        }
    ]


def test_result_speaks_its_own_text_without_message_received(qtbot):
    """Streaming sessions never emit message_received; the text rides on the result."""
    w = StubWidget()
    service, client, _ = make_service(qtbot, [w])
    w.agent.result_received.emit({"session_id": "s1", "text": "Pong, still here."})
    assert client.spoken == [
        {
            "text": "Pong, still here.",
            "voice": "af_heart",
            "ttl": 10.0,
            "priority": "normal",
        }
    ]


def test_result_text_wins_over_buffered_message(qtbot):
    w = StubWidget()
    service, client, _ = make_service(qtbot, [w])
    w.agent.message_received.emit("Let me check.")
    w.agent.result_received.emit({"text": "All done."})
    assert client.spoken == [
        {"text": "All done.", "voice": "af_heart", "ttl": 10.0, "priority": "normal"}
    ]
    client.spoken.clear()
    w.agent.result_received.emit({"text": ""})  # buffer was consumed too
    assert client.spoken == []


def test_mcp_tool_call_is_announced(qtbot):
    w = StubWidget()
    service, client, _ = make_service(qtbot, [w])
    w.agent.tool_called.emit("mcp__lightfall__lightfall_read_device", {})
    assert client.spoken == [
        {"text": "Reading device", "voice": "af_heart", "ttl": 10.0, "priority": "normal"}
    ]


def test_attached_agent_names(qtbot):
    a, b = StubWidget(), BusEndpointStubWidget(agent_name="observer")
    service, _, _ = make_service(qtbot, [a, b])
    assert service.attached_agent_names() == ["lightfall", "observer"]


def test_result_without_prior_message_is_silent(qtbot):
    w = StubWidget()
    service, client, _ = make_service(qtbot, [w])
    w.agent.result_received.emit({"total_cost_usd": 0.01, "session_id": "s1"})
    assert client.spoken == []


def test_result_buffer_clears_after_speaking(qtbot):
    w = StubWidget()
    service, client, _ = make_service(qtbot, [w])
    w.agent.message_received.emit("First answer.")
    w.agent.result_received.emit({"session_id": "s1"})
    assert client.spoken == [
        {"text": "First answer.", "voice": "af_heart", "ttl": 10.0, "priority": "normal"}
    ]
    client.spoken.clear()
    # Second result with no new message in between must stay silent.
    w.agent.result_received.emit({"session_id": "s1"})
    assert client.spoken == []


def test_error_speaks_attention_phrase(qtbot):
    w = StubWidget()
    service, client, _ = make_service(qtbot, [w])
    w.agent.error_occurred.emit("boom")
    assert client.spoken == [
        {
            "text": "Agent error. Attention needed",
            "voice": "af_heart",
            "ttl": None,
            "priority": "high",
        }
    ]


def test_new_session_tab_speaks_without_configuration(qtbot):
    w = StubWidget(agent_name="observer")
    service, client, _ = make_service(qtbot, [w])
    w.agent.tool_called.emit("Read", {"file_path": "C:/x/a.py"})
    w.agent.result_received.emit({"session_id": "s1", "text": "done"})
    assert [c["text"] for c in client.spoken] == ["Reading a", "done"]


def test_muted_session_is_silent_for_low_priority(qtbot):
    w = StubWidget(agent_name="alignment")
    settings = VoiceSettings(agent_overrides={"alignment": AgentOverride(muted=True)})
    service, client, _ = make_service(qtbot, [w], settings=settings)
    w.agent.tool_called.emit("Read", {"file_path": "C:/x/a.py"})
    w.agent.message_received.emit("done")
    w.agent.result_received.emit({"session_id": "s1"})
    assert client.spoken == []


def test_subagent_error_is_name_prefixed(qtbot):
    w = StubWidget(agent_name="alignment")
    service, client, _ = make_service(qtbot, [w])
    w.agent.error_occurred.emit("boom")
    assert client.spoken == [
        {
            "text": "Agent alignment error. Attention needed",
            "voice": "af_heart",
            "ttl": None,
            "priority": "high",
        }
    ]


def test_subagent_permission_is_name_prefixed(qtbot):
    w = StubWidget(agent_name="alignment")
    service, client, _ = make_service(qtbot, [w])
    w.agent.permission_requested.emit("req1", "Edit", {})
    assert client.spoken == [
        {
            "text": "Agent alignment: permission needed to edit a file",
            "voice": "af_heart",
            "ttl": None,
            "priority": "high",
        }
    ]


def test_rescan_is_idempotent(qtbot):
    w = StubWidget()
    service, client, _ = make_service(qtbot, [w])
    service.rescan()
    service.rescan()
    w.agent.tool_called.emit("Bash", {"command": "ls"})
    assert len(client.spoken) == 1  # exactly one connection despite 3 scans


def test_widget_without_agent_attr_is_skipped(qtbot):
    class Degraded(QObject):
        pass
    service, client, _ = make_service(qtbot, [Degraded()])  # must not raise


def test_start_and_stop_control_client(qtbot):
    service, client, _ = make_service(qtbot, [])
    service.start()
    assert client.started == 1
    service.stop()
    assert client.stopped == 1


def test_apply_settings_updates_policy_and_client(qtbot):
    w = StubWidget(agent_name="alignment")
    service, client, _ = make_service(qtbot, [w])
    new = VoiceSettings(
        speed=1.25,
        quality="int8",
        model_dir="D:/models",
        output_device="dev1",
        agent_overrides={"alignment": AgentOverride(muted=False, voice="af_sky")},
    )
    service.apply_settings(new)
    assert client.applied == [new.to_output()]
    w.agent.tool_called.emit("Grep", {})
    assert client.spoken == [
        {"text": "Searching", "voice": "af_sky", "ttl": 10.0, "priority": "normal"}
    ]


def test_apply_settings_preserves_tool_rate_limit_state(qtbot):
    w = StubWidget()
    service, client, clock = make_service(qtbot, [w])
    w.agent.tool_called.emit("Bash", {})
    assert len(client.spoken) == 1
    client.spoken.clear()

    # Rebuild settings (e.g. saving preferences) shortly after -- the rate
    # limit window for this tool must survive, not reset.
    service.apply_settings(VoiceSettings())
    clock["t"] += 1.0  # still inside TOOL_RATE_LIMIT_SECONDS
    w.agent.tool_called.emit("Bash", {})
    assert client.spoken == []


def test_speak_test_forwards_to_client(qtbot):
    service, client, _ = make_service(qtbot, [])
    service.speak_test()
    assert client.spoken == [
        {
            "text": "Voice notifications are working",
            "voice": "af_heart",
            "ttl": None,
            "priority": "normal",
        }
    ]


def test_silent_tools_do_not_speak(qtbot):
    w = StubWidget()
    service, client, _ = make_service(qtbot, [w])
    w.agent.tool_called.emit("TodoWrite", {})
    w.agent.tool_called.emit("Task", {})
    assert client.spoken == []


def test_handler_exceptions_do_not_propagate(qtbot):
    w = StubWidget()
    service, client, _ = make_service(qtbot, [w])

    def boom(*_args, **_kwargs):
        raise RuntimeError("client exploded")

    client.on_speak = boom
    w.agent.tool_called.emit("Bash", {})  # must not raise through emit


def test_bus_endpoint_name_resolves_agent_name(qtbot):
    """Real widgets have no .spec; the agent name must come from
    bus_endpoint.name so per-agent muting/prefixing works in production."""
    w = BusEndpointStubWidget(agent_name="alignment")
    service, client, _ = make_service(qtbot, [w])

    w.agent.tool_called.emit("Read", {"file_path": "C:/x/a.py"})
    assert client.spoken == [
        {"text": "Reading a", "voice": "af_heart", "ttl": 10.0, "priority": "normal"}
    ]
    client.spoken.clear()

    # High-priority events are name-prefixed using the resolved name.
    w.agent.error_occurred.emit("boom")
    assert client.spoken == [
        {
            "text": "Agent alignment error. Attention needed",
            "voice": "af_heart",
            "ttl": None,
            "priority": "high",
        }
    ]
