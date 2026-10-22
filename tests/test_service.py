"""Service tests drive stub QObjects with QtClaudeAgent-shaped signals."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from PySide6.QtCore import QObject, Signal

from lightfall_voice.policy import AgentOverride, VoiceSettings
from lightfall_voice.service import VoiceService


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
    client = MagicMock()
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
    client.speak.assert_called_once_with("Reading loader", voice=None, ttl=10.0)


def test_permission_request_speaks_high_priority_phrase(qtbot):
    w = StubWidget()
    service, client, _ = make_service(qtbot, [w])
    w.agent.permission_requested.emit("req1", "Bash", {})
    client.speak.assert_called_once_with(
        "Permission needed to run a command", voice=None, ttl=None
    )


def test_result_speaks_buffered_message(qtbot):
    w = StubWidget()
    service, client, _ = make_service(qtbot, [w])
    w.agent.message_received.emit("Fixed the flaky test.")
    w.agent.result_received.emit({"total_cost_usd": 0.01, "session_id": "s1"})
    client.speak.assert_called_once_with("Fixed the flaky test.", voice=None, ttl=10.0)


def test_result_speaks_its_own_text_without_message_received(qtbot):
    """Streaming sessions never emit message_received; the text rides on the result."""
    w = StubWidget()
    service, client, _ = make_service(qtbot, [w])
    w.agent.result_received.emit({"session_id": "s1", "text": "Pong, still here."})
    client.speak.assert_called_once_with("Pong, still here.", voice=None, ttl=10.0)


def test_result_text_wins_over_buffered_message(qtbot):
    w = StubWidget()
    service, client, _ = make_service(qtbot, [w])
    w.agent.message_received.emit("Let me check.")
    w.agent.result_received.emit({"text": "All done."})
    client.speak.assert_called_once_with("All done.", voice=None, ttl=10.0)
    client.speak.reset_mock()
    w.agent.result_received.emit({"text": ""})  # buffer was consumed too
    client.speak.assert_not_called()


def test_mcp_tool_call_is_announced(qtbot):
    w = StubWidget()
    service, client, _ = make_service(qtbot, [w])
    w.agent.tool_called.emit("mcp__lightfall__lightfall_read_device", {})
    client.speak.assert_called_once_with("Reading device", voice=None, ttl=10.0)


def test_attached_agent_names(qtbot):
    a, b = StubWidget(), BusEndpointStubWidget(agent_name="observer")
    service, _, _ = make_service(qtbot, [a, b])
    assert service.attached_agent_names() == ["lightfall", "observer"]


def test_result_without_prior_message_is_silent(qtbot):
    w = StubWidget()
    service, client, _ = make_service(qtbot, [w])
    w.agent.result_received.emit({"total_cost_usd": 0.01, "session_id": "s1"})
    client.speak.assert_not_called()


def test_result_buffer_clears_after_speaking(qtbot):
    w = StubWidget()
    service, client, _ = make_service(qtbot, [w])
    w.agent.message_received.emit("First answer.")
    w.agent.result_received.emit({"session_id": "s1"})
    client.speak.assert_called_once_with("First answer.", voice=None, ttl=10.0)
    client.speak.reset_mock()
    # Second result with no new message in between must stay silent.
    w.agent.result_received.emit({"session_id": "s1"})
    client.speak.assert_not_called()


def test_error_speaks_attention_phrase(qtbot):
    w = StubWidget()
    service, client, _ = make_service(qtbot, [w])
    w.agent.error_occurred.emit("boom")
    client.speak.assert_called_once_with(
        "Agent error. Attention needed", voice=None, ttl=None
    )


def test_new_session_tab_speaks_without_configuration(qtbot):
    w = StubWidget(agent_name="observer")
    service, client, _ = make_service(qtbot, [w])
    w.agent.tool_called.emit("Read", {"file_path": "C:/x/a.py"})
    w.agent.result_received.emit({"session_id": "s1", "text": "done"})
    assert [c.args[0] for c in client.speak.call_args_list] == ["Reading a", "done"]


def test_muted_session_is_silent_for_low_priority(qtbot):
    w = StubWidget(agent_name="alignment")
    settings = VoiceSettings(agent_overrides={"alignment": AgentOverride(muted=True)})
    service, client, _ = make_service(qtbot, [w], settings=settings)
    w.agent.tool_called.emit("Read", {"file_path": "C:/x/a.py"})
    w.agent.message_received.emit("done")
    w.agent.result_received.emit({"session_id": "s1"})
    client.speak.assert_not_called()


def test_subagent_error_is_name_prefixed(qtbot):
    w = StubWidget(agent_name="alignment")
    service, client, _ = make_service(qtbot, [w])
    w.agent.error_occurred.emit("boom")
    client.speak.assert_called_once_with(
        "Agent alignment error. Attention needed", voice=None, ttl=None
    )


def test_subagent_permission_is_name_prefixed(qtbot):
    w = StubWidget(agent_name="alignment")
    service, client, _ = make_service(qtbot, [w])
    w.agent.permission_requested.emit("req1", "Edit", {})
    client.speak.assert_called_once_with(
        "Agent alignment: permission needed to edit a file", voice=None, ttl=None
    )


def test_rescan_is_idempotent(qtbot):
    w = StubWidget()
    service, client, _ = make_service(qtbot, [w])
    service.rescan()
    service.rescan()
    w.agent.tool_called.emit("Bash", {"command": "ls"})
    client.speak.assert_called_once()  # exactly one connection despite 3 scans


def test_widget_without_agent_attr_is_skipped(qtbot):
    class Degraded(QObject):
        pass
    service, client, _ = make_service(qtbot, [Degraded()])  # must not raise


def test_apply_settings_updates_policy_and_client(qtbot):
    w = StubWidget(agent_name="alignment")
    service, client, _ = make_service(qtbot, [w])
    new = VoiceSettings(
        daemon_host="127.0.0.1", daemon_port=6000,
        agent_overrides={"alignment": AgentOverride(muted=False, voice="af_sky")},
    )
    service.apply_settings(new)
    client.configure.assert_called_with("127.0.0.1", 6000)
    w.agent.tool_called.emit("Grep", {})
    client.speak.assert_called_once_with("Searching", voice="af_sky", ttl=10.0)


def test_apply_settings_preserves_tool_rate_limit_state(qtbot):
    w = StubWidget()
    service, client, clock = make_service(qtbot, [w])
    w.agent.tool_called.emit("Bash", {})
    client.speak.assert_called_once()
    client.speak.reset_mock()

    # Rebuild settings (e.g. saving preferences) shortly after -- the rate
    # limit window for this tool must survive, not reset.
    service.apply_settings(VoiceSettings())
    clock["t"] += 1.0  # still inside TOOL_RATE_LIMIT_SECONDS
    w.agent.tool_called.emit("Bash", {})
    client.speak.assert_not_called()


def test_speak_test_forwards_to_client(qtbot):
    service, client, _ = make_service(qtbot, [])
    service.speak_test()
    client.speak.assert_called_once_with("Voice notifications are working")


def test_silent_tools_do_not_speak(qtbot):
    w = StubWidget()
    service, client, _ = make_service(qtbot, [w])
    w.agent.tool_called.emit("TodoWrite", {})
    w.agent.tool_called.emit("Task", {})
    client.speak.assert_not_called()


def test_handler_exceptions_do_not_propagate(qtbot):
    w = StubWidget()
    service, client, _ = make_service(qtbot, [w])
    client.speak.side_effect = RuntimeError("client exploded")
    w.agent.tool_called.emit("Bash", {})  # must not raise through emit


def test_bus_endpoint_name_resolves_agent_name(qtbot):
    """Real widgets have no .spec; the agent name must come from
    bus_endpoint.name so per-agent muting/prefixing works in production."""
    w = BusEndpointStubWidget(agent_name="alignment")
    service, client, _ = make_service(qtbot, [w])

    w.agent.tool_called.emit("Read", {"file_path": "C:/x/a.py"})
    client.speak.assert_called_once_with("Reading a", voice=None, ttl=10.0)
    client.speak.reset_mock()

    # High-priority events are name-prefixed using the resolved name.
    w.agent.error_occurred.emit("boom")
    client.speak.assert_called_once_with(
        "Agent alignment error. Attention needed", voice=None, ttl=None
    )
