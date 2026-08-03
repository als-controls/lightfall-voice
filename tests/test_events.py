from lightfall_voice.events import (
    VoiceEvent,
    error_event,
    permission_event,
    summary_event,
    tool_event,
)


def test_tool_event_is_low_priority():
    e = tool_event("lightfall", "Read", "Reading loader", now=100.0)
    assert e == VoiceEvent(
        agent_name="lightfall", kind="tool", text="Reading loader",
        priority="low", timestamp=100.0, tool_name="Read",
    )


def test_summary_event_is_low_priority():
    e = summary_event("lightfall", "Task complete", now=1.0)
    assert e.kind == "summary" and e.priority == "low" and e.tool_name is None


def test_permission_event_is_high_priority():
    e = permission_event("lightfall", "Permission needed to run a command", now=1.0)
    assert e.kind == "permission" and e.priority == "high"


def test_error_event_is_high_priority():
    e = error_event("alignment", "Agent alignment error. Attention needed", now=1.0)
    assert e.kind == "error" and e.priority == "high"


def test_events_are_frozen():
    import dataclasses, pytest
    e = summary_event("a", "b", now=0.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        e.text = "mutated"
