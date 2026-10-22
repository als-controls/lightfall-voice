from lightfall_voice.events import (
    error_event,
    permission_event,
    summary_event,
    tool_event,
)
from lightfall_voice.policy import AgentOverride, VoicePolicy, VoiceSettings


def make_policy(**settings_kwargs):
    return VoicePolicy(VoiceSettings(**settings_kwargs))


def test_primary_agent_tool_event_speaks():
    p = make_policy()
    assert p.decide(tool_event("lightfall", "Read", "Reading loader", 100.0), now=100.0) == (
        "Reading loader", None,
    )


def test_master_disable_silences_everything():
    p = make_policy(enabled=False)
    assert p.decide(permission_event("lightfall", "Permission needed", 0.0), now=0.0) is None


def test_event_kind_toggles():
    p = make_policy(tools=False)
    assert p.decide(tool_event("lightfall", "Read", "Reading x", 0.0), now=0.0) is None
    assert p.decide(summary_event("lightfall", "Task complete", 0.0), now=0.0) is not None


def test_non_primary_session_speaks_by_default():
    p = make_policy()
    assert p.decide(tool_event("observer", "Read", "Reading x", 0.0), now=0.0) == (
        "Reading x", None,
    )
    assert p.decide(summary_event("observer", "Done", 0.0), now=0.0) == ("Done", None)


def test_muted_override_silences_low_priority():
    p = make_policy(agent_overrides={"observer": AgentOverride(muted=True)})
    assert p.decide(tool_event("observer", "Read", "Reading x", 0.0), now=0.0) is None
    assert p.decide(summary_event("observer", "Done", 0.0), now=0.0) is None
    assert p.decide(tool_event("lightfall", "Read", "Reading x", 0.0), now=0.0) is not None


def test_voice_only_override_is_not_a_mute():
    p = make_policy(agent_overrides={"observer": AgentOverride(voice="af_sky")})
    assert p.decide(summary_event("observer", "Done", 0.0), now=0.0) == ("Done", "af_sky")


def test_subagent_high_priority_speaks_by_default():
    p = make_policy()
    out = p.decide(error_event("alignment", "Agent alignment error. Attention needed", 0.0), now=0.0)
    assert out == ("Agent alignment error. Attention needed", None)


def test_subagent_unmuted_override_speaks_with_voice():
    p = make_policy(agent_overrides={"alignment": AgentOverride(muted=False, voice="af_sky")})
    out = p.decide(tool_event("alignment", "Read", "Reading x", 0.0), now=0.0)
    assert out == ("Reading x", "af_sky")


def test_explicitly_muted_agent_still_speaks_high_priority():
    p = make_policy(agent_overrides={"alignment": AgentOverride(muted=True)})
    assert p.decide(permission_event("alignment", "Permission needed", 0.0), now=0.0) is not None


def test_primary_agent_voice_override():
    p = make_policy(agent_overrides={"lightfall": AgentOverride(muted=False, voice="af_nicole")})
    out = p.decide(summary_event("lightfall", "Task complete", 0.0), now=0.0)
    assert out == ("Task complete", "af_nicole")


def test_ttl_drops_stale_low_priority():
    p = make_policy()
    assert p.decide(tool_event("lightfall", "Read", "Reading x", 0.0), now=11.0) is None


def test_ttl_never_drops_high_priority():
    p = make_policy()
    assert p.decide(permission_event("lightfall", "Permission needed", 0.0), now=60.0) is not None


def test_tool_rate_limit_per_agent_and_tool():
    p = make_policy()
    assert p.decide(tool_event("lightfall", "Read", "Reading a", 0.0), now=0.0) is not None
    assert p.decide(tool_event("lightfall", "Read", "Reading b", 1.0), now=1.0) is None
    assert p.decide(tool_event("lightfall", "Grep", "Searching", 1.0), now=1.0) is not None
    assert p.decide(tool_event("lightfall", "Read", "Reading c", 4.0), now=4.0) is not None


def test_settings_preferences_roundtrip():
    class FakePrefs:
        def __init__(self):
            self.data = {}
        def get(self, key, default=None):
            return self.data.get(key, default)
        def set(self, key, value):
            self.data[key] = value

    prefs = FakePrefs()
    s = VoiceSettings(
        enabled=False, tools=False, daemon_port=9999,
        agent_overrides={"alignment": AgentOverride(muted=False, voice="af_sky")},
    )
    s.to_preferences(prefs)
    loaded = VoiceSettings.from_preferences(prefs)
    assert loaded == s
