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
        "Reading loader", "af_heart",
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
        "Reading x", "af_heart",
    )
    assert p.decide(summary_event("observer", "Done", 0.0), now=0.0) == ("Done", "af_heart")


def test_muted_override_silences_low_priority():
    p = make_policy(agent_overrides={"observer": AgentOverride(muted=True)})
    assert p.decide(tool_event("observer", "Read", "Reading x", 0.0), now=0.0) is None
    assert p.decide(summary_event("observer", "Done", 0.0), now=0.0) is None
    assert p.decide(tool_event("lightfall", "Read", "Reading x", 0.0), now=0.0) is not None


def test_default_voice_is_the_ascribe_link_voice():
    assert VoiceSettings().default_voice == "af_heart"


def test_default_voice_applies_to_agents_without_their_own():
    p = make_policy(default_voice="bf_emma",
                    agent_overrides={"observer": AgentOverride(voice="am_adam")})
    assert p.decide(summary_event("lightfall", "Done", 0.0), now=0.0) == ("Done", "bf_emma")
    assert p.decide(summary_event("observer", "Done", 0.0), now=0.0) == ("Done", "am_adam")


def test_empty_default_voice_uses_af_heart():
    p = make_policy(default_voice="")
    assert p.decide(summary_event("lightfall", "Done", 0.0), now=0.0) == ("Done", "af_heart")


def test_with_agent_voice_keeps_mute_and_drops_redundant_entries():
    from lightfall_voice.policy import with_agent_voice

    o = with_agent_voice({"x": AgentOverride(muted=True)}, "x", "am_adam")
    assert (o["x"].muted, o["x"].voice) == (True, "am_adam")
    assert with_agent_voice({"y": AgentOverride(voice="am_adam")}, "y", "") == {}


def test_voice_only_override_is_not_a_mute():
    p = make_policy(agent_overrides={"observer": AgentOverride(voice="af_sky")})
    assert p.decide(summary_event("observer", "Done", 0.0), now=0.0) == ("Done", "af_sky")


def test_subagent_high_priority_speaks_by_default():
    p = make_policy()
    out = p.decide(error_event("alignment", "Agent alignment error. Attention needed", 0.0), now=0.0)
    assert out == ("Agent alignment error. Attention needed", "af_heart")


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


class _FakePrefs:
    def __init__(self, initial=None):
        self.data = dict(initial or {})

    def get(self, key, default=None):
        return self.data.get(key, default)

    def set(self, key, value):
        self.data[key] = value


def test_old_daemon_prefs_ignored():
    prefs = _FakePrefs(
        {
            "voice.daemon_host": "10.0.0.1",
            "voice.daemon_port": 9999,
            "voice.enabled": False,
        }
    )
    loaded = VoiceSettings.from_preferences(prefs)
    assert loaded.enabled is False
    assert not hasattr(loaded, "daemon_host")
    assert not hasattr(loaded, "daemon_port")

    out = _FakePrefs()
    loaded.save(out)
    assert "voice.daemon_host" not in out.data
    assert "voice.daemon_port" not in out.data


def test_settings_round_trip_output_keys():
    from lightfall_voice.tts.client import OutputSettings

    prefs = _FakePrefs()
    s = VoiceSettings(
        enabled=False,
        tools=False,
        default_voice="bm_george",
        agent_overrides={"alignment": AgentOverride(muted=False, voice="af_sky")},
        speed=1.25,
        quality="int8",
        model_dir="D:/models/kokoro",
        output_device="speaker-1",
        python="C:/py/python.exe",
    )
    s.save(prefs)
    assert prefs.data["voice.output.speed"] == 1.25
    assert prefs.data["voice.output.quality"] == "int8"
    assert prefs.data["voice.output.model_dir"] == "D:/models/kokoro"
    assert prefs.data["voice.output.output_device"] == "speaker-1"
    assert prefs.data["voice.output.python"] == "C:/py/python.exe"

    loaded = VoiceSettings.from_preferences(prefs)
    assert loaded == s
    assert loaded.to_output() == OutputSettings(
        enabled=False,
        speed=1.25,
        quality="int8",
        model_dir="D:/models/kokoro",
        output_device="speaker-1",
        default_voice="bm_george",
    )
    assert VoiceSettings(default_voice="").to_output().default_voice == "af_heart"
