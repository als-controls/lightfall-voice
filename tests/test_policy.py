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

    def remove(self, key):
        self.data.pop(key, None)


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

    loaded.save(prefs)
    assert "voice.daemon_host" not in prefs.data
    assert "voice.daemon_port" not in prefs.data

    out = _FakePrefs()
    loaded.save(out)
    assert "voice.daemon_host" not in out.data
    assert "voice.daemon_port" not in out.data


def test_from_preferences_clamps_speed_and_validates_quality():
    for bad_speed in ("not-a-number", "", object(), float("nan"), float("inf"), True):
        s = VoiceSettings.from_preferences(_FakePrefs({"voice.output.speed": bad_speed}))
        assert s.speed == 1.0

    assert VoiceSettings.from_preferences(_FakePrefs({"voice.output.speed": 0.1})).speed == 0.5
    assert VoiceSettings.from_preferences(_FakePrefs({"voice.output.speed": -2.0})).speed == 0.5
    assert VoiceSettings.from_preferences(_FakePrefs({"voice.output.speed": 3.5})).speed == 2.0
    assert VoiceSettings.from_preferences(_FakePrefs({"voice.output.speed": "9.0"})).speed == 2.0
    assert VoiceSettings.from_preferences(_FakePrefs({"voice.output.speed": "1.5"})).speed == 1.5

    for bad_quality in ("fp32", "", None, 123):
        s = VoiceSettings.from_preferences(_FakePrefs({"voice.output.quality": bad_quality}))
        assert s.quality == "fp16"

    assert VoiceSettings.from_preferences(_FakePrefs({"voice.output.quality": "int8"})).quality == "int8"
    assert VoiceSettings.from_preferences(_FakePrefs({"voice.output.quality": "fp16"})).quality == "fp16"


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
        python="C:/py/python.exe",
    )
    assert VoiceSettings(default_voice="").to_output().default_voice == "af_heart"


def test_drop_legacy_keys_tolerates_refusal_and_missing_remove():
    from lightfall_voice.policy import drop_legacy_keys

    class Refusing(_FakePrefs):
        def remove(self, key):
            raise ValueError("bound scope")

    prefs = Refusing({"voice.daemon_host": "10.0.0.1"})
    drop_legacy_keys(prefs)  # must not raise
    assert prefs.data["voice.daemon_host"] == "10.0.0.1"

    class NoRemove:
        def get(self, key, default=None):
            return default

    drop_legacy_keys(NoRemove())  # no remove(): nothing to do


def test_drop_legacy_keys_only_touches_present_keys():
    from lightfall_voice.policy import drop_legacy_keys

    removed = []

    class Recording(_FakePrefs):
        def remove(self, key):
            removed.append(key)
            super().remove(key)

    prefs = Recording({"voice.daemon_port": 6000, "voice.enabled": True})
    drop_legacy_keys(prefs)
    assert removed == ["voice.daemon_port"]
    assert prefs.data == {"voice.enabled": True}
