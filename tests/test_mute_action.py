"""Title-bar mute button: follows the visible tab, toggles and persists mutes."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from PySide6.QtCore import QObject, Signal

from lightfall_voice.mute_action import SessionMuteAction
from lightfall_voice.policy import AgentOverride, VoiceSettings
from lightfall_voice.service import VoiceService


class FakePanel(QObject):
    current_session_changed = Signal()

    def __init__(self):
        super().__init__()
        self.actions = []
        self.current = None

    def add_title_bar_action(self, action):
        self.actions.append(action)

    def current_assistant_widget(self):
        return self.current


def _widget(name):
    return SimpleNamespace(bus_endpoint=SimpleNamespace(name=name))


def make(qtbot, settings=None):
    service = VoiceService(settings or VoiceSettings(), client=MagicMock(),
                           widget_scanner=lambda: [])
    panel = FakePanel()
    saved = []
    mute = SessionMuteAction(service, save=saved.append, panel_lookup=lambda: panel)
    return service, panel, mute, saved


def test_install_adds_one_checkable_title_button(qtbot):
    _, panel, mute, _ = make(qtbot)
    mute.install()
    mute.install()  # every agents_changed calls it again
    assert panel.actions == [mute.action]
    assert mute.action.isCheckable()


def test_install_waits_for_the_panel(qtbot):
    service = VoiceService(VoiceSettings(), client=MagicMock(), widget_scanner=lambda: [])
    holder = {"panel": None}
    mute = SessionMuteAction(service, save=lambda s: None,
                             panel_lookup=lambda: holder["panel"])
    mute.install()
    assert mute.action is None
    holder["panel"] = FakePanel()
    mute.install()
    assert holder["panel"].actions == [mute.action]


def test_install_retries_until_the_panel_is_registered(qtbot, monkeypatch):
    """agents_changed can fire before the docking manager registers the panel:
    install() keeps retrying on a timer, without another agents_changed."""
    monkeypatch.setattr("lightfall_voice.mute_action._RETRY_MS", 10)
    service = VoiceService(VoiceSettings(), client=MagicMock(), widget_scanner=lambda: [])
    holder = {"panel": None, "lookups": 0}

    def lookup():
        holder["lookups"] += 1
        return holder["panel"]

    mute = SessionMuteAction(service, save=lambda s: None, panel_lookup=lookup)
    mute.install()  # the only agents_changed: panel not registered yet
    assert mute.action is None
    qtbot.waitUntil(lambda: holder["lookups"] >= 3, timeout=2000)
    holder["panel"] = FakePanel()
    qtbot.waitUntil(lambda: mute.action is not None, timeout=2000)
    assert holder["panel"].actions == [mute.action]
    n = holder["lookups"]
    qtbot.wait(60)
    assert holder["lookups"] == n  # retries stop once installed


def test_install_retries_give_up(qtbot, monkeypatch):
    monkeypatch.setattr("lightfall_voice.mute_action._RETRY_MS", 5)
    monkeypatch.setattr("lightfall_voice.mute_action._MAX_RETRIES", 3)
    service = VoiceService(VoiceSettings(), client=MagicMock(), widget_scanner=lambda: [])
    calls = []
    mute = SessionMuteAction(service, save=lambda s: None,
                             panel_lookup=lambda: calls.append(1))
    mute.install()
    qtbot.wait(200)
    assert len(calls) == 3  # the first install plus two retries, then stop
    assert mute.action is None


def test_toggle_mutes_visible_session_and_persists(qtbot):
    service, panel, mute, saved = make(qtbot)
    panel.current = _widget("lightfall")
    mute.install()
    assert not mute.action.isChecked()

    mute.action.trigger()  # checks it
    assert service.is_agent_muted("lightfall")
    assert saved[-1].agent_overrides["lightfall"].muted is True
    assert "Unmute" in mute.action.toolTip()

    mute.action.trigger()  # unchecks: back to the default, nothing stored
    assert not service.is_agent_muted("lightfall")
    assert "lightfall" not in saved[-1].agent_overrides


def test_button_follows_tab_switches(qtbot):
    settings = VoiceSettings(agent_overrides={"observer": AgentOverride(muted=True)})
    _, panel, mute, _ = make(qtbot, settings)
    panel.current = _widget("lightfall")
    mute.install()
    assert not mute.action.isChecked()
    panel.current = _widget("observer")
    panel.current_session_changed.emit()
    assert mute.action.isChecked()


def test_button_tracks_mutes_made_in_preferences(qtbot):
    service, panel, mute, _ = make(qtbot)
    panel.current = _widget("lightfall")
    mute.install()
    service.apply_settings(
        VoiceSettings(agent_overrides={"lightfall": AgentOverride(muted=True)})
    )
    assert mute.action.isChecked()


def test_toggle_keeps_the_agents_voice(qtbot):
    settings = VoiceSettings(agent_overrides={"observer": AgentOverride(voice="af_sky")})
    service, panel, mute, saved = make(qtbot, settings)
    panel.current = _widget("observer")
    mute.install()
    mute.action.trigger()
    o = saved[-1].agent_overrides["observer"]
    assert (o.muted, o.voice) == (True, "af_sky")


def test_no_session_disables_the_button(qtbot):
    _, panel, mute, _ = make(qtbot)
    mute.install()
    assert not mute.action.isEnabled()


def test_mute_click_saves_only_the_agent_overrides(qtbot):
    from lightfall_voice.mute_action import save_agent_overrides

    class Prefs:
        def __init__(self):
            self.calls = []

        def set(self, key, value, **kwargs):
            self.calls.append((key, value, kwargs))

    prefs = Prefs()
    service = VoiceService(
        VoiceSettings(agent_overrides={"observer": AgentOverride(muted=False, voice="af_sky")}),
        client=MagicMock(), widget_scanner=lambda: [],
    )
    panel = FakePanel()
    mute = SessionMuteAction(service, save=save_agent_overrides(lambda: prefs), panel_lookup=lambda: panel)
    mute.install()
    panel.current = _widget("lightfall")
    panel.current_session_changed.emit()
    mute.action.trigger()
    assert prefs.calls == [(
        "voice.agent_overrides",
        {"observer": {"muted": False, "voice": "af_sky"}, "lightfall": {"muted": True, "voice": None}},
        {},
    )]
    assert service.is_agent_muted("lightfall")  # the service has the new overrides


def test_settings_page_mute_button_saves_only_the_agent_overrides(qtbot, voice_prefs, monkeypatch):
    from unittest.mock import patch

    from lightfall_voice import service as service_module
    from lightfall_voice import settings as settings_module

    panel = FakePanel()
    monkeypatch.setattr("lightfall_voice.mute_action._default_panel_lookup", lambda: panel)
    real_service = VoiceService(VoiceSettings(), client=MagicMock(), widget_scanner=lambda: [])
    with (
        patch.object(service_module, "VoiceService", return_value=real_service),
        patch.object(settings_module.controller_module, "VoiceInputController"),
    ):
        real_service.start = lambda: None
        settings_module.VoiceSettingsPlugin().on_loaded()
    try:
        mute = settings_module._mute_action
        panel.current = _widget("lightfall")
        mute.install()
        mute.action.trigger()
        assert voice_prefs.calls == [
            ("voice.agent_overrides", {"lightfall": {"muted": True, "voice": None}}, None)
        ]
    finally:
        service_module.set_service(None)
        settings_module.set_controller(None)
