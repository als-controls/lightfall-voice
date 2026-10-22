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
