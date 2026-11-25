"""Title-bar mute button on the Assistant panel, acting on the visible session.

A checkable QAction added through ``BasePanel.add_title_bar_action`` (rendered
as a ``PanelTitleButton``). Checked = the visible tab's agent is muted. It
follows tab switches via ``ClaudePanel.current_session_changed`` and mutes
made elsewhere (the preferences page) via ``VoiceService.settings_changed``.
Permission requests and errors still speak while muted (see policy.py).
"""

from __future__ import annotations

from typing import Callable

from PySide6.QtCore import QObject, QTimer
from PySide6.QtGui import QAction

from lightfall.utils.logging import logger

from lightfall_voice.policy import _KEYS, VoiceSettings
from lightfall_voice.stt.controller import CLAUDE_PANEL_ID, _alive, _default_window_lookup

_ICON_SPEAKING = "mdi6.volume-high"
_ICON_MUTED = "mdi6.volume-off"
_RETRY_MS = 500
_MAX_RETRIES = 240  # two minutes


def save_agent_overrides(get_prefs: Callable[[], object]) -> Callable[[VoiceSettings], None]:
    """A ``save`` for SessionMuteAction that writes only ``voice.agent_overrides``.

    A mute click changes nothing else, so no other voice key is (re)written.
    ``get_prefs`` is called on each save (the preferences store may be replaced).
    """

    def save(settings: VoiceSettings) -> None:
        get_prefs().set(
            _KEYS["agent_overrides"],
            {name: {"muted": o.muted, "voice": o.voice} for name, o in settings.agent_overrides.items()},
        )

    return save


def _default_panel_lookup():
    window = _default_window_lookup()
    get_panel = getattr(window, "get_panel", None)
    return get_panel(CLAUDE_PANEL_ID) if callable(get_panel) else None


def _icon(name: str):
    try:
        import qtawesome as qta

        from lightfall.ui.theme import ThemeManager

        color = ThemeManager.get_instance().colors.text_secondary
    except Exception:
        color = "#808080"
    try:
        return qta.icon(name, color=color)
    except Exception:
        return None


class SessionMuteAction(QObject):
    def __init__(
        self,
        service,
        save: Callable[[VoiceSettings], None],
        panel_lookup: Callable[[], object] | None = None,
    ) -> None:
        super().__init__()
        self._service = service
        self._save = save
        self._panel_lookup = panel_lookup or _default_panel_lookup
        self._panel = None
        self._action: QAction | None = None
        self._retries_left = 0
        self._retry_timer = QTimer(self)
        self._retry_timer.setSingleShot(True)
        self._retry_timer.setInterval(_RETRY_MS)
        self._retry_timer.timeout.connect(self._retry_install)
        service.settings_changed.connect(self.refresh)

    @property
    def action(self) -> QAction | None:
        return self._action

    def install(self) -> None:
        """Add the button to the Assistant panel once it exists; later calls
        (every AgentBus.agents_changed, e.g. a rebuilt session) just refresh."""
        if self._panel is not None and _alive(self._panel):
            self.refresh()
            return
        try:
            panel = self._panel_lookup()
        except Exception:
            logger.exception("lightfall-voice: Assistant panel lookup failed")
            return
        if panel is None or not hasattr(panel, "add_title_bar_action"):
            # agents_changed can fire while the Assistant panel is still being built,
            # before the docking manager registers it; with no later agents_changed the
            # button would never appear. Retry briefly until the panel shows up.
            self._schedule_retry()
            return
        self._retry_timer.stop()
        action = QAction(panel)
        action.setCheckable(True)
        action.triggered.connect(self._on_triggered)
        panel.add_title_bar_action(action)
        signal = getattr(panel, "current_session_changed", None)
        if signal is not None:
            signal.connect(self.refresh)
        self._panel, self._action = panel, action
        self.refresh()

    def _schedule_retry(self) -> None:
        if self._retry_timer.isActive():
            return
        if self._retries_left <= 0:
            self._retries_left = _MAX_RETRIES
        self._retry_timer.start()

    def _retry_install(self) -> None:
        self._retries_left -= 1
        if self._retries_left <= 0:
            # Give up until the next AgentBus.agents_changed calls install() again.
            logger.debug("lightfall-voice: Assistant panel not found; mute button not installed")
            return
        self.install()

    def _current_agent_name(self) -> str | None:
        if self._panel is None or not _alive(self._panel):
            return None
        widget = self._panel.current_assistant_widget()
        return self._service.agent_name_of(widget) if widget is not None else None

    def refresh(self) -> None:
        action = self._action
        if action is None or not _alive(action):
            return
        name = self._current_agent_name()
        muted = name is not None and self._service.is_agent_muted(name)
        action.setEnabled(name is not None)
        action.setChecked(muted)
        icon = _icon(_ICON_MUTED if muted else _ICON_SPEAKING)
        if icon is not None:
            action.setIcon(icon)
        if name is None:
            tip = "Voice notifications"
        elif muted:
            tip = f"Unmute voice notifications for '{name}'"
        else:
            tip = f"Mute voice notifications for '{name}'"
        action.setToolTip(tip)
        action.setText(tip)

    def _on_triggered(self, checked: bool) -> None:
        name = self._current_agent_name()
        if name is None:
            self.refresh()
            return
        try:
            settings = self._service.set_agent_muted(name, checked)
            self._save(settings)
        except Exception:
            logger.exception("lightfall-voice: toggling mute for '{}' failed", name)
        self.refresh()
