"""The per-session mic button shown beside Send in each agent tab's input row."""

from __future__ import annotations

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QToolButton

_PULSE_MS = 500
_LISTENING_STYLES = (
    "QToolButton { background-color: #c62828; border-radius: 4px; }",
    "QToolButton { background-color: #ef5350; border-radius: 4px; }",
)

# state -> (qtawesome icon name, enabled, checked)
_LOOK = {
    "idle": ("mdi6.microphone", True, False),
    "listening": ("mdi6.microphone", True, True),
    "transcribing": ("mdi6.dots-horizontal", False, False),
    "loading": ("mdi6.microphone-outline", False, False),
    "unavailable": ("mdi6.microphone-off", False, False),
}


def _icon(name: str):
    try:
        import qtawesome as qta

        return qta.icon(name)
    except Exception:  # noqa: BLE001 - an icon is cosmetic; never fail the button
        return None


class MicButton(QToolButton):
    """A checkable mic button; :meth:`set_state` applies one of the controller's states.

    Clicks are reported through the normal ``clicked`` signal; the controller decides
    what a click means and pushes the resulting state back, so the checked state always
    mirrors ``listening``.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setCheckable(True)
        self.setAutoRaise(True)
        self._state = ""
        self._pulse_on = False
        self._pulse = QTimer(self)
        self._pulse.setInterval(_PULSE_MS)
        self._pulse.timeout.connect(self._on_pulse)
        self.set_state("loading", "Loading speech model…")

    @property
    def state(self) -> str:
        return self._state

    def set_state(self, state: str, tooltip: str, *, enabled: bool | None = None) -> None:
        """Show ``state`` with ``tooltip``; ``enabled`` overrides the state's default."""
        icon_name, default_enabled, checked = _LOOK[state]
        if state != self._state:
            self._state = state
            icon = _icon(icon_name)
            if icon is not None:
                self.setIcon(icon)
            if state == "listening":
                self._pulse_on = False
                self._on_pulse()
                self._pulse.start()
            else:
                self._pulse.stop()
                self.setStyleSheet("")
        self.setEnabled(default_enabled if enabled is None else enabled)
        self.setChecked(checked)
        self.setToolTip(tooltip)

    def _on_pulse(self) -> None:
        self._pulse_on = not self._pulse_on
        self.setStyleSheet(_LISTENING_STYLES[int(self._pulse_on)])
