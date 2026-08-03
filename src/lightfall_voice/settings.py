"""Voice Notifications settings page. Owns the VoiceService lifecycle:
on_loaded() (preload) starts the service; save_settings() re-applies policy."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QGroupBox,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from lightfall.plugins.settings_plugin import SettingsPlugin
from lightfall.utils.logging import logger

from lightfall_voice import service as service_module
from lightfall_voice.policy import VoiceSettings


class VoiceSettingsPlugin(SettingsPlugin):
    def __init__(self) -> None:
        super().__init__()
        self._prefs = None  # test seam; resolved lazily via _preferences()
        self._widget: QWidget | None = None

    # -- identity ------------------------------------------------------------

    @property
    def name(self) -> str:
        return "voice"

    @property
    def display_name(self) -> str:
        return "Voice Notifications"

    @property
    def description(self) -> str:
        return "Spoken notifications for Claude agent sessions via the local voice daemon"

    @property
    def category(self) -> str:
        return "plugins"

    # -- lifecycle -------------------------------------------------------------

    def _preferences(self):
        if self._prefs is None:
            from lightfall.ui.preferences.manager import PreferencesManager

            self._prefs = PreferencesManager.get_instance()
        return self._prefs

    def on_loaded(self) -> None:
        try:
            settings = VoiceSettings.from_preferences(self._preferences())
            service = service_module.VoiceService(settings)
            service.start()
            service_module.set_service(service)
        except Exception:
            logger.exception("lightfall-voice: failed to start VoiceService")

    # -- widget ---------------------------------------------------------------

    def create_widget(self, parent: QWidget | None = None) -> QWidget:
        widget = QWidget(parent)
        layout = QVBoxLayout(widget)

        self._enabled_check = QCheckBox("Enable voice notifications")

        events_box = QGroupBox("Spoken events")
        events_form = QVBoxLayout(events_box)
        self._tools_check = QCheckBox("Tool announcements")
        self._summary_check = QCheckBox("Turn summaries")
        self._permission_check = QCheckBox("Permission requests")
        self._errors_check = QCheckBox("Errors")
        for cb in (self._tools_check, self._summary_check,
                   self._permission_check, self._errors_check):
            events_form.addWidget(cb)

        daemon_box = QGroupBox("Voice daemon")
        daemon_form = QFormLayout(daemon_box)
        self._host_edit = QLineEdit()
        self._port_spin = QSpinBox()
        self._port_spin.setRange(1, 65535)
        daemon_form.addRow("Host", self._host_edit)
        daemon_form.addRow("Port", self._port_spin)

        self._test_button = QPushButton("Test voice")
        self._test_button.clicked.connect(self._on_test_clicked)

        layout.addWidget(self._enabled_check)
        layout.addWidget(events_box)
        layout.addWidget(daemon_box)
        layout.addWidget(self._test_button)
        layout.addStretch()

        # Plugin outlives the dialog: drop widget refs on destruction
        # (same pattern as lightfall-vibe's settings plugin).
        widget.destroyed.connect(self._on_widget_destroyed)
        self._widget = widget
        return widget

    def _on_widget_destroyed(self) -> None:
        self._widget = None

    def _on_test_clicked(self) -> None:
        service = service_module.get_service()
        if service is not None:
            service._client.speak("Voice notifications are working")

    # -- load/save/validate -----------------------------------------------------

    def load_settings(self) -> None:
        s = VoiceSettings.from_preferences(self._preferences())
        self._enabled_check.setChecked(s.enabled)
        self._tools_check.setChecked(s.tools)
        self._summary_check.setChecked(s.summary)
        self._permission_check.setChecked(s.permission)
        self._errors_check.setChecked(s.errors)
        self._host_edit.setText(s.daemon_host)
        self._port_spin.setValue(s.daemon_port)

    def validate(self) -> list[str]:
        errors = []
        if self._widget is not None and not self._host_edit.text().strip():
            errors.append("Voice daemon host must not be empty")
        return errors

    def save_settings(self) -> None:
        prefs = self._preferences()
        existing = VoiceSettings.from_preferences(prefs)
        s = VoiceSettings(
            enabled=self._enabled_check.isChecked(),
            tools=self._tools_check.isChecked(),
            summary=self._summary_check.isChecked(),
            permission=self._permission_check.isChecked(),
            errors=self._errors_check.isChecked(),
            daemon_host=self._host_edit.text().strip(),
            daemon_port=self._port_spin.value(),
            agent_overrides=existing.agent_overrides,
        )
        s.to_preferences(prefs)
        service = service_module.get_service()
        if service is not None:
            service.apply_settings(s)
