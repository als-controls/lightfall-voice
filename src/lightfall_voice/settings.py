"""Voice settings page. Owns the VoiceService and VoiceInputController lifecycles:
on_loaded() (preload) starts them; save_settings() re-applies policy and voice input."""

from __future__ import annotations

from PySide6.QtCore import QCoreApplication
from PySide6.QtGui import QKeySequence
from PySide6.QtMultimedia import QMediaDevices
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QKeySequenceEdit,
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
from lightfall_voice.stt import controller as controller_module
from lightfall_voice.stt.controller import VoiceInputController, VoiceInputSettings

MODELS = ("base", "small", "medium", "large-v3", "distil-large-v3")
DEVICES = (("Auto (GPU if available)", "auto"), ("GPU", "gpu"), ("CPU", "cpu"))
LANGUAGES = (
    ("Auto-detect", ""),
    ("English", "en"),
    ("Spanish", "es"),
    ("French", "fr"),
    ("German", "de"),
    ("Italian", "it"),
    ("Portuguese", "pt"),
    ("Japanese", "ja"),
    ("Chinese", "zh"),
    ("Korean", "ko"),
)

_controller: VoiceInputController | None = None


def get_controller() -> VoiceInputController | None:
    return _controller


def set_controller(controller: VoiceInputController | None) -> None:
    global _controller
    _controller = controller


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
            return
        self._start_voice_input(service)

    def _start_voice_input(self, service) -> None:
        """Create and start the voice input controller (after VoiceService).

        Timing: preload plugins run before login, the main window and the service plugins.
        ``start()`` therefore only connects signals; the hotkey and the mic buttons attach
        on the first AgentBus ``agents_changed`` (the main window exists by then), and
        ``ensure_started("stt")`` is retried on ``services_changed`` until the ``stt``
        descriptor is registered.  ``stop()`` runs on ``aboutToQuit``.
        """
        try:
            controller = controller_module.VoiceInputController(
                VoiceInputSettings.from_preferences(self._preferences()), service
            )
            controller.start()
            set_controller(controller)
            app = QCoreApplication.instance()
            if app is not None:
                app.aboutToQuit.connect(controller.stop)
        except Exception:
            logger.exception("lightfall-voice: failed to start voice input")

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
        layout.addWidget(self._build_input_box())
        layout.addStretch()

        # Plugin outlives the dialog: drop widget refs on destruction
        # (same pattern as lightfall-vibe's settings plugin).
        widget.destroyed.connect(self._on_widget_destroyed)
        self._widget = widget
        return widget

    def _build_input_box(self) -> QGroupBox:
        box = QGroupBox("Voice input")
        form = QFormLayout(box)

        self._input_enabled_check = QCheckBox("Enable voice input")
        self._model_dir_edit = QLineEdit()
        self._model_dir_edit.setPlaceholderText("Empty: the Hugging Face cache")
        self._python_edit = QLineEdit()
        self._python_edit.setPlaceholderText("Empty: Lightfall's own interpreter")
        self._hotkey_edit = QKeySequenceEdit()
        self._model_combo = QComboBox()
        for model in MODELS:
            self._model_combo.addItem(model, model)
        self._device_combo = QComboBox()
        for label, value in DEVICES:
            self._device_combo.addItem(label, value)
        self._language_combo = QComboBox()
        for label, value in LANGUAGES:
            self._language_combo.addItem(label, value)
        self._input_device_combo = QComboBox()
        self._input_device_combo.addItem("System default", "")
        for device in QMediaDevices.audioInputs():
            self._input_device_combo.addItem(device.description(), bytes(device.id()).decode())
        self._silence_spin = QDoubleSpinBox()
        self._silence_spin.setRange(0.3, 5.0)
        self._silence_spin.setSingleStep(0.1)
        self._silence_spin.setDecimals(1)
        self._silence_spin.setSuffix(" s")

        form.addRow(self._input_enabled_check)
        form.addRow("Model directory", self._with_browse(self._model_dir_edit, directory=True))
        form.addRow("Python interpreter", self._with_browse(self._python_edit, directory=False))
        form.addRow("Hotkey", self._hotkey_edit)
        form.addRow("Model", self._model_combo)
        form.addRow("Device", self._device_combo)
        form.addRow("Language", self._language_combo)
        form.addRow("Input device", self._input_device_combo)
        form.addRow("Silence", self._silence_spin)
        return box

    @staticmethod
    def _with_browse(edit: QLineEdit, *, directory: bool) -> QWidget:
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        button = QPushButton("Browse…")

        def browse() -> None:
            if directory:
                chosen = QFileDialog.getExistingDirectory(row, "Model directory", edit.text())
            else:
                chosen, _ = QFileDialog.getOpenFileName(row, "Python interpreter", edit.text())
            if chosen:
                edit.setText(chosen)

        button.clicked.connect(browse)
        h.addWidget(edit, 1)
        h.addWidget(button)
        return row

    @staticmethod
    def _select(combo: QComboBox, value: str) -> None:
        """Select the entry carrying ``value``; keep an unlisted saved value as an entry."""
        index = combo.findData(value)
        if index < 0:
            combo.addItem(value, value)  # e.g. an unplugged microphone's id
            index = combo.count() - 1
        combo.setCurrentIndex(index)

    def _on_widget_destroyed(self) -> None:
        self._widget = None

    def _on_test_clicked(self) -> None:
        service = service_module.get_service()
        if service is not None:
            service.speak_test()

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

        i = VoiceInputSettings.from_preferences(self._preferences())
        self._input_enabled_check.setChecked(i.enabled)
        self._model_dir_edit.setText(i.model_dir)
        self._python_edit.setText(i.python)
        self._hotkey_edit.setKeySequence(QKeySequence(i.hotkey))
        self._select(self._model_combo, i.model)
        self._select(self._device_combo, i.device)
        self._select(self._language_combo, i.language)
        self._select(self._input_device_combo, i.input_device)
        self._silence_spin.setValue(i.silence_s)

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

        old_input = VoiceInputSettings.from_preferences(prefs)
        hotkey = self._hotkey_edit.keySequence().toString(QKeySequence.SequenceFormat.PortableText)
        new_input = VoiceInputSettings(
            enabled=self._input_enabled_check.isChecked(),
            model_dir=self._model_dir_edit.text().strip(),
            hotkey=hotkey or old_input.hotkey,
            model=self._model_combo.currentData(),
            device=self._device_combo.currentData(),
            input_device=self._input_device_combo.currentData() or "",
            silence_s=round(self._silence_spin.value(), 1),
            language=self._language_combo.currentData() or "",
            python=self._python_edit.text().strip(),
        )
        new_input.save(prefs)
        controller = get_controller()
        if controller is not None:
            controller.apply_settings(new_input)
