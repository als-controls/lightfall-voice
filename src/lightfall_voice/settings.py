"""Voice Notifications settings page (a scope-aware form page).

Owns the VoiceService and VoiceInputController lifecycles: on_loaded() (preload)
starts them; after_save() re-applies the saved notification and voice input
settings to them. This machine's voice output and voice input keys (device,
microphone, speaker, model folders, interpreters) belong to this console; every
other ``voice.*`` key is a per-user setting.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from PySide6.QtCore import QCoreApplication
from PySide6.QtMultimedia import QMediaDevices

from lightfall.plugins.settings_plugin import SettingsPlugin
from lightfall.ui.preferences.form.schema import (
    Action,
    Choice,
    Custom,
    Number,
    Path,
    Section,
    SettingField,
    Toggle,
)
from lightfall.ui.preferences.manager import PreferencesManager
from lightfall.utils.logging import logger

from lightfall_voice import service as service_module
from lightfall_voice.mute_action import SessionMuteAction, save_agent_overrides
from lightfall_voice.policy import VoiceSettings, drop_legacy_keys
from lightfall_voice.settings_editors import agent_voices_editor, hotkey_editor, voice_choice_editor
from lightfall_voice.stt import controller as controller_module
from lightfall_voice.stt.controller import VoiceInputController, VoiceInputSettings

if TYPE_CHECKING:
    from lightfall_settings import SettingSpec

    from lightfall.ui.preferences.form.schema import EditorSpec, FormValues

QUALITIES = (("Higher (fp16)", "fp16"), ("Standard (int8)", "int8"))
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

AGENTS_NOTE = (
    "Checked agents are muted; permission requests and errors always speak. "
    "Pick a voice per agent, or leave it on the default voice."
)

# This console's keys: its own voice output and voice input hardware and paths.
CONSOLE_KEYS = (
    "voice.output.model_dir",
    "voice.output.output_device",
    "voice.output.python",
    "voice.input.model_dir",
    "voice.input.python",
    "voice.input.input_device",
    "voice.input.device",
)

# Per-user keys: (key, type, label).
_USER_SPECS = (
    ("voice.enabled", bool, "Enable voice notifications"),
    ("voice.events.tools", bool, "Tool announcements"),
    ("voice.events.summary", bool, "Turn summaries"),
    ("voice.events.permission", bool, "Permission requests"),
    ("voice.events.errors", bool, "Errors"),
    ("voice.default_voice", str, "Default voice"),
    ("voice.agent_overrides", dict, "Agents"),
    ("voice.output.speed", float, "Speed"),
    ("voice.output.quality", str, "Quality"),
    ("voice.input.enabled", bool, "Enable voice input"),
    ("voice.input.hotkey", str, "Hotkey"),
    ("voice.input.model", str, "Model"),
    ("voice.input.language", str, "Language"),
    ("voice.input.silence_s", float, "Silence"),
)
_LABELS = {key: label for key, _type, label in _USER_SPECS}

_controller: VoiceInputController | None = None
_mute_action: SessionMuteAction | None = None


def get_controller() -> VoiceInputController | None:
    return _controller


def set_controller(controller: VoiceInputController | None) -> None:
    global _controller
    _controller = controller


def _output_devices() -> list[tuple[str, str]]:
    """"System default" ("") then this machine's audio outputs, listed when the page is shown."""
    options = [("", "System default")]
    for device in QMediaDevices.audioOutputs():
        options.append((bytes(device.id()).decode(), device.description()))
    return options


def _input_devices() -> list[tuple[str, str]]:
    """"System default" ("") then this machine's microphones, listed when the page is shown."""
    options = [("", "System default")]
    for device in QMediaDevices.audioInputs():
        options.append((bytes(device.id()).decode(), device.description()))
    return options


def _test_voice() -> str:
    """Speak the test phrase through the running service (runs on a worker thread)."""
    service = service_module.get_service()
    if service is None:
        return "Voice service not running"
    state = service.tts_state
    if isinstance(state, str) and state != "ready":
        error = service.tts_error
        if isinstance(error, str) and error:
            return f"Voice service {state}: {error}"
        return f"Voice service {state}"
    service.speak_test()
    return "Spoken"


def _field(key: str, editor: EditorSpec, *, label: str = "", **kwargs: Any) -> SettingField:
    return SettingField(key, editor, label=label or _LABELS.get(key, ""), **kwargs)


class VoiceSettingsPlugin(SettingsPlugin):
    # -- identity ------------------------------------------------------------

    @property
    def name(self) -> str:
        return "voice"

    @property
    def display_name(self) -> str:
        return "Voice Notifications"

    @property
    def description(self) -> str:
        return "Spoken notifications for Claude agent sessions via the local tts service"

    @property
    def category(self) -> str:
        return "plugins"

    # -- lifecycle -------------------------------------------------------------

    @staticmethod
    def _preferences():
        return PreferencesManager.get_instance()

    def on_loaded(self) -> None:
        service = None
        try:
            prefs = self._preferences()
            drop_legacy_keys(prefs)
            settings = VoiceSettings.from_preferences(prefs)
            service = service_module.VoiceService(settings)
            service.start()
            service_module.set_service(service)
            app = QCoreApplication.instance()
            if app is not None:
                app.aboutToQuit.connect(service.stop)
        except Exception:
            # Voice input does not need spoken notifications: carry on without holds.
            logger.exception("lightfall-voice: failed to start VoiceService")
            service = None
        if service is not None:
            self._start_mute_action(service)
        self._start_voice_input(service)

    def _start_mute_action(self, service) -> None:
        """Title-bar mute button. The Assistant panel doesn't exist yet at
        preload, so install on every AgentBus.agents_changed (idempotent)."""
        global _mute_action
        try:
            _mute_action = SessionMuteAction(service, save=save_agent_overrides(self._preferences))
            from lightfall.agents.bus import AgentBus

            AgentBus.get_instance().agents_changed.connect(_mute_action.install)
            _mute_action.install()
        except Exception:
            logger.exception("lightfall-voice: failed to set up the mute button")

    def _start_voice_input(self, service) -> None:
        """Create and start the voice input controller (after VoiceService).

        Timing: preload plugins run before login, the main window and the service plugins.
        ``start()`` therefore only connects signals; the hotkey and the mic buttons attach
        on the first AgentBus ``agents_changed`` (the main window exists by then), and
        ``ensure_started("stt")`` is retried on ``services_changed`` until the ``stt``
        descriptor is registered.  ``stop()`` runs on ``aboutToQuit``.

        ``service`` may be None (VoiceService failed): the controller then skips the
        notification hold/release. A controller whose ``start()`` raises is stopped and
        dropped.
        """
        try:
            controller = controller_module.VoiceInputController(
                VoiceInputSettings.from_preferences(self._preferences()), service
            )
        except Exception:
            logger.exception("lightfall-voice: failed to start voice input")
            return
        try:
            controller.start()
        except Exception:
            logger.exception("lightfall-voice: failed to start voice input")
            try:
                controller.stop()
            except Exception:
                logger.exception("lightfall-voice: stopping the failed controller failed")
            return
        try:
            set_controller(controller)
            app = QCoreApplication.instance()
            if app is not None:
                app.aboutToQuit.connect(controller.stop)
        except Exception:
            logger.exception("lightfall-voice: failed to start voice input")

    # -- schema ----------------------------------------------------------------

    def console_keys(self) -> Sequence[str]:
        return CONSOLE_KEYS

    def specs(self) -> Sequence[SettingSpec]:
        from lightfall_settings import SettingSpec

        return tuple(
            SettingSpec(key=key, type=type_, scopes={"user"}, label=label)
            for key, type_, label in _USER_SPECS
        )

    def sections(self) -> Sequence[Section]:
        return (
            Section("Voice notifications", (
                _field("voice.enabled", Toggle()),
                _field("voice.events.tools", Toggle()),
                _field("voice.events.summary", Toggle()),
                _field("voice.events.permission", Toggle()),
                _field("voice.events.errors", Toggle()),
                _field("voice.default_voice", Custom(voice_choice_editor)),
                Action("Test voice", run=lambda _values: _test_voice()),
            )),
            Section("Agents", (
                _field("voice.agent_overrides", Custom(agent_voices_editor), description=AGENTS_NOTE),
            )),
            Section("Voice output", (
                _field("voice.output.speed", Number(0.5, 2.0, 0.1), placeholder="1.0"),
                _field("voice.output.quality", Choice([(v, label) for label, v in QUALITIES])),
            )),
            Section("Voice input", (
                _field("voice.input.enabled", Toggle()),
                _field("voice.input.hotkey", Custom(hotkey_editor), placeholder="F9"),
                _field("voice.input.model", Choice([(m, m) for m in MODELS])),
                _field("voice.input.language", Choice([(v, label) for label, v in LANGUAGES])),
                _field("voice.input.silence_s", Number(0.3, 5.0, 0.1), unit="s", placeholder="1.0"),
            )),
            Section("This machine", (
                _field("voice.output.output_device", Choice(_output_devices), label="Output device"),
                _field(
                    "voice.output.model_dir",
                    Path(mode="dir"),
                    label="Output model directory",
                    placeholder="Empty: the user cache",
                ),
                _field(
                    "voice.output.python",
                    Path(mode="file"),
                    label="Output Python interpreter",
                    placeholder="Empty: Lightfall's own interpreter",
                ),
                _field("voice.input.device", Choice([(v, label) for label, v in DEVICES]), label="Device"),
                _field("voice.input.input_device", Choice(_input_devices), label="Input device"),
                _field(
                    "voice.input.model_dir",
                    Path(mode="dir"),
                    label="Model directory",
                    placeholder="Empty: the Hugging Face cache",
                ),
                _field(
                    "voice.input.python",
                    Path(mode="file"),
                    label="Python interpreter",
                    placeholder="Empty: Lightfall's own interpreter",
                ),
            )),
        )

    def validate_values(self, values: FormValues) -> list[str]:
        return []

    # -- save ------------------------------------------------------------------

    def after_save(self, scope: str, changed: Mapping[str, Any]) -> None:
        """Re-apply the saved settings to the running service and controller.

        Both settings objects are rebuilt from the resolved store, so running this
        once per saved scope of one Apply gives the same result each time.
        """
        if not any(key.startswith("voice.") for key in changed):
            return
        prefs = self._preferences()
        service = service_module.get_service()
        if service is not None:
            service.apply_settings(VoiceSettings.from_preferences(prefs))
        controller = get_controller()
        if controller is not None:
            controller.apply_settings(VoiceInputSettings.from_preferences(prefs))
