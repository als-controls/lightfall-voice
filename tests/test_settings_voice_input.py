from unittest.mock import MagicMock, patch

import pytest
from PySide6.QtWidgets import QComboBox, QLineEdit, QToolButton

from lightfall_voice import service as service_module
from lightfall_voice import settings as settings_module
from lightfall_voice.settings import VoiceSettingsPlugin
from lightfall_voice.stt.controller import VoiceInputSettings

USER, LOCAL = "user:u1", "local"  # as in conftest


def _input_prefs(prefs, **values):
    """Store ``voice.input.<k>`` values (unscoped, i.e. at User in the fake store)."""
    for k, v in values.items():
        prefs.set(f"voice.input.{k}", v)
    prefs.calls.clear()
    return prefs


@pytest.fixture(autouse=True)
def _clean_globals():
    yield
    service_module.set_service(None)
    settings_module.set_controller(None)


def load(plugin):
    with (
        patch.object(service_module, "VoiceService") as service_cls,
        patch.object(settings_module.controller_module, "VoiceInputController") as ctl_cls,
    ):
        plugin.on_loaded()
    return service_cls, ctl_cls


def test_defaults_loaded(voice_prefs):
    plugin = VoiceSettingsPlugin()
    service_cls, ctl_cls = load(plugin)
    args = ctl_cls.call_args.args
    assert args[0] == VoiceInputSettings()  # defaults: disabled
    assert args[1] is service_cls.return_value
    ctl_cls.return_value.start.assert_called_once_with()
    assert settings_module.get_controller() is ctl_cls.return_value


def test_enabled_starts_service_on_load(voice_prefs):
    """The (real) controller asks the ServiceManager for "stt" only when enabled."""
    from PySide6.QtCore import QObject, Signal

    class Manager(QObject):
        client_created = Signal(str)
        services_changed = Signal()

        def __init__(self):
            super().__init__()
            self.started = []

        def client(self, name):
            return None

        def ensure_started(self, name):
            self.started.append(name)

        def statuses(self):
            return []

    for enabled, expected in ((False, []), (True, ["stt"])):
        _input_prefs(voice_prefs, enabled=enabled)
        manager = Manager()
        real = settings_module.VoiceInputController

        def factory(settings, service, _m=manager, _real=real):
            return _real(settings, service, manager=_m, widget_scanner=lambda: [],
                         window_lookup=lambda: None)

        plugin = VoiceSettingsPlugin()
        with (
            patch.object(service_module, "VoiceService"),
            patch.object(settings_module.controller_module, "VoiceInputController", factory),
        ):
            plugin.on_loaded()
        assert manager.started == expected
        settings_module.get_controller().stop()
        settings_module.set_controller(None)


def test_voice_service_failure_still_starts_voice_input(voice_prefs):
    _input_prefs(voice_prefs, enabled=True)
    plugin = VoiceSettingsPlugin()
    with (
        patch.object(service_module, "VoiceService", side_effect=RuntimeError("tts failed")),
        patch.object(settings_module.controller_module, "VoiceInputController") as ctl_cls,
    ):
        plugin.on_loaded()
    assert ctl_cls.call_args.args[1] is None  # voice_service=None: no holds
    ctl_cls.return_value.start.assert_called_once_with()
    assert settings_module.get_controller() is ctl_cls.return_value


def test_controller_start_failure_stops_and_drops_it(voice_prefs):
    _input_prefs(voice_prefs, enabled=True)
    plugin = VoiceSettingsPlugin()
    with (
        patch.object(service_module, "VoiceService"),
        patch.object(settings_module.controller_module, "VoiceInputController") as ctl_cls,
    ):
        ctl_cls.return_value.start.side_effect = RuntimeError("boom")
        plugin.on_loaded()
    ctl_cls.return_value.stop.assert_called_once_with()
    assert settings_module.get_controller() is None


# ---------------------------------------------------------------------------
# Form page: voice input fields
# ---------------------------------------------------------------------------


def _dialog(qtbot, scope=USER):
    from lightfall.ui.preferences.dialog import PreferencesDialog

    dialog = PreferencesDialog()
    qtbot.addWidget(dialog)
    dialog._prompt_restart = lambda: None
    dialog.select_scope(scope)
    return dialog


def _editor(dialog, key):
    from lightfall.ui.preferences.form.view import SettingsForm

    form = next(f for f in dialog.findChildren(SettingsForm) if f._plugin.name == "voice")
    return form._rows[key].editor


def _combo_items(editor):
    box = editor.findChild(QComboBox)
    return [box.itemText(i) for i in range(box.count())]


def _pick(dialog, key, text):
    box = _editor(dialog, key).findChild(QComboBox)
    index = box.findText(text)
    assert index >= 0, (key, text, _combo_items(_editor(dialog, key)))
    box.setCurrentIndex(index)


def _type(dialog, key, text):
    edit = _editor(dialog, key).findChild(QLineEdit)
    edit.setText(text)
    edit.textEdited.emit(text)


def test_input_device_choices_from_qmediadevices(qtbot, voice_env, monkeypatch):
    from PySide6.QtMultimedia import QMediaDevices

    mic = MagicMock()
    mic.description.return_value = "USB mic"
    mic.id.return_value = b"usb-1"
    monkeypatch.setattr(QMediaDevices, "audioInputs", staticmethod(lambda: [mic]))
    voice_env.prefs.layers[(LOCAL, "voice.input.input_device")] = "gone"
    dialog = _dialog(qtbot, LOCAL)
    editor = _editor(dialog, "voice.input.input_device")
    assert _combo_items(editor) == ["Not set", "System default", "USB mic", "gone"]  # unlisted kept
    assert editor.value() == "gone"
    _pick(dialog, "voice.input.input_device", "System default")
    assert editor.value() == ""
    _pick(dialog, "voice.input.input_device", "USB mic")
    assert dialog._apply() is True
    assert voice_env.prefs.calls == [("voice.input.input_device", "usb-1", LOCAL)]


def test_voice_input_fields_save_at_their_scopes(qtbot, voice_env, monkeypatch):
    from PySide6.QtWidgets import QFileDialog

    controller = MagicMock()
    settings_module.set_controller(controller)
    dialog = _dialog(qtbot)
    assert _combo_items(_editor(dialog, "voice.input.language"))[:2] == ["Not set", "Auto-detect"]
    _pick(dialog, "voice.input.enabled", "On")
    _pick(dialog, "voice.input.model", "medium")
    _pick(dialog, "voice.input.language", "Auto-detect")
    _type(dialog, "voice.input.silence_s", "1.5")

    dialog.select_scope(LOCAL)
    _pick(dialog, "voice.input.device", "CPU")
    monkeypatch.setattr(
        QFileDialog, "getExistingDirectory", staticmethod(lambda *a, **k: "D:/models")
    )
    _editor(dialog, "voice.input.model_dir").findChild(QToolButton).click()
    _type(dialog, "voice.input.python", " C:/py/python.exe ")
    assert dialog._apply() is True

    assert sorted(voice_env.prefs.calls) == sorted([
        ("voice.input.enabled", True, USER),
        ("voice.input.model", "medium", USER),
        ("voice.input.language", "", USER),
        ("voice.input.silence_s", 1.5, USER),
        ("voice.input.device", "cpu", LOCAL),
        ("voice.input.model_dir", "D:/models", LOCAL),
        ("voice.input.python", "C:/py/python.exe", LOCAL),
    ])
    applied = controller.apply_settings.call_args.args[0]
    assert applied == VoiceInputSettings(
        enabled=True,
        model_dir="D:/models",
        hotkey="F9",
        model="medium",
        device="cpu",
        input_device="",
        silence_s=1.5,
        language="",
        python="C:/py/python.exe",
    )
