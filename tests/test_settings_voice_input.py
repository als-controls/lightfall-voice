from unittest.mock import MagicMock, patch

import pytest

from lightfall_voice import service as service_module
from lightfall_voice import settings as settings_module
from lightfall_voice.settings import VoiceSettingsPlugin
from lightfall_voice.stt.controller import VoiceInputSettings


class FakePrefs:
    def __init__(self, **values):
        self.data = {f"voice.input.{k}": v for k, v in values.items()}

    def get(self, key, default=None):
        return self.data.get(key, default)

    def set(self, key, value):
        self.data[key] = value


def make_plugin(prefs):
    plugin = VoiceSettingsPlugin()
    plugin._prefs = prefs
    return plugin


@pytest.fixture(autouse=True)
def _clean_globals():
    yield
    service_module.set_service(None)
    settings_module.set_controller(None)


def load(plugin, prefs):
    with (
        patch.object(service_module, "VoiceService") as service_cls,
        patch.object(settings_module.controller_module, "VoiceInputController") as ctl_cls,
    ):
        plugin.on_loaded()
    return service_cls, ctl_cls


def test_defaults_loaded():
    prefs = FakePrefs()
    plugin = make_plugin(prefs)
    service_cls, ctl_cls = load(plugin, prefs)
    args = ctl_cls.call_args.args
    assert args[0] == VoiceInputSettings()  # defaults: disabled
    assert args[1] is service_cls.return_value
    ctl_cls.return_value.start.assert_called_once_with()
    assert settings_module.get_controller() is ctl_cls.return_value


def test_enabled_starts_service_on_load():
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
        prefs = FakePrefs(enabled=enabled)
        manager = Manager()
        real = settings_module.VoiceInputController

        def factory(settings, service, _m=manager, _real=real):
            return _real(settings, service, manager=_m, widget_scanner=lambda: [],
                         window_lookup=lambda: None)

        plugin = make_plugin(prefs)
        with (
            patch.object(service_module, "VoiceService"),
            patch.object(settings_module.controller_module, "VoiceInputController", factory),
        ):
            plugin.on_loaded()
        assert manager.started == expected
        settings_module.get_controller().stop()
        settings_module.set_controller(None)


def test_save_applies_to_controller(qtbot):
    prefs = FakePrefs()
    plugin = make_plugin(prefs)
    widget = plugin.create_widget()
    qtbot.addWidget(widget)
    plugin.load_settings()

    plugin._input_enabled_check.setChecked(True)
    plugin._model_dir_edit.setText("D:/models")
    plugin._python_edit.setText("C:/py/python.exe")
    plugin._model_combo.setCurrentText("medium")
    plugin._device_combo.setCurrentIndex(plugin._device_combo.findData("cpu"))
    plugin._language_combo.setCurrentIndex(plugin._language_combo.findData(""))
    plugin._silence_spin.setValue(1.5)

    controller = MagicMock()
    settings_module.set_controller(controller)
    plugin.save_settings()

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
    assert VoiceInputSettings.from_preferences(prefs) == applied


def test_widget_has_all_fields(qtbot):
    prefs = FakePrefs(model="large-v3", language="", hotkey="Ctrl+F8", input_device="gone")
    plugin = make_plugin(prefs)
    widget = plugin.create_widget()
    qtbot.addWidget(widget)
    plugin.load_settings()

    for name in ("_input_enabled_check", "_model_dir_edit", "_python_edit", "_hotkey_edit",
                 "_model_combo", "_device_combo", "_language_combo", "_input_device_combo",
                 "_silence_spin"):
        assert hasattr(plugin, name), name
    assert plugin._language_combo.itemText(0) == "Auto-detect"
    assert plugin._language_combo.currentData() == ""
    assert plugin._input_device_combo.itemData(0) == ""
    assert plugin._input_device_combo.itemText(0) == "System default"
    assert plugin._input_device_combo.currentData() == "gone"  # unlisted value kept
    assert plugin._model_combo.currentData() == "large-v3"
    assert plugin._hotkey_edit.keySequence().toString() == "Ctrl+F8"
    assert plugin._silence_spin.minimum() == 0.3
    assert plugin._silence_spin.maximum() == 5.0
    assert [plugin._model_combo.itemText(i) for i in range(plugin._model_combo.count())] == [
        "base", "small", "medium", "large-v3", "distil-large-v3"
    ]


def test_save_without_controller_only_persists(qtbot):
    prefs = FakePrefs()
    plugin = make_plugin(prefs)
    widget = plugin.create_widget()
    qtbot.addWidget(widget)
    plugin.load_settings()
    plugin._silence_spin.setValue(2.0)
    plugin.save_settings()
    assert prefs.data["voice.input.silence_s"] == 2.0
