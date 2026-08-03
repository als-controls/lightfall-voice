from unittest.mock import MagicMock, patch

from lightfall_voice.policy import VoiceSettings
from lightfall_voice import service as service_module
from lightfall_voice.settings import VoiceSettingsPlugin


class FakePrefs:
    def __init__(self):
        self.data = {}
    def get(self, key, default=None):
        return self.data.get(key, default)
    def set(self, key, value):
        self.data[key] = value


def make_plugin(prefs=None):
    plugin = VoiceSettingsPlugin()
    plugin._prefs = prefs or FakePrefs()  # test seam, see implementation
    return plugin


def test_identity():
    p = make_plugin()
    assert p.name == "voice"
    assert p.display_name == "Voice Notifications"
    assert p.category == "plugins"


def test_on_loaded_starts_service():
    p = make_plugin()
    with patch.object(service_module, "VoiceService") as service_cls:
        p.on_loaded()
        service_cls.assert_called_once()
        service_cls.return_value.start.assert_called_once()
    assert service_module.get_service() is service_cls.return_value
    service_module.set_service(None)


def test_widget_roundtrip_saves_settings(qtbot):
    prefs = FakePrefs()
    p = make_plugin(prefs)
    widget = p.create_widget()
    qtbot.addWidget(widget)
    p.load_settings()

    p._enabled_check.setChecked(False)
    p._tools_check.setChecked(False)
    p._port_spin.setValue(6000)

    service = MagicMock()
    service_module.set_service(service)
    try:
        p.save_settings()
    finally:
        service_module.set_service(None)

    loaded = VoiceSettings.from_preferences(prefs)
    assert loaded.enabled is False
    assert loaded.tools is False
    assert loaded.daemon_port == 6000
    service.apply_settings.assert_called_once()


def test_validate_rejects_bad_port(qtbot):
    p = make_plugin()
    widget = p.create_widget()
    qtbot.addWidget(widget)
    p._host_edit.setText("")
    errors = p.validate()
    assert errors  # empty host is invalid


def test_test_button_speaks(qtbot):
    p = make_plugin()
    widget = p.create_widget()
    qtbot.addWidget(widget)
    service = MagicMock()
    service_module.set_service(service)
    try:
        p._on_test_clicked()
        service._client.speak.assert_called_once()
    finally:
        service_module.set_service(None)
