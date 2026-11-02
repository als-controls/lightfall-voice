"""Characterization of the hand-built Voice page (removed by the form rewrite)."""
from __future__ import annotations

from PySide6.QtMultimedia import QMediaDevices

from lightfall_voice import service as service_module
from lightfall_voice import settings as settings_module
from lightfall_voice.settings import VoiceSettingsPlugin


def test_old_page_writes_all_keys_unscoped(qtbot, monkeypatch):
    """The untouched old page writes every voice key, unscoped, on save."""
    calls = []

    class FakePrefs:
        def get(self, key, default=None):
            return default

        def set(self, key, value, **kwargs):
            calls.append((key, value, kwargs))

    monkeypatch.setattr(QMediaDevices, "audioInputs", staticmethod(lambda: []))
    service_module.set_service(None)
    settings_module.set_controller(None)
    plugin = VoiceSettingsPlugin()
    plugin._prefs = FakePrefs()
    widget = plugin.create_widget()
    qtbot.addWidget(widget)
    plugin.load_settings()
    plugin.save_settings()
    assert calls == [
        ("voice.enabled", True, {}),
        ("voice.events.tools", True, {}),
        ("voice.events.summary", True, {}),
        ("voice.events.permission", True, {}),
        ("voice.events.errors", True, {}),
        ("voice.daemon_host", "127.0.0.1", {}),
        ("voice.daemon_port", 52849, {}),
        ("voice.default_voice", "af_heart", {}),
        ("voice.agent_overrides", {}, {}),
        ("voice.input.enabled", False, {}),
        ("voice.input.model_dir", "", {}),
        ("voice.input.hotkey", "F9", {}),
        ("voice.input.model", "small", {}),
        ("voice.input.device", "auto", {}),
        ("voice.input.input_device", "", {}),
        ("voice.input.silence_s", 1.0, {}),
        ("voice.input.language", "en", {}),
        ("voice.input.python", "", {}),
    ]
