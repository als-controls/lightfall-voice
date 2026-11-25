"""The ``tts`` ServicePlugin and its manifest entry."""

from __future__ import annotations

import subprocess
import sys


def test_descriptor_values():
    from lightfall.plugins.service_plugin import ServicePlugin
    from lightfall_voice.tts_plugin import TtsServicePlugin

    plugin = TtsServicePlugin()
    assert isinstance(plugin, ServicePlugin)
    assert plugin.name == "tts"
    d = plugin.descriptor()
    assert d.name == "tts"
    assert d.scope == "{prefix}.host.{hostname}.{client_instance}"
    assert d.contract == "1.0"
    assert d.launch == "local"
    assert d.command == ("-m", "lightfall_voice.tts_service")
    assert d.stop_with_client is True
    assert d.single_instance is True
    assert d.interpreter_pref == "voice.output.python"
    assert d.hint == "Install lightfall-voice[tts-service]"


def test_manifest_has_tts_entry():
    from lightfall_voice.manifest import manifest

    assert (
        manifest.description
        == "Voice notifications and voice input for Lightfall agent sessions"
    )
    entries = [e for e in manifest.get_plugins_by_type("service") if e.name == "tts"]
    assert len(entries) == 1
    entry = entries[0]
    assert entry.name == "tts"
    assert entry.import_path == "lightfall_voice.tts_plugin:TtsServicePlugin"
    assert entry.preload is False


def test_plugin_imports_without_kokoro():
    code = (
        "import sys; sys.modules['kokoro_onnx'] = None; sys.modules['onnxruntime'] = None; "
        "import lightfall_voice.tts_plugin, lightfall_voice.tts_service.models, lightfall_voice.manifest"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
