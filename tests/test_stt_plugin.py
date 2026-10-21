"""The ``stt`` ServicePlugin and its manifest entry."""

from __future__ import annotations

import subprocess
import sys


def test_descriptor_values():
    from lightfall.plugins.service_plugin import ServicePlugin
    from lightfall_voice.stt_plugin import SttServicePlugin

    plugin = SttServicePlugin()
    assert isinstance(plugin, ServicePlugin)
    assert plugin.name == "stt"
    d = plugin.descriptor()
    assert d.name == "stt"
    assert d.scope == "{prefix}.host.{hostname}.{client_instance}"
    assert d.contract == "1.0"
    assert d.launch == "local"
    assert d.command == ("-m", "lightfall_voice.stt_service")
    assert d.stop_with_client is True
    assert d.single_instance is True
    assert d.interpreter_pref == "voice.input.python"


def test_manifest_has_service_entry():
    from lightfall_voice.manifest import manifest

    entries = manifest.get_plugins_by_type("service")
    assert len(entries) == 1
    entry = entries[0]
    assert entry.name == "stt"
    assert entry.import_path == "lightfall_voice.stt_plugin:SttServicePlugin"
    assert entry.preload is False


def test_plugin_imports_without_faster_whisper():
    code = (
        "import sys; sys.modules['faster_whisper'] = None; sys.modules['ctranslate2'] = None; "
        "import lightfall_voice.stt_plugin, lightfall_voice.manifest"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
