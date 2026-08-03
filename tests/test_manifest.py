def test_manifest_declares_preloaded_settings_plugin():
    from lightfall_voice.manifest import manifest

    entries = manifest.get_plugins_by_type("settings")
    assert len(entries) == 1
    entry = entries[0]
    assert entry.name == "voice"
    assert entry.preload is True
    assert entry.import_path == "lightfall_voice.settings:VoiceSettingsPlugin"
