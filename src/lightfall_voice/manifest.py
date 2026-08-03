"""Plugin manifest for lightfall-voice."""

from lightfall.plugins.manifest import PluginEntry, PluginManifest

manifest = PluginManifest(
    name="lightfall-voice",
    version="0.1.0",
    description="Spoken notifications for Claude agent sessions via the local voice daemon",
    plugins=[
        # preload=True so on_loaded() runs at startup and starts the VoiceService.
        PluginEntry(
            type_name="settings",
            name="voice",
            import_path="lightfall_voice.settings:VoiceSettingsPlugin",
            preload=True,
        ),
    ],
)
