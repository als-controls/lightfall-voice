"""Plugin manifest for lightfall-voice."""

from importlib.metadata import PackageNotFoundError, version

from lightfall.plugins.manifest import PluginEntry, PluginManifest

try:
    _version = version("lightfall-voice")
except PackageNotFoundError:
    _version = "0.0.0"

manifest = PluginManifest(
    name="lightfall-voice",
    version=_version,
    description="Voice notifications and voice input for Lightfall agent sessions",
    plugins=[
        # preload=True so on_loaded() runs at startup and starts the VoiceService.
        PluginEntry(
            type_name="settings",
            name="voice",
            import_path="lightfall_voice.settings:VoiceSettingsPlugin",
            preload=True,
        ),
        PluginEntry(
            type_name="service",
            name="stt",
            import_path="lightfall_voice.stt_plugin:SttServicePlugin",
        ),
        PluginEntry(
            type_name="service",
            name="tts",
            import_path="lightfall_voice.tts_plugin:TtsServicePlugin",
        ),
    ],
)
