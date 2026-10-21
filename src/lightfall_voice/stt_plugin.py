"""ServicePlugin declaring the local ``stt`` (speech-to-text) service.

Lightfall's ServiceManager launches it on demand as
``<interpreter> -m lightfall_voice.stt_service --scope ... --nats-url ... --token-stdin``,
where the interpreter is the ``voice.input.python`` preference (empty: Lightfall's own).
Importing this module never imports faster-whisper, ctranslate2 or CUDA.
"""

from __future__ import annotations

from lightfall.plugins.service_plugin import ServicePlugin
from lightfall_utils.ipc.services.descriptor import ServiceDescriptor

NAME = "stt"
CONTRACT = "1.0"
INTERPRETER_PREF = "voice.input.python"


class SttServicePlugin(ServicePlugin):
    """Declares the ``stt`` service (Whisper transcription of microphone utterances)."""

    @property
    def name(self) -> str:
        return NAME

    @property
    def description(self) -> str:
        return "Local speech-to-text service (faster-whisper)"

    def descriptor(self) -> ServiceDescriptor:
        return ServiceDescriptor(
            name=NAME,
            scope="{prefix}.host.{hostname}.{client_instance}",
            contract=CONTRACT,
            launch="local",
            command=("-m", "lightfall_voice.stt_service"),
            stop_with_client=True,
            single_instance=True,
            interpreter_pref=INTERPRETER_PREF,
            hint="Install lightfall-voice[stt-service]",
        )
