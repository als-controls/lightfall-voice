"""ServicePlugin declaring the local ``tts`` (text-to-speech) service.

Lightfall's ServiceManager launches it on demand as
``<interpreter> -m lightfall_voice.tts_service --scope ... --nats-url ... --token-stdin``,
where the interpreter is the ``voice.output.python`` preference (empty: Lightfall's own).
Importing this module never imports kokoro_onnx or onnxruntime.
"""

from __future__ import annotations

from lightfall.plugins.service_plugin import ServicePlugin
from lightfall_utils.ipc.services.descriptor import ServiceDescriptor

NAME = "tts"
CONTRACT = "1.0"
INTERPRETER_PREF = "voice.output.python"


class TtsServicePlugin(ServicePlugin):
    """Declares the ``tts`` service (Kokoro speech synthesis)."""

    @property
    def name(self) -> str:
        return NAME

    @property
    def description(self) -> str:
        return "Local text-to-speech service (kokoro-onnx)"

    def descriptor(self) -> ServiceDescriptor:
        return ServiceDescriptor(
            name=NAME,
            scope="{prefix}.host.{hostname}.{client_instance}",
            contract=CONTRACT,
            launch="local",
            command=("-m", "lightfall_voice.tts_service"),
            stop_with_client=True,
            single_instance=True,
            interpreter_pref=INTERPRETER_PREF,
            hint="Install lightfall-voice[tts-service]",
        )
