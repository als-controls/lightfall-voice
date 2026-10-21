"""Speech-to-text backends for the stt service.

The only module that imports faster-whisper (and ctranslate2), and only inside
:meth:`FasterWhisperTranscriber.load`, so the service imports without them.
"""

from __future__ import annotations

import importlib.util
import logging
import os
from collections.abc import Callable
from typing import Any, Protocol

import numpy as np

from lightfall_voice.audio import RATE

log = logging.getLogger(__name__)

DEVICES = ("auto", "gpu", "cpu")
_NVIDIA_PACKAGES = ("nvidia.cublas", "nvidia.cudnn")


class Transcriber(Protocol):
    device: str  # "cuda" or "cpu" once loaded

    def transcribe(self, audio_16k: np.ndarray, language: str | None) -> str: ...


def add_nvidia_dll_dirs() -> list[str]:
    """On Windows, add the ``bin`` dirs of installed ``nvidia-cublas-cu12`` /
    ``nvidia-cudnn-cu12`` wheels to the DLL search path (ctranslate2 loads them lazily)."""
    if os.name != "nt":
        return []
    added: list[str] = []
    for name in _NVIDIA_PACKAGES:
        try:
            spec = importlib.util.find_spec(name)
        except (ImportError, ValueError):
            spec = None
        for loc in (spec.submodule_search_locations or []) if spec is not None else []:
            bin_dir = os.path.join(loc, "bin")
            if not os.path.isdir(bin_dir):
                continue
            try:
                os.add_dll_directory(bin_dir)
            except OSError:
                log.debug("add_dll_directory(%s) failed", bin_dir, exc_info=True)
                continue
            os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
            added.append(bin_dir)
    return added


class FasterWhisperTranscriber:
    """faster-whisper backend.

    ``device``: ``auto`` tries ``cuda``/``float16`` and falls back to ``cpu``/``int8`` on any
    load error; ``gpu`` forces ``cuda``/``float16`` (an error is final); ``cpu`` is ``int8``.
    A non-empty ``model_dir`` is the ``download_root`` and loading is offline.
    """

    def __init__(
        self,
        model: str,
        device: str,
        model_dir: str = "",
        language: str = "en",
        progress: Callable[[str], None] = lambda s: None,
    ) -> None:
        if device not in DEVICES:
            raise ValueError(f"device must be one of {DEVICES}, got {device!r}")
        self.model = model
        self.requested_device = device
        self.model_dir = model_dir
        self.language = language
        self.progress = progress
        self.device = ""
        self._model: Any = None

    def load(self) -> None:
        """Load the model (blocking); may download it first unless ``model_dir`` is set."""
        add_nvidia_dll_dirs()
        import faster_whisper  # lazy: only the stt-service extra installs it

        self.progress("loading" if self._is_local(faster_whisper) else "downloading")
        if self.requested_device == "cpu":
            self._model = self._make(faster_whisper, "cpu", "int8")
            self.device = "cpu"
        elif self.requested_device == "gpu":
            self._model = self._make(faster_whisper, "cuda", "float16")
            self.device = "cuda"
        else:
            self._model, self.device = self._load_auto(faster_whisper)

    def _load_auto(self, fw: Any) -> tuple[Any, str]:
        if _cuda_device_count() != 0:
            try:
                model = self._make(fw, "cuda", "float16")
                # cuBLAS/cuDNN are loaded on first use: run one tiny decode so a missing
                # library falls back here rather than failing the first utterance.
                segments, _ = model.transcribe(
                    np.zeros(RATE // 2, dtype=np.float32),
                    language=self.language or None,
                    beam_size=1,
                )
                list(segments)
                return model, "cuda"
            except Exception as e:
                log.warning("CUDA load failed (%s); falling back to CPU", e)
        return self._make(fw, "cpu", "int8"), "cpu"

    def _make(self, fw: Any, device: str, compute_type: str) -> Any:
        kw: dict[str, Any] = {"device": device, "compute_type": compute_type}
        if self.model_dir:
            kw["download_root"] = self.model_dir
            kw["local_files_only"] = True
        return fw.WhisperModel(self.model, **kw)

    def _is_local(self, fw: Any) -> bool:
        if self.model_dir or os.path.isdir(self.model):
            return True
        try:
            fw.download_model(self.model, local_files_only=True)
        except Exception:
            return False
        return True

    def transcribe(self, audio_16k: np.ndarray, language: str | None) -> str:
        if self._model is None:
            raise RuntimeError("model not loaded")
        segments, _info = self._model.transcribe(
            np.asarray(audio_16k, dtype=np.float32),
            language=language or None,
            vad_filter=True,
        )
        return "".join(s.text for s in segments).strip()


def _cuda_device_count() -> int:
    """CUDA devices visible to ctranslate2; -1 when unknown (then just try)."""
    try:
        import ctranslate2

        return int(ctranslate2.get_cuda_device_count())
    except Exception:
        return -1
