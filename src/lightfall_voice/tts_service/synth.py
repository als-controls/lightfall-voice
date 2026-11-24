"""Sentence splitter and Kokoro ONNX / fake synthesizers for the TTS service."""

from __future__ import annotations

import os
import re
from typing import Any

import numpy as np

RATE = 24000
MAX_SENTENCE_LEN = 400
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?;])\s+|\r?\n")


def _split_long(part: str, max_len: int = MAX_SENTENCE_LEN) -> list[str]:
    out: list[str] = []
    while len(part) > max_len:
        comma = part.rfind(",", 0, max_len)
        if comma > 0:
            head, part = part[: comma + 1].strip(), part[comma + 1 :].strip()
        else:
            space = part.rfind(" ", 0, max_len)
            if space > 0:
                head, part = part[:space].strip(), part[space + 1 :].strip()
            else:
                head, part = part[:max_len].strip(), part[max_len:].strip()
        if head:
            out.append(head)
    if part:
        out.append(part)
    return out


def split_sentences(text: str) -> list[str]:
    """Split ``text`` into speakable sentence chunks of at most 400 characters."""
    if not text:
        return []
    result: list[str] = []
    for raw in _SENTENCE_SPLIT_RE.split(text):
        part = raw.strip()
        if not part:
            continue
        if len(part) > MAX_SENTENCE_LEN:
            result.extend(_split_long(part))
        else:
            result.append(part)
    return result


class KokoroSynth:
    """Kokoro ONNX text-to-speech synthesizer (imports ``kokoro_onnx`` lazily)."""

    RATE = RATE

    def __init__(
        self,
        model_path: str | os.PathLike[str],
        voices_path: str | os.PathLike[str],
    ) -> None:
        from kokoro_onnx import Kokoro

        self.model_path = os.fspath(model_path)
        self.voices_path = os.fspath(voices_path)
        self._kokoro = Kokoro(self.model_path, self.voices_path)

    def voices(self) -> set[str]:
        return set(self._kokoro.get_voices())

    def synthesize(
        self,
        sentence: str,
        voice: str = "af_heart",
        speed: float = 1.0,
        lang: str = "en-us",
    ) -> np.ndarray:
        samples, sr = self._kokoro.create(
            sentence, voice=voice, speed=speed, lang=lang
        )
        arr = np.asarray(samples, dtype=np.float32).reshape(-1)
        if sr != self.RATE:
            from lightfall_voice.audio import resample

            arr = resample(arr, int(sr), self.RATE)
        return arr


class FakeSynth:
    """Test-only synthesizer generating a 440 Hz tone (0.05 s per word at amplitude 0.3)."""

    RATE = RATE

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def voices(self) -> set[str]:
        return {"af_heart", "bm_george"}

    def synthesize(
        self,
        sentence: str,
        voice: str = "af_heart",
        speed: float = 1.0,
        lang: str = "en-us",
    ) -> np.ndarray:
        words = max(1, len(sentence.split()))
        n_samples = int(self.RATE * 0.05 * words)
        t = np.arange(n_samples, dtype=np.float32) / self.RATE
        return (0.3 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)
