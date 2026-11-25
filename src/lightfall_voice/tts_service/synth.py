"""Sentence splitter and Kokoro ONNX / fake synthesizers for the TTS service."""

from __future__ import annotations

import os
import re
from typing import Any

import numpy as np

RATE = 24000
MAX_SENTENCE_LEN = 200
_ABBREVIATIONS = frozenset(
    {
        "dr",
        "mr",
        "mrs",
        "ms",
        "prof",
        "st",
        "vs",
        "etc",
        "e.g",
        "i.e",
        "approx",
        "no",
        "fig",
        "sr",
        "jr",
    }
)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?;])\s+")


def _is_non_breaking_period(token: str) -> bool:
    if not token.endswith("."):
        return False
    word = token[:-1].lstrip("([{\"'")
    if len(word) == 1 and word.isupper():
        return True
    return word.lower() in _ABBREVIATIONS


def _split_long(part: str, max_len: int = MAX_SENTENCE_LEN) -> list[str]:
    out: list[str] = []
    while len(part) > max_len:
        punct = max(part.rfind(c, 0, max_len) for c in (",", ";", ":"))
        if punct > 0:
            head, part = part[: punct + 1].strip(), part[punct + 1 :].strip()
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
    """Split ``text`` into speakable sentence chunks of at most 200 characters."""
    if not text:
        return []
    result: list[str] = []
    for line in text.splitlines():
        start = 0
        raw_parts: list[str] = []
        for match in _SENTENCE_SPLIT_RE.finditer(line):
            segment = line[start : match.start()]
            words = segment.rsplit(None, 1)
            if words and _is_non_breaking_period(words[-1]):
                continue
            raw_parts.append(segment)
            start = match.end()
        raw_parts.append(line[start:])
        for raw in raw_parts:
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
        import onnxruntime as rt
        from kokoro_onnx import Kokoro

        self.model_path = os.fspath(model_path)
        self.voices_path = os.fspath(voices_path)
        # Spinning intra-op threads collapse under any competing CPU load: measured on a
        # 24-core host, the same sentence went 0.9 s -> 5 s after the first call with
        # spinning on, and held at 1.7 s with it off.
        opts = rt.SessionOptions()
        opts.intra_op_num_threads = min(8, os.cpu_count() or 1)
        opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
        session = rt.InferenceSession(
            self.model_path, sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self._kokoro = Kokoro.from_session(session, self.voices_path)

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

    def warm_up(self, lang: str = "en-us") -> None:
        """Run one short synthesis so the first real one isn't slowed by session warm-up."""
        voices = self.voices()
        voice = "af_heart" if "af_heart" in voices else min(voices)
        self.synthesize("Ready.", voice=voice, lang=lang)


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
