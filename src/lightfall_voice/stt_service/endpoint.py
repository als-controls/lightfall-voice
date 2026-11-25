"""Pure-numpy end-of-speech detector."""

from __future__ import annotations

from typing import Literal

import numpy as np

from lightfall_voice.audio import RATE, rms

Event = Literal["listening", "silence", "no_speech", "max"]


class EndpointDetector:
    def __init__(
        self,
        silence_s: float = 1.0,
        threshold: float = 0.01,
        window_s: float = 0.05,
        min_speech_s: float = 0.5,
        no_speech_s: float = 8.0,
        max_s: float = 60.0,
    ) -> None:
        self.silence_s = silence_s
        self.threshold = threshold
        self.min_speech_s = min_speech_s
        self.no_speech_s = no_speech_s
        self.max_s = max_s
        self._win = max(1, int(round(window_s * RATE)))
        self._chunks: list[np.ndarray] = []
        self._rem = np.zeros(0, dtype=np.float32)
        self._windows = 0
        self._speech_s = 0.0
        self._trailing_s = 0.0
        self._detected = False

    def audio(self) -> np.ndarray:
        if not self._chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self._chunks)

    def feed(self, samples: np.ndarray) -> Event | None:
        """Feed samples; return "listening" once, then the end reason when reached.

        If speech is first detected and an end is reached within the same
        chunk, only the end reason is returned.
        """
        samples = np.asarray(samples, dtype=np.float32)
        self._chunks.append(samples)
        buf = np.concatenate([self._rem, samples]) if len(self._rem) else samples
        n = len(buf) // self._win
        self._rem = buf[n * self._win :]
        wsec = self._win / RATE
        listening = False
        for i in range(n):
            self._windows += 1
            elapsed = self._windows * wsec
            is_speech = rms(buf[i * self._win : (i + 1) * self._win]) >= self.threshold
            if is_speech:
                self._speech_s += wsec
                self._trailing_s = 0.0
            else:
                self._speech_s = 0.0
                self._trailing_s += wsec
            if not self._detected and self._speech_s >= self.min_speech_s - 1e-9:
                self._detected = True
                listening = True
            if elapsed >= self.max_s - 1e-9:
                return "max"
            if self._detected:
                if self._trailing_s >= self.silence_s - 1e-9:
                    return "silence"
            elif elapsed >= self.no_speech_s - 1e-9:
                return "no_speech"
        return "listening" if listening else None
