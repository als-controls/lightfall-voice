"""Audio helpers shared by the STT service and the recorder (no Qt)."""

from __future__ import annotations

import numpy as np

RATE = 16000
CHUNK_SAMPLES = 1600


def pcm16_to_float32(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype="<i2").astype(np.float32) / 32768.0


def float32_to_pcm16(a) -> bytes:
    arr = np.clip(np.asarray(a, dtype=np.float32), -1.0, 1.0)
    return np.round(arr * 32767.0).astype("<i2").tobytes()


def resample(a: np.ndarray, src_rate: int, dst_rate: int = RATE) -> np.ndarray:
    """Linear resample via np.interp."""
    a = np.asarray(a, dtype=np.float32)
    if src_rate == dst_rate or len(a) == 0:
        return a
    n_out = int(round(len(a) * dst_rate / src_rate))
    x_out = np.arange(n_out) * (src_rate / dst_rate)
    return np.interp(x_out, np.arange(len(a)), a).astype(np.float32)


def rms(a) -> float:
    arr = np.asarray(a, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(arr * arr)))


class StreamResampler:
    """Stateful mono resampler: windowed-sinc low-pass, then linear interpolation.

    Filter history and the fractional read position carry across ``process`` calls,
    so block size does not affect the output. numpy only.
    """

    TAPS = 65

    def __init__(self, src_rate: int, dst_rate: int = RATE):
        self.src_rate = src_rate
        self.dst_rate = dst_rate
        self._passthrough = src_rate == dst_rate
        self._step = src_rate / dst_rate
        fc = 0.45 * min(src_rate, dst_rate) / src_rate  # cycles per input sample
        n = np.arange(self.TAPS) - (self.TAPS - 1) / 2
        h = 2 * fc * np.sinc(2 * fc * n) * np.hamming(self.TAPS)
        self._h = h / h.sum()
        self._hist = np.zeros(self.TAPS - 1, dtype=np.float64)
        self._buf = np.zeros(0, dtype=np.float64)
        self._pos = 0.0

    def process(self, a) -> np.ndarray:
        a = np.asarray(a, dtype=np.float32)
        if self._passthrough or len(a) == 0:
            return a
        x = np.concatenate([self._hist, a.astype(np.float64)])
        filtered = np.convolve(x, self._h, mode="valid")  # len(a) samples
        self._hist = x[-(self.TAPS - 1):]
        buf = np.concatenate([self._buf, filtered])
        last = len(buf) - 1
        if self._pos > last:
            self._buf = buf
            return np.zeros(0, dtype=np.float32)
        n_out = int((last - self._pos) // self._step) + 1
        xs = self._pos + np.arange(n_out) * self._step
        out = np.interp(xs, np.arange(len(buf)), buf).astype(np.float32)
        new_pos = self._pos + n_out * self._step
        k = min(int(new_pos), len(buf))  # excess stays in pos for the next block
        self._buf = buf[k:]
        self._pos = new_pos - k
        return out
