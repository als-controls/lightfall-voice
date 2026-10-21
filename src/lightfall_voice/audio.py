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
