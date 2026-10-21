import numpy as np

from lightfall_voice.audio import (
    CHUNK_SAMPLES,
    RATE,
    float32_to_pcm16,
    pcm16_to_float32,
    resample,
    rms,
)


def test_resample_48k_to_16k_length():
    a = np.random.rand(48000).astype(np.float32)
    assert len(resample(a, 48000, 16000)) == 16000
    assert resample(a, 16000, 16000) is a or len(resample(a, 16000)) == 48000


def test_pcm_roundtrip():
    a = np.linspace(-0.9, 0.9, 1000).astype(np.float32)
    b = float32_to_pcm16(a)
    assert len(b) == 2000
    back = pcm16_to_float32(b)
    assert back.dtype == np.float32
    assert np.max(np.abs(back - a)) < 1e-3


def test_constants_and_rms():
    assert (CHUNK_SAMPLES, RATE) == (1600, 16000)
    assert rms(np.ones(10)) == 1.0
    assert rms(np.zeros(0)) == 0.0
