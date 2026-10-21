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


def _sine(f, rate, n):
    import numpy as np

    return (0.5 * np.sin(2 * np.pi * f * np.arange(n) / rate)).astype(np.float32)


def _peak_hz(y, rate=16000):
    import numpy as np

    spec = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    return spec.argmax() * rate / len(y), spec.max()


def test_stream_resampler_block_invariance():
    import numpy as np

    from lightfall_voice.audio import StreamResampler

    for src in (48000, 44100):
        x = np.random.default_rng(1).uniform(-0.5, 0.5, src).astype(np.float32)
        whole = StreamResampler(src).process(x)
        rs = StreamResampler(src)
        parts = [rs.process(x[i:i + 1001]) for i in range(0, len(x), 1001)]
        blocks = np.concatenate(parts)
        assert len(blocks) == len(whole)
        assert np.max(np.abs(blocks - whole)) < 1e-5


def test_stream_resampler_tones():
    from lightfall_voice.audio import StreamResampler

    f, _ = _peak_hz(StreamResampler(48000).process(_sine(1000, 48000, 48000)))
    assert abs(f - 1000) < 2
    f, _ = _peak_hz(StreamResampler(48000).process(_sine(7500, 48000, 48000)))
    assert abs(f - 7500) < 2
    y = StreamResampler(48000).process(_sine(11000, 48000, 48000))
    _, loud = _peak_hz(StreamResampler(48000).process(_sine(1000, 48000, 48000)))
    f, peak = _peak_hz(y)
    assert peak < loud * 0.1  # at least 20 dB down; no alias at 5 kHz


def test_stream_resampler_length_and_passthrough():
    import numpy as np

    from lightfall_voice.audio import StreamResampler

    for src in (48000, 44100):
        rs = StreamResampler(src)
        total = sum(len(rs.process(np.zeros(1001, np.float32))) for _ in range(50))
        assert abs(total - 50 * 1001 * 16000 / src) <= 2
    x = np.arange(10, dtype=np.float32)
    assert np.array_equal(StreamResampler(16000).process(x), x)
