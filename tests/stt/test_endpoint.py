import numpy as np

from lightfall_voice.stt_service.endpoint import EndpointDetector

RATE = 16000


def _tone(seconds, amp=0.3):
    t = np.arange(int(seconds * RATE)) / RATE
    return (amp * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def _zeros(seconds):
    return np.zeros(int(seconds * RATE), dtype=np.float32)


def _run(det, audio, chunk=1600):
    events = []
    for i in range(0, len(audio), chunk):
        ev = det.feed(audio[i : i + chunk])
        if ev:
            events.append(ev)
            if ev != "listening":
                break
    return events


def test_tone_then_silence_ends_with_silence():
    det = EndpointDetector()
    assert _run(det, np.concatenate([_tone(1.0), _zeros(1.2)])) == ["listening", "silence"]


def test_only_silence_gives_no_speech_after_8s():
    det = EndpointDetector()
    assert _run(det, _zeros(7.5)) == []
    assert _run(det, _zeros(1.0)) == ["no_speech"]


def test_max_length_cap():
    det = EndpointDetector()
    assert _run(det, _tone(61.0)) == ["listening", "max"]


def test_short_blip_not_speech():
    det = EndpointDetector()
    events = _run(det, np.concatenate([_tone(0.2), _zeros(9.0)]))
    assert events == ["no_speech"]


def test_custom_silence_s():
    det = EndpointDetector(silence_s=0.3)
    assert _run(det, np.concatenate([_tone(1.0), _zeros(0.5)])) == ["listening", "silence"]
    det = EndpointDetector(silence_s=2.0)
    assert _run(det, np.concatenate([_tone(1.0), _zeros(1.5)])) == ["listening"]


def test_unaligned_chunks_and_audio():
    det = EndpointDetector()
    a = np.concatenate([_tone(1.0), _zeros(1.2)])
    events = _run(det, a, chunk=777)
    assert events == ["listening", "silence"]
    assert len(det.audio()) <= len(a)
    assert len(det.audio()) >= 2 * RATE
