import numpy as np
from PySide6.QtCore import QIODevice, QObject, Signal
from PySide6.QtMultimedia import QAudio, QAudioDevice, QAudioFormat, QMediaDevices

from lightfall_voice.audio import float32_to_pcm16
from lightfall_voice.stt.recorder import Recorder


class Pipe(QIODevice):
    """In-memory read-only device; feed() appends bytes and emits readyRead."""

    def __init__(self):
        super().__init__()
        self._buf = bytearray()
        self.open(QIODevice.OpenModeFlag.ReadOnly)

    def feed(self, data: bytes):
        self._buf += data
        self.readyRead.emit()

    def readData(self, maxlen):
        out = bytes(self._buf[:maxlen])
        del self._buf[:maxlen]
        return out

    def writeData(self, data):
        return -1

    def bytesAvailable(self):
        return len(self._buf) + super().bytesAvailable()


class FakeSource(QObject):
    stateChanged = Signal(object)

    def __init__(self, fmt):
        super().__init__()
        self._fmt = fmt
        self.pipe = Pipe()
        self._error = QAudio.Error.NoError
        self.stopped = False

    def format(self):
        return self._fmt

    def start(self):
        return self.pipe

    def stop(self):
        self.stopped = True

    def error(self):
        return self._error

    def fail(self):
        self._error = QAudio.Error.IOError
        self.stateChanged.emit(QAudio.State.StoppedState)


def make(rate=16000, channels=1):
    fmt = QAudioFormat()
    fmt.setSampleRate(rate)
    fmt.setChannelCount(channels)
    fmt.setSampleFormat(QAudioFormat.SampleFormat.Int16)
    holder = {}

    def factory(_fmt, _dev):
        holder["src"] = FakeSource(fmt)
        return holder["src"]

    rec = Recorder(source_factory=factory)
    chunks, levels, errors = [], [], []
    rec.chunk.connect(chunks.append)
    rec.level.connect(levels.append)
    rec.failed.connect(errors.append)
    rec.start()
    return rec, holder["src"], chunks, levels, errors


def test_emits_exact_3200_byte_chunks(qtbot):
    rec, src, chunks, _, _ = make()
    data = bytes(range(256)) * 25  # 6400 bytes = 2 chunks
    src.pipe.feed(data[:1000])
    assert chunks == []
    src.pipe.feed(data[1000:5000])
    src.pipe.feed(data[5000:])
    assert [len(c) for c in chunks] == [3200, 3200]
    assert b"".join(chunks) == data
    assert rec.is_recording


def test_resamples_48k_stereo(qtbot):
    rec, src, chunks, _, _ = make(48000, 2)
    n = 48000 // 5  # 200 ms of frames
    t = np.arange(n) / 48000
    mono = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    stereo = np.stack([mono, mono], axis=1).ravel()
    pcm = (stereo * 32767).astype("<i2").tobytes()
    src.pipe.feed(pcm + b"\x01")  # trailing partial frame must be held back
    assert [len(c) for c in chunks] == [3200, 3200]
    out = np.frombuffer(chunks[0], dtype="<i2").astype(np.float32) / 32768
    assert abs(float(np.sqrt(np.mean(out**2))) - 0.5 / np.sqrt(2)) < 0.02


def test_level_signal(qtbot):
    rec, src, chunks, levels, _ = make()  # keep rec alive
    src.pipe.feed(float32_to_pcm16(np.full(1600, 0.25, dtype=np.float32)))
    assert len(levels) == 1
    assert abs(levels[0] - 0.25) < 1e-3


def test_no_device_emits_failed(qtbot, monkeypatch):
    monkeypatch.setattr(QMediaDevices, "defaultAudioInput", staticmethod(QAudioDevice))
    monkeypatch.setattr(QMediaDevices, "audioInputs", staticmethod(lambda: []))
    rec = Recorder()
    errors = []
    rec.failed.connect(errors.append)
    rec.start()
    assert errors == ["No microphone found"]
    assert not rec.is_recording


def test_device_error_emits_failed(qtbot):
    rec, src, chunks, _, errors = make()
    src.fail()
    assert len(errors) == 1 and "IOError" in errors[0]
    assert not rec.is_recording
    assert src.stopped


class _ShadowEnum:
    """Mimics a PySide6 enum member that is value-equal to ``QAudio.Error`` but a
    distinct object whose ``==``/``!=`` disagree with the module-level enum.

    On PySide6 6.11.0 ``QAudioSource.error()`` returns such a shadow of
    ``QAudio.Error.NoError``: ``err == QAudio.Error.NoError`` is ``False`` even
    though ``err.value == 0``. The recorder must decide by ``.value``.
    """

    def __init__(self, value, name):
        self.value = value
        self.name = name

    def __eq__(self, other):  # never equal to the real enum (the quirk)
        return False

    def __ne__(self, other):
        return True


def test_noerror_state_change_does_not_fail(qtbot):
    """A stateChanged whose error() is NoError-by-value (shadow enum) must not
    be reported as a failure — the regression behind "Audio input error: NoError"."""
    rec, src, chunks, _, errors = make()
    src._error = _ShadowEnum(QAudio.Error.NoError.value, "NoError")
    src.stateChanged.emit(QAudio.State.IdleState)
    assert errors == []
    assert rec.is_recording


def test_shadow_enum_real_error_still_reported(qtbot):
    rec, src, chunks, _, errors = make()
    src._error = _ShadowEnum(QAudio.Error.IOError.value, "IOError")
    src.stateChanged.emit(QAudio.State.StoppedState)
    assert len(errors) == 1 and "IOError" in errors[0]
    assert not rec.is_recording


def test_stop_flushes_nothing_partial(qtbot):
    rec, src, chunks, _, _ = make()
    src.pipe.feed(b"\x01\x00" * 1000)  # less than one chunk
    rec.stop()
    assert chunks == []
    assert not rec.is_recording
    assert src.stopped


def test_unknown_device_id_fails_without_fallback(qtbot):
    rec = Recorder("no-such-device")
    errors = []
    rec.failed.connect(errors.append)
    rec.start()
    assert errors == ["Microphone not found: no-such-device"]
    assert not rec.is_recording


def test_float32_stereo_decode(qtbot):
    fmt = QAudioFormat()
    fmt.setSampleRate(16000)
    fmt.setChannelCount(2)
    fmt.setSampleFormat(QAudioFormat.SampleFormat.Float)
    holder = {}

    def factory(_f, _d):
        holder["src"] = FakeSource(fmt)
        return holder["src"]

    rec = Recorder(source_factory=factory)
    chunks = []
    rec.chunk.connect(chunks.append)
    rec.start()
    left = np.full(1600, 0.5, dtype="<f4")
    right = np.full(1600, 0.25, dtype="<f4")
    holder["src"].pipe.feed(np.stack([left, right], axis=1).tobytes())
    assert len(chunks) == 1
    out = np.frombuffer(chunks[0], dtype="<i2").astype(np.float32) / 32768
    assert abs(float(out.mean()) - 0.375) < 1e-3


def test_unsupported_sample_format_fails(qtbot):
    fmt = QAudioFormat()
    fmt.setSampleRate(16000)
    fmt.setChannelCount(1)
    rec = Recorder(source_factory=lambda _f, _d: FakeSource(fmt))  # Unknown format
    errors = []
    rec.failed.connect(errors.append)
    rec.start()
    assert errors == ["Unsupported sample format"]
    assert not rec.is_recording


def test_sync_start_error_is_reported(qtbot):
    fmt = QAudioFormat()
    fmt.setSampleRate(16000)
    fmt.setChannelCount(1)
    fmt.setSampleFormat(QAudioFormat.SampleFormat.Int16)

    def factory(_f, _d):
        src = FakeSource(fmt)
        src._error = QAudio.Error.OpenError
        return src

    rec = Recorder(source_factory=factory)
    errors = []
    rec.failed.connect(errors.append)
    rec.start()
    assert len(errors) == 1 and "OpenError" in errors[0]
    assert not rec.is_recording
