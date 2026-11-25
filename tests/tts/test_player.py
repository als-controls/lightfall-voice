from __future__ import annotations

import numpy as np
from PySide6.QtCore import QBuffer, QIODevice, QObject, Signal
from PySide6.QtMultimedia import QAudio, QAudioDevice, QAudioFormat, QMediaDevices

from lightfall_voice.audio import float32_to_pcm16
from lightfall_voice.tts.player import Player


class FakeSink(QObject):
    stateChanged = Signal(object)

    def __init__(self, fmt: QAudioFormat, bytes_free: int = 65536):
        super().__init__()
        self._fmt = fmt
        self.buffer = QBuffer()
        self.buffer.open(QIODevice.OpenModeFlag.WriteOnly)
        self.bytes_free = bytes_free
        self._error = QAudio.Error.NoError
        self.stopped = False

    def format(self) -> QAudioFormat:
        return self._fmt

    def start(self) -> QBuffer:
        return self.buffer

    def stop(self) -> None:
        self.stopped = True

    def bytesFree(self) -> int:
        return self.bytes_free

    def error(self):
        return self._error

    def written(self) -> bytes:
        return bytes(self.buffer.data())

    def fail(self) -> None:
        self._error = QAudio.Error.IOError
        self.stateChanged.emit(QAudio.State.StoppedState)


def make(rate: int = 24000, channels: int = 1, sample_format=QAudioFormat.SampleFormat.Int16, bytes_free: int = 65536):
    fmt = QAudioFormat()
    fmt.setSampleRate(rate)
    fmt.setChannelCount(channels)
    fmt.setSampleFormat(sample_format)
    holder: dict[str, FakeSink] = {}

    def factory(_fmt, _dev):
        holder["sink"] = FakeSink(fmt, bytes_free=bytes_free)
        return holder["sink"]

    player = Player(sink_factory=factory)
    finished: list[str] = []
    errors: list[str] = []
    player.finished.connect(finished.append)
    player.failed.connect(errors.append)
    player.begin("s1", 24000, 1)
    return player, holder["sink"], finished, errors


def test_plays_chunks_in_order(qtbot):
    player, sink, finished, errors = make(bytes_free=64)
    c1 = b"\x01\x00" * 25
    c2 = b"\x02\x00" * 25
    c3 = b"\x03\x00" * 25
    player.feed("s1", c1, final=False)
    player.feed("s1", c2, final=False)
    player.feed("s1", c3, final=False)
    qtbot.waitUntil(lambda: sink.written() == c1 + c2 + c3)
    assert sink.written() == c1 + c2 + c3
    assert finished == []
    assert errors == []
    assert player.is_playing
    player.stop()


def test_finished_after_final_and_drain(qtbot):
    player, sink, finished, _ = make(bytes_free=100)
    data = b"\x05\x00" * 125  # 250 bytes: takes 3 timer ticks at 100 bytes/tick
    player.feed("s1", data, final=True)
    sink.stateChanged.emit(QAudio.State.IdleState)
    assert finished == []
    qtbot.waitUntil(lambda: len(sink.written()) == len(data))
    assert finished == []
    assert player.is_playing
    sink.stateChanged.emit(QAudio.State.IdleState)
    assert finished == ["s1"]
    assert not player.is_playing
    assert sink.stopped


def test_stop_drops_buffer_no_finished(qtbot):
    player, sink, finished, _ = make(bytes_free=100)
    data = b"\x01\x00" * 500  # 1000 bytes
    player.feed("s1", data, final=True)
    player.stop()
    sink.stateChanged.emit(QAudio.State.IdleState)
    qtbot.wait(30)
    assert sink.written() == b""
    assert finished == []
    assert not player.is_playing
    assert sink.stopped


def test_ignores_other_stream_ids(qtbot):
    player, sink, finished, _ = make()
    good = b"\x01\x00" * 40
    player.feed("other", b"\xff\x7f" * 40, final=True)
    player.feed("s1", good, final=True)
    qtbot.waitUntil(lambda: sink.written() == good)
    sink.stateChanged.emit(QAudio.State.IdleState)
    assert sink.written() == good
    assert finished == ["s1"]


def test_resamples_to_device_format_48k_stereo_float(qtbot):
    player, sink, _, errors = make(
        rate=48000,
        channels=2,
        sample_format=QAudioFormat.SampleFormat.Float,
    )
    n = 24000 // 5  # 200 ms at 24 kHz = 4800 mono samples
    t = np.arange(n, dtype=np.float32) / 24000
    mono = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    pcm = float32_to_pcm16(mono)
    player.feed("s1", pcm, final=True)
    expected_frames = n * 2  # 9600 stereo float32 frames
    qtbot.waitUntil(lambda: len(sink.written()) >= (expected_frames - 10) * 8)
    assert errors == []
    out = np.frombuffer(sink.written(), dtype="<f4").reshape(-1, 2)
    assert abs(len(out) - expected_frames) <= 2
    assert np.allclose(out[:, 0], out[:, 1])
    rms_out = float(np.sqrt(np.mean(out[:, 0] ** 2)))
    assert abs(rms_out - 0.5 / np.sqrt(2)) < 0.02
    player.stop()


def test_unknown_device_fails(qtbot, monkeypatch):
    monkeypatch.setattr(QMediaDevices, "audioOutputs", staticmethod(lambda: []))
    player = Player("no-such-device")
    errors: list[str] = []
    player.failed.connect(errors.append)
    player.begin("s1", 24000, 1)
    assert errors == ["Speaker not found: no-such-device"]
    assert not player.is_playing


def test_no_device_fails(qtbot, monkeypatch):
    monkeypatch.setattr(QMediaDevices, "defaultAudioOutput", staticmethod(QAudioDevice))
    monkeypatch.setattr(QMediaDevices, "audioOutputs", staticmethod(lambda: []))
    player = Player()
    errors: list[str] = []
    player.failed.connect(errors.append)
    player.begin("s1", 24000, 1)
    assert errors == ["No audio output found"]
    assert not player.is_playing


def test_sink_error_fails(qtbot):
    player, sink, finished, errors = make()
    sink.fail()
    assert len(errors) == 1 and "IOError" in errors[0]
    assert finished == []
    assert not player.is_playing
    assert sink.stopped


class _ShadowEnum:
    """Mimics PySide6 6.11.0 shadow enum where ``==`` disagrees with ``.value``."""

    def __init__(self, value, name):
        self.value = value
        self.name = name

    def __eq__(self, other):
        return False

    def __ne__(self, other):
        return True


def test_noerror_state_change_does_not_fail(qtbot):
    player, sink, finished, errors = make()
    sink._error = _ShadowEnum(QAudio.Error.NoError.value, "NoError")
    sink.stateChanged.emit(_ShadowEnum(QAudio.State.IdleState.value, "IdleState"))
    assert errors == []
    assert finished == []
    assert player.is_playing
    player.stop()


def test_shadow_enum_real_error_still_reported(qtbot):
    player, sink, _, errors = make()
    sink._error = _ShadowEnum(QAudio.Error.IOError.value, "IOError")
    sink.stateChanged.emit(QAudio.State.StoppedState)
    assert len(errors) == 1 and "IOError" in errors[0]
    assert not player.is_playing


def test_unsupported_sample_format_fails(qtbot):
    fmt = QAudioFormat()
    fmt.setSampleRate(24000)
    fmt.setChannelCount(1)
    player = Player(sink_factory=lambda _f, _d: FakeSink(fmt))
    errors: list[str] = []
    player.failed.connect(errors.append)
    player.begin("s1", 24000, 1)
    assert errors == ["Unsupported sample format"]
    assert not player.is_playing

