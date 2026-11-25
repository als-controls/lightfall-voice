"""Speaker playback via QtMultimedia: push-mode QAudioSink with 20 ms buffered writes."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtMultimedia import QAudio, QAudioFormat, QAudioSink, QMediaDevices

from lightfall_voice.audio import (
    StreamResampler,
    float32_to_pcm16,
    pcm16_to_float32,
)

TICK_MS = 20

_SAMPLE_WIDTHS = {
    QAudioFormat.SampleFormat.UInt8: ("u1", 1),
    QAudioFormat.SampleFormat.Int16: ("<i2", 2),
    QAudioFormat.SampleFormat.Int32: ("<i4", 4),
    QAudioFormat.SampleFormat.Float: ("<f4", 4),
}


def _requested_format(rate: int, channels: int) -> QAudioFormat:
    fmt = QAudioFormat()
    fmt.setSampleRate(rate)
    fmt.setChannelCount(channels)
    fmt.setSampleFormat(QAudioFormat.SampleFormat.Int16)
    return fmt


def _is_error(err) -> bool:
    """True if ``err`` is a real QAudio error (not ``NoError``); compare by ``.value``."""
    return getattr(err, "value", err) != QAudio.Error.NoError.value


def _err_name(err) -> str:
    return getattr(err, "name", str(err))


def _is_idle(state) -> bool:
    return getattr(state, "value", state) == QAudio.State.IdleState.value


def _is_native(fmt: QAudioFormat, rate: int, channels: int) -> bool:
    return (
        fmt.sampleRate() == rate
        and fmt.channelCount() == channels
        and fmt.sampleFormat() == QAudioFormat.SampleFormat.Int16
    )


def _to_mono_float(data: bytes, channels: int) -> np.ndarray:
    """Decode interleaved s16le samples to mono float32 in [-1, 1]."""
    a = pcm16_to_float32(data)
    ch = max(1, channels)
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)
    return a


def _from_mono_float(mono: np.ndarray, fmt: QAudioFormat) -> bytes:
    """Duplicate mono float32 in [-1, 1] across ``fmt``'s channels and encode."""
    a = np.clip(np.asarray(mono, dtype=np.float32), -1.0, 1.0)
    ch = max(1, fmt.channelCount())
    if ch > 1:
        a = np.repeat(a, ch)
    code, _ = _SAMPLE_WIDTHS[fmt.sampleFormat()]
    if code == "<i2":
        return float32_to_pcm16(a)
    if code == "<f4":
        return a.astype("<f4").tobytes()
    if code == "<i4":
        return np.round(a * 2147483647.0).astype("<i4").tobytes()
    return np.clip(np.round(a * 128.0 + 128.0), 0.0, 255.0).astype("u1").tobytes()


class Player(QObject):
    """Push-mode audio output player for streamed speech."""

    finished = Signal(str)
    failed = Signal(str)

    def __init__(self, device_id: str = "", *, sink_factory: Callable | None = None, parent=None):
        super().__init__(parent)
        self._device_id = device_id
        self._factory = sink_factory
        self._sink = None
        self._io = None
        self._fmt: QAudioFormat | None = None
        self._in_rate = 24000
        self._in_channels = 1
        self._native = True
        self._rs: StreamResampler | None = None
        self._raw = bytearray()  # undecoded input (partial frames)
        self._buf = bytearray()  # converted device-format bytes awaiting push write
        self._stream_id: str | None = None
        self._playing = False
        self._final = False
        self._sink_idle = True
        self._connected = False
        self._timer = QTimer(self)
        self._timer.setInterval(TICK_MS)
        self._timer.timeout.connect(self._on_timer)

    @property
    def is_playing(self) -> bool:
        return self._playing

    def _find_device(self):
        if self._device_id:
            for dev in QMediaDevices.audioOutputs():
                if bytes(dev.id()).decode("utf-8", "replace") == self._device_id:
                    return dev
            return None
        return QMediaDevices.defaultAudioOutput()

    def begin(self, stream_id: str, rate: int = 24000, channels: int = 1) -> None:
        if self._playing:
            self.stop()
        device = None
        fmt = _requested_format(rate, channels)
        if self._factory is None:
            device = self._find_device()
            if device is None:
                self.failed.emit(f"Speaker not found: {self._device_id}")
                return
            if device.isNull():
                self.failed.emit("No audio output found")
                return
            if not device.isFormatSupported(fmt):
                fmt = device.preferredFormat()
        try:
            self._sink = self._factory(fmt, device) if self._factory else QAudioSink(device, fmt)
            self._io = self._sink.start()
        except Exception as exc:  # noqa: BLE001 - surface any backend failure
            self._teardown()
            self.failed.emit(f"Could not open speaker: {exc}")
            return
        if self._io is None:
            err = self._sink.error()
            self._teardown()
            self.failed.emit(f"Could not open speaker: {_err_name(err)}")
            return
        actual = self._sink.format() if callable(getattr(self._sink, "format", None)) else fmt
        if actual.sampleFormat() not in _SAMPLE_WIDTHS:
            self._teardown()
            self.failed.emit("Unsupported sample format")
            return
        self._fmt = actual
        self._in_rate = rate
        self._in_channels = channels
        self._native = _is_native(actual, rate, channels)
        self._rs = None if self._native else StreamResampler(rate, actual.sampleRate())
        self._raw.clear()
        self._buf.clear()
        self._stream_id = stream_id
        self._playing = True
        self._final = False
        self._sink_idle = True
        self._sink.stateChanged.connect(self._on_state)
        self._connected = True
        self._timer.start()
        self._on_state(None)  # an error raised synchronously by start()

    def feed(self, stream_id: str, data: bytes, final: bool = False) -> None:
        if not self._playing or stream_id != self._stream_id:
            return
        if data:
            if self._native:
                self._buf += data
            else:
                self._raw += data
                frame = 2 * max(1, self._in_channels)
                usable = len(self._raw) - len(self._raw) % frame
                if usable:
                    block = bytes(self._raw[:usable])
                    del self._raw[:usable]
                    mono = _to_mono_float(block, self._in_channels)
                    resampled = self._rs.process(mono)
                    if len(resampled):
                        self._buf += _from_mono_float(resampled, self._fmt)
        if final:
            self._final = True
            self._raw.clear()
            if not self._buf and self._sink_idle:
                self._finish()

    def stop(self) -> None:
        if not self._playing:
            return
        self._playing = False
        self._teardown()

    def _finish(self) -> None:
        sid = self._stream_id
        self._playing = False
        self._teardown()
        if sid is not None:
            self.finished.emit(sid)

    def _teardown(self) -> None:
        self._timer.stop()
        sink = self._sink
        self._sink = self._io = None
        self._stream_id = None
        self._final = False
        self._raw.clear()
        self._buf.clear()
        connected, self._connected = self._connected, False
        if connected and sink is not None:
            try:
                sink.stateChanged.disconnect(self._on_state)
            except (RuntimeError, TypeError):
                pass
        if sink is not None:
            sink.stop()
            if hasattr(sink, "deleteLater"):
                sink.deleteLater()  # never delete inside its own signal

    def _on_timer(self) -> None:
        if not self._playing or self._sink is None or self._io is None:
            return
        if self._buf:
            free = (
                self._sink.bytesFree()
                if callable(getattr(self._sink, "bytesFree", None))
                else len(self._buf)
            )
            if free > 0:
                n = min(len(self._buf), free)
                written = self._io.write(bytes(self._buf[:n]))
                if written > 0:
                    del self._buf[:written]
                    self._sink_idle = False
        elif self._final and self._sink_idle:
            self._finish()

    def _on_state(self, state) -> None:
        if not self._playing or self._sink is None:
            return
        err = self._sink.error()
        if _is_error(err):
            self._playing = False
            self._teardown()
            self.failed.emit(f"Audio output error: {_err_name(err)}")
            return
        if state is not None:
            if _is_idle(state):
                self._sink_idle = True
                if self._final and not self._buf:
                    self._finish()
            else:
                self._sink_idle = False
