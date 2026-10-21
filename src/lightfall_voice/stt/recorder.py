"""Microphone capture via QtMultimedia, emitting 100 ms 16 kHz mono s16le chunks."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from PySide6.QtCore import QObject, Signal
from PySide6.QtMultimedia import QAudio, QAudioFormat, QAudioSource, QMediaDevices

from lightfall_voice.audio import (
    CHUNK_SAMPLES,
    RATE,
    float32_to_pcm16,
    pcm16_to_float32,
    resample,
    rms,
)

CHUNK_BYTES = CHUNK_SAMPLES * 2

_SAMPLE_WIDTHS = {
    QAudioFormat.SampleFormat.UInt8: ("u1", 1),
    QAudioFormat.SampleFormat.Int16: ("<i2", 2),
    QAudioFormat.SampleFormat.Int32: ("<i4", 4),
    QAudioFormat.SampleFormat.Float: ("<f4", 4),
}


def _requested_format() -> QAudioFormat:
    fmt = QAudioFormat()
    fmt.setSampleRate(RATE)
    fmt.setChannelCount(1)
    fmt.setSampleFormat(QAudioFormat.SampleFormat.Int16)
    return fmt


def _is_native(fmt: QAudioFormat) -> bool:
    return (
        fmt.sampleRate() == RATE
        and fmt.channelCount() == 1
        and fmt.sampleFormat() == QAudioFormat.SampleFormat.Int16
    )


def _to_mono_float(data: bytes, fmt: QAudioFormat) -> np.ndarray:
    """Decode interleaved samples in ``fmt`` to mono float32 in [-1, 1]."""
    code, _ = _SAMPLE_WIDTHS[fmt.sampleFormat()]
    a = np.frombuffer(data, dtype=code)
    if code == "u1":
        a = (a.astype(np.float32) - 128.0) / 128.0
    elif code == "<i2":
        a = a.astype(np.float32) / 32768.0
    elif code == "<i4":
        a = a.astype(np.float32) / 2147483648.0
    else:
        a = a.astype(np.float32)
    ch = max(1, fmt.channelCount())
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)
    return a


class Recorder(QObject):
    """Capture from an input device and emit fixed-size 16 kHz mono PCM chunks."""

    chunk = Signal(bytes)
    level = Signal(float)
    failed = Signal(str)

    def __init__(self, device_id: str = "", *, source_factory: Callable | None = None, parent=None):
        super().__init__(parent)
        self._device_id = device_id
        self._factory = source_factory
        self._source = None
        self._io = None
        self._fmt: QAudioFormat | None = None
        self._native = True
        self._raw = bytearray()  # undecoded input (partial frames)
        self._out = bytearray()  # converted 16 kHz mono s16le awaiting chunking
        self._recording = False

    @property
    def is_recording(self) -> bool:
        return self._recording

    def _find_device(self):
        if self._device_id:
            for dev in QMediaDevices.audioInputs():
                if bytes(dev.id()).decode("utf-8", "replace") == self._device_id:
                    return dev
        return QMediaDevices.defaultAudioInput()

    def start(self) -> None:
        if self._recording:
            return
        device = None
        fmt = _requested_format()
        if self._factory is None:
            device = self._find_device()
            if device.isNull():
                self.failed.emit("No microphone found")
                return
            if not device.isFormatSupported(fmt):
                fmt = device.preferredFormat()
        try:
            self._source = self._factory(fmt, device) if self._factory else QAudioSource(device, fmt)
            self._io = self._source.start()
        except Exception as exc:  # noqa: BLE001 - surface any backend failure
            self._teardown()
            self.failed.emit(f"Could not open microphone: {exc}")
            return
        if self._io is None:
            err = self._source.error()
            self._teardown()
            self.failed.emit(f"Could not open microphone: {getattr(err, 'name', err)}")
            return
        actual = self._source.format() if callable(getattr(self._source, "format", None)) else fmt
        self._fmt = actual
        self._native = _is_native(actual)
        self._raw.clear()
        self._out.clear()
        self._recording = True
        self._source.stateChanged.connect(self._on_state)
        self._io.readyRead.connect(self._on_ready)
        self._on_ready()  # pick up anything already buffered

    def stop(self) -> None:
        if not self._recording:
            return
        self._recording = False
        self._teardown()

    def _teardown(self) -> None:
        source, io = self._source, self._io
        self._source = self._io = None
        self._raw.clear()
        self._out.clear()
        if io is not None:
            try:
                io.readyRead.disconnect(self._on_ready)
            except (RuntimeError, TypeError):
                pass
        if source is not None:
            try:
                source.stateChanged.disconnect(self._on_state)
            except (RuntimeError, TypeError):
                pass
            source.stop()

    def _on_state(self, _state) -> None:
        if not self._recording or self._source is None:
            return
        err = self._source.error()
        if err != QAudio.Error.NoError:
            self._recording = False
            self._teardown()
            self.failed.emit(f"Audio input error: {getattr(err, 'name', err)}")

    def _on_ready(self) -> None:
        if not self._recording or self._io is None:
            return
        data = bytes(self._io.readAll())
        if not data:
            return
        if self._native:
            self._out += data
        else:
            self._raw += data
            _, width = _SAMPLE_WIDTHS[self._fmt.sampleFormat()]
            frame = width * max(1, self._fmt.channelCount())
            usable = len(self._raw) - len(self._raw) % frame
            if usable:
                block = bytes(self._raw[:usable])
                del self._raw[:usable]
                mono = _to_mono_float(block, self._fmt)
                self._out += float32_to_pcm16(resample(mono, self._fmt.sampleRate()))
        while self._recording and len(self._out) >= CHUNK_BYTES:
            piece = bytes(self._out[:CHUNK_BYTES])
            del self._out[:CHUNK_BYTES]
            self.level.emit(rms(pcm16_to_float32(piece)))
            self.chunk.emit(piece)
