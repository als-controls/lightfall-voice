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
    StreamResampler,
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


def _is_error(err) -> bool:
    """True if ``err`` is a real QAudio error (not ``NoError``).

    Compare by numeric ``.value`` rather than by enum identity/equality: on some
    PySide6 builds (observed on 6.11.0) the ``QAudio.Error`` instance returned by
    ``QAudioSource.error()`` belongs to a *different* enum class object than the
    module-level ``QAudio.Error.NoError`` attribute, so ``err == QAudio.Error.NoError``
    is ``False`` — and ``!=`` is ``True`` — even when both are ``NoError`` (value 0).
    That false positive made the recorder tear down and report "Audio input error:
    NoError" on the first normal ``stateChanged`` to ``IdleState``.
    """
    return getattr(err, "value", err) != QAudio.Error.NoError.value


def _err_name(err) -> str:
    return getattr(err, "name", str(err))


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
        self._rs: StreamResampler | None = None
        self._raw = bytearray()  # undecoded input (partial frames)
        self._out = bytearray()  # converted 16 kHz mono s16le awaiting chunking
        self._recording = False
        self._connected = False

    @property
    def is_recording(self) -> bool:
        return self._recording

    def _find_device(self):
        if self._device_id:
            for dev in QMediaDevices.audioInputs():
                if bytes(dev.id()).decode("utf-8", "replace") == self._device_id:
                    return dev
            return None
        return QMediaDevices.defaultAudioInput()

    def start(self) -> None:
        if self._recording:
            return
        device = None
        fmt = _requested_format()
        if self._factory is None:
            device = self._find_device()
            if device is None:
                self.failed.emit(f"Microphone not found: {self._device_id}")
                return
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
            self.failed.emit(f"Could not open microphone: {_err_name(err)}")
            return
        actual = self._source.format() if callable(getattr(self._source, "format", None)) else fmt
        if actual.sampleFormat() not in _SAMPLE_WIDTHS:
            self._teardown()
            self.failed.emit("Unsupported sample format")
            return
        self._fmt = actual
        self._native = _is_native(actual)
        self._rs = None if self._native else StreamResampler(actual.sampleRate())
        self._raw.clear()
        self._out.clear()
        self._recording = True
        self._source.stateChanged.connect(self._on_state)
        self._io.readyRead.connect(self._on_ready)
        self._connected = True
        self._on_state(None)  # an error raised synchronously by start()
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
        connected, self._connected = self._connected, False
        if connected:
            for sig, slot in ((io.readyRead, self._on_ready), (source.stateChanged, self._on_state)):
                try:
                    sig.disconnect(slot)
                except (RuntimeError, TypeError):
                    pass
        if source is not None:
            source.stop()
            if hasattr(source, "deleteLater"):
                source.deleteLater()  # never delete inside its own signal

    def _on_state(self, _state) -> None:
        if not self._recording or self._source is None:
            return
        err = self._source.error()
        if _is_error(err):
            self._recording = False
            self._teardown()
            self.failed.emit(f"Audio input error: {_err_name(err)}")

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
                self._out += float32_to_pcm16(self._rs.process(mono))
        while self._recording and len(self._out) >= CHUNK_BYTES:
            piece = bytes(self._out[:CHUNK_BYTES])
            del self._out[:CHUNK_BYTES]
            self.level.emit(rms(pcm16_to_float32(piece)))
            self.chunk.emit(piece)
