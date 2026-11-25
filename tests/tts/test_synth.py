"""Tests for the TTS sentence splitter and synthesizers."""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from lightfall_voice.tts_service.synth import (
    RATE,
    FakeSynth,
    KokoroSynth,
    split_sentences,
)


def test_split_sentences_basic():
    text = "First sentence. Second one! Is this third? Fourth; fifth\nsixth"
    assert split_sentences(text) == [
        "First sentence.",
        "Second one!",
        "Is this third?",
        "Fourth;",
        "fifth",
        "sixth",
    ]


def test_split_keeps_punctuation():
    parts = split_sentences("Hello. World! Ready? Go; now.")
    assert parts == ["Hello.", "World!", "Ready?", "Go;", "now."]


def test_split_long_sentence_at_comma():
    clause1 = ("alpha " * 35).strip()
    clause2 = ("beta " * 50).strip()
    long_text = f"{clause1}, {clause2}."
    assert len(long_text) > 400
    parts = split_sentences(long_text)
    assert len(parts) == 2
    assert all(len(p) <= 400 for p in parts)
    assert parts[0] == f"{clause1},"
    assert parts[1] == f"{clause2}."

    no_comma = ("gamma " * 80).strip() + "."
    assert len(no_comma) > 400
    space_parts = split_sentences(no_comma)
    assert len(space_parts) == 2
    assert all(len(p) <= 400 for p in space_parts)


def test_split_drops_empty():
    assert split_sentences("") == []
    assert split_sentences("   \n\n  \t  ") == []
    assert split_sentences("  Hello.  \n\n   World!   \n  ") == ["Hello.", "World!"]


def test_fake_synth_length_and_dtype():
    synth = FakeSynth()
    assert synth.RATE == 24000
    assert RATE == 24000
    assert synth.voices() == {"af_heart", "bm_george"}

    audio = synth.synthesize("one two three", "af_heart", 1.0, "en-us")
    assert isinstance(audio, np.ndarray)
    assert audio.dtype == np.float32
    assert audio.ndim == 1
    assert len(audio) == 3 * int(0.05 * 24000)
    assert np.isclose( float(np.max(np.abs(audio))), 0.3, atol=1e-2)


def test_kokoro_synth_with_fake_module(monkeypatch):
    calls: list[dict] = []

    class FakeKokoro:
        def __init__(self, session, voices_path):
            self.session = session
            self.voices_path = voices_path

        @classmethod
        def from_session(cls, session, voices_path):
            return cls(session, voices_path)

        def get_voices(self):
            return ["af_heart", "bm_george", "af_sky"]

        def create(self, text, *, voice, speed, lang):
            calls.append({"text": text, "voice": voice, "speed": speed, "lang": lang})
            return np.full(2400, 0.25, dtype=np.float64), 24000

    class FakeOptions:
        def __init__(self):
            self.intra_op_num_threads = 0
            self.config: dict[str, str] = {}

        def add_session_config_entry(self, key, value):
            self.config[key] = value

    class FakeSession:
        def __init__(self, path, sess_options=None, providers=None):
            self.path, self.options, self.providers = path, sess_options, providers

    mod = types.ModuleType("kokoro_onnx")
    mod.Kokoro = FakeKokoro
    monkeypatch.setitem(sys.modules, "kokoro_onnx", mod)
    rt = types.ModuleType("onnxruntime")
    rt.SessionOptions = FakeOptions
    rt.InferenceSession = FakeSession
    monkeypatch.setitem(sys.modules, "onnxruntime", rt)

    synth = KokoroSynth("model.onnx", "voices.bin")
    session = synth._kokoro.session
    assert session.path == "model.onnx"
    assert session.providers == ["CPUExecutionProvider"]
    assert session.options.config == {"session.intra_op.allow_spinning": "0"}
    assert 1 <= session.options.intra_op_num_threads <= 8
    assert synth.RATE == 24000
    assert synth.voices() == {"af_heart", "bm_george", "af_sky"}
    out = synth.synthesize("Hello.", "af_heart", 1.2, "en-us")
    assert out.dtype == np.float32
    assert out.shape == (2400,)
    assert calls == [{"text": "Hello.", "voice": "af_heart", "speed": 1.2, "lang": "en-us"}]


def test_kokoro_real():
    pytest.importorskip("kokoro_onnx")
    model_dir = os.environ.get("LIGHTFALL_TTS_MODEL_DIR", "")
    if not model_dir:
        pytest.skip("LIGHTFALL_TTS_MODEL_DIR not set")
    root = Path(model_dir)
    from lightfall_voice.tts_service.models import MODEL_FILES

    model_path = root / MODEL_FILES["fp16"][0]
    voices_path = root / MODEL_FILES["voices"][0]
    if not (model_path.is_file() and voices_path.is_file()):
        pytest.skip("Kokoro model files not found in LIGHTFALL_TTS_MODEL_DIR")
    synth = KokoroSynth(str(model_path), str(voices_path))
    audio = synth.synthesize("Hello.", "af_heart", 1.0, "en-us")
    assert audio.dtype == np.float32
    assert len(audio) > int(0.3 * KokoroSynth.RATE)
