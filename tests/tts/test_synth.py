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
        def __init__(self, model_path, voices_path):
            self.model_path = model_path
            self.voices_path = voices_path

        def get_voices(self):
            return ["af_heart", "bm_george", "af_sky"]

        def create(self, text, *, voice, speed, lang):
            calls.append({"text": text, "voice": voice, "speed": speed, "lang": lang})
            return np.full(2400, 0.25, dtype=np.float64), 24000

    mod = types.ModuleType("kokoro_onnx")
    mod.Kokoro = FakeKokoro
    monkeypatch.setitem(sys.modules, "kokoro_onnx", mod)

    synth = KokoroSynth("model.onnx", "voices.bin")
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
    model_path = root / "kokoro-v1.0.onnx"
    voices_path = root / "voices-v1.0.bin"
    if not (model_path.is_file() and voices_path.is_file()):
        pytest.skip("Kokoro model files not found in LIGHTFALL_TTS_MODEL_DIR")
    synth = KokoroSynth(str(model_path), str(voices_path))
    audio = synth.synthesize("Hello.", "af_heart", 1.0, "en-us")
    assert audio.dtype == np.float32
    assert len(audio) > int(0.3 * KokoroSynth.RATE)
