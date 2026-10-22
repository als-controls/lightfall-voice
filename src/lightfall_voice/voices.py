"""Kokoro voices offered in the preferences. The daemon accepts any Kokoro
voice id; the combos are editable, so ids missing here can be typed."""

from __future__ import annotations

# The ascribe-link default, so Lightfall and ascribe-link sound alike.
DEFAULT_VOICE = "af_heart"

KOKORO_VOICES: tuple[tuple[str, str], ...] = (
    ("af_heart", "Heart (American female, warm)"),
    ("af_bella", "Bella (American female, expressive)"),
    ("af_jessica", "Jessica (American female, articulate)"),
    ("af_nicole", "Nicole (American female, friendly)"),
    ("af_sarah", "Sarah (American female, professional)"),
    ("af_sky", "Sky (American female)"),
    ("am_adam", "Adam (American male, neutral)"),
    ("am_michael", "Michael (American male, authoritative)"),
    ("bf_emma", "Emma (British female)"),
    ("bm_george", "George (British male)"),
)
