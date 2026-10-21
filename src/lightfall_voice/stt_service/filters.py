"""Junk-transcript filter (Whisper hallucinations on near-silence)."""

from __future__ import annotations

import re

_BLOCKLIST = frozenset(
    {"", "you", "thankyou", "thanks", "thanksforwatching", "bye", "okay", "ok", "uh", "um", "hmm", "so"}
)
_STRIP = re.compile(r"[\W_]+", re.UNICODE)


def _normalise(text: str) -> str:
    return _STRIP.sub("", text.lower().strip())


def is_junk(text: str) -> bool:
    return _normalise(text or "") in _BLOCKLIST


def clean(text: str) -> str:
    return "" if is_junk(text) else text.strip()
