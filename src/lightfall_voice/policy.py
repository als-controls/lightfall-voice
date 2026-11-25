"""Decides whether/how a VoiceEvent is spoken. Pure logic, injectable time."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

from lightfall_voice.events import VoiceEvent
from lightfall_voice.tts.client import OutputSettings
from lightfall_voice.voices import DEFAULT_VOICE

TTL_SECONDS = 10.0
TOOL_RATE_LIMIT_SECONDS = 3.0

_LEGACY_KEYS = ("voice.daemon_host", "voice.daemon_port")

_KEYS = {
    "enabled": "voice.enabled",
    "tools": "voice.events.tools",
    "summary": "voice.events.summary",
    "permission": "voice.events.permission",
    "errors": "voice.events.errors",
    "default_voice": "voice.default_voice",
    "agent_overrides": "voice.agent_overrides",
    "speed": "voice.output.speed",
    "quality": "voice.output.quality",
    "model_dir": "voice.output.model_dir",
    "output_device": "voice.output.output_device",
    "python": "voice.output.python",
}


@dataclass(frozen=True)
class AgentOverride:
    muted: bool = False
    voice: str | None = None


@dataclass
class VoiceSettings:
    enabled: bool = True
    tools: bool = True
    summary: bool = True
    permission: bool = True
    errors: bool = True
    # Voice for agents without their own; "" means af_heart.
    default_voice: str = DEFAULT_VOICE
    agent_overrides: dict[str, AgentOverride] = field(default_factory=dict)
    speed: float = 1.0
    quality: str = "fp16"
    model_dir: str = ""
    output_device: str = ""
    python: str = ""

    @classmethod
    def from_preferences(cls, prefs) -> "VoiceSettings":
        raw_overrides = prefs.get(_KEYS["agent_overrides"], {}) or {}
        overrides = {
            name: AgentOverride(
                muted=bool(o.get("muted", False)), voice=o.get("voice") or None
            )
            for name, o in raw_overrides.items()
        }
        raw_speed = prefs.get(_KEYS["speed"], 1.0)
        try:
            speed = (
                1.0
                if raw_speed is None or isinstance(raw_speed, bool)
                else float(raw_speed)
            )
        except (TypeError, ValueError):
            speed = 1.0
        if not math.isfinite(speed):
            speed = 1.0
        speed = min(2.0, max(0.5, speed))
        raw_quality = prefs.get(_KEYS["quality"], "fp16")
        quality = raw_quality if raw_quality in ("fp16", "int8") else "fp16"
        return cls(
            enabled=bool(prefs.get(_KEYS["enabled"], True)),
            tools=bool(prefs.get(_KEYS["tools"], True)),
            summary=bool(prefs.get(_KEYS["summary"], True)),
            permission=bool(prefs.get(_KEYS["permission"], True)),
            errors=bool(prefs.get(_KEYS["errors"], True)),
            default_voice=str(prefs.get(_KEYS["default_voice"], DEFAULT_VOICE) or DEFAULT_VOICE),
            agent_overrides=overrides,
            speed=speed,
            quality=quality,
            model_dir=str(prefs.get(_KEYS["model_dir"], "") or ""),
            output_device=str(prefs.get(_KEYS["output_device"], "") or ""),
            python=str(prefs.get(_KEYS["python"], "") or ""),
        )

    def save(self, prefs) -> None:
        remove = getattr(prefs, "remove", None)
        if callable(remove):
            for key in _LEGACY_KEYS:
                remove(key)
        prefs.set(_KEYS["enabled"], self.enabled)
        prefs.set(_KEYS["tools"], self.tools)
        prefs.set(_KEYS["summary"], self.summary)
        prefs.set(_KEYS["permission"], self.permission)
        prefs.set(_KEYS["errors"], self.errors)
        prefs.set(_KEYS["default_voice"], self.default_voice)
        prefs.set(
            _KEYS["agent_overrides"],
            {
                name: {"muted": o.muted, "voice": o.voice}
                for name, o in self.agent_overrides.items()
            },
        )
        prefs.set(_KEYS["speed"], self.speed)
        prefs.set(_KEYS["quality"], self.quality)
        prefs.set(_KEYS["model_dir"], self.model_dir)
        prefs.set(_KEYS["output_device"], self.output_device)
        prefs.set(_KEYS["python"], self.python)

    def to_preferences(self, prefs) -> None:
        self.save(prefs)

    def to_output(self) -> OutputSettings:
        return OutputSettings(
            enabled=self.enabled,
            speed=self.speed,
            quality=self.quality,
            model_dir=self.model_dir,
            output_device=self.output_device,
            default_voice=self.default_voice or DEFAULT_VOICE,
            python=self.python,
        )


def muted_by_default(agent_name: str) -> bool:
    """Mute state of an agent with no override in voice.agent_overrides.

    Every session tab speaks unless muted: agent_overrides is a pure opt-out.
    (Subagents never reach the policy: they have no widget, and their tool
    calls arrive through the parent session's signals under its name.)
    """
    return False


def with_agent_voice(
    overrides: dict[str, AgentOverride], agent_name: str, voice: str | None
) -> dict[str, AgentOverride]:
    """``overrides`` with ``agent_name``'s voice set (None/"" = the default
    voice); its mute is kept. Entries restating the default are dropped."""
    out = dict(overrides)
    muted = out[agent_name].muted if agent_name in out else muted_by_default(agent_name)
    voice = voice or None
    if muted == muted_by_default(agent_name) and voice is None:
        out.pop(agent_name, None)
    else:
        out[agent_name] = AgentOverride(muted=muted, voice=voice)
    return out


def with_agent_muted(
    overrides: dict[str, AgentOverride], agent_name: str, muted: bool
) -> dict[str, AgentOverride]:
    """``overrides`` with ``agent_name``'s mute set; its voice is kept.

    An entry that would only restate the default is dropped, so the stored
    overrides stay a pure opt-out list.
    """
    out = dict(overrides)
    voice = out[agent_name].voice if agent_name in out else None
    if muted == muted_by_default(agent_name) and voice is None:
        out.pop(agent_name, None)
    else:
        out[agent_name] = AgentOverride(muted=muted, voice=voice)
    return out


class VoicePolicy:
    def __init__(self, settings: VoiceSettings) -> None:
        self.settings = settings
        self._last_tool_time: dict[tuple[str, str], float] = {}

    def decide(self, event: VoiceEvent, now: float) -> tuple[str, str | None] | None:
        s = self.settings
        if not s.enabled:
            return None

        kind_enabled = {
            "tool": s.tools,
            "summary": s.summary,
            "permission": s.permission,
            "error": s.errors,
        }.get(event.kind, False)
        if not kind_enabled:
            return None

        override = s.agent_overrides.get(event.agent_name)
        # Explicit overrides win; high-priority events (permission, error)
        # always break through mutes.
        if event.priority != "high":
            muted = (
                override.muted if override is not None
                else muted_by_default(event.agent_name)
            )
            if muted:
                return None
            if now - event.timestamp > TTL_SECONDS:
                return None
            if event.kind == "tool" and event.tool_name:
                key = (event.agent_name, event.tool_name)
                last = self._last_tool_time.get(key, float("-inf"))
                if now - last < TOOL_RATE_LIMIT_SECONDS:
                    return None
                self._last_tool_time[key] = now

        voice = (
            (override.voice if override is not None else None)
            or s.default_voice
            or DEFAULT_VOICE
        )
        return (event.text, voice)
