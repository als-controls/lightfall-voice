"""Decides whether/how a VoiceEvent is spoken. Pure logic, injectable time."""

from __future__ import annotations

from dataclasses import dataclass, field

from lightfall_voice.events import VoiceEvent

TTL_SECONDS = 10.0
TOOL_RATE_LIMIT_SECONDS = 3.0

_KEYS = {
    "enabled": "voice.enabled",
    "tools": "voice.events.tools",
    "summary": "voice.events.summary",
    "permission": "voice.events.permission",
    "errors": "voice.events.errors",
    "daemon_host": "voice.daemon_host",
    "daemon_port": "voice.daemon_port",
    "agent_overrides": "voice.agent_overrides",
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
    daemon_host: str = "127.0.0.1"
    daemon_port: int = 52849
    agent_overrides: dict[str, AgentOverride] = field(default_factory=dict)

    @classmethod
    def from_preferences(cls, prefs) -> "VoiceSettings":
        raw_overrides = prefs.get(_KEYS["agent_overrides"], {}) or {}
        overrides = {
            name: AgentOverride(
                muted=bool(o.get("muted", False)), voice=o.get("voice") or None
            )
            for name, o in raw_overrides.items()
        }
        return cls(
            enabled=bool(prefs.get(_KEYS["enabled"], True)),
            tools=bool(prefs.get(_KEYS["tools"], True)),
            summary=bool(prefs.get(_KEYS["summary"], True)),
            permission=bool(prefs.get(_KEYS["permission"], True)),
            errors=bool(prefs.get(_KEYS["errors"], True)),
            daemon_host=str(prefs.get(_KEYS["daemon_host"], "127.0.0.1")),
            daemon_port=int(prefs.get(_KEYS["daemon_port"], 52849)),
            agent_overrides=overrides,
        )

    def to_preferences(self, prefs) -> None:
        prefs.set(_KEYS["enabled"], self.enabled)
        prefs.set(_KEYS["tools"], self.tools)
        prefs.set(_KEYS["summary"], self.summary)
        prefs.set(_KEYS["permission"], self.permission)
        prefs.set(_KEYS["errors"], self.errors)
        prefs.set(_KEYS["daemon_host"], self.daemon_host)
        prefs.set(_KEYS["daemon_port"], self.daemon_port)
        prefs.set(
            _KEYS["agent_overrides"],
            {
                name: {"muted": o.muted, "voice": o.voice}
                for name, o in self.agent_overrides.items()
            },
        )


def muted_by_default(agent_name: str) -> bool:
    """Mute state of an agent with no override in voice.agent_overrides.

    Every session tab speaks unless muted: agent_overrides is a pure opt-out.
    (Subagents never reach the policy: they have no widget, and their tool
    calls arrive through the parent session's signals under its name.)
    """
    return False


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

        voice = override.voice if override is not None else None
        return (event.text, voice)
