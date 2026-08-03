# lightfall-voice — Voice notifications plugin for Lightfall

**Date:** 2026-08-03
**Status:** Approved design

## Purpose

Spoken notifications for Lightfall's embedded Claude agent sessions, replacing the
out-of-process Claude Code hook scripts (`~/.claude/hooks/voice_notifications`) for
work done inside Lightfall. Ships as a standalone plugin repo; no changes to the
main `lightfall` repo.

Design goals carried over from experience with the hook-based system:

- **No stale speech.** The hook system sometimes spoke events minutes late
  (detached-process respawn chain). Here events are consumed in-process from Qt
  signals and carry a TTL; late events are dropped, never spoken.
- **Subagent noise control.** Subagents are muted by default; only their
  high-priority events (permission requests, errors) speak, name-prefixed.
- **Professional tone.** One neutral, concise phrase set. No personalities, no
  nicknames.

## Architecture

The plugin is a **thin client of the existing voice daemon** (the Kokoro TTS daemon
from the hooks repo, JSON line protocol on `127.0.0.1:52849`). It contains no TTS,
torch, kokoro, or numpy dependencies. If the daemon is unreachable, events are
dropped silently (optionally falling back to nothing in v1 — the daemon autostart
remains the hooks repo's job).

```
QtClaudeAgent signals ──> VoiceService (normalize, filter, rate-limit, TTL)
                              │
                              └──> daemon client thread ──TCP──> voice daemon (Kokoro)
```

### Integration point (verified against lightfall @ current master)

- `lightfall.claude.agent.QtClaudeAgent` exposes Qt signals: `tool_called(str, dict)`,
  `result_received(dict)`, `error_occurred(str)`, `query_completed()`,
  `permission_requested(str, str, dict)`, and `task_started/task_progress/task_finished`
  for in-session subagents (`agent.py:201-222`).
- Each session is a `ClaudeAssistantWidget` (`lightfall.claude.widget`) holding
  `.agent` and `.spec` (agent name). Sessions register with
  `lightfall.agents.bus.AgentBus`, which emits `agents_changed` on register/unregister.

**Discovery:** on `AgentBus.agents_changed` (plus once at startup), scan
`QApplication.allWidgets()` for `ClaudeAssistantWidget` instances and connect to any
`.agent` not already tracked (identity via weak refs; disconnect handled by Qt object
destruction). This keeps the plugin fully external. If lightfall later grows a
first-class `agent_created` signal, swap discovery to it.

## Components (src/lightfall_voice/)

| Module | Responsibility |
|---|---|
| `manifest.py` | `PluginManifest` with one entry: `PluginEntry(type_name="settings", name="voice", import_path="lightfall_voice.settings:VoiceSettingsPlugin", preload=True)` — preload so `on_loaded()` starts the service at startup. |
| `settings.py` | `VoiceSettingsPlugin` (category `"plugins"`). Widget: master enable, per-event toggles (tool announcements / turn summaries / permission requests / errors), daemon host+port, per-agent overrides table (mute, Kokoro voice name), "Test voice" button. Persists via `PreferencesManager` under `voice.*` keys. `on_loaded()` instantiates `VoiceService` and starts discovery. Follows the vibe-plugin gotcha: disconnect widget signals and null refs on `widget.destroyed`. |
| `service.py` | `VoiceService` (QObject singleton). Discovery (above), per-agent connection bookkeeping, policy application (subagent mute, per-agent overrides), rate limiting, TTL drop, dispatch to daemon client. |
| `events.py` | `VoiceEvent(agent_name, kind, text, priority, timestamp)` dataclass + normalizers from each Qt signal. Priorities: permission/error = HIGH, tool/summary = LOW (matches daemon interrupt semantics). |
| `phrases.py` | Professional phrase set, pure functions: `tool_announcement(tool_name, tool_input)` ("Reading configuration file", "Running a command"), `permission_phrase(tool_name)` ("Permission needed to run a command"), `completion_phrase(result)` (short summary of final text, ≤200 chars, else "Task complete"), `error_phrase(...)` ("Agent error — attention needed"). No personality layer. |
| `daemon_client.py` | Worker-thread TCP client speaking the daemon's JSON line protocol: `{"action": "speak", "args": {"message": ..., "voice": ...}}` fire-and-forget with short connect timeout; never blocks the GUI thread; drops on failure. The daemon already honors `args.voice`; no priority field is needed — the daemon derives HIGH priority from the words "permission"/"attention", which the phrase set includes deliberately. Requires one small hooks-repo patch: a direct-message branch in `handle_speak_request` (hookless requests are otherwise skipped by `should_announce`). |

## Event policy

- **Primary agent** (`spec.name == "lightfall"` / main assistant): all enabled event
  kinds speak.
- **Other agent sessions and in-session subagents (`task_*` signals):** muted by
  default; HIGH-priority events (permission, error) still speak, prefixed with the
  agent name ("Alignment agent needs permission to run a command"). Per-agent
  settings can unmute or assign a distinct voice.
- **Rate limiting:** tool announcements limited to one per tool per 3 s per agent.
- **TTL:** LOW-priority events older than 10 s at dispatch time are dropped.
  HIGH-priority events always speak.
- **Turn summary:** on `result_received`, speak a trimmed final-message summary;
  on `query_completed` with nothing meaningful, stay silent (no filler "Done").

## Error handling

- Daemon down → silent drop (log at debug). No autostart from the plugin in v1.
- Discovery scan failures / degraded widgets (no `.agent` attr) → skip quietly.
- All signal handlers wrapped so an exception can never propagate into the agent's
  signal emission path.

## Packaging & repo

- `~/PycharmProjects/ncs/lightfall-voice`, `master` branch, src-layout.
- `pyproject.toml`: hatchling + hatch-vcs (`fallback-version = "0.0.0"`),
  `requires-python >= 3.11`, deps: `lightfall` only; dev extras: `pytest`, `pytest-qt`.
- Entry point: `[project.entry-points."lightfall.plugins"] lightfall_voice = "lightfall_voice.manifest:manifest"`.
- README.md, Sphinx-less for v1 (README documents settings keys and daemon protocol).

## Testing

pytest + pytest-qt, no audio, no daemon dependency:

- Fake daemon: threaded TCP server fixture capturing JSON lines; assert payloads.
- Unit tests: phrase selection per tool/event; TTL drop; rate limiting; subagent
  mute policy incl. HIGH-priority passthrough and name prefixing; per-agent
  overrides; settings round-trip through a stubbed `PreferencesManager`.
- Signal-path test: stub QObject with the same signal signatures as `QtClaudeAgent`,
  drive signals, assert daemon-client dispatches.

## Out of scope (v1)

- Daemon autostart and in-process Kokoro. (The one hooks-repo change v1 does make
  is the direct-message daemon patch above.)
- Per-agent overrides UI (table) — persistence and policy support ship in v1;
  the table widget is a follow-up (overrides editable via preferences meanwhile).
- Core lightfall changes (an `agent_created` signal would be a nice-to-have later).
- Personalities / configurable phrase packs.
