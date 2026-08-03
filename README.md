# lightfall-voice

Voice notifications plugin for Lightfall. Speaks Claude agent events through
the local Kokoro voice daemon from `~/.claude/hooks/voice_notifications`
(JSON line protocol, TCP `127.0.0.1:52849`).

## What it speaks

- **Permission requests** — "Permission needed to run a command" (high
  priority, non-interruptible)
- **Turn summaries** — trimmed final assistant text, or "Task complete"
- **Tool announcements** — "Reading loader", "Running a command"
  (rate-limited to one per tool per 3 s)
- **Errors** — "Agent error. Attention needed"

Subagent sessions are muted by default; their permission requests and errors
still speak, prefixed with the agent name. Low-priority events older than
10 s are dropped, never spoken late.

## Install

    pip install -e .

Registers via the `lightfall.plugins` entry point; the settings page appears
under Preferences → Plugins → Voice Notifications.

## Requirements

- The voice daemon must accept hookless direct messages
  (`voice_daemon.py` commit "Add hookless direct-message path"). The plugin
  does not start the daemon; if it is unreachable, notifications are
  silently dropped.

## Settings keys

`voice.enabled`, `voice.events.{tools,summary,permission,errors}`,
`voice.daemon_host`, `voice.daemon_port`, `voice.agent_overrides`
(`{agent_name: {"muted": bool, "voice": "af_sky"}}` — no UI yet, edit via
preferences).

## Tests

    .venv\Scripts\python -m pytest
