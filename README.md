# lightfall-voice

Voice notifications plugin for Lightfall. Speaks Claude agent events through
the local Kokoro voice daemon from `~/.claude/hooks/voice_notifications`
(JSON line protocol, TCP `127.0.0.1:52849`).

## What it speaks

- **Permission requests** — "Permission needed to run a command" (high
  priority, non-interruptible)
- **Turn summaries** — trimmed final assistant text, or "Task complete"
- **Tool announcements** — "Reading loader", "Running a command"; MCP tools
  from their name ("Reading device" for `lightfall_read_device`). TodoWrite,
  Task, Agent and ToolSearch stay silent. Rate-limited to one per tool per 3 s
- **Errors** — "Agent error. Attention needed"

Every session tab speaks by default. Mute the visible tab with the speaker
button in the Assistant panel's title bar, or any agent under **Preferences →
Voice Notifications → Agents**; a muted agent's permission requests and
errors still speak.

Voices: **Preferences → Voice Notifications → Voice notifications → Default voice** sets
the voice for every agent (default `af_heart`, the voice ascribe-link uses);
the Agents list picks a voice per agent. Both combos are editable, so any
Kokoro voice id can be typed. An empty default leaves the choice to the
daemon's `config.json` (`kokoro_voice`).

Agents other than the main `lightfall` assistant are named in
those phrases. Low-priority events older than 10 s are dropped, never
spoken late.

## Install

    pip install lightfall-voice

For development, from a clone: `pip install -e ".[dev]"`.

Registers via the `lightfall.plugins` entry point; the settings page appears
under Preferences → Plugins → Voice Notifications.

## Voice input (speech to text)

Push-to-talk dictation into Claude agent tabs. A mic button sits next to each agent's input
box, and a hotkey (default `F9`) starts and stops recording on the visible agent tab. Esc
cancels. The recording goes to a local `stt` service (faster-whisper) that Lightfall launches
on demand; the transcript is submitted as a prompt (queued while the agent is busy).

Install the service dependencies into the interpreter that will run it:

    pip install "lightfall-voice[stt-service]"   # CPU
    pip install "lightfall-voice[gpu]"           # CUDA 12 (cuBLAS + cuDNN wheels)

The base install never imports faster-whisper. If the service dependencies live in a
separate environment, point `Python interpreter` at it.

Settings (Preferences -> Plugins -> Voice Notifications; keys `voice.input.*`). Device, Input device,
Model directory and Python interpreter are set at This console ("Voice input on this machine");
the rest are per user ("Voice input"):

| Setting | Key | Default |
|---|---|---|
| Enable voice input | `enabled` | off |
| Model directory | `model_dir` | empty (Hugging Face cache; downloads on first use) |
| Hotkey | `hotkey` | `F9` |
| Model | `model` | `small` (`base`, `small`, `medium`, `large-v3`, `distil-large-v3`) |
| Device | `device` | `auto` (`auto`, `gpu`, `cpu`; auto tries CUDA, then falls back to CPU) |
| Input device | `input_device` | system default |
| Silence | `silence_s` | 1.0 s of trailing silence ends the utterance |
| Language | `language` | `en` (empty: auto-detect) |
| Python interpreter | `python` | empty (Lightfall's own) |

Offline use: set `Model directory` to a folder holding a converted CTranslate2 model (for
example a `faster-whisper-small` snapshot); nothing is downloaded when the model is found
there.

While you dictate, spoken notifications are held and released afterwards.

## Requirements

- The voice daemon must accept hookless direct messages
  (`voice_daemon.py` commit "Add hookless direct-message path"). The plugin
  does not start the daemon; if it is unreachable, notifications are
  silently dropped.

## Settings keys

`voice.enabled`, `voice.events.{tools,summary,permission,errors}`,
`voice.daemon_host`, `voice.daemon_port`, `voice.default_voice`, `voice.input.*`
(above), `voice.agent_overrides` (`{agent_name: {"muted": bool, "voice": "af_sky"}}`,
set from the Agents list).

`voice.daemon_host`, `voice.daemon_port` and the machine's voice input keys
(`device`, `input_device`, `model_dir`, `python`) belong to this console and are
set at This console; every other key is a per-user setting. Only edited keys
are saved; a blank daemon host means the default `127.0.0.1`.

## Tests

    .venv\Scripts\python -m pytest
