# lightfall-voice

Voice notifications and push-to-talk voice input plugin for Lightfall. Speaks
Claude agent events through a local `tts` service (`kokoro-onnx`) and
transcribes microphone utterances through a local `stt` service
(`faster-whisper`), both launched on demand by Lightfall's `ServiceManager`.

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
the voice for every agent (default `af_heart`, "Heart (default)", the voice
ascribe-link uses); the Agents list picks a voice per agent. Both combos are
editable, so any Kokoro voice id can be typed.

Agents other than the main `lightfall` assistant are named in
those phrases. Low-priority events older than 10 s are dropped, never
spoken late.

## Install

    pip install lightfall-voice[tts-service]   # spoken notifications (kokoro-onnx)
    pip install lightfall-voice[stt-service]   # voice input (faster-whisper, CPU)
    pip install lightfall-voice[voice]         # both speech output and voice input
    pip install lightfall-voice[gpu]           # voice input with CUDA 12 (cuBLAS + cuDNN wheels)

For development, from a clone: `pip install -e ".[dev]"`.

Registers via the `lightfall.plugins` entry point; the settings page appears
under Preferences → Plugins → Voice Notifications.

## Voice output (text to speech)

Spoken notifications run through a local `tts` service (`kokoro-onnx`) that Lightfall
starts when `voice.enabled` is on. The base install never imports `kokoro_onnx` or
`onnxruntime`; if the service dependencies live in a separate environment, point
`voice.output.python` at that interpreter.

On first use, the `tts` service downloads the `fp16` Kokoro model (~205 MB total:
`kokoro-v1.0.fp16.onnx` + `voices-v1.0.bin`) into the user cache directory
(`platformdirs.user_cache_dir("lightfall-voice")/kokoro`) and verifies SHA-256
checksums.

Offline machines: set `voice.output.model_dir` to a folder holding:
- `kokoro-v1.0.fp16.onnx` (for `fp16`, "Higher (fp16)", the default) or
  `kokoro-v1.0.int8.onnx` (for `int8`, "Standard (int8)")
- `voices-v1.0.bin`

Both files are downloadable from
https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.0. Nothing is
downloaded when `voice.output.model_dir` is set.

| Setting | Key | Default |
|---|---|---|
| Enable voice notifications | `voice.enabled` | on |
| Speed | `voice.output.speed` | `1.0` (`0.5`–`2.0`, step `0.1`) |
| Quality | `voice.output.quality` | `fp16` (`fp16` "Higher (fp16)", `int8` "Standard (int8)") |
| Output device | `voice.output.output_device` | system default |
| Output model directory | `voice.output.model_dir` | empty (`platformdirs.user_cache_dir("lightfall-voice")/kokoro`) |
| Output Python interpreter | `voice.output.python` | empty (Lightfall's own) |

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

## Settings keys

`voice.enabled`, `voice.events.{tools,summary,permission,errors}`,
`voice.default_voice`, `voice.agent_overrides`
(`{agent_name: {"muted": bool, "voice": "af_sky"}}`, set from the Agents list),
`voice.output.{speed,quality,model_dir,output_device,python}` (above), and
`voice.input.{enabled,hotkey,model,language,silence_s,device,input_device,model_dir,python}`
(above).

This machine's voice output keys (`voice.output.model_dir`,
`voice.output.output_device`, `voice.output.python`) and voice input keys
(`voice.input.device`, `voice.input.input_device`, `voice.input.model_dir`,
`voice.input.python`) belong to this console and are set at This console; every
other `voice.*` key is a per-user setting. Only edited keys are saved.

## Tests

    .venv\Scripts\python -m pytest
