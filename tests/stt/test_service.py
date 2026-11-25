"""The stt ServiceHost, in process, against a real local broker and a fake transcriber."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import threading

import numpy as np
import pytest

from lightfall_utils.ipc.services.client import ServiceClient
from lightfall_utils.ipc.services.core import NatsCore
from lightfall_utils.ipc.services.descriptor import ServiceDescriptor
from lightfall_voice.audio import CHUNK_SAMPLES, RATE, float32_to_pcm16
from lightfall_voice.stt_service.service import INSTALL_HINT, make_host

SCOPE = "lfstt"
START = {"rate": 16000, "format": "s16le", "channels": 1}
EVENTS = ("listening", "ended", "transcript", "error")


class FakeTranscriber:
    device = "cpu"

    def __init__(self, text: str) -> None:
        self.text = text
        self.calls: list[tuple[int, str | None]] = []

    def transcribe(self, audio_16k, language):
        self.calls.append((len(audio_16k), language))
        return self.text


class Factory:
    """A transcriber_factory recording its calls; optionally blocks until released."""

    def __init__(self, text: str = "hello world", *, block: bool = False) -> None:
        self.text = text
        self.calls: list[dict] = []
        self.made: list[FakeTranscriber] = []
        self.gate = threading.Event()
        if not block:
            self.gate.set()

    def __call__(self, **cfg):
        self.calls.append(cfg)
        cfg["progress"]("loading")
        self.gate.wait(10)
        t = FakeTranscriber(self.text)
        self.made.append(t)
        return t


def _tone(seconds, amp=0.3):
    t = np.arange(int(seconds * RATE)) / RATE
    return (amp * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def _zeros(seconds):
    return np.zeros(int(seconds * RATE), dtype=np.float32)


def _chunks(audio):
    return [
        float32_to_pcm16(audio[i : i + CHUNK_SAMPLES])
        for i in range(0, len(audio), CHUNK_SAMPLES)
    ]


async def _until(pred, timeout=5.0):
    loop = asyncio.get_running_loop()
    end = loop.time() + timeout
    while not pred():
        if loop.time() > end:
            raise AssertionError("condition not met in time")
        await asyncio.sleep(0.01)


class Rig:
    def __init__(self, host, svc, cli, client):
        self.host, self.svc, self.cli, self.client = host, svc, cli, client
        self.events: list[tuple[str, dict]] = []
        for name in EVENTS:
            client.on_event(
                f"utterance.{name}", lambda ev, n=name: self.events.append((n, ev))
            )

    def names(self, sid=None):
        return [n for n, ev in self.events if sid is None or ev.get("stream_id") == sid]

    def get(self, name):
        return [ev for n, ev in self.events if n == name]

    async def ready(self):
        await _until(lambda: self.host.summary.get("status") == "ready")

    async def start(self):
        reply = await self.client.call("utterance.start", START)
        assert reply["status"] == "ok", reply
        sid = reply["stream_id"]
        self.client.adopt_stream(sid, "audio")
        return sid

    async def send(self, sid, audio, seq0=0):
        seq = seq0
        for chunk in _chunks(audio):
            if sid not in self.client.streams:
                break
            await self.client.send_chunk(sid, seq, chunk)
            seq += 1
        return seq

    async def close(self):
        await self.client.close()
        await self.host.stop()
        await self.svc.close()
        await self.cli.close()


async def _rig(broker_url, factory=None, *, session_token=None, **kw):
    kw.setdefault("heartbeat_s", 0.1)
    kw.setdefault("single_instance", False)
    kw.setdefault("initial_load_delay_s", 0.0)
    host = make_host(SCOPE, transcriber_factory=factory, **kw)
    svc = NatsCore(broker_url)
    await svc.connect()
    await host.start(svc)
    cli = NatsCore(broker_url)
    await cli.connect()
    desc = ServiceDescriptor(name="stt", scope=SCOPE, contract="1.0", launch="external")
    client = ServiceClient(
        desc,
        cli,
        scope=SCOPE,
        app_name="tester",
        app_version="1.0",
        session_token=session_token,
    )
    rig = Rig(host, svc, cli, client)
    await client.start()
    return rig


def _run(coro):
    return asyncio.run(asyncio.wait_for(coro, 30))


def test_stream_tone_silence_gives_transcript_event(broker_url):
    fac = Factory("hello world")

    async def go():
        r = await _rig(broker_url, fac)
        try:
            await r.ready()
            sid = await r.start()
            await r.send(sid, np.concatenate([_tone(1.0), _zeros(1.5)]))
            await _until(lambda: "transcript" in r.names())
            return r, sid
        finally:
            await r.close()

    r, sid = _run(go())
    assert r.names(sid) == ["listening", "ended", "transcript"]
    assert r.get("ended")[0]["reason"] == "silence"
    assert r.get("transcript")[0] == {
        **r.get("transcript")[0],
        "stream_id": sid,
        "text": "hello world",
    }
    (t,) = fac.made
    ((n, lang),) = t.calls
    assert lang == "en"
    assert n >= int(1.9 * RATE)


def test_near_silence_hallucination_sends_empty_transcript(broker_url):
    fac = Factory("Thank you.")

    async def go():
        r = await _rig(broker_url, fac)
        try:
            await r.ready()
            sid = await r.start()
            await r.send(sid, np.concatenate([_tone(1.0), _zeros(1.5)]))
            await _until(lambda: "transcript" in r.names())
            return r
        finally:
            await r.close()

    r = _run(go())
    assert r.get("transcript")[0]["text"] == ""


def test_second_start_is_busy(broker_url):
    async def go():
        r = await _rig(broker_url, Factory())
        try:
            await r.ready()
            await r.start()
            return await r.client.call("utterance.start", START)
        finally:
            await r.close()

    reply = _run(go())
    assert reply["status"] == "error"
    assert reply["code"] == "busy"


def test_bad_audio_format_rejected(broker_url):
    async def go():
        r = await _rig(broker_url, Factory())
        try:
            await r.ready()
            return await r.client.call("utterance.start", {**START, "rate": 44100})
        finally:
            await r.close()

    reply = _run(go())
    assert reply["code"] == "bad_request"


def test_client_stream_open_of_audio_refused(broker_url):
    async def go():
        r = await _rig(broker_url, Factory())
        try:
            await r.ready()
            return await r.client.call(
                "stream.open", {"kind": "audio", "params": START}
            )
        finally:
            await r.close()

    reply = _run(go())
    assert reply["status"] == "error"


def test_cancel_emits_ended_cancel_no_transcript(broker_url):
    fac = Factory()

    async def go():
        r = await _rig(broker_url, fac)
        try:
            await r.ready()
            sid = await r.start()
            await r.send(sid, _tone(0.8))
            await _until(lambda: "listening" in r.names())
            reply = await r.client.call("utterance.cancel", {"stream_id": sid})
            await _until(lambda: "ended" in r.names())
            await asyncio.sleep(0.3)
            again = await r.client.call("utterance.start", START)  # no longer busy
            return r, sid, reply, again
        finally:
            await r.close()

    r, sid, reply, again = _run(go())
    assert reply["status"] == "ok"
    assert r.names(sid) == ["listening", "ended"]
    assert r.get("ended")[0]["reason"] == "cancel"
    assert fac.made[0].calls == []
    assert again["status"] == "ok"


def test_no_speech(broker_url):
    fac = Factory()

    async def go():
        r = await _rig(broker_url, fac)
        try:
            await r.ready()
            sid = await r.start()
            await r.send(sid, _zeros(8.5))
            await _until(lambda: "ended" in r.names())
            await asyncio.sleep(0.3)
            return r, sid
        finally:
            await r.close()

    r, sid = _run(go())
    assert r.names(sid) == ["ended"]
    assert r.get("ended")[0]["reason"] == "no_speech"
    assert fac.made[0].calls == []


def _hold_listening(host):
    """Hold the stream handler at its utterance.listening emit until the returned event is
    set, so the chunks after it are still queued when a following rpc runs."""
    release = asyncio.Event()
    orig = host.emit

    async def emit(session, name, payload):
        if name == "utterance.listening":
            await release.wait()
        return await orig(session, name, payload)

    host.emit = emit
    return release


def test_stop_right_after_chunks_keeps_all_audio(broker_url):
    """Chunks still queued when utterance.stop runs are fed, not dropped."""
    n_chunks = 30
    fac = Factory("all of it")

    async def go():
        r = await _rig(broker_url, fac)
        try:
            await r.ready()
            release = _hold_listening(r.host)
            sid = await r.start()
            sent = await r.send(sid, _tone(n_chunks * CHUNK_SAMPLES / RATE))
            reply = await r.client.call("utterance.stop", {"stream_id": sid})
            release.set()
            await _until(lambda: "transcript" in r.names())
            return r, sent, reply
        finally:
            await r.close()

    r, sent, reply = _run(go())
    assert sent == n_chunks
    assert reply["status"] == "ok"
    assert r.get("ended")[0]["reason"] == "stop"
    ((n, _lang),) = fac.made[0].calls
    assert n == n_chunks * CHUNK_SAMPLES


def test_final_chunk_data_before_stop_is_kept(broker_url):
    fac = Factory()

    async def go():
        r = await _rig(broker_url, fac)
        try:
            await r.ready()
            release = _hold_listening(r.host)
            sid = await r.start()
            chunks = _chunks(_tone(1.0))
            for seq, c in enumerate(chunks[:-1]):
                await r.client.send_chunk(sid, seq, c)
            await r.client.send_chunk(sid, len(chunks) - 1, chunks[-1], final=True)
            await r.client.call("utterance.stop", {"stream_id": sid})
            release.set()
            await _until(lambda: "transcript" in r.names())
            return r, len(chunks)
        finally:
            await r.close()

    r, n_chunks = _run(go())
    assert r.get("ended")[0]["reason"] == "stop"
    ((n, _lang),) = fac.made[0].calls
    assert n == n_chunks * CHUNK_SAMPLES


def test_reload_mid_utterance_keeps_its_transcriber(broker_url):
    fac = Factory("old model")

    async def go():
        r = await _rig(broker_url, fac)
        try:
            await r.ready()
            sid = await r.start()
            await r.send(sid, _tone(1.0))
            fac.gate.clear()  # the reload blocks: no model while it runs
            await r.client.call("config.set", {"model": "base"})
            await r.client.call("utterance.stop", {"stream_id": sid})
            await _until(lambda: "transcript" in r.names() or "error" in r.names())
            fac.gate.set()
            await r.ready()
            return r
        finally:
            fac.gate.set()
            await r.close()

    r = _run(go())
    assert "error" not in r.names()
    assert r.get("transcript")[0]["text"] == "old model"


def test_summary_device_is_validated(broker_url):
    class Odd(Factory):
        def __call__(self, **cfg):
            t = super().__call__(**cfg)
            t.device = "cuda:0"
            return t

    async def go():
        r = await _rig(broker_url, Odd())
        try:
            await r.ready()
            return dict(r.host.summary)
        finally:
            await r.close()

    assert _run(go())["device"] == ""


def test_stop_finalizes_early(broker_url):
    fac = Factory("stop here")

    async def go():
        r = await _rig(broker_url, fac)
        try:
            await r.ready()
            sid = await r.start()
            await r.send(sid, _tone(1.0))
            reply = await r.client.call("utterance.stop", {"stream_id": sid})
            await _until(lambda: "transcript" in r.names())
            return r, sid, reply
        finally:
            await r.close()

    r, sid, reply = _run(go())
    assert reply["status"] == "ok"
    assert r.names(sid) == ["listening", "ended", "transcript"]
    assert r.get("ended")[0]["reason"] == "stop"
    assert r.get("transcript")[0]["text"] == "stop here"
    ((n, _lang),) = fac.made[0].calls
    assert n == RATE  # every chunk sent before stop was transcribed


def test_transcription_error_emits_error(broker_url):
    class Boom(Factory):
        def __call__(self, **cfg):
            t = super().__call__(**cfg)

            def fail(audio, language):
                raise RuntimeError("decoder exploded")

            t.transcribe = fail
            return t

    async def go():
        r = await _rig(broker_url, Boom())
        try:
            await r.ready()
            sid = await r.start()
            await r.send(sid, _tone(1.0))
            await r.client.call("utterance.stop", {"stream_id": sid})
            await _until(lambda: "error" in r.names())
            return r, sid
        finally:
            await r.close()

    r, sid = _run(go())
    assert "transcript" not in r.names()
    err = r.get("error")[0]
    assert err["stream_id"] == sid
    assert "decoder exploded" in err["message"]


def test_not_ready_rejects_start(broker_url):
    fac = Factory(block=True)

    async def go():
        r = await _rig(broker_url, fac)
        try:
            await _until(lambda: fac.calls)
            reply = await r.client.call("utterance.start", START)
            fac.gate.set()
            await r.ready()
            return reply
        finally:
            fac.gate.set()
            await r.close()

    reply = _run(go())
    assert reply["status"] == "error"
    assert reply["code"] == "unknown"
    assert "model not ready" in reply["message"]


def test_config_set_reloads_model(broker_url):
    fac = Factory()

    async def go():
        r = await _rig(broker_url, fac)
        try:
            await r.ready()
            lang = await r.client.call(
                "config.set", {"language": "de", "silence_s": 0.7}
            )
            await asyncio.sleep(0.2)
            n_after_lang = len(fac.calls)
            reply = await r.client.call(
                "config.set", {"model": "base", "device": "cpu"}
            )
            await _until(lambda: len(fac.calls) == 2)
            await r.ready()
            got = await r.client.call("config.get", {})
            return r, lang, n_after_lang, reply, got
        finally:
            await r.close()

    r, lang, n_after_lang, reply, got = _run(go())
    assert lang["status"] == "ok"
    assert n_after_lang == 1  # language / silence_s apply without a reload
    assert reply["status"] == "ok"
    assert fac.calls[1]["model"] == "base"
    assert fac.calls[1]["device"] == "cpu"
    assert got["model"] == "base"
    assert got["language"] == "de"
    assert got["silence_s"] == 0.7
    assert r.host.summary["model"] == "base"


def test_config_set_validates(broker_url):
    async def go():
        r = await _rig(broker_url, Factory())
        try:
            await r.ready()
            return [
                await r.client.call("config.set", {"device": "tpu"}),
                await r.client.call("config.set", {"silence_s": -1}),
                await r.client.call("config.set", {"model": ""}),
            ]
        finally:
            await r.close()

    for reply in _run(go()):
        assert reply["code"] == "bad_request"


def test_auth_without_stdin_token_refused(broker_url):
    async def go():
        r = await _rig(broker_url, Factory(), stdin_token="t")
        try:
            await asyncio.sleep(0.2)
            return r.client.health, r.client.last_error, r.host.sessions
        finally:
            await r.close()

    health, err, sessions = _run(go())
    assert health == "unauthorized"
    assert "token" in err
    assert sessions == {}


def test_summary_status_transitions(broker_url):
    fac = Factory(block=True)

    async def go():
        r = await _rig(broker_url, fac)
        try:
            await _until(lambda: fac.calls)
            before = (r.host.state, dict(r.host.summary))
            await _until(
                lambda: (
                    (r.client.info or {}).get("summary", {}).get("status") == "loading"
                )
            )
            fac.gate.set()
            await r.ready()
            await _until(
                lambda: (
                    (r.client.info or {}).get("summary", {}).get("status") == "ready"
                )
            )
            return before, (r.host.state, dict(r.host.summary))
        finally:
            fac.gate.set()
            await r.close()

    (s0, sum0), (s1, sum1) = _run(go())
    assert s0 == "starting"
    assert sum0 == {"model": "small", "device": "", "status": "loading", "error": ""}
    assert s1 == "ready"
    assert sum1 == {"model": "small", "device": "cpu", "status": "ready", "error": ""}


def test_missing_faster_whisper_reports_install_hint(broker_url, monkeypatch):
    from lightfall_voice.stt_service import transcriber as tr

    # The old test-only env var is ignored: production never swaps in a fake.
    monkeypatch.setenv("LIGHTFALL_STT_FAKE", "should be ignored")
    monkeypatch.setitem(sys.modules, "faster_whisper", None)
    monkeypatch.setattr(tr, "add_nvidia_dll_dirs", lambda: [])

    async def go():
        r = await _rig(broker_url)  # the real factory
        try:
            await _until(lambda: r.host.summary.get("status") == "error")
            cfg = await r.client.call("config.get", {})
            start = await r.client.call("utterance.start", START)
            return r.host.state, dict(r.host.summary), r.client.health, cfg, start
        finally:
            await r.close()

    state, summary, health, cfg, start = _run(go())
    assert state == "error"
    assert summary["status"] == "error"
    assert summary["error"] == INSTALL_HINT == "Install lightfall-voice[stt-service]"
    assert health == "up"  # the service stays up so the mic can show the reason
    assert cfg["status"] == "ok"
    assert start["code"] == "unknown"


def test_subprocess_with_fake_transcriber_flag(broker_url):
    """The hidden ``--fake-transcriber`` (test-only) makes ``python -m
    lightfall_voice.stt_service`` use a fake transcriber."""
    env = {k: v for k, v in os.environ.items() if k != "LIGHTFALL_STT_FAKE"}
    scope = "lfsttproc"
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "lightfall_voice.stt_service",
            "--scope",
            scope,
            "--nats-url",
            broker_url,
            "--token-stdin",
            "--fake-transcriber",
            "from the subprocess",
        ],
        stdin=subprocess.PIPE,
        env=env,
    )
    proc.stdin.write(b"tok123\n")
    proc.stdin.flush()

    async def go():
        cli = NatsCore(broker_url)
        await cli.connect()
        desc = ServiceDescriptor(
            name="stt", scope=scope, contract="1.0", launch="external"
        )
        client = ServiceClient(
            desc,
            cli,
            scope=scope,
            app_name="t",
            app_version="1",
            session_token="tok123",
        )
        texts: list[str] = []
        client.on_event("utterance.transcript", lambda ev: texts.append(ev["text"]))
        try:
            await client.start()
            await _until(lambda: client.health == "up", 15)
            await _until(
                lambda: (client.info or {}).get("summary", {}).get("status") == "ready",
                15,
            )
            reply = await client.call("utterance.start", START)
            sid = reply["stream_id"]
            client.adopt_stream(sid, "audio")
            for seq, chunk in enumerate(_chunks(_tone(1.0))):
                await client.send_chunk(sid, seq, chunk)
            await client.call("utterance.stop", {"stream_id": sid})
            await _until(lambda: texts, 10)
            await client.call("shutdown", {})
            return texts
        finally:
            await client.close()
            await cli.close()

    try:
        texts = asyncio.run(asyncio.wait_for(go(), 40))
        proc.stdin.close()
        assert proc.wait(15) == 0
    finally:
        if proc.poll() is None:
            proc.kill()
    assert texts == ["from the subprocess"]


@pytest.mark.parametrize("mod", ["lightfall_voice.stt_service.service"])
def test_service_imports_without_faster_whisper(mod):
    code = (
        "import sys; sys.modules['faster_whisper'] = None; sys.modules['ctranslate2'] = None; "
        f"import {mod}"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_initial_load_waits_for_config_set(broker_url):
    """A config.set inside the initial window gives exactly one load, with that config."""
    fac = Factory()

    async def go():
        r = await _rig(broker_url, fac, initial_load_delay_s=2.0)
        try:
            await _until(lambda: r.client.health == "up")
            before = (len(fac.calls), dict(r.host.summary))
            reply = await r.client.call(
                "config.set", {"model": "base", "model_dir": "D:/models", "device": "cpu"}
            )
            await r.ready()
            await asyncio.sleep(2.5)  # past the original window: no second load
            return before, reply
        finally:
            await r.close()

    (n_before, summ_before), reply = _run(go())
    assert n_before == 0
    assert summ_before["status"] == "loading"
    assert reply["status"] == "ok"
    assert reply["reloading"] is True
    assert len(fac.calls) == 1
    assert fac.calls[0]["model"] == "base"
    assert fac.calls[0]["model_dir"] == "D:/models"
    assert fac.calls[0]["device"] == "cpu"


def test_initial_load_without_config_set_uses_defaults(broker_url):
    fac = Factory()

    async def go():
        r = await _rig(broker_url, fac, initial_load_delay_s=0.3)
        try:
            await r.ready()
            await asyncio.sleep(0.3)
        finally:
            await r.close()

    _run(go())
    assert len(fac.calls) == 1
    assert fac.calls[0]["model"] == "small"
    assert fac.calls[0]["device"] == "auto"


class _Blocking(Factory):
    """Transcribers that block in transcribe() until ``release`` is set."""

    def __init__(self, text="late"):
        super().__init__(text)
        self.release = threading.Event()
        self.entered = threading.Event()

    def __call__(self, **cfg):
        t = super().__call__(**cfg)
        orig = t.transcribe

        def slow(audio, language):
            self.entered.set()
            self.release.wait(10)
            return orig(audio, language)

        t.transcribe = slow
        return t


def test_cancel_while_transcribing_frees_slot_and_drops_transcript(broker_url):
    fac = _Blocking("late")

    async def go():
        r = await _rig(broker_url, fac)
        try:
            await r.ready()
            sid = await r.start()
            await r.send(sid, _tone(1.0))
            await r.client.call("utterance.stop", {"stream_id": sid})
            await _until(fac.entered.is_set)
            cancel = await r.client.call("utterance.cancel", {"stream_id": sid})
            again = await r.client.call("utterance.start", START)
            fac.release.set()
            await asyncio.sleep(0.5)
            return r, sid, cancel, again
        finally:
            fac.release.set()
            await r.close()

    r, sid, cancel, again = _run(go())
    assert cancel["status"] == "ok"
    assert again["status"] == "ok", again  # not busy
    assert "transcript" not in r.names(sid)
    assert r.get("transcript") == []


def test_idle_stream_emits_idle_error(broker_url, monkeypatch):
    from lightfall_utils.ipc.services import protocol

    monkeypatch.setattr(protocol, "STREAM_IDLE_S", 0.5)

    async def go():
        r = await _rig(broker_url, Factory())
        try:
            await r.ready()
            sid = await r.start()
            await r.send(sid, _tone(0.3))
            await _until(lambda: "error" in r.names(), 5)
            again = await r.client.call("utterance.start", START)
            return r, sid, again
        finally:
            await r.close()

    r, sid, again = _run(go())
    (err,) = r.get("error")
    assert err == {
        **err,
        "stream_id": sid,
        "code": "idle",
        "message": "Microphone stopped sending audio",
    }
    assert again["status"] == "ok"


def test_detector_end_after_stop_stops_feeding_later_chunks(broker_url):
    """If queued chunks after utterance.stop reach a detector end (e.g. silence),
    subsequent queued chunks must not be fed into the detector."""
    fac = Factory("trimmed")

    async def go():
        r = await _rig(broker_url, fac)
        try:
            await r.ready()
            release = _hold_listening(r.host)
            sid = await r.start()
            # 1.0s tone + 1.0s silence reaches detector "silence" at 2.0s; the extra
            # 1.0s of tone queued after it must not be fed once the detector has ended.
            audio = np.concatenate([_tone(1.0), _zeros(1.0), _tone(1.0)])
            await r.send(sid, audio)
            await r.client.call("utterance.stop", {"stream_id": sid})
            release.set()
            await _until(lambda: "transcript" in r.names())
            return r
        finally:
            await r.close()

    r = _run(go())
    assert r.get("ended")[0]["reason"] == "stop"
    ((n, _lang),) = fac.made[0].calls
    assert n == 2 * RATE


def test_stop_shuts_down_executor(broker_url, monkeypatch):
    from lightfall_voice.stt_service import service as svc_mod

    executors = []
    real_executor = svc_mod.ThreadPoolExecutor

    def track_executor(*a, **kw):
        ex = real_executor(*a, **kw)
        executors.append(ex)
        return ex

    monkeypatch.setattr(svc_mod, "ThreadPoolExecutor", track_executor)

    async def go():
        r = await _rig(broker_url, Factory())
        await r.ready()
        assert len(executors) == 1 and not executors[0]._shutdown
        await r.close()
        assert executors[0]._shutdown

    _run(go())


def test_stt_smoke_cpu_device_check_and_not_ready_constant(broker_url):
    import argparse
    import tools.stt_smoke as stt_smoke

    assert stt_smoke.NOT_READY_CODE == "unknown"

    class DeviceFactory(Factory):
        def __init__(self, device: str) -> None:
            super().__init__("smoke ok")
            self.reported_device = device

        def __call__(self, **cfg):
            t = super().__call__(**cfg)
            t.device = self.reported_device
            return t

    async def go(reported_device: str, requested_device: str) -> int:
        host = make_host(
            stt_smoke.SCOPE,
            transcriber_factory=DeviceFactory(reported_device),
            heartbeat_s=0.05,
            single_instance=False,
            initial_load_delay_s=0.0,
        )
        svc = NatsCore(broker_url)
        await svc.connect()
        await host.start(svc)
        try:
            args = argparse.Namespace(
                model="small",
                device=requested_device,
                model_dir="",
                load_timeout=10.0,
                realtime=False,
            )
            return await stt_smoke.run(broker_url, "tok", _tone(1.0), args)
        finally:
            await host.stop()
            await svc.close()

    assert _run(go("cpu", "cpu")) == 0
    assert _run(go("cuda", "cpu")) == 1
