"""The tts ServiceHost, in process, against a real local broker and a fake synthesizer."""

from __future__ import annotations

import asyncio
from pathlib import Path
import threading

from lightfall_utils.ipc.services.client import ServiceClient
from lightfall_utils.ipc.services.core import NatsCore
from lightfall_utils.ipc.services.descriptor import ServiceDescriptor
from lightfall_voice.tts_service.models import ModelFilesMissing
from lightfall_voice.tts_service.service import make_host
from lightfall_voice.tts_service.synth import RATE, FakeSynth

SCOPE = "lftts"


class FakeResolver:
    """Fake ``models.resolve`` recording calls, optionally reporting download progress or blocking."""

    def __init__(
        self,
        *,
        download: bool = False,
        block: bool = False,
        error: Exception | None = None,
    ) -> None:
        self.download = download
        self.error = error
        self.calls: list[tuple[str, str]] = []
        self.entered = threading.Event()
        self.gate = threading.Event()
        if not block:
            self.gate.set()

    def __call__(self, quality: str, model_dir: str = "", *, progress=None, **_kw):
        self.calls.append((quality, model_dir))
        if self.download and progress is not None:
            progress("downloading")
        self.entered.set()
        self.gate.wait(10)
        if self.error is not None:
            raise self.error
        return Path(f"fake-{quality}.onnx"), Path("fake-voices.bin")


class SynthFactory:
    """Fake ``synth_factory`` recording created FakeSynth instances."""

    def __init__(self, synth_cls=FakeSynth) -> None:
        self.synth_cls = synth_cls
        self.calls: list[tuple[Path, Path]] = []
        self.made: list[FakeSynth] = []

    def __call__(self, model_path: Path, voices_path: Path):
        self.calls.append((model_path, voices_path))
        s = self.synth_cls(model_path, voices_path)
        self.made.append(s)
        return s


async def _until(pred, timeout: float = 5.0) -> None:
    loop = asyncio.get_running_loop()
    end = loop.time() + timeout
    while not pred():
        if loop.time() > end:
            raise AssertionError("condition not met in time")
        await asyncio.sleep(0.01)


class Rig:
    def __init__(self, host, svc, cli, client, resolver, factory) -> None:
        self.host = host
        self.svc = svc
        self.cli = cli
        self.client = client
        self.resolver = resolver
        self.factory = factory
        self.chunks: dict[str, list[tuple[int, bytes, bool]]] = {}
        self.chunk_order: list[tuple[str, int, bytes, bool]] = []
        self.errors: list[dict] = []
        client.on_stream(self._on_chunk)
        client.on_event("speech.error", self.errors.append)

    def _on_chunk(self, sid: str, seq: int, data: bytes, final: bool) -> None:
        self.chunks.setdefault(sid, []).append((seq, data, final))
        self.chunk_order.append((sid, seq, data, final))

    def stream_done(self, sid: str) -> bool:
        items = self.chunks.get(sid, [])
        return any(final for _seq, _data, final in items)

    async def ready(self) -> None:
        await _until(lambda: self.host.summary.get("status") == "ready")

    async def close(self) -> None:
        await self.client.close()
        await self.host.stop()
        await self.svc.close()
        await self.cli.close()


async def _rig(
    broker_url: str,
    *,
    resolver=None,
    synth_factory=None,
    initial_load_delay_s: float = 0.0,
    **kw,
) -> Rig:
    if resolver is None:
        resolver = FakeResolver()
    if synth_factory is None:
        synth_factory = SynthFactory()
    kw.setdefault("heartbeat_s", 0.1)
    kw.setdefault("single_instance", False)
    host = make_host(
        SCOPE,
        synth_factory=synth_factory,
        resolve=resolver,
        initial_load_delay_s=initial_load_delay_s,
        **kw,
    )
    svc = NatsCore(broker_url)
    await svc.connect()
    await host.start(svc)
    cli = NatsCore(broker_url)
    await cli.connect()
    desc = ServiceDescriptor(name="tts", scope=SCOPE, contract="1.0", launch="external")
    client = ServiceClient(
        desc,
        cli,
        scope=SCOPE,
        app_name="tester",
        app_version="1.0",
    )
    rig = Rig(host, svc, cli, client, resolver, synth_factory)
    await client.start()
    return rig


def _run(coro):
    return asyncio.run(asyncio.wait_for(coro, 30))


def test_config_set_before_delay_loads_once_with_that_config(broker_url):
    resolver = FakeResolver()
    fac = SynthFactory()

    async def go():
        r = await _rig(
            broker_url, resolver=resolver, synth_factory=fac, initial_load_delay_s=0.6
        )
        try:
            await _until(lambda: r.client.health == "up")
            before = (len(resolver.calls), dict(r.host.summary))
            reply = await r.client.call(
                "config.set",
                {"model_dir": "D:/kokoro", "quality": "int8", "lang": "en-gb"},
            )
            await r.ready()
            await asyncio.sleep(0.8)  # past the initial delay window: no second load
            got = await r.client.call("config.get", {})
            return before, reply, got, dict(r.host.summary)
        finally:
            await r.close()

    (n_before, summ_before), reply, got, summ_after = _run(go())
    assert n_before == 0
    assert summ_before == {"model": "fp16", "status": "loading", "error": ""}
    assert reply["status"] == "ok"
    assert reply["reloading"] is True
    assert resolver.calls == [("int8", "D:/kokoro")]
    assert len(fac.calls) == 1
    assert got["model_dir"] == "D:/kokoro"
    assert got["quality"] == "int8"
    assert got["lang"] == "en-gb"
    assert summ_after == {"model": "int8", "status": "ready", "error": ""}


def test_default_load_after_delay(broker_url):
    resolver = FakeResolver()
    fac = SynthFactory()

    async def go():
        r = await _rig(
            broker_url, resolver=resolver, synth_factory=fac, initial_load_delay_s=0.2
        )
        try:
            await r.ready()
            await asyncio.sleep(0.2)
            got = await r.client.call("config.get", {})
            return got, dict(r.host.summary)
        finally:
            await r.close()

    got, summ = _run(go())
    assert resolver.calls == [("fp16", "")]
    assert len(fac.calls) == 1
    assert got["model_dir"] == ""
    assert got["quality"] == "fp16"
    assert got["lang"] == "en-us"
    assert summ == {"model": "fp16", "status": "ready", "error": ""}


def test_summary_downloading_then_ready(broker_url):
    resolver = FakeResolver(download=True, block=True)
    fac = SynthFactory()

    async def go():
        r = await _rig(broker_url, resolver=resolver, synth_factory=fac)
        try:
            await _until(lambda: resolver.entered.is_set())
            await _until(
                lambda: (
                    (r.client.info or {}).get("summary", {}).get("status")
                    == "downloading"
                )
            )
            during = (r.host.state, dict(r.host.summary))
            resolver.gate.set()
            await r.ready()
            await _until(
                lambda: (
                    (r.client.info or {}).get("summary", {}).get("status") == "ready"
                )
            )
            after = (r.host.state, dict(r.host.summary))
            return during, after
        finally:
            resolver.gate.set()
            await r.close()

    (s0, sum0), (s1, sum1) = _run(go())
    assert s0 == "starting"
    assert sum0 == {"model": "fp16", "status": "downloading", "error": ""}
    assert s1 == "ready"
    assert sum1 == {"model": "fp16", "status": "ready", "error": ""}


def test_missing_files_error_status(broker_url):
    resolver = FakeResolver(
        error=ModelFilesMissing("Missing kokoro-v1.0.fp16.onnx in /models")
    )

    async def go():
        r = await _rig(broker_url, resolver=resolver)
        try:
            await _until(lambda: r.host.summary.get("status") == "error")
            speak_reply = await r.client.call("speak", {"text": "Hello."})
            return r.host.state, dict(r.host.summary), speak_reply
        finally:
            await r.close()

    state, summary, speak_reply = _run(go())
    assert state == "error"
    assert summary["model"] == "fp16"
    assert summary["status"] == "error"
    assert "Missing kokoro-v1.0.fp16.onnx" in summary["error"]
    assert speak_reply["status"] == "error"
    assert speak_reply["code"] == "unknown"
    assert speak_reply["message"] == "model not ready"


def test_speak_streams_chunks_then_final(broker_url):
    text = (
        "one two three four five six seven eight nine ten. "
        "alpha beta gamma delta epsilon."
    )
    ref = FakeSynth()
    expected_samples = len(
        ref.synthesize("one two three four five six seven eight nine ten.")
    ) + len(ref.synthesize("alpha beta gamma delta epsilon."))

    async def go():
        r = await _rig(broker_url)
        try:
            await r.ready()
            reply = await r.client.call(
                "speak", {"text": text, "voice": "af_heart", "speed": 1.0}
            )
            assert reply["status"] == "ok", reply
            sid = reply["stream_id"]
            await _until(lambda: r.stream_done(sid))
            return reply, r.chunks[sid]
        finally:
            await r.close()

    reply, chunks = _run(go())
    assert reply["rate"] == RATE == 24000
    assert reply["format"] == "s16le"
    assert reply["channels"] == 1
    assert len(chunks) > 2
    seqs = [seq for seq, _data, _final in chunks]
    assert seqs == list(range(len(chunks)))
    for _seq, data, final in chunks[:-1]:
        assert final is False
        assert 0 < len(data) <= 9600
        assert len(data) % 2 == 0
    last_seq, last_data, last_final = chunks[-1]
    assert last_seq == len(chunks) - 1
    assert last_final is True
    assert last_data == b""
    total_samples = sum(len(data) // 2 for _seq, data, _final in chunks)
    assert total_samples == expected_samples


def test_speak_empty_text_bad_request(broker_url):
    async def go():
        r = await _rig(broker_url)
        try:
            await r.ready()
            return [
                await r.client.call("speak", {"text": ""}),
                await r.client.call("speak", {"text": "   \n\t  "}),
            ]
        finally:
            await r.close()

    for reply in _run(go()):
        assert reply["status"] == "error"
        assert reply["code"] == "bad_request"


def test_speak_unknown_voice(broker_url):
    async def go():
        r = await _rig(broker_url)
        try:
            await r.ready()
            return await r.client.call(
                "speak", {"text": "Hello there.", "voice": "unknown_voice"}
            )
        finally:
            await r.close()

    reply = _run(go())
    assert reply["status"] == "error"
    assert reply["code"] == "bad_request"
    assert reply["message"] == "unknown voice 'unknown_voice'"


def test_speak_not_ready(broker_url):
    resolver = FakeResolver(block=True)

    async def go():
        r = await _rig(broker_url, resolver=resolver)
        try:
            await _until(lambda: resolver.entered.is_set())
            reply = await r.client.call("speak", {"text": "Hello there."})
            resolver.gate.set()
            await r.ready()
            return reply
        finally:
            resolver.gate.set()
            await r.close()

    reply = _run(go())
    assert reply["status"] == "error"
    assert reply["code"] == "unknown"
    assert reply["message"] == "model not ready"


def test_cancel_mid_stream_emits_final_and_no_more_chunks(broker_url):
    entered_second = threading.Event()
    release_second = threading.Event()

    class GatedSynth(FakeSynth):
        def __init__(self, *_a, **_kw) -> None:
            super().__init__()
            self.call_count = 0

        def synthesize(
            self,
            sentence: str,
            voice: str = "af_heart",
            speed: float = 1.0,
            lang: str = "en-us",
        ):
            self.call_count += 1
            if self.call_count == 2:
                entered_second.set()
                release_second.wait(10)
            return super().synthesize(sentence, voice=voice, speed=speed, lang=lang)

    async def go():
        r = await _rig(broker_url, synth_factory=SynthFactory(GatedSynth))
        try:
            await r.ready()
            reply = await r.client.call(
                "speak",
                {
                    "text": (
                        "First sentence with several words. "
                        "Second sentence with more words. "
                        "Third sentence that should never run."
                    )
                },
            )
            sid = reply["stream_id"]
            await _until(lambda: entered_second.is_set() and len(r.chunks.get(sid, [])) >= 1)
            before_cancel = len(r.chunks[sid])
            cancel_reply = await r.client.call("cancel", {"stream_id": sid})
            release_second.set()
            await _until(lambda: r.stream_done(sid))
            await asyncio.sleep(0.1)
            unknown_cancel = await r.client.call("cancel", {"stream_id": "unknown_id"})
            finished_cancel = await r.client.call("cancel", {"stream_id": sid})
            return (
                r.chunks[sid],
                before_cancel,
                cancel_reply,
                unknown_cancel,
                finished_cancel,
                r.factory.made[0].call_count,
            )
        finally:
            release_second.set()
            await r.close()

    chunks, before_cancel, cancel_reply, unknown_cancel, finished_cancel, calls = _run(
        go()
    )
    assert cancel_reply["status"] == "ok"
    assert unknown_cancel["status"] == "ok"
    assert finished_cancel["status"] == "ok"
    assert calls == 2  # stopped before third sentence
    non_final = [c for c in chunks if not c[2]]
    assert len(non_final) == before_cancel
    assert chunks[-1][2] is True
    assert chunks[-1][1] == b""


def test_two_speaks_fifo(broker_url):
    entered_first = threading.Event()
    release_first = threading.Event()
    synth_sentences: list[str] = []

    class OrderedSynth(FakeSynth):
        def synthesize(
            self,
            sentence: str,
            voice: str = "af_heart",
            speed: float = 1.0,
            lang: str = "en-us",
        ):
            synth_sentences.append(sentence)
            if len(synth_sentences) == 1:
                entered_first.set()
                release_first.wait(10)
            return super().synthesize(sentence, voice=voice, speed=speed, lang=lang)

    async def go():
        r = await _rig(broker_url, synth_factory=SynthFactory(OrderedSynth))
        try:
            await r.ready()
            rep1 = await r.client.call("speak", {"text": "First utterance."})
            sid1 = rep1["stream_id"]
            await _until(entered_first.is_set)
            rep2 = await r.client.call("speak", {"text": "Second utterance."})
            sid2 = rep2["stream_id"]
            assert rep2["status"] == "ok"
            release_first.set()
            await _until(lambda: r.stream_done(sid1) and r.stream_done(sid2))
            return sid1, sid2, list(r.chunk_order)
        finally:
            release_first.set()
            await r.close()

    sid1, sid2, order = _run(go())
    assert synth_sentences == ["First utterance.", "Second utterance."]
    sids_seen = [sid for sid, _seq, _data, _final in order]
    last_sid1 = max(i for i, s in enumerate(sids_seen) if s == sid1)
    first_sid2 = min(i for i, s in enumerate(sids_seen) if s == sid2)
    assert last_sid1 < first_sid2


def test_synth_exception_emits_final_and_error_event(broker_url):
    class BoomSynth(FakeSynth):
        def synthesize(
            self,
            sentence: str,
            voice: str = "af_heart",
            speed: float = 1.0,
            lang: str = "en-us",
        ):
            raise RuntimeError("kokoro synth exploded")

    async def go():
        r = await _rig(broker_url, synth_factory=SynthFactory(BoomSynth))
        try:
            await r.ready()
            reply = await r.client.call("speak", {"text": "Hello world."})
            sid = reply["stream_id"]
            await _until(lambda: r.stream_done(sid) and len(r.errors) >= 1)
            return sid, r.chunks[sid], r.errors
        finally:
            await r.close()

    sid, chunks, errors = _run(go())
    assert chunks == [(0, b"", True)]
    assert len(errors) == 1
    assert errors[0]["stream_id"] == sid
    assert errors[0]["code"]
    assert "kokoro synth exploded" in errors[0]["message"]


def test_speed_clamping_and_config_merge_and_session_end(broker_url):
    seen_args: list[tuple[str, float, str]] = []
    entered = threading.Event()
    release = threading.Event()

    class RecordingSynth(FakeSynth):
        def synthesize(
            self,
            sentence: str,
            voice: str = "af_heart",
            speed: float = 1.0,
            lang: str = "en-us",
        ):
            seen_args.append((voice, speed, lang))
            if sentence == "Block here.":
                entered.set()
                release.wait(10)
            return super().synthesize(sentence, voice=voice, speed=speed, lang=lang)

    resolver = FakeResolver()
    fac = SynthFactory(RecordingSynth)

    async def go():
        r = await _rig(broker_url, resolver=resolver, synth_factory=fac)
        try:
            await r.ready()
            lang_reply = await r.client.call("config.set", {"lang": "en-gb"})
            assert lang_reply["reloading"] is False
            assert len(resolver.calls) == 1

            rep_low = await r.client.call("speak", {"text": "Low.", "speed": 0.1})
            await _until(lambda: r.stream_done(rep_low["stream_id"]))
            rep_high = await r.client.call(
                "speak", {"text": "High.", "voice": "bm_george", "speed": 9.0}
            )
            await _until(lambda: r.stream_done(rep_high["stream_id"]))

            rep_sess = await r.client.call(
                "speak", {"text": "Block here. Never reached."}
            )
            sid_sess = rep_sess["stream_id"]
            await _until(entered.is_set)
            tok = next(iter(r.host.sessions))
            await r.host._end_session(tok)
            release.set()
            await _until(lambda: r.stream_done(sid_sess) and len(r.errors) >= 1)
            return sid_sess, r.chunks[sid_sess], r.errors
        finally:
            release.set()
            await r.close()

    sid_sess, sess_chunks, errors = _run(go())
    assert seen_args[0] == ("af_heart", 0.5, "en-gb")
    assert seen_args[1] == ("bm_george", 2.0, "en-gb")
    assert len(seen_args) == 3  # "Never reached." was skipped after session end
    assert sess_chunks == [(0, b"", True)]
    assert errors[0]["stream_id"] == sid_sess

