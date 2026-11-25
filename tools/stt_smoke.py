"""End-to-end smoke test of the stt service with a real faster-whisper model.

Starts a temporary local NATS broker, launches ``python -m lightfall_voice.stt_service``
as a subprocess (session token on stdin), configures the model and device over rpc,
streams a WAV file (resampled to 16 kHz mono) in 100 ms chunks and prints the transcript,
the device, the model and the time taken.

    python tools/stt_smoke.py speech.wav --device gpu --model small

``--device gpu`` is forced CUDA (a load failure is reported, never hidden by the CPU
fallback); ``auto`` may fall back to CPU.  Exit status 0 on a transcript, 1 otherwise.
"""

from __future__ import annotations

import argparse
import asyncio
import secrets
import socket
import subprocess
import sys
import time
import wave

import numpy as np

from lightfall_utils.ipc import LocalNatsServer
from lightfall_utils.ipc.services.client import ServiceClient
from lightfall_utils.ipc.services.core import NatsCore
from lightfall_utils.ipc.services.descriptor import ServiceDescriptor
from lightfall_voice.audio import CHUNK_SAMPLES, RATE, float32_to_pcm16, resample

SCOPE = "lfsttsmoke"
START = {"rate": RATE, "format": "s16le", "channels": 1}
NOT_READY_CODE = "unknown"  # utterance.start reply code before the model is loaded
_WIDTH_DTYPES = {1: np.uint8, 2: np.dtype("<i2"), 4: np.dtype("<i4")}


def read_wav_16k(path: str) -> np.ndarray:
    """Read a PCM WAV as float32 mono at 16 kHz."""
    with wave.open(path, "rb") as w:
        rate, width, channels = w.getframerate(), w.getsampwidth(), w.getnchannels()
        raw = w.readframes(w.getnframes())
    if width not in _WIDTH_DTYPES:
        raise SystemExit(f"unsupported sample width {width} bytes")
    a = np.frombuffer(raw, dtype=_WIDTH_DTYPES[width]).astype(np.float32)
    if width == 1:
        a = (a - 128.0) / 128.0
    else:
        a /= float(2 ** (8 * width - 1))
    if channels > 1:
        a = a.reshape(-1, channels).mean(axis=1)
    return resample(a, rate, RATE)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def _until(pred, timeout: float, what: str) -> None:
    end = time.monotonic() + timeout
    while not pred():
        if time.monotonic() > end:
            raise TimeoutError(f"timed out waiting for {what}")
        await asyncio.sleep(0.05)


def _summary(client: ServiceClient) -> dict:
    s = (client.info or {}).get("summary")
    return s if isinstance(s, dict) else {}


async def run(url: str, token: str, audio: np.ndarray, args: argparse.Namespace) -> int:
    core = NatsCore(url)
    await core.connect()
    desc = ServiceDescriptor(name="stt", scope=SCOPE, contract="1.0", launch="external")
    client = ServiceClient(
        desc, core, scope=SCOPE, app_name="stt_smoke", app_version="1", session_token=token
    )
    events: list[tuple[str, dict]] = []
    for name in ("listening", "ended", "transcript", "error"):
        client.on_event(
            f"utterance.{name}", lambda ev, n=name: events.append((n, ev))
        )
    try:
        await client.start()
        await _until(lambda: client.health == "up", 30, "service session")
        # Like Lightfall's controller, send config.set right away: the service defers its
        # first load for a moment so this config is the only one loaded.
        boot = _summary(client)
        print(f"before config:  status={boot.get('status')!r} (expected 'loading')")

        t0 = time.perf_counter()
        conf = {"model": args.model, "device": args.device, "language": "en"}
        if args.model_dir:
            conf["model_dir"] = args.model_dir
        cfg = await client.call("config.set", conf)
        if cfg.get("status") != "ok":
            print(f"config.set failed: {cfg}", file=sys.stderr)
            return 1
        # utterance.start fails with "model not ready" until the requested model is loaded.
        sid = None
        end = time.monotonic() + args.load_timeout
        while sid is None:
            summ = _summary(client)
            if summ.get("status") == "error" and summ.get("model") == args.model:
                print(f"model load FAILED: {summ.get('error')}", file=sys.stderr)
                return 1
            reply = await client.call("utterance.start", START)
            if reply.get("status") == "ok":
                sid = reply["stream_id"]
                break
            if reply.get("code") != NOT_READY_CODE:
                print(f"utterance.start failed: {reply}", file=sys.stderr)
                return 1
            if time.monotonic() > end:
                print(f"model not ready after {args.load_timeout}s: {summ}", file=sys.stderr)
                return 1
            await asyncio.sleep(0.2)
        load_s = time.perf_counter() - t0
        await _until(
            lambda: _summary(client).get("status") == "ready"
            and _summary(client).get("model") == args.model,
            10,
            "ready heartbeat",
        )
        summ = _summary(client)

        client.adopt_stream(sid, "audio")
        t1 = time.perf_counter()
        n = 0
        for seq, i in enumerate(range(0, len(audio), CHUNK_SAMPLES)):
            await client.send_chunk(sid, seq, float32_to_pcm16(audio[i : i + CHUNK_SAMPLES]))
            n += 1
            if args.realtime:
                await asyncio.sleep(CHUNK_SAMPLES / RATE)
        t_sent = time.perf_counter()
        await client.call("utterance.stop", {"stream_id": sid})
        await _until(
            lambda: any(k in ("transcript", "error") for k, _ in events)
            or any(k == "ended" and e.get("reason") in ("no_speech", "cancel") for k, e in events),
            120,
            "transcript",
        )
        t_done = time.perf_counter()
        await client.call("shutdown", {})

        text = next((e["text"] for k, e in events if k == "transcript"), None)
        ended = [e.get("reason") for k, e in events if k == "ended"]
        errors = [e for k, e in events if k == "error"]
        print(f"model:          {summ.get('model')}")
        print(f"device:         {summ.get('device')} (requested {args.device})")
        print(f"audio:          {len(audio) / RATE:.2f} s in {n} chunks")
        print(
            f"load:           {load_s:.2f} s (config.set -> utterance.start ok;"
            f" reloaded={cfg.get('reloading')})"
        )
        print(f"send:           {t_sent - t1:.2f} s")
        print(f"stop->result:   {t_done - t_sent:.2f} s")
        print(f"total:          {t_done - t0:.2f} s")
        print(f"ended reasons:  {ended}")
        if errors:
            print(f"errors:         {errors}")
        print(f"transcript:     {text!r}")
        if args.device == "gpu" and summ.get("device") != "cuda":
            print("FAIL: --device gpu did not run on cuda", file=sys.stderr)
            return 1
        if args.device == "cpu" and summ.get("device") != "cpu":
            print("FAIL: --device cpu did not run on cpu", file=sys.stderr)
            return 1
        return 0 if text else 1
    finally:
        await client.close()
        await core.close()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("wav", help="PCM WAV file to transcribe")
    p.add_argument("--device", choices=("auto", "gpu", "cpu"), default="auto")
    p.add_argument("--model", default="small")
    p.add_argument(
        "--model-dir",
        default="",
        help="model cache directory (default: the faster-whisper default)",
    )
    p.add_argument(
        "--load-timeout",
        type=float,
        default=900.0,
        help="seconds to wait for the model (the first run downloads it)",
    )
    p.add_argument(
        "--realtime",
        action="store_true",
        help="pace chunks at 100 ms (default: send as fast as possible)",
    )
    args = p.parse_args(argv)

    audio = read_wav_16k(args.wav)
    port = _free_port()
    server = LocalNatsServer(port=port, host="127.0.0.1")
    server.start(timeout_s=10.0)
    url = f"nats://127.0.0.1:{port}"
    token = secrets.token_hex(16)
    proc = subprocess.Popen(
        [sys.executable, "-m", "lightfall_voice.stt_service",
         "--scope", SCOPE, "--nats-url", url, "--token-stdin"],
        stdin=subprocess.PIPE,
    )
    try:
        proc.stdin.write(token.encode() + b"\n")
        proc.stdin.flush()
        rc = asyncio.run(run(url, token, audio, args))
        proc.stdin.close()
        try:
            proc.wait(20)
        except subprocess.TimeoutExpired:
            print("service did not exit; killing", file=sys.stderr)
        return rc
    finally:
        if proc.poll() is None:
            proc.kill()
        server.stop()


if __name__ == "__main__":
    raise SystemExit(main())
