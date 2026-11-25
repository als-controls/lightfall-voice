"""End-to-end smoke test of the tts service with a real Kokoro ONNX model.

Starts a temporary local NATS broker, launches ``python -m lightfall_voice.tts_service``
as a subprocess (session token on stdin), configures the model quality over rpc, waits for
ready, calls ``speak`` and writes the streamed PCM chunks to a WAV file, then prints the
model load time, the time to first chunk, the total time and the audio length.

    python tools/tts_smoke.py "Permission needed: run pytest in lightfall-voice." --out out.wav
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

from lightfall_utils.ipc import LocalNatsServer
from lightfall_utils.ipc.services.client import ServiceClient
from lightfall_utils.ipc.services.core import NatsCore
from lightfall_utils.ipc.services.descriptor import ServiceDescriptor

SCOPE = "lfttssmoke"


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


async def run(url: str, token: str, args: argparse.Namespace) -> int:
    core = NatsCore(url)
    await core.connect()
    desc = ServiceDescriptor(name="tts", scope=SCOPE, contract="1.0", launch="external")
    client = ServiceClient(
        desc, core, scope=SCOPE, app_name="tts_smoke", app_version="1", session_token=token
    )
    chunks: dict[str, list[tuple[int, bytes, bool]]] = {}
    first_chunk_at: dict[str, float] = {}
    errors: list[dict] = []

    def on_chunk(sid: str, seq: int, data: bytes, final: bool) -> None:
        if not final and sid not in first_chunk_at:
            first_chunk_at[sid] = time.perf_counter()
        chunks.setdefault(sid, []).append((seq, data, final))

    client.on_stream(on_chunk)
    client.on_event("speech.error", errors.append)
    try:
        await client.start()
        await _until(lambda: client.health == "up", 30, "service session")
        boot = _summary(client)
        print(f"before config:  status={boot.get('status')!r} (expected 'loading')")

        t0 = time.perf_counter()
        conf: dict[str, str] = {"quality": args.quality}
        if args.model_dir:
            conf["model_dir"] = args.model_dir
        cfg = await client.call("config.set", conf)
        if cfg.get("status") != "ok":
            print(f"config.set failed: {cfg}", file=sys.stderr)
            return 1

        end = time.monotonic() + args.load_timeout
        while True:
            summ = _summary(client)
            if summ.get("status") == "error" and summ.get("model") == args.quality:
                print(f"model load FAILED: {summ.get('error')}", file=sys.stderr)
                return 1
            if summ.get("status") == "ready" and summ.get("model") == args.quality:
                break
            if time.monotonic() > end:
                print(
                    f"model not ready after {args.load_timeout}s: {summ}", file=sys.stderr
                )
                return 1
            await asyncio.sleep(0.05)
        load_s = time.perf_counter() - t0
        summ = _summary(client)

        t_speak = time.perf_counter()
        reply = await client.call("speak", {"text": args.text, "voice": args.voice})
        if reply.get("status") != "ok":
            print(f"speak failed: {reply}", file=sys.stderr)
            return 1
        sid = reply["stream_id"]
        rate = int(reply.get("rate", 24000))
        channels = int(reply.get("channels", 1))

        await _until(
            lambda: any(final for _seq, _data, final in chunks.get(sid, [])),
            120,
            "speech stream completion",
        )
        t_done = time.perf_counter()
        await client.call("shutdown", {})

        stream_chunks = sorted(chunks.get(sid, []), key=lambda item: item[0])
        non_final = [data for _seq, data, final in stream_chunks if not final]
        pcm = b"".join(non_final)
        n_samples = len(pcm) // (2 * channels)
        audio_s = n_samples / rate if rate else 0.0
        first_s = (
            first_chunk_at[sid] - t_speak if sid in first_chunk_at else t_done - t_speak
        )

        with wave.open(args.out, "wb") as w:
            w.setnchannels(channels)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes(pcm)

        print(f"quality:        {summ.get('model')}")
        print(f"voice:          {args.voice}")
        print(
            f"load:           {load_s:.2f} s (config.set -> ready;"
            f" reloaded={cfg.get('reloading')})"
        )
        print(f"first chunk:    {first_s:.2f} s (speak -> first chunk)")
        print(f"speak->done:    {t_done - t_speak:.2f} s")
        print(f"total:          {t_done - t0:.2f} s")
        print(
            f"audio:          {audio_s:.2f} s ({n_samples} samples at {rate} Hz"
            f" in {len(non_final)} chunks) -> {args.out}"
        )
        if errors:
            print(f"errors:         {errors}", file=sys.stderr)
            return 1
        return 0 if pcm else 1
    finally:
        await client.close()
        await core.close()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("text", help="text to synthesize")
    p.add_argument("--voice", default="af_heart")
    p.add_argument("--quality", choices=("fp16", "int8"), default="fp16")
    p.add_argument(
        "--model-dir",
        default="",
        help="model cache directory (default: the lightfall-voice kokoro cache)",
    )
    p.add_argument("--out", default="out.wav", help="output WAV path")
    p.add_argument(
        "--load-timeout",
        type=float,
        default=900.0,
        help="seconds to wait for the model (the first run downloads it)",
    )
    args = p.parse_args(argv)

    port = _free_port()
    server = LocalNatsServer(port=port, host="127.0.0.1")
    server.start(timeout_s=10.0)
    url = f"nats://127.0.0.1:{port}"
    token = secrets.token_hex(16)
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "lightfall_voice.tts_service",
            "--scope",
            SCOPE,
            "--nats-url",
            url,
            "--token-stdin",
        ],
        stdin=subprocess.PIPE,
    )
    try:
        proc.stdin.write(token.encode() + b"\n")
        proc.stdin.flush()
        rc = asyncio.run(run(url, token, args))
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
