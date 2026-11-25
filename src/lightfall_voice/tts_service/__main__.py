import argparse
from pathlib import Path
import sys

from lightfall_utils.ipc.services.cli import run_service
from lightfall_voice.tts_service.service import make_host
from lightfall_voice.tts_service.synth import FakeSynth


def _fake_resolve(
    quality: str, model_dir: str = "", *, progress=None, **_kw
) -> tuple[Path, Path]:
    return Path(f"kokoro-v1.0.{quality}.onnx"), Path("voices-v1.0.bin")


def main(argv: list[str]) -> int:
    # Test-only, hidden: ``--fake-synth`` uses FakeSynth with dummy model paths.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--fake-synth", action="store_true", help=argparse.SUPPRESS)
    opts, rest = pre.parse_known_args(argv)
    if opts.fake_synth:
        host = make_host(synth_factory=FakeSynth, resolve=_fake_resolve)
    else:
        host = make_host()
    return run_service(host, rest)


raise SystemExit(main(sys.argv[1:]))
