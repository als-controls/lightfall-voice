import argparse
import sys

from lightfall_utils.ipc.services.cli import run_service
from lightfall_voice.stt_service.service import fake_factory, make_host


def main(argv: list[str]) -> int:
    # Test-only, hidden: ``--fake-transcriber TEXT`` answers every utterance with TEXT.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--fake-transcriber", default=None, help=argparse.SUPPRESS)
    opts, rest = pre.parse_known_args(argv)
    fake = opts.fake_transcriber
    factory = fake_factory(fake) if fake is not None else None
    return run_service(make_host(transcriber_factory=factory), rest)


raise SystemExit(main(sys.argv[1:]))
