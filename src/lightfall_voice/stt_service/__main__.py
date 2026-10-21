import sys

from lightfall_utils.ipc.services.cli import run_service
from lightfall_voice.stt_service.service import make_host

raise SystemExit(run_service(make_host(), sys.argv[1:]))
