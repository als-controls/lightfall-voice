import time

from lightfall_voice.daemon_client import DaemonClient


def wait_for(predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def test_speak_sends_hookless_payload(fake_daemon):
    client = DaemonClient(port=fake_daemon.port)
    try:
        client.speak("Task complete", voice="af_sky")
        assert wait_for(lambda: len(fake_daemon.requests) == 1)
        req = fake_daemon.requests[0]
        assert req == {
            "action": "speak",
            "args": {"message": "Task complete", "voice": "af_sky"},
        }
    finally:
        client.stop()


def test_speak_without_voice_omits_voice_key_value(fake_daemon):
    client = DaemonClient(port=fake_daemon.port)
    try:
        client.speak("Searching")
        assert wait_for(lambda: len(fake_daemon.requests) == 1)
        assert fake_daemon.requests[0]["args"]["voice"] is None
    finally:
        client.stop()


def test_daemon_down_is_silent():
    client = DaemonClient(port=1, connect_timeout=0.2)  # nothing listens on port 1
    try:
        client.speak("hello")  # must not raise
        time.sleep(0.5)
    finally:
        client.stop()


def test_speak_never_blocks_caller(fake_daemon):
    client = DaemonClient(port=fake_daemon.port)
    try:
        start = time.monotonic()
        for _ in range(20):
            client.speak("msg")
        assert time.monotonic() - start < 0.1
    finally:
        client.stop()


def test_configure_redirects(fake_daemon):
    client = DaemonClient(port=1)
    try:
        client.configure("127.0.0.1", fake_daemon.port)
        client.speak("after reconfigure")
        assert wait_for(lambda: len(fake_daemon.requests) == 1)
    finally:
        client.stop()


def test_expired_ttl_message_is_never_sent(fake_daemon):
    client = DaemonClient(port=fake_daemon.port)
    try:
        client.speak("stale", ttl=0.0)
        client.speak("fresh")
        assert wait_for(lambda: len(fake_daemon.requests) == 1)
        assert fake_daemon.requests[0]["args"]["message"] == "fresh"
    finally:
        client.stop()


def test_none_ttl_message_is_always_sent(fake_daemon):
    client = DaemonClient(port=fake_daemon.port)
    try:
        client.speak("no ttl", ttl=None)
        assert wait_for(lambda: len(fake_daemon.requests) == 1)
        assert fake_daemon.requests[0]["args"]["message"] == "no ttl"
    finally:
        client.stop()
