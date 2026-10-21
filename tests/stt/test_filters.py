from lightfall_voice.stt_service.filters import clean, is_junk


def test_junk_filter_blocklist():
    for t in ("Thank you.", " you ", "Um...", ""):
        assert is_junk(t)
        assert clean(t) == ""
    assert not is_junk("move motor x to 3")
    assert clean("move motor x to 3") == "move motor x to 3"
