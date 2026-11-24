from __future__ import annotations

from lightfall_voice.tts.scheduler import SpeechItem, SpeechScheduler


def _item(
    text: str,
    priority: str = "normal",
    deadline: float | None = None,
    voice: str = "af_heart",
    speed: float = 1.0,
) -> SpeechItem:
    return SpeechItem(
        text=text,
        voice=voice,
        speed=speed,
        priority=priority,  # type: ignore[arg-type]
        deadline=deadline,
    )


def test_normal_interrupts_normal() -> None:
    started: list[SpeechItem] = []
    cancelled: list[SpeechItem] = []
    sched = SpeechScheduler(start=started.append, cancel=cancelled.append)

    n1 = _item("first", "normal")
    n2 = _item("second", "normal")

    sched.submit(n1)
    assert sched.current is n1
    assert started == [n1]
    assert cancelled == []

    sched.submit(n2)
    assert sched.current is n2
    assert cancelled == [n1]
    assert started == [n1, n2]

    sched.finished(n2)
    assert sched.current is None
    assert started == [n1, n2]


def test_high_interrupts_normal() -> None:
    started: list[SpeechItem] = []
    cancelled: list[SpeechItem] = []
    sched = SpeechScheduler(start=started.append, cancel=cancelled.append)

    n1 = _item("normal-1", "normal")
    h1 = _item("high-1", "high")

    sched.submit(n1)
    sched.submit(h1)

    assert cancelled == [n1]
    assert started == [n1, h1]
    assert sched.current is h1

    sched.finished(h1)
    assert sched.current is None
    assert started == [n1, h1]


def test_high_not_interrupted_by_normal_newest_pending_wins() -> None:
    started: list[SpeechItem] = []
    cancelled: list[SpeechItem] = []
    sched = SpeechScheduler(start=started.append, cancel=cancelled.append)

    h1 = _item("high-1", "high")
    n1 = _item("normal-1", "normal")
    n2 = _item("normal-2", "normal")

    sched.submit(h1)
    sched.submit(n1)
    sched.submit(n2)

    assert cancelled == []
    assert started == [h1]
    assert sched.current is h1

    sched.finished(h1)
    assert sched.current is n2
    assert started == [h1, n2]

    sched.finished(n2)
    assert sched.current is None


def test_high_queues_behind_high() -> None:
    started: list[SpeechItem] = []
    cancelled: list[SpeechItem] = []
    sched = SpeechScheduler(start=started.append, cancel=cancelled.append)

    h1 = _item("high-1", "high")
    h2 = _item("high-2", "high")
    h3 = _item("high-3", "high")

    sched.submit(h1)
    sched.submit(h2)
    sched.submit(h3)

    assert cancelled == []
    assert started == [h1]
    assert sched.current is h1

    sched.finished(h1)
    assert sched.current is h2
    assert started == [h1, h2]

    sched.finished(h2)
    assert sched.current is h3
    assert started == [h1, h2, h3]

    sched.finished(h3)
    assert sched.current is None


def test_finish_order_high_queue_then_pending() -> None:
    started: list[SpeechItem] = []
    cancelled: list[SpeechItem] = []
    sched = SpeechScheduler(start=started.append, cancel=cancelled.append)

    h1 = _item("high-1", "high")
    n1 = _item("normal-1", "normal")
    h2 = _item("high-2", "high")
    n2 = _item("normal-2", "normal")
    h3 = _item("high-3", "high")

    sched.submit(h1)
    sched.submit(n1)
    sched.submit(h2)
    sched.submit(n2)
    sched.submit(h3)

    assert started == [h1]
    sched.finished(h1)
    assert started == [h1, h2]
    sched.finished(h2)
    assert started == [h1, h2, h3]
    sched.finished(h3)
    assert started == [h1, h2, h3, n2]
    sched.finished(n2)
    assert sched.current is None
    assert cancelled == []


def test_expired_items_skipped_at_start() -> None:
    now = 100.0
    started: list[SpeechItem] = []
    cancelled: list[SpeechItem] = []
    sched = SpeechScheduler(
        start=started.append,
        cancel=cancelled.append,
        clock=lambda: now,
    )

    h1 = _item("high-1", "high", deadline=None)
    h2 = _item("high-expired", "high", deadline=105.0)
    h3 = _item("high-valid", "high", deadline=120.0)
    n1 = _item("normal-expired", "normal", deadline=110.0)

    sched.submit(h1)
    sched.submit(h2)
    sched.submit(h3)
    sched.submit(n1)

    now = 115.0
    sched.finished(h1)
    assert sched.current is h3
    assert started == [h1, h3]

    sched.finished(h3)
    assert sched.current is None
    assert started == [h1, h3]
    assert cancelled == []


def test_stop_all_returns_interrupted_high() -> None:
    started: list[SpeechItem] = []
    cancelled: list[SpeechItem] = []
    sched = SpeechScheduler(start=started.append, cancel=cancelled.append)

    h1 = _item("high-1", "high")
    h2 = _item("high-2", "high")
    n1 = _item("normal-1", "normal")

    sched.submit(h1)
    sched.submit(h2)
    sched.submit(n1)

    interrupted = sched.stop_all()
    assert interrupted is h1
    assert cancelled == [h1]
    assert sched.current is None

    sched.finished(h1)
    assert sched.current is None
    assert started == [h1]


def test_stop_all_with_normal_returns_none() -> None:
    started: list[SpeechItem] = []
    cancelled: list[SpeechItem] = []
    sched = SpeechScheduler(start=started.append, cancel=cancelled.append)

    assert sched.stop_all() is None
    assert cancelled == []

    n1 = _item("normal-1", "normal")
    sched.submit(n1)

    interrupted = sched.stop_all()
    assert interrupted is None
    assert cancelled == [n1]
    assert sched.current is None


def test_finished_for_non_current_ignored() -> None:
    started: list[SpeechItem] = []
    cancelled: list[SpeechItem] = []
    sched = SpeechScheduler(start=started.append, cancel=cancelled.append)

    n1 = _item("normal-1", "normal")
    n2 = _item("normal-2", "normal")
    h1 = _item("high-1", "high")
    n3 = _item("normal-3", "normal")

    sched.submit(n1)
    sched.submit(n2)
    sched.submit(h1)
    sched.submit(n3)

    sched.finished(n1)
    sched.finished(n2)
    assert sched.current is h1
    assert started == [n1, n2, h1]

    sched.finished(h1)
    assert sched.current is n3
    assert started == [n1, n2, h1, n3]
